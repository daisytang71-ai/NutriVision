"""Train CustomCNN from scratch, selecting by validation dish calorie MAE."""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from . import TARGETS
from .data import (LabelScaler, NutritionDataset, load_manifest, sha256,
                   split_summary, unique_dishes)
from .evaluate import collect_predictions, report_predictions
from .metrics import aggregate_dishes, regression_metrics
from .model import CustomCNN
from .utils import (atomic_checkpoint, device_from, environment, load_checkpoint,
                    restore_rng, rng_state, save_json, seed_all, seed_worker)


def plot_history(history, output, kind, scaled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train (augmented)")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Val (all images)")
    axes[0].set(xlabel="Epoch", ylabel="Standardized MSE" if scaled else "Raw mixed-unit MSE")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_calories_MAE"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation dish calorie MAE (kcal)")
    fig.suptitle(f"{kind} | no test-set model selection")
    fig.tight_layout()
    fig.savefig(Path(output) / "loss_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-kind", required=True, choices=["synthetic", "real"])
    p.add_argument("--epochs", type=int, default=50, help="Total epochs, including resumed epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp", action="store_true", help="CUDA float16 mixed precision")
    p.add_argument("--sampling", choices=["all-images", "one-per-dish"], default="all-images",
                   help="All training images once per epoch, or one random image per dish")
    p.add_argument("--log-every", type=int, default=100, help="Print every N training batches")
    p.add_argument("--patience", type=int, default=7,
                   help="Stop after N consecutive epochs without lower validation dish calorie MAE; 0 disables")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--resume", help="Resume last.pt into a fresh output directory")
    args = p.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.image_size < 16 or args.num_workers < 0
            or args.threads < 1 or args.log_every < 1 or args.patience < 0 or args.lr <= 0
            or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        p.error("Invalid training hyperparameters")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        p.error("out-dir must be empty; resume into a new directory to preserve old evidence")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = device_from(args.device)
    if args.amp and device.type != "cuda":
        p.error("--amp requires --device cuda (or CUDA available with auto)")
    config = vars(args).copy()
    print("Loading manifest and checking all registered images...", flush=True)
    records = load_manifest(args.manifest, args.data_root)
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "val"]
    scaler = LabelScaler.fit(training, enabled=not args.no_standardize)
    train_mean = np.array([[r[k] for k in TARGETS] for r in unique_dishes(training)]).mean(0).tolist()
    digest = sha256(args.manifest)
    shutil.copyfile(args.manifest, out / "manifest.csv")
    model = CustomCNN(args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    generator = torch.Generator().manual_seed(args.seed)
    train_set = NutritionDataset(training, scaler, args.image_size, training=True,
                                 augment=not args.no_augment, sampling=args.sampling)
    val_set = NutritionDataset(validation, scaler, args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              worker_init_fn=seed_worker, generator=generator)
    # Independent generator keeps validation worker seeding out of model RNG state.
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    history, best_mae, start, best_model, best_epoch = [], float("inf"), 1, None, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if digest != checkpoint["manifest_sha256"]:
            raise ValueError("Cannot resume with a different manifest")
        # Checkpoints from the original release used one random view per dish.
        if args.sampling != checkpoint["config"].get("sampling", "one-per-dish"):
            raise ValueError("Resume sampling policy differs; start a new experiment")
        for key in ("batch_size", "image_size", "lr", "weight_decay", "dropout", "seed",
                    "num_workers", "amp", "no_standardize", "no_augment", "data_kind"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp_scaler.load_state_dict(checkpoint["amp_scaler"])
        scaler = LabelScaler(**checkpoint["label_scaler"])
        train_set.scaler = val_set.scaler = scaler
        train_mean = checkpoint["train_mean"]
        history = checkpoint["history"]
        best_mae, best_model, best_epoch = (checkpoint["best_mae"], checkpoint["best_model"],
                                            checkpoint["best_epoch"])
        start = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"], generator)
        if start > args.epochs:
            raise ValueError("epochs must exceed checkpoint epoch")
        # Recover the full no-improvement streak, including checkpoints from the
        # release before patience was added. Ties count as no improvement.
        if args.patience and checkpoint["epoch"] - best_epoch >= args.patience:
            raise ValueError("Checkpoint already meets early-stopping patience; evaluate best.pt "
                             "or explicitly increase/disable patience in a documented new run")
    save_json(out / "config.json", config)
    save_json(out / "label_scaler.json", scaler.state_dict())
    save_json(out / "environment.json", environment(device))
    save_json(out / "data_audit.json", {"manifest_sha256": digest, "splits": split_summary(records),
                                      "training_sampling": args.sampling,
                                      "training_samples_per_epoch": len(train_set),
                                      "training_batches_per_epoch": len(train_loader),
                                      "training_loss_weighting": (
                                          "equal images" if args.sampling == "all-images" else "equal dishes"),
                                      "parameters": sum(x.numel() for x in model.parameters()),
                                      "data_kind": args.data_kind})
    print(f"Sampling={args.sampling}; training samples/epoch={len(train_set)}; "
          f"batches/epoch={len(train_loader)}; validation images={len(val_set)}", flush=True)
    print(f"Early stopping: patience={args.patience}, min_delta=0, "
          "monitor=validation dish calorie MAE (kcal); 0 disables patience", flush=True)
    stop_reason = "max_epochs"
    for epoch in range(start, args.epochs + 1):
        began = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        print(f"Epoch {epoch}/{args.epochs}: training", flush=True)
        for step, (x, y, _, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total += loss.item() * y.numel()
            count += y.numel()
            if step % args.log_every == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs}, batch {step}/{len(train_loader)}, "
                      f"train_loss={total / count:.6f}", flush=True)
        print(f"Epoch {epoch}/{args.epochs}: validating all images", flush=True)
        collected = collect_predictions(model, val_loader, scaler, device)
        _, vy, vp = aggregate_dishes(collected["ids"], collected["y"], collected["pred"])
        val_metrics = regression_metrics(vy, vp)
        mae = val_metrics["calories"]["MAE"]
        row = {"epoch": epoch, "train_loss": total / count, "val_loss": collected["loss"],
               "val_calories_MAE": mae, "seconds": time.perf_counter() - began}
        history.append(row)
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = epoch - best_epoch
        should_stop = args.patience > 0 and bad_epochs >= args.patience
        state = {"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "amp_scaler": amp_scaler.state_dict(),
                 "label_scaler": scaler.state_dict(), "config": config,
                 "manifest_sha256": digest, "train_mean": train_mean, "history": history,
                 "best_mae": best_mae, "best_epoch": best_epoch, "best_model": best_model,
                 "early_stopping": {"monitor": "val_dish_calories_MAE", "mode": "min",
                                    "patience": args.patience, "min_delta": 0.0,
                                    "bad_epochs": bad_epochs, "stopped": should_stop},
                 "rng": rng_state(generator)}
        atomic_checkpoint(out / "last.pt", state)
        # best.pt contains a compact inference checkpoint; last.pt contains resume state.
        atomic_checkpoint(out / "best.pt", {k: v for k, v in state.items()
                                           if k in {"format_version", "label_scaler", "config",
                                                    "manifest_sha256", "train_mean"}} |
                          {"epoch": best_epoch, "model": best_model})
        with (out / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        save_json(out / "history.json", history)
        print(json.dumps(row), flush=True)
        if args.patience:
            print(f"Epochs without improvement: {bad_epochs}/{args.patience}", flush=True)
        if should_stop:
            stop_reason = "early_stopping"
            print(f"Early stopping at epoch {epoch}: no lower validation dish calorie MAE "
                  f"for {bad_epochs} consecutive epochs. Best epoch={best_epoch}.", flush=True)
            break
    save_json(out / "training_summary.json", {
        "data_kind": args.data_kind, "manifest_sha256": digest,
        "stop_reason": stop_reason, "requested_total_epochs": args.epochs,
        "completed_total_epochs": history[-1]["epoch"], "best_epoch": best_epoch,
        "best_val_dish_calories_MAE": best_mae, "patience": args.patience,
        "epochs_without_improvement": history[-1]["epoch"] - best_epoch,
        "resumed_from": args.resume,
    })
    print("Exporting loss curves and validation predictions from the best model...", flush=True)
    plot_history(history, out, args.data_kind, not args.no_standardize)
    model.load_state_dict(best_model)
    collected = collect_predictions(model, val_loader, scaler, device)
    report_predictions(out / "val", collected,
                       {"data_kind": args.data_kind, "split": "val", "checkpoint_epoch": best_epoch,
                        "manifest_sha256": digest}, train_mean)
    print(f"Best epoch={best_epoch}, validation calorie MAE={best_mae:.6f} kcal. Test not evaluated.")


if __name__ == "__main__":
    main()
