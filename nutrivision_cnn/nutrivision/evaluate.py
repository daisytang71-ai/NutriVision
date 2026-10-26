"""Standalone evaluation. Does not refit normalization or select a model."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from . import TARGETS
from .data import LabelScaler, NutritionDataset, load_manifest, sha256, split_summary
from .metrics import aggregate_dishes, regression_metrics
from .model import CustomCNN
from .utils import device_from, load_checkpoint, save_json


@torch.inference_mode()
def collect_predictions(model, loader, scaler, device):
    model.eval()
    truth, pred, ids, paths = [], [], [], []
    total, count = 0.0, 0
    for x, y, dish, image in loader:
        out = model(x.to(device)).float().cpu()
        total += torch.square(out - y).sum().item()
        count += y.numel()
        truth.append(scaler.decode(y.numpy()))
        pred.append(scaler.decode(out.numpy()))
        ids.extend(dish)
        paths.extend(image)
    if not count:
        raise ValueError("Empty evaluation loader")
    y, p = np.concatenate(truth), np.concatenate(pred)
    if not np.isfinite(p).all() or not np.isfinite(total):
        raise FloatingPointError("Non-finite predictions/loss")
    return {"y": y, "pred": p, "ids": ids, "paths": paths, "loss": total / count}


def report_predictions(output, collected, provenance, mean_prediction=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = collected
    ids, y, p = aggregate_dishes(c["ids"], c["y"], c["pred"])
    report = {"provenance": provenance, "targets": list(TARGETS),
              "prediction_policy": "raw, no clipping; dish predictions average available image predictions",
              "image_count": len(c["ids"]), "dish_count": len(ids),
              "image_mse_in_training_label_space": c["loss"],
              "dish_metrics": regression_metrics(y, p),
              "image_metrics": regression_metrics(c["y"], c["pred"])}
    if mean_prediction is not None:
        report["train_mean_predictor_dish_metrics"] = regression_metrics(
            y, np.broadcast_to(mean_prediction, y.shape))
    save_json(output / "metrics.json", report)
    for filename, rows in (("image_predictions.csv", zip(c["ids"], c["paths"], c["y"], c["pred"])),
                           ("dish_predictions.csv", zip(ids, [""] * len(ids), y, p))):
        with (output / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "dish_id", "y_cal", "y_p", "y_c", "y_f",
                             "pred_cal", "pred_p", "pred_c", "pred_f"])
            for dish, path, yi, pi in rows:
                writer.writerow([path, dish, *map(float, yi), *map(float, pi)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for i, ax in enumerate(axes.flat):
        ax.scatter(y[:, i], p[:, i], s=10, alpha=0.6)
        lo, hi = min(y[:, i].min(), p[:, i].min()), max(y[:, i].max(), p[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--")
        ax.set(xlabel="Ground truth", ylabel="Prediction", title=TARGETS[i])
    fig.suptitle(f"{provenance['data_kind']} | {provenance['split']} | dish averages")
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_true.png", dpi=150)
    plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("Invalid batch size, worker count or thread count")
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("out-dir must be empty to preserve existing evaluation evidence")
    torch.set_num_threads(args.threads)
    ckpt = load_checkpoint(args.checkpoint)
    if sha256(args.manifest) != ckpt["manifest_sha256"]:
        raise ValueError("Manifest differs from training snapshot; use the saved manifest.csv")
    records = load_manifest(args.manifest, args.data_root)
    scaler = LabelScaler(**ckpt["label_scaler"])
    device = device_from(args.device)
    model = CustomCNN(ckpt["config"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    selected = [r for r in records if r["split"] == args.split]
    dataset = NutritionDataset(selected, scaler, ckpt["config"]["image_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    collected = collect_predictions(model, loader, scaler, device)
    provenance = {"data_kind": ckpt["config"]["data_kind"], "split": args.split,
                  "checkpoint_epoch": ckpt["epoch"], "manifest_sha256": ckpt["manifest_sha256"],
                  "split_summary": split_summary(records)}
    report = report_predictions(args.out_dir, collected, provenance, ckpt["train_mean"])
    print(report["dish_metrics"])


if __name__ == "__main__":
    main()
