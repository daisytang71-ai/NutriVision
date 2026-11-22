"""NutriVision unified evaluation - stage 1: export per-frame predictions.

All six models are evaluated with ONE script on the SAME 676 test dishes /
20,686 test frames, and export the SAME CSV schema:

    dish_id,frame_path,true_calories,pred_calories,
    true_protein,pred_protein,true_carbs,pred_carbs,true_fat,pred_fat

Models
    1  MeanPredictor            non-learning baseline (training-dish mean)
    2  CustomCNN                cnn_best.pt
    3  ResNet18_baseline        Resnet_baseline.pt  (frozen backbone, stage 1)
    4  ResNet18_finetune        Resnet_finetune.pt  (partial fine-tune, stage 2)
    5  MobileNetV2_baseline     MobileNetV2_baseline.pt
    6  MobileNetV2_finetune     MobileNetV2_finetune.pt

Checkpoint selection was done on the VALIDATION split by the training runs
(val_dish_calories_MAE_kcal / early stopping on validation dish-level MAE);
no test-set information was used to choose any checkpoint.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18

ROOT = Path(os.environ.get("NV_REPO_ROOT", Path(__file__).resolve().parents[2]))
REPO = ROOT  # repo checkout containing src/models/nutrivision_cnn
MODELS = Path(os.environ.get("NV_MODELS_DIR", ROOT / "models"))
MANIFEST = Path(os.environ.get("NV_MANIFEST", ROOT / "data" / "manifests" / "full_frame_manifest.csv"))
DATA_ROOT = Path(os.environ.get("NV_DATA_ROOT", ROOT / "side_frames_valid"))
OUT = Path(os.environ.get("NV_OUT_DIR", ROOT / "predictions" / "unified"))
TARGETS = ["calories", "protein", "carbohydrate", "fat"]
FIELDS = ["dish_id", "frame_path", "true_calories", "pred_calories",
          "true_protein", "pred_protein", "true_carbs", "pred_carbs",
          "true_fat", "pred_fat"]
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Custom CNN lives in the repository source tree.
sys.path.insert(0, str(REPO / "src" / "models" / "nutrivision_cnn"))
from nutrivision.model import CustomCNN  # noqa: E402


def cnn_image_tensor(image, size=224):
    """Exact preprocessing used by the CNN training/evaluation code."""
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = (np.asarray(image, dtype=np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def tv_transform(size=224):
    """Exact preprocessing used by the ResNet18 / MobileNetV2 code."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


class FrameDataset(Dataset):
    def __init__(self, df, kind, size=224):
        self.df = df.reset_index(drop=True)
        self.kind = kind
        self.tf = tv_transform(size) if kind != "cnn" else None
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["__path"]) as im:
            x = cnn_image_tensor(im, self.size) if self.kind == "cnn" else self.tf(im.convert("RGB"))
        y = np.asarray([r[c] for c in TARGETS], dtype=np.float32)
        return x, y, str(r["dish_id"]), str(r["image_path"])


def load_manifest():
    df = pd.read_csv(MANIFEST)
    df["__path"] = df["image_path"].map(lambda p: str(DATA_ROOT / p))
    test = df[df["split"] == "test"].copy()
    assert len(test) == 20686 and test["dish_id"].nunique() == 676, "unexpected test split"
    assert all(Path(p).is_file() for p in test["__path"]), "missing test image"
    # Training-dish statistics (identical to the checkpoint scalers).
    tr = df[df["split"] == "train"].drop_duplicates("dish_id")[TARGETS]
    mean = tr.mean().to_numpy(np.float64)
    std = tr.std(ddof=0).to_numpy(np.float64)
    std[std < 1e-8] = 1.0
    return test, mean, std


def build_model(spec, mean, std):
    """Return (model, kind, decode_mean, decode_std, metadata)."""
    ck = torch.load(MODELS / spec["file"], map_location="cpu", weights_only=True)
    if spec["kind"] == "cnn":
        model = CustomCNN(ck["config"]["dropout"])
        model.load_state_dict(ck["model"])
        dmean = np.asarray(ck["label_scaler"]["mean"], dtype=np.float64)
        dstd = np.asarray(ck["label_scaler"]["std"], dtype=np.float64)
        meta = dict(epoch=ck["epoch"], selection="validation dish-level calorie MAE (early stopping)")
    else:
        if spec["kind"] == "resnet":
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
        else:
            model = mobilenet_v2(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 4)
        model.load_state_dict(ck["model_state_dict"])
        dmean = np.asarray(ck["target_mean"], dtype=np.float64)
        dstd = np.asarray(ck["target_std"], dtype=np.float64)
        meta = dict(stage=ck["stage"], epoch=ck["epoch"],
                    val_dish_calories_MAE_kcal=float(ck["val_dish_calories_MAE_kcal"]),
                    selection="validation dish-level calorie MAE")
    model.eval()
    assert np.allclose(dmean, mean, atol=1e-3) and np.allclose(dstd, std, atol=1e-3), "scaler mismatch"
    return model, spec["kind"], dmean, dstd, meta


def run_model(name, spec, df, mean, std, batch_size, workers, threads, limit, out_dir):
    torch.set_num_threads(threads)
    model, kind, dmean, dstd, meta = build_model(spec, mean, std)
    ds = FrameDataset(df.head(limit) if limit else df, kind)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=False, persistent_workers=workers > 0)
    out_path = out_dir / f"predictions_{name}.csv"
    t0 = time.time()
    n = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f, torch.inference_mode():
        w = csv.writer(f)
        w.writerow(FIELDS)
        for x, y, dish, path in loader:
            if kind == "mean":
                raise RuntimeError("mean predictor is handled separately")
            p = model(x).numpy().astype(np.float64) * dstd + dmean
            if not np.isfinite(p).all():
                raise ValueError(f"non-finite prediction from {name}")
            for i in range(len(x)):
                w.writerow([dish[i], path[i],
                            float(y[i, 0]), float(p[i, 0]),
                            float(y[i, 1]), float(p[i, 1]),
                            float(y[i, 2]), float(p[i, 2]),
                            float(y[i, 3]), float(p[i, 3])])
            n += len(x)
            if n % 2000 < batch_size:
                print(f"  {name}: {n}/{len(ds)} frames  {time.time()-t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"DONE {name}: {n} frames in {dt:.1f}s -> {out_path}", flush=True)
    return dict(model=name, frames=n, dishes=int(df.head(limit)["dish_id"].nunique()) if limit else 676,
                seconds=round(dt, 1), **meta)


def run_mean(df, mean, out_dir):
    """Sixth model: predict the training-dish mean for every frame (no learning)."""
    out_path = out_dir / "predictions_MeanPredictor.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in df.itertuples(index=False):
            w.writerow([r.dish_id, r.image_path,
                        r.calories, mean[0], r.protein, mean[1],
                        r.carbohydrate, mean[2], r.fat, mean[3]])
    print(f"DONE MeanPredictor: {len(df)} frames -> {out_path}", flush=True)
    return dict(model="MeanPredictor", frames=len(df), dishes=676, seconds=0.0,
                selection="non-learning baseline; training-dish means only")


SPECS = [
    ("CustomCNN", dict(file="cnn_best.pt", kind="cnn")),
    ("ResNet18_baseline", dict(file="Resnet_baseline.pt", kind="resnet")),
    ("ResNet18_finetune", dict(file="Resnet_finetune.pt", kind="resnet")),
    ("MobileNetV2_baseline", dict(file="MobileNetV2_baseline.pt", kind="mobilenet")),
    ("MobileNetV2_finetune", dict(file="MobileNetV2_finetune.pt", kind="mobilenet")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N frames")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only these model names")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: predictions/unified, or predictions/smoke with --limit)")
    a = ap.parse_args()
    for name, value in [("--manifest", MANIFEST), ("--data-root", DATA_ROOT),
                        ("--models-dir (or NV_MODELS_DIR)", MODELS)]:
        if not Path(value).exists():
            ap.error(f"{name} not found: {value}. Point it at the manifest CSV, the "
                     f"side_frames_valid image root, and the checkpoint directory.")
    out_dir = a.out_dir or (OUT / ("smoke" if a.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    test, mean, std = load_manifest()
    print(f"test split: {len(test)} frames / {test['dish_id'].nunique()} dishes | "
          f"train mean={mean.round(3).tolist()} std={std.round(3).tolist()}", flush=True)
    meta = []
    if a.only is None or "MeanPredictor" in a.only:
        meta.append(run_mean(test.head(a.limit) if a.limit else test, mean, out_dir))
    for name, spec in SPECS:
        if a.only and name not in a.only:
            continue
        meta.append(run_model(name, spec, test, mean, std, a.batch_size, a.workers,
                              a.threads, a.limit, out_dir))
    import json
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
