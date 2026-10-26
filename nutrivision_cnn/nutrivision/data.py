"""Validated CSV data interface, dish sampling, train-only label scaling."""
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from . import TARGETS

FIELDS = ["image_path", "dish_id", "group_id", "split", *TARGETS]
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, r["dish_id"] if k == "group_id" else "") for k in FIELDS}
                         for r in records)


def load_manifest(path, root=None, verify_images=True):
    path = Path(path).resolve()
    root = Path(root).resolve() if root else path.parent
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - {"group_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("Empty manifest")
    dish_info, group_split, images = {}, {}, set()
    for r in records:
        if not r["dish_id"].strip() or not r["image_path"].strip():
            raise ValueError("Empty dish_id or image_path")
        r["group_id"] = r.get("group_id") or r["dish_id"]
        if r["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {r['split']}")
        y = np.array([float(r[k]) for k in TARGETS], dtype=np.float64)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError(f"Invalid nutrition labels: {r['dish_id']}")
        for k, value in zip(TARGETS, y):
            r[k] = float(value)
        info = (r["split"], r["group_id"], tuple(y))
        if r["dish_id"] in dish_info and dish_info[r["dish_id"]] != info:
            raise ValueError(f"Dish leakage or inconsistent labels/group: {r['dish_id']}")
        dish_info[r["dish_id"]] = info
        if r["group_id"] in group_split and group_split[r["group_id"]] != r["split"]:
            raise ValueError(f"Physical plate group crosses splits: {r['group_id']}")
        group_split[r["group_id"]] = r["split"]
        p = (root / r["image_path"]).resolve()
        if p in images:
            raise ValueError(f"Duplicate image path: {p}")
        images.add(p)
        r["_path"] = str(p)
        if verify_images:
            if not p.is_file():
                raise FileNotFoundError(p)
            with Image.open(p) as im:
                im.verify()
    if {r["split"] for r in records} != {"train", "val", "test"}:
        raise ValueError("Manifest must contain non-empty train, val and test splits")
    return records


def unique_dishes(records):
    return list({r["dish_id"]: r for r in records}.values())


def split_summary(records):
    return {s: {"images": len(rs := [r for r in records if r["split"] == s]),
                "dishes": len({r["dish_id"] for r in rs}),
                "groups": len({r["group_id"] for r in rs})} for s in ("train", "val", "test")}


class LabelScaler:
    def __init__(self, mean, std, enabled=True):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.enabled = enabled

    @classmethod
    def fit(cls, records, enabled=True):
        if not records or any(r["split"] != "train" for r in records):
            raise ValueError("Scaler must be fitted on training records only")
        y = np.array([[r[k] for k in TARGETS] for r in unique_dishes(records)], dtype=np.float64)
        mean, std = y.mean(0), y.std(0, ddof=0)
        # A constant dimension uses unit scale (no division by zero).
        std[std < 1e-6] = 1.0
        return cls(mean if enabled else np.zeros(4), std if enabled else np.ones(4), enabled)

    def encode(self, y):
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std

    def decode(self, y):
        return np.asarray(y) * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "enabled": self.enabled}


def image_tensor(image, size, augment=False):
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Resize whole image: no crop that could remove part of the meal.
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    if augment:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.rotate(random.uniform(-10, 10), resample=Image.Resampling.BILINEAR,
                             fillcolor=(124, 116, 104))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
    x = (np.asarray(image, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class NutritionDataset(Dataset):
    def __init__(self, records, scaler, image_size=224, training=False, augment=True,
                 sampling="all-images"):
        if sampling not in {"all-images", "one-per-dish"}:
            raise ValueError(f"Unknown sampling policy: {sampling}")
        self.records, self.scaler = records, scaler
        self.image_size, self.training, self.augment = image_size, training, augment
        self.sampling = sampling
        groups = defaultdict(list)
        for r in records:
            groups[r["dish_id"]].append(r)
        # All images are visited once per epoch by the shuffled DataLoader.
        # The legacy policy remains explicit for reproducing earlier runs.
        self.items = (list(groups.values()) if training and sampling == "one-per-dish"
                      else [[r] for r in records])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        r = (random.choice(self.items[index])
             if self.training and self.sampling == "one-per-dish" else self.items[index][0])
        with Image.open(r["_path"]) as im:
            x = image_tensor(im, self.image_size, self.training and self.augment)
        y = self.scaler.encode([r[k] for k in TARGETS])
        return x, torch.tensor(y, dtype=torch.float32), r["dish_id"], r["image_path"]
