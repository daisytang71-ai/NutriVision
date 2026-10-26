from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def make_transform(size=224, train=False, aug=None):
    aug = aug or {}
    ops = [transforms.Resize((size, size))]
    if train:
        if float(aug.get("horizontal_flip_p", .5)) > 0:
            ops.append(transforms.RandomHorizontalFlip(float(aug.get("horizontal_flip_p", .5))))
        if float(aug.get("rotation_degrees", 10)) > 0:
            ops.append(transforms.RandomRotation(float(aug.get("rotation_degrees", 10))))
        if float(aug.get("brightness", .15)) > 0:
            ops.append(transforms.ColorJitter(brightness=float(aug.get("brightness", .15))))
    ops += [transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406], [.229,.224,.225])]
    return transforms.Compose(ops)

def load_manifest(path, image_root, image_col, dish_col, split_col, targets):
    path = Path(path).expanduser().resolve()
    if not path.exists(): raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required = [image_col, dish_col, split_col] + list(targets)
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing manifest columns {missing}; found {list(df.columns)}")
    df = df.copy()
    df[dish_col] = df[dish_col].astype(str)
    splitmap = {"train":"train","val":"val","valid":"val","validation":"val","test":"test"}
    df["__split"] = df[split_col].astype(str).str.lower().str.strip().map(splitmap)
    df = df[df["__split"].notna()].copy()
    for c in targets: df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[list(targets)].isna().any().any(): raise ValueError("Missing or nonnumeric nutrition labels found.")
    root = Path(image_root).expanduser().resolve() if image_root else None
    resolved = []
    missing_paths = []
    for raw in df[image_col].astype(str):
        p = Path(raw).expanduser()
        if not p.is_absolute():
            candidate = (path.parent / p).resolve()
            if not candidate.exists() and root is not None: candidate = (root / p).resolve()
        else: candidate = p
        resolved.append(str(candidate))
        if not candidate.exists(): missing_paths.append(str(candidate))
    df["__path"] = resolved
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} image paths missing. Examples: {missing_paths[:8]}")
    overlap = df.groupby(dish_col)["__split"].nunique()
    overlap = overlap[overlap > 1].index.tolist()
    if overlap: raise ValueError(f"Dish IDs occur in multiple splits (data leakage): {overlap[:10]}")
    label_counts = df.groupby(dish_col)[list(targets)].nunique()
    inconsistent = label_counts[(label_counts > 1).any(axis=1)].index.tolist()
    if inconsistent: raise ValueError(f"Labels differ between frames for dishes: {inconsistent[:10]}")
    print(df.groupby("__split").agg(frames=(image_col,"size"), dishes=(dish_col,"nunique")))
    return df

def target_stats(train_df, dish_col, targets):
    labels = train_df[[dish_col] + list(targets)].drop_duplicates(dish_col)[list(targets)]
    mean = labels.mean().to_numpy(dtype=np.float32)
    std = labels.std(ddof=0).to_numpy(dtype=np.float32)
    std[std < 1e-8] = 1.0
    return mean, std

class FrameDataset(Dataset):
    def __init__(self, df, targets, transform, mean, std, dish_col):
        self.df = df.reset_index(drop=True); self.targets = list(targets)
        self.transform = transform; self.mean = np.asarray(mean); self.std = np.asarray(std)
        self.dish_col = dish_col
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["__path"]) as im: x = self.transform(im.convert("RGB"))
        y = r[self.targets].to_numpy(dtype=np.float32)
        y = (y-self.mean)/self.std
        return x, torch.tensor(y, dtype=torch.float32), str(r[self.dish_col]), str(r["__path"])

class DummyDataset(Dataset):
    def __init__(self, n=64, size=224, seed=42):
        self.n=n; self.size=size; self.rng=np.random.default_rng(seed)
        self.y=self.rng.normal(size=(n,4)).astype("float32")
    def __len__(self): return self.n
    def __getitem__(self,i):
        return torch.rand(3,self.size,self.size), torch.tensor(self.y[i]), f"dish{i}", f"dummy{i}.jpg"
