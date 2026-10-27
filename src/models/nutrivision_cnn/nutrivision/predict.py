"""Predict one RGB image using a saved inference checkpoint."""
import argparse
import json
from PIL import Image
import torch
from . import TARGETS, UNITS
from .data import LabelScaler, image_tensor
from .model import CustomCNN
from .utils import device_from, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    ckpt = load_checkpoint(a.checkpoint)
    device = device_from(a.device)
    model = CustomCNN(ckpt["config"]["dropout"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    scaler = LabelScaler(**ckpt["label_scaler"])
    with Image.open(a.image) as im:
        x = image_tensor(im, ckpt["config"]["image_size"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        values = scaler.decode(model(x).cpu().numpy())[0]
    print(json.dumps({"data_kind": ckpt["config"]["data_kind"],
                      "policy": "raw predictions, no clipping",
                      "prediction": {k: {"value": float(v), "unit": u}
                                     for k, v, u in zip(TARGETS, values, UNITS)}}, indent=2))


if __name__ == "__main__":
    main()
