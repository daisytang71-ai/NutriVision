"""Paths resolve from the repository root, with optional NV_* overrides."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def env(name,default): return Path(os.environ.get(name,str(default))).expanduser().resolve()
MANIFEST=env('NV_MANIFEST',ROOT/'data/manifests/full_frame_manifest.csv')
IMAGE_ROOT=env('NV_DATA_ROOT',ROOT/'side_frames_valid')
MODELS_DIR=env('NV_MODELS_DIR',ROOT/'models')
PRED_DIR=env('NV_PRED_DIR',ROOT/'predictions/unified')
OUT=env('NV_ANALYSIS_OUT',ROOT/'results/color_shape')
ONNX_DIR=env('NV_ONNX_DIR',MODELS_DIR/'onnx')
SEGMENTATION_CHECKPOINT=env('NV_SEGMENTATION_CHECKPOINT',MODELS_DIR/'food_segmentation.pth')
