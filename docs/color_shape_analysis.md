# Color and food shape analysis

## Included evidence

Results are under `results/color_shape/`. The six new per-frame color prediction CSVs and all 100 mask/overlay images are in the separately supplied `NutriVision_Color_Shape_Results.zip` (intended as a GitHub Release asset). The archive is not bundled twice in Git. After downloading that asset, extract its `predictions/` directory into `results/color_shape/predictions/` to recompute color metrics. Original predictions already live in `predictions/unified/`.

Reported color results cover all 676 test dishes / 20,686 frames, with fixed CNN, ResNet18 FT and MobileNetV2 FT checkpoints. Saturation factor is 0.3. Grayscale converts back to three RGB channels. Resize precedes perturbation and ImageNet normalization. No retraining or prediction clipping occurs. Single-frame dish-balanced and dish-level averaged metrics are separate.

Shape results use 100 randomly sampled distinct test dishes (seed 42), lexicographically first frame per dish. A FoodSeg103-trained DeepLabV3+ MobileNetV2 supplies automatic masks. Source: https://huggingface.co/mawiie/food-segmentation-mobilenet . Download `best_model.pth` and store locally as `food_segmentation.pth` or set `NV_SEGMENTATION_CHECKPOINT`. The original download used hf-mirror.com because the primary host was unreachable. Weight source and limitations are recorded in shape/method.json. Do not commit this checkpoint to Git.

Visual screening retained 70 masks and excluded 30. `src/analysis/shape_qa_decisions.csv` stores dish/frame identities, review reasons and mask hashes. `shape_features_reviewed.csv` stores all 100 records including exclusions. Masks are approximate and do not have human ground-truth IoU validation. The review was AI-assisted visual inspection, not a separately validated annotation study. New masks or a changed sample require new screening decisions.

## Setup and paths

```bash
python -m pip install -r requirements-analysis.txt
```

Default paths from the repository root:

| Input | Default | Override |
|---|---|---|
| Manifest | data/manifests/full_frame_manifest.csv | NV_MANIFEST |
| Images | side_frames_valid/ | NV_DATA_ROOT |
| Three checkpoints | models/ | NV_MODELS_DIR |
| Original CSVs | predictions/unified/ | NV_PRED_DIR |
| Analysis outputs | results/color_shape/ | NV_ANALYSIS_OUT |
| Food segmentation checkpoint | models/food_segmentation.pth | NV_SEGMENTATION_CHECKPOINT |

The recorded Python/package versions are in `docs/analysis_environment.json`. Requirements specify ranges for setup convenience; not every version combination or platform has been tested.

## Recompute from existing predictions and reviewed features

First restore the six color prediction CSVs from the Release archive as described above. To preserve committed results, copy the analysis result folder to a fresh local directory and point NV_ANALYSIS_OUT there.

```bash
python src/analysis/run_color.py --summarize
python src/analysis/finish_analysis.py --shape-only
python src/analysis/finish_analysis.py
```

The first command recalculates color metrics. The second recalculates shape associations from recorded features and QA decisions. The third generates color figures, paired whole-dish bootstrap intervals and a result archive. These steps do not run model inference. The shape script uses the original predictions in NV_PRED_DIR.

## Optional new inference

```bash
export NV_DATA_ROOT=/path/to/side_frames_valid
export NV_MODELS_DIR=/path/to/checkpoints
export NV_ANALYSIS_OUT=work/new_color_shape
python src/analysis/run_color.py --smoke
python src/analysis/run_color.py
python src/analysis/run_shape.py --count 100
python src/analysis/finish_analysis.py --shape-only
python src/analysis/finish_analysis.py
```

In PowerShell set `$env:NV_DATA_ROOT="..."` etc. instead of `export`. The smoke run checks 16 original images against supplied predictions and runs 64 frames for new perturbations. A fresh inference output requires images/checkpoints. Full color inference can take tens of minutes on CPU. It is unnecessary if you only need to inspect existing results. Existing full prediction files require `--resume` or a fresh output directory.

For optional ONNX acceleration install `requirements-onnx.txt`, run `python src/analysis/prepare_onnx.py`, then use `run_color.py --ort`. CPU PyTorch and ONNX FP32 results were checked for numerical agreement. The originally reported run used PyTorch for its first 3,200 frames and ONNX for the remainder. Runtime metadata retains that provenance.

## Interpretation

Area fraction is food pixels divided by image pixels. Aspect ratio uses the minimum-area rectangle. Circularity uses 4*pi*A/P² over external contours; contour complexity is its inverse, not independent evidence. Associations are exploratory. Food type, framing, viewpoint, portion size and mask selection can confound them. Partial rank correlations control only for the true target. Do not claim causal shape dependence or improved prediction accuracy from these analyses alone. Rotation/flip are not reported as completed experiments.
