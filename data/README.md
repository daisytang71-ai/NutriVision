# NutriVision Dataset

NutriVision uses Nutrition5k side-view RGB frames with dish-level nutritional labels. The finalized dataset contains **140,662 frames from 4,605 dishes**.

## Dataset Splits

| Split | Frames | Dishes |
|---|---:|---:|
| Training | 102,087 | 3,340 |
| Validation | 17,889 | 589 |
| Test | 20,686 | 676 |
| **Total** | **140,662** | **4,605** |

The split assignments are fixed. Each dish belongs to exactly one split, and all frames from that dish remain together. The validation set comes from the training portion.

## Repository Files

| File | Contents |
|---|---|
| `manifests/full_frame_manifest.csv` | Image paths, dish IDs, nutritional labels and split assignments |
| `../splits/train_ids.txt` | Training dish IDs |
| `../splits/val_ids.txt` | Validation dish IDs |
| `../splits/test_ids.txt` | Test dish IDs |
| `checks/local_data_audit.json` | Recorded data audit results |
| `checks/SHA256SUMS.json` | Manifest and split-file checksums |

The manifest is the authoritative source for image–label correspondence and split assignments.

## Manifest Schema

The manifest contains these columns:

```text
dish_id,camera,frame_idx,image_path,calories,protein,carbohydrate,fat,split
```

- `dish_id` identifies the dish.
- `camera` identifies the source camera.
- `frame_idx` identifies the extracted frame.
- `image_path` is relative to `side_frames_valid/`.
- `calories` uses kilocalories.
- `protein`, `carbohydrate` and `fat` use grams.
- `split` contains `train`, `val` or `test`.

All frames belonging to the same dish share the same nutritional labels and split assignment.

## Data Verification

The finalized dataset satisfies the following checks:

- All image paths in the manifest exist in the prepared image directory.
- Image paths are unique.
- Required manifest fields contain no missing values.
- Nutritional labels are consistent within each dish.
- Dish IDs do not overlap across splits.
- The three split-ID files match the manifest assignments.
- All 20,686 test images pass image-file verification.

The manifest preserves the original nutritional values, including the two extreme calorie labels discussed in the evaluation report. Supplementary subset analysis leaves the full-test evaluation unchanged.

## Image Preprocessing

Models use RGB images resized directly to **224 × 224 pixels**, followed by ImageNet channel normalization.

Training uses model-specific augmentation. Validation and test preprocessing are deterministic.

Target-standardization statistics are calculated from **unique training dishes only**. Predictions are converted back to kilocalories and grams before evaluation.

## Evaluation Protocol

### Primary — Single-frame, dish-balanced evaluation

Each frame receives an independent prediction. Absolute errors are averaged within each dish and then across dishes. Every test dish receives equal weight.

For RMSE, squared errors are averaged within each dish and across dishes before taking the square root. R² uses the same dish-balanced weighting.

### Secondary — Dish-level prediction averaging

Frame predictions are averaged within each dish first. Metrics are then calculated across the resulting dish-level predictions.

### Reference — Frame-level evaluation

All frames receive equal weight. This protocol gives dishes with more frames greater influence and serves as a supplementary reference.

The project reports these evaluation protocols separately.

## Local Image Directory

Full image files are stored outside Git. The expected layout is:

```text
data_root/
└── side_frames_valid/
    └── dish_.../
        ├── camera_A_f01.jpg
        ├── camera_A_f02.jpg
        └── ...
```

The manifest path remains:

```text
data/manifests/full_frame_manifest.csv
```

Set the image root to the local `side_frames_valid/` directory when running inference or training. See [data access and directory instructions](DOWNLOAD.md).

## CNN Manifest Compatibility

The repository manifest uses `carbohydrate`. The original CNN training manifest uses `carbs`.

The unified evaluation script reads the repository manifest directly. CNN-only evaluation with this manifest uses the compatibility-aware entry point documented in [CNN testing](../docs/cnn_testing.md).

That entry point checks the split assignments and training statistics and records both manifest hashes. The original training snapshot and checkpoint metadata remain unchanged.

See [unified evaluation](../docs/unified_evaluation.md) for the complete evaluation workflow.
