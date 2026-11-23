# Unified Six-Model Evaluation

The evaluation pipeline compares six models on the same **676 test dishes and 20,686 test frames**. It exports per-frame predictions in a common CSV format and computes metrics under explicitly separated evaluation protocols.

The workflow contains three parts:

1. **Model comparison:** calorie MAE, RMSE and R², together with MAE for all four nutritional targets.
2. **Evaluation-level comparison:** single-frame, dish-balanced evaluation as the primary protocol, and dish-level prediction averaging as the secondary protocol.
3. **Error analysis:** predicted versus actual values, errors by calorie range, and representative best and worst predictions.

The primary metric is **calorie MAE under single-frame, dish-balanced evaluation**. Protein, carbohydrate and fat MAE are secondary metrics.

All learned checkpoints were selected using validation data. Training monitored validation dish-level calorie MAE for checkpoint selection and early stopping. Test results were not used to select checkpoints.

## Models and Checkpoints

| Model | Checkpoint filename | Validation dish-level calorie MAE |
|---|---|---:|
| Mean predictor | No checkpoint | Not applicable |
| Custom CNN | `cnn_best.pt` | 81.92 kcal |
| ResNet18 baseline, frozen backbone | `Resnet_baseline.pt` | 91.38 kcal |
| ResNet18 fine-tuned | `Resnet_finetune.pt` | 56.54 kcal |
| MobileNetV2 baseline, frozen backbone | `MobileNetV2_baseline.pt` | 88.58 kcal |
| MobileNetV2 fine-tuned | `MobileNetV2_finetune.pt` | 59.33 kcal |

The mean predictor returns the nutritional means of unique training dishes. It provides a non-learning reference for comparison with the five learned models.

Model weights are stored separately from the source code. The project’s model download location is [GitHub Releases](https://github.com/daisytang71-ai/NutriVision/releases).

## Prediction Files

Prediction files are stored in `predictions/unified/`, with one UTF-8 CSV per model.

Each file uses the following schema:

```text
dish_id,frame_path,true_calories,pred_calories,true_protein,pred_protein,true_carbs,pred_carbs,true_fat,pred_fat
```

Each row represents one frame.

- `dish_id` identifies the dish.
- `frame_path` is relative to `side_frames_valid/`.
- Calories use kilocalories.
- Protein, carbohydrates and fat use grams.
- Predictions are restored to their original units before export.
- Negative predictions are retained without clipping.

The prediction CSV uses `true_carbs` and `pred_carbs` for the carbohydrate target. The input manifest uses `carbohydrate`.

`run_metadata.json` records model metadata and inference times.

The reported evaluation metrics are calculated by:

```text
src/evaluation/compute_unified_metrics.py
```

The older `evaluate.py` interface uses different column names and frame-equal scoring. Use `compute_unified_metrics.py` to reproduce the reported primary and secondary results.

## Evaluation Protocols

### Primary — Single-frame, dish-balanced evaluation

Every frame receives an independent prediction. Errors are averaged within each dish and then across the 676 dishes.

Every dish receives equal weight, regardless of its number of frames.

Let $D=676$ denote the number of test dishes. For dish $i$, let $n_i$ be its number of frames, $y_i$ its ground-truth value, and $e_{ij}=\hat{y}_{ij}-y_i$ the prediction error for frame $j$.

$$
\mathrm{MAE}
=
\frac{1}{D}
\sum_{i=1}^{D}
\left(
\frac{1}{n_i}
\sum_{j=1}^{n_i}|e_{ij}|
\right)
$$

$$
\mathrm{MSE}
=
\frac{1}{D}
\sum_{i=1}^{D}
\left(
\frac{1}{n_i}
\sum_{j=1}^{n_i}e_{ij}^{2}
\right)
$$

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

$$
R^{2}
=
1-\frac{\mathrm{MSE}}{\mathrm{Var}_{\mathrm{db}}(y)}
$$

The dish-balanced variance is:

$$
\mathrm{Var}_{\mathrm{db}}(y)
=
\frac{1}{D}\sum_{i=1}^{D}(y_i-\bar{y})^{2},
\qquad
\bar{y}=\frac{1}{D}\sum_{i=1}^{D}y_i
$$

Squared errors are averaged within dishes and then across dishes before taking the square root. Individual dish RMSE values are not averaged.

### Secondary — Dish-level prediction averaging

Predictions from all frames of each dish are averaged first. MAE, RMSE and R² are then calculated over the 676 dish-level predictions.

This protocol measures the benefit of combining multiple views of the same dish.

### Reference — Frame-level evaluation

All 20,686 frames receive equal weight. Dishes with more frames therefore contribute more heavily.

This protocol serves as a supplementary reference. It is reported separately from the primary and secondary protocols.

## Results and Figures

Evaluation outputs are stored in `results/unified/`.

| File | Contents |
|---|---|
| `metrics_all_levels.csv` | Complete metrics by model, target and evaluation level |
| `table1_six_model_calorie.csv` | Primary calorie MAE, RMSE and R² for six models |
| `table1b_all_targets_all_metrics.csv` | Primary metrics for all four targets |
| `table2_four_targets_MAE.csv` | Primary MAE for calories, protein, carbohydrates and fat |
| `table3_single_vs_dish.csv` | Single-frame and dish-level comparison |
| `error_by_calorie_range.csv` | Errors grouped by true calorie range |
| `robust_subset_metrics.csv` | Full-test and supplementary subset metrics |
| `extreme_dishes_over_900kcal.csv` | Test dishes with true calories of at least 900 kcal |
| `best_worst_cases.csv` | Representative predictions with small and large errors |
| `dish_predictions_error_analysis.csv` | Dish-level predictions used for error analysis |
| `summary.json` | Evaluation definitions and summary results |
| `figures/` | Eight evaluation figures |
| `REPORT.md` | Evaluation report |

### Extreme Calorie Labels

Two original dataset labels are approximately **9,486 and 9,170 kcal**. Their squared errors strongly affect full-test RMSE and R².

The complete 676-dish test results are retained. Supplementary analysis also reports results for the **665 dishes with true calories below 900 kcal**.

This subset excludes all 11 dishes with labels of at least 900 kcal, including the two extreme cases. It does not replace the full-test evaluation.

The learned models achieve dish-level R² values of approximately **0.63–0.85** on this supplementary subset. These values belong to the secondary, dish-level prediction-averaging protocol.

## Reproduction

Run commands from the repository root.

The evaluation environment requires Python, PyTorch, torchvision, NumPy, pandas, Pillow, Matplotlib and scikit-learn. The metric script requires **pandas 2.2 or later**. The unified inference script runs on CPU.

The image directory and checkpoints are stored outside Git:

```text
/path/to/side_frames_valid/
/path/to/checkpoints/
```

### 1. Generate Predictions

Set the paths to the prepared manifest, image directory and checkpoint directory.

For macOS or Linux:

```bash
export NV_MANIFEST="data/manifests/full_frame_manifest.csv"
export NV_DATA_ROOT="/path/to/side_frames_valid"
export NV_MODELS_DIR="/path/to/checkpoints"
export NV_IMG_ROOT="$NV_DATA_ROOT"

python src/evaluation/run_unified_evaluation.py
```

For Windows PowerShell:

```powershell
$env:NV_MANIFEST="data/manifests/full_frame_manifest.csv"
$env:NV_DATA_ROOT="D:/Nutrition5k/side_frames_valid"
$env:NV_MODELS_DIR="D:/Nutrition5k/checkpoints"
$env:NV_IMG_ROOT=$env:NV_DATA_ROOT

python src/evaluation/run_unified_evaluation.py
```

The default output directory is `predictions/unified/`.

This step regenerates model predictions. The repository already includes prediction CSVs, so metric and figure reproduction can begin at Step 2 without repeating inference.

### 2. Compute Metrics

```bash
python src/evaluation/compute_unified_metrics.py
```

This command generates the evaluation tables, supplementary analyses and `summary.json`.

### 3. Generate Figures

```bash
python src/evaluation/make_unified_figures.py
```

This command writes figures to `results/unified/figures/`. Food-image case illustrations use the image directory specified by `NV_IMG_ROOT`.

### 4. Generate the Report

```bash
python src/evaluation/generate_unified_report.py
```

This command writes `results/unified/REPORT.md`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `NV_REPO_ROOT` | Repository root |
| `NV_MANIFEST` | Input manifest CSV |
| `NV_DATA_ROOT` | Prepared image root for inference |
| `NV_MODELS_DIR` | Checkpoint directory |
| `NV_OUT_DIR` | Prediction output directory |
| `NV_PRED_DIR` | Prediction input directory for metric calculation |
| `NV_RES_DIR` | Metric and report output directory |
| `NV_FIG_DIR` | Figure output directory |
| `NV_IMG_ROOT` | Image directory for visual case analysis |

To preserve committed results during a new reproduction run, direct outputs to a separate directory. Keep the prediction and result paths consistent across all four steps.
