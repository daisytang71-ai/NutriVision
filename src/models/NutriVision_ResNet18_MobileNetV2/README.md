# Nutrition5k pretrained-model project

Implements ResNet18 and MobileNetV2 for four-target regression from extracted JPG frames.

## Manifest
Required columns:
`dish_id,camera,frame_idx,image_path,calories,protein,carbohydrate,fat,split`

`split` must use `train`, `val`/`valid`/`validation`, or `test`. All frames from a dish must belong to the same split. The script checks for dish leakage, missing image paths, and inconsistent nutrition labels across frames.

## Configure and run
1. Install packages: `pip install -r requirements.txt`
2. Edit `configs/resnet18.yaml` and `configs/mobilenetv2.yaml`, especially `data.manifest`, optional `data.image_root`, and `output_dir`.
3. Run:
   `python train.py --config configs/resnet18.yaml`
   `python train.py --config configs/mobilenetv2.yaml`
4. Optional synthetic pipeline test (not real performance):
   `python train.py --config configs/resnet18.yaml --dummy`

Use forward slashes in Windows YAML paths, e.g. `D:/Nutrition5k/manifest.csv`.

## Method
- RGB, direct resize to 224×224 (no crop).
- Train augmentation: random horizontal flip, random rotation (±10°), random brightness jitter.
- Validation/test: resize and ImageNet normalization only, no random augmentation.
- Labels standardized using the unique training dishes' mean/std.
- Loss: mean MSE across batch and all four standardized targets.
- Stage 1 trains the regression head; Stage 2 fine-tunes later backbone layers with a lower backbone learning rate.
- Every frame is predicted independently. Frame predictions are averaged per `dish_id` for dish-level metrics.
- Checkpoint selection uses validation **dish-level calorie MAE (kcal)**. Test is used only for final evaluation.
- Reports MAE, RMSE and R² for calories (kcal), protein, carbohydrate and fat (grams).

Outputs: `metrics.csv`, `best_model.pt`, stage checkpoints, frame/dish prediction CSVs, `final_metrics.json`, and `run_config.json`.

The code has been prepared from the stated manifest schema; update paths and verify the actual split names/label units before training. Dummy-mode results must not be used as experimental results.
