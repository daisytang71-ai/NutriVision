# Training protocol

| Item | Setting |
|---|---|
| Image preprocessing | RGB; direct Resize to 224 × 224, no crop |
| Training augmentation | Random horizontal flip (p=0.5), random rotation (±10°), brightness jitter (0.15) |
| Validation/test augmentation | No random augmentation; resize and ImageNet normalization only |
| Targets | calories, protein, carbohydrate, fat |
| Target scaling | Training-split unique-dish mean and standard deviation |
| Loss | `MSELoss(reduction="mean")` over all batch items and four standardized targets |
| Stage 1 | Freeze backbone; train regression head, LR 1e-3 |
| Stage 2 | ResNet18: unfreeze layer4; MobileNetV2: unfreeze final 3 feature blocks; backbone LR 1e-5, head LR 1e-4 |
| Optimizer | AdamW; weight decay 1e-4 |
| Early stopping/model selection | Validation dish-level calorie MAE, patience 7 |
| Multi-frame inference | Predict each frame, then average predictions by dish_id |
| Metrics | MAE, RMSE, R² for all targets, in original units |
| Split protection | Same dish must not occur in multiple splits |
