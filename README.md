# NutriVision

NutriVision predicts **calories, protein, carbohydrate, and fat from RGB food images**. We compare a custom CNN trained from scratch with transfer-learning models using Nutrition5k data.

## Group

**Group 7:** 蔡雨静, 曹可昕, 滕姝慧, 汤远航.

## Data and Preparation

We use Nutrition5k side-view RGB frames with dish-level nutritional labels. Because overhead images were unavailable for some dishes, we extracted eight frames from each available video across four camera views, yielding up to 32 frames per dish.

The prepared dataset contains **140,662 frames from 4,605 dish IDs**.

| Split | Frames | Dish IDs |
|---|---:|---:|
| Training | 102,087 | 3,340 |
| Validation | 17,889 | 589 |
| Test | 20,686 | 676 |
| **Total** | **140,662** | **4,605** |

We retain the official train/test partition and hold out approximately 15% of the training portion for validation. All frames from the same dish ID remain in one split.

Data preparation includes:

- Matching image paths, dish IDs, nutritional labels, and split assignments in a manifest.
- Checking file availability, duplicate paths, label consistency, and dish overlap across splits.
- Converting images to RGB, resizing them to 224 × 224 without cropping, and applying ImageNet channel normalization for the CNN.
- Applying light augmentation only during training: horizontal flipping, rotation, brightness, and saturation changes. Validation and test preprocessing are deterministic.

Raw images are stored separately from this repository. Use the prepared manifest and the corresponding image directory to reproduce the experiments.

## Models

| Model | Initialization | Purpose |
|---|---|---|
| Mean predictor | Mean nutritional labels of unique training dishes | Naive regression baseline |
| Custom CNN | Random initialization; no pretrained weights | From-scratch baseline |
| ResNet18 | ImageNet-pretrained weights, followed by fine-tuning | Transfer-learning comparison |
| MobileNetV2 | ImageNet-pretrained model planned for fine-tuning | Additional comparison |

### Custom CNN

The architecture consists of:

1. RGB input of shape 3 × 224 × 224.
2. Three convolutional blocks with 32, 64, and 128 channels. Each block contains a 3 × 3 convolution, batch normalization, ReLU, and 2 × 2 max pooling.
3. Global average pooling and flattening to 128 features.
4. A fully connected layer from 128 to 64 units, followed by ReLU and dropout of 0.3.
5. A linear layer producing four standardized nutritional predictions.

| Training setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Loss | Mean MSE over four standardized targets |
| Maximum epochs | 50 |
| Early stopping | Patience 7; monitor validation dish-level calorie MAE |
| Precision | FP32 |
| Seed | 42 |

Label-standardization statistics are computed from unique training dishes only. Every training frame is used once per epoch. At evaluation time, predictions are converted back to kcal and grams; frame predictions are averaged within each dish for dish-level metrics.

### ResNet18 and MobileNetV2

ResNet18 uses ImageNet-pretrained weights and a four-output regression head. Its fine-tuning procedure first trains the regression head with the backbone frozen, then unfreezes later backbone layers.

The next experiments include completing the ResNet18 and MobileNetV2 fine-tuning and their corresponding baseline comparisons. Model-specific configurations should be reported with their results.

## Evaluation and Results

We report **MAE, RMSE, and R²** for all four targets in their original units: calories in kcal, and protein, carbohydrate, and fat in grams. Frame-level and dish-level results are reported separately. Checkpoints are selected using the validation set.

The completed full-image CNN run stopped at **epoch 16**, with the best checkpoint selected at **epoch 9**.

| Model | Split | Evaluation level | Calorie MAE |
|---|---|---|---:|
| Custom CNN | Validation | Dish-level mean across frames | **81.92 kcal** |

CNN test-set evaluation has also been completed. Its numerical metrics, and the results of the other models, will be added to the results summary after consolidation. The value above is the validation result.

## Trained Model Downloads

Download available checkpoints from [GitHub Releases](https://github.com/daisytang71-ai/NutriVision/releases).

| Checkpoint | Corresponding model |
|---|---|
| `cnn_best.pt` | Custom CNN trained from scratch |
| `resnet18_best.pt` | Fine-tuned ImageNet-pretrained ResNet18 |

Use each checkpoint with its corresponding architecture and preprocessing pipeline. CNN and ResNet18 use different checkpoint formats. Both include their training configuration and target-standardization parameters.

## Running the Custom CNN

The CNN source code is under [`nutrivision_cnn/nutrivision`](nutrivision_cnn/nutrivision). Run the following commands from the `nutrivision_cnn` directory.

### Environment

The supplied pinned CNN requirements were tested with Python 3.12. For an existing GPU environment, retain a CUDA-compatible PyTorch installation and check the dependencies before changing it.

```bash
cd nutrivision_cnn
python3 -m pip install -r requirements.txt
```

### Evaluate a trained checkpoint

Replace `/path/to/...` with your local paths. Use the exact `manifest.csv` saved by the corresponding training run: evaluation checks its SHA-256 against the checkpoint. For the CNN, target columns are `calories`, `protein`, `carbs`, and `fat`; the manifest must also contain `image_path`, `dish_id`, and `split`.

```bash
python3 -m nutrivision.evaluate \
  --checkpoint /path/to/cnn_best.pt \
  --manifest /path/to/manifest.csv \
  --data-root /path/to/side_frames_valid \
  --split test \
  --out-dir evaluation/cnn_test \
  --device auto
```

The output directory must be empty. Evaluation exports `metrics.json`, `image_predictions.csv`, `dish_predictions.csv`, and `predicted_vs_true.png`.

### Predict one image

```bash
python3 -m nutrivision.predict \
  --checkpoint /path/to/cnn_best.pt \
  --image /path/to/food_image.jpg \
  --device auto
```

### Train from scratch

For an available NVIDIA GPU, the configuration below reproduces the main CNN training settings:

```bash
python3 -u -m nutrivision.train \
  --manifest /path/to/manifest.csv \
  --data-root /path/to/side_frames_valid \
  --data-kind real \
  --out-dir runs/cnn_all_images \
  --device cuda \
  --epochs 50 \
  --batch-size 32 \
  --image-size 224 \
  --num-workers 4 \
  --sampling all-images \
  --patience 7 \
  --seed 42
```

Choose a fresh output directory for each run. The command uses FP32 and exports the best checkpoint, training history, loss curves, and validation predictions. Test evaluation is run separately.

## Next Steps

- Complete fine-tuning and baseline experiments for ResNet18 and MobileNetV2.
- Consolidate the four-target test metrics and compare models on the same test split.
- Report frame-level and dish-level results separately.
- Generate comparison plots and analyze dishes with large prediction errors.

