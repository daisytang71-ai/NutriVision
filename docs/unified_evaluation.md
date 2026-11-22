# Unified Six-Model Evaluation（统一六模型评估）

负责人：evaluation 部分（2026-11-22）。本目录中的脚本在**完全相同**的 676 个 test dishes /
20,686 个 test frames 上评估全部六个模型，并输出统一格式的预测 CSV，供指标复算
（如汤远航的误差分析）直接使用。方案与老师建议完全对齐：

1. **第一层（必须）**：六模型 Calorie MAE / RMSE / R² + 四种营养目标 MAE
2. **第二层（必须）**：Primary = Single-frame dish-balanced；Secondary = Dish-level averaging；两者分开报告
3. **第三层（推荐）**：Predicted vs. Actual、Error by Calorie Range、Best/Worst Cases

Primary Metric 为 Calorie MAE（single-frame, dish-balanced 口径）；Protein / Carbs / Fat MAE
为 Secondary Metrics。所有 checkpoint 均基于 Validation Set 选择（训练时 early stopping 监控
验证集 dish-level Calorie MAE），未使用测试集信息做模型选择。

## 六个模型

| # | 模型 | Checkpoint（GitHub Releases） | 验证集 dish Calorie MAE |
|---|---|---|---|
| 1 | Mean predictor（非学习基线，训练集 dish 均值） | — | — |
| 2 | Custom CNN | `cnn_best.pt` | （val 训练记录见 checkpoint） |
| 3 | ResNet18 baseline（frozen backbone） | `Resnet_baseline.pt` | 91.38 kcal |
| 4 | ResNet18 fine-tuned | `Resnet_finetune.pt` | 56.54 kcal |
| 5 | MobileNetV2 baseline（frozen backbone） | `MobileNetV2_baseline.pt` | 88.58 kcal |
| 6 | MobileNetV2 fine-tuned | `MobileNetV2_finetune.pt` | 59.33 kcal |

Mean predictor 来自仓库 README 的模型列表，用于补齐"六模型"比较并给出学习模型必须超越的下限。

## 预测文件（predictions/unified/）

每模型一份 UTF-8 CSV，列与老师要求的统一格式一致：

```text
dish_id,frame_path,true_calories,pred_calories,true_protein,pred_protein,true_carbs,pred_carbs,true_fat,pred_fat
```

每行一帧；frame_path 是相对 `side_frames_valid` 的路径（仅用于身份核对）。预测已恢复原单位
（kcal / g），负预测未截断。`run_metadata.json` 记录每个 checkpoint 的 stage / epoch /
验证集 MAE 与推理耗时。指标计算入口为 `src/evaluation/compute_unified_metrics.py`；
如需旧版 `evaluate.py` 接口（`y_cal/pred_cal` 列名），把列重命名即可（true_cal→y_cal，
pred_cal→pred_cal 等，target 列名 carbs→c 对应 carbohydrate）。

## 结果与图表（results/unified/）

- `metrics_all_levels.csv`：model × target × evaluation level 全量指标
- `table1_six_model_calorie.csv`：六模型 Calorie MAE / RMSE / R²（PPT 用）
- `table2_four_targets_MAE.csv`：四目标 MAE（PPT 用）
- `table3_single_vs_dish.csv`：Single-frame vs Dish-level 对比（PPT 用）
- `error_by_calorie_range.csv`、`robust_subset_metrics.csv`、`extreme_dishes_over_900kcal.csv`、
  `best_worst_cases.csv`、`dish_predictions_error_analysis.csv`：误差分析
- `summary.json`：口径定义、公式与关键结论
- `figures/`：8 张可直接插入 PPT 的图（160 dpi，白底）
- `REPORT.md`：完整评估报告（中文叙述 + 英文表格）

**重要发现**：测试集中 2 个 dish 的真实热量为 9,486 / 9,170 kcal（训练集标签最大约 1,300 kcal），
它们的平方误差主导了所有模型的 RMSE 与 R²（包括均值基线）。解读 RMSE / R² 时请结合
`robust_subset_metrics.csv`（true < 900 kcal 的 665 dish 子集，R² 0.63–0.85）。

## 复现步骤

需要 Python 3.11+、PyTorch、torchvision、pandas、numpy、scikit-learn（不需要 GPU；CPU 约 35 分钟）。

图片数据与 checkpoint 不入 Git：图片为 Nutrition5k 侧视图帧（本地目录 `side_frames_valid`，
由负责人同步给需要的成员），checkpoint 从 [GitHub Releases](https://github.com/daisytang71-ai/NutriVision/releases) 下载。
所有路径通过命令行参数或 `NV_*` 环境变量提供，脚本内不含本机绝对路径。

```powershell
# 1) 推理：6 个模型 -> predictions/unified/*.csv（把 PATH/TO 换成本机实际路径）
python src/evaluation/run_unified_evaluation.py \
  --manifest data/manifests/full_frame_manifest.csv \
  --data-root PATH/TO/side_frames_valid \
  --models-dir PATH/TO/checkpoints

# 2) 三层指标 -> results/unified/*.csv + summary.json
python src/evaluation/compute_unified_metrics.py

# 3) 图表 -> results/unified/figures/*.png
python src/evaluation/make_unified_figures.py

# 4) 报告 -> results/unified/REPORT.md
python src/evaluation/generate_unified_report.py
```

可用环境变量（与命令行参数等价）：`NV_REPO_ROOT`、`NV_MANIFEST`、`NV_DATA_ROOT`、
`NV_MODELS_DIR`、`NV_OUT_DIR`、`NV_PRED_DIR`、`NV_RES_DIR`、`NV_FIG_DIR`、`NV_IMG_ROOT`。

### 口径公式

```text
Single-frame, dish-balanced（D = 676 dishes，n_i = dish i 的帧数）：
  MAE  = (1/D) Σ_i (1/n_i) Σ_j |e_ij|
  MSE  = (1/D) Σ_i (1/n_i) Σ_j e_ij²        RMSE = √MSE
  R²   = 1 − MSE / Var_db(y)（Var_db 使用同一 dish-balanced 权重）
  平方误差先在每个 dish 内汇总，再对 676 道菜总体计算；不把各 dish 的 RMSE 简单平均。

Dish-level averaging：
  先对每道菜所有帧预测取平均 p̄_i，再对 676 个 (y_i, p̄_i) 计算 MAE / RMSE / R²。

Frame-level（全部帧等权）仅作为对照列出，因为它会给帧数多的菜更高权重。
```
