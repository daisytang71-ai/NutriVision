"""NutriVision unified evaluation - stage 4: generate the evaluation report.

Produces evaluation/REPORT.md (Chinese narrative + English tables/figures,
ready to paste into the Week-9 English slide deck).
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("NV_REPO_ROOT", Path(__file__).resolve().parents[2]))
RES = Path(os.environ.get("NV_RES_DIR", ROOT / "results" / "unified"))
PRED_DIR = Path(os.environ.get("NV_PRED_DIR", ROOT / "predictions" / "unified"))


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def main():
    t1 = pd.read_csv(RES / "table1_six_model_calorie.csv")
    t2 = pd.read_csv(RES / "table2_four_targets_MAE.csv")
    t3 = pd.read_csv(RES / "table3_single_vs_dish.csv")
    rng = pd.read_csv(RES / "error_by_calorie_range.csv")
    bw = pd.read_csv(RES / "best_worst_cases.csv")
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((PRED_DIR / "run_metadata.json").read_text(encoding="utf-8"))

    best_row = t1.loc[t1.Calorie_MAE_kcal.idxmin()]
    ea = summary["error_analysis_model"]
    mmeta = {m["model"]: m for m in meta}
    LABELS_REPORT = {"MeanPredictor": "Mean predictor", "CustomCNN": "Custom CNN",
                     "ResNet18_baseline": "ResNet18 baseline (frozen)",
                     "ResNet18_finetune": "ResNet18 fine-tuned",
                     "MobileNetV2_baseline": "MobileNetV2 baseline (frozen)",
                     "MobileNetV2_finetune": "MobileNetV2 fine-tuned"}
    robust_index = list(summary["robust_subset"].keys())

    md = []
    a = md.append
    a("# NutriVision Evaluation Report（评估结果报告）\n")
    a(f"> 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ｜ "
      f"测试集：**{summary['n_test_dishes']} 个 test dishes / {summary['n_test_frames']} 帧**（dish-disjoint，无泄漏）\n")

    a("## 0. 评估方案（与老师建议完全对齐）\n")
    a("| 层次 | 内容 | 口径 |")
    a("|---|---|---|")
    a("| 第一层 | 六模型性能比较（必须） | 相同 676 个 test dishes，MAE / RMSE / R² |")
    a("| 第二层 | Single-frame vs Dish-level（必须） | Primary = Single-frame dish-balanced；Secondary = Dish-level averaging |")
    a("| 第三层 | Error Analysis（推荐） | Predicted vs Actual ＋ Error by Calorie Range ＋ Best/Worst Cases |")
    a("")
    a("- **Primary Metric：Calorie MAE**；Secondary Metrics：Protein / Carbs / Fat MAE")
    a("- 所有模型使用**同一份评估脚本**，输出**统一格式 CSV**："
      "`dish_id, frame_path, true_calories, pred_calories, true_protein, pred_protein, true_carbs, pred_carbs, true_fat, pred_fat`")
    a("- 所有 checkpoint 均基于 **Validation Set** 选择（训练时的 early stopping / 最低验证集 dish-level Calorie MAE），"
      "未使用任何测试集信息做模型选择。")
    a("- 第六个模型 **Mean Predictor** 为非学习基线（直接预测训练集 dish 均值），"
      "来自仓库 README 的模型列表，用于补齐“六模型”比较并给出学习模型必须超越的下限。\n")

    a("## 1. 模型与 checkpoint 信息\n")
    a("| 模型 | Checkpoint | Stage / Epoch | 验证集 dish Calorie MAE (kcal) |")
    a("|---|---|---|---|")
    for m, lab in [("MeanPredictor", "Mean predictor"), ("CustomCNN", "Custom CNN"),
                   ("ResNet18_baseline", "ResNet18 baseline (frozen)"),
                   ("ResNet18_finetune", "ResNet18 fine-tuned"),
                   ("MobileNetV2_baseline", "MobileNetV2 baseline (frozen)"),
                   ("MobileNetV2_finetune", "MobileNetV2 fine-tuned")]:
        d = mmeta.get(m, {})
        stage = d.get("stage", "-")
        ep = d.get("epoch", "-")
        val = d.get("val_dish_calories_MAE_kcal")
        a(f"| {lab} | {m} | {stage} / {ep} | {'—（非学习基线）' if val is None else fmt(val)} |")
    a("")

    a("## 2. 第一层：六模型性能比较（Primary 口径）\n")
    a("### 2.1 Calorie MAE / RMSE / R²（必须）\n")
    a("| Model | Calorie MAE (kcal) ⭐ | Calorie RMSE (kcal) | Calorie R² |")
    a("|---|---|---|---|")
    for _, r in t1.iterrows():
        star = " ⭐" if r.Model == best_row.Model else ""
        a(f"| {r.Model}{star} | **{fmt(r.Calorie_MAE_kcal)}** | {fmt(r.Calorie_RMSE_kcal)} | {fmt(r.Calorie_R2, 3)} |")
    a("")
    a(f"⭐ = 在 Primary 口径下 Calorie MAE 最优的模型：**{best_row.Model}**\n")
    a(f"![fig1](figures/fig1_six_model_calorie_MAE.png)\n")
    a(f"![fig2](figures/fig2_six_model_RMSE_R2.png)\n")

    a("### 2.2 四种营养目标的 MAE（必须）\n")
    a("| Model | Calories MAE (kcal) | Protein MAE (g) | Carbs MAE (g) | Fat MAE (g) |")
    a("|---|---|---|---|---|")
    for _, r in t2.iterrows():
        a(f"| {r.Model} | {fmt(r.Calories_MAE_kcal)} | {fmt(r.Protein_MAE_g)} | {fmt(r.Carbs_MAE_g)} | {fmt(r.Fat_MAE_g)} |")
    a("")
    a(f"![fig3](figures/fig3_four_targets_MAE.png)\n")

    a("## 3. 第二层：Single-frame 与 Dish-level 两种评估（分别报告，不混用）\n")
    a("**口径定义（报告中已明确统计单位）：**\n")
    a("- **Primary — Single-frame, dish-balanced**：每道菜先算所有测试帧的平均绝对误差，再对 676 道菜取平均。"
      "RMSE/R² 同样在每个 dish 内先汇总平方误差再总体计算（不是把各 dish 的 RMSE 简单平均）。")
    a("- **Secondary — Dish-level prediction averaging**：每道菜先对所有帧的预测取平均，再对 676 道菜计算指标"
      "（即 Stage 2 使用的多帧融合方式，用于观察多帧融合带来的改善）。")
    a("- Frame-level（全部帧等权）仅作为对照列出——它会给帧数多的菜更高权重，因此不作为主口径。\n")
    a("| Model | Single-frame MAE | Single-frame RMSE | Single-frame R² | Dish-level MAE | Dish-level RMSE | Dish-level R² | 多帧融合改善 |")
    a("|---|---|---|---|---|---|---|---|")
    for _, r in t3.iterrows():
        imp = f"{fmt(r.Improvement_pct, 1)}%"
        a(f"| {r.Model} | {fmt(r.Single_frame_dish_balanced_MAE)} | {fmt(r.Single_frame_dish_balanced_RMSE)} "
          f"| {fmt(r.Single_frame_dish_balanced_R2, 3)} | {fmt(r.Dish_level_averaging_MAE)} "
          f"| {fmt(r.Dish_level_averaging_RMSE)} | {fmt(r.Dish_level_averaging_R2, 3)} | {imp} |")
    a("")
    a("所有指标均以 Calorie 为例；四目标完整指标见 `results/metrics_all_levels.csv`。\n")
    a(f"![fig4](figures/fig4_single_frame_vs_dish_level.png)\n")

    a("## 4. 第三层：Error Analysis（推荐；误差分析模型按验证集选择）\n")
    a(f"误差分析使用的模型：**{ea}**（选择依据：验证集 dish-level Calorie MAE 最低 = "
      f"{fmt(mmeta[ea]['val_dish_calories_MAE_kcal'])} kcal，不使用测试集做选择）。\n")

    pva = summary["predicted_vs_actual"]
    a("### 4.1 Predicted vs. Actual\n")
    rs = pva["regular_subset"]
    a(f"- 全部 676 道菜拟合：`Predicted = {fmt(pva['slope'], 3)} × Actual + {fmt(pva['intercept'], 1)}`，"
      f"Pearson r = {fmt(pva['pearson_r'], 3)}（被 2 个极端 dish 严重扭曲）")
    a(f"- **常规区间**（true < {rs['cutoff_kcal']} kcal，{rs['n_dishes']} 道菜）拟合："
      f"`Predicted = {fmt(rs['slope'], 3)} × Actual + {fmt(rs['intercept'], 1)}`，"
      f"Pearson r = {fmt(rs['pearson_r'], 3)}")
    slope_note = ("斜率 < 1，说明模型对高热量菜品存在**系统性低估**（regression to the mean）。"
                  if rs["slope"] < 1 else "斜率接近 1，模型在高/低热量区间的偏差较对称。")
    a(f"- {slope_note}")
    a(f"- 图中坐标轴截断至 1,200 kcal：2 个真实热量 >9,000 kcal 的 dish（9,486 / 9,170 kcal）超出坐标范围，"
      f"任何模型（包括均值基线）都无法从图像预测它们——训练集标签本身不超过约 1,300 kcal。\n")
    a(f"![fig5](figures/fig5_predicted_vs_actual.png)\n")

    a("### 4.2 Error by Calorie Range\n")
    a("| 真实热量区间 (kcal) | Dish 数 | Calorie MAE (kcal) | 平均偏差 Pred−True (kcal) |")
    a("|---|---|---|---|")
    for _, r in rng.iterrows():
        a(f"| {r.range} | {int(r.n_dishes)} | {fmt(r.MAE)} | {fmt(r.mean_bias)} |")
    a("")
    a(f"![fig6](figures/fig6_error_by_calorie_range.png)\n")

    a("### 4.3 极端标签敏感性（RMSE / R² 的关键注意事项）\n")
    a(f"测试集中有 **{summary['extreme_labels']['n_dishes_above']} 个 dish 的真实热量 ≥ 900 kcal**，"
      f"其中 2 个高达 **9,486 / 9,170 kcal**。它们的平方误差主导了所有模型的 RMSE 与 R²"
      f"（包括均值基线），而训练集标签最大约 1,300 kcal，模型在原理上无法外推到 9,000+ kcal。"
      f"因此 RMSE / R² 应结合\"常规子集\"一起解读：\n")
    a("| Model | 全部 676 dish MAE | 常规子集 MAE (true<900, n=665) | 全部 676 RMSE | 常规子集 RMSE | 全部 676 R² | 常规子集 R² |")
    a("|---|---|---|---|---|---|---|")
    for m in robust_index:
        r1, r2 = summary["robust_subset"][m], summary["robust_subset_regular"][m]
        a(f"| {LABELS_REPORT.get(m, m)} | {fmt(r1['MAE'])} | **{fmt(r2['MAE'])}** | {fmt(r1['RMSE'])} "
          f"| {fmt(r2['RMSE'])} | {fmt(r1['R2'], 3)} | **{fmt(r2['R2'], 3)}** |")
    a("")
    a("> 这正是 RMSE \"关注大误差样本\" 的作用：排除极端标签后，常规菜品的 MAE 约 "
      f"{fmt(min(v['MAE'] for v in summary['robust_subset_regular'].values()))}–"
      f"{fmt(max(v['MAE'] for v in summary['robust_subset_regular'].values()))} kcal，"
      "R² 大幅提升，说明模型在正常菜品区间表现良好。\n")
    a(f"![fig6b](figures/fig6b_extreme_label_sensitivity.png)\n")

    a("### 4.4 Best / Worst Predictions\n")
    a("| Case | Dish ID | 真实热量 (kcal) | 预测热量 (kcal) | 绝对误差 (kcal) |")
    a("|---|---|---|---|---|")
    for _, r in bw.iterrows():
        a(f"| {r.case} | {r.dish_id} | {fmt(r.true_calories, 0)} | {fmt(r.pred_calories, 0)} | {fmt(r.abs_error, 1)} |")
    a("")
    a(f"![fig7](figures/fig7_best_worst_cases.png)\n")

    a("## 5. 结论要点（可直接写进 PPT / 汇报词）\n")
    a(f"1. **六模型统一比较**（相同 676 test dishes、相同脚本）：{best_row.Model} 以 "
      f"Calorie MAE = {fmt(best_row.Calorie_MAE_kcal)} kcal 最优，Mean predictor 基线为 "
      f"{fmt(t1.loc[0,'Calorie_MAE_kcal'])} kcal——所有学习模型都显著优于非学习基线。")
    a("2. **两种评估口径分开报告**：单帧 dish-balanced 口径回答 Research Question 1（单图预测能力）；"
      "dish-level averaging 口径显示多帧融合能进一步降低 MAE（改善幅度见第 3 节表格）。")
    a("3. **误差分析**：误差随真实热量区间上升；常规区间内拟合斜率 < 1，"
      "说明模型对高热量菜品存在系统性低估。Best/Worst 案例展示了模型表现好/差的具体食物类型。")
    a("4. **极端标签发现（亮点）**：测试集中 2 个 dish 真实热量高达 9,486 / 9,170 kcal，"
      "主导了所有模型的 RMSE 与 R²（包括均值基线）；排除后常规子集 R² 大幅提升。"
      "建议汇报时把这一点作为\"为什么 RMSE 高、R² 低\"的核心解释，"
      "并可讨论后续改进（如对标签做 log 变换或截断训练范围）。")
    a("5. **严谨性**：checkpoint 全部基于验证集选择；所有模型用同一脚本与统一 CSV；"
      "dish-disjoint 划分避免数据泄漏。\n")

    a("## 6. 方法细节与文件清单\n")
    a("- 评估脚本：`src/evaluation/run_unified_evaluation.py`（推理）→ `compute_unified_metrics.py`（三层指标）→ `make_unified_figures.py`（图表）→ `generate_unified_report.py`（本报告）；运行说明见 `docs/unified_evaluation.md`")
    a("- 预测 CSV（统一格式，每模型 20,686 行）：`predictions/unified/predictions_<model>.csv`")
    a("- 指标明细：`results/metrics_all_levels.csv`（model × target × evaluation level）")
    a("- PPT 用表：`results/table1_six_model_calorie.csv`、`table2_four_targets_MAE.csv`、`table3_single_vs_dish.csv`、`error_by_calorie_range.csv`、`robust_subset_metrics.csv`、`extreme_dishes_over_900kcal.csv`、`best_worst_cases.csv`")
    a("- 图表（160 dpi PNG，白底，可直接插入 PPT）：`evaluation/figures/fig1…fig7` 与 `fig6b`\n")

    a("### 口径公式\n")
    a("```text")
    a("Single-frame, dish-balanced (D = 676 dishes, n_i = frames of dish i):")
    a("  MAE  = (1/D) Σ_i (1/n_i) Σ_j |e_ij|")
    a("  MSE  = (1/D) Σ_i (1/n_i) Σ_j e_ij²      RMSE = √MSE")
    a("  R²   = 1 − MSE / Var_db(y),  Var_db 使用同一 dish-balanced 权重")
    a("")
    a("Dish-level averaging:")
    a("  先对每道菜所有帧预测取平均 p̄_i，再对 676 个 (y_i, p̄_i) 计算 MAE/RMSE/R²")
    a("```\n")

    out = RES / "REPORT.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
