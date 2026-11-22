"""NutriVision unified evaluation - stage 2: compute the three evaluation layers.

Layer 1  Six-model comparison (mandatory): MAE / RMSE / R2 for calories and
         MAE for all four nutritional targets, on the SAME 676 test dishes.
Layer 2  Primary   = Single-frame, dish-balanced evaluation
                     (average |error| within each dish, then average over dishes;
                      squared errors are aggregated within each dish FIRST, so
                      RMSE/R2 are NOT an average of per-dish RMSE values).
         Secondary = Dish-level prediction averaging
                     (average all frame predictions per dish, then score 676 dishes).
         Frame-level (all 20,686 frames equally weighted) is reported as a
         reference/contrast only - it over-weights dishes with more frames.
Layer 3  Error analysis: predicted vs actual, error by calorie range,
         best / worst predictions.

Recommended final scheme (per the project brief):
  six-model comparison + single-frame primary + dish-level secondary +
  three error analyses. Calorie MAE is the PRIMARY metric; protein / carbs /
  fat MAE are SECONDARY metrics.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NV_REPO_ROOT", Path(__file__).resolve().parents[2]))
PRED_DIR = Path(os.environ.get("NV_PRED_DIR", ROOT / "predictions" / "unified"))
RES = Path(os.environ.get("NV_RES_DIR", ROOT / "results" / "unified"))
EXPECTED_FRAMES = int(os.environ.get("NV_EXPECTED_FRAMES", 20686))
EXPECTED_DISHES = int(os.environ.get("NV_EXPECTED_DISHES", 676))
RES.mkdir(parents=True, exist_ok=True)

TARGETS = ["calories", "protein", "carbs", "fat"]
UNITS = {"calories": "kcal", "protein": "g", "carbs": "g", "fat": "g"}
MODEL_ORDER = ["MeanPredictor", "CustomCNN", "ResNet18_baseline", "ResNet18_finetune",
               "MobileNetV2_baseline", "MobileNetV2_finetune"]
LABELS = {"MeanPredictor": "Mean predictor", "CustomCNN": "Custom CNN",
          "ResNet18_baseline": "ResNet18 baseline (frozen)",
          "ResNet18_finetune": "ResNet18 fine-tuned",
          "MobileNetV2_baseline": "MobileNetV2 baseline (frozen)",
          "MobileNetV2_finetune": "MobileNetV2 fine-tuned"}


def load(name):
    f = PRED_DIR / f"predictions_{name}.csv"
    df = pd.read_csv(f)
    assert len(df) == EXPECTED_FRAMES, f"{name}: expected {EXPECTED_FRAMES} frames, got {len(df)}"
    assert df.dish_id.nunique() == EXPECTED_DISHES, f"{name}: expected {EXPECTED_DISHES} dishes"
    assert df.notna().all().all(), f"{name}: contains NaN"
    return df


def dish_balanced(df, t, p):
    """PRIMARY: single-frame, dish-balanced evaluation."""
    g = df.groupby("dish_id", sort=False)
    per = g.apply(lambda x: pd.Series({
        "mae": np.abs(x[t] - x[p]).mean(),
        "mse": ((x[t] - x[p]) ** 2).mean(),
        "y": x[t].iloc[0],
        "n": len(x),
    }), include_groups=False)
    mae = float(per.mae.mean())
    mse = float(per.mse.mean())
    mu = float(per.y.mean())
    var = float(((per.y - mu) ** 2).mean())
    r2 = 1.0 - mse / var if var > 0 else np.nan
    return dict(MAE=mae, RMSE=float(np.sqrt(mse)), R2=float(r2),
                n_frames=int(len(df)), n_dishes=int(len(per)))


def dish_level(df, t, p):
    """SECONDARY: average all frame predictions per dish, then score 676 dishes."""
    g = df.groupby("dish_id", sort=False).agg(y=(t, "first"), p=(p, "mean"), n=(t, "size"))
    e = g.y - g.p
    mse = float((e ** 2).mean())
    var = float(((g.y - g.y.mean()) ** 2).mean())
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt(mse)),
                R2=float(1 - mse / var) if var > 0 else np.nan,
                n_frames=int(len(df)), n_dishes=int(len(g)))


def frame_level(df, t, p):
    """Reference only: all frames equally weighted."""
    e = df[t] - df[p]
    mse = float((e ** 2).mean())
    var = float(((df[t] - df[t].mean()) ** 2).mean())
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt(mse)),
                R2=float(1 - mse / var) if var > 0 else np.nan,
                n_frames=int(len(df)), n_dishes=int(df.dish_id.nunique()))


def main():
    data = {m: load(m) for m in MODEL_ORDER}
    rows = []
    for m in MODEL_ORDER:
        df = data[m]
        for t in TARGETS:
            tc, pc = f"true_{t}", f"pred_{t}"
            for level, fn in [("primary_single_frame_dish_balanced", dish_balanced),
                              ("secondary_dish_level_averaging", dish_level),
                              ("reference_frame_level", frame_level)]:
                r = fn(df, tc, pc)
                rows.append(dict(model=m, model_label=LABELS[m], target=t, unit=UNITS[t],
                                 evaluation_level=level, **r))
    full = pd.DataFrame(rows)
    full.to_csv(RES / "metrics_all_levels.csv", index=False, encoding="utf-8-sig")

    # ----- Layer 1 tables (PRIMARY = dish-balanced single-frame) -----
    prim = full[full.evaluation_level == "primary_single_frame_dish_balanced"]
    cal = prim[prim.target == "calories"].set_index("model").loc[MODEL_ORDER].reset_index()
    table1 = cal[["model_label", "MAE", "RMSE", "R2", "n_frames", "n_dishes"]].copy()
    table1.columns = ["Model", "Calorie_MAE_kcal", "Calorie_RMSE_kcal", "Calorie_R2",
                      "Test_frames", "Test_dishes"]
    table1.to_csv(RES / "table1_six_model_calorie.csv", index=False, encoding="utf-8-sig")

    table2 = prim.pivot(index="model", columns="target", values="MAE").loc[MODEL_ORDER].reset_index()
    table2.insert(0, "Model", table2.pop("model").map(LABELS))
    table2 = table2[["Model", "calories", "protein", "carbs", "fat"]]
    table2.columns = ["Model", "Calories_MAE_kcal", "Protein_MAE_g", "Carbs_MAE_g", "Fat_MAE_g"]
    table2.to_csv(RES / "table2_four_targets_MAE.csv", index=False, encoding="utf-8-sig")

    # full MAE/RMSE/R2 for all targets (primary level)
    wide = prim.pivot_table(index="model", columns="target", values=["MAE", "RMSE", "R2"])
    wide = wide.reindex(MODEL_ORDER)
    wide.to_csv(RES / "table1b_all_targets_all_metrics.csv", encoding="utf-8-sig")

    # ----- Layer 2 table: single-frame vs dish-level -----
    cmp_rows = []
    for m in MODEL_ORDER:
        df = data[m]
        a = dish_balanced(df, "true_calories", "pred_calories")
        b = dish_level(df, "true_calories", "pred_calories")
        c = frame_level(df, "true_calories", "pred_calories")
        cmp_rows.append(dict(Model=LABELS[m],
                             Single_frame_dish_balanced_MAE=a["MAE"],
                             Single_frame_dish_balanced_RMSE=a["RMSE"],
                             Single_frame_dish_balanced_R2=a["R2"],
                             Dish_level_averaging_MAE=b["MAE"],
                             Dish_level_averaging_RMSE=b["RMSE"],
                             Dish_level_averaging_R2=b["R2"],
                             Frame_level_reference_MAE=c["MAE"],
                             MAE_improvement_by_averaging=a["MAE"] - b["MAE"],
                             Improvement_pct=100 * (a["MAE"] - b["MAE"]) / a["MAE"]))
    table3 = pd.DataFrame(cmp_rows)
    table3.to_csv(RES / "table3_single_vs_dish.csv", index=False, encoding="utf-8-sig")

    # ----- Layer 3: error analysis -----
    meta = json.loads((PRED_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    val = {d["model"]: d.get("val_dish_calories_MAE_kcal") for d in meta}
    neural = [m for m in MODEL_ORDER if m != "MeanPredictor" and val.get(m) is not None]
    best_model = min(neural, key=lambda m: val[m])  # selected on VALIDATION, never on test
    print(f"Error-analysis model (best validation dish calorie MAE): {best_model}")

    df = data[best_model]
    g = df.groupby("dish_id", sort=False).agg(
        true_calories=("true_calories", "first"),
        pred_calories=("pred_calories", "mean"),
        n_frames=("true_calories", "size"),
        frame_path=("frame_path", "first"))
    g["abs_error"] = (g.true_calories - g.pred_calories).abs()
    g["signed_error"] = g.pred_calories - g.true_calories
    g.to_csv(RES / "dish_predictions_error_analysis.csv", encoding="utf-8-sig")

    # 3.2 error by calorie range
    bins = [-np.inf, 300, 600, 900, np.inf]
    labels = ["0-300", "300-600", "600-900", "900+"]
    g["range"] = pd.cut(g.true_calories, bins=bins, labels=labels, right=False)
    rng = g.groupby("range", observed=False).agg(
        n_dishes=("abs_error", "size"), MAE=("abs_error", "mean"),
        mean_bias=("signed_error", "mean")).reset_index()
    rng.to_csv(RES / "error_by_calorie_range.csv", index=False, encoding="utf-8-sig")

    # ----- Robust subset: extreme-label sensitivity -----
    # Two test dishes have true calories > 9,000 kcal (9,486 / 9,170). Their
    # squared errors dominate RMSE and R2 for EVERY model (including the mean
    # baseline) and are effectively unpredictable from images because training
    # labels never exceed ~1,300 kcal. Report a "regular subset" contrast.
    CUT = 900
    extreme = g[g.true_calories >= CUT].sort_values("true_calories", ascending=False)
    extreme.reset_index()[["dish_id", "true_calories", "pred_calories", "abs_error",
                           "signed_error", "n_frames"]].to_csv(
        RES / "extreme_dishes_over_900kcal.csv", index=False, encoding="utf-8-sig")
    robust_rows = []
    for m in MODEL_ORDER:
        dfm = data[m]
        gg = dfm.groupby("dish_id", sort=False).agg(
            y=("true_calories", "first"), p=("pred_calories", "mean"))
        reg = gg[gg.y < CUT]
        for label, sub in [("all_676_dishes", gg), (f"regular_subset_true_lt_{CUT}", reg)]:
            e = sub.p - sub.y
            mse = float((e ** 2).mean())
            var = float(((sub.y - sub.y.mean()) ** 2).mean())
            robust_rows.append(dict(model=m, model_label=LABELS[m], subset=label,
                                    n_dishes=int(len(sub)),
                                    MAE=float(np.abs(e).mean()),
                                    RMSE=float(np.sqrt(mse)),
                                    R2=float(1 - mse / var) if var > 0 else np.nan))
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(RES / "robust_subset_metrics.csv", index=False, encoding="utf-8-sig")
    all676 = robust[robust.subset == "all_676_dishes"].set_index("model")
    reg = robust[robust.subset != "all_676_dishes"].set_index("model")

    # systematic under/over estimation: regression of prediction on truth
    # (fitted on the regular subset AND on all dishes, both reported)
    x, y = g.true_calories.to_numpy(), g.pred_calories.to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    x2, y2 = reg_true, reg_pred = (g[g.true_calories < CUT].true_calories.to_numpy(),
                                   g[g.true_calories < CUT].pred_calories.to_numpy())
    slope_reg, intercept_reg = np.polyfit(x2, y2, 1)
    corr_reg = float(np.corrcoef(x2, y2)[0, 1])

    # 3.3 best / worst cases (top 8 each)
    s = g.reset_index().sort_values("abs_error")
    bw = pd.concat([s.head(8).assign(case="best"), s.tail(8).assign(case="worst")])
    bw = bw[["case", "dish_id", "frame_path", "true_calories", "pred_calories",
             "abs_error", "signed_error", "n_frames"]]
    bw.to_csv(RES / "best_worst_cases.csv", index=False, encoding="utf-8-sig")

    summary = dict(
        n_test_dishes=676, n_test_frames=20686, models=MODEL_ORDER,
        primary_metric="Calorie MAE (single-frame, dish-balanced)",
        secondary_metrics=["Protein MAE", "Carbs MAE", "Fat MAE"],
        checkpoint_selection="validation dish-level calorie MAE; no test-set selection",
        error_analysis_model=best_model,
        error_analysis_model_selection_basis="lowest validation dish-level calorie MAE",
        predicted_vs_actual=dict(slope=float(slope), intercept=float(intercept), pearson_r=corr,
                                 regular_subset=dict(slope=float(slope_reg), intercept=float(intercept_reg),
                                                     pearson_r=corr_reg, cutoff_kcal=CUT,
                                                     n_dishes=int(len(x2)))),
        extreme_labels=dict(cutoff_kcal=CUT,
                            n_dishes_above=int(len(extreme)),
                            max_true_calories=float(g.true_calories.max()),
                            note="Two dishes have true calories >9,000 kcal (9,486 / 9,170); "
                                 "they dominate RMSE/R2 for every model and are impossible to "
                                 "predict from images given the training-label range."),
        robust_subset=all676[["MAE", "RMSE", "R2"]].round(4).to_dict("index"),
        robust_subset_regular=reg[["MAE", "RMSE", "R2"]].round(4).to_dict("index"),
        best_primary_calorie_mae_model=str(table1.loc[table1.Calorie_MAE_kcal.idxmin(), "Model"]),
        mean_predictor_note="Non-learning baseline predicting training-dish means; included so the comparison covers six models.",
        csv_schema=["dish_id", "frame_path", "true_calories", "pred_calories",
                    "true_protein", "pred_protein", "true_carbs", "pred_carbs",
                    "true_fat", "pred_fat"],
        definitions=dict(
            single_frame_dish_balanced="MAE=(1/D)sum_i (1/n_i)sum_j |e_ij|; MSE=(1/D)sum_i (1/n_i)sum_j e_ij^2; RMSE=sqrt(MSE); R2=1-MSE/Var(y) with the same dish-balanced weighting.",
            dish_level_averaging="Average all frame predictions per dish, then compute MAE/RMSE/R2 over the 676 dish-level pairs.",
            frame_level_reference="All frames equally weighted; reported only as a contrast because it over-weights dishes with more frames."),
    )
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")

    print("\n=== Table 1: six-model calorie comparison (PRIMARY, dish-balanced single-frame) ===")
    print(table1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Table 2: MAE of four nutritional targets (PRIMARY) ===")
    print(table2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Table 3: single-frame vs dish-level (calories) ===")
    print(table3.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Error by calorie range ===")
    print(rng.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Extreme-label sensitivity (calorie, dish-level averaging) ===")
    print(robust.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nPredicted-vs-Actual (all 676): slope={slope:.4f} intercept={intercept:.2f} r={corr:.4f}")
    print(f"Predicted-vs-Actual (true<{CUT}, n={len(x2)}): slope={slope_reg:.4f} intercept={intercept_reg:.2f} r={corr_reg:.4f}")
    print(f"\nSaved results to {RES}")


if __name__ == "__main__":
    main()
