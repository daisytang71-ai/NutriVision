"""NutriVision unified evaluation - stage 3: publication-ready figures.

All figures are English-labelled (the group's slide deck is an English
template), white background, 150 dpi, ready to drop into PowerPoint.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("NV_REPO_ROOT", Path(__file__).resolve().parents[2]))
RES = Path(os.environ.get("NV_RES_DIR", ROOT / "results" / "unified"))
FIG = Path(os.environ.get("NV_FIG_DIR", RES / "figures"))
FIG.mkdir(parents=True, exist_ok=True)
IMG_ROOT = Path(os.environ.get("NV_IMG_ROOT", ROOT / "side_frames_valid"))

BLUE = "#2F6FED"
BLUE_DARK = "#1D4ED8"
LIGHT = "#A9C1F5"
GRAY = "#94A3B8"
ORANGE = "#F59E0B"
GREEN = "#10B981"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .7,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ORDER = ["Mean predictor", "Custom CNN", "ResNet18 baseline (frozen)",
         "ResNet18 fine-tuned", "MobileNetV2 baseline (frozen)",
         "MobileNetV2 fine-tuned"]
SHORT = ["Mean\npredictor", "Custom\nCNN", "ResNet18\nbaseline",
         "ResNet18\nfine-tune", "MobileNetV2\nbaseline", "MobileNetV2\nfine-tune"]


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


def bar(ax, values, labels, colors, title, ylabel, fmt="{:.1f}", fontsize=8):
    x = np.arange(len(values))
    b = ax.bar(x, values, color=colors, width=.62, edgecolor="white", linewidth=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_axisbelow(True)
    for rect, v in zip(b, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(values) * .015,
                fmt.format(v), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)


def main():
    t1 = pd.read_csv(RES / "table1_six_model_calorie.csv").set_index("Model").loc[ORDER].reset_index()
    t2 = pd.read_csv(RES / "table2_four_targets_MAE.csv").set_index("Model").loc[ORDER].reset_index()
    t3 = pd.read_csv(RES / "table3_single_vs_dish.csv").set_index("Model").loc[ORDER].reset_index()
    rng = pd.read_csv(RES / "error_by_calorie_range.csv")
    rng["MAE"] = rng["MAE"].fillna(0.0)
    dish = pd.read_csv(RES / "dish_predictions_error_analysis.csv", index_col=0)
    bw = pd.read_csv(RES / "best_worst_cases.csv")
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    best_idx = int(t1.Calorie_MAE_kcal.idxmin())

    colors = [GRAY] + [BLUE] * 5
    colors[best_idx] = ORANGE

    # ---- Fig 1: primary metric, six-model calorie MAE ----
    fig, ax = plt.subplots(figsize=(9, 4.3))
    bar(ax, t1.Calorie_MAE_kcal.to_numpy(), SHORT, colors,
        "Six-model comparison - Calorie MAE (PRIMARY METRIC)", "MAE (kcal)")
    ax.text(.99, .95, "Single-frame, dish-balanced evaluation\n676 test dishes / 20,686 frames",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#475569")
    save(fig, "fig1_six_model_calorie_MAE.png")

    # ---- Fig 2: RMSE and R2 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    bar(axes[0], t1.Calorie_RMSE_kcal.to_numpy(), SHORT, colors,
        "Calorie RMSE (large-error sensitive)", "RMSE (kcal)")
    r2 = t1.Calorie_R2.to_numpy()
    fig2colors = [GRAY] + [BLUE] * 5
    fig2colors[int(np.argmax(r2))] = ORANGE
    b = axes[1].bar(np.arange(6), r2, color=fig2colors, width=.62, edgecolor="white")
    axes[1].axhline(0, color="#334155", lw=1)
    axes[1].set_xticks(np.arange(6)); axes[1].set_xticklabels(SHORT, fontsize=8.5)
    axes[1].set_title("Calorie R² (variance explained)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("R²")
    for rect, v in zip(b, r2):
        axes[1].text(rect.get_x() + rect.get_width() / 2, v + .015, f"{v:.3f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    axes[1].set_ylim(min(0, r2.min() * 1.15), max(r2) * 1.2 + .05)
    save(fig, "fig2_six_model_RMSE_R2.png")

    # ---- Fig 3: four nutritional targets MAE (two panels, different units) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    bar(axes[0], t2.Calories_MAE_kcal.to_numpy(), SHORT, colors,
        "Calories (PRIMARY)", "MAE (kcal)")
    macro = t2[["Protein_MAE_g", "Carbs_MAE_g", "Fat_MAE_g"]].to_numpy()
    x = np.arange(6); w = .26
    for i, (name, c) in enumerate(zip(["Protein", "Carbs", "Fat"], [BLUE, ORANGE, GREEN])):
        axes[1].bar(x + (i - 1) * w, macro[:, i], width=w, label=name, color=c, edgecolor="white")
        for j, v in enumerate(macro[:, i]):
            axes[1].text(x[j] + (i - 1) * w, v + .3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(SHORT, fontsize=8.5)
    axes[1].set_title("Protein / Carbs / Fat (SECONDARY)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("MAE (g)"); axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(0, macro.max() * 1.22)
    save(fig, "fig3_four_targets_MAE.png")

    # ---- Fig 4: Layer 2, single-frame vs dish-level ----
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    a = t3.Single_frame_dish_balanced_MAE.to_numpy()
    b = t3.Dish_level_averaging_MAE.to_numpy()
    x = np.arange(6); w = .36
    ax.bar(x - w/2, a, width=w, label="Single-frame, dish-balanced (PRIMARY)", color=BLUE, edgecolor="white")
    ax.bar(x + w/2, b, width=w, label="Dish-level averaging (SECONDARY)", color=LIGHT, edgecolor="white")
    for i in range(6):
        ax.text(x[i] - w/2, a[i] + 1.5, f"{a[i]:.1f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(x[i] + w/2, b[i] + 1.5, f"{b[i]:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8.5)
    ax.set_ylabel("Calorie MAE (kcal)")
    ax.set_title("Layer 2: two evaluation protocols reported separately",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(a.max(), b.max()) * 1.2)
    save(fig, "fig4_single_frame_vs_dish_level.png")

    # ---- Fig 5: Predicted vs Actual ----
    # Two dishes have true calories >9,000 kcal; plotting them linearly would
    # crush all 674 other dishes into the corner. Plot the regular range and
    # annotate the outliers.
    LIM = 1200
    inr = dish[dish.true_calories < LIM]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    x = inr.true_calories.to_numpy(); y = inr.pred_calories.to_numpy()
    lim = LIM
    ax.scatter(x, y, s=16, alpha=.5, color=BLUE, edgecolors="none")
    ax.plot([0, lim], [0, lim], "--", color="#334155", lw=1.4, label="Ideal (y = x)")
    p = np.polyfit(x, y, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, np.polyval(p, xs), color=ORANGE, lw=2,
            label=f"Fit: y = {p[0]:.2f}x + {p[1]:.0f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual calories (kcal)"); ax.set_ylabel("Predicted calories (kcal)")
    r_all = summary["predicted_vs_actual"]["pearson_r"]
    ax.set_title(f"Predicted vs Actual  |  {summary['error_analysis_model']}\n"
                 "676 test dishes (dish-level averaged predictions)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(.98, .03, f"Pearson r = {r_all:.3f} (all 676 dishes)\n"
                      f"2 dishes >9,000 kcal lie outside axes\n(9,486 / 9,170 kcal, predicted ≤ 800)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#475569")
    save(fig, "fig5_predicted_vs_actual.png")

    # ---- Fig 6: error by calorie range ----
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    v = rng.MAE.to_numpy()
    b = ax.bar(rng.range.astype(str), v, color=[LIGHT, BLUE, BLUE_DARK, ORANGE],
               width=.6, edgecolor="white")
    for rect, val, n in zip(b, v, rng.n_dishes):
        ax.text(rect.get_x() + rect.get_width()/2, val + max(v)*.02, f"{val:.0f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.text(rect.get_x() + rect.get_width()/2, 10, f"n = {int(n)}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.annotate("dominated by 2 dishes with\ntrue 9,486 / 9,170 kcal\n(predicted ≤ 800 kcal)",
                xy=(3, v[3]), xytext=(2.35, v[3] * .72), fontsize=8.5, color="#7C2D12",
                arrowprops=dict(arrowstyle="->", color="#7C2D12", lw=1.2))
    ax.set_ylabel("Calorie MAE (kcal)")
    ax.set_xlabel("Actual calorie range (kcal)")
    ax.set_title("Error by Calorie Range - are high-calorie dishes harder?",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(v) * 1.22)
    save(fig, "fig6_error_by_calorie_range.png")

    # ---- Fig 6b: extreme-label sensitivity (all vs regular subset) ----
    rb = pd.read_csv(RES / "robust_subset_metrics.csv")
    models = rb.model.unique()
    allv = rb[rb.subset == "all_676_dishes"].set_index("model").loc[models]
    regv = rb[rb.subset != "all_676_dishes"].set_index("model").loc[models]
    order = ["MeanPredictor", "CustomCNN", "ResNet18_baseline", "ResNet18_finetune",
             "MobileNetV2_baseline", "MobileNetV2_finetune"]
    allv = allv.loc[order]; regv = regv.loc[order]
    labs = [s.replace("\n", " ") for s in SHORT]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(6); w = .38
    ax.bar(x - w/2, allv.MAE, width=w, label="All 676 dishes (incl. 2 outliers >9,000 kcal)",
           color=GRAY, edgecolor="white")
    ax.bar(x + w/2, regv.MAE, width=w, label="Regular subset: true < 900 kcal (665 dishes)",
           color=BLUE, edgecolor="white")
    for i in range(6):
        ax.text(x[i] - w/2, allv.MAE.iloc[i] + 4, f"{allv.MAE.iloc[i]:.0f}", ha="center", fontsize=8)
        ax.text(x[i] + w/2, regv.MAE.iloc[i] + 4, f"{regv.MAE.iloc[i]:.0f}", ha="center",
                fontsize=8, fontweight="bold", color=BLUE_DARK)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("Calorie MAE (kcal)")
    ax.set_title("Extreme-label sensitivity: dish-level Calorie MAE, all vs regular subset",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, allv.MAE.max() * 1.18)
    save(fig, "fig6b_extreme_label_sensitivity.png")

    # ---- Fig 7: best / worst prediction contact sheet ----
    n = len(bw)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, rows * 3.05))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, bw.iterrows()):
        ax.axis("off")
        p = IMG_ROOT / r.frame_path
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                ax.imshow(im)
        except Exception:
            ax.text(.5, .5, "image unavailable", ha="center", transform=ax.transAxes)
        good = r.case == "best"
        col = GREEN if good else "#DC2626"
        ax.set_title(f"{'BEST' if good else 'WORST'}  |  {r.dish_id}\n"
                     f"True {r.true_calories:.0f} kcal   Pred {r.pred_calories:.0f} kcal\n"
                     f"|Error| {r.abs_error:.1f} kcal",
                     fontsize=8.2, color=col, fontweight="bold", pad=4)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Best / Worst Predictions - {summary['error_analysis_model']} "
                 "(8 smallest and 8 largest absolute errors)",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(FIG / "fig7_best_worst_cases.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", FIG / "fig7_best_worst_cases.png")

    print("\nAll figures saved to", FIG)


if __name__ == "__main__":
    main()
