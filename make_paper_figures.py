#!/usr/bin/env python3
"""
Two publication-quality figures (300 dpi, colorblind-safe Okabe-Ito palette,
minimal chartjunk), saved as PNG + SVG + PDF, each with its backing data CSV.

  fig_capacity_scaling : x = model capacity (LR-k4, LR-k6, RF-k4, RF-k6),
                         y = accuracy drop (homology - original), one line per
                         dataset; leaky datasets highlighted, clean greyed; the
                         random-re-split control shown as a dashed line.
  fig_controls         : grouped bars per leaky dataset, original vs random vs
                         homology-aware accuracy (best original model).

Reads only already-computed, validated CSVs (no model fitting):
  results/capacity_scaling_final.csv, results/summary_final.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
CONFIGS = [(4, "LR"), (6, "LR"), (4, "RF"), (6, "RF")]
XLAB = ["LR k4", "LR k6", "RF k4", "RF k6"]

# Okabe-Ito colorblind-safe palette
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GREYS = ["#9a9a9a", "#b0b0b0", "#7f7f7f", "#c4c4c4", "#8a8a8a"]

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300,
})

NICE = {
    "human_nontata_promoters": "nonTATA promoters",
    "human_enhancers_ensembl": "enhancers (Ensembl)",
    "demo_coding_vs_intergenomic_seqs": "coding vs intergenomic",
    "human_enhancers_cohn": "enhancers (Cohn)",
    "human_ocr_ensembl": "OCR (Ensembl)",
    "demo_human_or_worm": "human vs worm",
    "drosophila_enhancers_stark": "Drosophila enhancers",
}

cap = pd.read_csv(os.path.join(R, "capacity_scaling_final.csv")).set_index("dataset")
summ = pd.read_csv(os.path.join(R, "summary_final.csv")).set_index("dataset")

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
ORDER = ["human_nontata_promoters", "human_enhancers_ensembl",
         "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
         "drosophila_enhancers_stark", "demo_human_or_worm", "human_enhancers_cohn"]


def savefig(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------- fig_capacity_scaling ----------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
xs = np.arange(4)
rows_csv = []
grey_i = 0
for d in ORDER:
    ys = [cap.loc[d, f"hom_delta_{m}_k{k}"] * 100 for (k, m) in CONFIGS]
    leak07 = summ.loc[d, "leak_full_0p7"]
    rows_csv.append({"dataset": d, **{XLAB[i]: round(ys[i], 3) for i in range(4)},
                     "leak_full_0.7": leak07, "group": "leaky" if d in LEAKY else "clean"})
    if d in LEAKY:
        col = BLUE if d == "human_nontata_promoters" else VERM
        ax.plot(xs, ys, marker="o", lw=2.8, ms=8, color=col, zorder=5,
                label=f"{NICE[d]} (leak {leak07:.2f})")
    else:
        ax.plot(xs, ys, marker="o", lw=1.2, ms=4, color=GREYS[grey_i % len(GREYS)],
                alpha=0.7, zorder=2,
                label=f"{NICE[d]} (leak {leak07:.2f})")
        grey_i += 1
# random-control mean (across all datasets)
rand_mean = [np.mean([cap.loc[d, f"rand_delta_{m}_k{k}"] for d in ORDER]) * 100 for (k, m) in CONFIGS]
ax.plot(xs, rand_mean, ls="--", lw=2.0, color="black", marker="s", ms=5, zorder=4,
        label="random re-split (mean, control)")
rows_csv.append({"dataset": "RANDOM_CONTROL_MEAN",
                 **{XLAB[i]: round(rand_mean[i], 3) for i in range(4)},
                 "leak_full_0.7": np.nan, "group": "control"})
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(XLAB)
ax.set_xlabel("model capacity  (low → high)")
ax.set_ylabel("accuracy drop under homology re-split (points)")
ax.set_title("Homology-leakage accuracy drop scales with model capacity")
ax.legend(fontsize=8.2, frameon=False, loc="upper left")
savefig(fig, "fig_capacity_scaling")
pd.DataFrame(rows_csv).to_csv(os.path.join(FIG, "fig_capacity_scaling_data.csv"), index=False)


# ---------------- fig_controls ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.8))
bar_rows = []
xs = np.arange(len(LEAKY))
w = 0.26
specs = [("original_acc", "original split", BLUE, None),
         ("random_acc", "random re-split (control)", "#8a8a8a", None),
         ("homology_acc", "near-duplicate-aware split", VERM, "homology_acc_std")]
for i, (col, lab, color, errcol) in enumerate(specs):
    vals = [summ.loc[d, col] * 100 for d in LEAKY]
    errs = [summ.loc[d, errcol] * 100 if errcol else 0 for d in LEAKY]
    ax.bar(xs + (i - 1) * w, vals, w, label=lab, color=color,
           yerr=errs, capsize=3, ecolor="#333")
    for d, v in zip(LEAKY, vals):
        bar_rows.append({"dataset": d, "split": lab, "accuracy_pct": round(v, 2),
                         "best_model": summ.loc[d, "best_model"]})
ax.set_xticks(xs)
ax.set_xticklabels([f"{NICE[d]}\n({summ.loc[d,'best_model'].replace('_k',' ($k$=')+')'}, leak@0.7={summ.loc[d,'leak_full_0p7']:.2f})" for d in LEAKY])
ax.set_ylabel("test accuracy (%)")
ax.set_ylim(0, 100)
# y-floor at 0 (honest scale; the earlier truncation at 50 visually exaggerated the drop).
ax.set_title("Negative control: only the near-duplicate-aware split removes the inflation",
             pad=14)
ax.legend(fontsize=9, frameon=False, loc="upper right")
savefig(fig, "fig_controls")
pd.DataFrame(bar_rows).to_csv(os.path.join(FIG, "fig_controls_data.csv"), index=False)

print("wrote fig_capacity_scaling.{png,svg,pdf} + fig_controls.{png,svg,pdf} and backing CSVs to", FIG)
