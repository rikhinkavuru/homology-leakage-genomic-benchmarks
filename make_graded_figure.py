#!/usr/bin/env python3
"""fig_graded_performance: per-model accuracy as a function of the test-sequence
similarity bin (original split, k=6). Panels for the 2 leaky datasets + 2 clean
controls. Hollow markers flag low-confidence bins (n<50). Reads graded_performance.csv
(which is itself the backing data CSV)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(R, "figures")
BINLAB = ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
PANELS = ["human_nontata_promoters", "human_enhancers_ensembl",
          "human_enhancers_cohn", "demo_human_or_worm"]
NICE = {"human_nontata_promoters": "nonTATA promoters", "human_enhancers_ensembl": "enhancers (Ensembl)",
        "human_enhancers_cohn": "enhancers (Cohn)", "demo_human_or_worm": "human vs worm"}
LEAKY = {"human_nontata_promoters", "human_enhancers_ensembl"}
MCOL = {"LR": "#0072B2", "LinearSVC": "#56B4E9", "RF": "#D55E00", "HGB": "#009E73"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

g = pd.read_csv(os.path.join(R, "graded_performance.csv"))
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, d in zip(axes.ravel(), PANELS):
    sub = g[g.dataset == d]
    n_by_bin = [int(sub[sub.model == "RF"].set_index("sim_bin").reindex(BINLAB).loc[b, "n"]) for b in BINLAB]
    for m in ["LR", "LinearSVC", "RF", "HGB"]:
        s = sub[sub.model == m].set_index("sim_bin").reindex(BINLAB)
        ys = [s.loc[b, "accuracy"] * 100 if pd.notna(s.loc[b, "accuracy"]) else np.nan for b in BINLAB]
        lw = 2.6 if m == "RF" else 1.6
        zo = 5 if m == "RF" else 2
        # Connect only consecutive bins that are BOTH high-confidence (n>=50); a
        # low-confidence bin (n<50) is shown as an isolated hollow point with no
        # connecting segment, so the tiny-n swings do not read as a real trend.
        for i in range(3):
            if (n_by_bin[i] >= 50 and n_by_bin[i + 1] >= 50
                    and not np.isnan(ys[i]) and not np.isnan(ys[i + 1])):
                ax.plot([i, i + 1], [ys[i], ys[i + 1]], "-", color=MCOL[m], lw=lw, zorder=zo)
        labeled = False
        for x, yy, nn in zip(range(4), ys, n_by_bin):
            if not np.isnan(yy):
                ax.plot(x, yy, "o", color=MCOL[m], ms=8,
                        mfc=MCOL[m] if nn >= 50 else "white", mew=1.5, zorder=6,
                        label=(m if not labeled else None))
                labeled = True
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(BINLAB, n_by_bin)], fontsize=8)
    ax.set_title(f"{NICE[d]}  {'(LEAKY)' if d in LEAKY else '(clean control)'}", fontsize=11)
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(45, 102); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Accuracy vs test→train similarity bin (original split, k=6)\n"
             "hollow markers: n<50 (low-confidence bin)", fontsize=12)
fig.text(0.5, 0.005, "max 8-mer Jaccard of a test sequence to the training set", ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig_graded_performance.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig_graded_performance.{png,svg,pdf}")
