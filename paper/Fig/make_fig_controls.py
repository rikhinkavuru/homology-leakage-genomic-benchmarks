#!/usr/bin/env python3
"""
Rebuild fig_controls with BOTH a leaky panel and a clean panel.

Two things were wrong with the previous version:

  1. It plotted only the two leaky datasets, so the caption's second half
     ("and only on leaky datasets") had no support in the image.
  2. It was a bar chart, which forces a zero baseline; with accuracies spanning
     67-94% that wasted two thirds of the panel and compressed the very effect
     the figure exists to show, prompting an apologetic caption sentence.

This version shows all seven binary datasets -- the two leaky ones on the left,
the borderline and four clean ones on the right -- as a dumbbell plot. Point
markers encode position rather than area, so a non-zero axis is legitimate and
needs no apology.

All three points per dataset are derived from one baseline and the two reported
deltas, so the plotted values agree with the deltas quoted in the text:
    original  = original accuracy
    random    = original - delta_random          (negative control)
    corrected = original - delta_homology        (near-duplicate-aware split)

Leaky-dataset baselines and corrected accuracies use the reconciled convention
(.predict()/argmax, 5-seed-mean corrected) so they match the headline numbers in
the text (nontata 11.8 pts, ensembl 16.5 pts).

Reads only already-computed, validated CSVs. No model fitting, no CPU load.

Usage:  ./venv/bin/python paper/Fig/make_fig_controls.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUMMARY = os.path.join(ROOT, "results", "summary_final.csv")

# Okabe-Ito colorblind-safe palette, matching the suite's other figures.
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#8a8a8a"

# Helvetica Neue rather than the matplotlib default (DejaVu Sans), which reads as
# an unstyled default plot. Sans-serif in figures is the convention in this field.
# mathtext is pointed at the same family so the italic k in "k=6" matches the
# surrounding labels instead of falling back to DejaVu.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica Neue",
    "mathtext.it": "Helvetica Neue:italic",
    "mathtext.bf": "Helvetica Neue:bold",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.0, "axes.axisbelow": True,
    "axes.labelsize": 10.5, "figure.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed as TrueType, not Type 3
})

NICE = {
    "human_nontata_promoters": "nonTATA\npromoters",
    "human_enhancers_ensembl": "enhancers\n(Ensembl)",
    "demo_coding_vs_intergenomic_seqs": "coding vs\nintergenomic",
    "human_ocr_ensembl": "OCR\n(Ensembl)",
    "demo_human_or_worm": "human\nor worm",
    "drosophila_enhancers_stark": "Drosophila\nenhancers",
    "human_enhancers_cohn": "enhancers\n(Cohn)",
}

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
         "demo_human_or_worm", "drosophila_enhancers_stark",
         "human_enhancers_cohn"]

# Reconciled headline values for the two leaky datasets (see the NOTE block at the
# top of paper/main.tex). Clean datasets have a single convention and need no override.
RECONCILED = {
    "human_nontata_promoters": (0.9284, 0.8101),
    "human_enhancers_ensembl": (0.8596, 0.6951),
}

summ = pd.read_csv(SUMMARY).set_index("dataset")


def points_for(d):
    """(original, random, corrected) accuracies in percent: one baseline + two deltas."""
    if d in RECONCILED:
        orig, corr = RECONCILED[d]
    else:
        orig = summ.loc[d, "original_acc"]
        corr = orig - summ.loc[d, "delta_homology"]
    rand = orig - summ.loc[d, "delta_random"]
    return orig * 100, rand * 100, corr * 100


def ticklabel(d):
    return f"{NICE[d]}\nleak {summ.loc[d, 'leak_full_0p7']:.2f}"


fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.2, 3.6), sharey=True,
    gridspec_kw={"width_ratios": [len(LEAKY), len(CLEAN)], "wspace": 0.05},
)

SPECS = [("original split", BLUE, "o", 7.5),
         ("random re-split (control)", GREY, "s", 6.0),
         ("near-duplicate-aware split", VERM, "D", 6.5)]
DX = 0.17  # horizontal offset so coincident points stay distinguishable
rows = []

for ax, group, title in [(axL, LEAKY, "leaky"), (axR, CLEAN, "borderline + clean")]:
    xs = np.arange(len(group), dtype=float)
    for x, d in zip(xs, group):
        orig, rand, corr = points_for(d)
        # connector spanning the full range of the three conditions
        ax.plot([x, x], [min(orig, rand, corr), max(orig, rand, corr)],
                color="#bbbbbb", lw=1.4, zorder=1, solid_capstyle="round")
        for i, ((lab, color, marker, ms), val) in enumerate(zip(SPECS, (orig, rand, corr))):
            yerr = summ.loc[d, "homology_acc_std"] * 100 if i == 2 else None
            ax.errorbar(x + (i - 1) * DX, val, yerr=yerr, fmt=marker, ms=ms,
                        color=color, mec="white", mew=0.8, capsize=2.5,
                        ecolor="#555", elinewidth=1.0, zorder=3,
                        label=lab if (d == group[0]) else None)
            rows.append({"dataset": d, "group": title, "split": lab,
                         "accuracy_pct": round(float(val), 2),
                         "best_model": summ.loc[d, "best_model"],
                         "leak_full_0.7": summ.loc[d, "leak_full_0p7"]})
        # Annotate the drop only where it is material, i.e. on the leaky panel.
        # Labelling the clean deltas too would give a null the visual weight of an effect.
        if d in LEAKY:
            ax.annotate(f"−{orig - corr:.1f} pts", xy=(x + DX, corr),
                        xytext=(x + DX, corr - 2.2), fontsize=8.5, color=VERM,
                        va="top", ha="center", weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([ticklabel(d) for d in group], fontsize=7.6)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlim(-0.6, len(group) - 0.4)
    ax.yaxis.grid(True, alpha=0.25)

axL.set_ylabel("test accuracy (%)")
axL.set_ylim(62, 100)
axL.set_yticks(np.arange(65, 101, 5))
axR.tick_params(axis="y", length=0)

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.04), columnspacing=1.6,
           handletextpad=0.4, numpoints=1)
fig.subplots_adjust(top=0.80)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig_controls.{ext}"), bbox_inches="tight")
plt.close(fig)

pd.DataFrame(rows).to_csv(os.path.join(HERE, "fig_controls_data.csv"), index=False)
print("wrote fig_controls.{pdf,png} + fig_controls_data.csv to", HERE)
for r in rows:
    print(f"  {r['dataset']:34s} {r['split']:28s} {r['accuracy_pct']:6.2f}")
