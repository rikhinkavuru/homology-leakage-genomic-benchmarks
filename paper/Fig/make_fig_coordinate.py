#!/usr/bin/env python3
"""
Figure 2 -- the coordinate signature.

Panel A: coordinate multiplicity of the positive class, leaky against clean.
Panel B: max-over-classes cross-split >=50% reciprocal coordinate overlap for all
         eight suite datasets and the four BEND partitions, against the 0.1 cut.

Reads only already-computed CSVs (construction_signatures.csv,
bend_coordinate_census.csv). No model fitting.

Usage:  ./venv/bin/python paper/Fig/make_fig_coordinate.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")

BLUE, VERM, GREY, GREEN = "#0072B2", "#D55E00", "#8a8a8a", "#009E73"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
})

sig = pd.read_csv(os.path.join(RES, "construction_signatures.csv"))
bend = pd.read_csv(os.path.join(RES, "bend_coordinate_census.csv"))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                               gridspec_kw={"width_ratios": [1.0, 1.55]})

# ---------------------------------------------------------------- Panel A
# human_enhancers_ensembl positives: 36,487 coordinates carry two rows, 4,447 one.
# human_ocr_ensembl positives: 87,378 coordinates, every one of multiplicity 1.
ens = sig[(sig.dataset == "human_enhancers_ensembl") & (sig.cls == "positive")].iloc[0]
ocr = sig[(sig.dataset == "human_ocr_ensembl") & (sig.cls == "positive")].iloc[0]

n_dup = int(ens.n_raw - ens.n_unique_coords)          # 36,487
n_sing = int(ens.n_unique_coords - n_dup)             # 4,447
ocr_sing = int(ocr.n_unique_coords)                   # 87,378

x = np.array([0.0, 1.0])
w = 0.34
axA.bar(x - w / 2, [n_sing, n_dup], width=w, color=VERM,
        label="h. enhancers ensembl (LEAKY)")
axA.bar(x + w / 2, [ocr_sing, 0], width=w, color=BLUE,
        label="h. ocr ensembl (clean)")
for xi, v in [(0 - w / 2, n_sing), (1 - w / 2, n_dup), (0 + w / 2, ocr_sing)]:
    axA.text(xi, v + 2500, f"{v:,}", ha="center", va="bottom", fontsize=7)
axA.text(1 + w / 2, 2500, "0", ha="center", va="bottom", fontsize=7, color=BLUE)
axA.set_xticks(x)
axA.set_xticklabels(["1 row", "2 rows"])
axA.set_xlim(-0.55, 1.55)
axA.set_xlabel("rows sharing one shipped coordinate")
axA.set_ylabel("distinct coordinates")
axA.set_ylim(0, 138000)
axA.set_yticks([0, 25000, 50000, 75000, 100000])
axA.set_yticklabels(["0", "25k", "50k", "75k", "100k"])
axA.legend(frameon=False, fontsize=6.5, loc="upper center", handlelength=1.0,
           borderpad=0.1, labelspacing=0.35, handletextpad=0.5)
axA.set_title("A   Positive-class coordinate multiplicity", fontsize=8,
              loc="left", pad=5)

# ---------------------------------------------------------------- Panel B
rows = []
for ds, label in [
    ("human_nontata_promoters", "h. nontata promoters"),
    ("human_enhancers_ensembl", "h. enhancers ensembl"),
    ("demo_human_or_worm", "demo human or worm"),
    ("drosophila_enhancers_stark", "d. enhancers stark"),
    ("demo_coding_vs_intergenomic_seqs", "demo coding vs interg."),
    ("human_ocr_ensembl", "h. ocr ensembl"),
    ("human_enhancers_cohn", "h. enhancers cohn"),
    ("human_ensembl_regulatory", "h. ensembl regulatory"),
]:
    sub = sig[(sig.dataset == ds) & (sig.cls != "ALL")]
    rows.append((label, float(sub["xsplit_ge50pct"].max()), "GB"))
for _, r in bend.iterrows():
    rows.append(("BEND " + r.task.replace("_", " "), float(r.xsplit_ge50pct), "BEND"))

labels = [r[0] for r in rows]
vals = np.array([r[1] for r in rows])
suite = [r[2] for r in rows]
y = np.arange(len(rows))[::-1]

cols = [VERM if v > 0.1 else (GREEN if s == "BEND" else BLUE)
        for v, s in zip(vals, suite)]
# F2: a measured 0.000 must not read as a missing bar. Non-zero values get a filled
# bar with a floor so the smallest stays visible; exact zeros get an OPEN hairline
# stub of the same floor width -- present on the page, and a different glyph from any
# filled bar, so "measured zero" and "small but non-zero" cannot be confused.
STUB = 0.011
is_zero = vals == 0.0
axB.barh(y[~is_zero], np.maximum(vals[~is_zero], STUB),
         color=[c for c, z in zip(cols, is_zero) if not z], height=0.62)
axB.barh(y[is_zero], np.full(is_zero.sum(), STUB), height=0.62,
         facecolor="none", edgecolor=[c for c, z in zip(cols, is_zero) if z],
         linewidth=0.9)
axB.axvline(0.1, color="k", lw=1.0, ls="--", zorder=5)
axB.text(0.115, y[0] + 0.75, "LEAKY cut, 0.1", fontsize=6.8, va="center")
for yi, v in zip(y, vals):
    txt = "0.000 (open stub)" if v == 0.0 else f"{v:.3f}"
    axB.text(max(v, STUB) + 0.015, yi, txt, va="center", fontsize=6.8,
             zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))
axB.set_yticks(y)
axB.set_yticklabels(labels, fontsize=7)
axB.set_ylim(-0.8, len(rows) - 0.1)
axB.set_xlim(0, 1.18)
axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axB.set_xlabel("test intervals overlapping a train interval\nat $\\geq$50% reciprocal overlap",
               fontsize=7.5)
axB.set_title("B   Cross-split coordinate overlap, maximum over classes",
              fontsize=8, loc="left", pad=5)
axB.spines["left"].set_visible(False)
axB.tick_params(axis="y", length=0)

fig.tight_layout(pad=0.4, w_pad=1.4)
out = os.path.join(HERE, "fig_coordinate_signature")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
pd.DataFrame({"item": labels, "xsplit_ge50pct": vals, "suite": suite}).to_csv(
    out + "_data.csv", index=False)
print("wrote", out + ".pdf")
