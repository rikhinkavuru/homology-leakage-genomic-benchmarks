#!/usr/bin/env python3
"""
Figure 4 -- the census landscape.

All fifty-one dataset-task censuses, one point each, x = near-duplicate leak
fraction at Jaccard 0.7, grouped by suite, marker by task type, colour by source
organism, with the 0.1 LEAKY cut as a vertical rule and the eleven GUE tasks
registered in advance as leaky drawn as open symbols.

Reads leakage_report_card.csv, crosssuite_census.csv, gue_census.csv,
bend_coordinate_census.csv (the last for the count only; BEND is coordinate-space
and carries no sequence-space leak fraction, so it is annotated, not plotted).

Usage:  ./venv/bin/python paper/Fig/make_fig_census.py
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito, one hue per source organism.
SPECIES = {
    "human":        "#0072B2",
    "mouse":        "#E69F00",
    "Drosophila":   "#009E73",
    "C. elegans":   "#CC79A7",
    "S. cerevisiae": "#56B4E9",
    "SARS-CoV-2":   "#D55E00",
}
# One marker per task type.
TASK = {
    "promoter": "o", "enhancer": "s", "splice site": "^",
    "histone / nucleosome": "D", "TF binding": "v", "other": "P",
}

gb = pd.read_csv(os.path.join(RES, "leakage_report_card.csv"))
nt = pd.read_csv(os.path.join(RES, "crosssuite_census.csv"))
gue = pd.read_csv(os.path.join(RES, "gue_census.csv"))
bend = pd.read_csv(os.path.join(RES, "bend_coordinate_census.csv"))

rows = []   # (suite, name, leak, species, task, registered_leaky, correction_unsafe)

GB_META = {
    "human_enhancers_ensembl": ("human", "enhancer"),
    "human_nontata_promoters": ("human", "promoter"),
    "human_ocr_ensembl": ("human", "other"),
    "human_enhancers_cohn": ("human", "enhancer"),
    "drosophila_enhancers_stark": ("Drosophila", "enhancer"),
    "demo_coding_vs_intergenomic_seqs": ("human", "other"),
    "demo_human_or_worm": ("C. elegans", "other"),
    "human_ensembl_regulatory (3-class)": ("human", "other"),
}
for _, r in gb.iterrows():
    sp, tk = GB_META[r.dataset]
    # correction_unsafe is READ FROM THE VERDICT, not hard-coded, so the figure and
    # Table 1 cannot drift apart: both derive from results/leakage_report_card.csv.
    rows.append(("Genomic Benchmarks", r.dataset, float(r.leak_at_0p7), sp, tk, False,
                 str(r.verdict) == "leaky-but-correction-unsafe"))


def nt_meta(task):
    if task.startswith("promoter"):
        return "human", "promoter"
    if task.startswith("enhancer"):
        return "human", "enhancer"
    if task.startswith("splice"):
        return "human", "splice site"
    return "S. cerevisiae", "histone / nucleosome"


for _, r in nt.iterrows():
    sp, tk = nt_meta(r.task)
    rows.append(("Nucleotide Transformer", f"{r.suite} {r.task}",
                 float(r.leak_jaccard_0p7), sp, tk, False, False))


def gue_meta(task):
    if task.startswith("emp_"):
        return "S. cerevisiae", "histone / nucleosome"
    if task.startswith("human_tf"):
        return "human", "TF binding"
    if task.startswith("mouse"):
        return "mouse", "TF binding"
    if task.startswith("prom"):
        return "human", "promoter"
    return "SARS-CoV-2", "other"


for _, r in gue.iterrows():
    sp, tk = gue_meta(r.task)
    rows.append(("GUE", r.task, float(r.leak_jaccard_0p7), sp, tk,
                 r.preregistered == "LEAKY", False))

df = pd.DataFrame(rows, columns=["suite", "name", "leak", "species", "task",
                                 "prereg_leaky", "correction_unsafe"])
assert len(df) == 51, len(df)

SUITES = ["Genomic Benchmarks", "Nucleotide Transformer", "GUE"]
fig, ax = plt.subplots(figsize=(7.0, 2.85))

rng = np.random.default_rng(0)
for i, s in enumerate(SUITES):
    sub = df[df.suite == s]
    y0 = len(SUITES) - 1 - i
    jit = np.linspace(-0.26, 0.26, len(sub)) if len(sub) > 1 else np.array([0.0])
    jit = jit[rng.permutation(len(sub))]
    for (_, r), dy in zip(sub.iterrows(), jit):
        # C5: the third verdict tier needs its own visual channel, and fill (prereg)
        # and edge colour (species) are both already spent -- so a grey halo drawn
        # underneath the marker, which composes with any shape/fill combination.
        if bool(getattr(r, "correction_unsafe", False)):
            ax.scatter(r.leak, y0 + dy, marker=TASK[r.task], s=118,
                       facecolor="none", edgecolor="0.45", lw=1.6, zorder=2)
        ax.scatter(r.leak, y0 + dy, marker=TASK[r.task], s=32,
                   facecolor="none" if r.prereg_leaky else SPECIES[r.species],
                   edgecolor=SPECIES[r.species], lw=1.1, zorder=3, alpha=0.95)

ax.axvline(0.1, color="k", ls="--", lw=1.0, zorder=1)

# Square-root x so the sub-0.1 region, where 44 of the 51 censuses sit, is legible.
ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                     lambda v: np.clip(v, 0, None) ** 2))
ticks = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.set_xlim(-0.004, 1.10)

ax.text(0.107, 2.70, "LEAKY cut, 0.1", fontsize=7, va="center")
ax.set_yticks([2, 1, 0])
# D1: one point per censused dataset-task ROW, but a row is not an independent task.
# The Nucleotide Transformer contributes 13 shipped tasks in two releases (26 rows),
# collapsing to 11 independent tasks; GUE ships 17 tasks, of which the 11 registered in
# advance collapse to 7 independent test partitions. Both conventions are on the panel so
# that 26 can be reconciled with 13 and with 11 without going to the text.
INDEP = {
    "Genomic Benchmarks": "8 rows = 8 datasets",
    "Nucleotide Transformer": "26 rows = 13 tasks $\\times$ 2 releases\n11 independent tasks",
    "GUE": "17 rows = 17 tasks\n11 registered $\\to$ 7 independent partitions",
}
ax.set_yticklabels([f"{s_}\n{INDEP[s_]}" for s_ in SUITES], fontsize=6.6)
ax.tick_params(axis="y", length=0, pad=6)
ax.set_ylim(-0.62, 3.05)
ax.set_xlabel("near-duplicate leak fraction at $8$-mer Jaccard $0.7$, full scale\n"
              "position on this axis is $\\sqrt{\\text{leak fraction}}$, not the leak "
              "fraction itself", fontsize=8)
ax.spines["left"].set_visible(False)

hs = [plt.Line2D([], [], marker=m, color="k", ls="", ms=5, mfc="none", label=t)
      for t, m in TASK.items()]
hp = [plt.Line2D([], [], marker="s", color=c, ls="", ms=5, label=sp)
      for sp, c in SPECIES.items()]
hp.append(plt.Line2D([], [], marker="o", mfc="none", mec="k", ls="", ms=5,
                     label="registered leaky in advance"))
hp.append(plt.Line2D([], [], marker="o", mfc="none", mec="0.45", ls="", ms=9, mew=1.6,
                     label="leaky, correction-unsafe"))
leg1 = ax.legend(handles=hs, frameon=False, fontsize=6.5, ncol=2, loc="upper right",
                 bbox_to_anchor=(1.005, 1.04), handletextpad=0.3,
                 columnspacing=1.0, labelspacing=0.28)
ax.add_artist(leg1)
ax.legend(handles=hp, frameon=False, fontsize=6.5, ncol=2, loc="center right",
          bbox_to_anchor=(1.005, 0.40), handletextpad=0.3,
          columnspacing=0.9, labelspacing=0.28)

fig.tight_layout(pad=0.4)
out = os.path.join(HERE, "fig_census")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
df.to_csv(out + "_data.csv", index=False)
print("wrote", out + ".pdf", "|", len(df), "censuses")
print(df.groupby("task").size().to_string())
print(df.groupby("species").size().to_string())
print("GUE prereg-leaky:", int(df.prereg_leaky.sum()),
      "range", df[df.prereg_leaky].leak.min(), df[df.prereg_leaky].leak.max())
