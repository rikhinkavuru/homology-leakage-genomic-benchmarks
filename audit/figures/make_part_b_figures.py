#!/usr/bin/env python3
"""
Part A/B figures + tables from the extended-model run.
  * capacity_scaling_extended.csv : homology drop for all 4 models x k=4/6 per dataset
  * fig_capacity_extended.{png,svg,pdf} : 8-point capacity curve (model x k), per dataset
  * fig_ranking_inversion.{png,svg,pdf} : rank slopegraph (orig vs corrected), leaky + clean
  * prints: LR/RF reproduction check vs frozen, and inversion magnitude (gap) analysis
Reads results/extended_models_long.csv + results/ranking_inversion.csv.
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
FIG = os.path.join(R, "figures")
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
POINTS = [(m, k) for m in MODELS for k in (4, 6)]              # 8 capacity points
LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
ORDER = ["human_nontata_promoters", "human_enhancers_ensembl",
         "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
         "demo_human_or_worm", "drosophila_enhancers_stark", "human_enhancers_cohn"]
NICE = {"human_nontata_promoters": "nonTATA promoters", "human_enhancers_ensembl": "enhancers (Ensembl)",
        "demo_coding_vs_intergenomic_seqs": "coding vs intergenomic", "human_ocr_ensembl": "OCR (Ensembl)",
        "demo_human_or_worm": "human vs worm", "drosophila_enhancers_stark": "Drosophila enh.",
        "human_enhancers_cohn": "enhancers (Cohn)"}
BLUE, VERM = "#0072B2", "#D55E00"
GREYS = ["#9a9a9a", "#b0b0b0", "#7f7f7f", "#c4c4c4", "#8a8a8a"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300})

df = pd.read_csv(os.path.join(R, "extended_models_long.csv"))
rank = pd.read_csv(os.path.join(R, "ranking_inversion.csv"))


def acc(dset, k, split, model, seed=None):
    q = (df.dataset == dset) & (df.k == k) & (df.split_type == split) & (df.model == model)
    if seed is not None:
        q = q & (df.seed == seed)
    return float(df[q]["accuracy"].mean())


# ---------- capacity_scaling_extended.csv ----------
rows = []
for d in ORDER:
    row = {"dataset": d, "leaky": d in LEAKY}
    for (m, k) in POINTS:
        o = acc(d, k, "original", m)
        h = acc(d, k, "homology@0.7", m)
        row[f"hom_delta_{m}_k{k}"] = round(o - h, 4)
    rows.append(row)
cap = pd.DataFrame(rows)
cap.to_csv(os.path.join(R, "capacity_scaling_extended.csv"), index=False)

# ---------- reproduction check vs frozen LR/RF ----------
print("== LR/RF reproduction check (extended run vs frozen full-scale) ==")
for d in LEAKY:
    for (m, k) in [("RF", 6), ("LR", 6)]:
        print(f"   {d} {m} k{k}: orig={acc(d,k,'original',m):.4f} hom={acc(d,k,'homology@0.7',m):.4f} "
              f"-> delta={acc(d,k,'original',m)-acc(d,k,'homology@0.7',m):+.4f}")

# ---------- Fig 1: 8-point capacity curve ----------
fig, ax = plt.subplots(figsize=(8.4, 5.2))
xs = np.arange(len(POINTS))
gi = 0
for d in ORDER:
    ys = [(acc(d, k, "original", m) - acc(d, k, "homology@0.7", m)) * 100 for (m, k) in POINTS]
    if d in LEAKY:
        col = BLUE if d == "human_nontata_promoters" else VERM
        ax.plot(xs, ys, marker="o", lw=2.8, ms=7, color=col, zorder=5, label=NICE[d])
    else:
        ax.plot(xs, ys, marker="o", lw=1.1, ms=3.5, color=GREYS[gi % len(GREYS)], alpha=0.7, label=NICE[d]); gi += 1
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels([f"{m}\nk{k}" for (m, k) in POINTS], fontsize=9)
ax.set_xlabel("model x feature-k  (capacity: low → high)")
ax.set_ylabel("accuracy drop under near-duplicate-aware re-split (points)")
ax.set_title("Capacity curve with 4 classical models (LR, LinearSVC, RF, HGB)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig_capacity_extended.{ext}"), bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: ranking slopegraph (k=6, accuracy) ----------
PANELS = ["human_nontata_promoters", "human_enhancers_ensembl", "demo_human_or_worm"]
MCOL = {"LR": "#0072B2", "LinearSVC": "#56B4E9", "RF": "#D55E00", "HGB": "#009E73"}
MARK = {"LR": "o", "LinearSVC": "^", "RF": "s", "HGB": "D"}   # distinct markers (grayscale/colorblind-safe)
LSTY = {"LR": "-", "LinearSVC": "--", "RF": "-", "HGB": "-"}    # LR solid vs LinearSVC dashed
fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
for ax, d in zip(axes, PANELS):
    o = {m: acc(d, 6, "original", m) for m in MODELS}
    c = {m: acc(d, 6, "homology@0.7", m) for m in MODELS}
    ro = sorted(MODELS, key=lambda m: -o[m])   # rank 1 = best
    rc = sorted(MODELS, key=lambda m: -c[m])
    for m in MODELS:
        y0, y1 = ro.index(m) + 1, rc.index(m) + 1
        lw = 3.2 if m == "RF" else 1.8
        ax.plot([0, 1], [y0, y1], color=MCOL[m], linestyle=LSTY[m], marker=MARK[m], lw=lw, ms=7, label=m)
        ax.text(-0.04, y0, m, ha="right", va="center", fontsize=8.5, color=MCOL[m])
        ax.text(1.04, y1, m, ha="left", va="center", fontsize=8.5, color=MCOL[m])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(4.5, 0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["original\n(leaky)", "near-dup.-\naware"])
    ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(["1st", "2nd", "3rd", "4th"] if d == PANELS[0] else [])
    # D6: Kendall tau is NOT annotated on the panels. The body text disclaims it as a
    # primary statistic at four models, and at four models it also orders the two leaky
    # panels opposite to the conclusions they carry. Printing it below keeps it available
    # without putting a disclaimed statistic on the figure.
    tau = float(rank[(rank.dataset == d) & (rank.k == 6) & (rank.metric == "accuracy")]["kendall_tau"].iloc[0])
    print(f"  [not plotted] kendall tau {d} k=6 accuracy = {tau:+.2f}")
    ax.set_title(f"{NICE[d]}\n{'LEAKY' if d in LEAKY else 'clean'}", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
fig.suptitle("Model ranking: original (leaky) vs near-duplicate-aware split  ($k$=6, accuracy)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig_ranking_inversion.{ext}"), bbox_inches="tight")
plt.close(fig)

# ---------- inversion magnitude analysis (leaky = large gap, clean = near-tie) ----------
print("\n== inversion magnitude (k=6 acc): accuracy gap of the dethroned pair, original split ==")
for d in ORDER:
    o = {m: acc(d, 6, "original", m) for m in MODELS}
    c = {m: acc(d, 6, "homology@0.7", m) for m in MODELS}
    gaps = []
    for a, b in itertools.combinations(MODELS, 2):
        if np.sign(o[a] - o[b]) != 0 and np.sign(c[a] - c[b]) != 0 and np.sign(o[a] - o[b]) != np.sign(c[a] - c[b]):
            gaps.append(abs(o[a] - o[b]))
    rf_orig_rank = sorted(MODELS, key=lambda m: -o[m]).index("RF") + 1
    rf_corr_rank = sorted(MODELS, key=lambda m: -c[m]).index("RF") + 1
    print(f"   {'LEAKY' if d in LEAKY else 'clean'} {d}: RF rank {rf_orig_rank}->{rf_corr_rank}; "
          f"max inversion gap={max(gaps) if gaps else 0:.4f}")
print("\nwrote capacity_scaling_extended.csv, fig_capacity_extended.*, fig_ranking_inversion.*")
