#!/usr/bin/env python3
"""
PART 1 artifact: the per-dataset "leakage report card" -- the practical takeaway
("which Genomic Benchmarks datasets can you trust?"). Pure aggregation of frozen CSVs.
Writes results/leakage_report_card.csv and a screenshot-ready table figure
results/figures/fig_report_card.{png,svg,pdf}.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(R, "figures")
M4 = ["LR", "LinearSVC", "RF", "HGB"]

leak = pd.read_csv(f"{R}/leakage_full.csv").set_index("dataset")
sfin = pd.read_csv(f"{R}/summary_final.csv").set_index("dataset")
subsum = pd.read_csv(f"{R}/summary.csv").set_index("dataset")
ext = pd.read_csv(f"{R}/extended_models_long.csv")

# Leaky-dataset "acc drop" uses the bootstrap-consistent point estimate (seed-0
# corrected split; 95% CIs in PAPER_NUMBERS S16.2), matching manuscript Table 2.
# Clean rows keep the 3-seed-mean delta (~0). 3-seed/5-seed values are in S16.
try:
    _s3 = pd.read_csv(f"{R}/step3_delta_ci.csv")
    BOOT_DROP = {r["dataset"]: round(float(r["delta_orig_minus_corr"]), 3)
                 for _, r in _s3.iterrows() if r["group"] == "leaky" and r["model"] == "RF_k6"}
except Exception:
    BOOT_DROP = {}

# order: leaky first, then clean by descending leakage
BINARY = ["human_nontata_promoters", "human_enhancers_ensembl",
          "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
          "demo_human_or_worm", "drosophila_enhancers_stark", "human_enhancers_cohn"]


def xacc(d, split, m):
    q = (ext.dataset == d) & (ext.k == 6) & (ext.split_type == split) & (ext.model == m)
    return float(ext[q]["accuracy"].mean())


rows = []
for d in BINARY:
    n = int(leak.loc[d, "n_train"] + leak.loc[d, "n_test"])
    l7, l9 = float(leak.loc[d, "leak_full_0.7"]), float(leak.loc[d, "leak_full_0.9"])
    verdict = "LEAKY" if l7 > 0.1 else "clean"
    o = {m: xacc(d, "original", m) for m in M4}
    c = {m: xacc(d, "homology@0.7", m) for m in M4}
    top_o, top_c = max(M4, key=lambda m: o[m]), max(M4, key=lambda m: c[m])
    rf_o = sorted(M4, key=lambda m: -o[m]).index("RF") + 1
    rf_c = sorted(M4, key=lambda m: -c[m]).index("RF") + 1
    inverts = (top_o != top_c) and (o[top_o] - o[top_c] > 0.01)
    rows.append(dict(dataset=d, n_full=n, leak_at_0p7=round(l7, 3), leak_at_0p9=round(l9, 3),
                     verdict=verdict, best_model=sfin.loc[d, "best_model"],
                     acc_drop_corrected=BOOT_DROP.get(d, round(float(sfin.loc[d, "delta_homology"]), 3)),
                     ranking_inverts="yes" if inverts else "no",
                     rf_rank_orig_to_corr=f"{rf_o}->{rf_c}"))
# 3-class regulatory, noted separately (numbers from the subsample suite)
d = "human_ensembl_regulatory"
rows.append(dict(dataset=d + " (3-class)", n_full=int(leak.loc[d, "n_train"] + leak.loc[d, "n_test"]),
                 leak_at_0p7=round(float(leak.loc[d, "leak_full_0.7"]), 3),
                 leak_at_0p9=round(float(leak.loc[d, "leak_full_0.9"]), 3),
                 verdict="clean", best_model=str(subsum.loc[d, "best_model"]),
                 acc_drop_corrected=round(float(subsum.loc[d, "delta_acc_homology"]), 3),
                 ranking_inverts="n/a", rf_rank_orig_to_corr="n/a"))

rc = pd.DataFrame(rows)
rc.to_csv(f"{R}/leakage_report_card.csv", index=False)
print(rc.to_string(index=False))

# ---- table figure ----
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis("off")
cols = ["dataset", "n", "leak@0.7", "leak@0.9", "verdict", "best model",
        "acc drop\n(corrected)", "ranking\ninverts", "RF rank\norig->corr"]
cell = [[r["dataset"], f"{r['n_full']:,}", f"{r['leak_at_0p7']:.3f}", f"{r['leak_at_0p9']:.3f}",
         r["verdict"], r["best_model"], f"{r['acc_drop_corrected']:+.3f}",
         r["ranking_inverts"], r["rf_rank_orig_to_corr"]] for r in rows]
tab = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.6)
for j in range(len(cols)):
    tab[0, j].set_facecolor("#34495e"); tab[0, j].get_text().set_color("white"); tab[0, j].get_text().set_weight("bold")
for i, r in enumerate(rows, start=1):
    color = "#f8d7da" if r["verdict"] == "LEAKY" else "#d4edda"
    tab[i, 4].set_facecolor(color)
    if r["verdict"] == "LEAKY":
        for j in range(len(cols)):
            tab[i, j].set_facecolor("#fdeef0" if j != 4 else "#f8d7da")
ax.set_title("Genomic Benchmarks homology-leakage report card\n"
             "(LEAKY = test/train near-duplicate fraction > 0.1 at Jaccard 0.7; full-scale leakage)",
             fontsize=11, weight="bold")
for ext_ in ("png", "svg", "pdf"):
    fig.savefig(f"{FIG}/fig_report_card.{ext_}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nwrote leakage_report_card.csv + fig_report_card.{png,svg,pdf}")
