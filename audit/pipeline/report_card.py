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

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
FIG = os.path.join(R, "figures")
M4 = ["LR", "LinearSVC", "RF", "HGB"]

leak = pd.read_csv(f"{R}/leakage_full.csv").set_index("dataset")
sfin = pd.read_csv(f"{R}/summary_final.csv").set_index("dataset")
subsum = pd.read_csv(f"{R}/summary.csv").set_index("dataset")
ext = pd.read_csv(f"{R}/extended_models_long.csv")

# Leaky-dataset "acc drop" is the seed-0 corrected-split point estimate. Table 2 of
# the manuscript reports the FIVE-SEED MEAN instead (0.118 / 0.165 against the
# 0.108 / 0.164 here), so the column is named acc_drop_seed0 rather than implying
# it matches the table. Both estimands are legitimate; conflating them is not.
# Clean rows keep the 3-seed-mean delta (~0). 3-seed/5-seed values are in S16.
try:
    _s3 = pd.read_csv(f"{R}/step3_delta_ci.csv")
    BOOT_DROP = {r["dataset"]: round(float(r["delta_orig_minus_corr"]), 3)
                 for _, r in _s3.iterrows() if r["group"] == "leaky" and r["model"] == "RF_k6"}
except Exception:
    BOOT_DROP = {}

# Full-scale containment, so the verdict can be three-way. Same source certify uses.
CONTAIN = {}
try:
    # exp_gb1_containment is the full-scale run for the two leaky datasets;
    # full_scale_containment covers the rest. certify.self_validate composes the same
    # two sources in the same order, so the card and the gate cannot disagree.
    _gb1 = pd.read_csv(f"{R}/exp_gb1_containment.csv").set_index("dataset")
    CONTAIN.update({d: float(v) for d, v in _gb1["leak_contain_0p7"].items()})
except Exception:
    pass
try:
    _fsc = pd.read_csv(f"{R}/full_scale_containment.csv").set_index("dataset")
    CONTAIN.update({d: float(v) for d, v in _fsc["leak_con_0p7"].items()})
except Exception:
    pass

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
    # Three-way, matching Table 2 and audit/tools/certify.verdict_from. The two-way
    # rule here labelled demo_coding "clean" while the paper calls it borderline on the
    # length-robust containment index -- the very finding the paper lists as a
    # contribution. The released card contradicted Table 2 on it until round 20.
    con7 = CONTAIN.get(d)
    verdict = ("LEAKY" if l7 > 0.1 else
               "borderline" if (con7 is not None and con7 > 0.1) else "clean")
    o = {m: xacc(d, "original", m) for m in M4}
    c = {m: xacc(d, "homology@0.7", m) for m in M4}
    top_o, top_c = max(M4, key=lambda m: o[m]), max(M4, key=lambda m: c[m])
    rf_o = sorted(M4, key=lambda m: -o[m]).index("RF") + 1
    rf_c = sorted(M4, key=lambda m: -c[m]).index("RF") + 1
    inverts = (top_o != top_c) and (o[top_o] - o[top_c] > 0.01)
    rows.append(dict(dataset=d, n_full=n, leak_at_0p7=round(l7, 3), leak_at_0p9=round(l9, 3),
                     leak_containment_0p7=(round(con7, 3) if con7 is not None else None),
                     verdict=verdict, best_model=sfin.loc[d, "best_model"],
                     acc_drop_seed0=BOOT_DROP.get(d, round(float(sfin.loc[d, "delta_homology"]), 3)),
                     top_model_changes_accuracy="yes" if inverts else "no",
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
fig, ax = plt.subplots(figsize=(15, 3.4))
ax.axis("off")
# Column set mirrors the CSV exactly. Round 20 renamed two row keys and left this
# block reading the old ones, so the script wrote a corrected CSV and then died with a
# KeyError before emitting the figure -- leaving the released figure frozen at the
# pre-correction state while the CSV beside it was right. The figure now carries
# containment and the three-way verdict, like Table 2.
cols = ["dataset", "n", "leak@0.7", "leak@0.9", "contain\n@0.7", "verdict", "best model",
        "acc drop\n(seed 0)", "top model\nchanges", "RF rank\norig->corr"]


def _cell(r):
    con = r.get("leak_containment_0p7")
    return [r["dataset"], f"{r['n_full']:,}", f"{r['leak_at_0p7']:.3f}",
            f"{r['leak_at_0p9']:.3f}",
            "n/a" if con is None or pd.isna(con) else f"{float(con):.3f}",
            r["verdict"], r["best_model"],
            f"{r.get('acc_drop_seed0', r.get('acc_drop_corrected', 0) or 0):+.3f}",
            r.get("top_model_changes_accuracy", r.get("ranking_inverts", "n/a")),
            r["rf_rank_orig_to_corr"]]


cell = [_cell(r) for r in rows]
tab = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.6)
# Without this the dataset column keeps matplotlib's default equal width and every
# name is clipped at both ends -- "an_enhancers_er" and "man_enhancers_" are two
# different datasets, and neither shows the suffix that tells them apart.
tab.auto_set_column_width(col=list(range(len(cols))))
for j in range(len(cols)):
    tab[0, j].set_facecolor("#34495e"); tab[0, j].get_text().set_color("white"); tab[0, j].get_text().set_weight("bold")
for i, r in enumerate(rows, start=1):
    color = ("#f8d7da" if r["verdict"] == "LEAKY"
             else "#fff3cd" if r["verdict"] == "borderline" else "#d4edda")
    tab[i, 4].set_facecolor(color)
    if r["verdict"] == "LEAKY":
        for j in range(len(cols)):
            tab[i, j].set_facecolor("#fdeef0" if j != 4 else "#f8d7da")
# Three-way rule, stated as Table 2 states it. The old title gave only the LEAKY
# condition, which left the borderline row unexplained once the verdict went three-way,
# and used "homology-leakage", the term the review asked us to retire in favour of
# "near-duplicate".
ax.set_title("Genomic Benchmarks near-duplicate leakage report card\n"
             "(LEAKY if the full-scale near-duplicate test fraction exceeds 0.1 at Jaccard 0.7; "
             "borderline if the length-robust containment index does)",
             fontsize=11, weight="bold")
for ext_ in ("png", "svg", "pdf"):
    fig.savefig(f"{FIG}/fig_report_card.{ext_}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nwrote leakage_report_card.csv + fig_report_card.{png,svg,pdf}")
