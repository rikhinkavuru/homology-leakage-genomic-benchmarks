#!/usr/bin/env python3
"""
CHECK 2: novel-sequence-only ranking (no refit; pure aggregation of frozen CSVs).
For each leaky dataset, rank the 4 models (k=6) three ways:
  (a) original leaky-split accuracy        (extended_models_long.csv, split=original)
  (b) homology-aware-split accuracy        (extended_models_long.csv, split=homology@0.7)
  (c) novel-only accuracy: accuracy on the <0.5-similarity test bin
                                           (graded_performance.csv, sim_bin=[0,0.5))
Kendall tau between each pair. (b) and (c) are two independent routes to "is RF's
leaky-split lead real?". Run: python check2_novelonly_ranking.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
M4 = ["LR", "LinearSVC", "RF", "HGB"]
LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
ext = pd.read_csv(f"{R}/extended_models_long.csv")
graded = pd.read_csv(f"{R}/graded_performance.csv")


def eacc(d, split, m):
    q = (ext.dataset == d) & (ext.k == 6) & (ext.split_type == split) & (ext.model == m)
    return float(ext[q]["accuracy"].mean())


def novel(d, m):
    q = (graded.dataset == d) & (graded.model == m) & (graded.sim_bin == "[0,0.5)")
    return float(graded[q]["accuracy"].iloc[0])


def rank(sc):
    return sorted(M4, key=lambda m: -sc[m])


def tau(r1, r2):
    return round(kendalltau([r1.index(m) for m in M4], [r2.index(m) for m in M4]).statistic, 3)


rows = []
for d in LEAKY:
    a = {m: eacc(d, "original", m) for m in M4}
    b = {m: eacc(d, "homology@0.7", m) for m in M4}
    c = {m: novel(d, m) for m in M4}
    ra, rb, rc = rank(a), rank(b), rank(c)
    rows.append(dict(
        dataset=d,
        a_leaky_ranking=">".join(ra), b_homology_ranking=">".join(rb), c_novelonly_ranking=">".join(rc),
        rf_rank_leaky=ra.index("RF") + 1, rf_rank_homology=rb.index("RF") + 1, rf_rank_novelonly=rc.index("RF") + 1,
        rf_acc_novel=round(c["RF"], 4), best_novel_model=rc[0], best_novel_acc=round(c[rc[0]], 4),
        tau_a_b=tau(ra, rb), tau_a_c=tau(ra, rc), tau_b_c=tau(rb, rc),
        homology_novelonly_agree=bool(rb == rc)))
out = pd.DataFrame(rows)
out.to_csv(f"{R}/novelonly_ranking.csv", index=False)
for _, r in out.iterrows():
    print(f"\n{r['dataset']}:")
    print(f"  (a) leaky-split   : {r['a_leaky_ranking']}   (RF #{r['rf_rank_leaky']})")
    print(f"  (b) homology-aware: {r['b_homology_ranking']}   (RF #{r['rf_rank_homology']})")
    print(f"  (c) novel-only    : {r['c_novelonly_ranking']}   (RF #{r['rf_rank_novelonly']}; "
          f"RF novel acc={r['rf_acc_novel']}, best novel={r['best_novel_model']} {r['best_novel_acc']})")
    print(f"  tau: a-b={r['tau_a_b']}  a-c={r['tau_a_c']}  b-c={r['tau_b_c']}  "
          f"(homology==novelonly: {r['homology_novelonly_agree']})")
print("\nwrote results/novelonly_ranking.csv")
