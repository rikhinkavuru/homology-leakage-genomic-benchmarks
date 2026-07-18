#!/usr/bin/env python3
"""
PART B: ranking inversion. Does correcting leakage change WHICH model looks best,
not just absolute scores? For each dataset and feature-k, rank the 4 models by test
accuracy (and AUROC) on the ORIGINAL (leaky) split vs the HOMOLOGY-AWARE split,
compute Kendall tau + Spearman between the two rankings, and flag pairwise inversions
that are STABLE across the 3 homology seeds. Control: clean datasets should keep
stable rankings (tau ~ 1), leaky datasets should scramble.

Reads results/extended_models_long.csv (from run_extended_models.py).
Writes results/ranking_inversion.csv + prints a summary.
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
SEEDS = [0, 1, 2]
LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
DATASETS = ["human_nontata_promoters", "human_enhancers_ensembl",
            "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
            "demo_human_or_worm", "drosophila_enhancers_stark", "human_enhancers_cohn"]

df = pd.read_csv(os.path.join(R, "extended_models_long.csv"))


def score(dset, k, split, model, metric, seed=None):
    q = ((df.dataset == dset) & (df.k == k) & (df.split_type == split) & (df.model == model))
    if seed is not None:
        q = q & (df.seed == seed)
    return float(df[q][metric].mean())


def ranking(scores):
    return sorted(MODELS, key=lambda m: -scores[m])


rows = []
for dset in DATASETS:
    for k in (4, 6):
        for metric in ("accuracy", "auroc"):
            orig = {m: score(dset, k, "original", m, metric) for m in MODELS}
            corr = {m: score(dset, k, "homology@0.7", m, metric) for m in MODELS}
            ro, rc = ranking(orig), ranking(corr)
            orank = [ro.index(m) for m in MODELS]
            crank = [rc.index(m) for m in MODELS]
            tau = kendalltau(orank, crank).statistic
            rho = spearmanr(orank, crank).statistic
            stable, allinv = [], []
            for a, b in itertools.combinations(MODELS, 2):
                o = np.sign(orig[a] - orig[b])
                c = np.sign(corr[a] - corr[b])
                if o != 0 and c != 0 and o != c:
                    # stability: does the flip hold on each of the 3 homology seeds?
                    flips = sum(1 for s in SEEDS
                                if np.sign(score(dset, k, "homology@0.7", a, metric, s)
                                           - score(dset, k, "homology@0.7", b, metric, s)) == c)
                    hi, lo = (a, b) if o > 0 else (b, a)   # hi outranks lo on original
                    tag = f"{hi}>{lo} (orig) -> {lo}>{hi} (corrected) [{flips}/3 seeds]"
                    allinv.append(tag)
                    if flips == len(SEEDS):
                        stable.append(tag)
            rows.append(dict(dataset=dset, leaky=dset in LEAKY, k=k, metric=metric,
                             original_ranking=">".join(ro), corrected_ranking=">".join(rc),
                             kendall_tau=round(tau, 3), spearman=round(rho, 3),
                             n_stable_inversions=len(stable),
                             stable_inversions=" ; ".join(stable),
                             all_inversions=" ; ".join(allinv)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(R, "ranking_inversion.csv"), index=False)

# ---- summary (headline uses k=6, accuracy) ----
print("\n===== RANKING INVERSION SUMMARY (k=6, accuracy) =====")
h = out[(out.k == 6) & (out.metric == "accuracy")]
for _, r in h.iterrows():
    print(f"{'LEAKY' if r.leaky else 'clean'} {r.dataset}: tau={r.kendall_tau:+.2f} "
          f"| orig [{r.original_ranking}] -> corr [{r.corrected_ranking}] "
          f"| stable inversions: {r.stable_inversions or 'none'}")
leaky = h[h.leaky]; clean = h[~h.leaky]
n_stable = int((h.n_stable_inversions > 0).sum())
print(f"\ndatasets with >=1 stable inversion (k6,acc): {n_stable}/{len(h)} "
      f"({', '.join(h[h.n_stable_inversions>0].dataset)})")
print(f"mean Kendall tau  LEAKY={leaky.kendall_tau.mean():+.3f}  CLEAN={clean.kendall_tau.mean():+.3f}")
print("wrote results/ranking_inversion.csv")
