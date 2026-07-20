#!/usr/bin/env python3
"""
INTERVAL STABILITY AND MULTIPLICITY for the NT `enhancers` decontamination result.

exp_nt_enhancers_decontam.py reports two paired margin shifts whose 95% percentile
cluster-bootstrap intervals exclude zero: LR-vs-RF, shift +0.0400 [0.0100, 0.0700],
and LR-vs-HGB, +0.0375 [0.0100, 0.0675]. Both lower bounds sit at 0.0100. The margin
statistic on n = 400 singleton clusters moves in steps of 1/400 = 0.0025, so each of
those bounds is FOUR grid steps from zero, and each was read off a single bootstrap
seed (20240524). Two questions follow, and neither is answered by that run:

  1. STABILITY. Does the exclusion survive a different bootstrap seed? A percentile
     endpoint estimated from 4,000 draws carries Monte Carlo error of its own.
  2. MULTIPLICITY. Six model pairs were tested. The manuscript paragraph these results
     would join already applies Benjamini-Hochberg to its cross-suite grid and calls
     unadjusted flags "nominal". Reporting two raw 95% intervals there without the same
     adjustment would be internally inconsistent.

This module refits both arms (deterministic, MODEL_SEED-fixed), then for every pair
re-runs the paired across-arm cluster bootstrap under NSEED distinct bootstrap seeds,
records how often the 95% interval excludes zero, and computes a bootstrap two-sided
p-value per pair plus its Benjamini-Hochberg adjustment across the six pairs.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_nt_enhancers_decontam_stability
Out:  results/nt_enhancers_decontam_stability.csv
"""
from __future__ import annotations
import itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES
from audit.experiments.cluster_bootstrap import test_internal_clusters
from audit.experiments.exp_nt_enhancers_shipped import MODELS, THR, SIM_K, FEAT_K, _order

R = os.path.join(HERE, "results")
REPO = SUITES["NT-original"]
TASK = "enhancers"
BOOT = 4000
NSEED = 25


def shift_boot(dA, dB, clusters, seed, boot=BOOT):
    """Paired across-arm cluster bootstrap of (margin_decontam - margin_shipped)."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(clusters)
    idx = [np.where(clusters == u)[0] for u in uniq]
    sizes = np.array([len(i) for i in idx], float)
    sumA = np.array([dA[i].sum() for i in idx], float)
    sumB = np.array([dB[i].sum() for i in idx], float)
    G = len(uniq)
    dd = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, size=G)
        n = sizes[pick].sum()
        dd[b] = sumB[pick].sum() / n - sumA[pick].sum() / n
    lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    # two-sided bootstrap p: proportion of draws on the far side of zero, doubled
    p = 2.0 * min((dd <= 0).mean(), (dd >= 0).mean())
    return lo, hi, min(p, 1.0)


def bh(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return adj


def main():
    t0 = time.time()
    tr_s, tr_y = read_fna(fetch(REPO, TASK, "train"))
    te_s, te_y = read_fna(fetch(REPO, TASK, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    tr = np.arange(len(tr_s))
    te = np.arange(len(tr_s), len(seqs))
    X = E.featurize(seqs, FEAT_K)

    sim_tr_to_te = E.max_sim_to_train(seqs, te, tr, k=SIM_K, mode="jaccard")
    keep = tr[sim_tr_to_te < THR]
    print(f"  removed {len(tr)-len(keep)} of {len(tr)}", flush=True)

    corr, acc = {}, {}
    for arm, ti in (("shipped", tr), ("decontam", keep)):
        corr[arm] = {m: E.correctness(E.models()[m], X, y, ti, te) for m in MODELS}
        acc[arm] = {m: float(corr[arm][m].mean()) for m in MODELS}
        print(f"  [{arm}] {_order(acc[arm])}", flush=True)

    blk, ng = test_internal_clusters([seqs[i] for i in te], THR)
    print(f"  test-internal clusters {ng}/{len(te)}", flush=True)

    rows = []
    for A, B in itertools.combinations(MODELS, 2):
        dA = corr["shipped"][A] - corr["shipped"][B]
        dB = corr["decontam"][A] - corr["decontam"][B]
        los, his, ps = [], [], []
        for s in range(NSEED):
            lo, hi, p = shift_boot(dA, dB, blk, seed=20240524 + 7919 * s)
            los.append(lo); his.append(hi); ps.append(p)
        excl = sum(1 for lo, hi in zip(los, his) if lo > 0 or hi < 0)
        rows.append(dict(
            suite="NT-original", task=TASK, model_A=A, model_B=B,
            shift=round(float(dB.mean() - dA.mean()), 4),
            n_boot_seeds=NSEED, n_seeds_excluding_zero=excl,
            ci_lo_min=round(min(los), 4), ci_lo_max=round(max(los), 4),
            ci_hi_min=round(min(his), 4), ci_hi_max=round(max(his), 4),
            p_boot_median=round(float(np.median(ps)), 4),
            p_boot_min=round(float(np.min(ps)), 4),
            p_boot_max=round(float(np.max(ps)), 4)))
        print(f"  {A} vs {B}: shift {rows[-1]['shift']:+.4f}  "
              f"excl0 in {excl}/{NSEED} seeds  lo in "
              f"[{min(los):.4f},{max(los):.4f}]  p~{np.median(ps):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df["p_bh_across_6_pairs"] = bh(df["p_boot_median"].values).round(4)
    df["survives_bh_05"] = df["p_bh_across_6_pairs"] < 0.05

    out = f"{R}/nt_enhancers_decontam_stability.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, out)
    print("\n== stability and multiplicity ==")
    print(df[["model_A", "model_B", "shift", "n_seeds_excluding_zero", "ci_lo_min",
              "p_boot_median", "p_bh_across_6_pairs", "survives_bh_05"]].to_string(index=False))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")
    print("EXP_NT_ENHANCERS_DECONTAM_STABILITY_DONE")


if __name__ == "__main__":
    main()
