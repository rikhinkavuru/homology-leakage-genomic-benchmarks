#!/usr/bin/env python3
"""
CONTROL for exp_nt_enhancers_shipped.py -- is the reordering de-leaking, or is it the
test set changing underneath us?

The shipped arm scores 0.73-0.77 and the cluster-corrected arm 0.86-0.89. Correction
RAISING accuracy by 12 points is the opposite of what leakage predicts, and it means the
shipped 400 are not an exchangeable draw from the pool: they are systematically harder.
So the shipped-vs-corrected contrast changes TWO things at once, leakage and test-set
composition, and the observed reordering cannot be attributed to leakage alone.

This module separates them with a third arm that changes composition WITHOUT removing
leakage: a plain RANDOM stratified split of the pooled data at the shipped test fraction
(n_test = 400), cluster-BLIND, so near-duplicates cross the boundary freely. Its leak
fraction is measured, not assumed.

  arm A  as-shipped        shipped composition,  leaky        (from the shipped module)
  arm B  random re-split   pooled composition,   leaky        <- THIS MODULE
  arm C  cluster-corrected pooled composition,   de-leaked    (from the shipped module)

A -> B isolates composition. B -> C isolates de-leaking. If B already orders like C, the
reordering is composition and the inversion claim does NOT survive. If B orders like A
and only C differs, de-leaking is doing the work.

Seeds: the random arm is run over 10 seeds, because a single 400-sequence draw is itself
noisy and one seed could land either way by chance. We report the ordering distribution
across seeds, not a single draw.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_nt_enhancers_control
Out:  results/nt_enhancers_control_arms.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES
from audit.experiments.cluster_bootstrap import test_internal_clusters
from audit.experiments.exp_nt_enhancers_shipped import (
    MODELS, THR, SIM_K, FEAT_K, _order, paired_margin_boot, mcnemar_counts)

R = os.path.join(HERE, "results")
REPO = SUITES["NT-original"]
TASK = "enhancers"
SEEDS = list(range(10))


def stratified_split(y, test_n, seed):
    """Random split holding the pooled class balance, cluster-BLIND."""
    rng = np.random.RandomState(seed)
    te = []
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        idx = np.where(y == c)[0]
        take = int(round(test_n * cnt / len(y)))
        te.append(rng.choice(idx, size=take, replace=False))
    te = np.sort(np.concatenate(te))
    tr = np.setdiff1d(np.arange(len(y)), te)
    return tr, te


def main():
    t0 = time.time()
    tr_s, tr_y = read_fna(fetch(REPO, TASK, "train"))
    te_s, te_y = read_fna(fetch(REPO, TASK, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    n_test = len(te_s)
    ship_tr = np.arange(len(tr_s))
    ship_te = np.arange(len(tr_s), len(seqs))
    print(f"[{TASK}] pooled n={len(seqs)}  shipped test n={n_test}", flush=True)
    print(f"  class balance  pooled={y.mean():.4f}  "
          f"shipped_train={y[ship_tr].mean():.4f}  shipped_test={y[ship_te].mean():.4f}",
          flush=True)

    X = E.featurize(seqs, FEAT_K)

    rows = []
    for seed in SEEDS:
        tr, te = stratified_split(y, n_test, seed)
        sim = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
        phi = float((sim >= 0.9).mean())
        acc, corr = {}, {}
        for m in MODELS:
            c = E.correctness(E.models()[m], X, y, tr, te)
            corr[m] = c
            acc[m] = float(c.mean())
        blk, _ = test_internal_clusters([seqs[i] for i in te], THR)
        mpt, lo, hi, se, excl = paired_margin_boot(corr["RF"], corr["HGB"], blk)
        b_only, c_only = mcnemar_counts(corr["RF"], corr["HGB"])
        print(f"  seed {seed}: phi={phi:.4f}  {_order(acc)}  "
              f"RF-HGB={mpt:+.4f} [{lo:.4f}, {hi:.4f}]  ({time.time()-t0:.0f}s)",
              flush=True)
        rows.append(dict(
            suite="NT-original", task=TASK, arm="random_resplit_cluster_blind",
            seed=seed, n_test=len(te), phi=round(phi, 4),
            **{f"acc_{m}": round(acc[m], 4) for m in MODELS},
            order=_order(acc), best=_order(acc).split(">")[0],
            margin_RF_minus_HGB=round(mpt, 4),
            margin_ci95=f"[{lo:.4f}, {hi:.4f}]", margin_se=round(se, 4),
            margin_excl0=excl, mcnemar_b=b_only, mcnemar_c=c_only,
            mde_from_boot_se=round(1.96 * se, 4)))

    df = pd.DataFrame(rows)
    out = f"{R}/nt_enhancers_control_arms.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, out)

    print("\n== random cluster-BLIND re-split at n_test=400, 10 seeds ==")
    print(df[["seed", "phi", "acc_LR", "acc_LinearSVC", "acc_RF", "acc_HGB",
              "best", "margin_RF_minus_HGB", "margin_ci95",
              "margin_excl0"]].to_string(index=False))
    print(f"\n  mean phi                    : {df.phi.mean():.4f}")
    print(f"  mean accuracy (over models) : "
          f"{df[[f'acc_{m}' for m in MODELS]].values.mean():.4f}")
    print("  best-model counts           : "
          f"{df.best.value_counts().to_dict()}")
    print(f"  RF>HGB in                   : {int((df.margin_RF_minus_HGB>0).sum())}"
          f"/{len(df)} seeds")
    print(f"  mean RF-HGB margin          : {df.margin_RF_minus_HGB.mean():+.4f}")
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")
    print("EXP_NT_ENHANCERS_CONTROL_DONE")


if __name__ == "__main__":
    main()
