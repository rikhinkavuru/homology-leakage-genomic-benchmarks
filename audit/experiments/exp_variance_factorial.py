#!/usr/bin/env python3
"""
exp_variance_factorial.py -- E5: quantify "conditional on the fit" instead of
disclaiming it.

THE OBJECTION
-------------
Every interval in the paper resamples the test set while holding the fitted model
fixed, and the manuscript says so. That disclosure is honest but unquantified: a reader
cannot tell whether the held-fixed component is negligible beside the resampling error
or dominates it. Two marginal standard deviations are already reported -- five re-split
seeds and, separately, five training seeds -- but marginals cannot separate the two
sources from their interaction.

DESIGN
------
A full SPLIT-SEED x TRAINING-SEED factorial for the headline delta on each leaky
dataset. Split seed drives the whole-cluster assignment of the corrected partition;
training seed drives only `random_state` on the estimator. Each cell is one corrected
accuracy, and the as-shipped accuracy is refit per training seed so the delta varies
with both factors exactly as the reported statistic does.

The decomposition is a two-way random-effects ANOVA on the delta:

    var_total = var_split + var_train + var_residual(interaction)

reported as variance components and as percentages, with the implied standard
deviations, so the sentence "every interval is conditional on the fitted models" can be
replaced by a number.

Run:  python audit/experiments/exp_variance_factorial.py [--grid 5] [--datasets ...]
Out:  results/variance_factorial_cells.csv    one row per (dataset, split seed, train seed)
      results/variance_factorial.csv          the decomposition per dataset
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from audit.core import expkit as E


def rf(seed):
    """The report card's forest, with the training seed varied and nothing else."""
    return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=_R.N_JOBS)


def run(dset, grid):
    seqs, y, otr, ote, tf = E.load(dset, full=True)
    y = np.asarray(y)
    X = E.featurize(seqs, 6)
    comp = E.cached_clusters(dset, seqs, 0.7, 8)
    rows, t0 = [], time.time()
    # as-shipped accuracy depends on the training seed only
    orig = {}
    for ts in range(grid):
        orig[ts] = float((rf(ts).fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).mean())
        print(f"  [{dset}] as-shipped train-seed {ts}: {orig[ts]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for ss in range(grid):
        ctr, cte = E.assign(comp, y, ss, tf)
        for ts in range(grid):
            acc = float((rf(ts).fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).mean())
            rows.append(dict(dataset=dset, split_seed=ss, train_seed=ts,
                             acc_orig=round(orig[ts], 4), acc_corr=round(acc, 4),
                             delta=round(orig[ts] - acc, 4)))
            print(f"  [{dset}] split {ss} x train {ts}: corr={acc:.4f} "
                  f"delta={orig[ts]-acc:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def decompose(g, grid):
    """Two-way random-effects components on the delta, balanced design."""
    d = g.pivot(index="split_seed", columns="train_seed", values="delta").to_numpy()
    a, b = d.shape
    grand = d.mean()
    row_m, col_m = d.mean(1), d.mean(0)
    ss_split = b * ((row_m - grand) ** 2).sum()
    ss_train = a * ((col_m - grand) ** 2).sum()
    ss_resid = ((d - row_m[:, None] - col_m[None, :] + grand) ** 2).sum()
    ms_split = ss_split / (a - 1)
    ms_train = ss_train / (b - 1)
    ms_resid = ss_resid / ((a - 1) * (b - 1))
    # random-effects estimators; a negative component means "indistinguishable from 0"
    v_split = max((ms_split - ms_resid) / b, 0.0)
    v_train = max((ms_train - ms_resid) / a, 0.0)
    v_resid = ms_resid
    tot = v_split + v_train + v_resid
    return dict(
        n_cells=d.size, grid=grid, delta_mean=round(float(grand), 5),
        delta_min=round(float(d.min()), 5), delta_max=round(float(d.max()), 5),
        delta_sd_total=round(float(d.std(ddof=1)), 5),
        var_split=v_split, var_train=v_train, var_resid=v_resid,
        sd_split=round(float(np.sqrt(v_split)), 5),
        sd_train=round(float(np.sqrt(v_train)), 5),
        sd_resid=round(float(np.sqrt(v_resid)), 5),
        pct_split=round(100 * v_split / tot, 1) if tot else np.nan,
        pct_train=round(100 * v_train / tot, 1) if tot else np.nan,
        pct_resid=round(100 * v_resid / tot, 1) if tot else np.nan,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--datasets", nargs="*", default=list(E.LEAKY))
    a = ap.parse_args()
    print(_R.describe(), flush=True)
    cells, summ = [], []
    for dset in a.datasets:
        g = run(dset, a.grid)
        cells.append(g)
        pd.concat(cells).to_csv(
            os.path.join(E.RESULTS, "variance_factorial_cells.csv"), index=False)
        summ.append(dict(dataset=dset, **decompose(g, a.grid)))
        pd.DataFrame(summ).to_csv(
            os.path.join(E.RESULTS, "variance_factorial.csv"), index=False)
        print(pd.DataFrame(summ).tail(1).to_string(index=False), flush=True)
    print("\n" + pd.DataFrame(summ).to_string(index=False))
    print("EXP_VARIANCE_FACTORIAL_DONE")
