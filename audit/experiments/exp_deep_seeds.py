#!/usr/bin/env python3
"""
Seed-replicated, interval-bearing CNN comparison on human_enhancers_ensembl.

WHY THIS EXISTS
---------------
Section 4.4 currently concedes that it cannot say whether a CNN-based leaderboard on
this suite is affected. The reason is a power problem, not a modelling one: the
pre-registered reference cell scores 0.8131 on the corrected split, which would place
it first, but that is a SINGLE run, and the nine-cell regularization grid's corrected
accuracies span 0.749-0.813 with LinearSVC's 0.7975 falling inside -- five cells above,
four below. One run inside a spread that straddles the comparison value supports no
ranking claim.

This script replicates the reference cell across seeds so the comparison acquires an
interval. It answers exactly one question: on the corrected split, does the CNN's
accuracy separate from LinearSVC's 0.7975, or not?

Either answer is publishable. If the CNN separates upward, the suite under-rates the
architecture family it publishes baselines for. If it does not separate, section 4.4's
existing refusal to claim a CNN win becomes a measured null rather than a hedge.

WHAT IT DOES NOT TOUCH
----------------------
Writes results/exp_deep_cnn_seeds.csv only. The committed 20-cell grid in
results/exp_deep_cnn.csv is quoted throughout the manuscript and is NOT regenerated
here -- exp_deep.py rewrites that file unconditionally on every run, so it must not be
invoked to produce these numbers.

`seed` varies training stochasticity only (weight init, dropout masks, batch order).
The split, the internal validation slice, and the data are identical across seeds, so
the spread measured here is training variance and nothing else.

USAGE
    ./venv/bin/python -m audit.experiments.exp_deep_seeds              # 5 seeds, ~45-60 min
    EXP_SEEDS_SMOKE=1 ./venv/bin/python -m audit.experiments.exp_deep_seeds   # crash test
    EXP_SEEDS_N=3 ./venv/bin/python -m audit.experiments.exp_deep_seeds       # fewer seeds
"""
from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd
import torch

from audit.core import expkit as E
from audit.experiments.exp_deep import (
    prep_dataset, train_and_correct, drop_ci, DEVICE,
)

SMOKE = os.environ.get("EXP_SEEDS_SMOKE") == "1"
N_SEEDS = int(os.environ.get("EXP_SEEDS_N", "5"))

DATASET = "human_enhancers_ensembl"
# The pre-registered standard-practice reference cell (results/deep_preregistration.md).
REF_DROPOUT, REF_WD = 0.3, 1e-4

# Published comparison value: LinearSVC's corrected accuracy on this dataset (Table 3).
# Used only to phrase the verdict; not an input to any computation.
LINEARSVC_CORRECTED = 0.7975

OUT = "results/exp_deep_cnn_seeds.csv"
# A smoke run must never be able to leave a file at the production path. Its numbers are
# meaningless -- it subsamples the head of the test index, which on this dataset is
# single-class, so it reports accuracy 1.0 by construction.
if SMOKE:
    OUT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "exp_deep_cnn_seeds_SMOKE.csv")


def main():
    t0 = time.time()
    seeds = list(range(N_SEEDS))
    max_epochs = 2 if SMOKE else None

    print(f"device={DEVICE} dataset={DATASET} cell=(dropout={REF_DROPOUT}, wd={REF_WD}) "
          f"seeds={seeds}{' [SMOKE]' if SMOKE else ''}", flush=True)

    P = prep_dataset(DATASET, t0)
    X = torch.from_numpy(P["X"])
    y = P["y"]
    otr, ote, ctr, cte = P["otr"], P["ote"], P["ctr"], P["cte"]
    ocl, ccl = P["ocl"], P["ccl"]

    if SMOKE:
        # Crash-test only: shrink the training sets so the loop runs in ~a minute.
        otr, ctr = otr[:2000], ctr[:2000]
        ote, cte = ote[:1000], cte[:1000]
        ocl, ccl = ocl[:1000], ccl[:1000]

    rows = []
    for s in seeds:
        tc = time.time()
        kw = {"seed": s}
        if max_epochs is not None:
            kw["max_epochs"] = max_epochs
        co, acc_o, ep_o = train_and_correct(X, y, otr, ote, REF_DROPOUT, REF_WD, False, **kw)
        cc, acc_c, ep_c = train_and_correct(X, y, ctr, cte, REF_DROPOUT, REF_WD, False, **kw)
        (slo, shi, sx), (klo, khi, kx) = drop_ci(co, cc, ocl, ccl)
        rows.append(dict(
            dataset=DATASET, seed=s, dropout=REF_DROPOUT, weight_decay=REF_WD,
            acc_orig=round(acc_o, 4), acc_corr=round(acc_c, 4),
            drop=round(acc_o - acc_c, 4),
            drop_ci_cluster_lo=round(klo, 4), drop_ci_cluster_hi=round(khi, 4),
            drop_excl0_cluster=kx,
            drop_ci_sample_lo=round(slo, 4), drop_ci_sample_hi=round(shi, 4),
            drop_excl0_sample=sx,
            epochs_orig=ep_o, epochs_corr=ep_c))
        print(f"  seed={s}: acc_o={acc_o:.4f} acc_c={acc_c:.4f} "
              f"drop={acc_o-acc_c:+.4f} clCI=[{klo:+.4f},{khi:+.4f}] excl0={kx} "
              f"({time.time()-tc:.0f}s, tot {time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint every seed

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    corr = df["acc_corr"].to_numpy()
    mean, sd = float(corr.mean()), float(corr.std(ddof=1)) if len(corr) > 1 else 0.0
    # Normal-approximation interval on the mean across seeds. With n=5 this is a
    # descriptive spread, not a strong inferential claim, and is reported as such.
    half = 1.96 * sd / np.sqrt(len(corr)) if len(corr) > 1 else 0.0
    lo, hi = mean - half, mean + half

    print("\n" + df.to_string(index=False), flush=True)
    print(f"\ncorrected accuracy over {len(corr)} seeds: mean={mean:.4f} sd={sd:.4f} "
          f"min={corr.min():.4f} max={corr.max():.4f}", flush=True)
    print(f"95% normal interval on the mean: [{lo:.4f}, {hi:.4f}]", flush=True)
    print(f"LinearSVC corrected (published): {LINEARSVC_CORRECTED:.4f}", flush=True)

    if lo > LINEARSVC_CORRECTED:
        verdict = "CNN SEPARATES ABOVE LinearSVC"
    elif hi < LINEARSVC_CORRECTED:
        verdict = "CNN SEPARATES BELOW LinearSVC"
    else:
        verdict = ("NO SEPARATION -- the seed interval straddles LinearSVC; "
                   "section 4.4's refusal to claim a CNN win is a measured null")
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\nEXP_DEEP_SEEDS_DONE {round(time.time()-t0)}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
