#!/usr/bin/env python3
"""
Paired CNN-vs-LinearSVC comparison on the corrected human_enhancers_ensembl split.

WHY THIS EXISTS
---------------
exp_deep_seeds.py established that the CNN's corrected accuracy is stable across
training seeds (0.8028-0.8131, mean 0.8086) and that every seed exceeds the linear
SVM's 0.7975. That is suggestive but it is NOT sufficient to claim the CNN outranks
the SVM, because the manuscript's own inversion criterion (Methods 3.3) requires a
PAIRED cluster-bootstrap interval excluding zero before any ranking claim is made, and
section 4.4 explicitly declines to violate that criterion. Comparing a seed-replicated
mean against a deterministic point estimate does not meet the bar the paper sets for
itself.

This script computes the interval the criterion asks for: both models scored on the
SAME corrected test set, differenced per example, and bootstrapped over within-test
near-duplicate clusters.

THE VALIDATION GATE
-------------------
Before anything is compared, LinearSVC is refitted on this script's corrected split and
its accuracy checked against the published 0.7975 (results/roster_rankings.csv). If it
does not reproduce, exp_deep's corrected partition is NOT the partition the roster used,
the comparison in section 4.4 is between numbers from different splits, and this script
aborts rather than emitting a paired interval that would silently inherit the mismatch.
That check is the reason this is a separate script and not an addendum.

LinearSVC is deterministic given a split (dual=False), so it contributes no training
variance; the CNN's spread is training stochasticity alone. Both are scored on one fixed
partition, so test-set sampling uncertainty is shared and cancels in the paired
difference -- which is exactly what makes the paired form the right instrument here.

Writes results/exp_deep_cnn_paired.csv only. Does not touch exp_deep_cnn.csv or
exp_deep_cnn_seeds.csv.

USAGE
    ./venv/bin/python -m audit.experiments.exp_deep_paired
    EXP_PAIRED_N=2 ./venv/bin/python -m audit.experiments.exp_deep_paired   # fewer seeds
"""
from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd
import torch

from audit.core import expkit as E
from audit.experiments.exp_deep import (
    prep_dataset, train_and_correct, DEVICE, BSEED, BOOT,
)

DATASET = "human_enhancers_ensembl"
REF_DROPOUT, REF_WD = 0.3, 1e-4
K = 6                                  # the roster's feature setting
N_SEEDS = int(os.environ.get("EXP_PAIRED_N", "5"))

# Published LinearSVC corrected accuracy for this dataset (results/roster_rankings.csv,
# quoted as 0.7975 in Table 3 and section 4.4). The gate below must reproduce it.
PUBLISHED_SVC_CORR = 0.7975
GATE_TOL = 0.0015

OUT = "results/exp_deep_cnn_paired.csv"


def main():
    t0 = time.time()
    seeds = list(range(N_SEEDS))
    print(f"device={DEVICE} dataset={DATASET} k={K} seeds={seeds}", flush=True)

    # --- split + clusters, from the same cache exp_deep_seeds used -------------
    P = prep_dataset(DATASET, t0)
    ctr, cte, ccl = P["ctr"], P["cte"], P["ccl"]
    y = P["y"]
    seqs, y2, _, _, _ = E.load(DATASET)
    assert len(seqs) == len(y) and np.array_equal(np.asarray(y2), np.asarray(y)), \
        "E.load ordering does not match the cached prep -- refusing to compare"
    print(f"corrected split: n_train={len(ctr)} n_test={len(cte)} "
          f"clusters={len(np.unique(ccl))} ({time.time()-t0:.0f}s)", flush=True)

    # --- LinearSVC on the corrected split, and the validation gate -------------
    Xf = E.featurize(seqs, K)
    print(f"featurized {Xf.shape} ({time.time()-t0:.0f}s)", flush=True)
    svc = E.models()["LinearSVC"]
    svc_correct = E.correctness(svc, Xf, y, ctr, cte)
    svc_acc = float(svc_correct.mean())
    del Xf                                          # 2.5 GB; free before the CNN
    print(f"LinearSVC corrected acc = {svc_acc:.4f} "
          f"(published {PUBLISHED_SVC_CORR:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    if abs(svc_acc - PUBLISHED_SVC_CORR) > GATE_TOL:
        raise SystemExit(
            f"\nGATE FAILED: LinearSVC scores {svc_acc:.4f} on this corrected split but "
            f"the manuscript publishes {PUBLISHED_SVC_CORR:.4f} (tolerance {GATE_TOL}).\n"
            f"The deep-model split and the roster split are therefore NOT the same "
            f"partition, and section 4.4's existing CNN-vs-LinearSVC comparison is "
            f"between numbers from different splits. That is a manuscript defect to "
            f"report, not something to paper over with a paired interval. Aborting.")
    print("GATE PASSED: same corrected partition as the roster.\n", flush=True)

    # --- CNN across seeds, paired against LinearSVC ---------------------------
    X = torch.from_numpy(P["X"])
    rows, cc_all = [], []
    for s in seeds:
        tc = time.time()
        cc, acc_c, ep = train_and_correct(X, y, ctr, cte, REF_DROPOUT, REF_WD,
                                          False, seed=s)
        cc_all.append(cc)
        diff = cc - svc_correct                     # paired, same examples
        draws = E.cluster_boot(diff, ccl, BOOT, seed=BSEED)
        lo, hi = E.ci(draws)
        rows.append(dict(
            dataset=DATASET, seed=s, cnn_acc_corr=round(float(acc_c), 4),
            svc_acc_corr=round(svc_acc, 4),
            paired_diff=round(float(diff.mean()), 4),
            paired_ci_lo=round(float(lo), 4), paired_ci_hi=round(float(hi), 4),
            paired_excl0=bool(lo > 0 or hi < 0), epochs=ep))
        print(f"  seed={s}: cnn={acc_c:.4f} svc={svc_acc:.4f} "
              f"diff={diff.mean():+.4f} clCI=[{lo:+.4f},{hi:+.4f}] "
              f"excl0={bool(lo>0 or hi<0)} ({time.time()-tc:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    # --- seed-averaged estimand ----------------------------------------------
    # Per-example expected correctness over seeds, differenced against the SVM. This
    # is the advantage of "the CNN procedure", not of any one run.
    cc_mean = np.mean(np.vstack(cc_all), axis=0)
    diff_mean = cc_mean - svc_correct
    draws = E.cluster_boot(diff_mean, ccl, BOOT, seed=BSEED)
    lo, hi = E.ci(draws)

    # COMBINED-SOURCE interval -- the one to quote.
    #
    # The `mean`-row bootstrap above resamples TEST CLUSTERS ONLY, applied to
    # per-example correctness that has already been averaged over seeds. That averaging
    # removes training variance from the point estimate, so the resulting interval is
    # narrower than any single seed's -- it describes a five-model ensemble nobody
    # trains, not the single CNN a practitioner runs. Quoting it would understate the
    # uncertainty the seed replication was run to expose.
    #
    # So we fold seed variance back in, by the same rule the manuscript already uses for
    # its other combined-source intervals (exp_clusterboot_full.py): add the raw
    # seed-to-seed SD in quadrature with the bootstrap SD. seed_sd is NOT divided by
    # sqrt(n), because the estimand is one run's advantage, not the mean of five.
    per_seed_diffs = np.array([r["paired_diff"] for r in rows], dtype=float)
    seed_sd = float(np.std(per_seed_diffs, ddof=1))
    boot_sd = float(np.std(draws))
    comb_sd = float(np.sqrt(boot_sd ** 2 + seed_sd ** 2))
    point = float(diff_mean.mean())
    clo, chi = point - 1.96 * comb_sd, point + 1.96 * comb_sd

    rows.append(dict(
        dataset=DATASET, seed="mean", cnn_acc_corr=round(float(cc_mean.mean()), 4),
        svc_acc_corr=round(svc_acc, 4), paired_diff=round(point, 4),
        paired_ci_lo=round(float(lo), 4), paired_ci_hi=round(float(hi), 4),
        paired_excl0=bool(lo > 0 or hi < 0), epochs="",
        seed_sd=round(seed_sd, 4), boot_sd=round(boot_sd, 4),
        combined_sd=round(comb_sd, 4),
        combined_ci_lo=round(clo, 4), combined_ci_hi=round(chi, 4),
        combined_excl0=bool(clo > 0 or chi < 0)))
    print(f"\ncombined-source (seed + test variance): {point:+.4f} "
          f"[{clo:+.4f}, {chi:+.4f}]  seed_sd={seed_sd:.4f} boot_sd={boot_sd:.4f}",
          flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + df.to_string(index=False), flush=True)
    n_excl = sum(1 for r in rows[:-1] if r["paired_excl0"])
    print(f"\nper-seed paired intervals excluding zero: {n_excl}/{len(seeds)}", flush=True)
    print(f"seed-averaged paired difference: {diff_mean.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]", flush=True)
    if lo > 0 and n_excl == len(seeds):
        verdict = ("CNN OUTRANKS LinearSVC under the manuscript's own criterion "
                   "(every seed's paired interval excludes zero)")
    elif lo > 0:
        verdict = (f"seed-averaged interval excludes zero but only {n_excl}/{len(seeds)} "
                   f"individual seeds do -- report the split, do not claim uniformity")
    else:
        verdict = "NO SEPARATION -- the paired interval includes zero; 4.4's refusal stands"
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\nEXP_DEEP_PAIRED_DONE {round(time.time()-t0)}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
