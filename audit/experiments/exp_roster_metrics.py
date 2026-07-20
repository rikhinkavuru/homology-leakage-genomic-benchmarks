#!/usr/bin/env python3
"""
Does the nine-model roster inversion on human_nontata_promoters survive a
threshold-free metric?

The four-model nontata demotion of the random forest dies under AUROC and F1 --
that is one of the five signals the manuscript cites for calling the dataset a
"cautionary partial case". The nine-model roster inverts under ACCURACY with
both swap margins excluding zero (results/roster_rankings.csv), but no AUROC or
F1 was ever computed for the roster, so the threshold-free question was never
actually asked of it. This module asks it.

Splits, features, threshold, seed and roster are taken from exp_roster.py
verbatim so the accuracy column here must reproduce roster_rankings.csv. That
reproduction is the correctness check on this file.

Paired CIs: for accuracy the per-example correctness difference is bootstrapped
over test-internal clusters exactly as exp_roster does. AUROC and F1 do not
decompose per example, so we resample clusters and RECOMPUTE both models'
metric on the same resample, taking the difference within each replicate --
paired at the replicate level, which is the closest available analogue.

Run:  python -m audit.experiments.exp_roster_metrics
Out:  results/roster_metrics_nontata.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_roster import (make_roster, MEMORIZERS, LINEARS, SEED,
                                          THR, SIM_K, FEAT_K, LEAKY)

R = os.path.join(HERE, "results")
BOOT = 2000
BSEED = 20240524


def _scores(model, X, y, tr, te):
    """Fit once; return correctness vector, decision score, acc/auroc/f1."""
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    yte = y[te]
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X[te])[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X[te])
    else:
        s = pred.astype(float)
    return (dict(acc=float(accuracy_score(yte, pred)),
                 auroc=float(roc_auc_score(yte, s)),
                 f1=float(f1_score(yte, pred, zero_division=0))),
            (pred == yte).astype(float), np.asarray(s, float),
            np.asarray(yte), np.asarray(pred))


def _cluster_draws(blocks, boot=BOOT, seed=BSEED):
    """Pre-draw the cluster resamples once so every model pair shares them."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(blocks)
    members = [np.flatnonzero(blocks == u) for u in uniq]
    G = len(uniq)
    for _ in range(boot):
        pick = rng.randint(0, G, G)
        yield np.concatenate([members[p] for p in pick])


def _paired_metric_ci(sA, sB, predA, predB, yte, blocks, metric, lo=2.5, hi=97.5):
    """CI on metric(A) - metric(B), resampling clusters and recomputing both."""
    out = []
    for idx in _cluster_draws(blocks):
        yy = yte[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "auroc":
            out.append(roc_auc_score(yy, sA[idx]) - roc_auc_score(yy, sB[idx]))
        else:
            out.append(f1_score(yy, predA[idx], zero_division=0)
                       - f1_score(yy, predB[idx], zero_division=0))
    out = np.asarray(out, float)
    return float(np.percentile(out, lo)), float(np.percentile(out, hi)), len(out)


def _paired_acc_ci(diff, blocks, lo=2.5, hi=97.5):
    out = np.array([diff[idx].mean() for idx in _cluster_draws(blocks)])
    return float(np.percentile(out, lo)), float(np.percentile(out, hi)), len(out)


def main(d="human_nontata_promoters"):
    t0 = time.time()
    # Mirror exp_roster.run_dataset's load exactly: leaky datasets at full scale,
    # the clean control on its 20,000-sequence subsample, so the control arm here
    # is the same control the accuracy result was computed on.
    seqs, y, tr, te, tf = E.load(d, full=(d in LEAKY), cap=20000)
    y = np.asarray(y)
    X = E.featurize(seqs, FEAT_K)
    print(f"[{d}] n={len(seqs)} train={len(tr)} test={len(te)}", flush=True)

    comp = E.cached_clusters(d, seqs, THR, SIM_K)
    if len(comp) != len(seqs):
        comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, tf)

    from audit.experiments.cluster_bootstrap import test_internal_clusters
    blk_o, _ = test_internal_clusters([seqs[i] for i in te], THR)
    blk_c, _ = test_internal_clusters([seqs[i] for i in cte], THR)

    names = list(make_roster())
    M, C, S, Y, P = {}, {}, {}, {}, {}
    for m in names:
        t1 = time.time()
        for arm, (a, b) in (("orig", (tr, te)), ("corr", (ctr, cte))):
            mt, c, s, yy, pr = _scores(make_roster()[m], X, y, a, b)
            M[(m, arm)], C[(m, arm)], S[(m, arm)] = mt, c, s
            Y[arm], P[(m, arm)] = yy, pr
        print(f"   {m:11s} "
              f"orig acc={M[(m,'orig')]['acc']:.4f} auroc={M[(m,'orig')]['auroc']:.4f} f1={M[(m,'orig')]['f1']:.4f} | "
              f"corr acc={M[(m,'corr')]['acc']:.4f} auroc={M[(m,'corr')]['auroc']:.4f} f1={M[(m,'corr')]['f1']:.4f} "
              f"({time.time()-t1:.0f}s)", flush=True)

    rows = []
    for m in names:
        r = dict(dataset=d, model=m,
                 family=("memorizer" if m in MEMORIZERS else
                         "linear" if m in LINEARS else "other"))
        for arm in ("orig", "corr"):
            for k in ("acc", "auroc", "f1"):
                r[f"{k}_{arm}"] = round(M[(m, arm)][k], 4)
        rows.append(r)
    df = pd.DataFrame(rows)
    for k in ("acc", "auroc", "f1"):
        for arm in ("orig", "corr"):
            df[f"rank_{k}_{arm}"] = df[f"{k}_{arm}"].rank(ascending=False,
                                                          method="min").astype(int)
    print("\n== per-metric leaders ==")
    swap = {}
    for k in ("acc", "auroc", "f1"):
        to = df.loc[df[f"{k}_orig"].idxmax(), "model"]
        tc = df.loc[df[f"{k}_corr"].idxmax(), "model"]
        swap[k] = (to, tc)
        print(f"  {k:6s} as-shipped top={to:11s} corrected top={tc:11s} "
              f"{'SWAP' if to != tc else 'no swap'}")

    # swap-margin intervals, in each direction, for each metric's own leader pair
    out = []
    for k in ("acc", "auroc", "f1"):
        to, tc = swap[k]
        if to == tc:
            out.append(dict(dataset=d, metric=k, top_orig=to, top_corr=tc,
                            swaps=False, margin_orig_ci=None, margin_corr_ci=None,
                            inverts_by_criterion=False))
            continue
        res = {}
        for arm, blk in (("orig", blk_o), ("corr", blk_c)):
            A, B = (to, tc) if arm == "orig" else (tc, to)
            if k == "acc":
                lo, hi, nb = _paired_acc_ci(C[(A, arm)] - C[(B, arm)], blk)
            else:
                lo, hi, nb = _paired_metric_ci(S[(A, arm)], S[(B, arm)],
                                               P[(A, arm)], P[(B, arm)],
                                               Y[arm], blk, k)
            res[arm] = (lo, hi, nb)
        inv = bool(res["orig"][0] > 0 and res["corr"][0] > 0)
        out.append(dict(dataset=d, metric=k, top_orig=to, top_corr=tc, swaps=True,
                        margin_orig_ci=f"[{res['orig'][0]:.4f}, {res['orig'][1]:.4f}]",
                        margin_corr_ci=f"[{res['corr'][0]:.4f}, {res['corr'][1]:.4f}]",
                        inverts_by_criterion=inv))
        print(f"  {k:6s} {to}>{tc} as-shipped CI [{res['orig'][0]:.4f}, {res['orig'][1]:.4f}]; "
              f"{tc}>{to} corrected CI [{res['corr'][0]:.4f}, {res['corr'][1]:.4f}] "
              f"-> inverts={inv}")

    sw = pd.DataFrame(out)
    tag = "nontata" if d == "human_nontata_promoters" else d
    df.to_csv(f"{R}/roster_metrics_{tag}.csv", index=False)
    sw.to_csv(f"{R}/roster_metrics_{tag}_swaps.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\n" + sw.to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}s\nEXP_ROSTER_METRICS_DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "human_nontata_promoters")
