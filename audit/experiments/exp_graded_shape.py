#!/usr/bin/env python3
"""
A1 (Rafi v2 contrast): the SHAPE of accuracy against training-set similarity.

Rafi et al. v2 (bioRxiv 10.1101/2025.01.22.634321, May 2026) report a NON-MONOTONIC
curve: models do well on distant sequences and on near-identical ones, but rely on
memorized associations at high AND intermediate similarity, so performance dips where
homologs have functionally diverged.

results/graded_performance.csv already bins test sequences by max 8-mer Jaccard to
train and reports raw accuracy per bin, but raw accuracy per bin is NOT comparable
across bins: class prevalence swings from 0.20 to 0.9997 between the novel and
near-duplicate strata (see graded_gap_corrected.csv, constant_classifier_raw_gap).
Any apparent gradient can therefore be manufactured by prevalence alone.

This script re-runs the same binning on the two leaky datasets and the two clean
controls and emits, per (dataset, model, bin): n, n_pos, n_neg, raw accuracy,
sensitivity, specificity, BALANCED accuracy (for which a constant classifier scores
exactly 0.5 in every bin, so it is comparable across bins), and AUROC. It then
classifies the shape of each curve as monotone-increasing, monotone-decreasing, or
non-monotonic, separately under raw and balanced accuracy, restricted to bins with
n >= MIN_N and both classes present.

Run: PYTHONPATH=. ./venv/bin/python audit/experiments/exp_graded_shape.py
  -> results/graded_shape.csv          (per dataset/model/bin)
  -> results/graded_shape_summary.csv  (per dataset/model curve shape)
"""
import os, time, gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.pipeline.run_extended_models import make_models, MODEL_ORDER

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "demo_human_or_worm"]
ORDER = LEAKY + CLEAN
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
BINLAB = ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
MIN_N = 50          # same low-confidence cut as run_graded.py


def shape_of(vals):
    """Classify a sequence of >=3 values. Returns (label, n_reversals)."""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(v) < 3:
        return "undetermined (<3 usable bins)", 0
    d = np.diff(v)
    ups = int((d > 0).sum()); downs = int((d < 0).sum())
    if downs == 0:
        return "monotone increasing", 0
    if ups == 0:
        return "monotone decreasing", 0
    # non-monotonic: is the extremum interior, and which way?
    imin = int(np.argmin(v)); imax = int(np.argmax(v))
    if 0 < imin < len(v) - 1:
        kind = "non-monotonic (interior DIP)"
    elif 0 < imax < len(v) - 1:
        kind = "non-monotonic (interior PEAK)"
    else:
        kind = "non-monotonic (no interior extremum)"
    return kind, min(ups, downs)


def run(dset):
    t0 = time.time()
    cap = 10**12 if dset in LEAKY else 20000
    df, info = S.discover_and_subsample(dset, cap=cap, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    X = RA.featurize(seqs, 6)
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
    max_sim = RA.max_jaccard_to_reference(M[ote], M[otr])
    Xtr, ytr, Xte, yte = X[otr], y[otr], X[ote], y[ote]
    print(f"=== {dset}  n_test={len(ote)} ===", flush=True)

    models = make_models()
    rows, summ = [], []
    for mname in MODEL_ORDER:
        mdl = models[mname]
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        score = (mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba")
                 else mdl.decision_function(Xte))
        correct = (pred == yte)
        mrows = []
        for (lo, hi), lab in zip(BINS, BINLAB):
            mask = (max_sim >= lo) & (max_sim < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            yb, cb = yte[mask], correct[mask]
            npos = int((yb == 1).sum()); nneg = int((yb == 0).sum())
            acc = float(cb.mean())
            sens = float(cb[yb == 1].mean()) if npos else np.nan
            spec = float(cb[yb == 0].mean()) if nneg else np.nan
            bal = float((sens + spec) / 2) if (npos and nneg) else np.nan
            au = (roc_auc_score(yb, score[mask]) if len(np.unique(yb)) > 1 else np.nan)
            mrows.append(dict(
                dataset=dset, leaky=dset in LEAKY, model=mname, sim_bin=lab,
                n=n, n_pos=npos, n_neg=nneg, pos_rate=round(npos / n, 4),
                accuracy=round(acc, 4),
                sensitivity=round(sens, 4) if npos else None,
                specificity=round(spec, 4) if nneg else None,
                balanced_accuracy=round(bal, 4) if (npos and nneg) else None,
                auroc=round(au, 4) if not np.isnan(au) else None,
                usable=bool(n >= MIN_N and npos and nneg),
            ))
        rows += mrows
        use = [r for r in mrows if r["usable"]]
        s_raw, rev_raw = shape_of([r["accuracy"] for r in use])
        s_bal, rev_bal = shape_of([r["balanced_accuracy"] for r in use])
        s_auc, rev_auc = shape_of([r["auroc"] for r in use])
        summ.append(dict(
            dataset=dset, leaky=dset in LEAKY, model=mname,
            n_usable_bins=len(use),
            usable_bins=";".join(r["sim_bin"] for r in use),
            shape_raw_accuracy=s_raw, reversals_raw=rev_raw,
            shape_balanced_accuracy=s_bal, reversals_balanced=rev_bal,
            shape_auroc=s_auc, reversals_auroc=rev_auc,
        ))
        print(f"   {mname:11s} raw={s_raw:34s} bal={s_bal:34s} auroc={s_auc}", flush=True)
    del X, M; gc.collect()
    print(f"  [{dset} done {time.time()-t0:.0f}s]", flush=True)
    return rows, summ


def main():
    R, Su = [], []
    for d in ORDER:
        try:
            r, s = run(d); R += r; Su += s
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(R).to_csv(os.path.join(RA.RESULTS_DIR, "graded_shape.csv"), index=False)
    pd.DataFrame(Su).to_csv(os.path.join(RA.RESULTS_DIR, "graded_shape_summary.csv"), index=False)
    print("\nSHAPE_DONE -> results/graded_shape.csv, results/graded_shape_summary.csv", flush=True)


if __name__ == "__main__":
    main()
