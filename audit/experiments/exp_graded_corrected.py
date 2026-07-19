#!/usr/bin/env python3
"""
Is the graded memorization gap confounded by class prevalence? (Yes. This fixes it.)

THE PROBLEM, found in round 6 of the adversarial audit
------------------------------------------------------
The paper's headline mechanism statistic is the graded memorization gap
    g_M = acc_M[sim >= 0.9] - acc_M[sim < 0.5],
described in Methods as "confound-immune". It is not. The two similarity strata have
wildly different class composition, and in OPPOSITE directions on the two leaky datasets:

    human_enhancers_ensembl : high bin 99.97% positive, novel bin 19.68% positive
    human_nontata_promoters : high bin  0.15% positive, novel bin 98.17% positive

So a constant classifier that always predicts the positive class -- which memorizes
nothing whatsoever -- scores a gap of +0.803 on ensembl and -0.980 on nonTATA. Any model
with a class-prediction bias inherits a large gap of the corresponding sign for free.
This fully explains the two negative gaps the roster prints (kNN-15 at -0.684,
GaussianNB at -0.234) without them indicating "negative memorization".

THE FIX
-------
Recompute the gap from BALANCED accuracy -- the unweighted mean of per-class recall --
within each stratum. A constant classifier scores 0.5 balanced accuracy in every stratum
regardless of prevalence, so its corrected gap is exactly 0. The corrected statistic
therefore measures what the paper wants it to measure: the extra ability to classify
near-duplicates over novel sequences, net of any class-prediction bias.

We report both, plus the trivial-classifier bound for each dataset, so a reader can see
the size of the confound the raw gap carries.

Run:  python -m audit.experiments.exp_graded_corrected
Out:  results/graded_gap_corrected.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
SEED, THR, SIM_K, FEAT_K = 0, 0.7, 8, 6
LEAKY = ["human_enhancers_ensembl", "human_nontata_promoters"]
CLEAN = ["human_enhancers_cohn"]


def balanced_acc(y_true, y_pred):
    """Unweighted mean of per-class recall. A constant predictor scores 0.5 on any
    binary problem regardless of prevalence -- which is exactly the property the raw
    accuracy gap lacks."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() == 0:
            continue
        recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def main():
    from audit.experiments.exp_roster import make_roster
    os.makedirs(R, exist_ok=True)
    rows = []
    for d in LEAKY + CLEAN:
        t0 = time.time()
        full = d in LEAKY
        seqs, y, tr, te, tf = E.load(d, full=full, cap=20000)
        y = np.asarray(y)
        X = E.featurize(seqs, FEAT_K)
        sim = E.cached_max_sim(d, seqs, tr, te, k=SIM_K, mode="jaccard")
        hi, lo = sim >= 0.9, sim < 0.5
        mid = (sim >= 0.5) & (sim < 0.9)
        yt = y[te]
        pos = yt.max()
        p_hi = float((yt[hi] == pos).mean()) if hi.any() else np.nan
        p_lo = float((yt[lo] == pos).mean()) if lo.any() else np.nan
        # what a memorization-free constant classifier scores under each statistic
        const_raw = p_hi - p_lo
        print(f"[{d}] high n={int(hi.sum())} pos={p_hi:.4f} | novel n={int(lo.sum())} "
              f"pos={p_lo:.4f} | mid band {float(mid.mean()):.4f} of test | "
              f"constant-classifier RAW gap = {const_raw:+.4f}", flush=True)

        roster = make_roster()
        for m in roster:
            mdl = make_roster()[m]
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            raw = (float((pred[hi] == yt[hi]).mean()) - float((pred[lo] == yt[lo]).mean())
                   if hi.any() and lo.any() else np.nan)
            bal = (balanced_acc(yt[hi], pred[hi]) - balanced_acc(yt[lo], pred[lo])
                   if hi.any() and lo.any() else np.nan)
            rows.append(dict(
                dataset=d, model=m, leaky=d in LEAKY,
                n_high=int(hi.sum()), n_novel=int(lo.sum()),
                mid_band_frac=round(float(mid.mean()), 4),
                pos_rate_high=round(p_hi, 4), pos_rate_novel=round(p_lo, 4),
                constant_classifier_raw_gap=round(const_raw, 4),
                constant_classifier_balanced_gap=0.0,
                graded_gap_raw=round(raw, 4),
                graded_gap_balanced=round(bal, 4),
                confound_share=(round(abs(raw - bal) / abs(raw), 3)
                                if raw and abs(raw) > 1e-9 else None)))
            print(f"   {m:11s} raw={raw:+.4f}  balanced={bal:+.4f}", flush=True)
        print(f"  [{d} done {time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    tmp = f"{R}/graded_gap_corrected.csv.tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, f"{R}/graded_gap_corrected.csv")

    print("\n== raw vs prevalence-corrected graded gap ==")
    print(df[["dataset", "model", "graded_gap_raw", "graded_gap_balanced",
              "constant_classifier_raw_gap"]].to_string(index=False))
    print("\n== does the mechanism survive the correction? ==")
    for d, g in df[df.leaky].groupby("dataset"):
        gg = g.set_index("model")
        mem = ["kNN-1", "kNN-15", "RF", "ExtraTrees"]
        non = ["LR", "LinearSVC", "HGB", "GaussianNB"]
        print(f"  {d}")
        print(f"    memorizers  balanced gap mean = "
              f"{gg.loc[mem,'graded_gap_balanced'].mean():+.4f}")
        print(f"    non-memoriz balanced gap mean = "
              f"{gg.loc[non,'graded_gap_balanced'].mean():+.4f}")
        print(f"    top by balanced gap: {gg.graded_gap_balanced.idxmax()} "
              f"({gg.graded_gap_balanced.max():+.4f})")
    print("\nEXP_GRADED_CORRECTED_DONE")


if __name__ == "__main__":
    main()
