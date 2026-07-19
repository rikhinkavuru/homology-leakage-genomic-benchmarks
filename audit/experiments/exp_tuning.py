#!/usr/bin/env python3
"""
Does tuning rescue you? What cross-validation selects on a leaky benchmark.

The sharpest objection to this paper's ranking result is that it concerns UNTUNED
defaults: the random forest holds rank 1 only near min_samples_leaf=1 and has fallen to
rank 4 by min_samples_leaf>=20 (exp_regpath.csv), so a reader may say the finding is about
library defaults rather than about the benchmark. The Discussion answers this with
argument. This module answers it with an experiment.

THE ARGUMENT THE EXPERIMENT TESTS
---------------------------------
A practitioner does not know the benchmark is leaky. They tune the way everyone tunes:
k-fold cross-validation on the TRAINING set. But the training set carries the same
near-duplicate structure as the split does -- copies of the same locus land in different
CV folds -- so each validation fold is itself contaminated by its own training folds. If
that contaminated CV rewards memorization, then tuning does not rescue the practitioner;
it actively selects the memorizer, and the "just tune it" objection collapses.

WHAT WE COMPARE
---------------
  naive CV    : standard stratified 5-fold on the as-shipped training set
  grouped CV  : 5-fold with whole near-duplicate CLUSTERS held out together
                (GroupKFold over the same 8-mer Jaccard>0.7 components used for the
                corrected split) -- i.e. the honest version of the same procedure
Both search the same grid over min_samples_leaf. We then report, for each selection,
the model's accuracy on the as-shipped test set and on the corrected split.

PRE-COMMITTED PREDICTION
------------------------
  Naive CV selects a small min_samples_leaf (a memorizing forest) on the leaky datasets;
  grouped CV selects a larger one. If naive CV instead selects a heavily regularized
  forest, the "just tune it" objection stands and we report that.

Run:  python -m audit.experiments.exp_tuning
Out:  results/tuning_selection.csv
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
SEED = 0
THR, SIM_K, FEAT_K = 0.7, 8, 6
GRID = [1, 5, 20, 50, 100, 200, 500]
DATASETS = ["human_nontata_promoters", "human_enhancers_ensembl", "human_enhancers_cohn"]
LEAKY = {"human_nontata_promoters", "human_enhancers_ensembl"}


def cv_select(X, y, idx, groups, grid, n_splits=5, grouped=False, n_jobs=4):
    """Return (best_param, per-param mean CV accuracy). `grouped` holds whole
    near-duplicate clusters out together; otherwise plain stratified k-fold."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    if grouped:
        splitter = GroupKFold(n_splits=n_splits).split(idx, y[idx], groups[idx])
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED).split(idx, y[idx])
    folds = list(splitter)
    scores = {}
    for p in grid:
        accs = []
        for tr_i, va_i in folds:
            tr, va = idx[tr_i], idx[va_i]
            m = RandomForestClassifier(n_estimators=150, min_samples_leaf=p,
                                       random_state=SEED, n_jobs=n_jobs)
            m.fit(X[tr], y[tr])
            accs.append(float((m.predict(X[va]) == y[va]).mean()))
        scores[p] = float(np.mean(accs))
    best = max(scores, key=scores.get)
    return best, scores


def run(d, cap=None):
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    full = d in LEAKY and cap is None
    seqs, y, tr, te, tf = E.load(d, full=full, cap=cap or 20000)
    y = np.asarray(y)
    X = E.featurize(seqs, FEAT_K)
    comp = E.cached_clusters(d, seqs, THR, SIM_K)
    if len(comp) != len(seqs):
        comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, tf)
    print(f"[{d}] n={len(seqs)} train={len(tr)} test={len(te)} "
          f"train-side clusters={len(np.unique(comp[tr]))}", flush=True)

    rows = []
    for grouped in (False, True):
        tag = "grouped_CV" if grouped else "naive_CV"
        best, scores = cv_select(X, y, tr, comp, GRID, grouped=grouped)
        # what that selection then scores, on both splits
        m = RandomForestClassifier(n_estimators=150, min_samples_leaf=best,
                                   random_state=SEED, n_jobs=4)
        acc_o = float((m.fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
        acc_c = float((m.fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).mean())
        rows.append(dict(dataset=d, leaky=d in LEAKY, selector=tag,
                         selected_min_samples_leaf=best,
                         cv_scores={k: round(v, 4) for k, v in scores.items()},
                         cv_best_score=round(scores[best], 4),
                         cv_score_at_default=round(scores[1], 4),
                         acc_as_shipped=round(acc_o, 4), acc_corrected=round(acc_c, 4),
                         drop=round(acc_o - acc_c, 4)))
        print(f"  {tag:11s} selects min_samples_leaf={best:<4d} "
              f"cv={scores[best]:.4f}  -> as-shipped {acc_o:.4f} / corrected {acc_c:.4f} "
              f"(drop {acc_o-acc_c:+.4f})  [{time.time()-t0:.0f}s]", flush=True)
        print(f"              cv curve: " +
              " ".join(f"{k}:{v:.4f}" for k, v in scores.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--cap", type=int)
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    rows = []
    for d in a.datasets:
        try:
            rows += run(d, cap=a.cap)
        except Exception as ex:                                   # noqa: BLE001
            print(f"  [{d}] FAILED: {type(ex).__name__}: {ex}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing completed"); return
    tmp = f"{R}/tuning_selection.csv.tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, f"{R}/tuning_selection.csv")
    print("\n== what cross-validation selects ==")
    print(df[["dataset", "leaky", "selector", "selected_min_samples_leaf",
              "acc_as_shipped", "acc_corrected", "drop"]].to_string(index=False))
    print("\nPREDICTION: naive CV selects a memorizing forest on the leaky datasets; "
          "grouped CV selects a more regularized one.")
    for d, g in df[df.leaky].groupby("dataset"):
        nv = g[g.selector == "naive_CV"].selected_min_samples_leaf.iloc[0]
        gr = g[g.selector == "grouped_CV"].selected_min_samples_leaf.iloc[0]
        print(f"  {d}: naive={nv}  grouped={gr}  -> "
              f"{'HOLDS' if nv < gr else 'FAILS' if nv > gr else 'TIE'}")
    print("\nEXP_TUNING_DONE")


if __name__ == "__main__":
    main()
