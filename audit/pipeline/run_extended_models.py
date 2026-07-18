#!/usr/bin/env python3
"""
PART A: extend the capacity axis with more classical models, on the SAME frozen
splits (regenerated deterministically -- identical seeds/loaders/edge logic, so the
splits are bit-identical to the frozen runs; we are NOT recomputing leakage or
changing the re-split logic, only adding models).

Models (CPU-only, ordered by a capacity proxy, justified in PAPER_NUMBERS):
  LR         LogisticRegression (linear, L2)          -- lowest capacity
  LinearSVC  linear SVM (hinge)                        -- ~ equal to LR (linear)
  RF         RandomForest(150)  (bagged deep trees)    -- nonlinear, bagging-regularized
  HGB        HistGradientBoosting (boosted trees)      -- highest effective capacity
(MLP/CNN skipped: would need long CPU time on the full 154k x 4096 enhancers set and
 no GPU is allowed; HGB anchors the high-capacity end. Noted in the writeup.)

Scale matches the frozen numbers: FULL for the 2 leaky datasets, 20k stratified
subsample (seed 0) for the 5 clean datasets.

Reuses RA.featurize / RA.make_models (LR,RF identical) + RA.kmer_binary_matrix /
S.edges_per_threshold / S.components_from_edges / RA.assign_clusters / S.random_split.
Run:  python run_extended_models.py  -> results/extended_models_long.csv
"""
import os, time, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "human_ocr_ensembl", "demo_human_or_worm",
         "demo_coding_vs_intergenomic_seqs", "drosophila_enhancers_stark"]
ORDER = CLEAN + LEAKY                      # clean first (fast feedback), leaky last
SEEDS = S.SPLIT_SEEDS                      # [0,1,2]
MAIN_THR = 0.7
MODEL_ORDER = ["LR", "LinearSVC", "RF", "HGB"]   # capacity low -> high


def make_models():
    base = RA.make_models()                # LR, RF -- EXACT same objects as frozen runs
    svc = make_pipeline(Normalizer(norm="l2"),
                        LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=RA.MODEL_SEED))
    hgb = HistGradientBoostingClassifier(random_state=RA.MODEL_SEED)
    return {"LR": base["LR"], "LinearSVC": svc, "RF": base["RF"], "HGB": hgb}


def eval_model(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X[te])[:, 1]
    else:
        score = model.decision_function(X[te])      # LinearSVC
    yt = y[te]
    return dict(accuracy=accuracy_score(yt, pred),
                auroc=roc_auc_score(yt, score) if len(np.unique(yt)) > 1 else float("nan"),
                f1=f1_score(yt, pred))


def run(dset):
    t0 = time.time()
    cap = 10**12 if dset in LEAKY else 20000
    df, info = S.discover_and_subsample(dset, cap=cap, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist(); n = len(df)
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    target = len(ote) / n
    print(f"=== {dset}  n={n} ({'full' if dset in LEAKY else 'subsample'})  test_frac={target:.3f} ===", flush=True)
    X = {k: RA.featurize(seqs, k) for k in (4, 6)}
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
    er, ec = S.edges_per_threshold(M, [MAIN_THR])[MAIN_THR]
    nc, lab = S.components_from_edges(n, er, ec)

    # build the split set: original (seed -1), random x3, homology@0.7 x3
    splits = [("original", -1, otr, ote)]
    for s in SEEDS:
        tr, te = S.random_split(y, target, s); splits.append(("random", s, tr, te))
    for s in SEEDS:
        tr, te = RA.assign_clusters(lab, y, s, target); splits.append(("homology@0.7", s, tr, te))

    rows = []
    for k in (4, 6):
        for (split_type, seed, tr, te) in splits:
            models = make_models()
            for mname in MODEL_ORDER:
                r = eval_model(models[mname], X[k], y, tr, te)
                rows.append(dict(dataset=dset, scale="full" if dset in LEAKY else "subsample",
                                 n_used=n, split_type=split_type, seed=seed, k=k, model=mname, **r))
        print(f"  k={k} done ({time.time()-t0:.0f}s)", flush=True)
    del X, M; gc.collect()
    print(f"  [{dset} done in {time.time()-t0:.0f}s]", flush=True)
    return rows


def main():
    all_rows = []
    for d in ORDER:
        try:
            all_rows += run(d)
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(RA.RESULTS_DIR, "extended_models_long.csv"), index=False)
    print("\nEXTENDED_DONE -> results/extended_models_long.csv", flush=True)


if __name__ == "__main__":
    main()
