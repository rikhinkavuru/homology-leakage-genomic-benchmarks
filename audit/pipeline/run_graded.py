#!/usr/bin/env python3
"""
PART 3: homology-graded performance (the memorization-mechanism experiment).

On the ORIGINAL (leaky) split, bin each TEST sequence by its maximum k-mer Jaccard
similarity to the TRAINING set, then report per-model accuracy + AUROC per bin for all
four models (LR, LinearSVC, RF, HGB) at k=6. Hypothesis: the high-capacity memorizer
(RF) is near-perfect on near-duplicate test sequences ([0.9,1.0]) and drops on
dissimilar ones ([0,0.5)), while linear models (LR/LinearSVC) stay flatter -- direct
evidence that RF's apparent edge is memorization of near-duplicates. Clean datasets are
included as controls (their high-similarity bins are nearly empty).

Reuses RA.featurize / RA.kmer_binary_matrix / RA.max_jaccard_to_reference and the exact
extended model set (run_extended_models.make_models). Scale matches frozen: full for
leaky, 20k subsample for clean. Same similarity values as the frozen leakage analysis.
Run:  python run_graded.py  -> results/graded_performance.csv
"""
import os, time, gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.pipeline.run_extended_models import make_models, MODEL_ORDER

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "demo_human_or_worm"]      # clean controls
ORDER = CLEAN + LEAKY
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]   # last bin includes 1.0
BINLAB = ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
MIN_N = 50                                                   # below this: low-confidence


def run(dset):
    t0 = time.time()
    cap = 10**12 if dset in LEAKY else 20000
    df, info = S.discover_and_subsample(dset, cap=cap, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    X = RA.featurize(seqs, 6)
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
    max_sim = RA.max_jaccard_to_reference(M[ote], M[otr])     # per TEST seq, vs train
    Xtr, ytr, Xte, yte = X[otr], y[otr], X[ote], y[ote]
    print(f"=== {dset}  n_test={len(ote)} ({'full' if dset in LEAKY else 'subsample'}) "
          f"bins n={[int(((max_sim>=lo)&(max_sim<hi)).sum()) for lo,hi in BINS]} ===", flush=True)

    models = make_models()
    rows = []
    for mname in MODEL_ORDER:
        mdl = models[mname]
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        score = (mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba")
                 else mdl.decision_function(Xte))
        correct = (pred == yte)
        for (lo, hi), lab in zip(BINS, BINLAB):
            mask = (max_sim >= lo) & (max_sim < hi)
            n = int(mask.sum())
            acc = float(correct[mask].mean()) if n else np.nan
            au = (roc_auc_score(yte[mask], score[mask])
                  if n and len(np.unique(yte[mask])) > 1 else np.nan)
            rows.append(dict(dataset=dset, leaky=dset in LEAKY, split="original",
                             model=mname, sim_bin=lab, n=n,
                             accuracy=round(acc, 4) if n else None,
                             auroc=round(au, 4) if (n and not np.isnan(au)) else None,
                             low_confidence=bool(0 < n < MIN_N)))
        a_hi = next(r for r in rows if r["model"] == mname and r["sim_bin"] == "[0.9,1.0]")
        a_lo = next(r for r in rows if r["model"] == mname and r["sim_bin"] == "[0,0.5)")
        gap = (a_hi["accuracy"] - a_lo["accuracy"]) if (a_hi["accuracy"] is not None and a_lo["accuracy"] is not None) else None
        print(f"   {mname:9s} acc by bin: "
              + " ".join(f"{r['accuracy'] if r['accuracy'] is not None else 'NA':>6}"
                         f"(n={r['n']})" for r in rows[-4:])
              + f"  | gap(>=0.9 - <0.5)={gap if gap is not None else 'NA'}", flush=True)
    del X, M; gc.collect()
    print(f"  [{dset} done {time.time()-t0:.0f}s]", flush=True)
    return rows


def main():
    allrows = []
    for d in ORDER:
        try:
            allrows += run(d)
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(RA.RESULTS_DIR, "graded_performance.csv"), index=False)
    print("\nGRADED_DONE -> results/graded_performance.csv", flush=True)


if __name__ == "__main__":
    main()
