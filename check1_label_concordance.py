#!/usr/bin/env python3
"""
CHECK 1: label-concordance of near-duplicates. On the ORIGINAL split, for each test
sequence with a training near-duplicate at Jaccard >= threshold, find its NEAREST
training neighbour and record whether that neighbour shares the test label. If the
same-label fraction is ~1.0, near-duplicates carry their labels across the split, so a
memorizing model gets them right for free -- the precise mechanism of the inflation
(and why RF hits 1.000 on the >=0.9 bin). Reuses RA.kmer_binary_matrix and the exact
frozen similarity (8-mer Jaccard); full scale for leaky, 20k subsample for clean.
Run:  python check1_label_concordance.py -> results/label_concordance.csv
"""
import os, time, gc
import numpy as np
import pandas as pd
import run_audit as RA
import run_suite as S

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "demo_human_or_worm"]
ORDER = CLEAN + LEAKY
THRS = [0.7, 0.9]
MIN_N = 50


def max_jaccard_argmax(Mq, Mr, batch=400):
    """Per query row: (max Jaccard, index of the argmax reference row)."""
    sq = np.asarray(Mq.sum(1)).ravel(); sr = np.asarray(Mr.sum(1)).ravel()
    RT = Mr.T.tocsr()
    mx = np.zeros(Mq.shape[0], np.float32)
    am = np.zeros(Mq.shape[0], np.int64)
    for s in range(0, Mq.shape[0], batch):
        e = min(s + batch, Mq.shape[0])
        inter = (Mq[s:e] @ RT).toarray()
        union = sq[s:e, None] + sr[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        am[s:e] = jac.argmax(1)
        mx[s:e] = jac[np.arange(e - s), am[s:e]]
    return mx, am


def main():
    rows = []
    for d in ORDER:
        t0 = time.time()
        cap = 10**12 if d in LEAKY else 20000
        df, info = S.discover_and_subsample(d, cap=cap, seed=S.SUBSAMPLE_SEED)
        y = df["label"].to_numpy(); seqs = df["seq"].tolist()
        otr = np.where(df["split"].to_numpy() == "train")[0]
        ote = np.where(df["split"].to_numpy() == "test")[0]
        M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
        mx, am = max_jaccard_argmax(M[ote], M[otr])
        yte, ytr = y[ote], y[otr]
        neigh = ytr[am]                                   # nearest-train-neighbour label
        for thr in THRS:
            mask = mx >= thr
            n = int(mask.sum())
            same = float((yte[mask] == neigh[mask]).mean()) if n else np.nan
            rows.append(dict(dataset=d, leaky=d in LEAKY, threshold=thr, n_pairs=n,
                             frac_same_label=round(same, 4) if n else None,
                             low_confidence=bool(0 < n < MIN_N)))
            print(f"  {d} (>= {thr}): n={n}  same_label_frac="
                  f"{same:.4f}" if n else f"  {d} (>= {thr}): n=0", flush=True)
        del M; gc.collect()
        print(f"   [{d} {time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RA.RESULTS_DIR, "label_concordance.csv"), index=False)
    print("\nCONCORDANCE_DONE -> results/label_concordance.csv", flush=True)


if __name__ == "__main__":
    main()
