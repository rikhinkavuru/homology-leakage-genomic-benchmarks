#!/usr/bin/env python3
"""
RF TRAINING-SEED robustness check (the two leaky datasets only).

Purpose: isolate random-forest *training* variance from the test-set sampling
variance the bootstrap CIs already capture, and confirm the RF demotion is robust
to the RF seed -- a DIFFERENT axis from step2_seed_variance.csv (which varies the
split assignment).

What it does, per leaky dataset:
  * LOADS the primary homology-aware re-split -- the seed-0 cluster->side
    assignment the headline 16.4/10.8-point deltas were computed from -- by
    reproducing it deterministically (same k=8 Jaccard>0.7 connected components +
    the same homology_split._assign(seed=0)). It does NOT create a new partition;
    this is exactly the split step_variance_ci.py used for the headline deltas.
  * Retrains ONLY the random forest (150 trees, k=6 L2-normalized counts) with
    random_state in {0,1,2,3,4}; everything else identical. LR/LinearSVC/HGB keep
    their fixed seed and are fit once (their corrected accuracies don't depend on
    the RF seed) so RF can be ranked among the four each seed.
  * Records per seed: corrected RF test accuracy and RF's rank among the four.
  * Does NOT re-bootstrap (only the across-seed mean+/-SD is needed).

Reuses the frozen pipeline verbatim: run_audit.featurize, homology_split,
run_extended_models.make_models. CPU-only, deterministic per seed, logs every seed.
Writes results/rf_seed_variance.csv (does NOT touch step2_seed_variance.csv).
"""
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.core import homology_split as H
from audit.pipeline.run_extended_models import make_models

LEAKY = ["human_enhancers_ensembl", "human_nontata_promoters"]
RF_SEEDS = [0, 1, 2, 3, 4]
M4 = ["LR", "LinearSVC", "RF", "HGB"]
R = RA.RESULTS_DIR


def components(seqs):
    """k=8 Jaccard > 0.7 connected components -- identical to step_variance_ci.py."""
    M = H.kmer_binary_matrix(seqs, 8)
    er, ec = H._edges(M, 0.7)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    _, comp = connected_components(g + g.T, directed=False)
    return comp


def fit_acc(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    return float((model.predict(X[te]) == y[te]).mean())


rows, summary = [], []
for d in LEAKY:
    df, info = S.discover_and_subsample(d, cap=10**12, seed=S.SUBSAMPLE_SEED)   # FULL scale, frozen subsample seed
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    tf = float((df["split"].to_numpy() == "test").mean())
    X = RA.featurize(seqs, 6)                                   # k=6 counts, frozen feature pipeline
    comp = components(seqs)
    tr0, te0 = H._assign(comp, y, 0, tf)                        # PRIMARY (seed-0) homology-aware split -- reproduced, NOT new

    orig_rf0 = fit_acc(make_models()["RF"], X, y, otr, ote)     # original-split RF (random_state=0) for the delta check
    other = {m: fit_acc(make_models()[m], X, y, tr0, te0)       # the 3 non-RF models on the corrected split (fixed seed, once)
             for m in ("LR", "LinearSVC", "HGB")}

    corr = []
    for s in RF_SEEDS:
        rf = make_models()["RF"]; rf.set_params(random_state=s)  # keep every RF param; vary ONLY the training seed
        a = fit_acc(rf, X, y, tr0, te0)
        corr.append(a)
        accs = dict(other); accs["RF"] = a
        rk = sorted(M4, key=lambda m: -accs[m]).index("RF") + 1   # RF rank among the 4 (1 = best)
        rows.append(dict(dataset=d, rf_seed=s, corrected_rf_acc=round(a, 4), rf_rank=rk,
                         LR_corrected=round(other["LR"], 4), LinearSVC_corrected=round(other["LinearSVC"], 4),
                         HGB_corrected=round(other["HGB"], 4), original_rf_acc_seed0=round(orig_rf0, 4)))
        print(f"[{d}] rf_seed={s}: corrected RF acc={a:.4f}  RF rank={rk}/4  "
              f"(corrected LR={other['LR']:.4f} LinearSVC={other['LinearSVC']:.4f} HGB={other['HGB']:.4f})", flush=True)

    arr = np.array(corr)
    ranks = [r["rf_rank"] for r in rows if r["dataset"] == d]
    summary.append(dict(dataset=d, corrected_rf_mean=round(arr.mean(), 4), corrected_rf_sd=round(arr.std(ddof=0), 4),
                        corrected_rf_min=round(arr.min(), 4), corrected_rf_max=round(arr.max(), 4),
                        corrected_rf_seed0=round(corr[0], 4), original_rf_seed0=round(orig_rf0, 4),
                        delta_seed0=round(orig_rf0 - corr[0], 4), rf_ranks=str(ranks),
                        rf_demoted_all_seeds=bool(all(r > 1 for r in ranks))))
    print(f"  >> {d}: corrected RF {arr.mean():.4f} +/- {arr.std(ddof=0):.4f} over 5 RF seeds "
          f"(min {arr.min():.4f}, max {arr.max():.4f}); seed0 corrected={corr[0]:.4f}, "
          f"original(seed0)={orig_rf0:.4f}, original-corrected(seed0)={orig_rf0 - corr[0]:+.4f}; ranks={ranks}", flush=True)
    if not all(r > 1 for r in ranks):
        print(f"  !!!! WARNING: RF stayed at RANK 1 for some seed on {d} -- demotion NOT seed-robust !!!!", flush=True)
    del X; gc.collect()

pd.DataFrame(rows).to_csv(f"{R}/rf_seed_variance.csv", index=False)
print("\n=== rf_seed_variance.csv (per RF seed) ===")
print(pd.DataFrame(rows).to_string(index=False), flush=True)
print("\n=== summary (mean+/-SD across 5 RF seeds; consistency vs the headline deltas) ===")
print(pd.DataFrame(summary).to_string(index=False), flush=True)
print("\nRF_SEED_DONE", flush=True)
