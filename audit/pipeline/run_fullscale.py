#!/usr/bin/env python3
"""
FULL-SCALE homology re-split for the datasets that the full-scale leakage scan
flagged as leaky (so the 20k subsample masked the effect). Measures the HONEST
accuracy drop at full dataset size, for the MAIN comparison only:
  original  vs  random re-split (x3 seeds)  vs  homology-aware @0.7 (x3 seeds),
all four model configs (LR/RF x k=4/6). Threshold sweep + drop-largest are
omitted here purely to bound runtime; the leakage in these datasets is
concentrated at >0.9 Jaccard so clustering at 0.7 captures essentially all of it.

Reuses run_audit.py + run_suite.py primitives unchanged.
Run:   python run_fullscale.py  -> results/fullscale_results.csv
"""
import os, time, gc
import numpy as np
import pandas as pd
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S

DSETS = [
    "human_nontata_promoters",          # known leaky; full-scale negative control + consistency
    "human_enhancers_ensembl",          # 0.384 @0.7 full scale -- masked by subsampling
    "demo_coding_vs_intergenomic_seqs", # 0.078 @0.7 full scale -- mild
]
SEEDS = S.SPLIT_SEEDS
THR = 0.7
CONFIGS = S.CONFIGS


def run_full(dset):
    t0 = time.time()
    df, info = S.discover_and_subsample(dset, cap=10**12, seed=0)  # cap huge => FULL
    y = df["label"].to_numpy(); seqs = df["seq"].tolist(); n = len(df)
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    target = len(ote) / n
    print(f"\n=== {dset} FULL n={n} (train {len(otr)}/test {len(ote)}, test_frac={target:.3f}) ===", flush=True)

    X = {k: RA.featurize(seqs, k) for k in (4, 6)}
    print(f"  featurized ({time.time()-t0:.0f}s)", flush=True)
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
    er, ec = S.edges_per_threshold(M, [THR])[THR]
    nc, lab = S.components_from_edges(n, er, ec)
    sizes = np.bincount(lab)
    print(f"  homology@{THR}: edges={len(er)} comps={nc} redund={(n-nc)/n:.3f} "
          f"max_cluster={int(sizes.max())} ({time.time()-t0:.0f}s)", flush=True)

    rows = []

    def rec(split, seed, tr, te):
        cfg = S.eval_all_configs(X, y, tr, te, RA.eval_split)
        for (k, m), r in cfg.items():
            rows.append(dict(dataset=dset, n_used=n, split_type=split, seed=seed,
                             k=k, model=m, **r))
        return cfg

    cfg = rec("original", -1, otr, ote)
    print("  ORIGINAL  " + "  ".join(f"{k}{m}={cfg[(k,m)]['accuracy']:.3f}" for (k, m) in CONFIGS)
          + f"  ({time.time()-t0:.0f}s)", flush=True)

    for s in SEEDS:
        tr, te = S.random_split(y, target, s)
        rec("random", s, tr, te)
    print(f"  random x{len(SEEDS)} done ({time.time()-t0:.0f}s)", flush=True)

    for si, s in enumerate(SEEDS):
        tr, te = RA.assign_clusters(lab, y, s, target)
        if si == 0:
            span = len(np.intersect1d(np.unique(lab[tr]), np.unique(lab[te])))
            resid = RA.max_jaccard_to_reference(M[te], M[tr])
            print(f"  VERIFY span={span} resid>{THR}={float((resid>THR).mean()):.4f} "
                  f"testfrac={len(te)/n:.3f} pos_tr/te={y[tr].mean():.3f}/{y[te].mean():.3f}", flush=True)
        rec(f"homology@{THR}", s, tr, te)
    print(f"  homology x{len(SEEDS)} done ({time.time()-t0:.0f}s)", flush=True)

    del X, M
    gc.collect()
    print(f"  [{dset} full-scale done in {time.time()-t0:.0f}s]", flush=True)
    return rows


def main():
    all_rows = []
    for d in DSETS:
        try:
            all_rows += run_full(d)
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RA.RESULTS_DIR, "fullscale_long.csv"), index=False)

    # summarize: per dataset x config, original vs random vs homology + deltas
    out = []
    for dset in df["dataset"].unique():
        for (k, m) in CONFIGS:
            def acc(split):
                s = df[(df.dataset == dset) & (df.split_type == split) & (df.k == k) & (df.model == m)]["accuracy"]
                return (float(s.mean()), float(s.std(ddof=0)))
            def auroc(split):
                s = df[(df.dataset == dset) & (df.split_type == split) & (df.k == k) & (df.model == m)]["auroc"]
                return float(s.mean())
            o, _ = acc("original"); rm, _ = acc("random"); hm, hs = acc(f"homology@{THR}")
            out.append(dict(dataset=dset, model=m, k=k,
                            original_acc=round(o, 4),
                            random_acc_mean=round(rm, 4),
                            homology_acc_mean=round(hm, 4), homology_acc_std=round(hs, 4),
                            delta_acc_homology=round(o - hm, 4),
                            delta_acc_random=round(o - rm, 4),
                            original_auroc=round(auroc("original"), 4),
                            homology_auroc_mean=round(auroc(f"homology@{THR}"), 4)))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(RA.RESULTS_DIR, "fullscale_summary.csv"), index=False)
    print("\n===== FULL-SCALE SUMMARY =====", flush=True)
    print(summ.to_string(), flush=True)
    print("\nFULLSCALE_DONE -> results/fullscale_summary.csv", flush=True)


if __name__ == "__main__":
    main()
