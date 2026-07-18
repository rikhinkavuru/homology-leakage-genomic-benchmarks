#!/usr/bin/env python3
"""
FULL-SCALE clustering-robustness for the two LEAKY datasets
(human_nontata_promoters, human_enhancers_ensembl), so both are treated
symmetrically. At full dataset scale, for each:
  * single-linkage threshold sweep {0.5, 0.7, 0.9}: corrected accuracy (all 4
    configs incl. RF k6 and LR k6), 3 seeds, mean +/- std, drop vs original;
  * drop-largest-component @0.7: remove the largest homology component, re-split
    the rest, retrain; report whether the drop + capacity-scaling persist;
  * invariant verification: residual test->train leakage > t == 0, 0 clusters
    span the split, class balance + train/test ratio held.

Reuses run_audit.py + run_suite.py primitives unchanged (identical featurization,
models, exact-Jaccard similarity, whole-cluster assignment).
Run:   python run_robustness_full.py
"""
import os, time, gc
import numpy as np
import pandas as pd
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S

DSETS = ["human_nontata_promoters", "human_enhancers_ensembl"]
THRESHOLDS = [0.5, 0.7, 0.9]
SEEDS = S.SPLIT_SEEDS
CONFIGS = S.CONFIGS


def run(dset):
    t0 = time.time()
    df, info = S.discover_and_subsample(dset, cap=10**12, seed=0)   # full scale
    y = df["label"].to_numpy(); seqs = df["seq"].tolist(); n = len(df)
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    target = len(ote) / n
    print(f"\n=== {dset} FULL n={n} (train {len(otr)}/test {len(ote)}, test_frac={target:.3f}) ===", flush=True)
    X = {k: RA.featurize(seqs, k) for k in (4, 6)}
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)
    edges = S.edges_per_threshold(M, THRESHOLDS)
    print(f"  featurized + edges done ({time.time()-t0:.0f}s)", flush=True)

    rows, clusters = [], []

    def rec(split, seed, tr, te):
        cfg = S.eval_all_configs(X, y, tr, te, RA.eval_split)
        for (k, m), r in cfg.items():
            rows.append(dict(dataset=dset, scale="full", split_type=split, seed=seed,
                             k=k, model=m, **r))

    rec("original", -1, otr, ote)   # reference for the drops
    print(f"  original done ({time.time()-t0:.0f}s)", flush=True)

    comp07 = None
    for t in THRESHOLDS:
        nc, lab = S.components_from_edges(n, *edges[t])
        sizes = np.bincount(lab)
        if abs(t - 0.7) < 1e-9:
            comp07 = (nc, lab, sizes)
        for si, s in enumerate(SEEDS):
            tr, te = RA.assign_clusters(lab, y, s, target)
            if si == 0:
                span = len(np.intersect1d(np.unique(lab[tr]), np.unique(lab[te])))
                resid = RA.max_jaccard_to_reference(M[te], M[tr])
                clusters.append(dict(
                    dataset=dset, threshold=t, n_components=int(nc),
                    n_singletons=int((sizes == 1).sum()),
                    n_multi=int((sizes > 1).sum()), max_cluster=int(sizes.max()),
                    top5_sizes=str(sorted(sizes.tolist())[-5:]),
                    n_redundant=int(n - nc), frac_redundant=float((n - nc) / n),
                    clusters_spanning_split=span,
                    resid_leak_frac=float((resid > t).mean()),
                    test_frac=len(te) / n, pos_train=float(y[tr].mean()),
                    pos_test=float(y[te].mean())))
                print(f"  hom@{t}: comps={nc} sing={int((sizes==1).sum())} "
                      f"max={int(sizes.max())} redund={(n-nc)/n:.3f} | VERIFY span={span} "
                      f"resid>{t}={float((resid>t).mean()):.4f} testfrac={len(te)/n:.3f} "
                      f"pos_tr/te={y[tr].mean():.3f}/{y[te].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
            rec(f"homology@{t}", s, tr, te)

    # drop-largest-component @0.7 (ALWAYS, regardless of size)
    nc, lab, sizes = comp07
    largest_id, lsize = int(sizes.argmax()), int(sizes.max())
    keep = np.where(lab != largest_id)[0]
    sub_lab, sub_y = lab[keep], y[keep]
    for s in SEEDS:
        trl, tel = RA.assign_clusters(sub_lab, sub_y, s, target)
        rec("droplargest@0.7", s, keep[trl], keep[tel])
    clusters.append(dict(dataset=dset, threshold="droplargest@0.7", n_components=-1,
                         n_singletons=-1, n_multi=-1, max_cluster=lsize,
                         top5_sizes=str(sorted(sizes.tolist())[-5:]),
                         n_redundant=-1, frac_redundant=-1.0, clusters_spanning_split=0,
                         resid_leak_frac=-1.0, test_frac=-1.0, pos_train=-1.0, pos_test=-1.0))
    print(f"  drop-largest@0.7: removed component size={lsize}, re-split remaining "
          f"{len(keep)} seqs ({time.time()-t0:.0f}s)", flush=True)

    del X, M
    gc.collect()
    print(f"  [{dset} full-scale robustness done in {time.time()-t0:.0f}s]", flush=True)
    return rows, clusters


def main():
    all_rows, all_cl = [], []
    for d in DSETS:
        try:
            r, c = run(d)
            all_rows += r; all_cl += c
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RA.RESULTS_DIR, "robustness_fullscale_long.csv"), index=False)
    pd.DataFrame(all_cl).to_csv(os.path.join(RA.RESULTS_DIR, "robustness_fullscale_clusters.csv"), index=False)

    out = []
    for d in df["dataset"].unique():
        for (k, m) in CONFIGS:
            o = df[(df.dataset == d) & (df.split_type == "original") & (df.k == k) & (df.model == m)]["accuracy"].mean()
            for label in [f"homology@{t}" for t in THRESHOLDS] + ["droplargest@0.7"]:
                sub = df[(df.dataset == d) & (df.split_type == label) & (df.k == k) & (df.model == m)]["accuracy"]
                out.append(dict(dataset=d, variant=label, model=m, k=k,
                                original_acc=round(float(o), 4),
                                corrected_acc_mean=round(float(sub.mean()), 4),
                                corrected_acc_std=round(float(sub.std(ddof=0)), 4),
                                delta_acc=round(float(o - sub.mean()), 4)))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(RA.RESULTS_DIR, "robustness_fullscale_summary.csv"), index=False)
    print("\n===== ROBUSTNESS FULL-SCALE SUMMARY =====", flush=True)
    print(summ.to_string(), flush=True)
    print("\nROBUSTNESS_DONE -> results/robustness_fullscale_summary.csv", flush=True)


if __name__ == "__main__":
    main()
