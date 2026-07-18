#!/usr/bin/env python3
"""
Reviewer 3.3 fix: cluster (block) bootstrap that respects homology-cluster
correlation among test sequences, vs the paper's sample-wise bootstrap.

The paper resamples INDIVIDUAL test sequences. R3.3: near-duplicate test
sequences are correlated, so treating them as independent underestimates
uncertainty. Here we resample whole TEST-INTERNAL near-duplicate clusters
(8-mer Jaccard > 0.7 connected components within each test set) instead, and
report whether any significance verdict ("delta excludes 0") changes.

CPU-only, deterministic. Reuses the frozen pipeline exactly (same featurize,
same models, same split, same k, same bootstrap seed).
"""
import sys, time, json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.core import homology_split as H
from audit.pipeline.run_extended_models import make_models

BOOT = 1000
BSEED = 20240524                 # same as the paper
LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]


def load(d):
    cap = 10**12 if d in LEAKY else 20000
    df, info = S.discover_and_subsample(d, cap=cap, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    return seqs, y, otr, ote, len(ote) / len(df)


def full_components(seqs, thr=0.7):
    """Connected components over ALL sequences at 8-mer Jaccard > thr (for the split)."""
    M = H.kmer_binary_matrix(seqs, 8)
    er, ec = H._edges(M, thr)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    _, comp = connected_components(g + g.T, directed=False)
    return comp


def test_internal_clusters(seqs_test, thr=0.7):
    """Cluster labels for a test set, from within-test 8-mer Jaccard > thr components.
    Singletons get their own id, so a test set with no near-dups => every sample its
    own cluster => cluster bootstrap reduces exactly to sample-wise bootstrap."""
    M = H.kmer_binary_matrix(seqs_test, 8)
    er, ec = H._edges(M, thr)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)),
                          shape=(M.shape[0], M.shape[0]))
    n_comp, comp = connected_components(g + g.T, directed=False)
    return comp, n_comp


def sample_boot(c, rng):
    n = len(c)
    return c[rng.randint(0, n, size=(BOOT, n))].mean(axis=1)


def cluster_boot(c, clusters, rng):
    """Resample whole clusters with replacement; accuracy = mean over drawn members.
    Draw the same NUMBER of clusters as observed, each cluster contributes all members."""
    c = np.asarray(c, float)
    uniq = np.unique(clusters)
    # precompute member correctness arrays and sizes per cluster
    members = [c[clusters == u] for u in uniq]
    sizes = np.array([len(m) for m in members])
    sums = np.array([m.sum() for m in members])
    G = len(uniq)
    draws = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        pick = rng.randint(0, G, size=G)
        draws[b] = sums[pick].sum() / sizes[pick].sum()
    return draws


def ci(draws, lo=2.5, hi=97.5):
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def delta_boot(c_orig, cl_orig, c_corr, cl_corr, mode, seed=BSEED):
    """Bootstrap the delta (orig acc - corrected acc); orig & corrected resampled
    independently (their test sets differ), matching the paper's construction."""
    rng = np.random.RandomState(seed)
    if mode == "sample":
        do = sample_boot(np.asarray(c_orig, float), rng)
        dc = sample_boot(np.asarray(c_corr, float), rng)
    else:
        do = cluster_boot(c_orig, cl_orig, rng)
        dc = cluster_boot(c_corr, cl_corr, rng)
    d = do - dc
    lo, hi = ci(d)
    return float(np.mean(c_orig) - np.mean(c_corr)), lo, hi, bool(lo > 0 or hi < 0)


def run(d, t0):
    seqs, y, otr, ote, tf = load(d)
    X = RA.featurize(seqs, 6)
    print(f"[{d}] n={len(y)} test_frac={tf:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # corrected split (seed 0), same construction as the paper
    comp = full_components(seqs, 0.7)
    ctr, cte = H._assign(comp, y, 0, tf)

    # test-internal clusters for each test set
    ocl, n_ocl = test_internal_clusters([seqs[i] for i in ote], 0.7)
    ccl, n_ccl = test_internal_clusters([seqs[i] for i in cte], 0.7)
    print(f"  original test: n={len(ote)} internal-clusters={n_ocl} "
          f"(deff basis: {len(ote)/max(n_ocl,1):.2f} seq/cluster); "
          f"corrected test: n={len(cte)} internal-clusters={n_ccl}", flush=True)

    out = []
    for m in ["RF", "LR"]:
        mdl = make_models()[m]; mdl.fit(X[otr], y[otr])
        c_orig = (mdl.predict(X[ote]) == y[ote]).astype(float)
        mdl2 = make_models()[m]; mdl2.fit(X[ctr], y[ctr])
        c_corr = (mdl2.predict(X[cte]) == y[cte]).astype(float)

        rng = np.random.RandomState(BSEED)
        s_orig = sample_boot(c_orig, rng);  s_lo, s_hi = ci(s_orig)
        rng = np.random.RandomState(BSEED)
        k_orig = cluster_boot(c_orig, ocl, rng); k_lo, k_hi = ci(k_orig)

        dm_s, dlo_s, dhi_s, ex_s = delta_boot(c_orig, ocl, c_corr, ccl, "sample")
        dm_k, dlo_k, dhi_k, ex_k = delta_boot(c_orig, ocl, c_corr, ccl, "cluster")

        row = dict(dataset=d, model=f"{m}_k6",
                   orig_acc=round(float(c_orig.mean()), 4),
                   corr_acc=round(float(c_corr.mean()), 4),
                   ci_sample=[round(s_lo, 4), round(s_hi, 4)],
                   ci_cluster=[round(k_lo, 4), round(k_hi, 4)],
                   ci_width_sample=round(s_hi - s_lo, 4),
                   ci_width_cluster=round(k_hi - k_lo, 4),
                   width_inflation=round((k_hi - k_lo) / max(s_hi - s_lo, 1e-9), 2),
                   delta=round(dm_s, 4),
                   delta_ci_sample=[round(dlo_s, 4), round(dhi_s, 4)], excl0_sample=ex_s,
                   delta_ci_cluster=[round(dlo_k, 4), round(dhi_k, 4)], excl0_cluster=ex_k,
                   verdict_flips=bool(ex_s != ex_k))
        out.append(row)
        print(f"  {m}_k6: orig acc CI sample={row['ci_sample']} cluster={row['ci_cluster']} "
              f"(x{row['width_inflation']} wider); delta excl0 sample={ex_s} cluster={ex_k} "
              f"FLIP={row['verdict_flips']} ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["human_nontata_promoters"]
    t0 = time.time()
    allrows = []
    for d in which:
        allrows += run(d, t0)
    import pandas as pd
    pd.DataFrame(allrows).to_csv("results/cluster_bootstrap.csv", index=False)
    print("\nCLUSTER_BOOTSTRAP_DONE", json.dumps(allrows, indent=1))
