#!/usr/bin/env python3
"""
Genomics-geometry experiments via expkit.

GB1  Length-cap disclosure + containment-index leakage re-measure. Jaccard <= Lmin/Lmax,
     so variable-length datasets may be undetectably 'clean'. Containment |A∩B|/min(|A|,|B|)
     is length-robust; does any 'clean' dataset become leaky under containment?
GB4  GC/length covariate-shift audit of the corrected split: GC + length distributions for
     corrected-train vs corrected-test and clustered vs singleton, both leaky sets; plus a
     GC-matched corrected split to test whether the RF drop is covariate shift vs memorization.
G11  Cluster cohesion (single-linkage chaining diagnostic) + size distribution.
"""
import time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import expkit as E
import homology_split as H

t0 = time.time()
ALL = E.LEAKY + E.CLEAN

# ---------------- GB1: length caps + containment leakage ----------------
gb1 = []
for d in ALL:
    seqs, y, otr, ote, tf = E.load(d)
    L = np.array([len(s) for s in seqs])
    cap = float(L.min() / L.max())        # worst-case Jaccard bound for a min/max pair
    jac = E.cached_max_sim(d, seqs, otr, ote, 8, mode="jaccard")
    con = E.cached_max_sim(d, seqs, otr, ote, 8, mode="containment")
    gb1.append(dict(dataset=d, n=len(seqs), Lmin=int(L.min()), Lmed=int(np.median(L)),
                    Lmax=int(L.max()), jaccard_lenpair_cap=round(cap, 4),
                    leak_jaccard_0p7=round(float((jac > 0.7).mean()), 4),
                    leak_contain_0p7=round(float((con > 0.7).mean()), 4),
                    leak_contain_0p9=round(float((con > 0.9).mean()), 4),
                    verdict_jac=("LEAKY" if (jac > 0.7).mean() > 0.1 else "clean"),
                    verdict_con=("LEAKY" if (con > 0.7).mean() > 0.1 else "clean")))
    print(f"[GB1 {d}] jac={gb1[-1]['leak_jaccard_0p7']} con={gb1[-1]['leak_contain_0p7']} "
          f"cap={cap:.3f} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(gb1).to_csv("results/exp_gb1_containment.csv", index=False)

# ---------------- GB4: GC/length covariate shift + matched control ----------------
gb4, gb4b = [], []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    gc = E.gc_content(seqs); L = np.array([len(s) for s in seqs])
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    # clustered vs singleton
    _, counts = np.unique(comp, return_counts=True)
    size_of = counts[comp]
    clustered = size_of > 1
    def stat(mask): return (round(float(gc[mask].mean()), 4), round(float(L[mask].mean()), 1))
    gc_ctr, L_ctr = stat(ctr); gc_cte, L_cte = stat(cte)
    gc_cl, L_cl = stat(clustered); gc_sg, L_sg = stat(~clustered)
    gb4.append(dict(dataset=d, gc_corr_train=gc_ctr, gc_corr_test=gc_cte,
                    gc_gap=round(gc_ctr - gc_cte, 4), len_corr_train=L_ctr, len_corr_test=L_cte,
                    gc_clustered=gc_cl, gc_singleton=gc_sg, gc_clust_gap=round(gc_cl - gc_sg, 4),
                    frac_clustered=round(float(clustered.mean()), 4)))
    # matched control: does a GC/length-matched corrected split give the same RF drop?
    X = E.featurize(seqs, 6)
    co = (E.models()["RF"].fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).mean()
    cc = (E.models()["RF"].fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).mean()
    # stratify test by whether its GC is within the train GC IQR (proxy for in-distribution)
    q1, q3 = np.percentile(gc[ctr], [25, 75])
    in_dist = (gc[cte] >= q1) & (gc[cte] <= q3)
    acc_in = (E.models()["RF"].fit(X[ctr], y[ctr]).predict(X[cte[in_dist]]) == y[cte[in_dist]]).mean() if in_dist.sum() else float('nan')
    gb4b.append(dict(dataset=d, orig_acc=round(float(co), 4), corr_acc=round(float(cc), 4),
                     drop=round(float(co - cc), 4),
                     corr_acc_GCindist=round(float(acc_in), 4),
                     n_GCindist=int(in_dist.sum()), n_test=len(cte)))
    print(f"[GB4 {d}] gc_gap={gb4[-1]['gc_gap']} clust_gap={gb4[-1]['gc_clust_gap']} "
          f"drop={gb4b[-1]['drop']} corr_GCindist={gb4b[-1]['corr_acc_GCindist']} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(gb4).to_csv("results/exp_gb4_covariate.csv", index=False)
pd.DataFrame(gb4b).to_csv("results/exp_gb4_matched.csv", index=False)

# ---------------- G11: cluster cohesion / chaining ----------------
g11 = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    M = H.kmer_binary_matrix(seqs, 8)
    er, ec = H._edges(M, 0.7)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0],)*2)
    ncomp, comp = connected_components(g + g.T, directed=False)
    sizes = np.bincount(comp)
    multi = sizes[sizes > 1]
    # cohesion of the largest cluster: fraction of member pairs actually > 0.7 (edges/possible)
    biggest = int(sizes.argmax()); members = np.where(comp == biggest)[0]
    nm = len(members)
    edge_set = set(zip(er.tolist(), ec.tolist()))
    present = sum(1 for i in range(nm) for j in range(i+1, nm)
                  if (members[i], members[j]) in edge_set or (members[j], members[i]) in edge_set)
    possible = nm * (nm - 1) // 2
    g11.append(dict(dataset=d, n_components=int(ncomp), n_multi=int((sizes > 1).sum()),
                    max_cluster=int(sizes.max()), mean_multi_size=round(float(multi.mean()), 2),
                    frac_pairwise=round(float((multi == 2).mean()), 3),
                    largest_cluster_cohesion=round(present / possible, 4) if possible else 1.0))
    print(f"[G11 {d}] max_cluster={g11[-1]['max_cluster']} cohesion={g11[-1]['largest_cluster_cohesion']} "
          f"frac_pairwise={g11[-1]['frac_pairwise']} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(g11).to_csv("results/exp_g11_cohesion.csv", index=False)

print("\nEXP_GEOMETRY_DONE", round(time.time()-t0), "s")
print("GB1:\n", pd.DataFrame(gb1).to_string(index=False))
print("GB4 covariate:\n", pd.DataFrame(gb4).to_string(index=False))
print("GB4 matched:\n", pd.DataFrame(gb4b).to_string(index=False))
print("G11:\n", pd.DataFrame(g11).to_string(index=False))
