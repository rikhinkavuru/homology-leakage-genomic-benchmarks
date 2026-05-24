#!/usr/bin/env python3
"""
Homology-leakage audit of the Genomic Benchmarks `human_nontata_promoters`
DNA sequence-classification dataset.

Question: does the standard train/test split report inflated accuracy because
near-identical (homologous) sequences leak across the split? We quantify the
accuracy drop after a homology-aware re-split.

CPU-only. Pure numpy / pandas / scikit-learn / scipy. No GPU, no deep learning,
no API calls. Deterministic: every random seed is set and reported.

Run:   python run_audit.py
Outputs are written to ./results/

Similarity method (justification):
    We measure homology with the EXACT Jaccard similarity of k-mer sets (k=8).
    Each sequence -> its set of 8-mers (encoded as integers in [0, 4^8)). With a
    sparse binary sequence x k-mer matrix M, shared-k-mer counts for all pairs are
    a single sparse product M @ M.T. Because each 8-mer occurs in only ~10^2
    sequences, the all-pairs work is ~10^9 multiply-adds -- cheap enough to compute
    Jaccard EXACTLY in batches, so we avoid MinHash estimation error altogether.
    (This is the exact form of the task's recommended "Option A".)
"""

import os
import sys
import glob
import time
import itertools
import collections
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config (all seeds and thresholds are reported in the outputs)
# --------------------------------------------------------------------------
DATASET            = "human_nontata_promoters"
HERE               = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR        = os.path.join(HERE, "results")

KMER_FEATURE_SIZES = [4, 6]          # featurization k for classification (256, 4096 features)
SIM_K              = 8               # k-mer size for similarity / homology detection
SIM_THRESHOLDS     = [0.5, 0.7, 0.9] # Jaccard thresholds for the leakage report
CLUSTER_THRESHOLD  = 0.7             # edge threshold for the homology-aware clustering
CLUSTER_SPLIT_SEEDS = [0, 1, 2]      # 3 random cluster->split assignments
MODEL_SEED         = 0               # seed for LR / RF
RF_TREES           = 150
TARGET_TEST_FRAC   = 0.25            # match the original ~75/25 ratio
BATCH              = 500             # rows per batch for the sparse similarity products

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(MODEL_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Byte -> 2-bit base code lookup (A,C,G,T -> 0,1,2,3 ; everything else -> -1)
_TRANS = np.full(256, -1, dtype=np.int64)
for _b, _code in zip(b"ACGT", range(4)):
    _TRANS[_b] = _code


def banner(msg):
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74)


# --------------------------------------------------------------------------
# Step 1 -- load the real sequences
# --------------------------------------------------------------------------
def load_dataset():
    from genomic_benchmarks.loc2seq import download_dataset
    base = download_dataset(DATASET, version=0)
    label_map = {"negative": 0, "positive": 1}
    seqs, labels, splits = [], [], []
    for split in ("train", "test"):
        for cls, lab in label_map.items():
            for fp in sorted(glob.glob(os.path.join(base, split, cls, "*.txt"))):
                with open(fp) as fh:
                    seqs.append(fh.read().strip().upper())
                labels.append(lab)
                splits.append(split)
    df = pd.DataFrame({"seq": seqs, "label": labels, "split": splits})
    return df, base


def describe(df, base):
    banner("STEP 1 -- dataset")
    print(f"dataset           : {DATASET}")
    print(f"download path     : {base}")
    print(f"total sequences   : {len(df)}")
    ct = df.groupby(["split", "label"]).size().unstack().reindex(index=["train", "test"])
    ct.columns = [f"label={c}(0=neg,1=pos)" for c in ct.columns]
    print("\nn per class per split:")
    print(ct.to_string())
    print("\nsplit totals      :", df["split"].value_counts().to_dict())
    print("train fraction    : %.4f" % (df["split"].eq("train").mean()))

    L = df["seq"].str.len()
    print("\nsequence length   : min=%d max=%d mean=%.2f median=%d std=%.3f"
          % (L.min(), L.max(), L.mean(), int(L.median()), L.std()))
    sample = "".join(df["seq"].sample(min(3000, len(df)), random_state=0))
    print("alphabet (sampled):", dict(collections.Counter(sample)))
    non_acgt = int(df["seq"].str.contains("[^ACGT]").sum())
    print("seqs w/ non-ACGT  :", non_acgt)
    print("example sequence  :", df["seq"].iloc[0][:60], "...")
    return L


# --------------------------------------------------------------------------
# Step 2 -- k-mer featurization + classifiers
# --------------------------------------------------------------------------
def featurize(seqs, k):
    """Overlapping k-mer counts over the fixed ACGT vocabulary (4^k features).

    Using a fixed, data-independent vocabulary means the feature space is
    identical for every split, so there is no possibility of vocabulary leakage
    between train and test."""
    vocab = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k),
                          vocabulary=vocab, lowercase=False)
    X = vec.fit_transform(seqs)        # fixed vocab => deterministic transform
    return np.asarray(X.todense(), dtype=np.float32)


def make_models():
    """Fresh, unfitted models each call (avoids any state reuse across splits)."""
    lr = make_pipeline(
        Normalizer(norm="l2"),
        LogisticRegression(max_iter=5000, C=1.0, random_state=MODEL_SEED),
    )
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=-1,
                                random_state=MODEL_SEED)
    return {"LR": lr, "RF": rf}


def eval_split(X, y, train_idx, test_idx):
    out = {}
    for name, model in make_models().items():
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        yt = y[test_idx]
        out[name] = dict(
            accuracy=accuracy_score(yt, pred),
            auroc=roc_auc_score(yt, proba) if len(np.unique(yt)) > 1 else float("nan"),
            f1=f1_score(yt, pred),
        )
    return out


# --------------------------------------------------------------------------
# Step 3/4 -- similarity via exact k-mer Jaccard (sparse matrix products)
# --------------------------------------------------------------------------
def kmer_binary_matrix(seqs, k):
    """Sparse binary matrix (n_seq x 4^k): entry = 1 iff that k-mer occurs."""
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    indptr = np.zeros(len(seqs) + 1, dtype=np.int64)
    all_idx = []
    for i, s in enumerate(seqs):
        v = _TRANS[np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)]
        if len(v) >= k:
            win = np.lib.stride_tricks.sliding_window_view(v, k)   # (m-k+1, k)
            valid = (win >= 0).all(axis=1)
            codes = np.unique((win @ powers)[valid])
        else:
            codes = np.empty(0, dtype=np.int64)
        all_idx.append(codes)
        indptr[i + 1] = indptr[i] + len(codes)
    indices = np.concatenate(all_idx) if all_idx else np.empty(0, dtype=np.int64)
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr),
                             shape=(len(seqs), 4 ** k), dtype=np.float32)


def max_jaccard_to_reference(M_query, M_ref, batch=BATCH):
    """For each query row, the max EXACT Jaccard to any reference row."""
    sz_q = np.asarray(M_query.sum(axis=1)).ravel()
    sz_r = np.asarray(M_ref.sum(axis=1)).ravel()
    M_ref_T = M_ref.T.tocsr()
    out = np.zeros(M_query.shape[0], dtype=np.float32)
    for s in range(0, M_query.shape[0], batch):
        e = min(s + batch, M_query.shape[0])
        inter = (M_query[s:e] @ M_ref_T).toarray()
        union = sz_q[s:e, None] + sz_r[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        out[s:e] = jac.max(axis=1)
    return out


def homology_edges(M, threshold, batch=BATCH):
    """All undirected pairs (i<j) with EXACT Jaccard > threshold."""
    sz = np.asarray(M.sum(axis=1)).ravel()
    M_T = M.T.tocsr()
    n = M.shape[0]
    rows, cols = [], []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        inter = (M[s:e] @ M_T).toarray()
        union = sz[s:e, None] + sz[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        bi, bj = np.where(jac > threshold)
        gi = bi + s
        keep = gi < bj                      # upper triangle => each edge once
        rows.append(gi[keep]); cols.append(bj[keep])
    return np.concatenate(rows), np.concatenate(cols)


def assign_clusters(labels, y, seed, target_test=TARGET_TEST_FRAC):
    """Assign whole clusters to train/test, per class, to hit ~target_test
    while keeping class balance. No cluster is ever split (by construction)."""
    rng = np.random.RandomState(seed)
    clusters = collections.defaultdict(list)
    for idx, c in enumerate(labels):
        clusters[c].append(idx)
    # majority class per cluster (ties -> class 1)
    major = {c: int(round(float(np.mean(y[idxs])))) for c, idxs in clusters.items()}

    test_mask = np.zeros(len(labels), dtype=bool)
    for cls in (0, 1):
        cl_ids = [c for c in clusters if major[c] == cls]
        rng.shuffle(cl_ids)
        total = sum(len(clusters[c]) for c in cl_ids)
        target = target_test * total
        cum = 0
        for c in cl_ids:
            if cum < target:
                for idx in clusters[c]:
                    test_mask[idx] = True
                cum += len(clusters[c])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    df, base = load_dataset()
    describe(df, base)

    y = df["label"].to_numpy()
    seqs = df["seq"].tolist()
    orig_train = np.where(df["split"].to_numpy() == "train")[0]
    orig_test  = np.where(df["split"].to_numpy() == "test")[0]

    # ---- featurize once per k (reused for every split) ----
    banner("STEP 2 -- featurization + baseline on the ORIGINAL split")
    X_by_k = {}
    for k in KMER_FEATURE_SIZES:
        t = time.time()
        X_by_k[k] = featurize(seqs, k)
        print("featurized k=%d -> %s  (%.1fs)" % (k, X_by_k[k].shape, time.time() - t))

    original_results = {}
    for k in KMER_FEATURE_SIZES:
        t = time.time()
        res = eval_split(X_by_k[k], y, orig_train, orig_test)
        original_results[k] = res
        for m, r in res.items():
            print("  ORIGINAL k=%d %s : acc=%.4f auroc=%.4f f1=%.4f"
                  % (k, m, r["accuracy"], r["auroc"], r["f1"]))
        print("  (k=%d fit+eval %.1fs)" % (k, time.time() - t))

    # ---- Step 3: homology between original test and original train ----
    banner("STEP 3 -- train/test homology (exact %d-mer Jaccard)" % SIM_K)
    t = time.time()
    M = kmer_binary_matrix(seqs, SIM_K)
    print("built binary %d-mer matrix %s  nnz=%d  (%.1fs)"
          % (SIM_K, M.shape, M.nnz, time.time() - t))

    t = time.time()
    max_sim = max_jaccard_to_reference(M[orig_test], M[orig_train])
    print("computed test->train max-Jaccard for %d test seqs (%.1fs)"
          % (len(max_sim), time.time() - t))
    print("max-sim distribution: min=%.3f mean=%.3f median=%.3f p90=%.3f p99=%.3f max=%.3f"
          % (max_sim.min(), max_sim.mean(), np.median(max_sim),
             np.percentile(max_sim, 90), np.percentile(max_sim, 99), max_sim.max()))

    leak_rows = []
    for thr in SIM_THRESHOLDS:
        frac = float((max_sim > thr).mean())
        leak_rows.append({"threshold": thr, "fraction_test_leaked": frac,
                          "n_test_leaked": int((max_sim > thr).sum()),
                          "n_test": len(max_sim)})
        print("  fraction of test with a train near-duplicate at sim>%.1f : %.4f (%d/%d)"
              % (thr, frac, int((max_sim > thr).sum()), len(max_sim)))
    pd.DataFrame(leak_rows).to_csv(os.path.join(RESULTS_DIR, "leakage_fraction.csv"), index=False)

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(max_sim, bins=50, range=(0, 1), color="#3b6ea5", edgecolor="white")
    for thr in SIM_THRESHOLDS:
        plt.axvline(thr, color="crimson", ls="--", lw=1)
        plt.text(thr, plt.ylim()[1] * 0.92, f" {thr}", color="crimson", fontsize=9)
    plt.xlabel(f"max Jaccard ({SIM_K}-mer) of a TEST sequence to any TRAIN sequence")
    plt.ylabel("number of test sequences")
    plt.title(f"{DATASET}: train/test homology (original split)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "similarity_hist.png"), dpi=130)
    plt.close()
    print("saved results/similarity_hist.png")

    # ---- Step 4: homology-aware re-split ----
    banner("STEP 4 -- homology-aware re-split (cluster @ Jaccard > %.1f)" % CLUSTER_THRESHOLD)
    t = time.time()
    er, ec = homology_edges(M, CLUSTER_THRESHOLD)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    g = g + g.T
    ncomp, comp_labels = connected_components(g, directed=False)
    sizes = np.bincount(comp_labels)
    print("edges (pairs sim>%.1f): %d   (%.1fs)" % (CLUSTER_THRESHOLD, len(er), time.time() - t))
    print("connected components : %d   (from %d sequences)" % (ncomp, M.shape[0]))
    print("singletons           : %d" % int((sizes == 1).sum()))
    print("multi-seq clusters   : %d   largest=%d  top5=%s"
          % (int((sizes > 1).sum()), sizes.max(), sorted(sizes)[-5:]))
    n_redundant = int(len(comp_labels) - ncomp)
    print("redundant sequences (n_seq - n_clusters): %d (%.1f%% of all data)"
          % (n_redundant, 100.0 * n_redundant / len(comp_labels)))

    # 3 cluster->split assignments
    corrected_splits = {}
    for seed in CLUSTER_SPLIT_SEEDS:
        tr, te = assign_clusters(comp_labels, y, seed)
        corrected_splits[seed] = (tr, te)
        # verify no cluster spans the split
        spanning = np.intersect1d(np.unique(comp_labels[tr]), np.unique(comp_labels[te]))
        # residual leakage of the corrected split (should be ~0 above the cluster threshold)
        resid = max_jaccard_to_reference(M[te], M[tr])
        print("  seed=%d  n_train=%d n_test=%d (test frac %.3f)  pos_train=%.3f pos_test=%.3f"
              "  clusters_spanning_split=%d  resid_test_leak>0.7=%.4f"
              % (seed, len(tr), len(te), len(te) / (len(tr) + len(te)),
                 y[tr].mean(), y[te].mean(), len(spanning),
                 float((resid > CLUSTER_THRESHOLD).mean())))

    # ---- Step 5: retrain SAME models on corrected split, build table ----
    banner("STEP 5 -- headline: original vs homology-aware")
    corrected_per_seed = []
    corrected_summary = {}   # (k, model) -> {metric: (mean, std)}
    for k in KMER_FEATURE_SIZES:
        acc = collections.defaultdict(list)
        for seed in CLUSTER_SPLIT_SEEDS:
            tr, te = corrected_splits[seed]
            res = eval_split(X_by_k[k], y, tr, te)
            for m, r in res.items():
                corrected_per_seed.append(dict(feature_k=k, model=m, seed=seed, **r))
                for metric in ("accuracy", "auroc", "f1"):
                    acc[(m, metric)].append(r[metric])
        for m in ("LR", "RF"):
            corrected_summary[(k, m)] = {
                metric: (float(np.mean(acc[(m, metric)])), float(np.std(acc[(m, metric)])))
                for metric in ("accuracy", "auroc", "f1")
            }
    pd.DataFrame(corrected_per_seed).to_csv(
        os.path.join(RESULTS_DIR, "corrected_per_seed.csv"), index=False)

    # wide summary table with deltas
    table_rows = []
    for k in KMER_FEATURE_SIZES:
        for m in ("LR", "RF"):
            o = original_results[k][m]
            c = corrected_summary[(k, m)]
            row = {"feature_k": k, "model": m}
            for metric in ("accuracy", "auroc", "f1"):
                cm, cs = c[metric]
                row[f"original_{metric}"]       = round(o[metric], 4)
                row[f"corrected_{metric}_mean"] = round(cm, 4)
                row[f"corrected_{metric}_std"]  = round(cs, 4)
                row[f"delta_{metric}"]          = round(o[metric] - cm, 4)
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)

    # pretty print the headline table
    print("\n%-3s %-3s | %-25s | %-34s | %-7s"
          % ("k", "mdl", "ORIGINAL acc/auroc/f1", "HOMOLOGY-AWARE acc/auroc/f1 (mean)", "dAcc"))
    print("-" * 84)
    for r in table_rows:
        print("%-3d %-3s | %6.4f / %6.4f / %6.4f   | %6.4f / %6.4f / %6.4f          | %+.4f"
              % (r["feature_k"], r["model"],
                 r["original_accuracy"], r["original_auroc"], r["original_f1"],
                 r["corrected_accuracy_mean"], r["corrected_auroc_mean"], r["corrected_f1_mean"],
                 r["delta_accuracy"]))

    # ---- results.md ----
    write_results_md(df, L_stats(df), max_sim, leak_rows, ncomp, sizes, n_redundant,
                     corrected_splits, original_results, corrected_summary, table_rows,
                     time.time() - t0)
    banner("DONE  (%.1fs total)" % (time.time() - t0))
    print("All outputs in:", RESULTS_DIR)


def L_stats(df):
    L = df["seq"].str.len()
    return dict(min=int(L.min()), max=int(L.max()), mean=float(L.mean()),
                median=int(L.median()), mode=int(L.mode().iloc[0]))


def write_results_md(df, Ls, max_sim, leak_rows, ncomp, sizes, n_redundant,
                     corrected_splits, original_results, corrected_summary,
                     table_rows, runtime):
    n_train = int(df["split"].eq("train").sum())
    n_test  = int(df["split"].eq("test").sum())
    n_all = n_train + n_test

    lines = []
    lines.append(f"# Homology-leakage audit: `{DATASET}`\n")
    lines.append("Genomic Benchmarks DNA sequence-classification dataset "
                 "(binary: non-TATA promoter vs. background). "
                 "CPU-only, pure numpy/pandas/scikit-learn/scipy. "
                 f"Seeds: model={MODEL_SEED}, cluster splits={CLUSTER_SPLIT_SEEDS}. "
                 f"Total runtime {runtime:.0f}s.\n")

    lines.append("## 1. Dataset\n")
    lines.append(f"- Sequences: **{n_all}** total - train **{n_train}**, "
                 f"test **{n_test}** (original split = {n_train/n_all:.1%} / "
                 f"{n_test/n_all:.1%}).")
    pos = df.groupby("split")["label"].mean()
    lines.append(f"- Class balance (fraction positive): train {pos['train']:.3f}, "
                 f"test {pos['test']:.3f}.")
    lines.append(f"- Sequence length: mode **{Ls['mode']} bp** "
                 f"(min {Ls['min']}, max {Ls['max']}, mean {Ls['mean']:.1f}). "
                 "Sequences are real ACGT strings.\n")

    lines.append("## 2. Train/test homology (original split)\n")
    lines.append(f"For each of the {len(max_sim)} test sequences we computed the maximum "
                 f"exact {SIM_K}-mer Jaccard similarity to any training sequence.\n")
    lines.append("| max-Jaccard threshold | fraction of test with a train near-duplicate |")
    lines.append("|---|---|")
    for r in leak_rows:
        lines.append(f"| > {r['threshold']} | **{r['fraction_test_leaked']:.4f}** "
                     f"({r['n_test_leaked']}/{r['n_test']}) |")
    lines.append("")
    lines.append(f"Median test->train max-similarity = {np.median(max_sim):.3f}; "
                 f"99th percentile = {np.percentile(max_sim,99):.3f}. "
                 "See `similarity_hist.png`.\n")

    lines.append("## 3. Homology-aware re-split\n")
    lines.append(f"Sequences (train+test pooled) were connected by edges where "
                 f"{SIM_K}-mer Jaccard > {CLUSTER_THRESHOLD}; connected components are clusters. "
                 f"**{ncomp}** clusters from {n_all} sequences; "
                 f"{int((sizes==1).sum())} singletons, largest cluster = {int(sizes.max())}. "
                 f"**{n_redundant} sequences ({100.0*n_redundant/n_all:.1f}%) are redundant** "
                 "(have a near-duplicate elsewhere in the dataset). "
                 "Whole clusters were assigned to train or test (never split), targeting "
                 f"{TARGET_TEST_FRAC:.0%} test while preserving class balance, "
                 f"for {len(corrected_splits)} random seeds.\n")

    lines.append("## 4. Headline result\n")
    lines.append("| split | feat k | model | accuracy | AUROC | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for r in table_rows:
        k, m = r["feature_k"], r["model"]
        lines.append(f"| original | {k} | {m} | {r['original_accuracy']:.4f} | "
                     f"{r['original_auroc']:.4f} | {r['original_f1']:.4f} |")
        cs = corrected_summary[(k, m)]
        lines.append(f"| homology-aware | {k} | {m} | "
                     f"{cs['accuracy'][0]:.4f} +/- {cs['accuracy'][1]:.4f} | "
                     f"{cs['auroc'][0]:.4f} +/- {cs['auroc'][1]:.4f} | "
                     f"{cs['f1'][0]:.4f} +/- {cs['f1'][1]:.4f} |")
    lines.append("")
    lines.append("**Deltas (original - homology-aware):**\n")
    lines.append("| feat k | model | d-accuracy | d-AUROC | d-F1 |")
    lines.append("|---|---|---|---|---|")
    for r in table_rows:
        lines.append(f"| {r['feature_k']} | {r['model']} | "
                     f"{r['delta_accuracy']:+.4f} | {r['delta_auroc']:+.4f} | "
                     f"{r['delta_f1']:+.4f} |")
    lines.append("")

    canon = next(r for r in table_rows if r["feature_k"] == max(KMER_FEATURE_SIZES) and r["model"] == "LR")
    A = canon["original_accuracy"] * 100
    B = canon["corrected_accuracy_mean"] * 100
    lines.append("### Poster headline\n")
    lines.append(f"> After a homology-aware re-split, test accuracy on **{DATASET}** "
                 f"(LR, k={canon['feature_k']}) dropped by **{(A-B):.1f} points** "
                 f"(from **{A:.1f}%** to **{B:.1f}%**), indicating the standard split "
                 "overstates generalization.\n")

    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("saved results/results.md")


if __name__ == "__main__":
    main()
