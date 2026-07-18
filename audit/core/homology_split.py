#!/usr/bin/env python3
"""
homology_split.py -- a drop-in, homology-aware train/test splitter for biological
sequence datasets. Pure numpy + scipy (CPU only, no GPU, no external services).

Problem it fixes
----------------
Random train/test splits of DNA/protein datasets leak near-duplicate (homologous)
sequences across the split, inflating reported accuracy. This splitter clusters
near-duplicates together (exact k-mer Jaccard >= threshold, single linkage via
connected components) and assigns WHOLE clusters to either train or test, so no
near-duplicate pair is split across the boundary.

Guarantees (see `verify_split`)
-------------------------------
* no cluster spans the split  -> residual cross-split similarity > threshold == 0
* class balance is preserved  (whole clusters assigned per class)
* the train/test ratio matches `test_frac` (up to cluster granularity)
* fully deterministic given `seed`

Usage
-----
>>> from homology_split import homology_aware_split
>>> seqs   = ["ACGT...", "ACGA...", ...]      # list of DNA strings (ACGT, case-insensitive)
>>> labels = [0, 1, 0, 1, ...]               # class labels (binary or multiclass)
>>> train_idx, test_idx = homology_aware_split(seqs, labels, test_frac=0.25,
...                                            threshold=0.7, seed=0)
>>> # X[train_idx] / X[test_idx] are now homology-disjoint at Jaccard > 0.7
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

__all__ = ["homology_aware_split", "kmer_binary_matrix",
           "max_jaccard_to_reference", "verify_split"]

_TRANS = np.full(256, -1, dtype=np.int64)
for _b, _c in zip(b"ACGT", range(4)):
    _TRANS[_b] = _c


def kmer_binary_matrix(sequences, k=8):
    """Sparse binary (n_seq x 4**k) matrix; entry 1 iff that k-mer occurs.
    k-mers containing non-ACGT characters are skipped."""
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    indptr = np.zeros(len(sequences) + 1, dtype=np.int64)
    chunks = []
    for i, s in enumerate(sequences):
        v = _TRANS[np.frombuffer(s.upper().encode("ascii", "ignore"), dtype=np.uint8)]
        if len(v) >= k:
            win = np.lib.stride_tricks.sliding_window_view(v, k)
            ok = (win >= 0).all(axis=1)
            codes = np.unique((win @ powers)[ok])
        else:
            codes = np.empty(0, dtype=np.int64)
        chunks.append(codes)
        indptr[i + 1] = indptr[i] + len(codes)
    idx = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(idx), np.float32), idx, indptr),
                             shape=(len(sequences), 4 ** k), dtype=np.float32)


def max_jaccard_to_reference(M_query, M_ref, batch=500):
    """For each query row, the max EXACT Jaccard to any reference row."""
    sq = np.asarray(M_query.sum(1)).ravel()
    sr = np.asarray(M_ref.sum(1)).ravel()
    RT = M_ref.T.tocsr()
    out = np.zeros(M_query.shape[0], dtype=np.float32)
    for s in range(0, M_query.shape[0], batch):
        e = min(s + batch, M_query.shape[0])
        inter = (M_query[s:e] @ RT).toarray()
        union = sq[s:e, None] + sr[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        out[s:e] = jac.max(axis=1) if e > s else out[s:e]
    return out


def _edges(M, threshold, batch=500):
    sz = np.asarray(M.sum(1)).ravel()
    MT = M.T.tocsr()
    rows, cols = [], []
    for s in range(0, M.shape[0], batch):
        e = min(s + batch, M.shape[0])
        inter = (M[s:e] @ MT).toarray()
        union = sz[s:e, None] + sz[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        bi, bj = np.where(jac > threshold)
        gi = bi + s
        keep = gi < bj
        rows.append(gi[keep]); cols.append(bj[keep])
    return np.concatenate(rows), np.concatenate(cols)


def _assign(labels_cluster, y, seed, test_frac):
    """Assign whole clusters to train/test, per class, ~test_frac, no cluster split."""
    rng = np.random.RandomState(seed)
    clusters = {}
    for idx, c in enumerate(labels_cluster):
        clusters.setdefault(c, []).append(idx)
    major = {c: int(np.bincount(y[ix]).argmax()) for c, ix in clusters.items()}
    test_mask = np.zeros(len(labels_cluster), dtype=bool)
    for cls in np.unique(y):
        cl = [c for c in clusters if major[c] == cls]
        rng.shuffle(cl)
        total = sum(len(clusters[c]) for c in cl)
        target, cum = test_frac * total, 0
        for c in cl:
            if cum < target:
                for ix in clusters[c]:
                    test_mask[ix] = True
                cum += len(clusters[c])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def homology_aware_split(sequences, labels, test_frac=0.25, threshold=0.7,
                         seed=0, k=8):
    """Return (train_idx, test_idx) such that no two sequences with k-mer Jaccard
    > `threshold` are split across train and test. Class balance and the train/test
    ratio (~`test_frac`) are preserved; deterministic given `seed`.

    Parameters
    ----------
    sequences : list[str]  DNA strings (ACGT, case-insensitive).
    labels    : array-like class labels (binary or multiclass).
    test_frac : float      target fraction of sequences in the test split.
    threshold : float      Jaccard cutoff defining "near-duplicate" (0..1).
    seed      : int        RNG seed for the cluster->split assignment.
    k         : int        k-mer length for the similarity (default 8).
    """
    y = np.asarray(labels)
    if len(sequences) != len(y):
        raise ValueError("sequences and labels must have the same length")
    M = kmer_binary_matrix(sequences, k)
    er, ec = _edges(M, threshold)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    _, comp = connected_components(g + g.T, directed=False)
    return _assign(comp, y, seed, test_frac)


def verify_split(sequences, train_idx, test_idx, threshold=0.7, k=8):
    """Return diagnostics confirming the guarantees: residual cross-split leakage
    (fraction of test sequences with a train neighbor at Jaccard > threshold; should
    be 0.0), plus the realized test fraction."""
    M = kmer_binary_matrix(sequences, k)
    resid = max_jaccard_to_reference(M[test_idx], M[train_idx])
    n = len(train_idx) + len(test_idx)
    return dict(residual_leak_fraction=float((resid > threshold).mean()),
                max_residual_jaccard=float(resid.max()) if len(resid) else 0.0,
                test_fraction=len(test_idx) / n,
                n_train=int(len(train_idx)), n_test=int(len(test_idx)))


def _read_fasta(path):
    """Minimal FASTA reader -> (ids, sequences). No external dependency."""
    ids, seqs, cur_id, cur = [], [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id); seqs.append("".join(cur))
                cur_id = line[1:].split()[0] if len(line) > 1 else str(len(ids))
                cur = []
            elif line:
                cur.append(line)
    if cur_id is not None:
        ids.append(cur_id); seqs.append("".join(cur))
    return ids, seqs


def _self_test():
    rng = np.random.RandomState(0)
    seqs, ys = [], []
    for c in range(2):
        for _ in range(200):
            s = "".join(rng.choice(list("ACGT"), 120))
            seqs.append(s); ys.append(c)
            if rng.rand() < 0.4:                       # inject a near-duplicate
                t = list(s); t[rng.randint(120)] = rng.choice(list("ACGT"))
                seqs.append("".join(t)); ys.append(c)
    tr, te = homology_aware_split(seqs, ys, test_frac=0.25, threshold=0.7, seed=0)
    v = verify_split(seqs, tr, te, threshold=0.7)
    print("self-test split:", v)
    assert v["residual_leak_fraction"] == 0.0, "residual leakage must be 0"
    print("OK: zero residual leakage, test_fraction=%.3f" % v["test_fraction"])


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser(
        description="Homology-aware train/test split for biological sequence datasets. "
                    "With no --fasta, runs a synthetic self-test.")
    ap.add_argument("--fasta", help="input FASTA of sequences")
    ap.add_argument("--labels", help="text file: one integer class label per FASTA record "
                                     "(same order). If omitted, all sequences share one class.")
    ap.add_argument("--out", default="splits.json", help="output JSON (train/test indices + ids)")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    if not a.fasta:
        _self_test(); sys.exit(0)
    ids, seqs = _read_fasta(a.fasta)
    if a.labels:
        with open(a.labels) as fh:
            labels = [int(x) for x in fh.read().split()]
    else:
        labels = [0] * len(seqs)
    if len(labels) != len(seqs):
        raise SystemExit(f"#labels ({len(labels)}) != #sequences ({len(seqs)})")
    tr, te = homology_aware_split(seqs, labels, test_frac=a.test_frac,
                                  threshold=a.threshold, seed=a.seed, k=a.k)
    v = verify_split(seqs, tr, te, threshold=a.threshold, k=a.k)
    out = {"params": {"test_frac": a.test_frac, "threshold": a.threshold, "seed": a.seed, "k": a.k},
           "verification": v,
           "train_idx": tr.tolist(), "test_idx": te.tolist(),
           "train_ids": [ids[i] for i in tr], "test_ids": [ids[i] for i in te]}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {a.out}: n_train={v['n_train']} n_test={v['n_test']} "
          f"residual_leak>{a.threshold}={v['residual_leak_fraction']:.6f} "
          f"test_frac={v['test_fraction']:.3f}")
