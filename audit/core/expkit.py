#!/usr/bin/env python3
"""
expkit.py -- shared, verified experiment kit for the homology-leakage revision.

Wraps the FROZEN pipeline (run_audit / run_suite / homology_split /
run_extended_models) so every new experiment is exactly consistent with the
published results. Nothing here changes the frozen primitives; it only exposes
them behind one tested surface, plus adds the new tools the revision needs
(canonical k-mers, containment clustering, cluster/block bootstrap, metrics).

Decision rule: sklearn `.predict()` (argmax / decision-function sign) EVERYWHERE.
This is the single canonical rule for T1.8; point estimates may differ trivially
from frozen tables that used a `proba>=0.5` rule for RF -- that is the intended
reconciliation.
"""
from __future__ import annotations
import os, itertools
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.core import homology_split as H
from audit.pipeline.run_extended_models import make_models as _make4

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "human_ocr_ensembl", "demo_human_or_worm",
         "demo_coding_vs_intergenomic_seqs", "drosophila_enhancers_stark"]
THREECLASS = "human_ensembl_regulatory"
MODEL_SEED = RA.MODEL_SEED
BSEED = 20240524
RESULTS = RA.RESULTS_DIR

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
DATACACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datacache")

def _load_df(dset, c, seed):
    """Stable on-disk cache of discover_and_subsample output, keyed by (dset,cap,seed).
    genomic_benchmarks re-extracts the dataset folder on EVERY download_dataset call
    (force_download), which makes parallel loads race and corrupt the cache. Pre-build
    this cache SERIALLY (prefetch.py) so all downstream loads read the pickle and never
    trigger a re-download."""
    os.makedirs(DATACACHE, exist_ok=True)
    path = os.path.join(DATACACHE, f"{dset}__cap{c}__seed{seed}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    df, info = S.discover_and_subsample(dset, cap=c, seed=seed)
    df.to_pickle(path)
    return df

def load(dset, full=None, cap=20000, seed=0):
    """Return seqs, y(int array), otr, ote (original split indices), test_frac.
    full=True forces no subsample; default: leaky->full, else 20k (matches frozen)."""
    if full is None:
        full = dset in LEAKY
    c = 10**12 if full else cap
    df = _load_df(dset, c, seed)
    y = df["label"].to_numpy()
    seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    return seqs, y, otr, ote, len(ote) / len(df)

# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def featurize(seqs, k):
    """Standard overlapping k-mer counts (4^k), same as the frozen pipeline."""
    return RA.featurize(seqs, k)

_COMP = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G on base-4 (A=0,C=1,G=2,T=3)

def _canon_map(k):
    """canon_id[i] = index of the canonical (min of kmer, revcomp) for k-mer i."""
    n = 4 ** k
    idx = np.arange(n)
    digits = np.zeros((n, k), dtype=np.int64)
    tmp = idx.copy()
    for pos in range(k - 1, -1, -1):
        digits[:, pos] = tmp % 4
        tmp //= 4
    comp = np.vectorize(_COMP.get)(digits)          # complement each base
    rc = comp[:, ::-1]                              # reverse -> reverse complement
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    rc_idx = rc @ powers
    return np.minimum(idx, rc_idx)

def featurize_canonical(seqs, k):
    """Reverse-complement-collapsed k-mer counts (strand-symmetric features).
    Columns sharing a canonical id are summed -> n_unique_canonical features."""
    X = RA.featurize(seqs, k)                       # (n, 4^k) float32
    canon = _canon_map(k)
    uniq, inv = np.unique(canon, return_inverse=True)
    Xc = np.zeros((X.shape[0], len(uniq)), dtype=np.float32)
    np.add.at(Xc.T, inv, X.T)                       # sum columns within each canon group
    return Xc

# ----------------------------------------------------------------------------
# similarity / clustering
# ----------------------------------------------------------------------------
def _sim_edges(M, threshold, mode="jaccard", batch=500):
    """Edges (i<j) with similarity > threshold. mode: 'jaccard' or 'containment'
    (|A∩B|/min(|A|,|B|), length-robust)."""
    sz = np.asarray(M.sum(1)).ravel()
    MT = M.T.tocsr()
    rows, cols = [], []
    for s in range(0, M.shape[0], batch):
        e = min(s + batch, M.shape[0])
        inter = (M[s:e] @ MT).toarray()
        if mode == "jaccard":
            denom = sz[s:e, None] + sz[None, :] - inter
        elif mode == "containment":
            denom = np.minimum(sz[s:e, None], sz[None, :])
        else:
            raise ValueError(mode)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, inter / denom, 0.0)
        bi, bj = np.where(sim > threshold)
        gi = bi + s
        keep = gi < bj
        rows.append(gi[keep]); cols.append(bj[keep])
    return np.concatenate(rows), np.concatenate(cols)

def clusters(seqs, threshold=0.7, k=8, mode="jaccard", linkage="single", canonical=False):
    """Connected-component (single-linkage) near-duplicate clusters.
    linkage='single' matches the frozen method. mode/canonical extend it."""
    if canonical:
        M = _binary_canonical(seqs, k)
    else:
        M = H.kmer_binary_matrix(seqs, k)
    er, ec = _sim_edges(M, threshold, mode=mode)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    if linkage == "single":
        _, comp = connected_components(g + g.T, directed=False)
        return comp
    raise ValueError("only single-linkage here; complete/average handled separately")

def _binary_canonical(seqs, k):
    """Binary presence matrix over canonical k-mers (for strand-aware leakage)."""
    M = H.kmer_binary_matrix(seqs, k)
    canon = _canon_map(k)
    uniq, inv = np.unique(canon, return_inverse=True)
    # fold columns: presence in any strand-equivalent kmer
    out = sparse.csr_matrix((M.shape[0], len(uniq)), dtype=np.float32)
    M = M.tocsc()
    cols = [None] * len(uniq)
    from collections import defaultdict
    groups = defaultdict(list)
    for j, g in enumerate(inv):
        groups[g].append(j)
    data_rows, data_cols = [], []
    Mcsr = M.tocsr()
    for gi, js in groups.items():
        sub = Mcsr[:, js]
        present = np.asarray((sub.sum(1) > 0)).ravel().nonzero()[0]
        data_rows.append(present); data_cols.append(np.full(len(present), gi))
    r = np.concatenate(data_rows); c = np.concatenate(data_cols)
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                             shape=(M.shape[0], len(uniq)))

def assign(comp, y, seed, test_frac):
    """Whole-cluster train/test assignment (frozen rule)."""
    return H._assign(comp, np.asarray(y), seed, test_frac)

def cached_clusters(dset, seqs, threshold=0.7, k=8, mode="jaccard", canonical=False):
    """Single-linkage components, cached to disk (the 155k-pairwise is computed ONCE
    per dataset/threshold and reused across experiments)."""
    tag = f"comp_{dset}_thr{threshold}_k{k}_{mode}{'_canon' if canonical else ''}.npy"
    path = os.path.join(DATACACHE, tag)
    if os.path.exists(path):
        return np.load(path)
    comp = clusters(seqs, threshold, k, mode=mode, canonical=canonical)
    np.save(path, comp)
    return comp

def cached_max_sim(dset, seqs, tr, te, k=8, mode="jaccard", canonical=False):
    """Max test->train similarity on the ORIGINAL split, cached to disk."""
    tag = f"sim_{dset}_k{k}_{mode}{'_canon' if canonical else ''}.npy"
    path = os.path.join(DATACACHE, tag)
    if os.path.exists(path):
        return np.load(path)
    sim = max_sim_to_train(seqs, tr, te, k, mode=mode, canonical=canonical)
    np.save(path, sim)
    return sim

def max_sim_to_train(seqs, tr_idx, te_idx, k=8, mode="jaccard", canonical=False):
    """Max similarity of each test seq to any train seq (for leakage fractions)."""
    if canonical:
        M = _binary_canonical(seqs, k)
    else:
        M = H.kmer_binary_matrix(seqs, k)
    if mode == "jaccard":
        return H.max_jaccard_to_reference(M[te_idx], M[tr_idx])
    # containment
    Mq, Mr = M[te_idx], M[tr_idx]
    sq = np.asarray(Mq.sum(1)).ravel(); sr = np.asarray(Mr.sum(1)).ravel()
    RT = Mr.T.tocsr(); out = np.zeros(Mq.shape[0], np.float32)
    for s in range(0, Mq.shape[0], 500):
        e = min(s + 500, Mq.shape[0])
        inter = (Mq[s:e] @ RT).toarray()
        denom = np.minimum(sq[s:e, None], sr[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, inter / denom, 0.0)
        out[s:e] = sim.max(1)
    return out

# ----------------------------------------------------------------------------
# models / evaluation (canonical decision rule = .predict())
# ----------------------------------------------------------------------------
def models():
    """The four classical models, identical objects to the frozen extended run."""
    return _make4()

def correctness(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    return (model.predict(X[te]) == y[te]).astype(float)

def metrics(model, X, y, tr, te):
    """acc / auroc / f1 with the canonical .predict() rule. Multiclass-safe."""
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    yte = y[te]
    acc = float(accuracy_score(yte, pred))
    classes = np.unique(y)
    f1 = float(f1_score(yte, pred, average=("binary" if len(classes) == 2 else "macro"),
                        zero_division=0))
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[te])
            if len(classes) == 2:
                auroc = float(roc_auc_score(yte, proba[:, 1]))
            else:
                auroc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(X[te])
            auroc = float(roc_auc_score(yte, dec)) if len(classes) == 2 else float("nan")
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")
    return dict(acc=acc, auroc=auroc, f1=f1, pred=pred)

# ----------------------------------------------------------------------------
# bootstrap (sample-wise + cluster/block) -- matches cluster_bootstrap.py
# ----------------------------------------------------------------------------
def sample_boot(c, boot=1000, seed=BSEED):
    rng = np.random.RandomState(seed)
    c = np.asarray(c, float); n = len(c)
    return c[rng.randint(0, n, size=(boot, n))].mean(1)

def cluster_boot(c, cl, boot=1000, seed=BSEED):
    """Resample whole clusters with replacement (pairs cluster bootstrap)."""
    rng = np.random.RandomState(seed)
    c = np.asarray(c, float)
    uniq = np.unique(cl)
    sizes = np.array([(cl == u).sum() for u in uniq])
    sums = np.array([c[cl == u].sum() for u in uniq])
    G = len(uniq); out = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, size=G)
        out[b] = sums[pick].sum() / sizes[pick].sum()
    return out

def ci(draws, lo=2.5, hi=97.5):
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))

def icc_oneway(c, cl):
    """One-way ICC of a binary/continuous outcome across clusters + design effect."""
    c = np.asarray(c, float); uniq = np.unique(cl)
    G = len(uniq); N = len(c)
    grand = c.mean()
    ni = np.array([(cl == u).sum() for u in uniq])
    means = np.array([c[cl == u].mean() for u in uniq])
    ssb = float((ni * (means - grand) ** 2).sum())
    ssw = float(sum(((c[cl == u] - means[i]) ** 2).sum() for i, u in enumerate(uniq)))
    msb = ssb / (G - 1) if G > 1 else 0.0
    msw = ssw / (N - G) if N > G else 0.0
    n0 = (N - (ni ** 2).sum() / N) / (G - 1) if G > 1 else 1.0
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
    mbar = N / G
    deff = 1 + (mbar - 1) * icc
    return dict(icc=float(icc), deff=float(deff), n_clusters=int(G), n=int(N),
                mean_cluster_size=float(mbar))

def gc_content(seqs):
    out = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        s = s.upper(); L = len(s)
        out[i] = (s.count("G") + s.count("C")) / L if L else 0.0
    return out


if __name__ == "__main__":
    # self-test on nontata (cheap): consistency with frozen numbers
    seqs, y, otr, ote, tf = load("human_nontata_promoters")
    X = featurize(seqs, 6)
    m = metrics(models()["RF"], X, y, otr, ote)
    print(f"[self-test] nontata RF k6 original: acc={m['acc']:.4f} auroc={m['auroc']:.4f} "
          f"f1={m['f1']:.4f} (frozen ~0.9284/0.9938/0.9342)")
    Xc = featurize_canonical(seqs, 6)
    print(f"[self-test] canonical k6 features: {X.shape[1]} -> {Xc.shape[1]} cols")
    comp = clusters(seqs, 0.7, 8)
    tr, te = assign(comp, y, 0, tf)
    v = H.verify_split(seqs, tr, te, 0.7, 8)
    print(f"[self-test] homology split resid={v['residual_leak_fraction']:.6f} tf={v['test_fraction']:.3f}")
    print("OK")
