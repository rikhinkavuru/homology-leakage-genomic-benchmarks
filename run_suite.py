#!/usr/bin/env python3
"""
Scale the homology-leakage audit across the Genomic Benchmarks suite.

REUSES the validated pipeline from run_audit.py by importing it directly:
featurization (k-mer counts k=4/6, L2-normalized for LR), models
(LogisticRegression + RandomForest, identical hyperparameters), exact-Jaccard
similarity (sparse M @ M.T, no MinHash), and the whole-cluster re-split logic.
Nothing in those primitives is changed -- cross-dataset comparability is exact.

Adds, for every dataset:
  * deterministic stratified subsampling (by split x class) for large datasets,
  * NEGATIVE CONTROL: a random re-split at the same train/test ratio and class
    balance, ignoring clusters (should show ~no drop and ~no capacity scaling),
  * clustering robustness: a single-linkage threshold sweep {0.5, 0.7, 0.9}
    (each provably zero residual leakage at its own threshold) plus a
    "drop the largest connected component" diagnostic for datasets that have a
    big chained cluster -- both preserve the resid>thr == 0 guarantee, which a
    complete-linkage refinement cannot (de-chaining can place a >thr pair across
    the split), so we use these instead of complete-linkage.
  * per-dataset verification (0 clusters span the split, residual leakage,
    balance, ratio) and 3-seed mean +/- std.

Run:   python run_suite.py
Outputs extend ./results/ (+ ./results/figures/).
"""
import os
import sys
import glob
import time
import collections
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_audit as RA          # <-- the validated single-dataset pipeline

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
BINARY_DSETS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "demo_human_or_worm",
    "demo_coding_vs_intergenomic_seqs",
    "drosophila_enhancers_stark",
]
SMOKE_DSETS      = ["dummy_mouse_enhancers_ensembl"]   # smoke test only
MULTICLASS_DSETS = ["human_ensembl_regulatory"]        # optional 3-class section

N_CAP          = 20000          # subsample cap (pooled train+test) per dataset
SUBSAMPLE_SEED = 0
SPLIT_SEEDS    = [0, 1, 2]      # cluster->split AND random-split seeds
THRESHOLDS     = [0.5, 0.7, 0.9]
MAIN_THR       = 0.7
CONFIGS        = [(4, "LR"), (6, "LR"), (4, "RF"), (6, "RF")]  # capacity order

RESULTS = RA.RESULTS_DIR
FIG     = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings("ignore")
np.random.seed(0)

SHORT = {
    "human_nontata_promoters": "nontata_prom",
    "human_enhancers_cohn": "enh_cohn",
    "human_enhancers_ensembl": "enh_ensembl",
    "human_ocr_ensembl": "ocr_ensembl",
    "demo_human_or_worm": "human_or_worm",
    "demo_coding_vs_intergenomic_seqs": "coding_vs_interg",
    "drosophila_enhancers_stark": "droso_stark",
    "dummy_mouse_enhancers_ensembl": "dummy_mouse",
    "human_ensembl_regulatory": "ensembl_regulatory",
}


def banner(msg):
    print("\n" + "#" * 78 + "\n# " + msg + "\n" + "#" * 78, flush=True)


# --------------------------------------------------------------------------
# loading with deterministic stratified subsampling (at the file-path level)
# --------------------------------------------------------------------------
def discover_and_subsample(dset, cap, seed):
    from genomic_benchmarks.loc2seq import download_dataset
    base = download_dataset(dset, version=0)
    strata = {}
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        if not os.path.isdir(sd):
            continue
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if os.path.isdir(cd):
                strata[(split, cls)] = sorted(glob.glob(os.path.join(cd, "*.txt")))
    classes = sorted({cls for (_, cls) in strata})
    label_of = {c: i for i, c in enumerate(classes)}
    n_full = sum(len(v) for v in strata.values())
    subsampled = n_full > cap
    rng = np.random.RandomState(seed)

    seqs, labels, splits = [], [], []
    for (split, cls), paths in strata.items():
        if subsampled:
            take = min(len(paths), max(1, int(round(cap * len(paths) / n_full))))
            sel = [paths[i] for i in rng.choice(len(paths), size=take, replace=False)]
        else:
            sel = paths
        for fp in sel:
            with open(fp) as fh:
                seqs.append(fh.read().strip().upper())
            labels.append(label_of[cls])
            splits.append(split)
    df = pd.DataFrame({"seq": seqs, "label": labels, "split": splits})
    info = dict(dataset=dset, classes=classes, n_full=n_full, n_used=len(df),
                subsampled=subsampled, cap=cap, subsample_seed=seed)
    return df, info


# --------------------------------------------------------------------------
# similarity helpers (built on RA primitives)
# --------------------------------------------------------------------------
def edges_per_threshold(M, thresholds, batch=RA.BATCH):
    """One pass over M @ M.T; return {threshold: (rows, cols)} for i<j pairs."""
    sz = np.asarray(M.sum(axis=1)).ravel()
    M_T = M.T.tocsr()
    n = M.shape[0]
    acc = {t: ([], []) for t in thresholds}
    for s in range(0, n, batch):
        e = min(s + batch, n)
        inter = (M[s:e] @ M_T).toarray()
        union = sz[s:e, None] + sz[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        for t in thresholds:
            bi, bj = np.where(jac > t)
            gi = bi + s
            keep = gi < bj
            acc[t][0].append(gi[keep]); acc[t][1].append(bj[keep])
    return {t: (np.concatenate(r), np.concatenate(c)) for t, (r, c) in acc.items()}


def components_from_edges(n, rows, cols):
    g = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    g = g + g.T
    return connected_components(g, directed=False)   # (n_comp, labels)


# --------------------------------------------------------------------------
# re-splits
# --------------------------------------------------------------------------
def random_split(y, target_test_frac, seed):
    """NEGATIVE CONTROL: stratified random split (preserves class balance and
    ratio exactly), ignoring clusters -- so near-duplicates leak across it."""
    rng = np.random.RandomState(1000 + seed)
    test_mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        ntest = int(round(target_test_frac * len(idx)))
        test_mask[idx[:ntest]] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def assign_clusters_multi(labels, y, seed, target_test):
    """Generalization of RA.assign_clusters to >2 classes (used for the 3-class
    dataset only). For binary datasets we call RA.assign_clusters unchanged."""
    rng = np.random.RandomState(seed)
    clusters = collections.defaultdict(list)
    for idx, c in enumerate(labels):
        clusters[c].append(idx)
    major = {c: int(np.bincount(y[idxs]).argmax()) for c, idxs in clusters.items()}
    test_mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(y):
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
# multiclass evaluation (binary path reuses RA.eval_split unchanged)
# --------------------------------------------------------------------------
def eval_split_multi(X, y, train_idx, test_idx):
    out = {}
    for name, model in RA.make_models().items():
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])
        pred = model.classes_[np.argmax(proba, axis=1)]
        yt = y[test_idx]
        try:
            auroc = roc_auc_score(yt, proba, multi_class="ovr", average="macro")
        except Exception:
            auroc = float("nan")
        out[name] = dict(accuracy=accuracy_score(yt, pred),
                         auroc=auroc, f1=f1_score(yt, pred, average="macro"))
    return out


def eval_all_configs(X, y, tr, te, evalf):
    out = {}
    for k in (4, 6):
        res = evalf(X[k], y, tr, te)        # fits LR and RF on this k
        for m in ("LR", "RF"):
            out[(k, m)] = res[m]
    return out


# --------------------------------------------------------------------------
# per-dataset driver
# --------------------------------------------------------------------------
def run_one_dataset(dset, multiclass=False, smoke=False, cap=N_CAP):
    banner(f"DATASET: {dset}"
           + ("  [3-class]" if multiclass else "")
           + ("  [SMOKE]" if smoke else ""))
    t0 = time.time()
    df, info = discover_and_subsample(dset, cap, SUBSAMPLE_SEED)
    y = df["label"].to_numpy()
    seqs = df["seq"].tolist()
    n = len(df)
    orig_train = np.where(df["split"].to_numpy() == "train")[0]
    orig_test  = np.where(df["split"].to_numpy() == "test")[0]
    target = len(orig_test) / n
    L = df["seq"].str.len()
    Lmin, Lmed, Lmax = int(L.min()), int(L.median()), int(L.max())
    print(f"classes={info['classes']}  n_full={info['n_full']}  n_used={n}  "
          f"subsampled={info['subsampled']}  test_frac={target:.3f}  "
          f"len(min/med/max)={Lmin}/{Lmed}/{Lmax}", flush=True)
    cls_counts = df.groupby(["split", "label"]).size().to_dict()
    print("  split/label counts:", cls_counts, flush=True)

    evalf   = eval_split_multi if multiclass else RA.eval_split
    assignf = assign_clusters_multi if multiclass else RA.assign_clusters
    pos_frac = (lambda idx: float("nan")) if multiclass else (lambda idx: float(np.mean(y[idx] == 1)))

    # features (reused) + similarity matrix (reused)
    X = {k: RA.featurize(seqs, k) for k in (4, 6)}
    M = RA.kmer_binary_matrix(seqs, RA.SIM_K)

    # ---- leakage: original test -> original train ----
    max_sim = RA.max_jaccard_to_reference(M[orig_test], M[orig_train])
    leak = {t: float((max_sim > t).mean()) for t in THRESHOLDS}
    print("  leakage (orig test has train near-dup): "
          + "  ".join(f">{t}:{leak[t]:.3f}" for t in THRESHOLDS)
          + f"   median_maxsim={np.median(max_sim):.3f}", flush=True)

    plt.figure(figsize=(7, 4.2))
    plt.hist(max_sim, bins=50, range=(0, 1), color="#3b6ea5", edgecolor="white")
    for t in THRESHOLDS:
        plt.axvline(t, color="crimson", ls="--", lw=1)
    plt.xlabel(f"max {RA.SIM_K}-mer Jaccard of a test seq to any train seq")
    plt.ylabel("# test sequences")
    plt.title(f"{dset} (n={n}{', subsampled' if info['subsampled'] else ''})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"sim_hist_{SHORT[dset]}.png"), dpi=120)
    plt.close()

    edges = edges_per_threshold(M, THRESHOLDS)

    tidy, cluster_rows = [], []

    def record(split_type, seed, tr, te):
        cfg = eval_all_configs(X, y, tr, te, evalf)
        for (k, m), r in cfg.items():
            tidy.append(dict(dataset=dset, multiclass=multiclass, smoke=smoke,
                             n_full=info["n_full"], n_used=n,
                             subsampled=info["subsampled"], split_type=split_type,
                             seed=seed, k=k, model=m,
                             accuracy=r["accuracy"], auroc=r["auroc"], f1=r["f1"],
                             n_train=len(tr), n_test=len(te),
                             pos_frac_train=pos_frac(tr), pos_frac_test=pos_frac(te),
                             target_test_frac=target))
        return cfg

    # ---- original split (the optimistic baseline) ----
    cfg = record("original", -1, orig_train, orig_test)
    print("  ORIGINAL  " + "  ".join(
        f"{k}{m}={cfg[(k,m)]['accuracy']:.3f}" for (k, m) in CONFIGS), flush=True)

    # ---- negative control: random re-split ----
    for s in SPLIT_SEEDS:
        tr, te = random_split(y, target, s)
        record("random", s, tr, te)

    # ---- homology-aware single-linkage, threshold sweep ----
    comp = {}
    for t in THRESHOLDS:
        nc, lab = components_from_edges(n, *edges[t])
        comp[t] = (nc, lab)
        sizes = np.bincount(lab)
        for si, s in enumerate(SPLIT_SEEDS):
            tr, te = assignf(lab, y, s, target)
            if si == 0:
                spanning = len(np.intersect1d(np.unique(lab[tr]), np.unique(lab[te])))
                resid = RA.max_jaccard_to_reference(M[te], M[tr])
                residfrac = float((resid > t).mean())
                cluster_rows.append(dict(
                    dataset=dset, threshold=t, n_components=int(nc),
                    n_singletons=int((sizes == 1).sum()),
                    n_multi=int((sizes > 1).sum()), max_cluster=int(sizes.max()),
                    top5_sizes=str(sorted(sizes.tolist())[-5:]),
                    n_redundant=int(n - nc), frac_redundant=float((n - nc) / n),
                    clusters_spanning_split=spanning,
                    resid_leak_frac=residfrac,
                    test_frac=len(te) / n,
                    pos_train=pos_frac(tr), pos_test=pos_frac(te)))
                print(f"  hom@{t}: comps={nc} sing/multi={int((sizes==1).sum())}/"
                      f"{int((sizes>1).sum())} max={int(sizes.max())} "
                      f"redund={(n-nc)/n:.3f} | VERIFY span={spanning} "
                      f"resid>{t}={residfrac:.4f} testfrac={len(te)/n:.3f} "
                      f"pos_tr/te={pos_frac(tr):.3f}/{pos_frac(te):.3f}", flush=True)
            record(f"homology@{t}", s, tr, te)

    # ---- robustness: drop the largest connected component @ MAIN_THR ----
    nc7, lab7 = comp[MAIN_THR]
    sizes7 = np.bincount(lab7)
    largest_id, largest_size = int(sizes7.argmax()), int(sizes7.max())
    did_drop = largest_size > max(20, int(0.003 * n))
    if did_drop:
        keep = np.where(lab7 != largest_id)[0]
        sub_lab, sub_y = lab7[keep], y[keep]
        for s in SPLIT_SEEDS:
            trl, tel = assignf(sub_lab, sub_y, s, target)
            record("homology_droplargest@0.7", s, keep[trl], keep[tel])
        print(f"  drop-largest@0.7: removed largest component (size={largest_size}); "
              f"re-split remaining {len(keep)} seqs", flush=True)

    print(f"  [{dset} done in {time.time()-t0:.0f}s]", flush=True)
    meta = dict(info=info, leak=leak, length=(Lmin, Lmed, Lmax),
                target=target, largest_size=largest_size, did_drop=did_drop,
                median_maxsim=float(np.median(max_sim)))
    return tidy, cluster_rows, meta


# --------------------------------------------------------------------------
# aggregation + outputs
# --------------------------------------------------------------------------
def aggregate_and_write(all_tidy, all_clusters, all_meta, runtime):
    df = pd.DataFrame(all_tidy)
    df.to_csv(os.path.join(RESULTS, "all_results_long.csv"), index=False)
    pd.DataFrame(all_clusters).to_csv(os.path.join(RESULTS, "cluster_stats.csv"), index=False)

    def agg(dset, split_type, k, m, metric):
        sub = df[(df.dataset == dset) & (df.split_type == split_type)
                 & (df.k == k) & (df.model == m)][metric]
        return (float(sub.mean()), float(sub.std(ddof=0))) if len(sub) else (np.nan, np.nan)

    datasets = [meta["info"]["dataset"] for meta in all_meta]

    # ---- per_dataset_results.csv (wide: original / random / homology@0.7) ----
    rows = []
    for dset in datasets:
        for (k, m) in CONFIGS:
            row = {"dataset": dset, "model": m, "k": k}
            for metric in ("accuracy", "auroc", "f1"):
                o, _ = agg(dset, "original", k, m, metric)
                rmean, rstd = agg(dset, "random", k, m, metric)
                hmean, hstd = agg(dset, f"homology@{MAIN_THR}", k, m, metric)
                row[f"original_{metric}"] = round(o, 4)
                row[f"random_{metric}_mean"] = round(rmean, 4)
                row[f"homology_{metric}_mean"] = round(hmean, 4)
                row[f"homology_{metric}_std"] = round(hstd, 4)
                row[f"delta_{metric}_homology"] = round(o - hmean, 4)
                row[f"delta_{metric}_random"] = round(o - rmean, 4)
            rows.append(row)
    per_ds = pd.DataFrame(rows)
    per_ds.to_csv(os.path.join(RESULTS, "per_dataset_results.csv"), index=False)

    # ---- capacity_scaling.csv (4 homology deltas + 4 random deltas per dataset) ----
    cap_rows = []
    for dset in datasets:
        row = {"dataset": dset}
        hd = {}
        for (k, m) in CONFIGS:
            o, _ = agg(dset, "original", k, m, "accuracy")
            hmean, _ = agg(dset, f"homology@{MAIN_THR}", k, m, "accuracy")
            rmean, _ = agg(dset, "random", k, m, "accuracy")
            row[f"hom_delta_{m}_k{k}"] = round(o - hmean, 4)
            row[f"rand_delta_{m}_k{k}"] = round(o - rmean, 4)
            hd[(k, m)] = o - hmean
        order = [hd[c] for c in CONFIGS]
        row["homology_monotone_increasing"] = bool(all(order[i] <= order[i+1] + 1e-9
                                                       for i in range(len(order)-1)))
        row["hom_delta_min"] = round(min(order), 4)
        row["hom_delta_max"] = round(max(order), 4)
        cap_rows.append(row)
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(os.path.join(RESULTS, "capacity_scaling.csv"), index=False)

    # ---- threshold_sensitivity.csv ----
    ts_rows = []
    leak_by = {meta["info"]["dataset"]: meta["leak"] for meta in all_meta}
    for dset in datasets:
        for t in THRESHOLDS:
            for (k, m) in CONFIGS:
                hmean, hstd = agg(dset, f"homology@{t}", k, m, "accuracy")
                o, _ = agg(dset, "original", k, m, "accuracy")
                ts_rows.append(dict(dataset=dset, threshold=t, model=m, k=k,
                                    leakage_fraction=round(leak_by[dset][t], 4),
                                    corrected_acc_mean=round(hmean, 4),
                                    corrected_acc_std=round(hstd, 4),
                                    delta_acc=round(o - hmean, 4)))
    pd.DataFrame(ts_rows).to_csv(os.path.join(RESULTS, "threshold_sensitivity.csv"), index=False)

    # ---- summary.csv (one row per dataset, best original model) ----
    sum_rows = []
    best_model_by = {}
    for meta in all_meta:
        dset = meta["info"]["dataset"]
        best, best_acc = None, -1
        for (k, m) in CONFIGS:
            o, _ = agg(dset, "original", k, m, "accuracy")
            if o > best_acc:
                best_acc, best = o, (k, m)
        best_model_by[dset] = best
        k, m = best
        o, _ = agg(dset, "original", k, m, "accuracy")
        hmean, hstd = agg(dset, f"homology@{MAIN_THR}", k, m, "accuracy")
        rmean, rstd = agg(dset, "random", k, m, "accuracy")
        oa, _ = agg(dset, "original", k, m, "auroc")
        ha, _ = agg(dset, f"homology@{MAIN_THR}", k, m, "auroc")
        dl_mean, dl_std = agg(dset, "homology_droplargest@0.7", k, m, "accuracy")
        sum_rows.append(dict(
            dataset=dset, multiclass=meta["info"].get("multiclass", False),
            n_full=meta["info"]["n_full"], n_used=meta["info"]["n_used"],
            subsampled=meta["info"]["subsampled"],
            len_min=meta["length"][0], len_med=meta["length"][1], len_max=meta["length"][2],
            leak_0p5=round(meta["leak"][0.5], 4), leak_0p7=round(meta["leak"][0.7], 4),
            leak_0p9=round(meta["leak"][0.9], 4),
            best_model=f"{m}_k{k}",
            original_acc=round(o, 4), original_auroc=round(oa, 4),
            random_acc_mean=round(rmean, 4),
            homology_acc_mean=round(hmean, 4), homology_acc_std=round(hstd, 4),
            homology_auroc_mean=round(ha, 4),
            delta_acc_homology=round(o - hmean, 4),
            delta_acc_random=round(o - rmean, 4),
            droplargest_acc_mean=(round(dl_mean, 4) if not np.isnan(dl_mean) else ""),
            largest_cluster=meta["largest_size"]))
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)

    make_figures(df, datasets, best_model_by, all_meta)
    write_results_md(df, per_ds, cap_df, summary, all_meta, best_model_by, runtime)
    return summary, cap_df, best_model_by


def make_figures(df, datasets, best_model_by, all_meta):
    bin_dsets = [m["info"]["dataset"] for m in all_meta
                 if not m["info"].get("multiclass", False) and not m["info"].get("smoke", False)]

    def agg_acc(dset, split_type, k, m):
        sub = df[(df.dataset == dset) & (df.split_type == split_type)
                 & (df.k == k) & (df.model == m)]["accuracy"]
        return float(sub.mean()) if len(sub) else np.nan

    # (b) dose-response: accuracy DROP vs capacity, one line per dataset
    plt.figure(figsize=(8.5, 5.5))
    xs = list(range(len(CONFIGS)))
    for dset in bin_dsets:
        ys = []
        for (k, m) in CONFIGS:
            o = agg_acc(dset, "original", k, m)
            h = agg_acc(dset, f"homology@{MAIN_THR}", k, m)
            ys.append((o - h) * 100)
        plt.plot(xs, ys, marker="o", label=SHORT[dset])
    plt.axhline(0, color="gray", lw=0.8)
    plt.xticks(xs, [f"{m} k{k}" for (k, m) in CONFIGS])
    plt.xlabel("model capacity (low -> high)")
    plt.ylabel("accuracy drop under homology re-split (points)")
    plt.title("Capacity-scaling of the homology-leakage drop (per dataset)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "capacity_scaling.png"), dpi=130)
    plt.close()

    # (c) original vs random vs homology grouped bars (best model)
    plt.figure(figsize=(10, 5.5))
    w = 0.26
    xs = np.arange(len(bin_dsets))
    for i, (split, lab, col) in enumerate([("original", "original", "#4c78a8"),
                                           ("random", "random re-split", "#999999"),
                                           (f"homology@{MAIN_THR}", "homology-aware", "#e45756")]):
        vals = []
        for dset in bin_dsets:
            k, m = best_model_by[dset]
            vals.append(agg_acc(dset, split, k, m) * 100)
        plt.bar(xs + (i - 1) * w, vals, w, label=lab, color=col)
    plt.xticks(xs, [SHORT[d] for d in bin_dsets], rotation=30, ha="right", fontsize=8)
    plt.ylabel("accuracy (%)  [best original model]")
    plt.title("Original vs random re-split vs homology-aware re-split")
    plt.legend()
    plt.ylim(40, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "orig_vs_random_vs_homology.png"), dpi=130)
    plt.close()
    print("saved figures/capacity_scaling.png, orig_vs_random_vs_homology.png", flush=True)


def write_results_md(df, per_ds, cap_df, summary, all_meta, best_model_by, runtime):
    bin_sum = summary[(~summary["multiclass"]) & (summary["dataset"] != "dummy_mouse_enhancers_ensembl")]
    L = []
    L.append("# Homology-leakage audit across the Genomic Benchmarks suite\n")
    L.append(f"Reuses the validated `run_audit.py` pipeline (identical featurization, "
             f"models, exact-Jaccard similarity, whole-cluster re-split). "
             f"Subsample cap N={N_CAP} (stratified by split x class, seed={SUBSAMPLE_SEED}); "
             f"{len(SPLIT_SEEDS)} re-split seeds {SPLIT_SEEDS}; model seed={RA.MODEL_SEED}. "
             f"Total runtime {runtime:.0f}s.\n")
    L.append("Datasets larger than the cap are subsampled; for those the **leakage "
             "fraction is a lower bound** (random subsampling breaks up near-duplicate "
             "pairs), while the accuracy **deltas are unbiased within the subsample** "
             "and conservative.\n")

    L.append("## Master table (best original model per dataset)\n")
    L.append("| dataset | n_used (full) | best model | leak@0.7 | original | random | homology | Δ(hom) | Δ(rand) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in bin_sum.iterrows():
        L.append(f"| {r['dataset']} | {r['n_used']} ({r['n_full']}) | {r['best_model']} "
                 f"| {r['leak_0p7']:.3f} | {r['original_acc']:.3f} | {r['random_acc_mean']:.3f} "
                 f"| {r['homology_acc_mean']:.3f}±{r['homology_acc_std']:.3f} "
                 f"| **{r['delta_acc_homology']:+.3f}** | {r['delta_acc_random']:+.3f} |")
    L.append("")
    hd = bin_sum["delta_acc_homology"]
    rd = bin_sum["delta_acc_random"]
    L.append(f"**Cross-dataset:** homology-aware accuracy drop (best model) mean "
             f"**{hd.mean():.3f}** (range {hd.min():+.3f}..{hd.max():+.3f}); "
             f"random re-split drop mean **{rd.mean():.3f}** "
             f"(range {rd.min():+.3f}..{rd.max():+.3f}).\n")

    L.append("## Capacity scaling (accuracy drop by model, homology re-split)\n")
    L.append("| dataset | LR k4 | LR k6 | RF k4 | RF k6 | monotone? |")
    L.append("|---|---|---|---|---|---|")
    for _, r in cap_df.iterrows():
        if r["dataset"] in set(bin_sum["dataset"]):
            L.append(f"| {r['dataset']} | {r['hom_delta_LR_k4']:+.3f} | {r['hom_delta_LR_k6']:+.3f} "
                     f"| {r['hom_delta_RF_k4']:+.3f} | {r['hom_delta_RF_k6']:+.3f} "
                     f"| {'yes' if r['homology_monotone_increasing'] else 'no'} |")
    L.append("")
    L.append("For contrast, the **random re-split** deltas (negative control):\n")
    L.append("| dataset | LR k4 | LR k6 | RF k4 | RF k6 |")
    L.append("|---|---|---|---|---|")
    for _, r in cap_df.iterrows():
        if r["dataset"] in set(bin_sum["dataset"]):
            L.append(f"| {r['dataset']} | {r['rand_delta_LR_k4']:+.3f} | {r['rand_delta_LR_k6']:+.3f} "
                     f"| {r['rand_delta_RF_k4']:+.3f} | {r['rand_delta_RF_k6']:+.3f} |")
    L.append("")
    with open(os.path.join(RESULTS, "results_suite.md"), "w") as fh:
        fh.write("\n".join(L))
    print("saved results/results_suite.md", flush=True)


def main():
    t0 = time.time()
    print(f"seeds: subsample={SUBSAMPLE_SEED} split={SPLIT_SEEDS} model={RA.MODEL_SEED} "
          f"| N_CAP={N_CAP} | thresholds={THRESHOLDS}", flush=True)
    all_tidy, all_clusters, all_meta = [], [], []

    for dset in BINARY_DSETS:
        tidy, clusters, meta = run_one_dataset(dset, multiclass=False)
        meta["info"]["multiclass"] = False; meta["info"]["smoke"] = False
        all_tidy += tidy; all_clusters += clusters; all_meta.append(meta)

    for dset in SMOKE_DSETS:
        try:
            tidy, clusters, meta = run_one_dataset(dset, multiclass=False, smoke=True)
            meta["info"]["multiclass"] = False; meta["info"]["smoke"] = True
            all_tidy += tidy; all_clusters += clusters; all_meta.append(meta)
        except Exception as e:
            print(f"SMOKE {dset} FAILED: {type(e).__name__}: {e}", flush=True)

    for dset in MULTICLASS_DSETS:
        try:
            tidy, clusters, meta = run_one_dataset(dset, multiclass=True)
            meta["info"]["multiclass"] = True; meta["info"]["smoke"] = False
            all_tidy += tidy; all_clusters += clusters; all_meta.append(meta)
        except Exception as e:
            print(f"MULTICLASS {dset} FAILED (optional, non-blocking): "
                  f"{type(e).__name__}: {e}", flush=True)

    runtime = time.time() - t0
    summary, cap_df, best = aggregate_and_write(all_tidy, all_clusters, all_meta, runtime)
    banner(f"SUITE DONE ({runtime:.0f}s)")
    print(summary.to_string(), flush=True)
    print("\nAll outputs in:", RESULTS, flush=True)


if __name__ == "__main__":
    main()
