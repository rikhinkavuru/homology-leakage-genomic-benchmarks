#!/usr/bin/env python3
"""
chromosome_holdout.py -- leave-one-chromosome-out (LOCO-style) control for the
two leaky datasets (human_enhancers_ensembl, human_nontata_promoters).

WHAT IT DOES
------------
1. Recovers each cached sequence's chromosome from the Genomic Benchmarks
   interval CSVs shipped in the source repo
       raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/
           datasets/<dataset>/<split>/<class>.csv.gz   (cols: id,region,start,end,strand)
   The `.txt` sequence files that genomic_benchmarks writes are named `<id>.txt`
   (loc2seq.download_dataset), so the filename stem IS the interval-CSV `id`.
   run_suite.discover_and_subsample() reads them in a fixed deterministic order
   (split in [train,test]; class in sorted(listdir); files in sorted(glob)), which
   is exactly the row order of the cached pickle that expkit.load() returns. We
   reproduce that order and join id -> region to get chromosome per row.
   VERIFICATION (no reference genome needed): (a) per-(split,class) file-id set ==
   interval-CSV id set; (b) total rows and per-split/label counts match the cached
   dataset exactly; (c) a sample of cached sequences is re-read and confirmed to
   align row-for-row with the recovered coordinates. (A full 0/154842 and 0/36131
   exact-seq realignment was verified during development.)

2. Builds a deterministic grouped-by-chromosome train/test split approximating
   the original test_frac: whole chromosomes are moved to the test side, smallest
   first (deterministic, ties broken by name), stopping at whichever prefix lands
   closest to test_frac. No chromosome is shared between train and test.

3. Refits LR/LinearSVC/RF/HGB at k=6 via expkit on (a) the ORIGINAL split and
   (b) the chromosome-holdout split; records per-model acc/auroc/f1, each model's
   rank-by-accuracy per split, and RF's rank.

4. Reports residual near-duplicate leakage that SURVIVES the chromosome split
   (E.max_sim_to_train test-vs-train at k=8, fraction > 0.7). A chromosome split
   does NOT remove INTRA-chromosome near-duplicates, so this quantifies what
   remains -- contrast with the homology-aware split's 0.0 (zero by construction).

Determinism / reproducibility notes:
  * expkit's canonical .predict() decision rule is used throughout.
  * RF results are bit-identical regardless of n_jobs (fixed random_state); we cap
    n_jobs only to bound peak memory / avoid swap on the full-scale ensembl fit.
    This changes NO number, only wall-time.
  * A small on-disk cache (datacache/cho_resume_cache.json) makes the run resumable
    cell-by-cell so a long full-scale run can be relaunched without recomputation.

Outputs: results/chromosome_holdout.csv (written incrementally). Creates only new
files; modifies nothing frozen.
"""
from __future__ import annotations
import os, glob, sys, json, gc, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit.core import expkit as E

GBROOT   = os.path.join(os.path.expanduser("~"), ".genomic_benchmarks")
IV_BASE  = "https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/datasets"
IV_CACHE = os.path.join(HERE, "datacache", "intervals")
RESUME   = os.path.join(HERE, "datacache", "cho_resume_cache.json")
LEAKY    = ["human_enhancers_ensembl", "human_nontata_promoters"]
K        = 6
SIM_K    = 8
LEAK_THRESHOLD = 0.7
RF_NJOBS = 3                  # result-invariant; bounds memory on full-scale ensembl
RESULTS  = os.path.join(HERE, "results")
OUTCSV   = os.path.join(RESULTS, "chromosome_holdout.csv")
VERIFY_SAMPLE = 1500
MODEL_ORDER = ("LR", "LinearSVC", "RF", "HGB")

# ---------------------------------------------------------------------------
# resumable cache
# ---------------------------------------------------------------------------
def _cache_load():
    if os.path.exists(RESUME):
        try:
            with open(RESUME) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _cache_save(c):
    os.makedirs(os.path.dirname(RESUME), exist_ok=True)
    tmp = RESUME + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, RESUME)

# ---------------------------------------------------------------------------
# coordinate recovery
# ---------------------------------------------------------------------------
def _fetch_interval_csvs(dset):
    """Download the per-(split,class) interval CSVs into datacache/intervals (cached)."""
    dd = os.path.join(IV_CACHE, dset)
    os.makedirs(dd, exist_ok=True)
    out = {}
    base = os.path.join(GBROOT, dset)
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        if not os.path.isdir(sd):
            continue
        for cls in sorted(os.listdir(sd)):
            if not os.path.isdir(os.path.join(sd, cls)):
                continue
            path = os.path.join(dd, f"{split}_{cls}.csv.gz")
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"{IV_BASE}/{dset}/{split}/{cls}.csv.gz", path)
            df = pd.read_csv(path, compression="gzip")
            df["id"] = df["id"].astype(str)
            out[(split, cls)] = df
    return out


def recover_chrom(dset, seqs, y, otr, ote):
    """Return (chrom array aligned to expkit.load order, verification dict)."""
    base = os.path.join(GBROOT, dset)
    csv = _fetch_interval_csvs(dset)
    region_map = {k: dict(zip(v["id"], v["region"].astype(str))) for k, v in csv.items()}
    idset = {k: set(v["id"]) for k, v in csv.items()}
    classdirs = set()
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        if os.path.isdir(sd):
            classdirs |= {c for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c))}
    classes = sorted(classdirs)                     # e.g. ['negative','positive']
    label_of = {c: i for i, c in enumerate(classes)}

    chrom, row_split, row_label, row_path = [], [], [], []
    idset_ok = True
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd):
                continue
            paths = sorted(glob.glob(os.path.join(cd, "*.txt")))
            rmap = region_map[(split, cls)]
            if {os.path.splitext(os.path.basename(p))[0] for p in paths} != idset[(split, cls)]:
                idset_ok = False
            for p in paths:
                _id = os.path.splitext(os.path.basename(p))[0]
                chrom.append(rmap[_id])
                row_split.append(split)
                row_label.append(label_of[cls])
                row_path.append(p)
    chrom = np.asarray(chrom)
    row_split = np.asarray(row_split)
    row_label = np.asarray(row_label, dtype=int)

    n_ok = (len(chrom) == len(seqs))
    lbl_ok = bool(np.array_equal(row_label, np.asarray(y, dtype=int))) if n_ok else False
    otr_set = set(np.asarray(otr).tolist())
    split_ok = n_ok and all((row_split[i] == "train") == (i in otr_set)
                            for i in range(len(seqs)))
    seq_mism, n_sample = 0, 0
    if n_ok:
        idx = np.unique(np.linspace(0, len(seqs) - 1,
                                    min(VERIFY_SAMPLE, len(seqs))).astype(int))
        n_sample = len(idx)
        for i in idx:
            with open(row_path[i]) as fh:
                if fh.read().strip().upper() != seqs[i]:
                    seq_mism += 1
    ver = dict(n_rows=len(chrom), n_seqs=len(seqs), rowcount_ok=bool(n_ok),
               idset_ok=bool(idset_ok), label_ok=bool(lbl_ok), split_ok=bool(split_ok),
               sampled=int(n_sample), sample_seq_mismatch=int(seq_mism),
               n_chrom=int(len(np.unique(chrom))))
    return chrom, ver


# ---------------------------------------------------------------------------
# chromosome-holdout split
# ---------------------------------------------------------------------------
def chromosome_holdout_split(chrom, test_frac):
    """Deterministic whole-chromosome test holdout ~ test_frac. Move chromosomes to
    test smallest-first (ties by name); keep the prefix whose cumulative fraction is
    closest to test_frac. Returns (tr_idx, te_idx, held_out_chroms)."""
    chrom = np.asarray(chrom)
    n = len(chrom)
    sizes = pd.Series(chrom).value_counts()
    order = sorted(sizes.index, key=lambda c: (int(sizes[c]), str(c)))
    cum, prefix, best_prefix, best_diff = 0, [], [], test_frac * n
    for c in order:
        prefix = prefix + [c]
        cum += int(sizes[c])
        diff = abs(cum - test_frac * n)
        if diff < best_diff:
            best_diff, best_prefix = diff, list(prefix)
    held = sorted(set(best_prefix), key=lambda c: (int(sizes[c]), str(c)))
    mask = np.isin(chrom, list(held))
    return np.where(~mask)[0], np.where(mask)[0], held


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def rank_models(metric_by_model, order=MODEL_ORDER):
    def key(m):
        d = metric_by_model[m]
        au = d["auroc"] if d["auroc"] == d["auroc"] else -1.0
        return (-d["acc"], -au, order.index(m))
    ranked = sorted(metric_by_model.keys(), key=key)
    return {m: i + 1 for i, m in enumerate(ranked)}, ">".join(ranked)


# ---------------------------------------------------------------------------
# per-dataset run (resumable)
# ---------------------------------------------------------------------------
def run_dataset(dset, cache):
    print(f"\n===== {dset} =====", flush=True)
    seqs, y, otr, ote, test_frac = E.load(dset)
    chrom, ver = recover_chrom(dset, seqs, y, otr, ote)
    print("  recovery verification:", ver, flush=True)
    assert ver["rowcount_ok"] and ver["idset_ok"] and ver["label_ok"] and ver["split_ok"], \
        "coordinate recovery failed verification"
    assert ver["sample_seq_mismatch"] == 0, "sampled sequence realignment mismatch"

    tr_c, te_c, held = chromosome_holdout_split(chrom, test_frac)
    ytr_c, yte_c = y[tr_c], y[te_c]
    assert len(np.unique(ytr_c)) > 1 and len(np.unique(yte_c)) > 1, "degenerate class split"
    print(f"  held-out chroms ({len(held)}): {held}", flush=True)
    print(f"  orig test_frac={test_frac:.4f}  chrom-holdout realized={len(te_c)/len(seqs):.4f} "
          f"(train={len(tr_c)} test={len(te_c)})", flush=True)

    splits = {"original": (np.asarray(otr), np.asarray(ote)),
              "chromosome_holdout": (tr_c, te_c)}

    def rkey(s):  return f"{dset}|resid|{s}"
    def mkey(s, m): return f"{dset}|metric|{s}|{m}"

    need_metric = any(mkey(s, m) not in cache for s in splits for m in MODEL_ORDER)
    X = None
    if need_metric:
        X = E.featurize(seqs, K)
        print(f"  featurized k={K}: X={X.shape}", flush=True)
        models = E.models()
        if hasattr(models["RF"], "n_jobs"):
            models["RF"].n_jobs = RF_NJOBS

    # residuals (cached per split)
    for s, (tr, te) in splits.items():
        if rkey(s) not in cache:
            sim = E.max_sim_to_train(seqs, tr, te, SIM_K)
            cache[rkey(s)] = dict(frac=float((sim > LEAK_THRESHOLD).mean()),
                                  maxsim=float(sim.max()), n_test=int(len(te)))
            _cache_save(cache)
            print(f"  residual leak [{s}] frac>{LEAK_THRESHOLD}={cache[rkey(s)]['frac']:.4f} "
                  f"maxsim={cache[rkey(s)]['maxsim']:.3f}", flush=True)

    # metrics (cached per split,model)
    for s, (tr, te) in splits.items():
        for m in MODEL_ORDER:
            if mkey(s, m) not in cache:
                r = E.metrics(models[m], X, y, tr, te)
                cache[mkey(s, m)] = dict(acc=r["acc"], auroc=r["auroc"], f1=r["f1"])
                _cache_save(cache)
                print(f"  [{s}] {m}: acc={r['acc']:.4f} auroc={r['auroc']:.4f} "
                      f"f1={r['f1']:.4f}", flush=True)
    if X is not None:
        del X; gc.collect()

    # assemble rows from cache
    rows = []
    for s, (tr, te) in splits.items():
        mbm = {m: cache[mkey(s, m)] for m in MODEL_ORDER}
        rank, rstr = rank_models(mbm)
        res = cache[rkey(s)]
        for m in MODEL_ORDER:
            d = mbm[m]
            rows.append({
                "dataset": dset, "split": s, "k": K, "model": m,
                "acc": round(d["acc"], 4), "auroc": round(d["auroc"], 4),
                "f1": round(d["f1"], 4),
                "rank_by_acc": rank[m], "rf_rank": rank["RF"], "ranking_acc": rstr,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "orig_test_frac": round(test_frac, 4),
                "realized_test_frac": round(len(te) / len(seqs), 4),
                "held_out_chroms": ("|".join(held) if s == "chromosome_holdout" else ""),
                "leak_threshold": LEAK_THRESHOLD,
                "resid_leak_frac": round(res["frac"], 4),
                "resid_max_sim": round(res["maxsim"], 3),
            })
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cache = _cache_load()
    all_rows = []
    for dset in LEAKY:
        all_rows.extend(run_dataset(dset, cache))
        # write CSV incrementally after each dataset completes
        pd.DataFrame(all_rows).to_csv(OUTCSV, index=False)
        print(f"  -> wrote partial {OUTCSV} ({len(all_rows)} rows)", flush=True)
    print(f"\nwrote {OUTCSV} ({len(all_rows)} rows) [ALL DATASETS COMPLETE]")
    df = pd.DataFrame(all_rows)
    for dset in LEAKY:
        for s in ("original", "chromosome_holdout"):
            sub = df[(df.dataset == dset) & (df.split == s)].iloc[0]
            print(f"{dset} [{s}] ranking={sub.ranking_acc} RF_rank={sub.rf_rank} "
                  f"resid>{LEAK_THRESHOLD}={sub['resid_leak_frac']}")


if __name__ == "__main__":
    main()
