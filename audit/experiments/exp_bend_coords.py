"""
exp_bend_coords.py
==================
Out-of-suite validation of the coordinate-space construction signature (S:m-coord)
on BEND (Marin et al. 2024), which ships BED files carrying their own split column.

Why this experiment exists
--------------------------
Every dataset audited in sequence space in this paper has a median length at or
below 2,142 bp. BEND's enhancer-annotation task uses 100,096 bp inputs, two orders
of magnitude longer. Because the coordinate screen needs only the shipped intervals
and no reference genome, we can run it there at zero marginal cost and obtain a
long-range datapoint that the k-mer census cannot supply.

Statistic (identical to exp_construction.cross_split_overlap):
  share of TEST intervals overlapping ANY TRAIN interval at >= t reciprocal
  overlap, reciprocal overlap = |intersection| / max(len_test, len_train).

Cost: ~162 MB of BED download, ~6 min single-core. No GPU, no reference genome.

Usage:
    python -m audit.experiments.exp_bend_coords --data-dir <dir> [--download]
Writes results/bend_coordinate_census.csv
"""

import argparse
import collections
import os

import numpy as np
import pandas as pd

BIN = 100_000
ERDA = "https://sid.erda.dk/share_redirect/f6hdp1zTzh/data"
TASKS = {
    "chromatin_accessibility": "chromatin_accessibility/chromatin_accessibility.bed",
    "histone_modification": "histone_modification/histone_modification.bed",
    "gene_finding": "gene_finding/gene_finding.bed",
    "enhancer_annotation": "enhancer_annotation/enhancer_annotation.bed",
}


def _index(df):
    idx = collections.defaultdict(list)
    s = df["start"].to_numpy()
    e = df["end"].to_numpy()
    reg = df["chromosome"].to_numpy()
    for i in range(len(df)):
        for b in range(s[i] // BIN, e[i] // BIN + 1):
            idx[(reg[i], b)].append(i)
    return idx, s, e


def cross_split_overlap(test_df, train_df):
    """Mirror of exp_construction.cross_split_overlap on BED column names."""
    idx, s_tr, e_tr = _index(train_df)
    s = test_df["start"].to_numpy()
    e = test_df["end"].to_numpy()
    reg = test_df["chromosome"].to_numpy()
    best = np.zeros(len(test_df))
    for i in range(len(test_df)):
        cands = set()
        for b in range(s[i] // BIN, e[i] // BIN + 1):
            cands.update(idx.get((reg[i], b), ()))
        if not cands:
            continue
        c = np.fromiter(cands, int, len(cands))
        ov = np.maximum(np.minimum(e[i], e_tr[c]) - np.maximum(s[i], s_tr[c]), 0)
        rec = ov / np.maximum(np.maximum(e[i] - s[i], e_tr[c] - s_tr[c]), 1)
        best[i] = rec.max()
    return dict(
        xsplit_ge1bp=float((best > 0).mean()),
        xsplit_ge50pct=float((best >= 0.5).mean()),
        xsplit_ge90pct=float((best >= 0.9).mean()),
        xsplit_exact=float((best >= 1.0).mean()),
    )


def download(data_dir):
    import urllib.request

    for task, rel in TASKS.items():
        dst = os.path.join(data_dir, f"{task}.bed")
        if os.path.exists(dst):
            continue
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{ERDA}/{rel}", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datacache/bend")
    ap.add_argument("--out", default="results/bend_coordinate_census.csv")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    if args.download:
        download(args.data_dir)

    rows = []
    for task in TASKS:
        path = os.path.join(args.data_dir, f"{task}.bed")
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df = df[["chromosome", "start", "end", "split"]]
        lengths = df["end"] - df["start"]
        splits = set(df["split"].unique())

        if {"train", "test"} <= splits:
            # train/valid/test releases: test against train
            tr, te = df[df.split == "train"], df[df.split == "test"]
            res = cross_split_overlap(te, tr)
            partition = ("identity-80pct" if set(tr.chromosome) & set(te.chromosome)
                         else "chromosome")
            folds = 1
        else:
            # 10-fold cross-validation release: worst fold
            per = {p: cross_split_overlap(df[df.split == p], df[df.split != p])
                   for p in sorted(splits)}
            res = {k: max(v[k] for v in per.values()) for k in next(iter(per.values()))}
            chrom_sets = [set(df[df.split == p].chromosome) for p in sorted(splits)]
            disjoint = all(not (a & b) for i, a in enumerate(chrom_sets)
                           for b in chrom_sets[i + 1:])
            partition = "chromosome" if disjoint else "unknown"
            folds = len(splits)

        rows.append(dict(
            suite="BEND", task=task, partition=partition, folds=folds,
            n_train=int((df.split == "train").sum()) or len(df) - int(len(df) / folds),
            n_test=int((df.split == "test").sum()) or None,
            n_total=len(df),
            len_med=int(lengths.median()), len_max=int(lengths.max()),
            **{k: round(v, 4) for k, v in res.items()},
            coord_verdict="LEAKY" if res["xsplit_ge50pct"] > 0.1 else "clean",
        ))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
