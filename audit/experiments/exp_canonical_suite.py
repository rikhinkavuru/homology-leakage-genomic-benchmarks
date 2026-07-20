#!/usr/bin/env python3
"""
exp_canonical_suite.py -- run the reverse-complement-collapsed (canonical) census
across the WHOLE suite, not only the two leaky datasets.

WHY
---
Every "clean" verdict in the report card is currently forward-strand-only: canonical
k-mers appear as a robustness arm on `human_nontata_promoters` and
`human_enhancers_ensembl` and nowhere else (`results/exp_canonical_leak.csv` has two
rows). A clean verdict that has never been measured strand-symmetrically is a weaker
statement than the table implies, because DNA is double-stranded and a
reverse-complemented near-duplicate is invisible to a forward-only k-mer set.

This measures, for all eight datasets and under both the Jaccard and the length-robust
containment index, the leak fraction with forward-strand k-mers and with canonical
(min of k-mer and its reverse complement) k-mers, and records whether the 0.1 verdict
cut moves on any dataset.

`exp_canonical.py` is left alone: it additionally refits the four-model comparison on
canonical FEATURES for the two leaky sets, which is a different question and is much
more expensive. This script is census-only.

Output
------
results/canonical_census_suite.csv   one row per dataset x metric
"""
from __future__ import annotations
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import numpy as np
import pandas as pd
from audit.core import expkit as E

CUT = 0.1          # the report card's verdict threshold
THRESHOLDS = (0.7, 0.9)


def census(dset):
    seqs, y, otr, ote, _ = E.load(dset)
    out = []
    for mode in ("jaccard", "containment"):
        fwd = E.cached_max_sim(dset, seqs, otr, ote, 8, mode=mode, canonical=False)
        can = E.cached_max_sim(dset, seqs, otr, ote, 8, mode=mode, canonical=True)
        for t in THRESHOLDS:
            lf = float((fwd > t).mean())
            lc = float((can > t).mean())
            out.append(dict(
                dataset=dset, n_test=len(ote), n_train=len(otr), metric=mode,
                threshold=t,
                leak_fwd=round(lf, 4), leak_canon=round(lc, 4),
                delta=round(lc - lf, 4),
                rel_increase=round((lc - lf) / lf, 4) if lf > 0 else np.nan,
                verdict_fwd="LEAKY" if lf > CUT else "clean",
                verdict_canon="LEAKY" if lc > CUT else "clean",
                verdict_moved=int((lf > CUT) != (lc > CUT)),
            ))
    return out


if __name__ == "__main__":
    todo = sys.argv[1:] or (list(E.LEAKY) + list(E.CLEAN) + [E.THREECLASS])
    print(_R.describe(), flush=True)
    rows, t0 = [], time.time()
    for d in todo:
        rows.extend(census(d))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(E.RESULTS, "canonical_census_suite.csv"), index=False)
        r = df[(df.dataset == d) & (df.metric == "jaccard") & (df.threshold == 0.7)].iloc[0]
        print(f"[{d}] jaccard@0.7 fwd={r.leak_fwd} canon={r.leak_canon} "
              f"({r.verdict_fwd}->{r.verdict_canon})  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    moved = df[df.verdict_moved == 1]
    print("\n" + df.to_string(index=False))
    print(f"\nverdicts moved under canonicalization: {len(moved)}")
    if len(moved):
        print(moved.to_string(index=False))
    print("EXP_CANONICAL_SUITE_DONE", round(time.time() - t0), "s")
