#!/usr/bin/env python3
"""
exp_datacache_manifest.py -- pin the exact bytes this audit was computed on.

WHY
---
`genomic_benchmarks` 1.0.0 re-extracts its archives on every call, so "we used the
Genomic Benchmarks datasets" does not identify a fixed input: a later release, or a
re-extraction that orders rows differently, silently changes what the report card
describes. The manuscript already names the package version; what it cannot currently
do is let a reader confirm they hold the same bytes.

This emits a checksum manifest of the local extraction cache. Two digests per cached
combination:

  sha256_file     the pickle on disk. Fast, but not portable -- pickle framing and
                  pandas' own version can change it without the DATA changing.
  sha256_content  a canonical digest over the data itself: for every row, in loader
                  order, the split, the label and the sequence, hashed as UTF-8 with
                  explicit separators. This is the one to quote and to compare
                  across machines, because it is invariant to pickle protocol,
                  pandas version and column dtype.

Both are recorded so a mismatch can be diagnosed rather than merely detected: file
digests differing while content digests agree means a serialization change, not a
data change.

Output
------
results/datacache_manifest.csv   one row per cached (dataset, cap, seed)
"""
from __future__ import annotations
import hashlib
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import pandas as pd
from audit.core import expkit as E
from audit.tools.prefetch import COMBOS

CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_content(df):
    """Canonical digest over the data, invariant to serialization details."""
    h = hashlib.sha256()
    for split, label, seq in zip(df["split"], df["label"], df["seq"]):
        h.update(f"{split}\t{int(label)}\t{seq}\n".encode("utf-8"))
    return h.hexdigest()


if __name__ == "__main__":
    print(_R.describe(), flush=True)
    rows, t0 = [], time.time()
    for dset, cap, seed in COMBOS:
        path = os.path.join(E.DATACACHE, f"{dset}__cap{cap}__seed{seed}.pkl")
        if not os.path.exists(path):
            print(f"[{dset} cap={cap}] MISSING {path} -- run audit.tools.prefetch first",
                  flush=True)
            continue
        df = pd.read_pickle(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        rows.append(dict(
            dataset=dset, cap=cap, seed=seed,
            n_rows=len(df),
            n_train=int((df["split"] == "train").sum()),
            n_test=int((df["split"] == "test").sum()),
            n_classes=int(df["label"].nunique()),
            pkl_bytes=os.path.getsize(path),
            sha256_file=sha256_file(path),
            sha256_content=sha256_content(df),
            extracted_utc=mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            path=os.path.relpath(path, os.path.dirname(E.RESULTS)),
        ))
        r = rows[-1]
        print(f"[{dset} cap={cap}] n={r['n_rows']:,} content={r['sha256_content'][:16]}... "
              f"extracted {r['extracted_utc']}  ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(E.RESULTS, "datacache_manifest.csv"), index=False)
    print(f"\nwrote results/datacache_manifest.csv ({len(out)} rows)")
    if len(out):
        print(f"extraction dates span {out.extracted_utc.min()} .. {out.extracted_utc.max()}")
    print("EXP_DATACACHE_MANIFEST_DONE", round(time.time() - t0), "s")
