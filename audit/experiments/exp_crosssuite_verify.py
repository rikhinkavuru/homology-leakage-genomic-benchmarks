#!/usr/bin/env python3
"""
Independent verification of the cross-suite census headline numbers.

The census (exp_crosssuite_census.py) reports leak fractions via the 8-mer Jaccard /
containment estimator. This module re-derives the LENGTH-INDEPENDENT part of that
result -- exact byte-identical duplication -- with a completely separate code path
(plain set membership on raw strings, no k-mers, no similarity), so the flagship
"25% byte-identical" figure does not rest on the estimator it is meant to justify.

WHY THIS MATTERS FOR THE ORIGINAL-vs-REVISED CONTRAST
-----------------------------------------------------
Sequence lengths differ between the two NT releases on most tasks, and 8-mer Jaccard
falls monotonically as sequences lengthen, so a naive reading of the Jaccard drop is
confounded by length. Exact duplication is immune to that: a byte-identical match is
byte-identical at any length. Reporting exact-duplicate counts alongside the length
distribution is what lets the curation claim survive the confound.

Run:  python -m audit.experiments.exp_crosssuite_verify
Out:  results/crosssuite_exact_verification.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES, SHARED_TASKS

R = os.path.join(HERE, "results")


def verify(suite, repo, task):
    try:
        tr, try_ = read_fna(fetch(repo, task, "train"))
        te, tey = read_fna(fetch(repo, task, "test"))
    except Exception as ex:                                    # noqa: BLE001
        print(f"  [{suite}/{task}] unavailable: {type(ex).__name__}", flush=True)
        return None
    tr_set = set(tr)
    hits = [s for s in te if s in tr_set]
    # label concordance on the exact duplicates: does the copy carry the same label?
    lab = {}
    for s, l in zip(tr, try_):
        lab.setdefault(s, set()).add(str(l))
    conc = [str(l) in lab[s] for s, l in zip(te, tey) if s in lab]
    Ltr = np.array([len(s) for s in tr])
    Lte = np.array([len(s) for s in te])
    row = dict(
        suite=suite, task=task,
        n_train=len(tr), n_train_unique=len(tr_set),
        train_internal_dup_rows=len(tr) - len(tr_set),
        n_test=len(te), n_test_unique=len(set(te)),
        exact_test_in_train_n=len(hits),
        exact_test_in_train_frac=round(len(hits) / len(te), 4) if te else 0.0,
        label_concordance_on_exact_dups=(round(float(np.mean(conc)), 4) if conc else None),
        train_len_min=int(Ltr.min()), train_len_med=int(np.median(Ltr)),
        train_len_max=int(Ltr.max()),
        test_len_min=int(Lte.min()), test_len_med=int(np.median(Lte)),
        test_len_max=int(Lte.max()))
    print(f"  [{suite}/{task}] exact {row['exact_test_in_train_n']}/{row['n_test']} "
          f"= {row['exact_test_in_train_frac']:.4f}  concordance="
          f"{row['label_concordance_on_exact_dups']}  test_len_med={row['test_len_med']}",
          flush=True)
    return row


def main():
    os.makedirs(R, exist_ok=True)
    rows = []
    for suite, repo in SUITES.items():
        print(f"=== {suite} ===", flush=True)
        for task in SHARED_TASKS:
            r = verify(suite, repo, task)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    out = f"{R}/crosssuite_exact_verification.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)

    print("\n== exact duplication (length-independent) and length distributions ==")
    o = df[df.suite == "NT-original"].set_index("task")
    r = df[df.suite == "NT-revised"].set_index("task")
    cmp = pd.DataFrame({
        "exact_orig": o.exact_test_in_train_frac, "exact_rev": r.exact_test_in_train_frac,
        "testlen_orig": o.test_len_med, "testlen_rev": r.test_len_med,
        "trainlen_orig": o.train_len_med, "trainlen_rev": r.train_len_med})
    cmp["length_changed"] = cmp.testlen_orig != cmp.testlen_rev
    print(cmp.to_string())
    same_len = cmp[~cmp.length_changed]
    print(f"\ntasks where TEST length is unchanged between releases: {len(same_len)}/{len(cmp)}")
    if len(same_len):
        print("  exact-duplication change on those length-matched tasks:")
        print(same_len[["exact_orig", "exact_rev"]].to_string())
    print(f"\nEXP_CROSSSUITE_VERIFY_DONE -> {out}")


if __name__ == "__main__":
    main()
