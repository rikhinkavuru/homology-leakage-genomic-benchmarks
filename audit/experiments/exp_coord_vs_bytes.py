#!/usr/bin/env python3
"""D8: derive, in code, the coordinate-space prediction of the byte-identical
test-copy count on human_enhancers_ensembl, and emit it to a CSV.

The manuscript compares two counts measured on two independent code paths:

  observed  the number of TEST sequences whose exact byte string also occurs in
            TRAIN (results/exact_dup_counts.csv, no coordinates involved), and
  predicted the number of TEST positive intervals whose shipped coordinate
            (region, start, end, strand) also occurs among TRAIN positive
            intervals (this module, no sequences involved).

We report the realised coordinate count -- which is a direct count on the shipped
split, not an estimate -- and, separately, the expectation of that count under a
class-stratified uniform split, so that the arithmetic behind it is visible.

Writes results/coord_vs_bytes.csv.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IV = os.path.join(HERE, "datacache", "intervals")
DSET = "human_enhancers_ensembl"
KEY = ["region", "start", "end", "strand"]


def load(split, cls):
    return pd.read_csv(os.path.join(IV, DSET, f"{split}_{cls}.csv.gz"))


rows = []
for cls in ("positive", "negative"):
    tr, te = load("train", cls), load("test", cls)
    n_raw = len(tr) + len(te)
    allc = pd.concat([tr, te], ignore_index=True)
    mult = allc.groupby(KEY).size()
    n_distinct = len(mult)
    n_mult2 = int((mult == 2).sum())
    n_mult1 = int((mult == 1).sum())

    # realised: test rows whose coordinate also appears in train
    trkeys = set(map(tuple, tr[KEY].itertuples(index=False, name=None)))
    realised = sum(
        1 for t in te[KEY].itertuples(index=False, name=None) if tuple(t) in trkeys)

    # expectation under a class-stratified uniform split, hypergeometric over
    # the multiplicity-2 coordinates: a pair contributes exactly one such test
    # row iff exactly one of its two rows lands in test.
    n_te, n_tr = len(te), len(tr)
    p_one = 2.0 * (n_te / n_raw) * (n_tr / (n_raw - 1))
    expected = n_mult2 * p_one

    rows.append(dict(
        dataset=DSET, cls=cls, n_rows=n_raw, n_train=n_tr, n_test=n_te,
        n_distinct_coords=n_distinct, n_coords_mult1=n_mult1,
        n_coords_mult2=n_mult2,
        frac_rows_on_dup_coord=round(1 - n_mult1 / n_raw, 4),
        p_exactly_one_in_test=round(p_one, 6),
        expected_test_coord_in_train=round(expected, 1),
        realised_test_coord_in_train=realised))

df = pd.DataFrame(rows)

# pooled over classes: this is what the byte-identical census counts, since that
# census does not know which class a duplicate belongs to.
obs = pd.read_csv(os.path.join(HERE, "results", "exact_dup_counts.csv"))
obs = obs[obs.dataset == DSET].iloc[0]
pooled_real = int(df.realised_test_coord_in_train.sum())
pooled_exp = float(df.expected_test_coord_in_train.sum())
df.loc[len(df)] = dict(
    dataset=DSET, cls="both", n_rows=int(df.n_rows.sum()),
    n_train=int(df.n_train.sum()), n_test=int(df.n_test.sum()),
    n_distinct_coords=int(df.n_distinct_coords.sum()),
    n_coords_mult1=int(df.n_coords_mult1.sum()),
    n_coords_mult2=int(df.n_coords_mult2.sum()),
    frac_rows_on_dup_coord=None, p_exactly_one_in_test=None,
    expected_test_coord_in_train=round(pooled_exp, 1),
    realised_test_coord_in_train=pooled_real)
df["observed_byte_identical_test"] = int(obs.exact_dup)
df["rel_gap_realised_vs_observed"] = (
    (df.realised_test_coord_in_train - int(obs.exact_dup)) / int(obs.exact_dup)).round(4)

out = os.path.join(HERE, "results", "coord_vs_bytes.csv")
df.to_csv(out + ".tmp", index=False)
os.replace(out + ".tmp", out)
print(df.to_string(index=False))
print(f"\nwrote {out}")
