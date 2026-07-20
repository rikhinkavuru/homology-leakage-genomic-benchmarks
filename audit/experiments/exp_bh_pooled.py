#!/usr/bin/env python3
"""
exp_bh_pooled.py -- E3: is the Benjamini-Hochberg conclusion an artifact of where the
family boundary was drawn?

THE OBJECTION
-------------
The manuscript corrects two grids separately: a declared family of 15 leaky-versus-clean
deltas (results/bh_correction_frozen.csv), and the cross-suite ranking grid
(results/crosssuite_ranking.csv), each at q=0.05. Two families corrected separately is
always more permissive than one family of their union, and a reader cannot tell from the
text whether the boundary was chosen before or after seeing which deltas survived. The
membership rule is now stated explicitly in the paper; this supplies the sensitivity
analysis that makes the statement checkable, by re-running BH over the POOLED union and
reporting what changes.

Pooling is NOT a one-sided stress test, though it is tempting to assume so, and this run
disproves the assumption on its own data. Benjamini-Hochberg is a step-up procedure that
rejects the k smallest p-values satisfying p_(i) <= q*i/m. Enlarging the family raises m,
which is restrictive, but it also raises the RANK i of any given p-value when the
incoming tests are more significant than the ones already there. Family A contributes
several p-values below 1e-5, so the rank effect dominates and one family-B delta --
NT-original enhancers|LinearSVC -- is rejected under pooling having been retained under
its own family. Pooling can therefore both add and remove rejections, and the sensitivity
analysis has to be run rather than reasoned about.

APPROXIMATION, INHERITED AND RESTATED
-------------------------------------
Both source files store percentile intervals rather than bootstrap draws, so two-sided
p-values are recovered under a normal reference, SE ~= (hi-lo)/(2*1.96). That is the same
approximation exp_bh_correction.py documents, and it is adequate for a verdict question
only because the deltas are either far from zero or plainly null. Any delta whose pooled
q-value lands within a factor of two of 0.05 is flagged in the output as
`near_boundary`, because for those the approximation could flip the call and the honest
report is that the method cannot resolve them.

Out: results/bh_pooled.csv
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import numpy as np
import pandas as pd
from scipy import stats

from audit.experiments.exp_bh_correction import bh, parse_ci, Q

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")


def approx_p(delta, lo, hi):
    """Two-sided p under a normal reference recovered from a 95% percentile interval."""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan, np.nan
    se = (hi - lo) / (2 * 1.959963985)
    if se <= 0:
        return np.nan, np.nan
    z = delta / se
    return float(2 * (1 - stats.norm.cdf(abs(z)))), float(z)


def load_family_a():
    d = pd.read_csv(os.path.join(R, "bh_correction_frozen.csv"))
    return pd.DataFrame(dict(
        family="A: per-dataset leaky-vs-clean deltas",
        label=d.dataset + "|" + d.model,
        delta=d.delta, p=d.p_approx,
        bh_sig_own_family=d.bh_significant.astype(bool),
    ))


def load_family_b():
    d = pd.read_csv(os.path.join(R, "crosssuite_ranking.csv"))
    ps, zs = [], []
    for _, r in d.iterrows():
        lo, hi = parse_ci(r.drop_ci_cluster)
        # r["drop"], not r.drop: attribute access resolves to DataFrame.drop, the method
        p, z = approx_p(float(r["drop"]), lo, hi)
        ps.append(p); zs.append(z)
    out = pd.DataFrame(dict(
        family="B: cross-suite ranking grid",
        label=d.suite + " " + d.task + "|" + d.model,
        delta=d["drop"], p=ps,
    ))
    ok = out.p.notna()
    sig = np.zeros(len(out), bool)
    sig[np.where(ok)[0]] = bh(out.p[ok].to_numpy())
    out["bh_sig_own_family"] = sig
    return out


if __name__ == "__main__":
    a, b = load_family_a(), load_family_b()
    both = pd.concat([a, b], ignore_index=True)
    ok = both.p.notna()
    pooled = np.zeros(len(both), bool)
    pooled[np.where(ok)[0]] = bh(both.p[ok].to_numpy())
    both["bh_sig_pooled"] = pooled
    both["changed"] = both.bh_sig_own_family != both.bh_sig_pooled
    # q-values, so "near the boundary" is a number and not a judgement call
    m = int(ok.sum())
    both["q_pooled"] = np.nan
    idx = np.where(ok)[0]
    order = idx[np.argsort(both.p.to_numpy()[idx])]
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        running = min(running, both.p.iloc[i] * m / (m - rank))
        both.loc[both.index[i], "q_pooled"] = running
    both["near_boundary"] = (both.q_pooled > Q / 2) & (both.q_pooled < Q * 2)
    both.to_csv(os.path.join(R, "bh_pooled.csv"), index=False)

    print(both.to_string(index=False))
    print(f"\nfamily sizes: A={len(a)}  B={len(b)}  pooled={len(both)} "
          f"({m} with a usable interval)")
    print(f"separate-family rejections: {int(both.bh_sig_own_family.sum())}")
    print(f"pooled-family  rejections: {int(both.bh_sig_pooled.sum())}")
    ch = both[both.changed]
    if len(ch):
        print("\nCHANGED under pooling:")
        print(ch[["family", "label", "delta", "p", "q_pooled",
                  "bh_sig_own_family", "bh_sig_pooled"]].to_string(index=False))
    else:
        print("\nNo delta changes status under pooling.")
    nb = both[both.near_boundary]
    if len(nb):
        print("\nNEAR THE BOUNDARY (normal-approximation p could flip these):")
        print(nb[["label", "p", "q_pooled"]].to_string(index=False))
    print("EXP_BH_POOLED_DONE")
