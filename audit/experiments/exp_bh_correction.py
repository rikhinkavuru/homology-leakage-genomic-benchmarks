#!/usr/bin/env python3
"""
Multiplicity correction for the frozen leaky-vs-clean delta grid.

WHY THIS EXISTS
---------------
paper/main.tex asserts: "Because we report roughly a dozen leaky-versus-clean delta
intervals, we also applied a Benjamini-Hochberg correction at q=0.05: it leaves the two
leaky random-forest drops and the human_nontata_promoters logistic-regression drop
significant and no clean or three-class delta significant, so multiplicity changes no
verdict."

A round-2 adversarial audit found that NO Benjamini-Hochberg computation exists anywhere
in the repository: no code, no p-value, no q-value, no artifact column. The sentence
describes a result that was never computed. This module computes it, so the claim is
either substantiated with an artifact or corrected.

METHOD, AND ITS LIMITATION (stated because it matters)
------------------------------------------------------
The frozen artifact `cluster_bootstrap_full.csv` stores percentile cluster-bootstrap
CONFIDENCE INTERVALS, not the underlying draws, and the draws were not retained. A
percentile interval does not determine a p-value exactly. We therefore recover an
approximate two-sided p-value under a normal reference:

    SE  ~= (hi - lo) / (2 * 1.96)        from the 95% percentile interval
    z    = delta / SE
    p    = 2 * (1 - Phi(|z|))

This is an approximation, and it is labelled as one. It is adequate for the question at
hand -- whether BH at q=0.05 changes any VERDICT -- because the deltas in this grid are
either far from zero (the leaky random-forest drops, |z| > 30) or plainly null (every
clean delta, |z| < 2), so no verdict sits near the decision boundary where the
approximation could flip it. Reconstructing exact bootstrap p-values would require
re-running the full cluster bootstrap and retaining the draws; that is the correct
long-term fix and is noted in the output.

Run:  python -m audit.experiments.exp_bh_correction
Out:  results/bh_correction_frozen.csv
"""
from __future__ import annotations
import os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(HERE, "results")
Q = 0.05
LEAKY = {"human_nontata_promoters", "human_enhancers_ensembl"}


def bh(pvals, q=Q):
    """Benjamini-Hochberg step-up. Returns a boolean mask of rejections."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.where(passed)[0])) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def parse_ci(s):
    if not isinstance(s, str):
        return None, None
    nums = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", s.replace("np.float64", ""))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[1])


def main():
    src = f"{R}/cluster_bootstrap_full.csv"
    df = pd.read_csv(src)
    rows = []
    for _, r in df.iterrows():
        lo, hi = parse_ci(r.get("delta_ci_cluster"))
        if lo is None:
            continue
        delta = float(r["delta"])
        se = (hi - lo) / (2 * 1.959964)
        z = delta / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append(dict(
            dataset=r["dataset"], model=r["model"],
            group="leaky" if r["dataset"] in LEAKY else "clean/3-class",
            delta=round(delta, 4), ci_cluster=f"[{lo:.4f}, {hi:.4f}]",
            se_approx=round(se, 5), z_approx=round(z, 2),
            p_approx=p, excl0_uncorrected=bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows)
    out["bh_significant"] = bh(out.p_approx.to_numpy(), Q)
    out["p_approx"] = out.p_approx.map(lambda v: f"{v:.3e}")
    out.to_csv(f"{R}/bh_correction_frozen.csv", index=False)

    print(f"Benjamini-Hochberg at q={Q} over {len(out)} frozen delta tests")
    print("(p-values are NORMAL-APPROXIMATE, recovered from percentile CIs; see docstring)\n")
    print(out[["dataset", "model", "group", "delta", "ci_cluster", "z_approx",
               "p_approx", "excl0_uncorrected", "bh_significant"]].to_string(index=False))

    sig = out[out.bh_significant]
    print(f"\nBH-significant ({len(sig)}): "
          f"{[f'{r.dataset}/{r.model}' for _, r in sig.iterrows()]}")
    flips = out[out.excl0_uncorrected != out.bh_significant]
    print(f"verdicts changed by BH: {len(flips)}")
    if len(flips):
        print(flips[["dataset", "model", "excl0_uncorrected", "bh_significant"]]
              .to_string(index=False))

    # Adjudicate the manuscript's specific sentence.
    claimed = {("human_enhancers_ensembl", "RF_k6"), ("human_nontata_promoters", "RF_k6"),
               ("human_nontata_promoters", "LR_k6")}
    got = {(r.dataset, r.model) for _, r in sig.iterrows()}
    print(f"\nmanuscript claims exactly these survive BH: {sorted(claimed)}")
    print(f"actually survive:                            {sorted(got)}")
    print("CLAIM SUPPORTED" if got == claimed else
          f"CLAIM NOT SUPPORTED AS WRITTEN -- extra: {sorted(got - claimed)}, "
          f"missing: {sorted(claimed - got)}")
    print("\nNOTE: exact bootstrap p-values require re-running the cluster bootstrap "
          "with draw retention; this normal approximation is a sensitivity check.")


if __name__ == "__main__":
    main()
