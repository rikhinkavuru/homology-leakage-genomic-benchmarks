#!/usr/bin/env python3
"""
exp_pretrain_power.py -- price the overlap-free foundation-model arm instead of
only disclaiming it.

WHY THIS EXISTS
---------------
Every genomic foundation model whose ranking a reader would want checked here
(DNABERT-2, HyenaDNA, the Nucleotide Transformer) pretrains on GRCh38, and the
sequences in this suite are drawn from GRCh38, so fine-tuning one on either arm
of our split measures a confounded quantity: pretraining has already seen the
test sequences. The manuscript declines that experiment. A declined experiment
is only a defensible limitation if the reader is told what the RIGHT experiment
is and whether it is powered -- otherwise "we did not run it" and "it cannot be
run" are indistinguishable.

HyenaDNA inherits the Enformer interval set and designates chromosomes 14 and X
as HELD-OUT pretraining test chromosomes. Test rows of this suite that fall on
those two chromosomes are therefore a stratum that model provably never saw
during pretraining, and an as-shipped-vs-corrected comparison restricted to that
stratum carries no pretraining confound. This script measures how large that
stratum actually is and what accuracy difference it could detect.

WHAT IT COMPUTES
----------------
1. Exact counts, per leaky dataset, of AS-SHIPPED TEST rows whose recovered
   interval lies on chr14 or chrX, overall and by class. Coordinate recovery and
   its four-part verification are reused unchanged from chromosome_holdout.py --
   this script fits no model and touches no split.
2. The minimum detectable paired accuracy difference on that stratum. Two models
   scored on the SAME sequences give a paired design, so the right test is
   McNemar's: with b and c the discordant counts, the normal-approximation
   two-sided test at level alpha and power 1-beta needs

       |b - c| >= (z_{1-alpha/2} + z_{1-beta}) * sqrt(b + c)                (1)

   Writing psi = (b+c)/n for the DISCORDANCE RATE -- the fraction of the stratum
   on which the two models disagree -- and delta = (b-c)/n for the accuracy
   difference, (1) becomes

       delta_min = (z_{1-alpha/2} + z_{1-beta}) * sqrt(psi / n)             (2)

   Two models that never disagree need no sample; two that disagree everywhere
   need the most. psi = 1 is therefore the WORST CASE and gives a bound that
   holds whatever the models turn out to do, which is the number to quote when
   the models have not been run. We report the worst case and a grid of
   plausible psi alongside it.

NOT A POWER ANALYSIS OF A RUN EXPERIMENT. No foundation model is fine-tuned
here; psi is not measured, it is swept. The output is a feasibility bound.

Output: results/pretrain_stratum_power.csv
Run:    python -m audit.experiments.exp_pretrain_power
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from audit.core import expkit as E
from audit.experiments.chromosome_holdout import recover_chrom

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "results", "pretrain_stratum_power.csv")

DATASETS = ["human_enhancers_ensembl", "human_nontata_promoters"]

# HyenaDNA's held-out pretraining test chromosomes, inherited from Enformer.
HELDOUT = {"14", "X", "chr14", "chrX"}

ALPHA = 0.05
POWER = 0.80
# Discordance rates to sweep. 1.0 is the worst case and is the bound we quote;
# the rest bracket what two reasonable models on this task would actually do.
PSI_GRID = [0.10, 0.20, 0.30, 0.50, 1.00]


def mde_paired(n, psi, alpha=ALPHA, power=POWER):
    """Minimum detectable paired accuracy difference -- equation (2) above."""
    if n <= 0:
        return float("nan")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * np.sqrt(psi / n)


def main():
    rows = []
    for dset in DATASETS:
        seqs, y, otr, ote, test_frac = E.load(dset)
        chrom, ver = recover_chrom(dset, seqs, y, otr, ote)
        # Same four gates chromosome_holdout.py asserts; a stratum count read off
        # a misaligned recovery would be silently wrong.
        assert ver["rowcount_ok"] and ver["idset_ok"] and ver["label_ok"] \
            and ver["split_ok"], f"coordinate recovery failed verification: {ver}"
        assert ver["sample_seq_mismatch"] == 0, "sampled sequence realignment mismatch"

        ote = np.asarray(ote)
        te_chrom = np.asarray(chrom)[ote]
        te_y = np.asarray(y, dtype=int)[ote]
        keep = np.isin(te_chrom, list(HELDOUT))

        n_test = int(len(ote))
        n_strat = int(keep.sum())
        frac = n_strat / n_test if n_test else float("nan")
        n_pos = int((te_y[keep] == 1).sum())
        n_neg = int((te_y[keep] == 0).sum())
        # Class share WITHIN each class of the full test set -- the asymmetry the
        # manuscript flags, and a confound any user of this stratum must handle.
        pos_share = n_pos / max(1, int((te_y == 1).sum()))
        neg_share = n_neg / max(1, int((te_y == 0).sum()))

        for psi in PSI_GRID:
            rows.append(dict(
                dataset=dset,
                n_test_shipped=n_test,
                n_overlap_free=n_strat,
                frac_overlap_free=round(frac, 4),
                n_pos=n_pos, n_neg=n_neg,
                pos_share_of_test_pos=round(pos_share, 4),
                neg_share_of_test_neg=round(neg_share, 4),
                discordance_psi=psi,
                alpha=ALPHA, power=POWER,
                mde_paired_acc=round(float(mde_paired(n_strat, psi)), 4),
            ))
        print(f"{dset}: test={n_test}  overlap-free(chr14+X)={n_strat} "
              f"({frac:.3%})  pos={n_pos} neg={n_neg}", flush=True)
        for psi in PSI_GRID:
            print(f"    psi={psi:.2f} -> MDE={mde_paired(n_strat, psi):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
