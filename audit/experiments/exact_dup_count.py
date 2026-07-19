#!/usr/bin/env python3
"""Exact train/test duplicate count per dataset (full scale). If most of the
'leakage fraction' is literal identical strings, the near-duplicate-vs-homology
terminology debate (R3.4/R2.a2/R1.3) is settled metric-independently: identical
sequences leak under 8-mer Jaccard, MMseqs2, CD-HIT, alignment, everything.

Also reports reverse-complement exact duplicates (strand-blindness check, NEW gap #5).
"""
import numpy as np
from audit.pipeline import run_suite as S

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
ALL = LEAKY + ["human_ocr_ensembl", "human_enhancers_cohn",
               "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm",
               "drosophila_enhancers_stark"]
_RC = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(_RC)[::-1]

print(f"{'dataset':38s} {'n_test':>7s} {'exact_dup':>9s} {'frac':>6s} {'+revcomp':>9s} {'rc_frac':>7s}")
# REPRODUCE's T26 advertises results/exact_dup_counts.csv; until round 25 this module
# only printed, so section 3.4's byte-identical figures had no committed artifact.
_rows = []

for d in ALL:
    cap = 10**12 if d in LEAKY else 20000
    df, _ = S.discover_and_subsample(d, cap=cap, seed=S.SUBSAMPLE_SEED)
    sp = df["split"].to_numpy(); seq = df["seq"].tolist()
    tr = set(seq[i].upper() for i in np.where(sp == "train")[0])
    te = [seq[i].upper() for i in np.where(sp == "test")[0]]
    n = len(te)
    exact = sum(1 for s in te if s in tr)
    rcadd = sum(1 for s in te if s not in tr and rc(s) in tr)
    print(f"{d:38s} {n:7d} {exact:9d} {exact/n:6.3f} {exact+rcadd:9d} {(exact+rcadd)/n:7.3f}")
    _rows.append(dict(dataset=d, n_test=n, exact_dup=exact,
                      exact_dup_frac=round(exact / n, 4),
                      exact_plus_revcomp=exact + rcadd,
                      exact_plus_revcomp_frac=round((exact + rcadd) / n, 4)))

import os as _os                                                        # noqa: E402
import pandas as _pd                                                    # noqa: E402
_out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))), "results", "exact_dup_counts.csv")
_pd.DataFrame(_rows).to_csv(_out + ".tmp", index=False)
_os.replace(_out + ".tmp", _out)
print(f"\nwrote {_out}")
