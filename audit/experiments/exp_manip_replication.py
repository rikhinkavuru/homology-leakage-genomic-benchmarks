#!/usr/bin/env python3
"""
C1 -- replicate the causal manipulation so it is a DESIGN, not an anecdote.

The paper's construct-and-break experiment (exp_construction_manip.py) runs
break-a-clean on exactly one donor (human_ocr_ensembl) and fix-a-leaky on exactly one
recipient (human_enhancers_ensembl). n=1 per direction. An external reviewer asked for
two things, and this module supplies both.

  (C1a) TWO MORE CLEAN DONORS. Run the identical break-a-clean construction on
        human_enhancers_cohn and drosophila_enhancers_stark. Identical means: the same
        TARGET share of positive rows on a multiplicity-2 coordinate (0.9426, the value
        measured on human_enhancers_ensembl), the same rate algebra r = T/(2-T), the same
        three arms (control / unmatched / size-and-balance-matched), the same split seed,
        the same evaluate() -- including its two-arm cluster bootstrap. Nothing is retuned
        per donor. Three clean donors all developing the same syndrome makes the causal
        claim a replicated design.

  (C1b) SIZE-MATCHED FIX-A-LEAKY COMPARATOR. The committed fix-a-leaky pair is
        B_control_unmerged_union (n=154,842) against B_manipulated_merged_balanced
        (n=81,868). The manipulated arm therefore trains on 47% fewer rows than its
        control, so "the RF drop vanished" is confounded with "the RF had half the
        training data". main_resubmission.tex:68 defends this as unavoidable because the
        intervention is defined by removing rows -- true of the manipulated arm, but it
        does not preclude subsampling the CONTROL down to the manipulated arm's size.
        This module adds B_control_unmerged_union_matched: the unmanipulated dataset
        subsampled to n=81,868 at 50/50, so both arms have equal training resources and
        the only remaining difference is near-duplicate content.

DONOR SCALE. human_enhancers_cohn is subsampled to the frozen CAP_CLEAN=20,000 that every
other clean-dataset arm in the paper uses, so the three donors are read on the same
convention rather than on three different sizes. drosophila_enhancers_stark is 6,914 rows
and runs at full size. Both are natively 50/50, as human_ocr_ensembl is, so the balance
arithmetic of manipulation A applies unchanged.

Run:  python -m audit.experiments.exp_manip_replication            # everything
      python -m audit.experiments.exp_manip_replication --only A   # donors only
      python -m audit.experiments.exp_manip_replication --only B   # size-match only
Out:  results/construction_manipulation_replication.csv
      (a separate file from construction_manipulation.csv -- the frozen table is not
       rewritten, so the committed causal result stays byte-stable and this is additive)
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_construction_manip import evaluate, random_split, SEED, CAP_CLEAN

R = os.path.join(HERE, "results")
OUT = os.path.join(R, "construction_manipulation_replication.csv")

# The share of positive ROWS sitting on a multiplicity-2 coordinate that
# human_enhancers_ensembl exhibits. Frozen; identical to exp_construction_manip.
TARGET = 0.9426
RATE = TARGET / (2 - TARGET)          # 0.8914 -- see the derivation in exp_construction_manip

# Donors for C1a. human_ocr_ensembl is deliberately NOT re-run: its three rows are already
# committed in construction_manipulation.csv and re-running them would only add model noise
# to a frozen number.
DONORS = [("human_enhancers_cohn", CAP_CLEAN), ("drosophila_enhancers_stark", None)]


def break_a_clean(dset, cap):
    """The manipulation-A construction, applied unchanged to one clean donor."""
    print(f"== C1a break-a-clean: {dset} ==", flush=True)
    full = cap is None
    seqs, y, _otr, _ote, tf = E.load(dset, full=full, cap=cap or CAP_CLEAN)
    y = np.asarray(y)
    n_ctrl = len(seqs)
    rows = []

    # ARM 1 -- control. Same rows, same random-split machinery, no intervention.
    tr, te = random_split(len(seqs), tf)
    rows.append(evaluate(seqs, y, tr, te, f"C_control__{dset}",
                         dict(dataset=dset, arm="control", manipulation="none",
                              dup_rate_applied=0.0, dup_share_of_positives=0.0,
                              target_share_on_dup_coord=0.0, donor_n_used=n_ctrl)))

    # ARM 2 -- intervention, unmatched. Duplicate a fraction RATE of positives so that
    # TARGET of positive rows sit on a multiplicity-2 coordinate, then re-split at random.
    rng = np.random.RandomState(SEED)
    pos = np.where(y == y.max())[0]
    dup = rng.choice(pos, size=int(round(RATE * len(pos))), replace=False)
    seqs2 = list(seqs) + [seqs[i] for i in dup]
    y2 = np.concatenate([y, y[dup]])
    tr2, te2 = random_split(len(seqs2), tf)
    rows.append(evaluate(seqs2, y2, tr2, te2, f"C_manipulated_unmerged_union__{dset}",
                         dict(dataset=dset, arm="manipulated_unmatched",
                              manipulation="duplicate positives (multiplicity 2)",
                              dup_rate_applied=round(RATE, 4),
                              target_share_on_dup_coord=TARGET, donor_n_used=n_ctrl)))

    # ARM 3 -- intervention, size AND balance matched to the control. Duplicating only
    # positives moves the set off 50/50 and adds rows; either alone can move accuracy, so
    # the load-bearing contrast is arm 3 against arm 1.
    pos2 = np.where(y2 == y2.max())[0]
    neg2 = np.where(y2 != y2.max())[0]
    half = min(n_ctrl // 2, len(pos2), len(neg2))
    sel = np.sort(np.concatenate([rng.choice(pos2, half, replace=False),
                                  rng.choice(neg2, half, replace=False)]))
    seqs3 = [seqs2[i] for i in sel]
    y3 = y2[sel]
    tr3, te3 = random_split(len(seqs3), tf)
    rows.append(evaluate(seqs3, y3, tr3, te3, f"C_manipulated_matched__{dset}",
                         dict(dataset=dset, arm="manipulated_matched",
                              manipulation="duplicate positives + match control size/balance",
                              dup_rate_applied=round(RATE, 4),
                              target_share_on_dup_coord=TARGET, donor_n_used=n_ctrl)))
    return rows


def size_matched_control():
    """C1b -- subsample the fix-a-leaky CONTROL to the manipulated arm's training size.

    B_manipulated_merged_balanced has n=81,868 at 50/50. We take the unmanipulated
    human_enhancers_ensembl and draw the same n at the same balance, so the two arms
    differ only in whether duplicate coordinates were collapsed."""
    d = "human_enhancers_ensembl"
    N_TARGET = 81868                    # exactly B_manipulated_merged_balanced's n
    print(f"== C1b size-matched fix-a-leaky control: {d} at n={N_TARGET} ==", flush=True)
    seqs, y, _otr, _ote, tf = E.load(d, full=True)
    y = np.asarray(y)
    pos = np.where(y == y.max())[0]
    neg = np.where(y != y.max())[0]
    half = N_TARGET // 2
    assert half <= min(len(pos), len(neg)), "cannot match size at 50/50"
    rng = np.random.RandomState(SEED)
    sel = np.sort(np.concatenate([rng.choice(pos, half, replace=False),
                                  rng.choice(neg, half, replace=False)]))
    seqs_m = [seqs[i] for i in sel]
    y_m = y[sel]
    tr, te = random_split(len(seqs_m), tf)
    # comp=None: recompute near-duplicate components on the subsample from scratch, the
    # same way B_manipulated_merged_balanced does. Restricting the cached full-dataset
    # components to `sel` would be cheaper but would coarsen the partition, making this
    # arm's correction methodologically different from the arm it is compared against.
    return [evaluate(seqs_m, y_m, tr, te, "B_control_unmerged_union_matched",
                     dict(dataset=d, arm="control_size_matched",
                          manipulation="none (control subsampled to manipulated arm size)",
                          kept_fraction=round(len(sel) / len(seqs), 4),
                          positive_fraction=0.5, donor_n_used=len(seqs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B"])
    ap.add_argument("--donor", help="run a single donor by name")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(R, exist_ok=True)
    rows = []
    if a.only in (None, "A"):
        for dset, cap in DONORS:
            if a.donor and dset != a.donor:
                continue
            rows += break_a_clean(dset, cap)
    if a.only in (None, "B") and not a.donor:
        rows += size_matched_control()

    df = pd.DataFrame(rows)
    # Merge by condition rather than overwrite, so a partial re-run cannot truncate the
    # committed table (the bug exp_dose_response, exp_roster and exp_gue all shipped).
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        prev = prev[~prev.condition.isin(set(df.condition))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    print("\n== C1 replication summary ==")
    print(df[["condition", "n", "positive_frac", "leak_jaccard_0p7", "acc_orig_RF",
              "acc_corr_RF", "rf_drop", "rf_drop_ci", "rf_rank_orig", "rf_rank_corr",
              "ranking_inverts"]].to_string(index=False))
    print(f"\nEXP_MANIP_REPLICATION_DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
