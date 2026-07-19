#!/usr/bin/env python3
"""
Dose-response: does the breakdown condition predict WHERE the ranking flips?

The paper contains two results that have so far been argued separately.

  (1) The condition. Writing phi for the near-duplicate test fraction, n_M for a model's
      accuracy on novel test sequences and g_M for its graded memorization gap, the
      as-shipped accuracy is approximately a_M = n_M + phi*g_M. A challenger B overtakes
      an as-shipped leader A exactly when phi exceeds

          phi* = delta / Dg,     delta = n_B - n_A,   Dg = g_A - g_B,

      so a curator can evaluate the condition from the as-shipped split alone.

  (2) The manipulation. Duplicating coordinates in a clean dataset manufactures the whole
      downstream syndrome, and collapsing them in a leaky one abolishes it.

Separately, (1) is bookkeeping and (2) is a single before/after contrast. Together they
support a much sharper claim, which this module tests: because the manipulation SETS phi,
and phi* is computed from quantities measured before any correction, the condition makes a
quantitative, falsifiable prediction about the dose at which the ranking should flip.

DESIGN
------
Take human_ocr_ensembl, which is clean, and impose the duplicated-coordinate construction
at a series of doses. Duplicating a fraction r of positives puts a share 2r/(1+r) of
positive rows on multiplicity-2 coordinates and raises phi monotonically. At each dose:

  * measure phi on the as-shipped (manipulated) split;
  * compute phi* for the leader/challenger pair from the AS-SHIPPED split only --
    novel-stratum accuracies and graded gaps, no corrected split, no peeking;
  * record whether the ranking actually inverts under the near-duplicate-aware re-split.

The condition predicts inversion exactly when phi > phi*. Every arm is downsampled to the
control's size and class balance, so arms differ only in near-duplicate content.

PRE-COMMITTED PREDICTION (recorded here before the run)
-------------------------------------------------------
  P6  Inversion occurs at every dose with phi > phi* and at no dose with phi < phi*,
      i.e. the sign of (phi - phi*) predicts `ranking_inverts` on every arm.
      If the observed flip point disagrees with phi* by more than one dose step, the
      condition does not predict the manipulation and we report that.

Run:  python -m audit.experiments.exp_dose_response
Out:  results/dose_response.csv
"""
from __future__ import annotations
import argparse, itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_construction_manip import evaluate, random_split, MODELS, SEED

R = os.path.join(HERE, "results")
DATASET = "human_ocr_ensembl"
CAP = 20000
# Doses as the TARGET share of positive rows sitting on a multiplicity-2 coordinate.
# 0.9426 is the value measured on human_enhancers_ensembl and used by the headline
# manipulation; the rest bracket it from below so the flip point can be located.
DOSES = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.60, 0.75, 0.9426]


def phi_star_from_as_shipped(seqs, y, tr, te):
    """Compute the breakdown point for the as-shipped leader against every challenger,
    using ONLY as-shipped quantities: novel-stratum accuracy n_M and graded gap g_M.

    Returns (leader, best_challenger, delta, Dg, phi_star) for the challenger with the
    smallest positive phi*, i.e. the one the condition says should overtake first."""
    X = E.featurize(seqs, 6)
    sim = E.max_sim_to_train(seqs, tr, te, k=8, mode="jaccard")
    novel, high = sim < 0.5, sim >= 0.9
    acc, strat = {}, {}
    for m in MODELS:
        c = E.correctness(E.models()[m], X, y, tr, te)
        acc[m] = float(c.mean())
        n_m = float(c[novel].mean()) if novel.any() else np.nan
        h_m = float(c[high].mean()) if high.any() else np.nan
        strat[m] = dict(n=n_m, g=h_m - n_m)
    leader = max(acc, key=acc.get)
    best = None
    for b in MODELS:
        if b == leader:
            continue
        delta = strat[b]["n"] - strat[leader]["n"]
        Dg = strat[leader]["g"] - strat[b]["g"]
        if not (np.isfinite(delta) and np.isfinite(Dg)) or Dg <= 0 or delta <= 0:
            continue
        ps = delta / Dg
        if best is None or ps < best[-1]:
            best = (leader, b, delta, Dg, ps)
    if best is None:
        return (leader, None, np.nan, np.nan, np.nan)
    return best


def one_dose(seqs, y, tf, target, n_ctrl):
    """Impose the duplicated-coordinate construction at `target`, then match the control's
    size and class balance so only near-duplicate content differs between arms."""
    y = np.asarray(y)
    rng = np.random.RandomState(SEED)
    if target <= 0:
        seqs_d, y_d = list(seqs), y
    else:
        # 2r/(1+r) = target  =>  r = target/(2-target)
        rate = target / (2 - target)
        pos = np.where(y == y.max())[0]
        dup = rng.choice(pos, size=int(round(rate * len(pos))), replace=False)
        seqs_d = list(seqs) + [seqs[i] for i in dup]
        y_d = np.concatenate([y, y[dup]])
    # match size AND balance to the control
    pos2 = np.where(y_d == y_d.max())[0]
    neg2 = np.where(y_d != y_d.max())[0]
    half = min(n_ctrl // 2, len(pos2), len(neg2))
    sel = np.sort(np.concatenate([rng.choice(pos2, half, replace=False),
                                  rng.choice(neg2, half, replace=False)]))
    seqs_m = [seqs_d[i] for i in sel]
    y_m = y_d[sel]
    tr, te = random_split(len(seqs_m), tf)
    return seqs_m, y_m, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", nargs="*", type=float, default=DOSES)
    ap.add_argument("--cap", type=int, default=CAP)
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    t0 = time.time()

    seqs, y, _otr, _ote, tf = E.load(DATASET, full=False, cap=a.cap)
    y = np.asarray(y)
    n_ctrl = len(seqs)
    print(f"== dose-response on {DATASET} (n={n_ctrl}) ==", flush=True)
    print("P6 (pre-committed): inversion at every dose with phi > phi*, none below.\n",
          flush=True)

    rows = []
    for target in a.doses:
        seqs_m, y_m, tr, te = one_dose(seqs, y, tf, target, n_ctrl)
        leader, chal, delta, Dg, ps = phi_star_from_as_shipped(seqs_m, y_m, tr, te)
        row = evaluate(seqs_m, y_m, tr, te, f"dose_{target:.4f}",
                       dict(dataset=DATASET, dose_target_share=target,
                            phi_star_leader=leader, phi_star_challenger=chal,
                            phi_star_delta=round(float(delta), 4)
                            if np.isfinite(delta) else None,
                            phi_star_Dg=round(float(Dg), 4)
                            if np.isfinite(Dg) else None,
                            phi_star=round(float(ps), 4) if np.isfinite(ps) else None))
        phi = row["leak_jaccard_0p9"]          # phi is the >=0.9 fraction, as in eq. (1)
        predicted = bool(np.isfinite(ps) and phi > ps)
        row.update(phi_used=phi, predicted_invert=predicted,
                   prediction_correct=bool(predicted == row["ranking_inverts"]))
        print(f"      phi={phi:.4f}  phi*={ps if np.isfinite(ps) else float('nan'):.4f} "
              f"({leader} vs {chal})  predict={'INVERT' if predicted else 'no'}  "
              f"observed={'INVERT' if row['ranking_inverts'] else 'no'}  "
              f"-> {'OK' if row['prediction_correct'] else 'WRONG'}", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Merge by dose rather than overwrite. Running a subset via --doses previously
    # replaced the whole table with just that subset: an auditor re-running two doses
    # truncated the committed ten-dose file to two rows. Same bug exp_roster, exp_tuning
    # and exp_gue all had, and the same fix.
    out = f"{R}/dose_response.csv"
    if os.path.exists(out):
        prev = pd.read_csv(out)
        if "dose_target_share" in prev.columns:
            prev = prev[~prev.dose_target_share.isin(set(df.dose_target_share))]
            df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values("dose_target_share").reset_index(drop=True)
    df.to_csv(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)

    print("\n== dose-response ==")
    print(df[["dose_target_share", "n", "positive_frac", "phi_used", "phi_star",
              "phi_star_leader", "phi_star_challenger", "predicted_invert",
              "ranking_inverts", "prediction_correct"]].to_string(index=False))
    ok = int(df.prediction_correct.sum())
    print(f"\nP6: the condition predicts the observed flip on {ok}/{len(df)} doses.")
    inv = df[df.ranking_inverts]
    if len(inv):
        print(f"  observed flip first at dose {inv.dose_target_share.iloc[0]:.4f} "
              f"(phi={inv.phi_used.iloc[0]:.4f})")
    print(f"\nEXP_DOSE_RESPONSE_DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
