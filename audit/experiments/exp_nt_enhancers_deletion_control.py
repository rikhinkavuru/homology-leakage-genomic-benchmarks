#!/usr/bin/env python3
"""
DELETION CONTROL for the NT `enhancers` train-side decontamination result.

exp_nt_enhancers_decontam.py deletes the 107 of 14,968 training sequences that
near-duplicate a test sequence (8-mer Jaccard >= 0.7), holds the shipped 400 test
sequences fixed, and finds the top model changes RF -> LR. It attributes that change
to contamination, on the grounds that the test set is byte-identical between arms.

That argument has a hole. TWO things changed, not one: contamination went to zero AND
the training set lost 107 rows. At n_test = 400 the four models sit inside 3.75 accuracy
points of each other and the bootstrap MDE is ~0.03, so a reordering could in principle
be produced by ANY perturbation of the training set of that size, contaminating or not.

This module supplies the missing arm: delete 107 UNIFORMLY RANDOM training sequences
(same count, contamination left intact), refit, rescore on the same 400, repeat over
seeds. It answers the one question the decontamination run cannot answer alone:

  Does a size-matched, leakage-PRESERVING deletion also reorder the leaderboard?

  * If random deletions leave RF on top, the decontamination reordering is attributable
    to contamination and the paper's claim stands.
  * If random deletions reorder it too, the reordering is deletion noise at n = 400 and
    the contamination attribution is not identified.

The shipped arm is refit here from scratch rather than read from the CSV, so this run
also independently reproduces the as-shipped accuracies.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_nt_enhancers_deletion_control
Out:  results/nt_enhancers_deletion_control.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES
from audit.experiments.exp_nt_enhancers_shipped import MODELS, THR, SIM_K, FEAT_K, _order

R = os.path.join(HERE, "results")
REPO = SUITES["NT-original"]
TASK = "enhancers"
NSEEDS = 10


def main():
    t0 = time.time()
    tr_s, tr_y = read_fna(fetch(REPO, TASK, "train"))
    te_s, te_y = read_fna(fetch(REPO, TASK, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    tr = np.arange(len(tr_s))
    te = np.arange(len(tr_s), len(seqs))
    X = E.featurize(seqs, FEAT_K)
    print(f"[{TASK}] shipped train={len(tr)} test={len(te)}", flush=True)

    # the real decontamination set, recomputed here (independent re-derivation)
    sim_tr_to_te = E.max_sim_to_train(seqs, te, tr, k=SIM_K, mode="jaccard")
    contaminating = tr[sim_tr_to_te >= THR]
    ndel = len(contaminating)
    keep_dec = np.setdiff1d(tr, contaminating)
    sim_ship = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    phi_ship = float((sim_ship >= 0.9).mean())
    print(f"  contaminating train rows = {ndel}  phi_shipped = {phi_ship:.4f}", flush=True)

    rows = []

    def score(train_idx, label, seed, phi):
        acc = {m: float(E.correctness(E.models()[m], X, y, train_idx, te).mean())
               for m in MODELS}
        o = _order(acc)
        rows.append(dict(suite="NT-original", task=TASK, arm=label, seed=seed,
                         n_train=len(train_idx), n_deleted=len(tr) - len(train_idx),
                         phi=round(phi, 4),
                         class_balance=round(float(y[train_idx].mean()), 4),
                         **{f"acc_{m}": round(acc[m], 4) for m in MODELS},
                         order=o, best=o.split(">")[0],
                         margin_RF_minus_LR=round(acc["RF"] - acc["LR"], 4)))
        print(f"  [{label} seed={seed}] {o}  RF-LR={acc['RF']-acc['LR']:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        return acc

    score(tr, "shipped_train", -1, phi_ship)

    sim_dec = E.max_sim_to_train(seqs, keep_dec, te, k=SIM_K, mode="jaccard")
    score(keep_dec, "decontam_train", -1, float((sim_dec >= 0.9).mean()))

    for s in range(NSEEDS):
        rng = np.random.RandomState(1000 + s)
        drop = rng.choice(len(tr), size=ndel, replace=False)
        keep_r = np.setdiff1d(tr, tr[drop])
        sim_r = E.max_sim_to_train(seqs, keep_r, te, k=SIM_K, mode="jaccard")
        score(keep_r, "random_deletion", s, float((sim_r >= 0.9).mean()))

    df = pd.DataFrame(rows)
    out = f"{R}/nt_enhancers_deletion_control.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, out)

    print("\n== summary ==")
    print(df[["arm", "seed", "n_train", "phi", "order", "best",
              "margin_RF_minus_LR"]].to_string(index=False))
    rd = df[df.arm == "random_deletion"]
    print(f"\nrandom-deletion arms: best-model counts {rd['best'].value_counts().to_dict()}")
    print(f"random-deletion RF-LR margin: mean {rd['margin_RF_minus_LR'].mean():+.4f}  "
          f"min {rd['margin_RF_minus_LR'].min():+.4f}  "
          f"max {rd['margin_RF_minus_LR'].max():+.4f}")
    print(f"decontam RF-LR margin: "
          f"{float(df[df.arm=='decontam_train']['margin_RF_minus_LR'].iloc[0]):+.4f}")
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")
    print("EXP_NT_ENHANCERS_DELETION_CONTROL_DONE")


if __name__ == "__main__":
    main()
