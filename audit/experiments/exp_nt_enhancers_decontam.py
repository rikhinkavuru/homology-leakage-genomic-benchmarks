#!/usr/bin/env python3
"""
TRAIN-SIDE DECONTAMINATION on NT `enhancers` -- the leakage-only contrast.

exp_nt_enhancers_shipped.py compares the shipped split against a cluster-corrected
re-split and finds the top model changes. That contrast is confounded: the corrected
arm scores ~12 accuracy points HIGHER, so its 400 test sequences are not exchangeable
with the shipped 400, and leakage is not the only thing that moved.

This module removes the confound by holding the TEST SET FIXED. It keeps the shipped
400 test sequences exactly as they are, and deletes from the SHIPPED TRAIN SET every
sequence that is a near-duplicate (8-mer Jaccard >= 0.7, the frozen threshold) of any
test sequence. Then it refits and rescores on the same 400.

  arm  shipped_train   14,968 train seqs, phi = 0.2500  }  SAME 400 test sequences,
  arm  decontam_train  N < 14,968 train seqs, phi = 0   }  so every margin is PAIRED

Because both arms score the same sequences, the cluster bootstrap can be paired ACROSS
arms as well as across models: it resamples test clusters once per draw and computes
both arms' margins on the identical draw. That is the highest-power comparison available
at n = 400, and it isolates leakage from composition.

If the RF-over-HGB lead survives decontamination, the reordering in the corrected-split
module is composition, not leakage. If the lead collapses here too, leakage is doing the
work. Either way the answer is attributable.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_nt_enhancers_decontam
Out:  results/nt_enhancers_decontam.csv
"""
from __future__ import annotations
import itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.core import homology_split as H
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES
from audit.experiments.cluster_bootstrap import test_internal_clusters
from audit.experiments.exp_nt_enhancers_shipped import (
    MODELS, THR, SIM_K, FEAT_K, _order, _rank, mcnemar_counts, INVERSION_MARGIN)

R = os.path.join(HERE, "results")
REPO = SUITES["NT-original"]
TASK = "enhancers"
BOOT = 4000


def paired_two_arm_boot(dA, dB, clusters, boot=BOOT, seed=20240524):
    """Cluster bootstrap of two per-sequence difference vectors on the SAME test set.

    dA, dB are per-sequence (correct_model1 - correct_model2) under arm A and arm B.
    One resample of test clusters per draw scores both arms, so the returned interval
    on (margin_B - margin_A) is fully paired: sequence difficulty and draw noise
    cancel. Returns (margin_A, margin_B, delta, ci_lo, ci_hi, excludes_zero) plus the
    per-arm intervals."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(clusters)
    idx = [np.where(clusters == u)[0] for u in uniq]
    sizes = np.array([len(i) for i in idx], float)
    sumA = np.array([dA[i].sum() for i in idx], float)
    sumB = np.array([dB[i].sum() for i in idx], float)
    G = len(uniq)
    drA = np.empty(boot); drB = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, size=G)
        n = sizes[pick].sum()
        drA[b] = sumA[pick].sum() / n
        drB[b] = sumB[pick].sum() / n
    dd = drB - drA
    pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    loA, hiA = pc(drA); loB, hiB = pc(drB); lo, hi = pc(dd)
    return dict(
        margin_A=float(dA.mean()), ciA=(loA, hiA), seA=float(drA.std(ddof=1)),
        exclA=bool(loA > 0 or hiA < 0),
        margin_B=float(dB.mean()), ciB=(loB, hiB), seB=float(drB.std(ddof=1)),
        exclB=bool(loB > 0 or hiB < 0),
        shift=float(dB.mean() - dA.mean()), ci_shift=(lo, hi),
        se_shift=float(dd.std(ddof=1)), excl_shift=bool(lo > 0 or hi < 0))


def main():
    t0 = time.time()
    tr_s, tr_y = read_fna(fetch(REPO, TASK, "train"))
    te_s, te_y = read_fna(fetch(REPO, TASK, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    tr = np.arange(len(tr_s))
    te = np.arange(len(tr_s), len(seqs))
    print(f"[{TASK}] shipped train={len(tr)} test={len(te)}", flush=True)

    X = E.featurize(seqs, FEAT_K)

    # ---- which TRAIN sequences are near-duplicates of any TEST sequence? -----------
    # max similarity is computed in the reverse direction from the census: for each
    # TRAIN sequence, its max Jaccard to any TEST sequence.
    sim_tr_to_te = E.max_sim_to_train(seqs, te, tr, k=SIM_K, mode="jaccard")
    contaminating = sim_tr_to_te >= THR
    keep = tr[~contaminating]
    print(f"  train sequences near-duplicating a test sequence (>= {THR}): "
          f"{int(contaminating.sum())} / {len(tr)}", flush=True)
    print(f"  decontaminated train n = {len(keep)}  "
          f"class balance {y[keep].mean():.4f} (was {y[tr].mean():.4f})", flush=True)

    # leak fraction of the shipped arm and of the decontaminated arm
    sim_ship = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    phi_ship = float((sim_ship >= 0.9).mean())
    sim_dec = E.max_sim_to_train(seqs, keep, te, k=SIM_K, mode="jaccard")
    phi_dec = float((sim_dec >= 0.9).mean())
    resid = float((sim_dec >= THR).mean())
    print(f"  phi shipped = {phi_ship:.4f}   phi decontaminated = {phi_dec:.4f}"
          f"   residual@{THR} = {resid:.6f}", flush=True)

    novel, high = sim_ship < 0.5, sim_ship >= 0.9

    # ---- both arms, SAME 400 test sequences ---------------------------------------
    corr, acc = {}, {}
    for armname, train_idx in (("shipped_train", tr), ("decontam_train", keep)):
        corr[armname], acc[armname] = {}, {}
        for m in MODELS:
            c = E.correctness(E.models()[m], X, y, train_idx, te)
            corr[armname][m] = c
            acc[armname][m] = float(c.mean())
        print(f"  [{armname}] n_train={len(train_idx)}  {_order(acc[armname])}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        for m in MODELS:
            print(f"      {m:<10} acc={acc[armname][m]:.4f}", flush=True)

    blk, ng = test_internal_clusters([seqs[i] for i in te], THR)
    print(f"  test-internal clusters: {ng}/{len(te)}", flush=True)

    top_s = _order(acc["shipped_train"]).split(">")[0]
    top_d = _order(acc["decontam_train"]).split(">")[0]
    material = bool(top_s != top_d and
                    (acc["shipped_train"][top_s] - acc["shipped_train"][top_d])
                    > INVERSION_MARGIN)

    rows = []
    for A, B in itertools.combinations(MODELS, 2):
        dA = corr["shipped_train"][A] - corr["shipped_train"][B]
        dB = corr["decontam_train"][A] - corr["decontam_train"][B]
        r = paired_two_arm_boot(dA, dB, blk)
        bs, cs = mcnemar_counts(corr["shipped_train"][A], corr["shipped_train"][B])
        bd, cd = mcnemar_counts(corr["decontam_train"][A], corr["decontam_train"][B])
        rows.append(dict(
            suite="NT-original", task=TASK, model_A=A, model_B=B,
            n_test=len(te), n_test_clusters=ng,
            n_train_shipped=len(tr), n_train_decontam=len(keep),
            n_train_removed=int(contaminating.sum()),
            phi_shipped=round(phi_ship, 4), phi_decontam=round(phi_dec, 4),
            acc_A_shipped=round(acc["shipped_train"][A], 4),
            acc_B_shipped=round(acc["shipped_train"][B], 4),
            acc_A_decontam=round(acc["decontam_train"][A], 4),
            acc_B_decontam=round(acc["decontam_train"][B], 4),
            margin_shipped=round(r["margin_A"], 4),
            margin_shipped_ci=f"[{r['ciA'][0]:.4f}, {r['ciA'][1]:.4f}]",
            margin_shipped_excl0=r["exclA"],
            margin_decontam=round(r["margin_B"], 4),
            margin_decontam_ci=f"[{r['ciB'][0]:.4f}, {r['ciB'][1]:.4f}]",
            margin_decontam_excl0=r["exclB"],
            margin_shift=round(r["shift"], 4),
            margin_shift_ci_paired=f"[{r['ci_shift'][0]:.4f}, {r['ci_shift'][1]:.4f}]",
            margin_shift_excl0=r["excl_shift"],
            mde_shift=round(1.96 * r["se_shift"], 4),
            mde_margin_shipped=round(1.96 * r["seA"], 4),
            mcnemar_shipped=f"{bs}/{cs}", mcnemar_decontam=f"{bd}/{cd}",
            order_shipped=_order(acc["shipped_train"]),
            order_decontam=_order(acc["decontam_train"]),
            best_shipped=top_s, best_decontam=top_d,
            material_inversion=material,
            residual_leak_at_thr=round(resid, 6)))

    df = pd.DataFrame(rows)
    out = f"{R}/nt_enhancers_decontam.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, out)

    print("\n== orderings on the SAME shipped 400 test sequences ==")
    print(f"  shipped train   : {_order(acc['shipped_train'])}  (phi={phi_ship:.4f})")
    print(f"  decontam train  : {_order(acc['decontam_train'])}  (phi={phi_dec:.4f})")
    print(f"  MATERIAL INVERSION (frozen >1pt rule): {material}")

    print("\n== RF vs HGB, paired across arms ==")
    s = df[(df.model_A == "RF") & (df.model_B == "HGB")]
    print(s[["margin_shipped", "margin_shipped_ci", "margin_shipped_excl0",
             "margin_decontam", "margin_decontam_ci", "margin_decontam_excl0",
             "margin_shift", "margin_shift_ci_paired", "margin_shift_excl0",
             "mde_shift"]].to_string(index=False))

    print("\n== all pairs ==")
    print(df[["model_A", "model_B", "margin_shipped", "margin_shipped_ci",
              "margin_decontam", "margin_decontam_ci", "margin_shift",
              "margin_shift_ci_paired", "margin_shift_excl0"]].to_string(index=False))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")
    print("EXP_NT_ENHANCERS_DECONTAM_DONE")


if __name__ == "__main__":
    main()
