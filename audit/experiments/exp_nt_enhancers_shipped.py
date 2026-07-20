#!/usr/bin/env python3
"""
Does near-duplicate leakage reorder the model ranking on the NT `enhancers` task
AT THE LEAK FRACTION THE TASK ACTUALLY SHIPS?

WHY THIS RUN EXISTS
-------------------
exp_crosssuite_ranking.py evaluated NT `enhancers` on a POOLED re-split (20% test)
because the shipped test set holds only 400 sequences. Pooling changed the leak
fraction: the shipped split leaks 25.0% byte-identical test-to-train, the pooled
re-split leaks phi = 0.0992. The inversion condition phi*Dg - delta with the
recorded Dg = 0.1839, delta = 0.0134 then gives 0.0049 at phi = 0.0992 -- immaterial,
hence the paper's "no material ranking inversion" for this pair. At phi = 0.25 the
same numbers give 0.0326, which IS material. The models were never evaluated there.

This module evaluates them there: train on the SHIPPED train split, evaluate on the
SHIPPED 400-sequence test split, four classical models, k=6 features, canonical
.predict() rule -- everything else identical to the rest of the audit. Then it builds
the near-duplicate-aware corrected split for the same task AT THE SHIPPED TEST
FRACTION (so both arms carry the same n and the same power) and evaluates again.

READOUTS
--------
  * as-shipped and corrected orderings of all four models
  * the RF-vs-HGB margin in each arm, with a PAIRED cluster bootstrap interval
    (both models scored on the same resampled test clusters, so the interval is on
    the margin itself and not on the difference of two independent accuracies)
  * whether a MATERIAL inversion occurs by the paper's frozen criterion
    (top model changes AND as-shipped margin > 1 accuracy point)
  * an explicit power statement: the minimum margin detectable at n = 400.

POWER IS THE POINT OF THE LAST READOUT. n = 400 is small. A wide interval that
includes zero here is UNDERPOWERED, not null, and the two must not be conflated.
The MDE is computed two ways -- from the paired bootstrap SE and from the observed
McNemar discordance -- and printed alongside every margin.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_nt_enhancers_shipped
Out:  results/nt_enhancers_shipped_split.csv
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

R = os.path.join(HERE, "results")
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
REPO = SUITES["NT-original"]
TASK = "enhancers"
THR, SIM_K, FEAT_K = 0.7, 8, 6
SEED = 0
BOOT = 4000
INVERSION_MARGIN = 0.01          # frozen rule, report_card.py:55


def _order(acc):
    return ">".join(sorted(MODELS, key=lambda m: -acc[m]))


def _rank(acc, m):
    return 1 + sum(acc[o] > acc[m] for o in MODELS)


def paired_margin_boot(cA, cB, clusters, boot=BOOT, seed=20240524):
    """Cluster bootstrap of the PAIRED margin acc_A - acc_B.

    Whole test-internal clusters are resampled with replacement; both models are
    scored on the SAME drawn indices, so per-sequence difficulty cancels and the
    interval is on the margin. Singleton clusters reduce this to the paired
    sample bootstrap, which is the correct degenerate case for a test set with no
    internal near-duplicates."""
    rng = np.random.RandomState(seed)
    d = np.asarray(cA, float) - np.asarray(cB, float)
    uniq = np.unique(clusters)
    members = [d[clusters == u] for u in uniq]
    sizes = np.array([len(m) for m in members], float)
    sums = np.array([m.sum() for m in members], float)
    G = len(uniq)
    draws = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, size=G)
        draws[b] = sums[pick].sum() / sizes[pick].sum()
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return float(d.mean()), lo, hi, float(draws.std(ddof=1)), bool(lo > 0 or hi < 0)


def mcnemar_counts(cA, cB):
    a, b = np.asarray(cA, float), np.asarray(cB, float)
    b_only = int(((a == 1) & (b == 0)).sum())     # A right, B wrong
    c_only = int(((a == 0) & (b == 1)).sum())     # B right, A wrong
    return b_only, c_only


def arm(name, seqs, y, tr, te, X, note=""):
    """Fit all four models on `tr`, score on `te`; return correctness vectors + acc."""
    t0 = time.time()
    corr, acc = {}, {}
    for m in MODELS:
        c = E.correctness(E.models()[m], X, y, tr, te)
        corr[m] = c
        acc[m] = float(c.mean())
    print(f"  [{name}] n_train={len(tr)} n_test={len(te)}  {_order(acc)}  "
          f"({time.time()-t0:.0f}s) {note}", flush=True)
    for m in MODELS:
        print(f"      {m:<10} acc={acc[m]:.4f}", flush=True)
    return corr, acc


def main():
    t0 = time.time()
    os.makedirs(R, exist_ok=True)
    tr_s, tr_y = read_fna(fetch(REPO, TASK, "train"))
    te_s, te_y = read_fna(fetch(REPO, TASK, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    tr = np.arange(len(tr_s))
    te = np.arange(len(tr_s), len(seqs))
    shipped_test_frac = len(te) / len(seqs)
    print(f"[{TASK}] shipped split: train={len(tr)} test={len(te)} "
          f"test_frac={shipped_test_frac:.4f}", flush=True)

    X = E.featurize(seqs, FEAT_K)

    # ---- leak fraction on the SHIPPED split ---------------------------------------
    sim = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    phi = float((sim >= 0.9).mean())
    novel, high = sim < 0.5, sim >= 0.9
    print(f"  shipped phi(>=0.9 jaccard) = {phi:.4f}   "
          f"n_leaked={int(high.sum())} n_novel={int(novel.sum())}", flush=True)

    # ---- ARM 1: as shipped ---------------------------------------------------------
    corr_o, acc_o = arm("as-shipped", seqs, y, tr, te, X)
    strata = {}
    for m in MODELS:
        c = corr_o[m]
        n_m = float(c[novel].mean()) if novel.any() else np.nan
        h_m = float(c[high].mean()) if high.any() else np.nan
        strata[m] = dict(n=n_m, h=h_m, g=h_m - n_m)

    # ---- ARM 2: corrected, AT THE SHIPPED TEST FRACTION ----------------------------
    comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, shipped_test_frac)
    ver = H.verify_split(seqs, ctr, cte, THR, SIM_K)
    print(f"  corrected split: residual_leak={ver['residual_leak_fraction']:.6f}",
          flush=True)
    corr_c, acc_c = arm("corrected", seqs, y, ctr, cte, X)

    blk_o, ng_o = test_internal_clusters([seqs[i] for i in te], THR)
    blk_c, ng_c = test_internal_clusters([seqs[i] for i in cte], THR)
    print(f"  test-internal clusters: shipped {ng_o}/{len(te)}  "
          f"corrected {ng_c}/{len(cte)}", flush=True)

    top_o = _order(acc_o).split(">")[0]
    top_c = _order(acc_c).split(">")[0]
    material = bool(top_o != top_c and (acc_o[top_o] - acc_o[top_c]) > INVERSION_MARGIN)

    rows = []
    for A, B in itertools.combinations(MODELS, 2):
        for armname, corr, acc, blk, ntest in (
                ("as_shipped", corr_o, acc_o, blk_o, len(te)),
                ("corrected", corr_c, acc_c, blk_c, len(cte))):
            mpt, lo, hi, se, excl = paired_margin_boot(corr[A], corr[B], blk)
            b_only, c_only = mcnemar_counts(corr[A], corr[B])
            disc = b_only + c_only
            # MDE: smallest |margin| a 95% two-sided test could resolve, from the
            # paired bootstrap SE, and from the McNemar discordance (SE = sqrt(disc)/n).
            mde_boot = 1.96 * se
            mde_mcn = 1.96 * np.sqrt(max(disc, 1)) / ntest
            rows.append(dict(
                suite="NT-original", task=TASK, arm=armname, model_A=A, model_B=B,
                n_test=ntest, n_test_clusters=int(len(np.unique(blk))),
                phi_shipped=round(phi, 4),
                acc_A=round(acc[A], 4), acc_B=round(acc[B], 4),
                margin_A_minus_B=round(mpt, 4),
                margin_ci95=f"[{lo:.4f}, {hi:.4f}]",
                margin_se=round(se, 4), margin_excl0=excl,
                mcnemar_b=b_only, mcnemar_c=c_only, discordant=disc,
                mde_from_boot_se=round(mde_boot, 4),
                mde_from_mcnemar=round(mde_mcn, 4),
                order=_order(acc), rank_A=_rank(acc, A), rank_B=_rank(acc, B),
                best=_order(acc).split(">")[0],
                material_inversion_task=material,
                residual_leak_after_split=round(float(ver["residual_leak_fraction"]), 6),
                novel_acc_A=round(strata[A]["n"], 4),
                novel_acc_B=round(strata[B]["n"], 4),
                graded_gap_A=round(strata[A]["g"], 4),
                graded_gap_B=round(strata[B]["g"], 4),
                delta_nB_minus_nA=round(strata[B]["n"] - strata[A]["n"], 4),
                Dg_A_minus_B=round(strata[A]["g"] - strata[B]["g"], 4),
                predicted_margin_at_phi=round(
                    phi * (strata[A]["g"] - strata[B]["g"])
                    - (strata[B]["n"] - strata[A]["n"]), 4),
            ))

    df = pd.DataFrame(rows)
    out = f"{R}/nt_enhancers_shipped_split.csv"
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, out)

    print("\n== orderings ==")
    print(f"  as-shipped : {_order(acc_o)}   (phi = {phi:.4f})")
    print(f"  corrected  : {_order(acc_c)}")
    print(f"  top model changes: {top_o != top_c}   "
          f"as-shipped margin between them: {acc_o[top_o]-acc_o[top_c]:.4f}")
    print(f"  MATERIAL INVERSION (frozen >1pt rule): {material}")

    print("\n== RF vs HGB (the pre-specified pair) ==")
    sub = df[(df.model_A == "RF") & (df.model_B == "HGB")]
    print(sub[["arm", "n_test", "acc_A", "acc_B", "margin_A_minus_B", "margin_ci95",
               "margin_se", "margin_excl0", "discordant", "mde_from_boot_se",
               "mde_from_mcnemar"]].to_string(index=False))

    print("\n== all pairs ==")
    print(df[["arm", "model_A", "model_B", "margin_A_minus_B", "margin_ci95",
              "margin_excl0", "mde_from_boot_se"]].to_string(index=False))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")
    print("EXP_NT_ENHANCERS_SHIPPED_DONE")


if __name__ == "__main__":
    main()
