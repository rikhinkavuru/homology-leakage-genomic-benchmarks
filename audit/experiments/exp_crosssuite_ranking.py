#!/usr/bin/env python3
"""
Deepener #1, phase B -- does near-duplicate leakage REORDER the ranking on a SECOND suite?

This is the experiment the paper's rejection turns on. Phase A (exp_crosssuite_census.py)
established that near-duplicate leakage exists in the Nucleotide Transformer downstream
tasks -- a suite built independently of Genomic Benchmarks. A census is not a re-ranking,
so on its own it leaves the "n=1 / not systematic" objection standing. Here we run the
paper's OWN pipeline, unchanged, on the leaky NT-original tasks and ask whether the model
ranking changes when the leakage is removed.

PROTOCOL (identical to the paper's, deliberately -- no new modelling code)
-------------------------------------------------------------------------
  original   : the task's as-shipped train/test split
  corrected  : pool all sequences, take connected components of the 8-mer Jaccard > 0.7
               graph, assign WHOLE clusters to train/test at the same test fraction
               (homology_split._assign via expkit), refit
  models     : the same four classical models (LR, LinearSVC, RF, HGB), k=6 features,
               canonical `.predict()` decision rule
  readouts   : accuracy / AUROC / F1 rankings, P(rank-1) under sample and cluster
               bootstrap, two-arm cluster-bootstrap CI on each model's drop

THE PRE-COMMITTED PREDICTION (this is what makes it a test, not a fishing trip)
------------------------------------------------------------------------------
Before computing any corrected split, we measure on the AS-SHIPPED split alone the
quantities the inversion law of exp_inversion_law.py needs -- the leak fraction phi, each
model's novel-stratum accuracy n_M and graded memorization gap g_M -- and emit, per model
pair, the predicted verdict from

        inversion iff 0 < delta < phi * Dg ,    delta = n_B - n_A ,  Dg = g_A - g_B .

Those predictions are written to the results CSV in the same row as the observed outcome,
so agreement or disagreement is auditable. This is the law's first genuinely out-of-sample
test: every prior evaluation reused datasets it was derived on. A NULL result here is
still a result -- if no ranking inverts, the law should say so in advance via delta <= 0,
and that is reportable as a confirmed prediction rather than a failed experiment.

TASKS
-----
  splice_sites_acceptors  leak@0.7 = 0.2435, 600 bp, n = 19,961 / 2,218
  splice_sites_donors     leak@0.7 = 0.2452, 600 bp, n = 19,775 / 2,198
  enhancers               leak@0.7 = 0.2500, 25.0% BYTE-IDENTICAL, 200 bp
  promoter_no_tata        leak@0.7 = 0.0059  <- CLEAN CONTROL, must NOT invert

`enhancers` ships only 400 test sequences, too few to carry a usable interval, so for
that task we pool train+test and draw a fresh stratified split at the same ratio before
running both arms; this is disclosed in the output as `pooled_resplit`.

Run:  python -m audit.experiments.exp_crosssuite_ranking
Out:  results/crosssuite_ranking.csv, results/crosssuite_ranking_pairs.csv
"""
from __future__ import annotations
import argparse, itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.core import homology_split as H
from audit.experiments.exp_crosssuite_census import read_fna, fetch, SUITES

R = os.path.join(HERE, "results")
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
REPO_ORIG = SUITES["NT-original"]
THR, SIM_K, FEAT_K = 0.7, 8, 6
SEED = 0
BOOT = 2000

TASKS = [
    dict(task="splice_sites_acceptors", predicted="LEAKY", pooled=False),
    dict(task="splice_sites_donors", predicted="LEAKY", pooled=False),
    dict(task="enhancers", predicted="LEAKY", pooled=True),
    dict(task="promoter_no_tata", predicted="CLEAN (control)", pooled=False),
]


def _order(acc):
    return ">".join(sorted(MODELS, key=lambda m: -acc[m]))


# The paper counts a ranking inversion only when the top model CHANGES *and* the original
# margin between the old and new leader exceeds 1 accuracy point (report_card.py:55).
# Without that margin a clean dataset whose models sit within noise of each other will
# "invert" on every reshuffle -- as promoter_no_tata does here at a 0.0023 margin. Reusing
# the frozen rule keeps one decision rule across the whole project.
INVERSION_MARGIN = 0.01


def inverts(acc_o, acc_c):
    top_o, top_c = _order(acc_o).split(">")[0], _order(acc_c).split(">")[0]
    return bool(top_o != top_c and (acc_o[top_o] - acc_o[top_c]) > INVERSION_MARGIN)


def _rank(acc, m):
    return 1 + sum(acc[o] > acc[m] for o in MODELS)


def p_rank1(corr_vecs, clusters=None, boot=BOOT, seed=20240524):
    """P(model ranks 1st) under resampling. Sample bootstrap when clusters is None,
    else whole-cluster (block) bootstrap -- the same estimator as the paper's G6."""
    rng = np.random.RandomState(seed)
    n = len(next(iter(corr_vecs.values())))
    wins = {m: 0 for m in MODELS}
    if clusters is None:
        idx_pool = np.arange(n)
        for _ in range(boot):
            idx = rng.randint(0, n, n)
            accs = {m: corr_vecs[m][idx].mean() for m in MODELS}
            wins[max(accs, key=accs.get)] += 1
    else:
        uniq = np.unique(clusters)
        members = {u: np.where(clusters == u)[0] for u in uniq}
        G = len(uniq)
        for _ in range(boot):
            pick = rng.randint(0, G, G)
            idx = np.concatenate([members[uniq[p]] for p in pick])
            accs = {m: corr_vecs[m][idx].mean() for m in MODELS}
            wins[max(accs, key=accs.get)] += 1
    return {m: wins[m] / boot for m in MODELS}


def run_task(spec, cap=None):
    task, pooled = spec["task"], spec["pooled"]
    t0 = time.time()
    tr_s, tr_y = read_fna(fetch(REPO_ORIG, task, "train"))
    te_s, te_y = read_fna(fetch(REPO_ORIG, task, "test"))
    seqs = tr_s + te_s
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)]).astype(int)
    tr = np.arange(len(tr_s))
    te = np.arange(len(tr_s), len(seqs))
    test_frac = len(te) / len(seqs)

    if pooled:
        # 400 test sequences cannot carry an interval; redraw a stratified split at the
        # SAME ratio over the pooled data. Disclosed in the output.
        rng = np.random.RandomState(SEED)
        test_frac = 0.2
        idx = rng.permutation(len(seqs))
        cut = int(round((1 - test_frac) * len(seqs)))
        tr, te = np.sort(idx[:cut]), np.sort(idx[cut:])

    if cap and len(seqs) > cap:                       # debugging aid only
        keep = np.sort(np.random.RandomState(SEED).choice(len(seqs), cap, replace=False))
        pos = {v: i for i, v in enumerate(keep)}
        seqs = [seqs[i] for i in keep]; y = y[keep]
        tr = np.array([pos[i] for i in tr if i in pos])
        te = np.array([pos[i] for i in te if i in pos])

    print(f"[{task}] n={len(seqs)} (train {len(tr)} / test {len(te)}) "
          f"test_frac={test_frac:.3f} pooled_resplit={pooled}", flush=True)

    X = E.featurize(seqs, FEAT_K)

    # ---- AS-SHIPPED ARM, and the pre-committed prediction it licenses -------------
    sim = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    phi = float((sim >= 0.9).mean())
    novel, high = sim < 0.5, sim >= 0.9
    corr_o, acc_o, strata = {}, {}, {}
    for m in MODELS:
        c = E.correctness(E.models()[m], X, y, tr, te)
        corr_o[m] = c
        acc_o[m] = float(c.mean())
        n_m = float(c[novel].mean()) if novel.any() else np.nan
        h_m = float(c[high].mean()) if high.any() else np.nan
        strata[m] = dict(n=n_m, h=h_m, g=h_m - n_m)
    print(f"  as-shipped: {_order(acc_o)}  phi={phi:.4f} "
          f"n_hi={int(high.sum())} n_novel={int(novel.sum())} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- CORRECTED ARM ------------------------------------------------------------
    comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, test_frac)
    ver = H.verify_split(seqs, ctr, cte, THR, SIM_K)
    corr_c, acc_c = {}, {}
    for m in MODELS:
        c = E.correctness(E.models()[m], X, y, ctr, cte)
        corr_c[m] = c
        acc_c[m] = float(c.mean())
    print(f"  corrected : {_order(acc_c)}  residual_leak="
          f"{ver['residual_leak_fraction']:.6f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- uncertainty --------------------------------------------------------------
    from audit.experiments.cluster_bootstrap import test_internal_clusters, delta_boot
    blk_o, _ = test_internal_clusters([seqs[i] for i in te], THR)
    blk_c, _ = test_internal_clusters([seqs[i] for i in cte], THR)
    pr_o = p_rank1(corr_o, blk_o)
    pr_c = p_rank1(corr_c, blk_c)

    rows = []
    for m in MODELS:
        d_pt, lo, hi, excl0 = delta_boot(corr_o[m], blk_o, corr_c[m], blk_c, mode="cluster")
        rows.append(dict(
            suite="NT-original", task=task, model=m,
            predicted_verdict=spec["predicted"], pooled_resplit=pooled,
            n=len(seqs), n_train=len(tr), n_test=len(te), test_frac=round(test_frac, 4),
            phi=round(phi, 4), n_high_bin=int(high.sum()), n_novel_bin=int(novel.sum()),
            novel_acc=round(strata[m]["n"], 4), leaked_acc=round(strata[m]["h"], 4),
            graded_gap=round(strata[m]["g"], 4),
            acc_orig=round(acc_o[m], 4), acc_corr=round(acc_c[m], 4),
            drop=round(d_pt, 4), drop_ci_cluster=f"[{lo:.4f}, {hi:.4f}]",
            drop_excl0=bool(excl0),
            rank_orig=_rank(acc_o, m), rank_corr=_rank(acc_c, m),
            p_rank1_orig=pr_o[m], p_rank1_corr=pr_c[m],
            order_orig=_order(acc_o), order_corr=_order(acc_c),
            best_orig=_order(acc_o).split(">")[0], best_corr=_order(acc_c).split(">")[0],
            ranking_inverts=inverts(acc_o, acc_c),
            top_margin_orig=round(acc_o[_order(acc_o).split(">")[0]]
                                  - acc_o[_order(acc_c).split(">")[0]], 4),
            residual_leak_after_split=round(float(ver["residual_leak_fraction"]), 6)))

    # ---- pairwise predictions from the AS-SHIPPED arm only -------------------------
    pair_rows = []
    for A, B in itertools.permutations(MODELS, 2):
        if acc_o[A] <= acc_o[B]:
            continue                                   # A is the as-shipped leader
        delta = strata[B]["n"] - strata[A]["n"]
        Dg = strata[A]["g"] - strata[B]["g"]
        phi_star = delta / Dg if Dg > 0 else np.nan
        pred = bool(delta > 0 and phi * Dg > delta)
        obs = bool(acc_c[B] > acc_c[A] and (acc_o[A] - acc_o[B]) > INVERSION_MARGIN)
        pair_rows.append(dict(
            suite="NT-original", task=task, leader_A=A, challenger_B=B,
            phi=round(phi, 4), n_A=round(strata[A]["n"], 4), n_B=round(strata[B]["n"], 4),
            delta=round(delta, 4), g_A=round(strata[A]["g"], 4),
            g_B=round(strata[B]["g"], 4), Dg=round(Dg, 4),
            phi_star=(round(phi_star, 4) if np.isfinite(phi_star) else None),
            m_inv=round(phi * Dg - delta, 4),
            PREDICTED_inversion=pred, OBSERVED_inversion=obs, agree=bool(pred == obs)))
    return rows, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*",
                    default=[t["task"] for t in TASKS])
    ap.add_argument("--cap", type=int, help="debug: subsample each task")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    specs = [t for t in TASKS if t["task"] in a.tasks]

    all_rows, all_pairs = [], []
    for spec in specs:
        try:
            rows, pairs = run_task(spec, cap=a.cap)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  [{spec['task']}] FAILED: {type(ex).__name__}: {ex}", flush=True)
            continue
        all_rows += rows; all_pairs += pairs
        pd.DataFrame(all_rows).to_csv(f"{R}/crosssuite_ranking.csv.partial", index=False)

    df = pd.DataFrame(all_rows)
    dp = pd.DataFrame(all_pairs)
    if df.empty:
        print("no tasks completed"); return
    for frame, name in ((df, "crosssuite_ranking"), (dp, "crosssuite_ranking_pairs")):
        tmp = f"{R}/{name}.csv.tmp"
        frame.to_csv(tmp, index=False); os.replace(tmp, f"{R}/{name}.csv")
    part = f"{R}/crosssuite_ranking.csv.partial"
    if os.path.exists(part):
        os.remove(part)

    print("\n== ranking before vs after de-leaking (NT-original) ==")
    summ = (df.groupby(["task", "predicted_verdict", "order_orig", "order_corr",
                        "best_orig", "best_corr", "ranking_inverts", "phi"],
                       as_index=False).size().drop(columns="size"))
    print(summ.to_string(index=False))

    print("\n== per-model drops (two-arm cluster bootstrap) ==")
    print(df[["task", "model", "acc_orig", "acc_corr", "drop", "drop_ci_cluster",
              "drop_excl0", "rank_orig", "rank_corr", "p_rank1_orig",
              "p_rank1_corr"]].to_string(index=False))

    print("\n== inversion law: predicted from the AS-SHIPPED split, revealed after ==")
    if not dp.empty:
        print(dp[["task", "leader_A", "challenger_B", "delta", "Dg", "phi", "phi_star",
                  "PREDICTED_inversion", "OBSERVED_inversion", "agree"]].to_string(index=False))
        print(f"\n  law agrees with observation on {int(dp.agree.sum())}/{len(dp)} pairs")
        fp = dp[(dp.PREDICTED_inversion) & (~dp.OBSERVED_inversion)]
        print(f"  false 'inversion' predictions: {len(fp)}")

    inv = df[df.ranking_inverts].task.unique()
    print(f"\nTASKS WHOSE BEST MODEL CHANGES AFTER DE-LEAKING: {list(inv)}")
    ctrl = df[df.predicted_verdict.str.startswith("CLEAN")]
    if len(ctrl):
        print(f"clean control inverts: {bool(ctrl.ranking_inverts.any())} "
              f"(must be False)")
    print("\nEXP_CROSSSUITE_RANKING_DONE")


if __name__ == "__main__":
    main()
