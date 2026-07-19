#!/usr/bin/env python3
"""
Expanded model roster: is the reordering an artifact of comparing four sklearn models?

The standing objection to the ranking claim is its comparison set -- four classical
models, of which three function as foils for one designated memorizer. This module
widens the roster to nine learners spanning the memorization axis end to end, on the
same splits, with the same decision rule, and asks whether the reordering survives and
whether it lands where the memorization account says it should.

THE ROSTER, ordered by a priori memorization propensity
-------------------------------------------------------
  kNN-1        1-nearest neighbour            -- the DEFINITIONAL memorizer: its
                                                 prediction on a near-duplicate IS the
                                                 training label. Strongest a priori
                                                 prediction in the study.
  kNN-15       15-nearest neighbours          -- same mechanism, averaged, so weaker
  RF           RandomForest(150), default     -- the paper's memorizer
  ExtraTrees   ExtraTrees(150), default       -- tests whether the effect is RF-specific
                                                 or generic to unregularized tree ensembles
  MLP          1 hidden layer, 128 units      -- a neural learner with no convolution
  HGB          HistGradientBoosting           -- highest nominal capacity, boosted
  LinearSVC    linear SVM                     -- the paper's corrected-split winner
  LR           logistic regression            -- linear baseline
  GaussianNB   naive Bayes                    -- lowest capacity floor

PRE-COMMITTED PREDICTIONS (recorded here before the run; scored in the output)
-----------------------------------------------------------------------------
  P1  kNN-1 has the largest graded memorization gap g of any model on the leaky sets.
  P2  kNN-1 has the largest split-induced drop of any model on the leaky sets.
  P3  The memorizers (kNN-1, kNN-15, RF, ExtraTrees) rank ABOVE the linear models on
      the as-shipped split and BELOW them after correction, on human_enhancers_ensembl.
  P4  On the clean control, no model's drop interval excludes zero.
If P1/P2 fail, the memorization account of the mechanism is wrong and we report that.

Every model is trained from scratch on each split; none is pretrained, so no model here
carries the pretraining-overlap confound that scopes foundation models out of this study.

Run:  python -m audit.experiments.exp_roster [--datasets ...] [--cap N]
Out:  results/roster_rankings.csv, results/roster_predictions.csv
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.core import homology_split as H

R = os.path.join(HERE, "results")
SEED = 0
THR, SIM_K, FEAT_K = 0.7, 8, 6
DATASETS = ["human_nontata_promoters", "human_enhancers_ensembl", "human_enhancers_cohn"]
LEAKY = {"human_nontata_promoters", "human_enhancers_ensembl"}
INVERSION_MARGIN = 0.01          # report_card.py:55 convention


def make_roster(n_jobs=4):
    """Nine learners, all from scratch, all on the same k=6 count features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline
    return {
        "kNN-1": KNeighborsClassifier(n_neighbors=1, n_jobs=n_jobs),
        "kNN-15": KNeighborsClassifier(n_neighbors=15, n_jobs=n_jobs),
        "RF": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=n_jobs),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=150, random_state=SEED, n_jobs=n_jobs),
        "MLP": make_pipeline(Normalizer(norm="l2"),
                             MLPClassifier(hidden_layer_sizes=(128,), max_iter=60,
                                           random_state=SEED, early_stopping=True)),
        "HGB": HistGradientBoostingClassifier(random_state=SEED),
        "LinearSVC": make_pipeline(Normalizer(norm="l2"),
                                   LinearSVC(C=1.0, dual=False, max_iter=5000,
                                             random_state=SEED)),
        "LR": LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=n_jobs),
        "GaussianNB": GaussianNB(),
    }


MEMORIZERS = {"kNN-1", "kNN-15", "RF", "ExtraTrees"}
LINEARS = {"LR", "LinearSVC"}


def _paired_ci(diff_vec, blocks, boot=2000, seed=20240524, lo=2.5, hi=97.5):
    """Cluster-bootstrap CI for the mean of a PAIRED per-example difference (model A
    correct minus model B correct on the same test examples). Paired, so the two models'
    shared difficulty cancels and the interval reflects only their disagreement."""
    rng = np.random.RandomState(seed)
    d = np.asarray(diff_vec, float)
    uniq = np.unique(blocks)
    sums = np.array([d[blocks == u].sum() for u in uniq])
    sizes = np.array([(blocks == u).sum() for u in uniq], float)
    G = len(uniq)
    out = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, G)
        out[b] = sums[pick].sum() / sizes[pick].sum()
    return float(np.percentile(out, lo)), float(np.percentile(out, hi))


def _order(acc, names):
    return ">".join(sorted(names, key=lambda m: -acc[m]))


def run_dataset(d, cap=None, boot_models=("kNN-1", "RF", "LinearSVC", "LR")):
    t0 = time.time()
    full = d in LEAKY
    seqs, y, tr, te, tf = E.load(d, full=full and cap is None,
                                 cap=cap or 20000)
    y = np.asarray(y)
    print(f"[{d}] n={len(seqs)} train={len(tr)} test={len(te)}", flush=True)
    X = E.featurize(seqs, FEAT_K)

    sim = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    phi = float((sim >= 0.9).mean())
    novel, high = sim < 0.5, sim >= 0.9

    # expkit's cluster cache is keyed by (dataset, threshold, k) only, NOT by the number
    # of sequences loaded, so a capped load would silently receive the full-scale
    # component array. Use the cache only when the lengths agree.
    comp = E.cached_clusters(d, seqs, THR, SIM_K)
    if len(comp) != len(seqs):
        comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, tf)

    from audit.experiments.cluster_bootstrap import test_internal_clusters, delta_boot
    blk_o, _ = test_internal_clusters([seqs[i] for i in te], THR)
    blk_c, _ = test_internal_clusters([seqs[i] for i in cte], THR)

    roster = make_roster()
    names = list(roster)
    acc_o, acc_c, rows = {}, {}, []
    corr_o, corr_c = {}, {}
    for m in names:
        t1 = time.time()
        c_o = E.correctness(make_roster()[m], X, y, tr, te)
        c_c = E.correctness(make_roster()[m], X, y, ctr, cte)
        acc_o[m], acc_c[m] = float(c_o.mean()), float(c_c.mean())
        corr_o[m], corr_c[m] = c_o, c_c
        n_m = float(c_o[novel].mean()) if novel.any() else np.nan
        h_m = float(c_o[high].mean()) if high.any() else np.nan
        d_pt, lo, hi, excl0 = delta_boot(c_o, blk_o, c_c, blk_c, mode="cluster")
        rows.append(dict(dataset=d, model=m, leaky=d in LEAKY, n=len(seqs),
                         phi=round(phi, 4), n_high_bin=int(high.sum()),
                         novel_acc=round(n_m, 4), leaked_acc=round(h_m, 4),
                         graded_gap=round(h_m - n_m, 4),
                         acc_orig=round(acc_o[m], 4), acc_corr=round(acc_c[m], 4),
                         drop=round(d_pt, 4), drop_ci=f"[{lo:.4f}, {hi:.4f}]",
                         drop_excl0=bool(excl0),
                         family=("memorizer" if m in MEMORIZERS else
                                 "linear" if m in LINEARS else "other")))
        print(f"   {m:11s} orig={acc_o[m]:.4f} corr={acc_c[m]:.4f} "
              f"drop={d_pt:+.4f} {('*' if excl0 else ' ')} gap={h_m-n_m:+.4f} "
              f"({time.time()-t1:.0f}s)", flush=True)

    # An inversion criterion built for four models is too permissive for nine. With a
    # wider roster the top two are often within noise, so "top model changed by >1 pt"
    # fires on a CLEAN control from reshuffling alone (observed: MLP/HGB on
    # human_enhancers_cohn, where every drop interval includes zero). We therefore
    # require, in addition, that the leaders actually SWAP with support: the old leader
    # must lead on the as-shipped split and the new leader must lead after correction,
    # each by a paired cluster-bootstrap difference whose interval excludes zero.
    order_o, order_c = _order(acc_o, names), _order(acc_c, names)
    top_o, top_c = order_o.split(">")[0], order_c.split(">")[0]
    inv, supp_o, supp_c = False, None, None
    if top_o != top_c:
        supp_o = _paired_ci(corr_o[top_o] - corr_o[top_c], blk_o)
        supp_c = _paired_ci(corr_c[top_c] - corr_c[top_o], blk_c)
        inv = bool(supp_o[0] > 0 and supp_c[0] > 0)
    for r in rows:
        r["rank_orig"] = 1 + sum(acc_o[o] > acc_o[r["model"]] for o in names)
        r["rank_corr"] = 1 + sum(acc_c[o] > acc_c[r["model"]] for o in names)
        r["order_orig"], r["order_corr"] = order_o, order_c
        r["top_orig"], r["top_corr"] = top_o, top_c
        r["lead_margin_orig_ci"] = (f"[{supp_o[0]:.4f}, {supp_o[1]:.4f}]" if supp_o else None)
        r["lead_margin_corr_ci"] = (f"[{supp_c[0]:.4f}, {supp_c[1]:.4f}]" if supp_c else None)
        r["ranking_inverts"] = inv
    print(f"  as-shipped: {rows[0]['order_orig']}")
    print(f"  corrected : {rows[0]['order_corr']}")
    print(f"  inverts   : {rows[0]['ranking_inverts']}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def score_predictions(df):
    """Score the four pre-committed predictions against the run."""
    out = []
    for d, g in df[df.leaky].groupby("dataset"):
        gg = g.set_index("model")
        top_gap = gg.graded_gap.idxmax()
        top_drop = gg["drop"].idxmax()
        out.append(dict(dataset=d, prediction="P1 kNN-1 has largest graded gap",
                        expected="kNN-1", observed=top_gap, holds=bool(top_gap == "kNN-1")))
        out.append(dict(dataset=d, prediction="P2 kNN-1 has largest drop",
                        expected="kNN-1", observed=top_drop, holds=bool(top_drop == "kNN-1")))
        if d == "human_enhancers_ensembl":
            mem_o = gg.loc[list(MEMORIZERS & set(gg.index)), "rank_orig"].mean()
            lin_o = gg.loc[list(LINEARS & set(gg.index)), "rank_orig"].mean()
            mem_c = gg.loc[list(MEMORIZERS & set(gg.index)), "rank_corr"].mean()
            lin_c = gg.loc[list(LINEARS & set(gg.index)), "rank_corr"].mean()
            out.append(dict(dataset=d,
                            prediction="P3 memorizers above linears as-shipped, below after",
                            expected="mem_rank<lin_rank then mem_rank>lin_rank",
                            observed=f"as-shipped mem={mem_o:.2f} lin={lin_o:.2f}; "
                                     f"corrected mem={mem_c:.2f} lin={lin_c:.2f}",
                            holds=bool(mem_o < lin_o and mem_c > lin_c)))
    for d, g in df[~df.leaky].groupby("dataset"):
        out.append(dict(dataset=d, prediction="P4 no drop excludes zero on a clean set",
                        expected="none", observed=(", ".join(g[g.drop_excl0].model) or "none"),
                        holds=bool(not g.drop_excl0.any())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--cap", type=int, help="subsample (debug / tractability)")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    rows = []
    for d in a.datasets:
        try:
            rows += run_dataset(d, cap=a.cap)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  [{d}] FAILED: {type(ex).__name__}: {ex}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(f"{R}/roster_rankings.csv.partial", index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing completed"); return
    for frame, name in ((df, "roster_rankings"), (score_predictions(df), "roster_predictions")):
        tmp = f"{R}/{name}.csv.tmp"; frame.to_csv(tmp, index=False); os.replace(tmp, f"{R}/{name}.csv")
    part = f"{R}/roster_rankings.csv.partial"
    if os.path.exists(part):
        os.remove(part)

    print("\n== expanded roster ==")
    print(df[["dataset", "model", "family", "graded_gap", "acc_orig", "acc_corr",
              "drop", "drop_ci", "drop_excl0", "rank_orig", "rank_corr"]].to_string(index=False))
    print("\n== pre-committed predictions ==")
    print(score_predictions(df).to_string(index=False))
    print("\nEXP_ROSTER_DONE")


if __name__ == "__main__":
    main()
