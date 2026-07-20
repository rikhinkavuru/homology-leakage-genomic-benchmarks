#!/usr/bin/env python3
"""
exp_spread_control.py -- a spread control for the "23 accuracy points" claim.

THE CLAIM UNDER TEST (main_resubmission.tex, Nine learners paragraph):
    "the as-shipped split leaves human_enhancers_ensembl's top three of nine learners
     within 0.0015, yet corrected they span 23 points ... the benchmark is not merely
     crowning the wrong model; it is blind to enormous real differences between models
     it ranks as equals."

The standing objection is that P4 controls DROPS on a clean dataset, not top-k SPREAD,
and that spread expands mechanically as accuracy falls off a ceiling: three models
pinned near an accuracy ceiling are forced together, and any manipulation that lowers
the operating point lets them separate whether or not the separation is "real".

Two arms:

  ARM 1 -- clean-dataset spread control (--arm roster)
      Nine-learner roster on human_enhancers_cohn and demo_human_or_worm, as-shipped
      and after the SAME whole-cluster re-split, reporting top-3 spread in both arms.
      If spread expands on a clean dataset under a re-split that changes nothing else,
      the re-split itself expands spread and the ensembl number is not evidence about
      models.  Spread is reported three ways, because they are not the same statistic:
        spread_asshipped_trio_orig : max-min over the AS-SHIPPED top 3, as-shipped split
        spread_asshipped_trio_corr : max-min over those SAME three, corrected split
                                     (this is the paper's "23 points")
        spread_corrected_trio_corr : max-min over the CORRECTED top 3, corrected split

  ARM 2 -- difficulty-matched control (--arm matched)
      Subsample the AS-SHIPPED human_enhancers_ensembl test set down to the corrected
      split's accuracy level and re-measure the spread of the as-shipped top three.
      The subsample is drawn on a difficulty axis computed from a HELD-OUT panel of
      models (LR, LinearSVC, HGB, GaussianNB) -- never from the three models whose
      spread is being measured, which would build the answer into the design.  Bands
      are re-weighted until the trio's mean accuracy matches the corrected operating
      point; within bands the draw is uniform.  200 draws give a spread distribution.

      Reading:  if difficulty-matched spread at the corrected operating point is close
      to 23 points, the spread expansion is a ceiling artefact and the sentence
      "blind to enormous real differences" must go.  If it stays near the as-shipped
      0.0015-0.02 range, the expansion is attributable to de-leaking, not to the
      operating point.

Out: results/spread_control.csv
"""
from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_roster import make_roster, MEMORIZERS, LINEARS

R = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "datacache", "spread_control")
SEED = 0
THR, SIM_K, FEAT_K = 0.7, 8, 6
BOOT = 2000
BSEED = 20240524


# ---------------------------------------------------------------- spread stats
def _spread(accs):
    return float(max(accs) - min(accs))


def _cluster_boot_spread(corr_map, trio, blocks, boot=BOOT, seed=BSEED):
    """Cluster bootstrap CI for max-min accuracy over `trio`, resampling whole test
    clusters. The three models share the resampled examples, so the shared difficulty
    of a draw cancels and the interval reflects only their disagreement."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(blocks)
    idx_by = [np.where(blocks == u)[0] for u in uniq]
    sizes = np.array([len(i) for i in idx_by], float)
    sums = {m: np.array([corr_map[m][i].sum() for i in idx_by]) for m in trio}
    G = len(uniq)
    out = np.empty(boot)
    for b in range(boot):
        pick = rng.randint(0, G, G)
        n = sizes[pick].sum()
        out[b] = _spread([sums[m][pick].sum() / n for m in trio])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------- arm 1
def run_roster_arm(d, cap, rows):
    t0 = time.time()
    seqs, y, tr, te, tf = E.load(d, full=(cap is None), cap=cap or 20000)
    y = np.asarray(y)
    print(f"[{d}] n={len(seqs)} train={len(tr)} test={len(te)}", flush=True)
    X = E.featurize(seqs, FEAT_K)

    comp = E.cached_clusters(d, seqs, THR, SIM_K)
    if len(comp) != len(seqs):
        print("   (cluster cache size mismatch -- recomputing)", flush=True)
        comp = E.clusters(seqs, THR, SIM_K)
    ctr, cte = E.assign(comp, y, SEED, tf)

    from audit.experiments.cluster_bootstrap import test_internal_clusters
    blk_o, _ = test_internal_clusters([seqs[i] for i in te], THR)
    blk_c, _ = test_internal_clusters([seqs[i] for i in cte], THR)

    names = list(make_roster())
    acc_o, acc_c, corr_o, corr_c = {}, {}, {}, {}
    for m in names:
        t1 = time.time()
        corr_o[m] = E.correctness(make_roster()[m], X, y, tr, te)
        corr_c[m] = E.correctness(make_roster()[m], X, y, ctr, cte)
        acc_o[m], acc_c[m] = float(corr_o[m].mean()), float(corr_c[m].mean())
        print(f"   {m:11s} orig={acc_o[m]:.4f} corr={acc_c[m]:.4f} "
              f"({time.time()-t1:.0f}s)", flush=True)

    trio_o = sorted(names, key=lambda m: -acc_o[m])[:3]
    trio_c = sorted(names, key=lambda m: -acc_c[m])[:3]

    def emit(arm, trio, accs, corr, blocks, n_test, note):
        lo, hi = _cluster_boot_spread(corr, trio, blocks)
        rows.append(dict(
            experiment="roster_spread", dataset=d, leaky=(d in E.LEAKY), n=len(seqs),
            arm=arm, trio=">".join(trio), n_test=n_test,
            acc1=round(accs[trio[0]], 4), acc2=round(accs[trio[1]], 4),
            acc3=round(accs[trio[2]], 4),
            top3_spread=round(_spread([accs[m] for m in trio]), 4),
            spread_ci_lo=round(lo, 4), spread_ci_hi=round(hi, 4),
            mean_acc=round(float(np.mean([accs[m] for m in trio])), 4),
            note=note))

    emit("as_shipped_split/as_shipped_top3", trio_o, acc_o, corr_o, blk_o, len(te),
         "top-3 chosen and scored on the as-shipped split")
    emit("corrected_split/as_shipped_top3", trio_o, acc_c, corr_c, blk_c, len(cte),
         "SAME three models scored on the corrected split -- the paper's 23-point statistic")
    emit("corrected_split/corrected_top3", trio_c, acc_c, corr_c, blk_c, len(cte),
         "top-3 chosen and scored on the corrected split")
    print(f"  [{d}] done in {time.time()-t0:.0f}s", flush=True)

    np.savez(os.path.join(CACHE, f"corr_{d}_n{len(seqs)}.npz"),
             **{f"o_{m}": corr_o[m] for m in names},
             **{f"c_{m}": corr_c[m] for m in names})
    return rows


# ---------------------------------------------------------------- arm 2
TRIO_PANEL = ["MLP", "kNN-1", "ExtraTrees"]          # ensembl as-shipped top 3
DIFF_PANEL = ["LR", "LinearSVC", "GaussianNB"]        # held-out difficulty panel


def _ensembl_correctness(cache_dir):
    """Per-example as-shipped correctness on human_enhancers_ensembl for the trio and
    the held-out difficulty panel. Checkpointed per model -- these are full-scale fits
    (n=154,842) on a contended machine, so a lost process must not cost a refit.
    Cheap models are fitted first so a partial run still yields the difficulty axis."""
    d = "human_enhancers_ensembl"
    order = DIFF_PANEL + TRIO_PANEL
    paths = {m: os.path.join(cache_dir, f"corr_ensembl_{m}.npy") for m in order}
    todo = [m for m in order if not os.path.exists(paths[m])]
    if todo:
        seqs, y, tr, te, tf = E.load(d, full=True)
        y = np.asarray(y)
        print(f"[ensembl] n={len(seqs)} featurizing k={FEAT_K}", flush=True)
        X = E.featurize(seqs, FEAT_K)
        for m in todo:
            t1 = time.time()
            c = E.correctness(make_roster()[m], X, y, tr, te)
            np.save(paths[m], c)
            print(f"   {m:11s} orig={c.mean():.4f} ({time.time()-t1:.0f}s)", flush=True)
        del X
    return {m: np.load(paths[m]) for m in order}


def run_matched_arm(rows, target_acc, draws=200):
    d = "human_enhancers_ensembl"
    os.makedirs(CACHE, exist_ok=True)
    corr = _ensembl_correctness(CACHE)
    n_test = len(corr[TRIO_PANEL[0]])
    trio_acc_full = {m: float(corr[m].mean()) for m in TRIO_PANEL}
    print(f"[matched] as-shipped trio {trio_acc_full}, n_test={n_test}", flush=True)

    # difficulty band = number of held-out panel models that get the example right
    band = np.zeros(n_test, int)
    for m in DIFF_PANEL:
        band += corr[m].astype(int)
    trio_mean = np.mean([corr[m] for m in TRIO_PANEL], axis=0)
    band_acc = np.array([trio_mean[band == b].mean() if (band == b).any() else np.nan
                         for b in range(len(DIFF_PANEL) + 1)])
    band_n = np.array([(band == b).sum() for b in range(len(DIFF_PANEL) + 1)])
    print(f"[matched] band n={band_n.tolist()}  trio acc by band="
          f"{np.round(band_acc, 4).tolist()}", flush=True)

    # Re-weight bands with a single monotone tilt exp(-lam*band) and solve for lam so
    # that the trio's mean accuracy equals the corrected operating point. One free
    # parameter, so the subsample is matched on accuracy and on nothing else.
    def acc_at(lam):
        w = np.exp(-lam * np.arange(len(band_n))) * band_n
        w = w / w.sum()
        return float(np.nansum(w * band_acc))

    lo, hi = 0.0, 50.0
    if acc_at(hi) > target_acc:
        raise SystemExit(f"cannot reach {target_acc}: floor is {acc_at(hi):.4f}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if acc_at(mid) > target_acc:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    w = np.exp(-lam * np.arange(len(band_n))) * band_n
    w = w / w.sum()
    print(f"[matched] lam={lam:.4f} -> trio mean acc {acc_at(lam):.4f} "
          f"(target {target_acc:.4f}); band weights {np.round(w,4).tolist()}", flush=True)

    # draw difficulty-matched subsamples of the SAME size as the corrected test set
    rng = np.random.RandomState(BSEED)
    idx_by_band = [np.where(band == b)[0] for b in range(len(band_n))]
    n_draw = int(n_test * 0.5)   # half the as-shipped test set, so every band can be
                                 # drawn without replacement at its re-weighted rate
    # Diagnostic the arm needs to be honest about: difficulty on a LEAKED test set is
    # correlated with novelty, so a difficulty tilt partly re-derives the novel-only
    # stratum. Report how far it goes, per band and in the drawn subsample.
    sim = np.load(os.path.join(HERE, "datacache",
                               "sim_human_enhancers_ensembl_k8_jaccard.npy"))
    leak_by_band = [float((sim[band == b] >= 0.9).mean()) if (band == b).any() else np.nan
                    for b in range(len(band_n))]
    leak_full = float((sim >= 0.9).mean())
    leak_matched = float(np.sum(w * np.nan_to_num(leak_by_band)))
    print(f"[matched] leak fraction (sim>=0.9): full test {leak_full:.4f}, "
          f"by band {np.round(leak_by_band,4).tolist()}, "
          f"difficulty-matched draw {leak_matched:.4f}", flush=True)

    spreads, accs_draw = [], {m: [] for m in TRIO_PANEL}
    for _ in range(draws):
        counts = rng.multinomial(n_draw, w)
        pick = np.concatenate([
            rng.choice(idx_by_band[b], size=min(counts[b], len(idx_by_band[b])),
                       replace=False)
            for b in range(len(band_n)) if counts[b] > 0 and len(idx_by_band[b]) > 0])
        a = {m: float(corr[m][pick].mean()) for m in TRIO_PANEL}
        for m in TRIO_PANEL:
            accs_draw[m].append(a[m])
        spreads.append(_spread(list(a.values())))
    spreads = np.array(spreads)
    mean_acc = float(np.mean([np.mean(accs_draw[m]) for m in TRIO_PANEL]))
    rows.append(dict(
        experiment="difficulty_matched", dataset=d, leaky=True, n=154842,
        arm=f"as_shipped_split/difficulty_matched_to_{target_acc:.4f}",
        trio=">".join(TRIO_PANEL), n_test=n_draw,
        acc1=round(float(np.mean(accs_draw[TRIO_PANEL[0]])), 4),
        acc2=round(float(np.mean(accs_draw[TRIO_PANEL[1]])), 4),
        acc3=round(float(np.mean(accs_draw[TRIO_PANEL[2]])), 4),
        top3_spread=round(float(spreads.mean()), 4),
        spread_ci_lo=round(float(np.percentile(spreads, 2.5)), 4),
        spread_ci_hi=round(float(np.percentile(spreads, 97.5)), 4),
        mean_acc=round(mean_acc, 4),
        note=(f"as-shipped test set re-weighted on a held-out difficulty panel "
              f"({'+'.join(DIFF_PANEL)}) to the corrected operating point "
              f"{target_acc:.4f}; {draws} draws of n={n_draw}; leak fraction "
              f"sim>=0.9 {leak_matched:.4f} in the draw against {leak_full:.4f} "
              f"in the full as-shipped test set")))

    # reference rows: the same trio on the untouched as-shipped and corrected splits
    ros = pd.read_csv(os.path.join(R, "roster_rankings.csv"))
    g = ros[ros.dataset == d].set_index("model")
    for col, arm, note in (("acc_orig", "as_shipped_split/as_shipped_top3",
                            "reference, from roster_rankings.csv"),
                           ("acc_corr", "corrected_split/as_shipped_top3",
                            "reference, the paper's 23-point statistic, "
                            "from roster_rankings.csv")):
        a = [float(g.loc[m, col]) for m in TRIO_PANEL]
        rows.append(dict(experiment="difficulty_matched", dataset=d, leaky=True,
                         n=154842, arm=arm, trio=">".join(TRIO_PANEL),
                         n_test=(30971 if col == "acc_orig" else np.nan),
                         acc1=round(a[0], 4), acc2=round(a[1], 4), acc3=round(a[2], 4),
                         top3_spread=round(_spread(a), 4),
                         spread_ci_lo=np.nan, spread_ci_hi=np.nan,
                         mean_acc=round(float(np.mean(a)), 4), note=note))
    return rows


def run_derive_arm(rows):
    """Point-estimate top-3 spreads read straight off the frozen nine-model roster
    (results/roster_rankings.csv). No refit, so no cluster-bootstrap interval; these
    rows exist so the three datasets are comparable on one table and so the clean
    control is present whether or not the fresh runs complete."""
    ros = pd.read_csv(os.path.join(R, "roster_rankings.csv"))
    for d, g in ros.groupby("dataset"):
        g = g.set_index("model")
        trio_o = g.acc_orig.sort_values(ascending=False).index[:3].tolist()
        trio_c = g.acc_corr.sort_values(ascending=False).index[:3].tolist()
        n = int(g.n.iloc[0])
        for arm, trio, col, note in (
            ("as_shipped_split/as_shipped_top3", trio_o, "acc_orig",
             "top-3 chosen and scored on the as-shipped split"),
            ("corrected_split/as_shipped_top3", trio_o, "acc_corr",
             "SAME three models on the corrected split -- the paper's 23-point statistic"),
            ("corrected_split/corrected_top3", trio_c, "acc_corr",
             "top-3 chosen and scored on the corrected split"),
        ):
            a = [float(g.loc[m, col]) for m in trio]
            rows.append(dict(experiment="roster_spread", dataset=d,
                             leaky=bool(g.leaky.iloc[0]), n=n, arm=arm,
                             trio=">".join(trio), n_test=np.nan,
                             acc1=round(a[0], 4), acc2=round(a[1], 4),
                             acc3=round(a[2], 4), top3_spread=round(_spread(a), 4),
                             spread_ci_lo=np.nan, spread_ci_hi=np.nan,
                             mean_acc=round(float(np.mean(a)), 4),
                             note=note + "; point estimates from roster_rankings.csv"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["roster", "matched", "derive"], required=True)
    ap.add_argument("--datasets", nargs="*",
                    default=["human_enhancers_cohn", "demo_human_or_worm"])
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--target-acc", type=float, default=0.6951,
                    help="corrected operating point to match (RF corrected on ensembl)")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    rows = []
    if a.arm == "roster":
        for d in a.datasets:
            run_roster_arm(d, a.cap, rows)
    elif a.arm == "derive":
        run_derive_arm(rows)
    else:
        run_matched_arm(rows, a.target_acc)
    df = pd.DataFrame(rows)
    path = os.path.join(R, "spread_control.csv")
    if os.path.exists(path):
        prev = pd.read_csv(path)
        key = ["experiment", "dataset", "arm"]
        prev = prev.merge(df[key].drop_duplicates(), on=key, how="left",
                          indicator=True)
        prev = prev[prev._merge == "left_only"].drop(columns="_merge")
        df = pd.concat([prev, df], ignore_index=True)
    tmp = path + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, path)
    print(df.to_string(index=False))
    print("\nEXP_SPREAD_CONTROL_DONE")


if __name__ == "__main__":
    main()
