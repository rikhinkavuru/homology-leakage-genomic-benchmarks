#!/usr/bin/env python3
"""
C6 -- put an interval on the inversion threshold instead of quoting three digits.

main_resubmission.tex currently says a ten-dose sweep "places the first ranking inversion at an
imposed leak fraction of 0.106". That 0.106 is the realised phi at dose 0.30 in
results/dose_response.csv, and it comes from ONE fit at ONE split seed at ONE model seed.
`ranking_inverts` is a discrete outcome computed by comparing two accuracy orderings whose
top-two margins at that dose are on the order of 0.01 -- comfortably inside the noise of a
single fit. A three-digit threshold read off a single realisation of a Bernoulli outcome
is not a measurement.

DESIGN
------
Take the three doses that bracket the observed transition -- 0.25 (no inversion), 0.30
(first inversion), 0.35 (inversion) -- and replicate each across a factorial of

    5 split seeds  x  3 model seeds  =  15 trials per dose, 45 trials total.

SPLIT SEED drives everything data-side: which positives are duplicated, which rows survive
the size/balance downsample, the random as-shipped split, and the whole-cluster assignment
of the corrected split. MODEL SEED drives only `random_state` on LR, LinearSVC, RF and
HGB. Separating them matters because they answer different questions: split-seed variance
is how much the threshold depends on which dataset you happened to construct, model-seed
variance is how much it depends on the fit alone.

Per trial we record the realised phi (>=0.9 near-duplicate test fraction, the quantity
eq. (1) uses), the as-shipped and corrected orderings, and whether the ranking inverts.
Per dose we then report the inversion PROBABILITY with a Clopper-Pearson binomial interval,
plus the split-seed-clustered spread, because the 15 trials are not independent -- they are
5 datasets seen by 3 fits each, so a naive binomial CI on n=15 overstates the information.
Both are reported and the clustered one is the honest one.

THE THRESHOLD AS AN INTERVAL
----------------------------
For each of the 15 (split seed, model seed) replicates we find the lowest dose at which
that replicate inverts. The threshold is quoted as the range of realised phi spanned by
those per-replicate first-inversion points, not as a single number. If a replicate inverts
at the lowest dose we report the threshold as censored below 0.25's phi; if it never
inverts, censored above 0.35's phi. Censoring is reported, not dropped.

If the interval is wide, that IS the finding and it is reported as such.

Run:  python -m audit.experiments.exp_dose_replication
Out:  results/dose_replication_trials.csv     one row per (dose, split seed, model seed)
      results/dose_replication_summary.csv    per-dose inversion probability + CIs
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
DATASET = "human_ocr_ensembl"
CAP = 20000
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
# Models whose random_state is inert. Verified at runtime, not assumed -- see DET_CHECK.
DETERMINISTIC = ["LR", "LinearSVC"]
RF_TREES = 150                       # matches audit/pipeline/run_audit.py:64
DOSES = [0.25, 0.30, 0.35]           # the three doses bracketing the committed transition
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEEDS = [0, 1, 2]
TRIALS = os.path.join(R, "dose_replication_trials.csv")
SUMMARY = os.path.join(R, "dose_replication_summary.csv")


RF_JOBS = -1          # overridden by --rf-jobs when several split seeds run concurrently


def make_models(seed):
    """The paper's four models with random_state driven by `seed`. Mirrors
    run_audit.make_models + run_extended_models.make_models exactly, except that the
    seed is a parameter instead of the module constant MODEL_SEED=0.

    n_jobs on the forest is a scheduling knob, not a modelling one: it changes wall-clock
    and nothing else, since RandomForestClassifier is deterministic given random_state."""
    return {
        "LR": make_pipeline(Normalizer(norm="l2"),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
        "LinearSVC": make_pipeline(Normalizer(norm="l2"),
                                   LinearSVC(C=1.0, dual=False, max_iter=5000,
                                             random_state=seed)),
        "RF": RandomForestClassifier(n_estimators=RF_TREES, n_jobs=RF_JOBS,
                                     random_state=seed),
        "HGB": HistGradientBoostingClassifier(random_state=seed),
    }


def cached_dose_clusters(dose, split_seed, seqs_m):
    """Near-duplicate components for one (dose, split seed) dataset realisation.

    Clustering 20,000 sequences costs ~160 s and is by far the dominant term, while the
    dataset it runs on is a deterministic function of (dose, split seed). Caching it keeps
    a re-run of the model-seed loop cheap and lets the 15 realisations be built once."""
    tag = f"dosecomp_{DATASET}_d{dose:.4f}_ss{split_seed}_thr0.7_k8.npy"
    path = os.path.join(E.DATACACHE, tag)
    if os.path.exists(path):
        comp = np.load(path)
        if len(comp) == len(seqs_m):
            return comp
    comp = E.clusters(seqs_m, 0.7, 8)
    np.save(path, comp)
    return comp


def build_dose(seqs, y, tf, target, n_ctrl, split_seed):
    """Impose the duplicated-coordinate construction at `target`, match the control's size
    and balance, and draw the as-shipped split -- all driven by `split_seed`.

    Identical arithmetic to exp_dose_response.one_dose; the only change is that the RNG
    and the split are seeded by split_seed rather than by the frozen SEED=0."""
    y = np.asarray(y)
    rng = np.random.RandomState(split_seed)
    rate = target / (2 - target)                    # 2r/(1+r) = target
    pos = np.where(y == y.max())[0]
    dup = rng.choice(pos, size=int(round(rate * len(pos))), replace=False)
    seqs_d = list(seqs) + [seqs[i] for i in dup]
    y_d = np.concatenate([y, y[dup]])
    pos2 = np.where(y_d == y_d.max())[0]
    neg2 = np.where(y_d != y_d.max())[0]
    half = min(n_ctrl // 2, len(pos2), len(neg2))
    sel = np.sort(np.concatenate([rng.choice(pos2, half, replace=False),
                                  rng.choice(neg2, half, replace=False)]))
    seqs_m = [seqs_d[i] for i in sel]
    y_m = y_d[sel]
    perm = rng.permutation(len(seqs_m))
    cut = int(round((1 - tf) * len(seqs_m)))
    return seqs_m, y_m, perm[:cut], perm[cut:]


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def merge_shards():
    """Concatenate every results/dose_replication_trials_*.csv shard into the canonical
    trials file. Concurrent workers each write their own shard, because two processes
    read-modify-writing one CSV lose rows."""
    import glob
    paths = sorted(glob.glob(os.path.join(R, "dose_replication_trials_*.csv")))
    if not paths:
        raise SystemExit("no shards to merge")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["dose_target_share", "split_seed", "model_seed"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    df = df.sort_values(key).reset_index(drop=True)
    print(f"merged {len(paths)} shards: {before} rows -> {len(df)} unique trials")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="merge worker shards and write the summary, running no fits")
    ap.add_argument("--doses", nargs="*", type=float, default=DOSES)
    ap.add_argument("--split-seeds", nargs="*", type=int, default=SPLIT_SEEDS)
    ap.add_argument("--model-seeds", nargs="*", type=int, default=MODEL_SEEDS)
    ap.add_argument("--rf-jobs", type=int, default=-1,
                    help="forest n_jobs; lower it when running split seeds concurrently")
    ap.add_argument("--out-suffix", default="",
                    help="write to results/dose_replication_trials<suffix>.csv so "
                         "concurrent workers cannot race on the same file")
    ap.add_argument("--no-summary", action="store_true",
                    help="write trials only; summarise later once shards are merged")
    a = ap.parse_args()
    global RF_JOBS, TRIALS, SUMMARY
    RF_JOBS = a.rf_jobs
    if a.out_suffix:
        TRIALS = os.path.join(R, f"dose_replication_trials{a.out_suffix}.csv")
        SUMMARY = os.path.join(R, f"dose_replication_summary{a.out_suffix}.csv")
    os.makedirs(R, exist_ok=True)
    t0 = time.time()

    if a.merge:
        df = merge_shards()
        df.to_csv(TRIALS + ".tmp", index=False)
        os.replace(TRIALS + ".tmp", TRIALS)
        summarise(df)
        print(f"\nEXP_DOSE_REPLICATION_DONE {time.time()-t0:.0f}s")
        return

    seqs, y, _otr, _ote, tf = E.load(DATASET, full=False, cap=CAP)
    y = np.asarray(y)
    n_ctrl = len(seqs)
    print(f"== C6 dose replication on {DATASET} (n={n_ctrl}) ==")
    print(f"   {len(a.doses)} doses x {len(a.split_seeds)} split seeds x "
          f"{len(a.model_seeds)} model seeds = "
          f"{len(a.doses)*len(a.split_seeds)*len(a.model_seeds)} trials\n", flush=True)

    rows = []
    det_checked, det_report = False, []
    for dose in a.doses:
        for ss in a.split_seeds:
            t1 = time.time()
            seqs_m, y_m, tr, te = build_dose(seqs, y, tf, dose, n_ctrl, ss)
            # Everything below the model loop depends only on (dose, split seed), so it is
            # computed once per dataset realisation rather than once per fit.
            X = E.featurize(seqs_m, 6)
            sim = E.max_sim_to_train(seqs_m, tr, te, k=8, mode="jaccard")
            phi = float((sim >= 0.9).mean())
            comp = cached_dose_clusters(dose, ss, seqs_m)
            ctr, cte = E.assign(comp, y_m, ss, len(te) / len(seqs_m))
            # Only RF and HGB are stochastic given the data. LogisticRegression(lbfgs) and
            # LinearSVC(dual=False, liblinear primal) are deterministic solvers whose
            # random_state is inert, so refitting them once per model seed would triple
            # the most expensive linear fit in the design and change nothing. We do not
            # assume that -- DET_CHECK below refits both at a second seed on the first
            # realisation and records whether the accuracies are bit-identical.
            det = {}
            for m in DETERMINISTIC:
                det[(m, "o")] = float(E.correctness(make_models(a.model_seeds[0])[m],
                                                    X, y_m, tr, te).mean())
                det[(m, "c")] = float(E.correctness(make_models(a.model_seeds[0])[m],
                                                    X, y_m, ctr, cte).mean())
            if not det_checked and len(a.model_seeds) > 1:
                for m in DETERMINISTIC:
                    alt = float(E.correctness(make_models(a.model_seeds[1])[m],
                                              X, y_m, tr, te).mean())
                    det_report.append((m, det[(m, "o")], alt, alt == det[(m, "o")]))
                    print(f"  DET_CHECK {m}: seed{a.model_seeds[0]}={det[(m,'o')]:.6f} "
                          f"seed{a.model_seeds[1]}={alt:.6f} "
                          f"{'identical' if alt == det[(m,'o')] else 'DIFFERS'}", flush=True)
                det_checked = True
            t_data = time.time() - t1
            for ms in a.model_seeds:
                acc_o, acc_c = {}, {}
                for m in MODELS:
                    if m in DETERMINISTIC:
                        acc_o[m], acc_c[m] = det[(m, "o")], det[(m, "c")]
                        continue
                    acc_o[m] = float(E.correctness(make_models(ms)[m], X, y_m, tr, te).mean())
                    acc_c[m] = float(E.correctness(make_models(ms)[m], X, y_m,
                                                   ctr, cte).mean())
                order_o = ">".join(sorted(MODELS, key=lambda m: -acc_o[m]))
                order_c = ">".join(sorted(MODELS, key=lambda m: -acc_c[m]))
                lead_o, lead_c = order_o.split(">")[0], order_c.split(">")[0]
                rank_o = 1 + sum(acc_o[m] > acc_o["RF"] for m in MODELS)
                rank_c = 1 + sum(acc_c[m] > acc_c["RF"] for m in MODELS)
                rows.append(dict(
                    dose_target_share=dose, split_seed=ss, model_seed=ms,
                    n=len(seqs_m), phi=round(phi, 4),
                    leak_jaccard_0p7=round(float((sim > 0.7).mean()), 4),
                    **{f"acc_orig_{m}": round(acc_o[m], 4) for m in MODELS},
                    **{f"acc_corr_{m}": round(acc_c[m], 4) for m in MODELS},
                    # the margin the discrete outcome actually turns on
                    top2_margin_orig=round(sorted(acc_o.values())[-1]
                                           - sorted(acc_o.values())[-2], 4),
                    rf_drop=round(acc_o["RF"] - acc_c["RF"], 4),
                    order_orig=order_o, order_corr=order_c,
                    leader_orig=lead_o, leader_corr=lead_c,
                    rf_rank_orig=rank_o, rf_rank_corr=rank_c,
                    ranking_inverts=bool(lead_o != lead_c),
                    # provenance for the "refit only the stochastic learners" shortcut
                    deterministic_models=";".join(DETERMINISTIC),
                    deterministic_verified=(";".join(f"{m}:{'same' if ok else 'DIFFERS'}"
                                                     for m, _p, _q, ok in det_report)
                                            or "not_checked"),
                    dataset=DATASET))
                print(f"  dose={dose:.2f} ss={ss} ms={ms} phi={phi:.4f} "
                      f"RF {acc_o['RF']:.4f}->{acc_c['RF']:.4f} "
                      f"rank {rank_o}->{rank_c} margin={rows[-1]['top2_margin_orig']:.4f} "
                      f"{lead_o}=>{lead_c} "
                      f"{'INVERT' if rows[-1]['ranking_inverts'] else '-'}", flush=True)
            print(f"    [dose {dose:.2f} ss {ss} data {t_data:.0f}s "
                  f"total {time.time()-t1:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(TRIALS):
        prev = pd.read_csv(TRIALS)
        key = ["dose_target_share", "split_seed", "model_seed"]
        merged = prev.merge(df[key].drop_duplicates(), on=key, how="left", indicator=True)
        df = pd.concat([prev[merged["_merge"].to_numpy() == "left_only"], df],
                       ignore_index=True)
    df = df.sort_values(["dose_target_share", "split_seed", "model_seed"]).reset_index(drop=True)
    df.to_csv(TRIALS + ".tmp", index=False)
    os.replace(TRIALS + ".tmp", TRIALS)
    if a.no_summary:
        print(f"\nwrote {len(df)} trials to {TRIALS} (summary deferred)")
        print(f"EXP_DOSE_REPLICATION_DONE {time.time()-t0:.0f}s")
        return

    summarise(df)
    print(f"\nEXP_DOSE_REPLICATION_DONE {time.time()-t0:.0f}s")


def summarise(df):
    """Per-dose inversion probability, and the threshold quoted as an interval."""
    # ---- per-dose inversion probability -------------------------------------------
    srows = []
    for dose, g in df.groupby("dose_target_share"):
        k, n = int(g.ranking_inverts.sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        # Cluster-aware alternative: the 15 trials are 5 datasets x 3 fits, so treat the
        # per-split-seed inversion rate as the unit and bootstrap over split seeds.
        per_ss = g.groupby("split_seed").ranking_inverts.mean().to_numpy()
        rng = np.random.RandomState(0)
        bs = np.array([rng.choice(per_ss, len(per_ss), replace=True).mean()
                       for _ in range(10000)])
        clo, chi = np.percentile(bs, [2.5, 97.5])
        srows.append(dict(
            dose_target_share=dose, phi_mean=round(float(g.phi.mean()), 4),
            phi_min=round(float(g.phi.min()), 4), phi_max=round(float(g.phi.max()), 4),
            n_trials=n, n_split_seeds=int(g.split_seed.nunique()),
            n_model_seeds=int(g.model_seed.nunique()),
            n_inversions=k, inversion_prob=round(k / n, 4),
            binomial_ci_lo=round(lo, 4), binomial_ci_hi=round(hi, 4),
            split_cluster_ci_lo=round(float(clo), 4),
            split_cluster_ci_hi=round(float(chi), 4),
            per_split_seed_rates=";".join(f"{v:.2f}" for v in per_ss),
            mean_top2_margin_orig=round(float(g.top2_margin_orig.mean()), 4),
            mean_rf_drop=round(float(g.rf_drop.mean()), 4)))
    sdf = pd.DataFrame(srows).sort_values("dose_target_share")

    # ---- threshold as an interval ---------------------------------------------------
    doses_sorted = sorted(df.dose_target_share.unique())
    phi_by_dose = {d: float(df[df.dose_target_share == d].phi.mean()) for d in doses_sorted}
    firsts, censored_lo, censored_hi = [], 0, 0
    for (ss, ms), g in df.groupby(["split_seed", "model_seed"]):
        g = g.sort_values("dose_target_share")
        inv = g[g.ranking_inverts]
        if len(inv) == 0:
            censored_hi += 1
        else:
            d0 = float(inv.dose_target_share.iloc[0])
            if d0 == doses_sorted[0]:
                censored_lo += 1
            firsts.append(phi_by_dose[d0])
    thr = dict(
        dose_target_share="THRESHOLD_SUMMARY",
        n_trials=int(df.groupby(["split_seed", "model_seed"]).ngroups),
        n_inversions=len(firsts),
        phi_min=round(min(firsts), 4) if firsts else None,
        phi_max=round(max(firsts), 4) if firsts else None,
        phi_mean=round(float(np.mean(firsts)), 4) if firsts else None,
        per_split_seed_rates=(f"censored_below={censored_lo};censored_above={censored_hi};"
                              f"single_fit_committed_value=0.106"))
    sdf = pd.concat([sdf, pd.DataFrame([thr])], ignore_index=True)
    sdf.to_csv(SUMMARY + ".tmp", index=False)
    os.replace(SUMMARY + ".tmp", SUMMARY)

    print("\n== per-dose inversion probability ==")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
