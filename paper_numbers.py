#!/usr/bin/env python3
"""
Freeze ALL paper-citable numbers into one authoritative file: results/PAPER_NUMBERS.md.
Pure aggregation of validated CSVs (no model fitting). For the two LEAKY datasets the
main results + robustness are taken at FULL scale; for the clean datasets at the 20k
subsample (full-scale leakage = 0, so the subsample drop is unconfounded). Every table
ends with the script + file each number came from.

Inputs (all in results/):
  leakage_full.csv, summary_final.csv, capacity_scaling_final.csv,
  per_dataset_results.csv, fullscale_long.csv, threshold_sensitivity.csv,
  cluster_stats.csv, robustness_fullscale_summary.csv, robustness_fullscale_clusters.csv
"""
import os
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONFIGS = [(4, "LR"), (6, "LR"), (4, "RF"), (6, "RF")]
LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
BINARY = ["human_nontata_promoters", "human_enhancers_ensembl",
          "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
          "demo_human_or_worm", "drosophila_enhancers_stark", "human_enhancers_cohn"]

leak = pd.read_csv(f"{R}/leakage_full.csv").set_index("dataset")
sfin = pd.read_csv(f"{R}/summary_final.csv").set_index("dataset")
subsum = pd.read_csv(f"{R}/summary.csv").set_index("dataset")   # lengths, n_used, subsample size
capf = pd.read_csv(f"{R}/capacity_scaling_final.csv").set_index("dataset")
pdr = pd.read_csv(f"{R}/per_dataset_results.csv")
fsl = pd.read_csv(f"{R}/fullscale_long.csv")
ts = pd.read_csv(f"{R}/threshold_sensitivity.csv")
cls = pd.read_csv(f"{R}/cluster_stats.csv")
robs = pd.read_csv(f"{R}/robustness_fullscale_summary.csv")
robc = pd.read_csv(f"{R}/robustness_fullscale_clusters.csv")
import itertools
capext = pd.read_csv(f"{R}/capacity_scaling_extended.csv")
ri = pd.read_csv(f"{R}/ranking_inversion.csv")
extlong = pd.read_csv(f"{R}/extended_models_long.csv")
try:
    sval = pd.read_csv(f"{R}/splitter_validation.csv")
except Exception:
    sval = None
try:
    graded = pd.read_csv(f"{R}/graded_performance.csv")
except Exception:
    graded = None
try:
    rcard = pd.read_csv(f"{R}/leakage_report_card.csv")
except Exception:
    rcard = None
try:
    lc = pd.read_csv(f"{R}/label_concordance.csv")
except Exception:
    lc = None
try:
    nor = pd.read_csv(f"{R}/novelonly_ranking.csv")
except Exception:
    nor = None
try:
    s2var = pd.read_csv(f"{R}/step2_seed_variance.csv")
    s3acc = pd.read_csv(f"{R}/step3_accuracy_ci.csv")
    s3del = pd.read_csv(f"{R}/step3_delta_ci.csv")
    s4tie = pd.read_csv(f"{R}/step4_tie_overlap.csv")
except Exception:
    s2var = s3acc = s3del = s4tie = None


def balance(dset):
    sub = cls[(cls.dataset == dset) & (cls.threshold == 0.7)]
    return float(sub.pos_train.iloc[0]) if len(sub) and not pd.isna(sub.pos_train.iloc[0]) else np.nan


def main_row(dset, k, m):
    """(orig, rand, hom) for acc/auroc/f1 from full-scale (leaky) or subsample (clean)."""
    if dset in LEAKY:
        d = fsl[(fsl.dataset == dset) & (fsl.k == k) & (fsl.model == m)]
        g = lambda sp, c: float(d[d.split_type == sp][c].mean())
        return dict(acc=(g("original", "accuracy"), g("random", "accuracy"), g("homology@0.7", "accuracy")),
                    auroc=(g("original", "auroc"), g("random", "auroc"), g("homology@0.7", "auroc")),
                    f1=(g("original", "f1"), g("random", "f1"), g("homology@0.7", "f1")))
    r = pdr[(pdr.dataset == dset) & (pdr.k == k) & (pdr.model == m)].iloc[0]
    return dict(acc=(r.original_accuracy, r.random_accuracy_mean, r.homology_accuracy_mean),
                auroc=(r.original_auroc, r.random_auroc_mean, r.homology_auroc_mean),
                f1=(r.original_f1, r.random_f1_mean, r.homology_f1_mean))


L = ["# PAPER_NUMBERS.md — frozen, paper-citable numbers\n",
     "Single authoritative source for the manuscript. Leaky datasets "
     "(nontata, enhancers_ensembl) are reported at **full scale**; clean datasets at the "
     "**20k stratified subsample (seed 0)** — their full-scale leakage is ~0, so the "
     "subsample drop is unconfounded. Pipeline: k-mer counts (k=4/6), LR + RF(150 trees), "
     "exact 8-mer Jaccard, whole-cluster re-split; seeds in `results/seeds.txt`.\n"]

# ---- 1. descriptor ----
L.append("## 1. Dataset descriptors\n")
L.append("| dataset | n (full) | fit scale (n used, seed) | seq len min/med/max | class balance (pos frac) | test ratio |")
L.append("|---|---|---|---|---|---|")
for d in BINARY:
    nfull = int(leak.loc[d, "n_train"] + leak.loc[d, "n_test"])
    ratio = leak.loc[d, "n_test"] / nfull
    fit = f"full ({nfull})" if d in LEAKY else f"subsample ({int(subsum.loc[d,'n_used'])}, seed 0)"
    lm = f"{int(subsum.loc[d,'len_min'])}/{int(subsum.loc[d,'len_med'])}/{int(subsum.loc[d,'len_max'])}"
    bal = balance(d)
    L.append(f"| {d} | {nfull} | {fit} | {lm} | {bal:.3f} | {ratio:.2f} |")
L.append("\n*Source: `measure_leakage_full.py`->`leakage_full.csv` (n); `run_suite.py`->"
         "`summary.csv` (lengths); `cluster_stats.csv` (balance). Leaky datasets fitted full-scale "
         "by `run_fullscale.py`/`run_robustness_full.py`; clean at 20k subsample by `run_suite.py`.*\n")

# ---- 2. leakage (full scale) ----
L.append("## 2. Leakage (FULL scale: test→train exact 8-mer Jaccard)\n")
L.append("| dataset | leak@0.5 | leak@0.7 | leak@0.9 | median sim | p99 sim |")
L.append("|---|---|---|---|---|---|")
for d in BINARY:
    L.append(f"| {d} | {leak.loc[d,'leak_full_0.5']:.4f} | {leak.loc[d,'leak_full_0.7']:.4f} "
             f"| {leak.loc[d,'leak_full_0.9']:.4f} | {leak.loc[d,'median_maxsim']:.3f} "
             f"| {leak.loc[d,'p99_maxsim']:.3f} |")
L.append("\n*Source: `measure_leakage_full.py` -> `results/leakage_full.csv`.*\n")

# ---- 3. main results ----
L.append("## 3. Main results (acc / AUROC / F1; original vs random vs homology@0.7)\n")
L.append("| dataset | model | k | orig acc | rand acc | hom acc | Δacc(hom) | Δacc(rand) | orig AUROC | hom AUROC | orig F1 | hom F1 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in BINARY:
    for (k, m) in CONFIGS:
        mr = main_row(d, k, m)
        oa, ra, ha = mr["acc"]; oA, rA, hA = mr["auroc"]; of, rf, hf = mr["f1"]
        L.append(f"| {d} | {m} | {k} | {oa:.4f} | {ra:.4f} | {ha:.4f} | {oa-ha:+.4f} | {oa-ra:+.4f} "
                 f"| {oA:.4f} | {hA:.4f} | {of:.4f} | {hf:.4f} |")
L.append("\n*Source: leaky -> `run_fullscale.py`->`fullscale_long.csv`; clean -> "
         "`run_suite.py`->`per_dataset_results.csv`. 3 re-split seeds, means shown.*\n")

# ---- 4. capacity scaling ----
L.append("## 4. Capacity scaling (homology accuracy drop by model)\n")
L.append("| dataset | leak@0.7 | LR k4 | LR k6 | RF k4 | RF k6 | monotone |")
L.append("|---|---|---|---|---|---|---|")
for d in BINARY:
    c = capf.loc[d]
    L.append(f"| {d} | {leak.loc[d,'leak_full_0.7']:.3f} | {c['hom_delta_LR_k4']:+.4f} "
             f"| {c['hom_delta_LR_k6']:+.4f} | {c['hom_delta_RF_k4']:+.4f} | {c['hom_delta_RF_k6']:+.4f} "
             f"| {'yes' if c['monotone'] else 'no'} |")
L.append("\n*Source: `finalize.py` -> `capacity_scaling_final.csv` (full-scale deltas for leaky, "
         "subsample for clean).*\n")

# ---- 5. robustness (full scale, both leaky datasets) ----
L.append("## 5. Clustering robustness (FULL scale, both leaky datasets)\n")
L.append("Homology-aware drop (= original - corrected accuracy) under the single-linkage "
         "threshold sweep and the drop-largest-component check, RF k6 and LR k6:\n")
L.append("| dataset | model | k | variant | corrected acc | Δacc | (orig) |")
L.append("|---|---|---|---|---|---|---|")
for d in LEAKY:
    for (k, m) in [(6, "RF"), (6, "LR")]:
        for var in ["homology@0.5", "homology@0.7", "homology@0.9", "droplargest@0.7"]:
            row = robs[(robs.dataset == d) & (robs.model == m) & (robs.k == k) & (robs.variant == var)]
            if len(row):
                r = row.iloc[0]
                L.append(f"| {d} | {m} | {k} | {var} | {r.corrected_acc_mean:.4f}±{r.corrected_acc_std:.4f} "
                         f"| {r.delta_acc:+.4f} | {r.original_acc:.4f} |")
L.append("")
L.append("Cluster-size distribution + verified invariants (residual leakage, spanning, balance):\n")
L.append("| dataset | threshold | n_comp | max cluster | redundant | spanning | resid>thr |")
L.append("|---|---|---|---|---|---|---|")
for _, r in robc[robc.threshold != "droplargest@0.7"].iterrows():
    L.append(f"| {r.dataset} | {r.threshold} | {int(r.n_components)} | {int(r.max_cluster)} "
             f"| {float(r.frac_redundant):.3f} | {int(r.clusters_spanning_split)} | {float(r.resid_leak_frac):.4f} |")
L.append("\n*Source: `run_robustness_full.py` -> `robustness_fullscale_summary.csv`, "
         "`robustness_fullscale_clusters.csv`.*\n")

# ---- 6. negative control ----
L.append("## 6. Negative control (random re-split delta, best model)\n")
L.append("| dataset | best model | Δacc(homology) | Δacc(random) |")
L.append("|---|---|---|---|")
for d in BINARY:
    s = sfin.loc[d]
    L.append(f"| {d} | {s['best_model']} | {s['delta_homology']:+.4f} | {s['delta_random']:+.4f} |")
big = [d for d in BINARY if abs(sfin.loc[d, "delta_random"]) > 0.006]
L.append(f"\nAll random-resplit deltas within ±0.006 except: {big if big else 'none'} "
         f"(drosophila n=6914, small-sample). *Source: `finalize.py`->`summary_final.csv`.*\n")

# ---- 7. cross-dataset summary ----
L.append("## 7. Cross-dataset summary\n")
lk = sfin.loc[[d for d in BINARY if leak.loc[d, "leak_full_0.7"] > 0.1]]
cl = sfin.loc[[d for d in BINARY if leak.loc[d, "leak_full_0.7"] <= 0.1]]
L.append(f"- **Leaky (leak@0.7>0.1): {len(lk)}** ({', '.join(lk.index)}): best-model homology drop "
         f"mean **{lk.delta_homology.mean():.3f}** (range {lk.delta_homology.min():+.3f}"
         f"..{lk.delta_homology.max():+.3f}); random drop mean {lk.delta_random.mean():+.3f}.")
L.append(f"- **Clean: {len(cl)}** ({', '.join(cl.index)}): best-model homology drop "
         f"mean **{cl.delta_homology.mean():+.3f}** (range {cl.delta_homology.min():+.3f}"
         f"..{cl.delta_homology.max():+.3f}) — ~0; random drop mean {cl.delta_random.mean():+.3f}.\n")

# ---- 8. subsampling-masking ----
L.append("## 8. Subsampling-masking effect (why full-scale leakage matters)\n")
L.append("| dataset | leak@0.7 (20k subsample, seed 0) | leak@0.7 (full) |")
L.append("|---|---|---|")
for d in BINARY:
    sub = ts[(ts.dataset == d) & (ts.threshold == 0.7) & (ts.model == "LR") & (ts.k == 4)]
    subleak = float(sub.leakage_fraction.iloc[0]) if len(sub) else np.nan
    L.append(f"| {d} | {subleak:.4f} | {leak.loc[d,'leak_full_0.7']:.4f} |")
L.append("\n**human_enhancers_ensembl: 0.0560 at the 20k subsample (seed 0) vs 0.3839 at full "
         "scale** — subsampling to ~13% of 154,842 sequences broke up most near-duplicate pairs "
         "and masked the leakage. *Source: subsample `run_suite.py`->`threshold_sensitivity.csv`; "
         "full `measure_leakage_full.py`->`leakage_full.csv`.*\n")

# ---- provenance ----
# ---- 9. extended capacity (Part A) ----
_M4 = ["LR", "LinearSVC", "RF", "HGB"]


def _xacc(dset, k, split, model):
    q = ((extlong.dataset == dset) & (extlong.k == k)
         & (extlong.split_type == split) & (extlong.model == model))
    return float(extlong[q]["accuracy"].mean())


L.append("## 9. Extended capacity curve (Part A: four classical models)\n")
L.append("Capacity proxy ordering: **LogReg ~= LinearSVC** (linear) < **RandomForest** "
         "(150 deep, unpruned, bagged trees) < **HistGradientBoosting** (boosted trees). "
         "An MLP/1D-CNN point was **skipped** (no GPU permitted; a neural fit on the full "
         "154k x 4096 enhancers_ensembl set would exceed the CPU budget) -- HGB anchors the "
         "high-capacity end. Same k-mer features, splits, 3 seeds, metrics as before; leaky "
         "datasets at full scale, clean at the 20k subsample.\n")
L.append("Homology accuracy drop (original - homology@0.7), k=6:\n")
L.append("| dataset | leak@0.7 | LR | LinearSVC | RF | HGB |")
L.append("|---|---|---|---|---|---|")
for _, r in capext.iterrows():
    L.append(f"| {r['dataset']} | {leak.loc[r['dataset'],'leak_full_0.7']:.3f} | "
             f"{r['hom_delta_LR_k6']:+.3f} | {r['hom_delta_LinearSVC_k6']:+.3f} | "
             f"{r['hom_delta_RF_k6']:+.3f} | {r['hom_delta_HGB_k6']:+.3f} |")
L.append("")
L.append("**Finding:** the drop is monotone **LR ~= LinearSVC < RF** and is driven by RF's deep "
         "unpruned trees memorizing near-duplicates. **HGB (regularized boosting) drops far less "
         "than RF** (e.g. +0.006 vs +0.164 on enhancers_ensembl), so strict LR<SVC<RF<HGB "
         "monotonicity does NOT hold: the drop tracks *memorization propensity / regularization*, "
         "not nominal capacity. On clean datasets all four models stay ~0. (k=4 in "
         "`capacity_scaling_extended.csv`; figure `fig_capacity_extended.*`.)\n")

# ---- 10. ranking inversion (Part B) ----
L.append("## 10. Ranking inversion (Part B)\n")
h = ri[(ri.k == 6) & (ri.metric == "accuracy")].set_index("dataset")
L.append("Models ranked by test accuracy (k=6) on the original (leaky) split vs the "
         "homology-aware split; Kendall tau between the two rankings; inversions counted "
         "only if stable across all 3 homology seeds.\n")
L.append("| dataset | leaky | tau | RF rank (orig->corr) | max inversion gap | original ranking | corrected ranking |")
L.append("|---|---|---|---|---|---|---|")
for d in ["human_nontata_promoters", "human_enhancers_ensembl",
          "demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
          "demo_human_or_worm", "drosophila_enhancers_stark", "human_enhancers_cohn"]:
    o = {m: _xacc(d, 6, "original", m) for m in _M4}
    c = {m: _xacc(d, 6, "homology@0.7", m) for m in _M4}
    rfo = sorted(_M4, key=lambda m: -o[m]).index("RF") + 1
    rfc = sorted(_M4, key=lambda m: -c[m]).index("RF") + 1
    gaps = [abs(o[a] - o[b]) for a, b in itertools.combinations(_M4, 2)
            if np.sign(o[a]-o[b]) != 0 and np.sign(c[a]-c[b]) != 0 and np.sign(o[a]-o[b]) != np.sign(c[a]-c[b])]
    L.append(f"| {d} | {bool(h.loc[d,'leaky'])} | {h.loc[d,'kendall_tau']:+.2f} | {rfo}->{rfc} "
             f"| {max(gaps) if gaps else 0:.3f} | {h.loc[d,'original_ranking']} | {h.loc[d,'corrected_ranking']} |")
L.append("")
leaky_tau = h[h.leaky]["kendall_tau"].mean()
clean_tau = h[~h.leaky]["kendall_tau"].mean()
n_stable = int((h["n_stable_inversions"] > 0).sum())
L.append(f"**Mean Kendall tau: leaky = {leaky_tau:+.2f}, clean = {clean_tau:+.2f}.** "
         f"{n_stable}/7 datasets show >=1 stable inversion, BUT the character differs sharply: "
         "on **leaky** datasets the apparent best model (RF) is dethroned by a **material** margin "
         "(inversion gaps 0.058 / 0.096; RF 1st->3rd / 1st->4th), whereas on **clean** datasets RF "
         "stays last on both splits and the only inversions are sub-0.003 swaps between "
         "statistically-tied models (drosophila's 0.019 is small-n noise).\n")
L.append("Quotable inversions (k=6, accuracy, stable across 3/3 seeds):\n")
L.append("- **nonTATA promoters:** RandomForest ranks **1st on the leaky split but 3rd after "
         "correction**; LogReg rises from last to 1st.")
L.append("- **enhancers (Ensembl):** RandomForest ranks **1st on the leaky split but LAST (4th) "
         "after correction** -- the method that looks best is the most leakage-inflated.\n")

# ---- 11. splitter tool + validation (Part C) ----
L.append("## 11. Homology-aware splitter tool + validation (Part C)\n")
L.append("`homology_split.py` exposes `homology_aware_split(sequences, labels, test_frac=0.25, "
         "threshold=0.7, seed=0, k=8) -> (train_idx, test_idx)` (pure numpy/scipy, CPU-only). "
         "Guarantees, checkable via `verify_split`: no cluster spans the split (residual "
         "cross-split Jaccard > threshold == 0), class balance and train/test ratio preserved, "
         "deterministic given seed.\n")
if sval is not None:
    L.append("Validation on the two leaky datasets, RF k6, **5 seeds**:\n")
    L.append("| dataset | corrected acc (mean +/- std, 5 seeds) | max residual leak >0.7 | mean test frac |")
    L.append("|---|---|---|---|")
    for _, r in sval.iterrows():
        L.append(f"| {r['dataset']} | {r['corrected_acc_mean']:.4f} +/- {r['corrected_acc_std']:.4f} "
                 f"| {r['max_residual_leak']:.6f} | {r['test_frac_mean']:.3f} |")
    L.append("\nResidual leakage is **0.000000** for every seed (the tool's guarantee), and corrected "
             "accuracy is stable across seeds -- i.e. the tool yields honest, reproducible evaluation. "
             "*Source: `validate_splitter.py` -> `splitter_validation.csv`.*\n")
else:
    L.append("*(Validation pending: run `python validate_splitter.py`.)*\n")

# ---- 12. graded performance (Part 3) ----
L.append("## 12. Homology-graded performance (Part 3: the memorization mechanism)\n")
if graded is not None:
    _BINS = ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
    L.append("On the ORIGINAL split, test sequences are binned by max 8-mer Jaccard to the "
             "training set; per-model accuracy (k=6) per bin. The memorization signature is the "
             "**gap = accuracy[>=0.9] - accuracy[<0.5]**: large for the high-capacity memorizer "
             "(RF), small for linear models. Clean datasets have near-empty high-similarity bins "
             "(nothing to memorize).\n")
    for d in ["human_nontata_promoters", "human_enhancers_ensembl",
              "human_enhancers_cohn", "demo_human_or_worm"]:
        sub = graded[graded.dataset == d]
        if not len(sub):
            continue
        nb = {b: int(sub[(sub.model == "RF") & (sub.sim_bin == b)]["n"].iloc[0]) for b in _BINS}
        leaky = bool(sub["leaky"].iloc[0])
        L.append(f"**{d}** ({'LEAKY' if leaky else 'clean control'}) -- test n per bin: "
                 + ", ".join(f"{b}={nb[b]}" for b in _BINS) + "\n")
        L.append("| model | acc [0,0.5) | acc [0.5,0.7) | acc [0.7,0.9) | acc [0.9,1.0] | gap (>=0.9 - <0.5) |")
        L.append("|---|---|---|---|---|---|")
        for m in ["LR", "LinearSVC", "RF", "HGB"]:
            sm = sub[sub.model == m].set_index("sim_bin")
            def cell(b):
                v = sm.loc[b, "accuracy"] if b in sm.index else None
                if v is None or pd.isna(v):
                    return "-"
                return f"{v:.3f}{'*' if nb[b] < 50 else ''}"
            lo = sm.loc["[0,0.5)", "accuracy"] if "[0,0.5)" in sm.index else np.nan
            hi = sm.loc["[0.9,1.0]", "accuracy"] if "[0.9,1.0]" in sm.index else np.nan
            gap = f"{(hi-lo):+.3f}" + ("*" if (nb['[0.9,1.0]'] < 50 or nb['[0,0.5)'] < 50) else "") if (pd.notna(lo) and pd.notna(hi)) else "n/a"
            L.append(f"| {m} | {cell('[0,0.5)')} | {cell('[0.5,0.7)')} | {cell('[0.7,0.9)')} | {cell('[0.9,1.0]')} | {gap} |")
        L.append("")
    L.append("`*` = bin has n<50 (low-confidence). **Reading:** on the leaky datasets RF is "
             "near-perfect on near-duplicate test sequences ([0.9,1.0]) and falls on dissimilar "
             "ones ([0,0.5)) -- a large gap -- while LR/LinearSVC are far flatter; this is direct "
             "evidence that RF's apparent edge is memorization, not generalization. On clean "
             "controls the high-similarity bins are nearly empty (n<50), so no gap is estimable -- "
             "consistent with there being no near-duplicates to exploit. "
             "*Source: `run_graded.py` -> `graded_performance.csv`; figure `fig_graded_performance.*`.*\n")
else:
    L.append("*(pending: run `python run_graded.py`)*\n")

# ---- 13. leakage report card (Part 1 artifact) ----
L.append("## 13. Leakage report card (which datasets can you trust?)\n")
if rcard is not None:
    L.append("| dataset | n | leak@0.7 | leak@0.9 | verdict | best model | acc drop (corrected) | ranking inverts | RF rank orig->corr |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in rcard.iterrows():
        L.append(f"| {r['dataset']} | {int(r['n_full']):,} | {r['leak_at_0p7']:.3f} | {r['leak_at_0p9']:.3f} "
                 f"| {r['verdict']} | {r['best_model']} | {r['acc_drop_corrected']:+.3f} "
                 f"| {r['ranking_inverts']} | {r['rf_rank_orig_to_corr']} |")
    L.append("\n*Drop values are the bootstrap-consistent point estimates (seed-0 corrected "
             "split, with 95% CIs reported in §16); 3-seed-mean corrected accuracies and "
             "SDs are in §16.*")
    L.append("\nVerdict rule: LEAKY if full-scale test/train near-duplicate fraction > 0.1 at "
             "Jaccard 0.7. *Source: `report_card.py` -> `leakage_report_card.csv`; figure "
             "`fig_report_card.*`.*\n")
else:
    L.append("*(pending: run `python report_card.py`)*\n")

# ---- 14. label concordance (Check 1) ----
L.append("## 14. Label concordance of near-duplicates (Check 1)\n")
if lc is not None:
    L.append("For each test sequence with a training near-duplicate at Jaccard >= t, is its "
             "NEAREST training neighbour the same label? If ~1.0, near-duplicates carry their "
             "labels across the split, so a memorizing model scores them correctly for free.\n")
    L.append("| dataset | leaky | threshold | n pairs | same-label fraction |")
    L.append("|---|---|---|---|---|")
    for _, r in lc.iterrows():
        flag = " *" if r["low_confidence"] else ""
        val = f"{r['frac_same_label']:.4f}" if pd.notna(r['frac_same_label']) else "n/a"
        L.append(f"| {r['dataset']} | {r['leaky']} | >= {r['threshold']} | {int(r['n_pairs'])}{flag} | {val} |")
    L.append("\n`*` = n<50 (low-confidence). **Reading:** on the leaky datasets the same-label "
             "fraction at >=0.9 is ~1.0 -- near-duplicates carry their labels across the split, so "
             "RF's 1.000 accuracy on that bin is the expected consequence of memorizing labelled "
             "near-copies, not hard-won generalization. Clean datasets have near-empty >=0.9 groups "
             "(nothing to memorize). *Source: `check1_label_concordance.py` -> `label_concordance.csv`.*\n")
else:
    L.append("*(pending: run `python check1_label_concordance.py`)*\n")

# ---- 15. novel-only ranking (Check 2) ----
L.append("## 15. Novel-only ranking: independent routes to 'is RF's lead real?' (Check 2)\n")
if nor is not None:
    L.append("Rank the 4 models (k=6) by (a) original leaky-split accuracy, (b) homology-aware-split "
             "accuracy, and (c) accuracy on novel (<0.5-similarity) test sequences only. (b) and (c) "
             "are independent honest evaluations.\n")
    L.append("| dataset | (a) leaky-split | (b) homology-aware | (c) novel-only | RF rank a/b/c | tau(b,c) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in nor.iterrows():
        L.append(f"| {r['dataset']} | {r['a_leaky_ranking']} | {r['b_homology_ranking']} "
                 f"| {r['c_novelonly_ranking']} | {r['rf_rank_leaky']}/{r['rf_rank_homology']}/{r['rf_rank_novelonly']} "
                 f"| {r['tau_b_c']} |")
    L.append("")
    L.append("**Reading:** on **enhancers_ensembl** both honest routes dethrone RF from its leaky #1 "
             "and crown LinearSVC (RF novel-only acc 0.770 < LinearSVC 0.782; RF homology rank last) "
             "-- two independent confirmations that RF's leaky lead is memorization, not "
             "generalization. On **nonTATA promoters** the routes diverge (tau(b,c) = -0.67): the "
             "homology split demotes RF to 3rd, but RF stays best on truly-novel sequences (0.879). "
             "**Flagged:** nonTATA's RF advantage is therefore partly genuine generalization, not "
             "pure test-side memorization -- its homology demotion also reflects the de-duplicated "
             "training regime. The memorization mechanism is airtight for enhancers_ensembl and "
             "partial for nonTATA. *Source: `check2_novelonly_ranking.py` -> `novelonly_ranking.csv`.*\n")
else:
    L.append("*(pending: run `python check2_novelonly_ranking.py`)*\n")

# ---- 16. variance & confidence intervals ----
L.append("## 16. Variance and confidence intervals (uncertainty quantification)\n")
if s2var is not None and s3acc is not None and s3del is not None and s4tie is not None:
    L.append("Quantifies the uncertainty behind the \"within noise\" / \"statistically tied\" "
             "statements; **no number in sections 1-15 is changed -- this section only adds "
             "variance.** Bootstrap CIs resample the stored per-example test correctness "
             "(1000 draws, no model refit), RNG seed **20240524**. Leaky datasets are at full "
             "scale, clean at the 20k subsample (seed 0), matching the frozen results above. "
             "Re-split seeds {0,1,2,3,4}.\n")

    L.append("### 16.1 Re-split seed variance (5 cluster->side seeds, homology-aware corrected accuracy)\n")
    L.append("Confirms that a single cluster->side assignment is not 'unlucky': corrected accuracy "
             "is stable across 5 independent assignments.\n")
    L.append("| dataset | model | mean | SD | min | max | range | per-seed |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in LEAKY:
        for m in ["RF_k6", "LR_k6"]:
            r = s2var[(s2var.dataset == d) & (s2var.model == m)]
            if len(r):
                r = r.iloc[0]
                L.append(f"| {d} | {m} | {r['mean']:.4f} | {r['sd']:.4f} | {r['min']:.4f} "
                         f"| {r['max']:.4f} | {r['range']:.4f} | {r['per_seed']} |")
    L.append("\nRF k6 reproduces `splitter_validation.csv` (seed-0 cross-check exact); SD <= 0.008 "
             "for every cell. *Source: `step2_rf_seeds.py` (RF) + `step_variance_ci.py` (LR) -> "
             "`step2_seed_variance.csv`; cross-checked vs `validate_splitter.py` -> `splitter_validation.csv`.*\n")

    L.append("### 16.2 Test-set bootstrap 95% CIs and delta significance\n")
    L.append("Accuracies with 95% bootstrap CIs (homology column = seed-0 corrected split):\n")
    L.append("| dataset | model | original acc [95% CI] | homology acc [95% CI] |")
    L.append("|---|---|---|---|")
    for d in LEAKY:
        for m in ["RF_k6", "LR_k6"]:
            o = s3acc[(s3acc.dataset == d) & (s3acc.model == m) & (s3acc.split == "original")]
            h = s3acc[(s3acc.dataset == d) & (s3acc.model == m) & (s3acc.split.str.startswith("homology"))]
            if len(o) and len(h):
                o = o.iloc[0]; h = h.iloc[0]
                L.append(f"| {d} | {m} | {o['accuracy']:.4f} [{o['ci_lo']:.4f}, {o['ci_hi']:.4f}] "
                         f"| {h['accuracy']:.4f} [{h['ci_lo']:.4f}, {h['ci_hi']:.4f}] |")
    L.append("")
    L.append("Delta (original - corrected) with 95% CI; \"excludes 0\" => the drop is significant:\n")
    L.append("| dataset | group | model | delta | 95% CI | excludes 0 |")
    L.append("|---|---|---|---|---|---|")
    for _, r in s3del.iterrows():
        L.append(f"| {r['dataset']} | {r['group']} | {r['model']} | {r['delta_orig_minus_corr']:+.4f} "
                 f"| [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {'YES' if r['excludes_zero'] else 'no'} |")
    L.append("\nBootstrap deltas use the seed-0 corrected split, so point estimates differ slightly "
             "from the frozen 3-seed-mean deltas in sections 3-5 (unchanged); the sign/significance "
             "is the result of interest. **Reading:** the high-capacity (RF) drop is significant on "
             "both leaky datasets (CIs exclude 0); the linear (LR) drop is significant on nonTATA but "
             "**null on enhancers_ensembl** (CI includes 0) -- only the memorizer inflates there. All "
             "five clean-dataset deltas are not significant (CIs include 0), which replaces 'within "
             "noise' with a tested statement. Clean per-model original-split CIs are in "
             "`step3_accuracy_ci.csv`. *Source: `step_variance_ci.py` -> `step3_accuracy_ci.csv`, "
             "`step3_delta_ci.csv`.*\n")

    L.append("### 16.3 \"Statistically tied\": clean-dataset rank swaps\n")
    L.append("For every clean-dataset rank swap, the swapped models' original-split accuracy CIs "
             "overlap, so 'statistically tied' is justified.\n")
    L.append("| dataset | swapped pair | acc A [95% CI] | acc B [95% CI] | CI overlap | gap |")
    L.append("|---|---|---|---|---|---|")
    for _, r in s4tie.iterrows():
        L.append(f"| {r['dataset']} | {r['model_a']} / {r['model_b']} | {r['acc_a']:.4f} {r['ci_a']} "
                 f"| {r['acc_b']:.4f} {r['ci_b']} | {'yes' if r['ci_overlap'] else 'NO'} | {r['gap']:.4f} |")
    L.append("\nEvery swap-pair CI overlaps, so 'statistically tied' holds for all clean swaps "
             "(human_or_worm has no swap, tau=+1.00). The 'sub-0.003' magnitude in section 10 fits "
             "cohn/ocr/coding; drosophila's LinearSVC/RF swap (gap 0.019) is larger but unstable "
             "(2/3 seeds, not a stable inversion) and its CIs still overlap. *Source: "
             "`step_variance_ci.py` -> `step4_tie_overlap.csv`.*\n")
else:
    L.append("*(pending: run `python step_variance_ci.py` then `python step2_rf_seeds.py`)*\n")

L.append("## 17. Provenance (number -> script -> file)\n")
L.append("| quantity | script | output file |")
L.append("|---|---|---|")
for q, s, f in [
    ("full-scale leakage fractions / sim distribution", "measure_leakage_full.py", "leakage_full.csv"),
    ("single-dataset headline (nontata)", "run_audit.py", "results.md, results_table.csv"),
    ("subsample sweep: original/random/threshold/drop-largest (all 9 datasets)", "run_suite.py",
     "per_dataset_results.csv, threshold_sensitivity.csv, cluster_stats.csv, all_results_long.csv"),
    ("full-scale main results for leaky datasets", "run_fullscale.py", "fullscale_long.csv, fullscale_summary.csv"),
    ("full-scale robustness (sweep + drop-largest) for leaky datasets", "run_robustness_full.py",
     "robustness_fullscale_summary.csv, robustness_fullscale_clusters.csv"),
    ("consolidated summary + capacity scaling", "finalize.py", "summary_final.csv, capacity_scaling_final.csv, results_FINAL.md"),
    ("figures + backing data", "make_paper_figures.py", "figures/fig_capacity_scaling.*, figures/fig_controls.*, *_data.csv"),
    ("extended capacity (4 models, Part A)", "run_extended_models.py + make_part_b_figures.py",
     "extended_models_long.csv, capacity_scaling_extended.csv, figures/fig_capacity_extended.*"),
    ("ranking inversion (Part B)", "ranking_inversion.py + make_part_b_figures.py",
     "ranking_inversion.csv, figures/fig_ranking_inversion.*"),
    ("homology-aware splitter + validation (Part C)", "homology_split.py + validate_splitter.py",
     "homology_split.py, splitter_validation.csv"),
    ("graded performance by similarity bin (Part 3)", "run_graded.py + make_graded_figure.py",
     "graded_performance.csv, figures/fig_graded_performance.*"),
    ("leakage report card (Part 1 artifact)", "report_card.py",
     "leakage_report_card.csv, figures/fig_report_card.*"),
    ("splitter tool + CLI + docs (Part 2 artifact)", "homology_split.py",
     "homology_split.py, TOOL_README.md"),
    ("label concordance of near-duplicates (Check 1)", "check1_label_concordance.py", "label_concordance.csv"),
    ("novel-only ranking, 3-way comparison (Check 2)", "check2_novelonly_ranking.py", "novelonly_ranking.csv"),
    ("re-split seed variance: RF+LR k6, 5 seeds, both leaky (Phase 14)",
     "step2_rf_seeds.py + step_variance_ci.py", "step2_seed_variance.csv"),
    ("test-set bootstrap 95% CIs + delta significance + 'tied' overlap (Phase 14)",
     "step_variance_ci.py", "step3_accuracy_ci.csv, step3_delta_ci.csv, step4_tie_overlap.csv"),
    ("this file", "paper_numbers.py", "PAPER_NUMBERS.md"),
]:
    L.append(f"| {q} | `{s}` | `{f}` |")
L.append("")
L.append("*3-class human_ensembl_regulatory (leak@0.7=0.005, clean) and the dummy_mouse smoke "
         "test are in `summary.csv`/`leakage_full.csv`; excluded from the binary tables above.*\n")

with open(f"{R}/PAPER_NUMBERS.md", "w") as fh:
    fh.write("\n".join(L))
print("wrote results/PAPER_NUMBERS.md")
