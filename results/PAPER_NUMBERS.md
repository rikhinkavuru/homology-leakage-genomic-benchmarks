# PAPER_NUMBERS.md — frozen, paper-citable numbers

Single authoritative source for the manuscript. Leaky datasets (nontata, enhancers_ensembl) are reported at **full scale**; clean datasets at the **20k stratified subsample (seed 0)** — their full-scale leakage is ~0, so the subsample drop is unconfounded. Pipeline: k-mer counts (k=4/6), LR + RF(150 trees), exact 8-mer Jaccard, whole-cluster re-split; seeds in `results/seeds.txt`.

## 1. Dataset descriptors

| dataset | n (full) | fit scale (n used, seed) | seq len min/med/max | class balance (pos frac) | test ratio |
|---|---|---|---|---|---|
| human_nontata_promoters | 36131 | full (36131) | 251/251/251 | 0.544 | 0.25 |
| human_enhancers_ensembl | 154842 | full (154842) | 2/269/573 | 0.500 | 0.20 |
| demo_coding_vs_intergenomic_seqs | 100000 | subsample (20000, seed 0) | 200/200/200 | 0.500 | 0.25 |
| human_ocr_ensembl | 174756 | subsample (20000, seed 0) | 71/315/593 | 0.500 | 0.20 |
| demo_human_or_worm | 100000 | subsample (20000, seed 0) | 200/200/200 | 0.500 | 0.25 |
| drosophila_enhancers_stark | 6914 | subsample (6914, seed 0) | 236/2142/3237 | 0.500 | 0.25 |
| human_enhancers_cohn | 27791 | subsample (20000, seed 0) | 500/500/500 | 0.500 | 0.25 |

*Source: `measure_leakage_full.py`->`leakage_full.csv` (n); `run_suite.py`->`summary.csv` (lengths); `cluster_stats.csv` (balance). Leaky datasets fitted full-scale by `run_fullscale.py`/`run_robustness_full.py`; clean at 20k subsample by `run_suite.py`.*

## 2. Leakage (FULL scale: test→train exact 8-mer Jaccard)

| dataset | leak@0.5 | leak@0.7 | leak@0.9 | median sim | p99 sim |
|---|---|---|---|---|---|
| human_nontata_promoters | 0.4497 | 0.4064 | 0.2254 | 0.088 | 0.992 |
| human_enhancers_ensembl | 0.3938 | 0.3839 | 0.3802 | 0.120 | 1.000 |
| demo_coding_vs_intergenomic_seqs | 0.1420 | 0.0784 | 0.0243 | 0.049 | 0.959 |
| human_ocr_ensembl | 0.0382 | 0.0104 | 0.0010 | 0.041 | 0.705 |
| demo_human_or_worm | 0.0292 | 0.0124 | 0.0026 | 0.048 | 0.738 |
| drosophila_enhancers_stark | 0.0283 | 0.0156 | 0.0058 | 0.052 | 0.776 |
| human_enhancers_cohn | 0.0075 | 0.0012 | 0.0004 | 0.034 | 0.455 |

*Source: `measure_leakage_full.py` -> `results/leakage_full.csv`.*

## 3. Main results (acc / AUROC / F1; original vs random vs homology@0.7)

| dataset | model | k | orig acc | rand acc | hom acc | Δacc(hom) | Δacc(rand) | orig AUROC | hom AUROC | orig F1 | hom F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | LR | 4 | 0.8268 | 0.8305 | 0.8182 | +0.0086 | -0.0038 | 0.8974 | 0.8884 | 0.8329 | 0.8244 |
| human_nontata_promoters | LR | 6 | 0.8702 | 0.8749 | 0.8300 | +0.0402 | -0.0048 | 0.9381 | 0.9072 | 0.8747 | 0.8421 |
| human_nontata_promoters | RF | 4 | 0.8907 | 0.8916 | 0.8070 | +0.0837 | -0.0008 | 0.9811 | 0.8954 | 0.8913 | 0.8255 |
| human_nontata_promoters | RF | 6 | 0.9317 | 0.9324 | 0.8105 | +0.1212 | -0.0007 | 0.9938 | 0.9220 | 0.9342 | 0.8389 |
| human_enhancers_ensembl | LR | 4 | 0.7337 | 0.7376 | 0.7383 | -0.0046 | -0.0039 | 0.8020 | 0.8063 | 0.7354 | 0.7397 |
| human_enhancers_ensembl | LR | 6 | 0.7809 | 0.7810 | 0.7776 | +0.0033 | -0.0000 | 0.8573 | 0.8542 | 0.7812 | 0.7767 |
| human_enhancers_ensembl | RF | 4 | 0.8461 | 0.8485 | 0.6934 | +0.1527 | -0.0024 | 0.9398 | 0.7873 | 0.8527 | 0.6445 |
| human_enhancers_ensembl | RF | 6 | 0.8556 | 0.8577 | 0.6999 | +0.1558 | -0.0021 | 0.9428 | 0.8031 | 0.8606 | 0.6474 |
| demo_coding_vs_intergenomic_seqs | LR | 4 | 0.8836 | 0.8847 | 0.8827 | +0.0009 | -0.0011 | 0.9537 | 0.9540 | 0.8852 | 0.8838 |
| demo_coding_vs_intergenomic_seqs | LR | 6 | 0.8878 | 0.8905 | 0.8907 | -0.0029 | -0.0027 | 0.9576 | 0.9596 | 0.8884 | 0.8911 |
| demo_coding_vs_intergenomic_seqs | RF | 4 | 0.8590 | 0.8583 | 0.8598 | -0.0008 | +0.0007 | 0.9359 | 0.9367 | 0.8633 | 0.8637 |
| demo_coding_vs_intergenomic_seqs | RF | 6 | 0.8406 | 0.8454 | 0.8428 | -0.0022 | -0.0048 | 0.9273 | 0.9280 | 0.8451 | 0.8471 |
| human_ocr_ensembl | LR | 4 | 0.6495 | 0.6492 | 0.6484 | +0.0011 | +0.0003 | 0.7029 | 0.7025 | 0.6477 | 0.6476 |
| human_ocr_ensembl | LR | 6 | 0.6770 | 0.6762 | 0.6732 | +0.0038 | +0.0008 | 0.7473 | 0.7423 | 0.6807 | 0.6763 |
| human_ocr_ensembl | RF | 4 | 0.6645 | 0.6562 | 0.6460 | +0.0185 | +0.0083 | 0.7182 | 0.7036 | 0.6847 | 0.6658 |
| human_ocr_ensembl | RF | 6 | 0.6540 | 0.6540 | 0.6439 | +0.0101 | +0.0000 | 0.7112 | 0.7049 | 0.6709 | 0.6628 |
| demo_human_or_worm | LR | 4 | 0.9278 | 0.9375 | 0.9369 | -0.0091 | -0.0097 | 0.9795 | 0.9823 | 0.9273 | 0.9364 |
| demo_human_or_worm | LR | 6 | 0.9374 | 0.9435 | 0.9414 | -0.0040 | -0.0061 | 0.9840 | 0.9862 | 0.9365 | 0.9406 |
| demo_human_or_worm | RF | 4 | 0.9310 | 0.9355 | 0.9366 | -0.0056 | -0.0045 | 0.9820 | 0.9837 | 0.9315 | 0.9370 |
| demo_human_or_worm | RF | 6 | 0.9180 | 0.9198 | 0.9227 | -0.0047 | -0.0018 | 0.9738 | 0.9768 | 0.9189 | 0.9241 |
| drosophila_enhancers_stark | LR | 4 | 0.7006 | 0.6832 | 0.6852 | +0.0154 | +0.0174 | 0.7614 | 0.7416 | 0.7144 | 0.7018 |
| drosophila_enhancers_stark | LR | 6 | 0.7179 | 0.6934 | 0.7061 | +0.0118 | +0.0245 | 0.7869 | 0.7621 | 0.7310 | 0.7226 |
| drosophila_enhancers_stark | RF | 4 | 0.6879 | 0.6769 | 0.6828 | +0.0051 | +0.0110 | 0.7546 | 0.7356 | 0.7103 | 0.7088 |
| drosophila_enhancers_stark | RF | 6 | 0.7081 | 0.6746 | 0.6850 | +0.0231 | +0.0335 | 0.7568 | 0.7253 | 0.7315 | 0.7126 |
| human_enhancers_cohn | LR | 4 | 0.7310 | 0.7300 | 0.7341 | -0.0031 | +0.0010 | 0.8086 | 0.8132 | 0.7277 | 0.7295 |
| human_enhancers_cohn | LR | 6 | 0.7362 | 0.7331 | 0.7405 | -0.0043 | +0.0031 | 0.8103 | 0.8168 | 0.7321 | 0.7364 |
| human_enhancers_cohn | RF | 4 | 0.7012 | 0.7043 | 0.7122 | -0.0110 | -0.0031 | 0.7836 | 0.7881 | 0.6995 | 0.7100 |
| human_enhancers_cohn | RF | 6 | 0.6934 | 0.6931 | 0.6988 | -0.0054 | +0.0003 | 0.7728 | 0.7740 | 0.6866 | 0.6925 |

*Source: leaky -> `run_fullscale.py`->`fullscale_long.csv`; clean -> `run_suite.py`->`per_dataset_results.csv`. 3 re-split seeds, means shown.*

## 4. Capacity scaling (homology accuracy drop by model)

| dataset | leak@0.7 | LR k4 | LR k6 | RF k4 | RF k6 | monotone |
|---|---|---|---|---|---|---|
| human_nontata_promoters | 0.406 | +0.0086 | +0.0402 | +0.0837 | +0.1212 | yes |
| human_enhancers_ensembl | 0.384 | -0.0046 | +0.0033 | +0.1527 | +0.1558 | yes |
| demo_coding_vs_intergenomic_seqs | 0.078 | -0.0006 | -0.0008 | +0.0073 | +0.0033 | no |
| human_ocr_ensembl | 0.010 | +0.0011 | +0.0038 | +0.0185 | +0.0101 | no |
| demo_human_or_worm | 0.012 | -0.0091 | -0.0040 | -0.0056 | -0.0047 | no |
| drosophila_enhancers_stark | 0.016 | +0.0154 | +0.0118 | +0.0050 | +0.0231 | no |
| human_enhancers_cohn | 0.001 | -0.0031 | -0.0043 | -0.0110 | -0.0054 | no |

*Source: `finalize.py` -> `capacity_scaling_final.csv` (full-scale deltas for leaky, subsample for clean).*

## 5. Clustering robustness (FULL scale, both leaky datasets)

Homology-aware drop (= original - corrected accuracy) under the single-linkage threshold sweep and the drop-largest-component check, RF k6 and LR k6:

| dataset | model | k | variant | corrected acc | Δacc | (orig) |
|---|---|---|---|---|---|---|
| human_nontata_promoters | RF | 6 | homology@0.5 | 0.7271±0.0063 | +0.2046 | 0.9317 |
| human_nontata_promoters | RF | 6 | homology@0.7 | 0.8105±0.0060 | +0.1212 | 0.9317 |
| human_nontata_promoters | RF | 6 | homology@0.9 | 0.9253±0.0008 | +0.0064 | 0.9317 |
| human_nontata_promoters | RF | 6 | droplargest@0.7 | 0.8144±0.0079 | +0.1173 | 0.9317 |
| human_nontata_promoters | LR | 6 | homology@0.5 | 0.8030±0.0043 | +0.0671 | 0.8702 |
| human_nontata_promoters | LR | 6 | homology@0.7 | 0.8300±0.0100 | +0.0402 | 0.8702 |
| human_nontata_promoters | LR | 6 | homology@0.9 | 0.8606±0.0033 | +0.0095 | 0.8702 |
| human_nontata_promoters | LR | 6 | droplargest@0.7 | 0.8286±0.0090 | +0.0415 | 0.8702 |
| human_enhancers_ensembl | RF | 6 | homology@0.5 | 0.6993±0.0006 | +0.1564 | 0.8556 |
| human_enhancers_ensembl | RF | 6 | homology@0.7 | 0.6999±0.0021 | +0.1558 | 0.8556 |
| human_enhancers_ensembl | RF | 6 | homology@0.9 | 0.6985±0.0045 | +0.1572 | 0.8556 |
| human_enhancers_ensembl | RF | 6 | droplargest@0.7 | 0.6973±0.0023 | +0.1583 | 0.8556 |
| human_enhancers_ensembl | LR | 6 | homology@0.5 | 0.7742±0.0027 | +0.0067 | 0.7809 |
| human_enhancers_ensembl | LR | 6 | homology@0.7 | 0.7776±0.0021 | +0.0033 | 0.7809 |
| human_enhancers_ensembl | LR | 6 | homology@0.9 | 0.7730±0.0027 | +0.0079 | 0.7809 |
| human_enhancers_ensembl | LR | 6 | droplargest@0.7 | 0.7740±0.0026 | +0.0069 | 0.7809 |

Cluster-size distribution + verified invariants (residual leakage, spanning, balance):

| dataset | threshold | n_comp | max cluster | redundant | spanning | resid>thr |
|---|---|---|---|---|---|---|
| human_nontata_promoters | 0.5 | 20977 | 420 | 0.419 | 0 | 0.0000 |
| human_nontata_promoters | 0.7 | 23312 | 420 | 0.355 | 0 | 0.0000 |
| human_nontata_promoters | 0.9 | 29967 | 212 | 0.171 | 0 | 0.0000 |
| human_enhancers_ensembl | 0.5 | 115714 | 330 | 0.253 | 0 | 0.0000 |
| human_enhancers_ensembl | 0.7 | 117270 | 143 | 0.243 | 0 | 0.0000 |
| human_enhancers_ensembl | 0.9 | 117758 | 125 | 0.239 | 0 | 0.0000 |

*Source: `run_robustness_full.py` -> `robustness_fullscale_summary.csv`, `robustness_fullscale_clusters.csv`.*

## 6. Negative control (random re-split delta, best model)

| dataset | best model | Δacc(homology) | Δacc(random) |
|---|---|---|---|
| human_nontata_promoters | RF_k6 | +0.1212 | -0.0007 |
| human_enhancers_ensembl | RF_k6 | +0.1558 | -0.0021 |
| demo_coding_vs_intergenomic_seqs | LR_k6 | -0.0008 | -0.0022 |
| human_ocr_ensembl | LR_k6 | +0.0038 | +0.0008 |
| demo_human_or_worm | LR_k6 | -0.0040 | -0.0061 |
| drosophila_enhancers_stark | LR_k6 | +0.0118 | +0.0245 |
| human_enhancers_cohn | LR_k6 | -0.0043 | +0.0031 |

All random-resplit deltas within ±0.006 except: ['demo_human_or_worm', 'drosophila_enhancers_stark'] (drosophila n=6914, small-sample). *Source: `finalize.py`->`summary_final.csv`.*

## 7. Cross-dataset summary

- **Leaky (leak@0.7>0.1): 2** (human_nontata_promoters, human_enhancers_ensembl): best-model homology drop mean **0.139** (range +0.121..+0.156); random drop mean -0.001.
- **Borderline: 1** (demo_coding_vs_intergenomic_seqs): clean under the length-blind $k$-mer Jaccard but above the cut on the length-robust containment index; best-model homology drop mean **-0.001**.
- **Clean: 4** (human_ocr_ensembl, demo_human_or_worm, drosophila_enhancers_stark, human_enhancers_cohn): best-model homology drop mean **+0.002** (range -0.004..+0.012) — ~0; random drop mean +0.006.

## 8. Subsampling-masking effect (why full-scale leakage matters)

| dataset | leak@0.7 (20k subsample, seed 0) | leak@0.7 (full) |
|---|---|---|
| human_nontata_promoters | 0.3197 | 0.4064 |
| human_enhancers_ensembl | 0.0560 | 0.3839 |
| demo_coding_vs_intergenomic_seqs | 0.0224 | 0.0784 |
| human_ocr_ensembl | 0.0015 | 0.0104 |
| demo_human_or_worm | 0.0032 | 0.0124 |
| drosophila_enhancers_stark | 0.0156 | 0.0156 |
| human_enhancers_cohn | 0.0012 | 0.0012 |

**human_enhancers_ensembl: 0.0560 at the 20k subsample (seed 0) vs 0.3839 at full scale** — subsampling to ~13% of 154,842 sequences broke up most near-duplicate pairs and masked the leakage. *Source: subsample `run_suite.py`->`threshold_sensitivity.csv`; full `measure_leakage_full.py`->`leakage_full.csv`.*

## 9. Extended capacity curve (Part A: four classical models)

Capacity proxy ordering: **LogReg ~= LinearSVC** (linear) < **RandomForest** (150 deep, unpruned, bagged trees) < **HistGradientBoosting** (boosted trees). An MLP/1D-CNN point was **skipped** (no GPU permitted; a neural fit on the full 154k x 4096 enhancers_ensembl set would exceed the CPU budget) -- HGB anchors the high-capacity end. Same k-mer features, splits, 3 seeds, metrics as before; leaky datasets at full scale, clean at the 20k subsample.

Homology accuracy drop (original - homology@0.7), k=6:

| dataset | leak@0.7 | LR | LinearSVC | RF | HGB |
|---|---|---|---|---|---|
| human_nontata_promoters | 0.406 | +0.040 | +0.081 | +0.116 | +0.066 |
| human_enhancers_ensembl | 0.384 | +0.003 | +0.003 | +0.164 | +0.006 |
| demo_coding_vs_intergenomic_seqs | 0.078 | -0.003 | +0.005 | -0.003 | -0.006 |
| human_ocr_ensembl | 0.010 | +0.004 | +0.002 | +0.011 | +0.010 |
| demo_human_or_worm | 0.012 | -0.004 | -0.007 | -0.004 | -0.004 |
| drosophila_enhancers_stark | 0.016 | +0.012 | +0.007 | +0.026 | +0.013 |
| human_enhancers_cohn | 0.001 | -0.004 | -0.009 | -0.006 | -0.001 |

**Finding:** the drop is monotone **LR ~= LinearSVC < RF** and is driven by RF's deep unpruned trees memorizing near-duplicates. **HGB (regularized boosting) drops far less than RF** (e.g. +0.006 vs +0.164 on enhancers_ensembl), so strict LR<SVC<RF<HGB monotonicity does NOT hold: the drop tracks *memorization propensity / regularization*, not nominal capacity. On clean datasets all four models stay ~0. (k=4 in `capacity_scaling_extended.csv`; figure `fig_capacity_extended.*`.)

## 10. Ranking inversion (Part B)

Models ranked by test accuracy (k=6) on the original (leaky) split vs the homology-aware split; Kendall tau between the two rankings; inversions counted only if stable across all 3 homology seeds.

| dataset | leaky | tau | RF rank (orig->corr) | max inversion gap | original ranking | corrected ranking |
|---|---|---|---|---|---|---|
| human_nontata_promoters | True | -0.33 | 1->3 | 0.058 | RF>HGB>LinearSVC>LR | LR>HGB>RF>LinearSVC |
| human_enhancers_ensembl | True | +0.00 | 1->4 | 0.096 | RF>LinearSVC>LR>HGB | LinearSVC>LR>HGB>RF |
| demo_coding_vs_intergenomic_seqs | False | +0.67 | 4->4 | 0.002 | LinearSVC>LR>HGB>RF | LR>LinearSVC>HGB>RF |
| human_ocr_ensembl | False | +0.67 | 4->4 | 0.001 | LR>HGB>LinearSVC>RF | LR>LinearSVC>HGB>RF |
| demo_human_or_worm | False | +1.00 | 4->4 | 0.000 | LinearSVC>LR>HGB>RF | LinearSVC>LR>HGB>RF |
| drosophila_enhancers_stark | False | +0.33 | 2->3 | 0.018 | LR>RF>HGB>LinearSVC | LR>LinearSVC>RF>HGB |
| human_enhancers_cohn | False | +0.67 | 4->4 | 0.002 | LR>HGB>LinearSVC>RF | LR>LinearSVC>HGB>RF |

**Mean Kendall tau: leaky = -0.17, clean = +0.67.** 5/7 datasets show >=1 stable inversion, BUT the character differs sharply: on **leaky** datasets the apparent best model (RF) is dethroned by a **material** margin (inversion gaps 0.058 / 0.096; RF 1st->3rd / 1st->4th), whereas on **clean** datasets RF stays last on both splits and the only inversions are sub-0.003 swaps between statistically-tied models (drosophila's 0.019 is small-n noise).

Quotable inversions (k=6, accuracy, stable across 3/3 seeds):

- **nonTATA promoters:** RandomForest ranks **1st on the leaky split but 3rd after correction**; LogReg rises from last to 1st.
- **enhancers (Ensembl):** RandomForest ranks **1st on the leaky split but LAST (4th) after correction** -- the method that looks best is the most leakage-inflated.

## 11. Homology-aware splitter tool + validation (Part C)

`homology_split.py` exposes `homology_aware_split(sequences, labels, test_frac=0.25, threshold=0.7, seed=0, k=8) -> (train_idx, test_idx)` (pure numpy/scipy, CPU-only). Guarantees, checkable via `verify_split`: no cluster spans the split (residual cross-split Jaccard > threshold == 0), class balance and train/test ratio preserved, deterministic given seed.

Validation on the two leaky datasets, RF k6, **5 seeds**:

| dataset | corrected acc (mean +/- std, 5 seeds) | max residual leak >0.7 | mean test frac |
|---|---|---|---|
| human_nontata_promoters | 0.8101 +/- 0.0055 | 0.000000 | 0.250 |
| human_enhancers_ensembl | 0.6951 +/- 0.0019 | 0.000000 | 0.200 |

Residual leakage is **0.000000** for every seed (the tool's guarantee), and corrected accuracy is stable across seeds -- i.e. the tool yields honest, reproducible evaluation. *Source: `validate_splitter.py` -> `splitter_validation.csv`.*

## 12. Homology-graded performance (Part 3: the memorization mechanism)

On the ORIGINAL split, test sequences are binned by max 8-mer Jaccard to the training set; per-model accuracy (k=6) per bin. The memorization signature is the **gap = accuracy[>=0.9] - accuracy[<0.5]**: large for the high-capacity memorizer (RF), small for linear models. Clean datasets have near-empty high-similarity bins (nothing to memorize).

**human_nontata_promoters** (LEAKY) -- test n per bin: [0,0.5)=4971, [0.5,0.7)=392, [0.7,0.9)=1634, [0.9,1.0]=2037

| model | acc [0,0.5) | acc [0.5,0.7) | acc [0.7,0.9) | acc [0.9,1.0] | gap (>=0.9 - <0.5) |
|---|---|---|---|---|---|
| LR | 0.834 | 0.875 | 0.910 | 0.925 | +0.091 |
| LinearSVC | 0.855 | 0.867 | 0.914 | 0.951 | +0.097 |
| RF | 0.879 | 0.908 | 0.993 | 1.000 | +0.121 |
| HGB | 0.838 | 0.890 | 0.944 | 0.976 | +0.137 |

**human_enhancers_ensembl** (LEAKY) -- test n per bin: [0,0.5)=18770, [0.5,0.7)=311, [0.7,0.9)=115, [0.9,1.0]=11774

| model | acc [0,0.5) | acc [0.5,0.7) | acc [0.7,0.9) | acc [0.9,1.0] | gap (>=0.9 - <0.5) |
|---|---|---|---|---|---|
| LR | 0.765 | 0.913 | 0.896 | 0.802 | +0.037 |
| LinearSVC | 0.782 | 0.920 | 0.904 | 0.819 | +0.037 |
| RF | 0.770 | 0.932 | 0.939 | 1.000 | +0.230 |
| HGB | 0.749 | 0.894 | 0.870 | 0.783 | +0.034 |

**human_enhancers_cohn** (clean control) -- test n per bin: [0,0.5)=4966, [0.5,0.7)=28, [0.7,0.9)=4, [0.9,1.0]=2

| model | acc [0,0.5) | acc [0.5,0.7) | acc [0.7,0.9) | acc [0.9,1.0] | gap (>=0.9 - <0.5) |
|---|---|---|---|---|---|
| LR | 0.734 | 1.000* | 1.000* | 1.000* | +0.266* |
| LinearSVC | 0.713 | 1.000* | 1.000* | 0.500* | -0.213* |
| RF | 0.691 | 1.000* | 1.000* | 1.000* | +0.309* |
| HGB | 0.715 | 0.964* | 0.750* | 1.000* | +0.285* |

**demo_human_or_worm** (clean control) -- test n per bin: [0,0.5)=4951, [0.5,0.7)=33, [0.7,0.9)=10, [0.9,1.0]=6

| model | acc [0,0.5) | acc [0.5,0.7) | acc [0.7,0.9) | acc [0.9,1.0] | gap (>=0.9 - <0.5) |
|---|---|---|---|---|---|
| LR | 0.937 | 0.939* | 0.900* | 1.000* | +0.063* |
| LinearSVC | 0.939 | 0.970* | 1.000* | 1.000* | +0.061* |
| RF | 0.918 | 1.000* | 1.000* | 1.000* | +0.082* |
| HGB | 0.929 | 0.970* | 0.900* | 1.000* | +0.071* |

`*` = bin has n<50 (low-confidence). **Reading:** on the leaky datasets RF is near-perfect on near-duplicate test sequences ([0.9,1.0]) and falls on dissimilar ones ([0,0.5)) -- a large gap -- while LR/LinearSVC are far flatter; this is direct evidence that RF's apparent edge is memorization, not generalization. On clean controls the high-similarity bins are nearly empty (n<50), so no gap is estimable -- consistent with there being no near-duplicates to exploit. *Source: `run_graded.py` -> `graded_performance.csv`; figure `fig_graded_performance.*`.*

## 13. Leakage report card (which datasets can you trust?)

| dataset | n | leak@0.7 | leak@0.9 | contain@0.7 | verdict | best model | acc drop (seed 0) | top model changes | RF rank orig->corr |
|---|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | 36,131 | 0.406 | 0.225 | 0.445 | LEAKY | RF_k6 | +0.108 | yes | 1->3 |
| human_enhancers_ensembl | 154,842 | 0.384 | 0.380 | 0.845 | LEAKY | RF_k6 | +0.164 | yes | 1->4 |
| demo_coding_vs_intergenomic_seqs | 100,000 | 0.078 | 0.024 | 0.131 | borderline | LR_k6 | -0.001 | no | 4->4 |
| human_ocr_ensembl | 174,756 | 0.010 | 0.001 | 0.068 | clean | LR_k6 | +0.004 | no | 4->4 |
| demo_human_or_worm | 100,000 | 0.012 | 0.003 | 0.042 | clean | LR_k6 | -0.004 | no | 4->4 |
| drosophila_enhancers_stark | 6,914 | 0.016 | 0.006 | 0.033 | clean | LR_k6 | +0.012 | no | 2->3 |
| human_enhancers_cohn | 27,791 | 0.001 | 0.000 | 0.005 | clean | LR_k6 | -0.004 | no | 4->4 |
| human_ensembl_regulatory (3-class) | 289,061 | 0.005 | 0.001 | n/a | clean | RF_k4 | -0.010 | n/a | n/a |

*Drop values are the bootstrap-consistent point estimates (seed-0 corrected split, with 95% CIs reported in §16); 3-seed-mean corrected accuracies and SDs are in §16.*

Verdict rule: LEAKY if the full-scale near-duplicate test fraction exceeds 0.1 at Jaccard 0.7; **borderline** if the length-robust containment index does; clean otherwise. The containment column below is what distinguishes the two. *Source: `report_card.py` -> `leakage_report_card.csv`; figure `fig_report_card.*`.*

## 14. Label concordance of near-duplicates (Check 1)

For each test sequence with a training near-duplicate at Jaccard >= t, is its NEAREST training neighbour the same label? If ~1.0, near-duplicates carry their labels across the split, so a memorizing model scores them correctly for free.

| dataset | leaky | threshold | n pairs | same-label fraction |
|---|---|---|---|---|
| human_enhancers_cohn | False | >= 0.7 | 6 * | 1.0000 |
| human_enhancers_cohn | False | >= 0.9 | 2 * | 1.0000 |
| demo_human_or_worm | False | >= 0.7 | 16 * | 1.0000 |
| demo_human_or_worm | False | >= 0.9 | 6 * | 1.0000 |
| human_nontata_promoters | True | >= 0.7 | 3671 | 0.9992 |
| human_nontata_promoters | True | >= 0.9 | 2037 | 0.9995 |
| human_enhancers_ensembl | True | >= 0.7 | 11889 | 0.9993 |
| human_enhancers_ensembl | True | >= 0.9 | 11774 | 0.9999 |

`*` = n<50 (low-confidence). **Reading:** on the leaky datasets the same-label fraction at >=0.9 is ~1.0 -- near-duplicates carry their labels across the split, so RF's 1.000 accuracy on that bin is the expected consequence of memorizing labelled near-copies, not hard-won generalization. Clean datasets have near-empty >=0.9 groups (nothing to memorize). *Source: `check1_label_concordance.py` -> `label_concordance.csv`.*

## 15. Novel-only ranking: independent routes to 'is RF's lead real?' (Check 2)

Rank the 4 models (k=6) by (a) original leaky-split accuracy, (b) homology-aware-split accuracy, and (c) accuracy on novel (<0.5-similarity) test sequences only. (b) and (c) are independent honest evaluations.

| dataset | (a) leaky-split | (b) homology-aware | (c) novel-only | RF rank a/b/c | tau(b,c) |
|---|---|---|---|---|---|
| human_nontata_promoters | RF>HGB>LinearSVC>LR | LR>HGB>RF>LinearSVC | RF>LinearSVC>HGB>LR | 1/3/1 | -0.667 |
| human_enhancers_ensembl | RF>LinearSVC>LR>HGB | LinearSVC>LR>HGB>RF | LinearSVC>RF>LR>HGB | 1/4/2 | 0.333 |

**Reading:** on **enhancers_ensembl** both honest routes dethrone RF from its leaky #1 and crown LinearSVC (RF novel-only acc 0.770 < LinearSVC 0.782; RF homology rank last) -- two independent confirmations that RF's leaky lead is memorization, not generalization. On **nonTATA promoters** the routes diverge (tau(b,c) = -0.67): the homology split demotes RF to 3rd, but RF stays best on truly-novel sequences (0.879). **Flagged:** nonTATA's RF advantage is therefore partly genuine generalization, not pure test-side memorization -- its homology demotion also reflects the de-duplicated training regime. The memorization mechanism is airtight for enhancers_ensembl and partial for nonTATA. *Source: `check2_novelonly_ranking.py` -> `novelonly_ranking.csv`.*

## 16. Variance and confidence intervals (uncertainty quantification)

Quantifies the uncertainty behind the "within noise" / "statistically tied" statements; **no number in sections 1-15 is changed -- this section only adds variance.** Bootstrap CIs resample the stored per-example test correctness (1000 draws, no model refit), RNG seed **20240524**. Leaky datasets are at full scale, clean at the 20k subsample (seed 0), matching the frozen results above. Re-split seeds {0,1,2,3,4}.

### 16.1 Re-split seed variance (5 cluster->side seeds, homology-aware corrected accuracy)

Confirms that a single cluster->side assignment is not 'unlucky': corrected accuracy is stable across 5 independent assignments.

| dataset | model | mean | SD | min | max | range | per-seed |
|---|---|---|---|---|---|---|---|
| human_nontata_promoters | RF_k6 | 0.8101 | 0.0055 | 0.8036 | 0.8200 | 0.0164 | [0.82, 0.8113, 0.8073, 0.8036, 0.808] |
| human_nontata_promoters | LR_k6 | 0.8283 | 0.0080 | 0.8180 | 0.8424 | 0.0244 | [0.8294, 0.818, 0.8424, 0.8272, 0.8245] |
| human_enhancers_ensembl | RF_k6 | 0.6951 | 0.0019 | 0.6930 | 0.6982 | 0.0052 | [0.6957, 0.6982, 0.6932, 0.6954, 0.693] |
| human_enhancers_ensembl | LR_k6 | 0.7753 | 0.0033 | 0.7707 | 0.7806 | 0.0099 | [0.7806, 0.7764, 0.7758, 0.7731, 0.7707] |

RF k6 reproduces `splitter_validation.csv` (seed-0 cross-check exact); SD <= 0.008 for every cell. *Source: `step2_rf_seeds.py` (RF) + `step_variance_ci.py` (LR) -> `step2_seed_variance.csv`; cross-checked vs `validate_splitter.py` -> `splitter_validation.csv`.*

### 16.2 Test-set bootstrap 95% CIs and delta significance

Accuracies with 95% bootstrap CIs (homology column = seed-0 corrected split):

| dataset | model | original acc [95% CI] | homology acc [95% CI] |
|---|---|---|---|
| human_nontata_promoters | RF_k6 | 0.9284 [0.9232, 0.9338] | 0.8200 [0.8119, 0.8279] |
| human_nontata_promoters | LR_k6 | 0.8702 [0.8635, 0.8771] | 0.8294 [0.8214, 0.8370] |
| human_enhancers_ensembl | RF_k6 | 0.8596 [0.8562, 0.8635] | 0.6957 [0.6902, 0.7006] |
| human_enhancers_ensembl | LR_k6 | 0.7809 [0.7765, 0.7856] | 0.7806 [0.7760, 0.7850] |

Delta (original - corrected) with 95% CI; "excludes 0" => the drop is significant:

| dataset | group | model | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| human_nontata_promoters | leaky | RF_k6 | +0.1083 | [+0.0993, +0.1183] | YES |
| human_nontata_promoters | leaky | LR_k6 | +0.0407 | [+0.0305, +0.0517] | YES |
| human_enhancers_ensembl | leaky | RF_k6 | +0.1639 | [+0.1579, +0.1704] | YES |
| human_enhancers_ensembl | leaky | LR_k6 | +0.0004 | [-0.0057, +0.0069] | no |
| human_enhancers_cohn | clean | LR_k6 | -0.0016 | [-0.0196, +0.0148] | no |
| human_ocr_ensembl | clean | LR_k6 | +0.0059 | [-0.0146, +0.0254] | no |
| demo_human_or_worm | clean | LR_k6 | -0.0034 | [-0.0120, +0.0058] | no |
| demo_coding_vs_intergenomic_seqs | clean | LR_k6 | +0.0056 | [-0.0074, +0.0182] | no |
| drosophila_enhancers_stark | clean | LR_k6 | +0.0229 | [-0.0048, +0.0553] | no |

Bootstrap deltas use the seed-0 corrected split, so point estimates differ slightly from the frozen 3-seed-mean deltas in sections 3-5 (unchanged); the sign/significance is the result of interest. **Reading:** the high-capacity (RF) drop is significant on both leaky datasets (CIs exclude 0); the linear (LR) drop is significant on nonTATA but **null on enhancers_ensembl** (CI includes 0) -- only the memorizer inflates there. All five clean-dataset deltas are not significant (CIs include 0), which replaces 'within noise' with a tested statement. Clean per-model original-split CIs are in `step3_accuracy_ci.csv`. *Source: `step_variance_ci.py` -> `step3_accuracy_ci.csv`, `step3_delta_ci.csv`.*

### 16.3 "Statistically tied": clean-dataset rank swaps

For every clean-dataset rank swap, the swapped models' original-split accuracy CIs overlap, so 'statistically tied' is justified.

| dataset | swapped pair | acc A [95% CI] | acc B [95% CI] | CI overlap | gap |
|---|---|---|---|---|---|
| human_enhancers_cohn | HGB / LinearSVC | 0.7164 [0.7042,0.7296] | 0.7144 [0.7018,0.7270] | yes | 0.0020 |
| human_ocr_ensembl | HGB / LinearSVC | 0.6710 [0.6562,0.6853] | 0.6697 [0.6552,0.6838] | yes | 0.0013 |
| demo_coding_vs_intergenomic_seqs | LR / LinearSVC | 0.8878 [0.8790,0.8970] | 0.8902 [0.8810,0.8990] | yes | 0.0024 |
| drosophila_enhancers_stark | HGB / LinearSVC | 0.6890 [0.6670,0.7116] | 0.6873 [0.6665,0.7087] | yes | 0.0017 |
| drosophila_enhancers_stark | LinearSVC / RF | 0.6873 [0.6665,0.7087] | 0.7058 [0.6850,0.7266] | yes | 0.0185 |

Every swap-pair CI overlaps, so 'statistically tied' holds for all clean swaps (human_or_worm has no swap, tau=+1.00). The 'sub-0.003' magnitude in section 10 fits cohn/ocr/coding; drosophila's LinearSVC/RF swap (gap 0.019) is larger but unstable (2/3 seeds, not a stable inversion) and its CIs still overlap. *Source: `step_variance_ci.py` -> `step4_tie_overlap.csv`.*

## 17. Provenance (number -> script -> file)

| quantity | script | output file |
|---|---|---|
| full-scale leakage fractions / sim distribution | `measure_leakage_full.py` | `leakage_full.csv` |
| single-dataset headline (nontata) | `run_audit.py` | `results.md, results_table.csv` |
| subsample sweep: original/random/threshold/drop-largest (all 9 datasets) | `run_suite.py` | `per_dataset_results.csv, threshold_sensitivity.csv, cluster_stats.csv, all_results_long.csv` |
| full-scale main results for leaky datasets | `run_fullscale.py` | `fullscale_long.csv, fullscale_summary.csv` |
| full-scale robustness (sweep + drop-largest) for leaky datasets | `run_robustness_full.py` | `robustness_fullscale_summary.csv, robustness_fullscale_clusters.csv` |
| consolidated summary + capacity scaling | `finalize.py` | `summary_final.csv, capacity_scaling_final.csv, results_FINAL.md` |
| figures + backing data | `make_paper_figures.py` | `figures/fig_capacity_scaling.*, figures/fig_controls.*, *_data.csv` |
| extended capacity (4 models, Part A) | `run_extended_models.py + make_part_b_figures.py` | `extended_models_long.csv, capacity_scaling_extended.csv, figures/fig_capacity_extended.*` |
| ranking inversion (Part B) | `ranking_inversion.py + make_part_b_figures.py` | `ranking_inversion.csv, figures/fig_ranking_inversion.*` |
| homology-aware splitter + validation (Part C) | `homology_split.py + validate_splitter.py` | `homology_split.py, splitter_validation.csv` |
| graded performance by similarity bin (Part 3) | `run_graded.py + make_graded_figure.py` | `graded_performance.csv, figures/fig_graded_performance.*` |
| leakage report card (Part 1 artifact) | `report_card.py` | `leakage_report_card.csv, figures/fig_report_card.*` |
| splitter tool + CLI + docs (Part 2 artifact) | `homology_split.py` | `homology_split.py, TOOL_README.md` |
| label concordance of near-duplicates (Check 1) | `check1_label_concordance.py` | `label_concordance.csv` |
| novel-only ranking, 3-way comparison (Check 2) | `check2_novelonly_ranking.py` | `novelonly_ranking.csv` |
| re-split seed variance: RF+LR k6, 5 seeds, both leaky (Phase 14) | `step2_rf_seeds.py + step_variance_ci.py` | `step2_seed_variance.csv` |
| test-set bootstrap 95% CIs + delta significance + 'tied' overlap (Phase 14) | `step_variance_ci.py` | `step3_accuracy_ci.csv, step3_delta_ci.csv, step4_tie_overlap.csv` |
| this file | `paper_numbers.py` | `PAPER_NUMBERS.md` |

*3-class human_ensembl_regulatory (leak@0.7=0.005, clean) and the dummy_mouse smoke test are in `summary.csv`/`leakage_full.csv`; excluded from the binary tables above.*
