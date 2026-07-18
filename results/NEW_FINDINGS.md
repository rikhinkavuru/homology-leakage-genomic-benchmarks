# NEW_FINDINGS.md — revision experiment results (running log)

Durable evidence log for the resubmission. Each entry: finding, key numbers, source script/CSV, reviewer item. Updated as experiments land. `[RAN]`=executed here.

## Confirmed (locked)

### R3.3 cluster/block bootstrap — `cluster_bootstrap.py` → `results/cluster_bootstrap.csv` [RAN]
Cluster bootstrap over within-test 8-mer-Jaccard>0.7 components. Both leaky datasets: CIs widen ×1.0–1.5, **no verdict flips**. RF drops still exclude 0 (nontata delta CI [0.093,0.126]; ensembl [0.156,0.172]); ensembl-LR stays null both ways. Design effect small because test-internal clustering ≈1.06–1.19 seq/cluster (near-dups are test-to-**train**, not test-to-test). → R3.3 answered; the correct tool changes nothing. (Full grid + ICC/deff/CRVE/combined-source CI: `exp_clusterboot_full.py`, pending.)

### R2.a1 balance provenance [RAN]
All GB datasets curated to 0.500 pos-fraction (positive/negative subfolders, train+test); nontata natural mild 0.544; 3-class 0.37/0.30/0.33. → "preprocessed-balanced, not natural"; imbalanced test warranted.

### T1.8 decision-rule + number reconciliation — `results/reconciled_numbers.md` [RAN]
Real cause of the 15.6/16.0/16.4 (ensembl) & 10.8/12.1/12.2 (nontata) spread = `RA.eval_split` uses `predict_proba>=0.5` (`run_audit.py:158`, used by `run_suite.py:249`/`run_fullscale.py:51`/`run_robustness_full.py:45`) while variance scripts use `.predict()`; PLUS estimator inconsistency (seed-0 vs 5-seed-mean). NOT thread-nondeterminism (RF is bit-deterministic; it's deterministic vote-tie breaking, ~0.4–1% of test). **Reconciled headline (`.predict()` + 5-seed-mean, uniform): nontata 11.8 pts (0.9284→0.8101±0.0055), ensembl 16.5 pts (0.8596→0.6951±0.0019).** One-line fix: `run_audit.py:158` → `.predict()`.

### AUROC reversal + train-side loss (from frozen PAPER_NUMBERS) [VERIFIED]
- nontata corrected AUROC: RF 0.9220 > LR 0.9072 → demotion is accuracy@0.5-only; ranking-change claim reduces to **1/2 leaky, accuracy-only**. (ensembl survives AUROC: RF 0.803 < SVC 0.854.)
- Leaky-trained RF novel-bin acc 0.770 (ensembl) vs corrected RF 0.696 → part of the drop is de-duplicated-**training** signal loss; corrected drop is a biased-high estimator (R1.2 quantified).

### Exact-duplicate census — `exact_dup_count.py` [RAN]
ensembl: **11,774 = 0.380 of test are byte-identical** train copies (leakage = literal duplication; "homology" indefensible; alignment a formality). nontata: **0 exact**, yet 0.225 near-dup at ≥0.9 (genuinely mutated → the more "homology-like" but statistically weaker case). Reverse-complement adds ≤10 anywhere. demo_human_or_worm (clean) has 2.5% literal dups.

### T1.1 rankings under acc/AUROC/F1 + P(rank1) — `results/exp_rankings.csv`, `exp_rank_prob.csv` [RAN]
**ensembl: RF demoted to rank 4 under ALL THREE metrics** (acc/AUROC/F1) after correction → airtight, metric-robust. **nontata: RF demoted to rank 3 under accuracy ONLY; stays rank 1 under BOTH AUROC and F1** → the demotion is an accuracy@0.5 artifact. Combined with chromosome-holdout (reproduces ensembl, not nontata) + novel-only τ=−0.67 → the "which model wins" claim is cleanly **n=1 (ensembl)** across FOUR independent axes (metric-robustness, chromosome control, novel-only ranking, CI-overlap).

### T1.6 / C-2 RF regularization path — `exp_regpath.py` → `results/exp_regpath.csv` [RAN, COMPLETE both datasets]
Airtight C-2 resolution. At default (min_samples_leaf=1, max_depth=None) RF scores **1.000 on the ≥0.9-sim near-dup bin** (pure-leaf memorization) with large drop (ensembl 0.164, nontata 0.108) and graded gap (0.230/0.121), RF rank 1. Regularizing collapses everything monotonically: min_samples_leaf 1→500 drops the homology drop to ≈0 (ensembl 0.164→−0.001; nontata 0.108→0.000), the graded gap shrinks (ensembl 0.230→0.020; nontata 0.121→0.061), and acc on the ≥0.9 bin falls from 1.000 to ~0.71–0.82. max_depth None→4 does the same. **Crucially rf_rank_orig flips from 1 (msl≤5) to 4 (msl≥20)**: the "RF wins on the leaky split" phenomenon REQUIRES the unregularized RF. → Owned, publishable finding: the leakage inflation + demotion is a property of the DEFAULT high-capacity RF practitioners deploy; heavy regularization removes the memorization but also the apparent lead — either way a homology-aware split reveals the truth. The graded gap is the confound-immune headline (tracks memorization propensity, not overall accuracy). Answers R3.2 and blocker C-2.

### GB7 / G5 canonical (revcomp) k-mers — `exp_canonical.py` → `results/exp_canonical_models.csv` [RAN, COMPLETE]
Strand-symmetric revcomp-collapsed features keep the RF demotion on BOTH: ensembl RF **1→4, drop 0.153**; nontata RF **1→3, drop 0.097** → the memorization is real, NOT forward-strand-only overfitting; canonicalization does not rescue RF. Leakage barely changes under canonicalization (nontata 0.406→0.416) → revcomp near-dup leakage empirically minor (G5 small).

### G6 P(rank-1) bootstrap — `results/exp_rank_prob.csv` [RAN]
The correct "which model wins" estimator (sidesteps CI-overlap): ensembl **P(RF=1st)=1.0 leaky → P(LinearSVC=1st)=1.0 corrected** (sample AND cluster bootstrap) = bulletproof unambiguous flip. nontata P(RF=1st)=1.0 leaky, but corrected has NO dominant winner (LR 0.63, HGB 0.31, RF 0.06) = genuine 3-way tie, not a clean inversion. Reinforces n=1 (ensembl).

### G8 clean-dataset RF delta CIs — `results/exp_clean_rf_ci.csv` [RAN]
All 5 clean datasets, RF k6+k4: every delta CI **includes 0** (drosophila RF_k6 +0.024 [−0.008,0.058] the largest, still includes 0). So "all clean deltas include 0" holds for the memorizer RF, not just LR — the positive control is validated for the relevant model.

### GB1 containment-index leakage — `exp_geometry.py` [RAN, partial]
Length-robust containment index reveals leakage Jaccard misses: **ensembl 0.384 (Jaccard) → 0.845 (containment)** (length cap 0.003 — ultra-short seqs + length disparity hide >half the leakage from Jaccard); nontata 0.406→0.445 (fixed length, cap 1.0, small change); cohn 0.0012→0.0042. → Jaccard leak fractions are LOWER BOUNDS; the containment metric is the length-robust fix. (Full re-measure across all datasets pending re-run.)

### T1.2 deep-model CNN dose-response — `exp_deep.py` → `results/exp_deep_cnn.csv` [RAN, in progress on MPS]
From-scratch 1D residual CNN, cluster-bootstrap CIs. nontata: **standard-practice regularized CNN (dropout 0.3, wd 1e-4) drop +0.043, cluster CI [0.025, 0.061] — excludes 0** → a properly-regularized deep net IS inflated by leakage on nontata. Unregularized CNN +0.061 [0.045,0.079]. → the mechanism generalizes to trained deep nets (answers R1.1/R2.a3/R3.1). Caveat: per-cell single-run variance non-trivial (novel-acc swings 0.71–0.85 from early-stopping/MPS); read the mechanism from the grid (unregularized-converged vs reference), not per-cell. (ensembl reference cell = the decider for the pre-registered refutation condition; pending.)

### T1.11 / GB2 chromosome-holdout (genomics-standard control) — `chromosome_holdout.py` → `results/chromosome_holdout.csv` [RAN]
Coordinates recovered; leave-one-chromosome-out split, 4 models k6, vs original split:
- **ensembl: RF demotion REPRODUCES** — original RF rank 1 (0.8596) → chromosome-holdout rank **4** (0.7069); new order LinearSVC>LR>HGB>RF, identical to the homology-aware split. The marquee "which model wins" result is **bulletproof on the deployment-relevant control**, not a clustering artifact.
- **nontata: RF is NOT demoted** — stays rank **1** (0.9284→0.8199; order RF>HGB>LR>LinearSVC). So nontata's homology-split demotion does **not** reproduce under the chromosome control → its demotion is partly an artifact of aggressive de-duplication; RF retains genuine generalization. This is a THIRD independent signal (with AUROC-reversal and novel-only τ=−0.67) that nontata is the cautionary partial case → central ranking-change claim is cleanly **n=1 (ensembl)**.
- Chromosome-holdout corrected accuracies (ensembl 0.707, nontata 0.820) closely match the homology-aware split (0.696, 0.810) → the two controls AGREE on the drop magnitude, validating the homology-aware split as a good proxy for the deployment control.
- Residual near-duplicate leakage surviving the chromosome split: ensembl 0.0082, nontata 0.0002 (vs homology-split's exact 0) → honest nuance: chromosome-holdout is standard but NOT leakage-free (intra-chromosome near-dups survive); the homology-aware split guarantees zero.

### T1.2 deep-model CNN — COMPLETE, both leaky datasets — `results/exp_deep_cnn.csv` [RAN]
From-scratch 1D residual CNN (one-hot, no k-mer features), 9-config dropout×wd dose-response + unregularized-converged manipulation check, cluster-bootstrap CIs.
- **Standard-practice regularized reference cell (dropout 0.3, wd 1e-4) drops on BOTH: nontata +0.0426 cluster-CI [0.0245,0.0614]; ensembl +0.0141 [0.0067,0.0211] — both exclude 0.** Per the pre-registered refutation condition (strong reading refuted only if CI includes 0 on both), the strong reading is **NOT refuted** → a properly-regularized deep net is inflated by near-duplicate leakage.
- **19 of 20 configs drop with cluster-CI excluding 0** (all 9 ensembl grid cells, 8 of 9 nontata grid cells, both manip checks); the one exception is nontata dropout 0.6/wd 1e-4 where heavy dropout underfits (acc rises 0.020) → robust, not cherry-picked; per-cell single-run variance non-trivial, read the grid-level contrast.
- **Unregularized-converged manipulation check memorizes hardest**: ensembl graded gap **+0.2217** (acc 0.974 on ≥0.9-sim bin vs 0.752 novel), drop +0.076 [0.068,0.084]; nontata drop +0.101 [0.085,0.117]. Mirrors the unregularized RF (gap 0.230) exactly.
- Reading: the deep net confirms the mechanism generalizes beyond classical models; drop magnitude tracks regularization (less regularization → bigger drop + bigger memorization gap), consistent with "memorization propensity, not capacity." Answers R1.1a/R2.a3/R3.1(i). DNABERT-2/HyenaDNA (R3.1 ii/iii) remain GPU-gated + pretraining-contaminated (scoped out) — pending user GPU decision.

### T1.4 / R2.a2 alignment validation (MMseqs2) — `exp_alignment.py` → `results/exp_alignment.csv` [RAN]
Re-cluster by MMseqs2 alignment identity (min-seq-id 0.7, c 0.8), same whole-cluster splitter, refit 4 models:
- **ensembl: RF drop metric-INDEPENDENT** — 8-mer Jaccard 0.1639 ≈ MMseqs2 alignment 0.1618; other models ~0 under both; RF still demoted to rank 4. Defuses the k-mer-circularity concern (R2.a2): the leakage effect is real regardless of near-duplicate definition (Jaccard/alignment). NOTE: MMseqs2 is itself k-mer-seeded → we claim "alignment-scored", not "k-mer-independent".
- **nontata: drop milder under alignment** (Jaccard 0.108 → MMseqs2 0.064; RF corrected acc 0.82→0.864) → partly k-mer-specific, reinforcing partial status. FIFTH independent confirmation of ensembl-airtight / nontata-partial (with metric-robustness, chromosome-holdout, novel-only, P-rank1).

### T1.7 / R2.a1 imbalanced data — `exp_imbalance.py` → `results/exp_imbalance_{counts,panel}.csv` [RAN]
Part A: GB datasets curated-balanced (confirmed). Part B: synthetic prevalence stress (RF, π∈{0.5,0.2,0.1}), prevalence-aware metrics:
- **Under imbalance the leakage inflation is REVEALED by AUPRC/MCC/minority-recall but MASKED by accuracy.** ensembl π=0.2: AUPRC 0.622→0.477, MCC 0.422→0.182, minority-recall 0.258→0.070, while accuracy barely moves (0.844→0.808). nontata same pattern. → accuracy@0.5 is the wrong metric under imbalance; the appropriate metrics show the inflation even more starkly. Connects to the AUROC/threshold-free finding.
- **The homology-aware splitter PRESERVES the target prevalence** (realized pos_frac 0.5001/0.2007/0.1001 ≈ target) → G9 addressed: the whole-cluster assigner handles imbalance without skewing the minority class.

### S9 / R2.a3 injected-leakage 3-class — `exp_inject3class.py` → `results/exp_inject3class.csv` [RAN]
Clean 3-class human_ensembl_regulatory (20k subsample; full 289k pairwise intractable, mechanism is scale-independent). Inject near-dup train→test copies at fraction f, RF vs LR:
- **RF drop grows monotonically with f**: f=0 → +0.0004; f=0.1 → +0.118; f=0.2 → +0.179; f=0.4 → +0.258. LR (non-memorizer) stays ~flat (0→0.016→0.032→0.066). Corrected accuracy stable ~0.58 regardless of f (homology split removes the injected leakage).
- → Leakage inflation appears iff f>0, scales with f, is removed by the homology-aware split, and hits only the memorizer — **generalizes to the 3-class setting**. Answers R2.a3 multiclass by controlled construction (GB ships no naturally-leaky multiclass set).

### GB1 / GB4 / G11 genomics geometry — `exp_geometry.py` → `results/exp_gb1_containment.csv`, `exp_gb4_*.csv`, `exp_g11_cohesion.csv` [RAN]
- **GB1 length-cap + containment (updated with FULL-SCALE recompute, `results/full_scale_containment.csv`):** ensembl containment leakage **0.845** vs Jaccard 0.384 (length cap 0.0035 from the 2 bp min → Jaccard undercounts >half via length disparity) → leak fractions are LOWER BOUNDS; containment is the length-robust metric. **Full-scale containment flips demo_coding to BORDERLINE (0.131 > 0.1; clean under Jaccard 0.078)** — the length-robust metric catches leakage the length-blind one misses (and the 20k subsample masked it further at 0.043, consistent with the full-scale-audit thesis). The other four clean stay clean under both (ocr 0.068, worm 0.042, drosophila 0.033, cohn 0.005). Honest verdict: **2 clearly leaky, 1 borderline (containment-only), 4 clean**. This is a strength — demo_coding is the proof-by-example that both metrics should be reported.
- **GB4 covariate shift:** restricting the corrected test to GC-in-distribution sequences gives LOWER accuracy (nontata 0.787, ensembl 0.631) than the full corrected acc, not the inflated original → the RF drop is NOT a GC covariate-shift artifact; it is genuine leakage removal.
- **G11 single-linkage chaining (the mechanistic key to ensembl-vs-nontata):** ensembl clusters are **99.6% pairwise, largest-cluster cohesion 0.96** = clean discrete ~2× locus duplication (correction is a clean fix → RF demotion genuine). nontata's 420-member largest cluster has cohesion **0.083** = heavy transitive chaining (24% pairwise) that over-removes genuine promoter/gene-family signal → explains why nontata RF keeps a real edge on novel sequences and why its demotion is the partial/cautionary case. Directly answers GB5 (over-correction) and the single-linkage-chaining concern.

### T1.5 cluster-bootstrap FULL rigor — `exp_clusterboot_full.py` → `results/cluster_bootstrap_full.csv` [RAN]
Complete R3.3 answer. Leaky RF/LR with ICC + design effect + CRVE + combined-source CI (BOOT=10k):
- nontata RF: ICC **0.907** but deff **1.169** (mean cluster size ~1.17); sample CI [0.099,0.118], cluster CI [0.092,0.126], **combined-source CI (test-boot + 5-seed re-split variance) [0.088,0.129]** — all exclude 0. CRVE orig-acc CI [0.923,0.934] agrees.
- ensembl RF: ICC **0.9925**, deff **1.056**; combined-source CI [0.154,0.174] — excludes 0. ensembl LR combined CI [-0.010,0.011] — includes 0 (null).
- **All 5 clean + 3-class cluster-bootstrap deltas include 0** (RF and LR). 
- Key: ICC is high but deff is SMALL (1.06–1.17) because near-dups are test-to-TRAIN not test-to-test → cluster bootstrap barely widens CIs; combined-source + CRVE change NO verdict. R3.3 fully honored, overturns nothing.

### GB3 repeat/low-complexity characterization — `exp_repeat.py` → `results/exp_repeat.csv` [RAN]
dustmasker on leaked (≥0.9-sim) vs novel (<0.5) test sequences: leaked are NOT low-complexity-enriched (nontata leaked 0.033 < novel 0.106; ensembl leaked 0.053 ≈ novel 0.043; <3.3% mostly-masked). → the leakage is genuine **discrete locus duplication** (benchmark-construction artifact), NOT a TE-family / microsatellite artifact. With G11 (99.6% pairwise, cohesion 0.96 on ensembl), confirms removal is a legitimate fix, not over-correction of repeat biology (GB3/GB6 answered).

## ALL 13 EXPERIMENTS COMPLETE
Every Tier-1/2 experiment executed and recorded above. The core narrative: **ensembl is an airtight existence proof** (RF→LinearSVC flip holds across accuracy/AUROC/F1, chromosome-holdout, P(rank1)=1.0→0.0, alignment metric-independence, cluster-bootstrap, containment, canonical features, GC-control; mechanism = clean discrete locus duplication memorized by the default unregularized RF; a regularized CNN reproduces the drop). **nontata is a cautionary partial case** (demotion appears only under accuracy@0.5; fails threshold-free metrics, chromosome-holdout, alignment-mildness, P(rank1) no dominant winner; single-linkage chaining over-removes genuine promoter signal). Central claim honestly scoped to cleanly n=1.
