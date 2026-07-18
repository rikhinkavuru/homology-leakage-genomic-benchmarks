# Tier-1 pre-registration and frozen predictions

**Registered:** 2026-07-18T20:07:54Z
**Repository state at registration:** `6b0360bd920ab26c62d405ab374ca6bea119b3a3`
**Scope:** EXECUTION_PLAN.md Tier 1 — Deepener #2 (inversion law), Deepener #3
(construction rule), Deepener #6-A (`certify`), Deepener #1 phase A (cross-suite census).

---

## 0. Honesty declaration — what is and is not pre-registered

This project has one prior pre-registration (`results/deep_preregistration.md`). This
document must not overstate its own status, so the distinction is drawn explicitly:

| Claim class | Status | Why |
|---|---|---|
| §2 inversion law validated on `exp_regpath.csv`, `exp_inject3class.csv`, `exp_alignment.csv`, `exp_deep_cnn.csv` | **RETROSPECTIVE** | Those CSVs were frozen in a prior session. The law was derived after they existed. It is *out-of-sample* with respect to its own inputs (it never sees the corrected split) but it is **not** temporally pre-registered. |
| §3 construction rule fitted on the 8 GB datasets | **RETROSPECTIVE**, zero free parameters | Threshold reused from `report_card.py:49`, not tuned here. |
| §1-A cross-suite predictions in `exp_crosssuite_census.py:PREREGISTERED` | **WEAK** | Written into the source before execution, but authored in the same session. Treat as a stated expectation, not an independent registration. |
| §4 forward predictions below | **STRONG / BINDING** | Made before the corresponding data exists locally. These are the ones a reviewer should hold us to. |

Anyone re-running this repo can verify the retrospective claims byte-for-byte; only §4
carries genuine predictive risk. We would rather label three quarters of this document
"retrospective" than claim a registration we did not earn.

**Commit anchoring.** An uncommitted file is not a registration. This document is
committed to the repository *before* the Tier-1 result CSVs and before
`results/TIER1_FINDINGS.md`, so `git log --follow` gives it a tamper-evident position in
history that provably precedes the results it predicts. The §4 forward predictions are
the ones that anchoring protects; §1–§3 remain retrospective regardless of commit order,
and no commit hash can change that.

**Outcome of the §1-A expectations (recorded here rather than quietly dropped).** The
suite-level call `NT-original → LEAKY` was **wrong as stated**: only 4 of 13 tasks are
leaky and 6 are clean, including all three promoter tasks. Leakage is a property of
particular task constructions, not of suites. The `NT-revised → CLEAN` call held (13/13).

---

## 1. The inversion law (frozen statement)

Stratify a leaky test set by maximum similarity to train into a leaked stratum
(fraction **φ**) and a novel stratum. For model *M* with leaked-stratum accuracy
*h_M* and novel-stratum accuracy *n_M*, define the graded memorization gap
*g_M = h_M − n_M*. Then

```
a_M      = n_M + φ·g_M                     (two-stratum identity)
Δ_M      = a_M − a_M^corr = φ·g_M          (split-induced drop)
```

For an ordered pair (A = as-shipped leader, B = challenger), with
**δ = n_B − n_A** and **Δg = g_A − g_B**:

```
the benchmark reports the WRONG winner   iff   0 < δ < φ·Δg
critical leak fraction (breakdown point) φ* = δ / Δg
```

**Frozen anchor values** (from `graded_performance.csv`, `exp_rankings.csv`):

| dataset | pair | δ | Δg | φ* | measured φ | predicted |
|---|---|---|---|---|---|---|
| `human_enhancers_ensembl` | RF → LinearSVC | +0.0122 | 0.1929 | **≈0.063** | 0.3802 | **inverts** (φ ≫ φ*) |
| `human_nontata_promoters` | RF → any | all δ ≤ 0 | — | — | 0.2255 | **no inversion on the novel stratum** |

φ* is quoted to three decimals deliberately: `inversion_law_pairs.csv` reports 0.0632
(computed from the 4-decimal δ and Δg printed in `graded_performance.csv`) while
`inversion_law_bootstrap.csv` reports 0.0633 (recomputed at full precision from the
per-example correctness vectors). The difference is input rounding, not a disagreement,
and it is immaterial against a measured φ of 0.380.

**Registered decision rule.** A pair is CLEAN-inverting iff the cluster-bootstrap CI
excludes 0 for both δ and the inversion margin *m_inv = φ·Δg − δ*; FRAGILE iff the δ CI
straddles 0. Blocks are within-test 8-mer-Jaccard>0.7 components
(`cluster_bootstrap.test_internal_clusters`), 10,000 replicates, seed 20240524.

**The observable form carries no predictive content.** `0 < μ < Δ_A − Δ_B` reduces
algebraically to `a_B^corr > a_A^corr`, i.e. it *is* the definition of a corrected-split
inversion. It is computed only as an arithmetic-consistency check on the frozen tables
and must pass 12/12 by algebra. All predictive content lives in the φ·g form, whose
inputs are measured on the as-shipped split alone.

---

## 2. The construction rule (frozen statement)

**Cause.** Leakage is produced when *some class* of a benchmark is an un-merged union of
regions that tiles recurring loci — not by "multi-cell-type data" per se.

**Statistic.** `xsplit_ge50pct_max` = the maximum over classes of the share of TEST
intervals overlapping any TRAIN interval at ≥50% reciprocal overlap.

**Threshold.** 0.1 — reused verbatim from `report_card.py:49`. **Zero free parameters.**

Two corrections were forced by the data and are recorded here because they contradict
the original plan text:

1. **Redundancy is not always in the positive class.** `human_nontata_promoters` carries
   its redundancy entirely in the **negative** class (R_self 0.9605 vs 0.0474 positive).
   The statistic must therefore be a max over classes.
2. **"≥1 bp overlap" over-calls.** `drosophila_enhancers_stark` tiles at 2142 bp median
   with only bp-scale touches (xsplit ≥1 bp = 0.406, but ≥50% reciprocal = 0.036). The
   statistic must use reciprocal overlap.

**Three distinct mechanisms, now separated:**

| mechanism | signature | example |
|---|---|---|
| exact coordinate duplication | `xsplit_exact` high | `human_enhancers_ensembl`: 77,421 positives on 40,934 unique coordinates; 94.3% of positive rows sit on a duplicated coordinate; **75.5% of test positives have their exact coordinate in train** |
| contiguous window tiling | `xsplit_ge50pct` high, `xsplit_exact` = 0 | `human_nontata_promoters` negatives: cross-split overlap 0.992 at ≥50% reciprocal (self-overlap R_self = 0.961), and **zero** exact duplicates |
| paralog/repeat homology | coordinates clean, sequence leaky | `demo_coding_vs_intergenomic_seqs`: coordinate 0.0064, containment-borderline at full scale. Note its `coding_seqs` class is keyed by ENST transcript accession (one region per row), so it is **not measurable** in coordinate space at all — the rule has no coverage there |

---

## 3. Registered refutation conditions (Tier 1)

The Tier-1 claims are **refuted** if any of the following holds:

- **R1** The φ·g identity fails to reconstruct as-shipped accuracy to within 0.01 on
  `human_enhancers_ensembl` (the dataset with <2% mid-bin mass).
- **R2** The law predicts a CLEAN inversion on any dataset the paper calls clean, or
  fails to predict the `human_enhancers_ensembl` RF→LinearSVC inversion.
- **R3** Predicted RF as-shipped rank disagrees with the observed `rf_rank_orig` on more
  than 10% of the 22 regularization-path cells.
- **R4** The injection-axis prediction, with *h* fitted at f = 0.1 only, misses the
  held-out f = 0.2 / 0.4 accuracies by more than 0.01.
- **R5** The coordinate rule at threshold 0.1 misclassifies any of the 8 GB datasets,
  counting `borderline` as its own level rather than folding it into `clean`.
- **R6** `certify --self-validate` fails to reproduce the published 8-dataset verdicts.

*Outcome: R1–R4 and R6 did not fire. **R5 DID fire** under the strict reading:
`demo_coding_vs_intergenomic_seqs` is coordinate-CLEAN but sequence-`borderline`, so the
rule is 7/8 strict and 8/8 only when `borderline` is folded into `clean`. Both numbers
are reported in `results/TIER1_FINDINGS.md`; the miss is the paralog-homology mechanism,
which has no coordinate signature by construction.*

---

## 4. BINDING forward predictions (data not yet analysed locally)

These are the predictions that carry real risk. They are recorded before the
corresponding experiments are run.

### 4.1 Cross-suite leak census (Deepener #1, remaining suites)

| suite / task family | predicted verdict | basis |
|---|---|---|
| GUE core-promoter (70 bp), TF-binding (100 bp), promoter (300 bp), human | **LEAKY** | short fixed-length human regulatory windows — the regime the construction rule flags |
| GUE yeast EMP, virus CVC (multi-species) | **CLEAN** | different genomes, not window-tiled |
| GLRB variant-effect (ref/alt SNP pairs) | **LEAKY**, and *test-to-test* rather than test-to-train | ref/alt pairs differ by one nucleotide and both sit in the test set |
| GLRB `regulatory_element_enhancer` | **LEAKY** | 200 bp windows labelled by ≥50% overlap with annotated cis-REs |
| BEND | **CLEAN** | 80%-identity / chromosome-aware by construction |
| DeepSTARR | **LEAKY** | genome-tiled by construction |

**Design-effect prediction (registered, and the sharpest test here).** The cluster/block
bootstrap will matter *quantitatively* only where near-duplicates are test-to-test. We
predict design effects near the paper's observed 1.06–1.17 for test-to-train regimes
(GUE, NT) but **substantially larger** (> 1.5) for GLRB variant-effect and DeepSTARR. If
the design effect is ~1 everywhere, the cluster bootstrap is not load-bearing and we will
say so.

### 4.2 Inversion law, applied out of suite

For any newly censused dataset we will publish (φ, g, δ) **before** computing its
corrected split, and predict inversion iff φ > φ* = δ/Δg. Registered expectations:

- Any dataset with φ > 0.2 **and** a memorizer/non-memorizer pair with Δg > 0.15 and
  small positive δ will inverta.
- **No dataset with φ < 0.05 will show a CLEAN inversion.** This is the strong,
  falsifiable half: a single clean-dataset CLEAN inversion refutes the law.

### 4.3 Honest-null branch

If no second suite yields a metric-robust inversion, the contribution is the multi-suite
report card plus the two laws, and the paper retains its scoped `n = 1` ranking claim —
now externally stress-tested rather than merely asserted. Either outcome is reportable;
the pre-registration is what makes both publishable.

---

## 5. Estimator declarations (one per quantity, per §5.3 of EXECUTION_PLAN.md)

| quantity | estimator | source |
|---|---|---|
| decision rule | sklearn `.predict()` (argmax) everywhere | `expkit.py`, `reconciled_numbers.md` §5 |
| leak fraction | max 8-mer Jaccard / containment of each TEST sequence to any TRAIN sequence, as-shipped split | `homology_split.max_jaccard_to_reference` |
| φ in the law | share of test in the `[0.9,1.0]` similarity bin, using the **same binning** that defines *n* and *g* | `run_graded.py:BINS` |
| bootstrap blocks | within-test 8-mer-Jaccard>0.7 connected components | `cluster_bootstrap.test_internal_clusters` |
| leak verdict cut | 0.1, on Jaccard first then full-scale containment | `report_card.py:49` |

Note φ = 0.2255 here uses the graded-bin count (2037/9034); `leakage_fraction.csv`
reports 0.22537 (2036/9034) from a strict `> 0.9` comparison. The one-sequence
difference is a `>=` vs `>` boundary and is immaterial, but the binned value is used
throughout the law so that φ, *n* and *g* are mutually consistent.
