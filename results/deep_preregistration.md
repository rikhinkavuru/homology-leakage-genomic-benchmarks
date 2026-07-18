# Pre-registration — deep-model dose-response leakage experiment (task T1.2)

**Written BEFORE any model is trained.** Addresses reviewers R1.1a / R2.a3 / R3.1(i),
who ask for a trained deep net (the classical-only suite is called outdated). This
document fixes the claim, the grid, the metrics, and — critically — the falsification
condition, so the verdict cannot be reverse-engineered from the result.

## Scope and confound control

We train a **from-scratch 1D residual CNN** (Basset/DeepSEA-style). We deliberately do
**NOT** use DNABERT-2 / HyenaDNA / any HG38-pretrained model here: those are pretrained
on the exact human reference regions that appear in these test sets, so any "deep nets
generalize / deep nets leak" reading would be confounded by pretraining contamination.
That comparison is **explicitly out of scope** for this experiment. The from-scratch CNN
sees each dataset's training split and nothing else, isolating the near-duplicate-leakage
mechanism from pretraining contamination.

## The narrowed claim (what we assert)

> Near-duplicate leakage inflates **high-capacity, insufficiently-regularized** learners
> — those exhibiting a large **graded memorization gap** — and demotes them under a
> homology-aware split. **Well-regularized** learners show a **null drop**.

This is a mechanism claim about capacity × regularization, NOT the strong universal claim
that *any* expressive learner is inflated by leakage.

## Refutation condition (pre-committed, binding)

> If the **standard-practice regularized CNN** (dropout=0.3, weight_decay=1e-4)'s
> homology-drop **95% cluster-bootstrap CI INCLUDES 0 on BOTH leaky datasets**, then the
> strong "any expressive learner is inflated" reading is **REFUTED**. We will report that
> as a **limitation** of the deep-net evidence — NOT relabel the outcome, move the
> reference cell, or redefine "standard practice" after the fact.

We report the verdict honestly against this condition regardless of which way it falls.
The mechanism claim (graded gap grows as regularization drops; the over-trained net
behaves like the over-capacity RF) is evaluated separately and can hold even if the
standard-practice cell shows a null drop.

## Pre-declared design

**Datasets (load-bearing):** the two LEAKY datasets, full scale from cache —
`human_nontata_promoters` (n=36,131; fixed length 251) and
`human_enhancers_ensembl` (n=154,842; variable length → pad/crop to 269).
Clean-dataset controls (`demo_human_or_worm`, `demo_coding`) would be added only if their
datacache pkls existed; they do not, so no clean controls are run.

**Encoding:** one-hot A,C,G,T → 4 channels; N/other → all-zero column. Fixed length per
dataset (nontata 251; ensembl 269), center-crop / zero-pad.

**Model:** 1D residual CNN — 3 conv blocks (Conv1d→BatchNorm→ReLU, residual where shapes
allow, widths ~64/128/128, kernels 9/5/3, maxpool between blocks), global average pool,
one dense hidden layer (128, ReLU, dropout), linear → 1 logit, binary cross-entropy, Adam,
batch 256, max 30 epochs, early stopping (patience 5) on a fixed 10%-of-train internal
validation slice. Deterministic: `torch.manual_seed(0)`, device MPS (CPU fallback).

**Dose-response grid (pre-declared):** dropout ∈ {0.0, 0.3, 0.6} × weight_decay ∈
{0.0, 1e-4, 1e-3} = 9 configs.
- **Reference cell = standard-practice regularized:** (dropout=0.3, weight_decay=1e-4).
- **Manipulation check (labelled):** (dropout=0.0, weight_decay=0.0) trained to
  convergence (early stopping disabled, full epoch budget) — predicted to memorize
  heavily (large graded gap, large drop), mirroring the unregularized RandomForest.

If the full 9-config grid is too slow, we may drop to dropout{0.0,0.3,0.6} ×
weight_decay{0,1e-4} (6 configs) but will KEEP both the standard-practice and the
unregularized cells and will state that the grid was reduced.

**Splits:** for each (dataset, config) we train twice — on the ORIGINAL split (`otr→ote`)
and on the homology-CORRECTED split (`E.assign(E.clusters(seqs,0.7,8), y, 0, test_frac)`),
reusing the frozen `expkit` primitives. We store the per-test-example correctness vector
(`pred>0.5 == y`) for each.

**Metrics per (dataset, config):**
- `acc_orig`, `acc_corr`, `drop = acc_orig − acc_corr`.
- Drop 95% CI via BOTH `E.sample_boot` (sample-wise) and `E.cluster_boot`
  (cluster/block bootstrap; within-test 8-mer-Jaccard>0.7 components via `E.clusters` on
  each test set). **The cluster-bootstrap CI is the one that binds the refutation
  condition** (it respects near-duplicate correlation among test sequences).
- Graded memorization gap = `mean(correct[sim≥0.9]) − mean(correct[sim<0.5])` on the
  ORIGINAL-trained correctness vector, with `sim = E.max_sim_to_train(seqs, otr, ote, 8)`.
- Novel-only accuracy = `correct[sim<0.5].mean()` (original-trained).

**Predicted directional pattern (for the mechanism test):**
1. Graded memorization gap increases monotonically as regularization decreases (dropout↓,
   weight_decay↓).
2. The unregularized/over-trained CNN shows a large gap and a large, CI-excludes-0 drop,
   like the over-capacity RF.
3. The standard-practice cell's drop is smaller; whether its cluster-bootstrap CI excludes
   0 is the open, pre-committed test above.

**Outputs:** `results/exp_deep_cnn.csv` (one row per dataset×config×any manipulation
variant) and `results/exp_deep_cnn.md` (verdict against the refutation condition).

**Constraints honored:** reuse `expkit` for splits/similarity/bootstrap; deterministic
seeds; no frozen script or existing result file is modified; only new files are added.
