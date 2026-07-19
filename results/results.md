# Homology-leakage audit: `human_nontata_promoters`

Genomic Benchmarks DNA sequence-classification dataset (binary: non-TATA promoter vs. background). CPU-only, pure numpy/pandas/scikit-learn/scipy. Seeds: model=0, cluster splits=[0, 1, 2]. Total runtime 98s.

## 1. Dataset

- Sequences: **36131** total - train **27097**, test **9034** (original split = 75.0% / 25.0%).
- Class balance (fraction positive): train 0.544, test 0.544.
- Sequence length: mode **251 bp** (min 251, max 251, mean 251.0). Sequences are real ACGT strings.

## 2. Train/test homology (original split)

For each of the 9034 test sequences we computed the maximum exact 8-mer Jaccard similarity to any training sequence.

| max-Jaccard threshold | fraction of test with a train near-duplicate |
|---|---|
| > 0.5 | **0.4497** (4063/9034) |
| > 0.7 | **0.4064** (3671/9034) |
| > 0.9 | **0.2254** (2036/9034) |

Median test->train max-similarity = 0.088; 99th percentile = 0.992. See `similarity_hist.png`.

## 3. Homology-aware re-split

Sequences (train+test pooled) were connected by edges where 8-mer Jaccard > 0.7; connected components are clusters. **23312** clusters from 36131 sequences; 20501 singletons, largest cluster = 420. **12819 sequences (35.5%) are redundant** (have a near-duplicate elsewhere in the dataset). Whole clusters were assigned to train or test (never split), targeting 25% test while preserving class balance, for 3 random seeds.

## 4. Headline result

| split | feat k | model | accuracy | AUROC | F1 |
|---|---|---|---|---|---|
| original | 4 | LR | 0.8268 | 0.8974 | 0.8329 |
| homology-aware | 4 | LR | 0.8183 +/- 0.0055 | 0.8884 +/- 0.0064 | 0.8244 +/- 0.0047 |
| original | 4 | RF | 0.8907 | 0.9811 | 0.8913 |
| homology-aware | 4 | RF | 0.8065 +/- 0.0034 | 0.8952 +/- 0.0050 | 0.8250 +/- 0.0028 |
| original | 6 | LR | 0.8702 | 0.9381 | 0.8747 |
| homology-aware | 6 | LR | 0.8300 +/- 0.0098 | 0.9073 +/- 0.0066 | 0.8422 +/- 0.0076 |
| original | 6 | RF | 0.9317 | 0.9938 | 0.9342 |
| homology-aware | 6 | RF | 0.8117 +/- 0.0053 | 0.9211 +/- 0.0049 | 0.8401 +/- 0.0038 |

**Deltas (original - homology-aware):**

| feat k | model | d-accuracy | d-AUROC | d-F1 |
|---|---|---|---|---|
| 4 | LR | +0.0085 | +0.0089 | +0.0084 |
| 4 | RF | +0.0842 | +0.0859 | +0.0662 |
| 6 | LR | +0.0402 | +0.0308 | +0.0326 |
| 6 | RF | +0.1200 | +0.0727 | +0.0941 |

### Poster headline

> After a homology-aware re-split, test accuracy on **human_nontata_promoters** (LR, k=6) dropped by **4.0 points** (from **87.0%** to **83.0%**), indicating the standard split overstates generalization.
