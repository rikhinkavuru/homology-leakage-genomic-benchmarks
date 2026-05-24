# Sanity check: original-split numbers vs. published baselines

This file compares our **original-split** classical-model results against the
deep-learning baselines published for the same dataset. The published numbers
below are quoted from the literature (NOT produced by our run); every number in
`results.md` and the CSVs comes from our actual run.

## Published baseline (Gresova et al. 2023)

*Genomic benchmarks: a collection of datasets for genomic sequence
classification*, BMC Genomic Data 24:25, Table 2 (CNN baseline):

| model | test accuracy | F1 |
|---|---|---|
| PyTorch CNN | 84.6% | 83.7 |
| TensorFlow CNN | 86.5% | 84.4 |

The paper's Table 1 lists this dataset as **36,131 sequences, 2 classes, class
ratio 1.2, median length 251 bp** -- which matches exactly what we loaded
(36,131 sequences; class balance 0.544 positive ~ ratio 1.19; all sequences
251 bp). This confirms we are auditing the same data.

## Our original-split results (this run)

| model | k | accuracy | F1 |
|---|---|---|---|
| LogisticRegression | 4 | 82.7% | 0.833 |
| LogisticRegression | 6 | 87.0% | 0.875 |
| RandomForest | 4 | 89.1% | 0.891 |
| RandomForest | 6 | **93.2%** | 0.934 |

## Reading

- Our LR sits right in the **published CNN ballpark** (82.7-87.0% vs 84.6-86.5%),
  so the pipeline is behaving sensibly -- classical k-mer models are expected to
  land slightly below/around the CNN, and they do.
- Our **RandomForest (k=6) at 93.2% exceeds the published CNN** by ~7-9 points.
  A bag-of-k-mers RF out-scoring a tuned CNN is implausible as genuine
  generalization; it is exactly the signature of a model exploiting the
  near-duplicate train/test leakage we quantify in `results.md`
  (40.6% of test sequences have a >0.7-Jaccard twin in train).
- Crucially, the **published CNN numbers were themselves measured on this same
  leaky split**, so they are inflated by the same homology leakage. Our
  homology-aware re-split drops the classical models to ~81-83% (below the
  published CNN), which is the honest difficulty of the task once near-duplicate
  leakage is removed.

Source: Gresova, K., Martinek, V., Cechak, D., Simecek, P., Alexiou, P. (2023).
*Genomic benchmarks: a collection of datasets for genomic sequence
classification.* BMC Genomic Data 24:25.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10150520/
