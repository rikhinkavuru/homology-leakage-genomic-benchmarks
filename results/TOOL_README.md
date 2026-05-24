# homology_split.py — homology-aware train/test splitter

A drop-in splitter for biological sequence datasets that prevents near-duplicate
(homologous) sequences from leaking across the train/test boundary — the failure mode
that inflates reported accuracy on several Genomic Benchmarks datasets (see the audit).

## Why

Random (or curated-but-unchecked) splits often place near-identical sequences in both
train and test. High-capacity models then memorize the near-duplicates, reporting
optimistic accuracy that does not generalize. This tool clusters near-duplicates and
keeps each cluster wholly on one side of the split.

## Dependencies

`numpy`, `scipy` only. CPU-only. No GPU, no network, no external services.

## Guarantee (checkable via `verify_split`)

- **Zero residual leakage:** no test sequence has Jaccard similarity > `threshold` to any
  train sequence (residual cross-split similarity > threshold == 0.0).
- Class balance and the train/test ratio (`test_frac`) are preserved.
- Deterministic given `seed`.

## Method

Each sequence -> its set of k-mers (default k=8). Pairs with exact k-mer Jaccard >
`threshold` are edges; connected components are near-duplicate clusters; whole clusters
are assigned per class to reach `test_frac`. Similarity is exact (not MinHash).

## Python API

```python
from homology_split import homology_aware_split, verify_split
train_idx, test_idx = homology_aware_split(sequences, labels,
                                           test_frac=0.25, threshold=0.7, seed=0, k=8)
X_train, X_test = X[train_idx], X[test_idx]          # homology-disjoint at Jaccard>0.7
print(verify_split(sequences, train_idx, test_idx))  # residual_leak_fraction == 0.0
```

## Command line

```bash
python homology_split.py --fasta seqs.fasta --labels labels.txt --out splits.json \
                         --test-frac 0.25 --threshold 0.7 --seed 0
```

- `--fasta`   : FASTA of sequences (DNA, ACGT; case-insensitive).
- `--labels`  : text file, one integer label per FASTA record, same order (optional).
- `--out`     : JSON with `train_idx`/`test_idx`, `train_ids`/`test_ids`, the parameters,
                and the `verification` block (including `residual_leak_fraction`).
- No `--fasta` -> runs the built-in synthetic self-test.

## Inputs / outputs

| input | output (JSON) |
|---|---|
| sequences (FASTA) + optional labels | `train_idx`, `test_idx` (0-based), `train_ids`, `test_ids`, `params`, `verification` |

## Runtime

Exact all-pairs Jaccard via sparse `M @ M.T` in batches. Validated end-to-end on the
full `human_enhancers_ensembl` (154,842 sequences) in a few minutes on a laptop CPU;
small datasets are sub-second. Memory scales with the binary sequence x k-mer matrix
(~`n_seq * avg_kmers` nonzeros).

## Validation

On the two leaky Genomic Benchmarks datasets, over 5 seeds: corrected RandomForest
(k=6) accuracy 0.8101 +/- 0.0055 (nonTATA promoters) and 0.6951 +/- 0.0019 (enhancers
Ensembl), with **residual leakage 0.000000 every seed**. See `splitter_validation.csv`.
