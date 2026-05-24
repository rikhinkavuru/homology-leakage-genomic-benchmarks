# Homology-leakage audit — final synthesis (Genomic Benchmarks suite)

Leakage is measured on the **full** dataset (test→train exact 8-mer Jaccard). The accuracy drop is measured at **full scale for the leaky datasets** (nontata, enhancers_ensembl, coding_vs_intergenomic) and on a 20k stratified subsample for the genuinely-clean datasets (where full-scale leakage ≈ 0, so the subsample drop is unconfounded). Pipeline identical across all: k-mer counts (k=4/6), LogisticRegression + RandomForest(150), exact 8-mer Jaccard, whole-cluster re-split; 3 re-split seeds; model seed 0.

## Master table (best original model per dataset)

| dataset | n | leak@0.7 (full) | best model | original | random | homology | Δ(hom) | Δ(rand) | drop scale |
|---|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | 36131 | 0.406 | RF_k6 | 0.932 | 0.932 | 0.810±0.006 | **+0.121** | -0.001 | full |
| human_enhancers_ensembl | 154842 | 0.384 | RF_k6 | 0.856 | 0.858 | 0.700±0.002 | **+0.156** | -0.002 | full |
| demo_coding_vs_intergenomic_seqs | 100000 | 0.078 | LR_k6 | 0.900 | 0.903 | 0.901±0.001 | **-0.001** | -0.002 | full |
| human_enhancers_cohn | 27791 | 0.001 | LR_k6 | 0.736 | 0.733 | 0.741±0.002 | **-0.004** | +0.003 | subsample |
| human_ocr_ensembl | 174756 | 0.010 | LR_k6 | 0.677 | 0.676 | 0.673±0.002 | **+0.004** | +0.001 | subsample |
| demo_human_or_worm | 100000 | 0.012 | LR_k6 | 0.937 | 0.944 | 0.941±0.000 | **-0.004** | -0.006 | subsample |
| drosophila_enhancers_stark | 6914 | 0.016 | LR_k6 | 0.718 | 0.693 | 0.706±0.009 | **+0.012** | +0.025 | subsample |

## Capacity scaling (homology-aware accuracy drop by model)

| dataset | leak@0.7 | LR k4 | LR k6 | RF k4 | RF k6 | monotone | scale |
|---|---|---|---|---|---|---|---|
| human_nontata_promoters | 0.406 | +0.009 | +0.040 | +0.084 | +0.121 | yes | full |
| human_enhancers_ensembl | 0.384 | -0.005 | +0.003 | +0.153 | +0.156 | yes | full |
| demo_coding_vs_intergenomic_seqs | 0.078 | -0.001 | -0.001 | +0.007 | +0.003 | no | full |
| human_enhancers_cohn | 0.001 | -0.003 | -0.004 | -0.011 | -0.005 | no | subsample |
| human_ocr_ensembl | 0.010 | +0.001 | +0.004 | +0.018 | +0.010 | no | subsample |
| demo_human_or_worm | 0.012 | -0.009 | -0.004 | -0.006 | -0.005 | no | subsample |
| drosophila_enhancers_stark | 0.016 | +0.015 | +0.012 | +0.005 | +0.023 | no | subsample |

Negative-control (random re-split) deltas, same best-model column, are all within noise of zero — see `capacity_scaling_final.csv` (`rand_delta_*`).

## Cross-dataset summary

- **Leaky datasets (leak@0.7 > 0.1): 2** — human_nontata_promoters, human_enhancers_ensembl.
  - best-model homology drop: mean **0.139** (range +0.121..+0.156); random-control drop mean -0.001.
- **Clean datasets: 5** — demo_coding_vs_intergenomic_seqs, human_enhancers_cohn, human_ocr_ensembl, demo_human_or_worm, drosophila_enhancers_stark.
  - best-model homology drop: mean **+0.001** (range -0.004..+0.012) — i.e. ~0.

**Headline:** the homology-aware accuracy drop is large only where homology leakage exists, scales with model capacity (RF ≫ LR), and vanishes under a random re-split of the same data — establishing that the inflation is caused by near-duplicate train/test leakage, not by re-splitting per se.
