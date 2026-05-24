# Homology-leakage audit across the Genomic Benchmarks suite

Reuses the validated `run_audit.py` pipeline (identical featurization, models, exact-Jaccard similarity, whole-cluster re-split). Subsample cap N=20000 (stratified by split x class, seed=0); 3 re-split seeds [0, 1, 2]; model seed=0. Total runtime 923s.

Datasets larger than the cap are subsampled; for those the **leakage fraction is a lower bound** (random subsampling breaks up near-duplicate pairs), while the accuracy **deltas are unbiased within the subsample** and conservative.

## Master table (best original model per dataset)

| dataset | n_used (full) | best model | leak@0.7 | original | random | homology | Δ(hom) | Δ(rand) |
|---|---|---|---|---|---|---|---|---|
| human_nontata_promoters | 20000 (36131) | RF_k6 | 0.320 | 0.894 | 0.895 | 0.835±0.007 | **+0.059** | -0.002 |
| human_enhancers_cohn | 20000 (27791) | LR_k6 | 0.001 | 0.736 | 0.733 | 0.741±0.002 | **-0.004** | +0.003 |
| human_enhancers_ensembl | 20000 (154842) | LR_k6 | 0.056 | 0.760 | 0.752 | 0.750±0.010 | **+0.011** | +0.009 |
| human_ocr_ensembl | 20000 (174756) | LR_k6 | 0.002 | 0.677 | 0.676 | 0.673±0.002 | **+0.004** | +0.001 |
| demo_human_or_worm | 20000 (100000) | LR_k6 | 0.003 | 0.937 | 0.944 | 0.941±0.000 | **-0.004** | -0.006 |
| demo_coding_vs_intergenomic_seqs | 20000 (100000) | LR_k6 | 0.022 | 0.888 | 0.890 | 0.891±0.007 | **-0.003** | -0.003 |
| drosophila_enhancers_stark | 6914 (6914) | LR_k6 | 0.016 | 0.718 | 0.693 | 0.706±0.009 | **+0.012** | +0.025 |

**Cross-dataset:** homology-aware accuracy drop (best model) mean **0.011** (range -0.004..+0.059); random re-split drop mean **0.004** (range -0.006..+0.025).

## Capacity scaling (accuracy drop by model, homology re-split)

| dataset | LR k4 | LR k6 | RF k4 | RF k6 | monotone? |
|---|---|---|---|---|---|
| human_nontata_promoters | +0.004 | +0.027 | +0.036 | +0.059 | yes |
| human_enhancers_cohn | -0.003 | -0.004 | -0.011 | -0.005 | no |
| human_enhancers_ensembl | -0.000 | +0.011 | +0.013 | +0.020 | yes |
| human_ocr_ensembl | +0.001 | +0.004 | +0.018 | +0.010 | no |
| demo_human_or_worm | -0.009 | -0.004 | -0.006 | -0.005 | no |
| demo_coding_vs_intergenomic_seqs | +0.001 | -0.003 | -0.001 | -0.002 | no |
| drosophila_enhancers_stark | +0.015 | +0.012 | +0.005 | +0.023 | no |

For contrast, the **random re-split** deltas (negative control):

| dataset | LR k4 | LR k6 | RF k4 | RF k6 |
|---|---|---|---|---|
| human_nontata_promoters | +0.004 | +0.006 | +0.000 | -0.002 |
| human_enhancers_cohn | +0.001 | +0.003 | -0.003 | +0.000 |
| human_enhancers_ensembl | +0.009 | +0.009 | +0.010 | +0.013 |
| human_ocr_ensembl | +0.000 | +0.001 | +0.008 | +0.000 |
| demo_human_or_worm | -0.010 | -0.006 | -0.004 | -0.002 |
| demo_coding_vs_intergenomic_seqs | -0.001 | -0.003 | +0.001 | -0.005 |
| drosophila_enhancers_stark | +0.017 | +0.025 | +0.011 | +0.034 |
