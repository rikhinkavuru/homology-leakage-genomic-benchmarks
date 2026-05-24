# Reproducing the homology-leakage audit

CPU-only, deterministic. Python 3.13. ~16 GB RAM recommended (the full-scale
`human_enhancers_ensembl` step peaks around 6-7 GB). No GPU, no deep learning,
no network except the one-time dataset download.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r results/requirements.txt
# (equivalently: pip install genomic-benchmarks scikit-learn pandas numpy scipy matplotlib)
```

Datasets are fetched by `genomic_benchmarks.loc2seq.download_dataset` on first use
and cached in `~/.genomic_benchmarks/` (a few minutes total, one-time).

## Pipeline (run from the project root; each script writes into `results/`)

| # | command | what it does | ~runtime |
|---|---|---|---|
| 1 | `python run_audit.py` | single-dataset headline (human_nontata_promoters, full 36 k): original vs homology@0.7, LR+RF, k=4/6 | ~2 min |
| 2 | `python run_suite.py` | 7 binary + smoke + 3-class, 20 k subsample: original, **random control**, threshold sweep {0.5,0.7,0.9}, drop-largest | ~15 min |
| 3 | `python measure_leakage_full.py` | **full-scale** test->train 8-mer Jaccard leakage per dataset | ~15 min |
| 4 | `python run_fullscale.py` | full-scale re-split (original / random / homology@0.7) for the leaky datasets | ~40 min |
| 5 | `python run_robustness_full.py` | full-scale threshold sweep + drop-largest for **both** leaky datasets | ~60 min |
| 6 | `python finalize.py` | consolidate -> `summary_final.csv`, `capacity_scaling_final.csv`, `results_FINAL.md` | <5 s |
| 7 | `python make_paper_figures.py` | the two publication figures (PNG/SVG/PDF) + backing CSVs | <5 s |
| 8 | `python run_extended_models.py` | **Part A**: LR, LinearSVC, RF, HGB on the frozen splits (full leaky / 20k clean), k=4/6, 3 seeds | ~50 min |
| 9 | `python ranking_inversion.py` | **Part B**: rank models per split, Kendall tau, stable inversions | <5 s |
| 10 | `python validate_splitter.py` | **Part C**: validate `homology_split.py` on the 2 leaky datasets, 5 seeds | ~10 min |
| 11 | `python make_part_b_figures.py` | extended capacity curve + ranking-inversion slopegraph (PNG/SVG/PDF) | <5 s |
| 12 | `python run_graded.py` | **Part 3**: homology-graded per-model accuracy by similarity bin (2 leaky + 2 clean), k=6, original split | ~12 min |
| 13 | `python report_card.py` | **Part 1 artifact**: per-dataset leakage report card (table + figure) | <5 s |
| 14 | `python make_graded_figure.py` | graded-performance figure (PNG/SVG/PDF) | <5 s |
| 15 | `python check1_label_concordance.py` | **Check 1**: nearest-train-neighbour label concordance for near-duplicates (leaky + clean) | ~6 min |
| 16 | `python check2_novelonly_ranking.py` | **Check 2**: novel-only ranking + 3-way ranking comparison (no refit) | <5 s |
| 17 | `python paper_numbers.py` | `results/PAPER_NUMBERS.md` (single consolidated source, with provenance) | <5 s |

Steps 1-5, 8, 10, 12, 15 do model fitting / similarity; 6-7, 9, 11, 13-14, 16-17 are pure aggregation.

**Splitter CLI (Part 2):** `python homology_split.py --fasta seqs.fasta --labels labels.txt --out splits.json` (see `results/TOOL_README.md`); no `--fasta` runs the self-test.

**Standalone tool:** `homology_split.py` is a self-contained, importable module
(`from homology_split import homology_aware_split`); run `python homology_split.py`
for its built-in self-test (zero residual leakage on synthetic near-duplicates).

## Headline reproduction (quickest check)

`python run_audit.py` reproduces the canonical result:

- human_nontata_promoters, **RF k6**: original **0.932** -> homology-aware **0.811** (**-12.1 pts**)
- LR k4: 0.827 -> 0.818 (-0.9 pts)  (monotone capacity scaling in between)

Full numbers land in `results/results.md`. `run_robustness_full.py` independently
re-loads and re-fits the full dataset from scratch and reproduces the same
nontata full-scale numbers, so the headline is reproduced by two independent paths.

## Determinism

Fixed numpy seed (0) + sklearn `random_state=0`; a fixed ACGT k-mer vocabulary
(featurization is data-independent); and an **exact, non-stochastic** Jaccard
similarity (no MinHash). All seeds are listed in `results/seeds.txt`.
