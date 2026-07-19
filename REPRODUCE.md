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

Datasets are fetched by `genomic_benchmarks.loc2seq.download_dataset`. Under the pinned
`genomic_benchmarks==1.0.0`, `download_dataset` re-extracts the dataset on every call, so
run `python -m audit.tools.prefetch` **once** to build a stable local `datacache/` (pickled
sequences) that every experiment then reads — this avoids repeated downloads/re-extraction
and the races they cause. Raw sequences are cached under `~/.genomic_benchmarks/`.

## Pipeline (run from the project root as modules; each script writes into `results/`)

Every script is a package module — invoke it with `python -m audit.<subpackage>.<module>`
from the repo root (see [`ARCHITECTURE.md`](ARCHITECTURE.md)).

| # | command | what it does | ~runtime |
|---|---|---|---|
| 1 | `python -m audit.pipeline.run_audit` | single-dataset headline (human_nontata_promoters, full 36 k): original vs homology@0.7, LR+RF, k=4/6 | ~2 min |
| 2 | `python -m audit.pipeline.run_suite` | 7 binary + smoke + 3-class, 20 k subsample: original, **random control**, threshold sweep {0.5,0.7,0.9}, drop-largest | ~15 min |
| 3 | `python -m audit.pipeline.measure_leakage_full` | **full-scale** test->train 8-mer Jaccard leakage per dataset | ~15 min |
| 4 | `python -m audit.pipeline.run_fullscale` | full-scale re-split (original / random / homology@0.7) for the leaky datasets | ~40 min |
| 5 | `python -m audit.pipeline.run_robustness_full` | full-scale threshold sweep + drop-largest for **both** leaky datasets | ~60 min |
| 6 | `python -m audit.pipeline.finalize` | consolidate -> `summary_final.csv`, `capacity_scaling_final.csv`, `results_FINAL.md` | <5 s |
| 7 | `python -m audit.figures.make_paper_figures` | the two publication figures (PNG/SVG/PDF) + backing CSVs | <5 s |
| 8 | `python -m audit.pipeline.run_extended_models` | **Part A**: LR, LinearSVC, RF, HGB on the frozen splits (full leaky / 20k clean), k=4/6, 3 seeds | ~50 min |
| 9 | `python -m audit.pipeline.ranking_inversion` | **Part B**: rank models per split, Kendall tau, stable inversions | <5 s |
| 10 | `python -m audit.pipeline.validate_splitter` | **Part C**: validate `homology_split.py` on the 2 leaky datasets, 5 seeds | ~10 min |
| 11 | `python -m audit.figures.make_part_b_figures` | extended capacity curve + ranking-inversion slopegraph (PNG/SVG/PDF) | <5 s |
| 12 | `python -m audit.pipeline.run_graded` | **Part 3**: homology-graded per-model accuracy by similarity bin (2 leaky + 2 clean), k=6, original split | ~12 min |
| 13 | `python -m audit.pipeline.report_card` | **Part 1 artifact**: per-dataset leakage report card (table + figure) | <5 s |
| 14 | `python -m audit.figures.make_graded_figure` | graded-performance figure (PNG/SVG/PDF) | <5 s |
| 15 | `python -m audit.pipeline.check1_label_concordance` | **Check 1**: nearest-train-neighbour label concordance for near-duplicates (leaky + clean) | ~6 min |
| 16 | `python -m audit.pipeline.check2_novelonly_ranking` | **Check 2**: novel-only ranking + 3-way ranking comparison (no refit) | <5 s |
| 17 | `python -m audit.pipeline.step_variance_ci` | **Phase 14**: re-split seed variance (LR k6, 5 seeds) + test-set bootstrap 95% CIs (1000 draws, no refit) + 'statistically tied' CI overlap; leaky full-scale, clean 20k | ~25 min |
| 18 | `python -m audit.pipeline.step2_rf_seeds` | **Phase 14**: RF k6 homology-aware corrected accuracy over 5 re-split seeds (both leaky) merged into the variance CSV | ~7 min |
| 19 | `python -m audit.pipeline.paper_numbers` | `results/PAPER_NUMBERS.md` (single consolidated source, with provenance) | <5 s |

Steps 1-5, 8, 10, 12, 15, 17-18 do model fitting / similarity; 6-7, 9, 11, 13-14, 16, 19 are pure aggregation.

**Splitter CLI (Part 2):** `python -m audit.core.homology_split --fasta seqs.fasta --labels labels.txt --out splits.json` (see `results/TOOL_README.md`); no `--fasta` runs the self-test.

**Standalone tool:** `audit/core/homology_split.py` is a self-contained, importable module
(`from audit.core.homology_split import homology_aware_split`); run `python -m audit.core.homology_split`
for its built-in self-test (zero residual leakage on synthetic near-duplicates).

## Headline reproduction (quickest check)

`python -m audit.pipeline.run_audit` reproduces the canonical result:

- human_nontata_promoters, **RF k6**: original **0.932** -> homology-aware **0.811** (**-12.1 pts**)
- LR k4: 0.827 -> 0.818 (-0.9 pts)  (monotone capacity scaling in between)

Full numbers land in `results/results.md`. `audit/pipeline/run_robustness_full.py` independently
re-loads and re-fits the full dataset from scratch and reproduces the same
nontata full-scale numbers, so the headline is reproduced by two independent paths.

## Determinism

Fixed numpy seed (0) + sklearn `random_state=0`; a fixed ACGT k-mer vocabulary
(featurization is data-independent); and an **exact, non-stochastic** Jaccard
similarity (no MinHash). All seeds are listed in `results/seeds.txt`.

## Tier-1 deepener modules (added after the first submission)

These produce the coordinate/construction, roster, cross-suite and diagnostic results.
All are CPU-only and run from the repo root as `python -m audit.<sub>.<module>`.
`huggingface_hub` (pinned in `results/requirements.txt`) is required by the two
cross-suite modules, which download the Nucleotide Transformer task files.

| # | Command | Output | Approx. runtime |
|---|---|---|---|
| T1 | `python -m audit.experiments.exp_inversion_law [--bootstrap]` | `inversion_law_*.csv`, `figures/fig_inversion_*.png` | 1 s / 9 min with `--bootstrap` |
| T2 | `python -m audit.experiments.exp_construction` | `construction_{signatures,provenance,rule,alignment_check}.csv` | ~1 min |
| T3 | `python -m audit.experiments.exp_construction_manip [--only A\|B]` | `construction_manipulation.csv` | ~45 min (B is full-scale) |
| T4 | `python -m audit.experiments.exp_crosssuite_census` | `crosssuite_census.csv` | ~5 min + download |
| T5 | `python -m audit.experiments.exp_crosssuite_verify` | `crosssuite_exact_verification.csv` | ~2 min |
| T6 | `python -m audit.experiments.exp_crosssuite_ranking` | `crosssuite_ranking{,_pairs}.csv` | ~15 min |
| T7 | `python -m audit.experiments.exp_roster` | `roster_{rankings,predictions}.csv` | ~60 min (full-scale ensembl) |
| T8 | `python -m audit.experiments.exp_graded_corrected` | `graded_gap_corrected.csv` | ~10 min |
| T9 | `python -m audit.experiments.exp_tuning` | `tuning_selection.csv` | ~15 min (nonTATA); the ensembl arm is hours |
| T10 | `python -m audit.experiments.exp_bh_correction` | `bh_correction_frozen.csv` | < 5 s |
| T11 | `python -m audit.tools.certify --self-validate` | `certify_self_validation.csv`; **exit 1 on drift** | ~2 s |

`certify` also runs end to end on a dataset (`--dataset NAME [--cap N]`) or on arbitrary
input (`--fasta X.fa --labels y.txt [--full-n N]`). Without `--full-n` the C1 full-scale
check reports `UNVERIFIABLE` and any clean verdict is downgraded to `provisional-clean`,
because the tool cannot tell a full dataset from a subsample by inspection.

Modules that shell out to external tools (`exp_alignment` needs MMseqs2, `exp_repeat`
needs `dustmasker`) write scratch files under `$AUDIT_SCRATCH` if set, otherwise the
platform temp directory.

### Figure 1 is built outside `audit/figures/`

`audit/figures/make_paper_figures.py` still emits a `fig_controls` image, but it is
**superseded** and is not the Figure 1 in the manuscript. The shipping figure is a
two-panel dumbbell plot (it needed a clean-dataset panel, since its caption contrasts
leaky against clean) built by:

```bash
./venv/bin/python paper/Fig/make_fig_controls.py     # reads results/summary_final.csv only
```

Run that, not the `audit/figures/` block, when regenerating Figure 1.
