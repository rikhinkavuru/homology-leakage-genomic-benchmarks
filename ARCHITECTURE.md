# Architecture

The code is a single importable Python package, `audit/`, organized by role. All
scripts are run **as modules from the repository root** so the package resolves:

```bash
python -m audit.<subpackage>.<module>
# e.g.
python -m audit.pipeline.run_audit
python -m audit.experiments.exp_stats
python -m audit.tools.prefetch
python -m audit.core.expkit          # expkit self-test
python -m audit.core.homology_split  # splitter self-test (no --fasta)
```

Running a file by path (`python audit/pipeline/run_audit.py`) will **not** work,
because the scripts use absolute package imports (`from audit.core import expkit`)
that require the `audit` package to be importable — which `-m` from the repo root
guarantees.

## Layout

```
audit/                         importable package
  core/                        shared library — depends on nothing internal
    expkit.py                  verified experiment kit (loaders, features, splits,
                               similarity, metrics, bootstrap); wraps the frozen pipeline
    homology_split.py          drop-in near-duplicate-aware splitter (numpy/scipy; --fasta CLI)
  pipeline/                    the frozen audit pipeline + aggregation/report scripts
    run_audit.py               single-dataset headline pipeline; defines RESULTS_DIR
    run_suite.py               subsample sweep across the dataset suite
    run_fullscale.py           full-scale re-split for the leaky datasets
    run_robustness_full.py     full-scale threshold sweep + drop-largest
    run_extended_models.py     LR / LinearSVC / RF / HGB on the frozen splits
    run_graded.py              homology-graded per-model accuracy by similarity bin
    finalize.py                consolidate -> summary/capacity CSVs + results_FINAL.md
    measure_leakage_full.py    full-scale test->train 8-mer Jaccard leakage per dataset
    ranking_inversion.py       rank models per split, Kendall tau, stable inversions
    report_card.py             per-dataset leakage report card (table + figure)
    paper_numbers.py           consolidated PAPER_NUMBERS.md with provenance
    prep_datasets.py           download/prepare datasets via genomic_benchmarks
    step2_rf_seeds.py          RF re-split seed variance (leaky datasets)
    step_variance_ci.py        re-split seed variance + test-set bootstrap CIs
    rf_seed_variance.py        RF training-seed robustness
    validate_splitter.py       validate homology_split.py on the leaky datasets
    check1_label_concordance.py  nearest-train-neighbour label concordance
    check2_novelonly_ranking.py  novel-only ranking + 3-way comparison
  experiments/                 revision experiments (each imports core via expkit)
    original revision:
      exp_stats.py, exp_regpath.py, exp_canonical.py, exp_alignment.py,
      exp_imbalance.py, exp_inject3class.py, exp_geometry.py, exp_clusterboot_full.py,
      exp_repeat.py, exp_deep.py, exp_transformer.py, cluster_bootstrap.py,
      exact_dup_count.py, chromosome_holdout.py, full_scale_containment.py
    resubmission (the six deepeners and their supports):
      exp_roster.py              nine-model roster; P1-P4 scored in the output
      exp_graded_corrected.py    prevalence-corrected graded gap (balanced accuracy)
      exp_construction.py        coordinate-space construction signatures
      exp_construction_manip.py  construct-and-break, with cluster-bootstrap intervals
      exp_inversion_law.py       the phi* = delta/Dg condition, in and out of sample
      exp_dose_response.py       ten-dose causal test of phi*; P6
      exp_tuning.py              what naive vs cluster-grouped CV selects; P5
      exp_gue.py                 GUE census, executing the pre-registered predictions
      exp_crosssuite_census.py   Nucleotide Transformer census (run uncapped)
      exp_crosssuite_ranking.py  re-ranking on the tasks the census flags
      exp_crosssuite_verify.py   byte-identical verification, no k-mers
      exp_estimator_sensitivity.py  what the detector detects, by sequence length
      exp_bh_correction.py       Benjamini-Hochberg over the frozen delta families
  figures/                       publication-figure generators (read results/)
    make_paper_figures.py        Figures 2-3 (its fig_controls block is SUPERSEDED)
    make_part_b_figures.py, make_graded_figure.py
    NOTE Figure 1 is built by paper/Fig/make_fig_controls.py, not from this package.
  tools/                         operational helpers and gates
    prefetch.py                  serial build of the local datacache/*.pkl
    run_serial.sh                serial experiment chain (one heavy job at a time)
    certify.py                   the executable certification standard (C1-C9);
                                 --self-validate is a regression gate over the
                                 published verdicts
    check_numbers.py             the numbers gate: every registered document claim
                                 recomputed from its CSV, plus retired-claim and
                                 cross-document consistency checks. --self-test proves
                                 the gate can fail.
```

`exp_transformer.py` is an extra, git-untracked experiment that follows the
`exp_*` convention and imports only `core`; it lives in `experiments/` for
consistency.

## Dependency direction

```
experiments/  ─┐
figures/       ├──►  core/        (core depends on nothing internal)
pipeline/     ─┘
                     ▲
experiments/, figures/, and most pipeline aggregators reach core through
`expkit`, which itself wraps the frozen pipeline primitives
(run_audit / run_suite / homology_split / run_extended_models).
```

- `core/homology_split.py` imports nothing from the package.
- `core/expkit.py` imports the frozen pipeline primitives
  (`run_audit`, `run_suite`, `homology_split`, `run_extended_models`) and re-exposes
  them behind one tested surface, so every experiment stays consistent with the
  published numbers.
- `pipeline/` modules import each other (`run_suite` -> `run_audit`;
  `run_extended_models` -> `run_audit`, `run_suite`) and `core/homology_split`.
- `experiments/`, `figures/`, and `tools/` depend on `core` (and, where needed,
  `pipeline`); nothing in `core` depends on them.

## Paths (unchanged behavior)

`results/`, `datacache/`, and `figures/` still resolve to the **repository root**,
not under `audit/`:

- `run_audit.RESULTS_DIR` and every `R = .../results` are computed as
  `os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
  + `"results"` — three `dirname`s climb from `audit/<subpackage>/<file>.py` back
  to the repo root.
- `expkit.DATACACHE` and `prefetch` resolve `datacache/` at the repo root the same way.

Because every module sits exactly two directories below the repo root
(`audit/<subpackage>/`), the three-`dirname` climb is uniform across the package.
