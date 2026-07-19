# Near-duplicate leakage can reorder model rankings on a genomic benchmark suite

Code and data for the paper *"Near-duplicate leakage can reorder model rankings on a
genomic benchmark suite: an audit of Genomic Benchmarks"* (Kavuru; resubmission of
BIOADV-2026-296, Bioinformatics Advances).

**Claim.** On the Genomic Benchmarks binary suite, near-duplicate sequences that span
the train/test split inflate the apparent performance of memorization-prone models and
can **reorder which model appears best** — demonstrated cleanly on `human_enhancers_ensembl`
(airtight across accuracy/AUROC/F1, a chromosome-holdout control, a winner-probability
bootstrap, and MMseqs2 alignment) and as an explicit cautionary partial case on
`human_nontata_promoters`. We ship a per-dataset leakage **report card** and a drop-in,
dependency-free **near-duplicate-aware splitter** so any dataset can be certified before use.

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib genomic-benchmarks
# deep-model checks additionally need:      pip install torch
# alignment validation additionally needs:  brew install mmseqs2 blast   (dustmasker ships with blast)

python -m audit.tools.prefetch    # build the local data cache once (serial; avoids download races)
```

Datasets download on demand via the `genomic-benchmarks` package (cached under
`~/.genomic_benchmarks`); a local pickle/npy cache is built lazily under `datacache/`
(git-ignored, **not** redistributed). See [`REPRODUCE.md`](REPRODUCE.md) for exact
commands and runtimes; seeds are in [`results/seeds.txt`](results/seeds.txt).

The code is packaged under `audit/` (see [`ARCHITECTURE.md`](ARCHITECTURE.md)). Run
every script **as a module from the repo root**: `python -m audit.<subpackage>.<module>`
(e.g. `python -m audit.pipeline.run_suite`). Each experiment writes CSVs to `results/`
and prints a summary:

| Script | Produces | Reviewer point |
|---|---|---|
| `audit/core/expkit.py` | shared verified helper (loaders, features, splits, similarity, bootstrap) — imported by all `exp_*` | — |
| `audit/pipeline/measure_leakage_full.py`, `audit/pipeline/run_suite.py`, `audit/pipeline/run_fullscale.py` | leakage fractions + main results | — |
| `audit/core/homology_split.py` | the drop-in near-duplicate-aware splitter (numpy/scipy; `--fasta` CLI) | R3.4 |
| `audit/experiments/cluster_bootstrap.py`, `audit/experiments/exp_clusterboot_full.py` | cluster/block bootstrap + ICC / design-effect / CRVE | R3.3 |
| `audit/experiments/exp_regpath.py` | random-forest regularization path + graded memorization gap | R3.2 |
| `audit/experiments/exp_stats.py` | AUROC/F1 rankings, P(model rank-1) bootstrap, clean-set CIs, short-seq census | R3, central claim |
| `audit/experiments/exp_alignment.py` | MMseqs2 alignment re-cluster + refit | R2.a2 |
| `audit/experiments/chromosome_holdout.py` | leave-chromosomes-out control (recovers hg38 coordinates) | genomics-standard |
| `audit/experiments/exp_geometry.py`, `audit/experiments/full_scale_containment.py` | length-cap + containment index, GC-shift, cluster cohesion | R1.3, R3.4 |
| `audit/experiments/exp_canonical.py` | reverse-complement-canonical k-mers (metric + features) | strand |
| `audit/experiments/exp_imbalance.py` | balance provenance + prevalence-aware imbalanced panel | R2.a1 |
| `audit/experiments/exp_inject3class.py` | injected-leakage multiclass construction | R2.a3 |
| `audit/experiments/exp_deep.py` | from-scratch 1D CNN dropout×weight-decay dose-response (MPS/CPU) | R1.1, R3.1(i) |
| `audit/experiments/exact_dup_count.py` | exact byte-identical train/test duplicate census | terminology |
| `paper/Fig/make_fig_controls.py` | Figure 1 (the `audit/figures/` block is superseded) |
| `audit/figures/make_paper_figures.py`, `audit/figures/make_part_b_figures.py`, `audit/figures/make_graded_figure.py` | Figures 2–3 | — |

Deep-model pre-registration (claim, dose-response grid, binding refutation condition):
[`results/deep_preregistration.md`](results/deep_preregistration.md).

## Key results
- Consolidated numbers with provenance: [`results/NEW_FINDINGS.md`](results/NEW_FINDINGS.md)
- Decision-rule / headline-number reconciliation: [`results/reconciled_numbers.md`](results/reconciled_numbers.md)
- Per-dataset leakage report card: [`results/leakage_report_card.csv`](results/leakage_report_card.csv) (Table 2 of the paper)

## Manuscript
`paper/main.tex` (compiles with `tectonic main.tex`), `paper/response_to_reviewers.md`,
`paper/cover_letter.md`.

## Data
This repository does **not** redistribute the Genomic Benchmarks sequence data; it is
obtained via the [`genomic-benchmarks`](https://pypi.org/project/genomic-benchmarks/)
package. Please cite the original dataset paper:

> Grešová, K., Martinek, V., Čechák, D., Šimeček, P., Alexiou, P. (2023). Genomic
> benchmarks: a collection of datasets for genomic sequence classification.
> *BMC Genomic Data* 24:25. doi:[10.1186/s12863-023-01123-8](https://doi.org/10.1186/s12863-023-01123-8)

## License
MIT — see [LICENSE](LICENSE).
