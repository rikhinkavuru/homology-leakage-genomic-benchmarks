# Homology leakage selectively inverts model rankings on genomic sequence-classification benchmarks

Standard train/test splits of DNA sequence-classification benchmarks often leave near-identical (homologous) sequences on both sides, letting models score well by memorizing near-duplicates rather than generalizing. We audit the seven binary Genomic Benchmarks datasets with an exact $k$-mer Jaccard homology measure and a homology-aware re-split, and find that two datasets are materially leaky --- correcting the split removes up to **15.6 accuracy points** and, crucially, **inverts which model looks best** (a random forest falls from first to last), while the clean datasets are unaffected. We provide a per-dataset leakage report card and a drop-in homology-aware splitter so any dataset can be certified before use.

## Reproduce

All experiments are CPU-only, deterministic, and reproducible. See **[REPRODUCE.md](REPRODUCE.md)** for the exact commands and expected runtimes. Seeds are logged in [`results/seeds.txt`](results/seeds.txt) and the Python environment is pinned in [`results/requirements.txt`](results/requirements.txt). Every figure quoted in the manuscript is traced to its source script and output file in [`results/PAPER_NUMBERS.md`](results/PAPER_NUMBERS.md).

## Splitter tool

The homology-aware splitter is a standalone, dependency-light module: **[`homology_split.py`](homology_split.py)** (pure `numpy`/`scipy`, CPU-only). It exposes `homology_aware_split(...)` and a `--fasta` command-line interface, and guarantees zero residual cross-split similarity above the chosen threshold. Inputs, outputs, the guarantee, and runtime are documented in **[results/TOOL_README.md](results/TOOL_README.md)**.

## Data

This repository does **not** redistribute the Genomic Benchmarks sequence data. The datasets are downloaded on demand via the [`genomic-benchmarks`](https://pypi.org/project/genomic-benchmarks/) package (`genomic_benchmarks.loc2seq.download_dataset`) and cached locally outside this tree. Please cite the original dataset paper:

> Grešová, K., Martinek, V., Čechák, D., Šimeček, P., Alexiou, P. (2023). Genomic benchmarks: a collection of datasets for genomic sequence classification. *BMC Genomic Data* 24:25. doi:[10.1186/s12863-023-01123-8](https://doi.org/10.1186/s12863-023-01123-8)

## Manuscript

The manuscript source (OUP Bioinformatics Advances *modern, large* template) and the compiled PDF are in [`paper/`](paper/).

## License

MIT --- see [LICENSE](LICENSE).
