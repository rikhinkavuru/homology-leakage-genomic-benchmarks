# Maintainer disclosure — draft, NOT YET FILED

Reviewer item C9 asks for a defect disclosure to the benchmark's maintainers, and for
the manuscript to record the date, the issue number and any response.

**Status: not filed.** This is drafted and ready to post to
<https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/issues>, but posting is a
public, outward-facing action under the author's identity and is the author's to send,
not the analysis pipeline's. Once filed, replace the manuscript sentence in the Data
Availability section with the date, the issue number and any response received.

---

## Suggested title

Near-duplicate leakage across the train/test split in `human_enhancers_ensembl` and `human_nontata_promoters`

## Suggested body

Hello, and thank you for maintaining Genomic Benchmarks — the suite's small size and
clean interface are exactly why it gets used, and that is also why this seemed worth
reporting rather than working around quietly.

We have been auditing near-duplicate leakage across train/test splits in genomic
benchmarks, and two datasets in this collection carry a substantial amount. We think the
cause is a specific, fixable step in construction rather than anything about the
sequences, and everything below is reproducible from the shipped files.

**What we measured.**

- `human_enhancers_ensembl`: the positive class ships 77,421 genomic intervals on only
  40,934 distinct coordinates. 36,487 coordinates appear exactly twice, so 94.3% of
  positive rows sit on a duplicated coordinate. Because the split is random, 75.5% of
  test positives have their exact coordinate present in training, and 11,774 test
  sequences (38.0% of the test set) are byte-identical to a training sequence.
- `human_nontata_promoters`: 0 byte-identical duplicates, but 22.5% of test sequences
  have an 8-mer Jaccard ≥ 0.9 to a training sequence. Its negative class is dominated by
  251 bp windows tiling single loci at a median 2 bp step (self-overlap redundancy 0.961
  in the negative class against 0.047 in the positive).

**Why it matters for users.** A near-default random forest scores 0.860 on the shipped
`human_enhancers_ensembl` split and 0.696 under a near-duplicate-aware re-split. On the
near-duplicate test sequences alone it scores 1.000, and on the rest 0.770 — below the
linear SVM it outranks on the shipped split. So the split can reorder which model looks
best, not merely inflate everyone equally.

**Where it appears to come from.** Reading the construction notebooks in `docs/`, we
could not find a merge or overlap-removal step for these datasets, in the notebooks or
in the `ensembl_scraper` code two of them delegate to. `seq2loc`'s `fasta2loc` keys
results by sequence, so exact duplicates collapse incidentally, but nothing merges
*overlapping* intervals. Deduplicating positive intervals by coordinate before splitting
removes the leakage: on our copy it takes the leak fraction from 0.390 to 0.012 and the
forest's inflation from +0.165 to −0.008.

Four of the seven binary datasets show none of this, so this is not a criticism of the
collection as a whole.

**What might help, if you agree it is worth changing.**

1. Merge or deduplicate overlapping positive intervals in coordinate space before the
   train/test split, for the datasets built from interval scrapes.
2. Alternatively, assign whole overlap-connected components to one side of the split.
3. Either way, it would help users a lot if the shipped metadata recorded the
   per-dataset near-duplicate fraction, so a leaky split is visible without re-deriving
   it.

We are happy to open a PR with the deduplication step and the re-derived splits if that
would be useful, and equally happy to be told we have misread the construction — in
which case we would like to correct our own write-up.

Code and the full report card: <https://github.com/rikhinkavuru/homology-leakage-genomic-benchmarks>

---

## Manuscript sentence to use once filed

> We reported these findings to the benchmark's maintainers on <DATE> (issue
> <NUMBER>); <the response, or "no response had been received at the time of
> submission">.
