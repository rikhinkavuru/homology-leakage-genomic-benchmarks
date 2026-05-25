# REVISIONS.md — Phase 2 proposals (judgment-required; NOT yet applied)

These touch scientific claims or need author input. **Nothing here has been edited into `main.tex`.** Review and tell me which to apply. Line numbers refer to `paper/main.tex` at commit `e73ee82`.

---

## 1. Abstract — telegraph clean-vs-partial mechanism

**Status: already satisfies — no change proposed.**

The abstract already carries the softened clean/partial distinction:

> "A homology-graded analysis indicates the mechanism is largely memorization of label-carrying near-duplicates---**cleanly on one leaky dataset and partially on the other.**"

This was added in an earlier pass and is exactly the hedge you asked for, so per your instruction I am not re-proposing it.

*Optional (only if you want the abstract more explicit — not proposed, just noted):* it could name the datasets and the number, e.g. "...cleanly on \textit{human\_enhancers\_ensembl} but only partially on \textit{human\_nontata\_promoters}, where the random forest retains a genuine edge on novel sequences (0.879)." This adds ~1 line and names datasets in the abstract; I left it out because you flagged the abstract as already-satisfied.

---

## 2. §4.3 Kendall τ paragraph — lead with concrete rank changes

**Status: already satisfies — no change proposed.**

§4.3 already leads with the concrete demotions and demotes τ to descriptive support, including the coarse-resolution caveat:

> "On both leaky datasets the apparent best model on the standard split---a random forest---is demoted after correction, falling from rank~1 to rank~3 on \textit{human\_nontata\_promoters} and from rank~1 to last (rank~4) on \textit{human\_enhancers\_ensembl} ... Summarized by rank correlation, the mean Kendall $\tau$ ... is $-0.17$ ... versus $+0.67$ ...; with only four models, however, $\tau$ is restricted to a coarse set of values ($0, \pm0.33, \pm0.67, \pm1$ ...), so we treat it as descriptive support for the concrete demotion above rather than as a primary statistic."

This is the rank-changes-first, τ-demoted structure you asked for, so I am not re-proposing it.

*Optional (not proposed):* if you want it even tighter, the parenthetical listing the τ value set could be shortened to "τ takes only five possible values with four models," saving ~half a line. Minor; left as-is.

---

## 3. Methods — specify the bootstrap (PROPOSED — new sentence)

**Status: change proposed.** The paper reports "95\% bootstrap CI" in the abstract, §4.2, and §4.4 but never defines the bootstrap in Methods. Add one sentence.

**Where:** end of §3.6 "Reproducibility" (it already discusses seeds/determinism, so the fixed-seed bootstrap fits there). Current last sentence of §3.6:

> "...The homology-aware splitter is released as a standalone tool (\texttt{homology\_split.py}, dependencies \texttt{numpy}/\texttt{scipy})."

**Proposed addition (append after that sentence):**

> "Reported 95\% confidence intervals are percentile bootstrap intervals from 1{,}000 resamples of the per-test-example correctness vector; the resampling unit is the individual test sequence and models are not refit, and a fixed random seed makes the intervals reproducible."

Specifies: **n = 1,000 resamples**, **percentile** method (not BCa), **resampling unit = individual test sequences**, **no model refit**, **fixed seed**. (These match the frozen analysis in `results/PAPER_NUMBERS.md` §16.2; seed `20240524`. I can cite the seed explicitly if you prefer, but Methods sentences usually omit the literal value.)

---

## 4. Reviewer-risk / experiments needed (FLAGGED ONLY — I cannot run these without your data/compute)

Two experiment-level objections a referee is likely to raise. Both need data/compute I don't have here; I am **not attempting** them, only laying out concrete plans.

### (a) Circularity: k-mer Jaccard similarity vs k-mer features
**Risk.** Leakage is measured by exact **8-mer Jaccard**, and the classifiers use **4/6-mer count features**. A reviewer can argue the homology-aware split removes precisely the k-mer overlap the features exploit, so the measured drop is partly definitional rather than a property of *homology*. This is the single most likely "reject/major-revision" hook.
**Experiment to defuse it.** Re-derive the homology-aware split from an **alignment-based** similarity independent of k-mer counts, and show the drop + RF demotion persist:
1. Cluster each leaky dataset's sequences with **MMseqs2** (`mmseqs easy-cluster --min-seq-id 0.7 -c 0.8`) or BLAST-based identity; treat clusters as the near-duplicate groups.
2. Re-split by whole MMseqs2 clusters (same whole-cluster, ratio/balance-preserving rule as now), giving an alignment-defined corrected split.
3. Re-evaluate LR/LinearSVC/RF/HGB (k=6) on `human_nontata_promoters` + `human_enhancers_ensembl`, original vs MMseqs2-corrected.
**Pass criterion.** RF still loses a material margin and is still demoted under the alignment-based split → the effect is homology, not k-mer self-reference.
**Cost/needs.** MMseqs2 binary + the raw FASTA sequences (not in repo). CPU-feasible (~155k seqs clusters in minutes–low hours). The splitter already accepts an arbitrary cluster assignment, so only the clustering step is new. **MMseqs2/CD-HIT are now cited (Agent A) so the framing is in place.**

### (b) Classical models only vs a neural-net motivation
**Risk.** The motivation invokes "models memorizing near-duplicates," and genomic benchmarks are dominated by CNNs/transformers, but we test only **classical** models (LR/LinearSVC/RF/HGB). A reviewer will ask whether the inflation/ranking effect holds for the deep models people actually rank on these suites.
**Experiment to defuse it.** Add **one** neural baseline:
1. A small **1D-CNN** on one-hot DNA (e.g. 2–3 conv blocks + global pool + dense), no k-mer features, trained on `human_nontata_promoters` + `human_enhancers_ensembl`.
2. Evaluate on **both** the original and the homology-aware split (and, ideally, the novel-only ≥... <0.5-similarity set).
**Pass criterion.** The CNN also drops materially under correction (memorization signature) and its leaky-split standing is inflated → the mechanism generalizes beyond classical models, matching the softened "we expect the mechanism to generalize" claim now in the Discussion.
**Cost/needs.** A GPU (or patient CPU) + the raw sequences — this **conflicts with the current CPU-only/\$0 constraint**, so it explicitly needs your decision on compute. Scope it to one CNN on the two leaky datasets to keep it minimal.

*Neither (a) nor (b) was attempted. If you greenlight either and provide the sequences/compute, I can implement it as a new analysis script consistent with the frozen pipeline.*
