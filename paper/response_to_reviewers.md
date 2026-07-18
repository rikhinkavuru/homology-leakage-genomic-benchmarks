# Response to Reviewers — BIOADV-2026-296 (revised resubmission)

We thank the reviewers and the Associate Editor for a rigorous and constructive
review. The revision makes three structural changes before addressing the
individual comments:

1. **Title and terminology.** The paper is retitled *"Near-duplicate leakage can
   reorder model rankings on a genomic benchmark suite: an audit of Genomic
   Benchmarks."* We adopt **near-duplicate leakage** as the primary term (an
   exact-duplicate census shows the strongest dataset, `human_enhancers_ensembl`, is
   38.0% *byte-identical* train/test copies), reserve *homology* for where
   alignment/containment evidence earns it, and rename the splitter
   **near-duplicate-aware** (addresses R3.4).

2. **Honest rescoping of the central claim.** The previous "changes which model
   wins on 2/2 leaky datasets" claim was too strong: on `human_nontata_promoters`
   the demotion holds only under accuracy@0.5 (it reverses under AUROC/F1, is a
   CI-overlap tie, and does not reproduce under a chromosome-holdout control). The
   ranking-change result is now claimed **cleanly for `human_enhancers_ensembl`**
   (an existence proof, airtight on every axis) and **as a cautionary partial
   case for `human_nontata_promoters`** (§4.3, Discussion).

3. **New experiments.** We added a from-scratch deep-model dose-response, an
   alignment-based re-measure, a chromosome-holdout control, a random-forest
   regularization path, a cluster/block bootstrap, a containment/canonical-k-mer/GC
   robustness battery, an imbalanced-prevalence panel, and an injected-leakage
   multiclass experiment. All numbers below are traceable to the results log.

---

## Associate Editor (Prof. Shanfeng Zhu)

The Associate Editor summarized five overarching concerns; each is now addressed and cross-referenced to the detailed responses below.

- **Limited model scope.** A from-scratch 1D residual CNN with a pre-registered dropout×weight-decay dose-response now shows the effect reaches a trained neural network (regularized reference cell drops on both leaky datasets, CIs excluding zero); the named pretrained foundation models are scoped out with a *measured* ≈100% pretraining-overlap confound rather than merely asserted (R1.1, R2.a3, R3.1).
- **Validation of homology detection.** MMseqs2 alignment identity reproduces the effect metric-independently, and a length-robust containment index is added — which honestly flags one previously-clean dataset as borderline (R1.3, R2.a2, R3.4).
- **Bootstrap methodology.** A cluster (block) bootstrap over within-test near-duplicate components, with intraclass correlation, design effect, and an analytic cluster-robust cross-check, now quantifies the correlated-sample uncertainty; it changes no significance verdict (R3.3).
- **Dataset characteristics.** Class balance is disclosed as curated (not natural), and a prevalence-aware imbalanced evaluation is added (R2.a1).
- **Hyperparameter choice.** A random-forest regularization path shows the effect is a property of the *default unregularized* forest, dissolving under regularization (R3.2).

Most importantly, the central claim is now honestly rescoped from "changes which model wins on 2/2 leaky datasets" to a clean single-dataset existence proof plus an explicit cautionary partial case — directly addressing the generality-and-robustness concern at the heart of the decision.

---

## Reviewer 1

### R1.1 — "The evaluation uses only classical models; a trained deep network is needed, and the reach beyond this domain is unclear."

**(a) Deep model.** We added a **from-scratch 1D residual CNN** (one-hot input, no
k-mer features, trained on each training split only — no pretraining confound),
with a **pre-registered** dropout×weight-decay dose-response and a binding
refutation condition pre-specified in a timestamped document in the code release (results/deep_preregistration.md). The standard-practice
regularized reference cell (dropout 0.3, weight decay 1e-4) drops on **both** leaky
datasets — `human_nontata_promoters` **+0.043** (cluster-bootstrap CI [0.025, 0.061]),
`human_enhancers_ensembl` **+0.014** ([0.007, 0.021]), both excluding zero — so the
pre-registered refutation condition is **not** triggered. 19 of the 20 cells drop
with cluster CIs excluding zero (the one exception is a single nontata cell,
dropout 0.6/wd 1e-4, where heavy dropout underfits so there is no inflation to lose);
per-cell single-run variance is non-trivial, so we read the mechanism from the
grid-level contrast, and the unregularized manipulation-check cell memorizes hardest
(ensembl graded gap **+0.222**, mirroring the unregularized RF).
The effect is therefore **not** a classical-model artifact. New **§4.6**; Methods
**§3.8**; pre-registration in `results/deep_preregistration.md`.

**(b) Reach.** We now separate two claims explicitly (Introduction): **Claim I**
(de-leaking lowers accuracy) is credited to prior work across domains
(Kapoor & Narayanan; hashFrag/Rafi; Bushuiev; PINDER) and reproduced only as a
control; **Claim II** (leakage can *reorder* rankings) is our contribution, scoped
to short human regulatory DNA plus a memorization-prone model. The title now carries
that scope.

### R1.2 — "The corrected ranking is treated as the true generalization ranking, but de-duplicating the training set also removes learnable signal."

Agreed, and we now **measure** rather than assert this. On `human_nontata_promoters`
the corrected and novel-only rankings diverge (τ = −0.67): the random forest is
demoted under the corrected split yet remains the best model on truly novel
sequences (accuracy 0.879, <0.5 similarity), so its advantage is partly genuine
generalization. We reframe both the corrected split and novel-only accuracy as
explicit near-duplicate-bounded proxies (Methods §3.4/§3.7), name
leave-one-chromosome-out as the functional ground truth (§4.8), and single-linkage
cohesion (0.083 for nontata vs 0.96 for ensembl, §4.10) supplies the biological
reason correction over-removes signal on nontata. This is the core of the
nontata-as-cautionary-case framing (§4.3, Discussion).

### R1.3 — "Validate leakage with an alignment-based measure and give threshold recommendations."

**Alignment:** we re-clustered both leaky datasets with **MMseqs2** alignment
identity, fed the external clusters into the *same* whole-cluster splitter (only the
edge metric changes), and refit all four models. The `human_enhancers_ensembl` effect
is metric-independent (RF drop **0.164** Jaccard ≈ **0.162** alignment; RF still rank
4); nontata is milder under alignment (0.108→0.064), reinforcing its partial status.
We describe this as "alignment-scored," not k-mer-independent, since MMseqs2 is
k-mer-seeded (new **§4.7**, Methods **§3.5**). **Thresholds:** the clustering threshold
sweep (0.5/0.7/0.9) changes no verdict (Methods §3.4); the length-robust containment
re-measure and the bimodal-vs-graded similarity structure (cohesion 0.96 vs 0.083)
provide the decision basis — a single blind-antimode threshold suffices for the
discrete-duplication case (ensembl) whereas the chaining case (nontata) must be
treated as partial-scope (§4.1, §4.10).

---

## Reviewer 2

### R2.a1 — "Class balance is not characterized, and imbalanced data are not tested."

We now disclose that every balanced dataset is **curated** to a 0.500 positive
fraction in both train and test (positive/negative subfolders); only nontata is a
natural mild imbalance (0.544) (Methods §3.1). Because balance is preprocessed, we
add a **prevalence stress test** (RF, π ∈ {0.5, 0.2, 0.1}) with prevalence-aware
metrics: the leakage inflation is **revealed by AUPRC/MCC/minority-recall but masked
by accuracy** (ensembl π=0.2: AUPRC 0.622→0.477, MCC 0.422→0.182, minority recall
0.258→0.070, accuracy barely moves 0.844→0.808). The prevalence-aware splitter
preserves the target prevalence (realized 0.5001/0.2007/0.1001). New **§4.11**;
Methods **§3.10**.

### R2.a2 — "The similarity measure and the features are both k-mer based, so the leakage finding may be circular."

Addressed by the alignment re-measure above (R1.3): with alignment identity as the
edge metric and the same splitter, the `human_enhancers_ensembl` RF drop is
essentially unchanged (0.164 vs 0.162) and the demotion persists, so the effect does
not depend on the k-mer definition of similarity. We additionally note that
ensembl's leakage is 38.0% byte-identical duplication (metric-independent by
construction). New **§4.7**; Methods **§3.3/§3.5**.

### R2.a3 — "Extend to deep models and to the multiclass setting."

**Deep models:** covered by the CNN (R1.1a, §4.6). **Multiclass:** the only
multiclass set is clean, so we answer by construction — injecting label-carrying
near-duplicates at fraction f ∈ {0, 0.1, 0.2, 0.4} into the clean 3-class
`human_ensembl_regulatory` set. The RF drop grows monotonically with f (+0.0004,
+0.118, +0.179, +0.258) while LR stays flat (≤0.066), and the split removes it
(corrected accuracy stable ≈0.58). New **§4.12**; Methods **§3.11**; we state
explicitly that leakage is injected because the suite ships no naturally leaky
multiclass set.

### R2.b1 — "Give a threshold-sensitivity analysis and concrete recommendations."

The verdict is robust to the clustering threshold (0.5/0.7/0.9 sweep, no change;
Methods §3.4), and we now report the geometry that drives the recommendation: the
length-robust containment index (§4.1/§4.10) as the metric to prefer when lengths
vary, and single-linkage cohesion as the diagnostic that separates a clean
discrete-duplication case (ensembl, treat with one threshold) from a transitive-chaining case
(nontata, treat as partial-scope and avoid over-removal). Report-card guidance and
the length-cap disclosure are in Table 2.

---

## Reviewer 3

### R3.1 — "Deep models are needed, specifically attention-based / foundation models (e.g. DNABERT-2, HyenaDNA)."

**(i) Trained deep net:** answered by a from-scratch 1D residual CNN (§4.6, one-hot
input, no k-mer features, pre-registered dropout×weight-decay dose-response) trained on
the training split only, whose regularized reference cell drops on both leaky datasets
(cluster-bootstrap CIs excluding zero) — the memorization mechanism reaches a trained
neural network with no pretraining confound. We also probed a from-scratch attention
(Transformer) encoder under the identical protocol; consistent with transformers being
data-hungry and weaker than convolutions on short, local-motif inputs, it underfits
these 251–600 bp tasks and does not yield a clean aggregate-accuracy comparison, so we
do not report it as evidence. The CNN is the appropriate trained-deep-net instrument
here, and the pretraining-contamination argument below specifically justifies excluding
*pretrained* attention models, not attention architectures per se. **(ii)/(iii) Named foundation models:**
DNABERT-2, HyenaDNA, and the Nucleotide Transformer are cited and discussed but scoped
**out** of the split-effect evidence, and we now **quantify** the confound rather than
merely assert it: the two leaky datasets' sequences are extracted from the human
reference genome (we recovered their hg38 chromosomal coordinates for the
chromosome-holdout control), so **≈100% of these test sequences already lie inside
HyenaDNA's whole-genome (hg38) pretraining corpus** and within the human component of
DNABERT-2's and the Nucleotide Transformer's multispecies corpora — a second leakage
channel that no train/test re-split can close. Fine-tuning a model that has already
seen the test sequences during pretraining yields a confounded lower bound, not
split-effect evidence, so we leave the deployed-foundation-model question **genuinely
open** (Discussion, Limitations) rather than answer it with a contaminated number.

### R3.2 — "The random forest runs at memorization-maximizing defaults; is the result just an untuned-RF artifact?"

We added a full **regularization path** (min_samples_leaf ∈ {1…500}, max_depth ∈
{None…4}, at full scale, with the largest settings exceeding the largest
near-duplicate cluster). At the default the forest scores 1.000 on the ≥0.9-similarity bin
with the full drop (0.164/0.108) and graded gap (0.230/0.121); regularizing collapses
the drop to ≈0 (0.164→−0.001; 0.108→0.000) and the graded gap, and the "RF wins on the
leaky split" phenomenon **requires** the unregularized forest (rf_rank_orig flips 1→4
once min_samples_leaf ≥ 20). We frame this as an owned finding: the inflation and
demotion are a property of the **default** high-capacity RF practitioners deploy, and
either regularization or a near-duplicate-aware split reveals the truth. New **§4.5**;
Methods **§3.7**. The graded memorization gap is used as the confound-immune headline
(immune to "regularized RF just got worse everywhere").

### R3.3 — "Near-duplicates violate independence; a cluster/block bootstrap is the correct uncertainty estimate."

Implemented and run: resampling whole within-test near-duplicate components widens CIs
by 1.0–1.5× but **flips no verdict** — the RF drop still excludes zero on both datasets
(ensembl [0.156, 0.172]; nontata [0.092, 0.126]) and the ensembl-LR drop stays null.
The design effect is small because the correlated near-duplicates are test-to-**train**,
not test-to-test (within-test clustering only 1.06–1.19 sequences/component). New
**§4.9**; Methods **§3.9**. Every deep-model drop CI (§4.6) already uses this cluster
bootstrap. (The full design-effect/ICC/CRVE/combined-source detail is marked as being
finalized in §3.9/§4.9.)

### R3.4 — "'Homology' is the wrong word for near-identical/duplicate leakage."

Agreed and adopted throughout: **near-duplicate leakage** is the primary term, the
splitter is **near-duplicate-aware**, and the title is changed. An exact-duplicate
census settles this metric-independently (`human_enhancers_ensembl` = 38.0%
byte-identical copies; `human_nontata_promoters` = 0 exact but 22.5% near-duplicates at
Jaccard ≥0.9). We keep "homology" only where the alignment/containment re-measure
earns it (Methods §3.3), and we downgrade the report card's green verdicts to "no
forward-strand near-duplicate leakage detected (length-cap disclosed)" (Table 2,
Discussion), since a clean verdict carries a heavier evidential burden than a leaky
one.

---

## Associate Editor — "This cannot be addressed in a short time / new experiments are needed."

The revision adds the two hard-gate experiments the panel expected — a trained deep
model (from-scratch CNN dose-response, §4.6) and alignment-based validation (MMseqs2,
§4.7) — plus the most decisive genomics-standard control, a **leave-one-chromosome-out
split** (§4.8), which independently reproduces the marquee `human_enhancers_ensembl`
demotion (RF rank 1→4) and independently confirms that `human_nontata_promoters` is
the partial case (RF stays rank 1). We also closed two internal issues a careful
reviewer would have flagged: the central claim is now honestly rescoped to
`human_enhancers_ensembl` (n=1 clean, partial on the second), and the previously
inconsistent headline drop numbers are reconciled to a single decision rule
(.predict()/argmax) and estimator (5-seed-mean + bootstrap CI), removing the earlier
three-coexisting-values spread and the incorrect "thread-nondeterminism" wording.
