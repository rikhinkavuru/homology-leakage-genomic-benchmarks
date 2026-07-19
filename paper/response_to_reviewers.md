# Response to Reviewers — BIOADV-2026-296 (revised resubmission)

We thank the reviewers and the Associate Editor for a rigorous and constructive
review. The decision turned on one question — *is the ranking result general, or
is it an artifact of this particular comparison?* — and we have tried to answer
it with experiments rather than argument.

## What is new, and why it answers the decision

Four additions carry the revision. We put them first because each targets the
generality concern directly.

1. **A nine-model roster (§4.4, Table 3).** The sharpest form of the objection is
   that a claim about "rankings" made over four models is a claim about nothing.
   We widened the comparison to nine learners spanning memorization propensity end
   to end — 1- and 15-nearest neighbours, random forest, extremely randomized
   trees, a multilayer perceptron, gradient boosting, linear SVM, logistic
   regression, Gaussian naive Bayes — all trained from scratch on the same splits
   with the same decision rule. The result is stronger than the four-model version
   and sharper: **the leaky split cannot separate its top three models at all**
   (0.8736 / 0.8729 / 0.8721, within 0.0015, paired interval on the top-two margin
   including zero) **yet those same three span 23 accuracy points once corrected**
   (0.768 / 0.537 / 0.628). 1-NN falls from rank 2 to rank 9, 0.873 → 0.537. The
   benchmark is not merely crowning the wrong model; it is blind to enormous real
   differences between models it reports as tied. The corrected winner is the
   linear SVM — the same model the four-model comparison selects, so the two
   analyses agree. Four predictions (P1–P4) were committed before the run and all
   four are scored in the text whether or not they held.

2. **A construct-and-break manipulation (§4.7, Table 4).** The previous submission
   argued the cause was a curation defect; it did not *test* it. We now intervene
   on the construction step alone, in both directions. Applying the omitted merge
   step to `human_enhancers_ensembl` collapses its leak fraction 0.390 → 0.012, its
   random-forest inflation +0.165 → −0.008, and the reordering disappears.
   Conversely, imposing the same defect on the clean `human_ocr_ensembl` — at
   matched size *and* balance — raises leakage 0.004 → 0.204, inflation
   +0.002 → +0.105, and **manufactures the reordering on demand** (RF promoted
   1st on the contaminated split, demoted to 4th by correction). Model code,
   split ratio and (in the fix-a-leaky direction) the sequences themselves are
   untouched; the intervention is on the construction step alone. This converts
   the causal claim from correlational to interventional.

3. **A cross-suite census (§4.10) — and an honest null.** We applied the census
   unchanged to the Nucleotide Transformer downstream tasks. Three of eleven
   independent tasks are leaky — one at 25.0% *byte-identical* train/test overlap,
   verified on a separate code path using no k-mers — and six more borderline. **No ranking inverts
   materially on any of them, and we report that null prominently rather than
   burying it.** One swap does occur — HistGradientBoosting overtakes the forest on
   `enhancers` after correction, 0.887 vs 0.869 — but the two were separated by only
   0.005 on the as-shipped split, so it is immaterial under our own inversion
   criterion, and we say so rather than counting it either way. The null
   is not an embarrassment but a test passed: our diagnostic condition (§3.13)
   predicts it in advance from the as-shipped split, because reordering requires
   not just leakage but a challenger that is genuinely better on novel sequences,
   and on 22 of 24 pairs the as-shipped leader is also the novel-sequence leader.
   On the two pairs where the condition had something to say it was right both
   times — including the one case a trivial "nothing swaps" baseline gets wrong.
   Only one of those two is admissible under our own Methods §3.13, which permits
   the condition only where the intermediate similarity band is small: the
   `splice_sites_acceptors` pair's band is 33.3%, so we rule that call out and do
   not count it. The condition therefore rests on **one** admissible out-of-sample
   call, and we state it that way in §4.10 rather than claiming the two.
   We also state plainly that the aggregate 24/24 score is nearly uninformative,
   since that baseline scores 23/24.

4. **Honest rescoping of the central claim.** The previous "changes which model
   wins on 2/2 leaky datasets" claim was too strong. On `human_nontata_promoters`
   the demotion holds only under accuracy@0.5: it reverses under AUROC and F1, is a
   CI-overlap tie, does not reproduce under chromosome holdout, is milder under
   alignment, and the prevalence-corrected graded gap does not separate the model
   families there at all. The ranking-change result is now claimed **cleanly for
   `human_enhancers_ensembl`** (an existence proof, holding on every axis tested)
   and **as an explicit cautionary partial case for `human_nontata_promoters`**.
   The title, abstract, Discussion and new Conclusion all carry that scope.

Three further structural changes: the paper adopts **near-duplicate leakage** as
its primary term and reserves *homology* for where alignment or containment
evidence earns it (R3.4); it adds a **Conclusion** (§6); and it now reports its
**software environment** with pinned versions (§3.14).

---

## Associate Editor (Prof. Shanfeng Zhu)

The Associate Editor summarized five overarching concerns; each is addressed and
cross-referenced below.

- **Limited model scope.** Answered twice over: the nine-model roster above
  (§4.4), and a from-scratch 1D residual CNN with a pre-registered
  dropout×weight-decay grid and a binding refutation condition (§4.9). The named
  pretrained foundation models are scoped out with a *quantified* ≈100%
  pretraining-overlap confound rather than an asserted one (R1.1, R2.a3, R3.1).
- **Validation of homology detection.** MMseqs2 alignment identity reproduces the
  effect metric-independently, and a length-robust containment index is added —
  which honestly flags one previously-clean dataset as borderline (R1.3, R2.a2,
  R3.4).
- **Bootstrap methodology.** A cluster (block) bootstrap over within-test
  near-duplicate components, with intraclass correlation, design effect, an
  analytic cluster-robust cross-check and a combined-source interval; it changes
  no significance verdict (R3.3).
- **Dataset characteristics.** Class balance is disclosed as curated, not natural,
  and a prevalence-aware imbalanced evaluation is added (R2.a1).
- **Hyperparameter choice.** A random-forest regularization path shows the effect
  is a property of the *default unregularized* forest — and, critically, that
  cross-validation on the as-shipped training set *selects* that configuration
  (R3.2, and see the Discussion).

---

## Reviewer 1

### R1.1 — "The evaluation uses only classical models; a trained deep network is needed, and the reach beyond this domain is unclear."

**(a) Deep model.** We added a **from-scratch 1D residual CNN** in the
Basset/DeepSEA lineage (one-hot input, no k-mer features, trained on each training
split only — no pretraining confound), with a **pre-registered** dropout×weight-decay
grid and a binding refutation condition pre-specified in a timestamped document in
the code release (`results/deep_preregistration.md`). The standard-practice
regularized reference cell (dropout 0.3, weight decay 1e-4) drops on **both** leaky
datasets — `human_nontata_promoters` **+0.043** (cluster-bootstrap CI [0.025, 0.061]),
`human_enhancers_ensembl` **+0.014** ([0.007, 0.021]), both excluding zero — so the
pre-registered refutation condition is **not** triggered. Nineteen of the 20 cells
drop with cluster CIs excluding zero; the sole exception is one nontata cell
(dropout 0.6 / wd 1e-4) where heavy dropout underfits, leaving no inflation to lose.
Per-cell single-run variance is non-trivial, so we read the mechanism from the
grid-level contrast, and the unregularized manipulation-check cell memorizes
hardest (ensembl graded gap **+0.222**, mirroring the unregularized RF). The effect
is therefore **not** a classical-model artifact. New **§4.9**; Methods **§3.8**.

We also report one pre-registered expectation that was **not** met: the
pre-registration called this a *dose-response* grid, and it is not one — the drop is
non-monotone in both dropout and weight decay. We state this rather than quietly
restating the grid, since a pre-registration reported selectively is worth nothing.

**(c) The deep model is not just affected — it is the model the leaky split
penalises.** An earlier draft of this revision declined to rank the CNN, because the
evidence was a single run whose value sat inside a nine-cell grid spanning
0.749–0.813 with the linear SVM's 0.7975 in the middle of it. We have since run the
replicated, interval-bearing comparison that situation called for (§4.4). Across
**five training seeds** the network scores 0.8028–0.8131 corrected (mean 0.8086,
sd 0.0040) against the SVM's 0.7975, and 0.8119–0.8359 as shipped. **On every seed it
ranks fifth on the leaky split and first on the corrected one.** Scoring both models
on the same corrected test set and differencing per example, the paired
cluster-bootstrap difference is **+0.0111, CI [0.0063, 0.0161]**, excluding zero — so
the comparison now satisfies the inversion criterion (§3.3) that the earlier draft
was unwilling to bypass.

So leakage does not only promote a memorizer; on this dataset it **demotes the
architecture family whose baselines this suite publishes**. We attach three
qualifications rather than leave them for a referee: four of the five seeds' paired
intervals exclude zero and the fifth does not (+0.0053, [−0.0006, 0.0109]), so the
advantage is uniform in rank but not in significance; the CNN was not in the
pre-registered nine-model roster, so this comparison is **post hoc** and labelled so;
and it is one dataset.

Two by-products of running it are worth reporting. First, we **verified rather than
assumed** that the deep-model and roster corrected splits are the same partition —
refitting the linear SVM on the deep-model split reproduces its published 0.7975
exactly. The manuscript had been comparing those two numbers without that check.
Second, the same replication **weakens** a number we had reported: the reference
cell's drop is 0.014 at seed 0 but ranges 0.0049–0.0269 across seeds, a five-fold
spread, and the smallest seed's interval includes zero. §4.9 now reports that range
and flags the +0.014 as one draw rather than the cell's value. The corrected accuracy
is stable to ±0.004 while the drop varies five-fold, so it is the as-shipped score
that moves.

**(b) Reach.** We now separate two claims explicitly (Introduction): **Claim I**
(de-leaking lowers accuracy) is credited to prior work across domains and
reproduced only as a within-suite control; **Claim II** (leakage can *reorder*
rankings) is our contribution, scoped to short human regulatory DNA plus a
memorization-prone model in the comparison set. The title carries that scope.

We also tested the reach **empirically**, on two further suites, rather than
arguing it. Applying the census unchanged to the **Nucleotide Transformer**
downstream tasks (§4.10) finds **three of eleven independent tasks leaky** and six
more borderline, leaving two clean — so the defect is not confined to Genomic
Benchmarks. We then pre-registered predictions for **GUE** before examining it, and
**they failed** (§4.10): we predicted its short fixed-length human regulatory tasks
(core promoter 70 bp, promoter 300 bp, TF binding 101 bp) would be leaky, and at
full scale **all eleven predicted-leaky tasks are clean** — near-duplicate fractions
0.009–0.041 by Jaccard, the largest under half the verdict cut. We count those
eleven conservatively as **seven** independent test partitions, since the `_all`
promoter test sets are exact unions of their own `notata` and `tata` test sets and
the 70 bp and 300 bp families are the same loci at two window widths.

We report that failure prominently because it is the **stronger** generalization
result. It refutes the natural reading of our own finding — that short human
regulatory DNA is inherently at risk — and replaces it with a sharper one:
**leakage tracks how a dataset was constructed, not what kind of task it is.**
Genomic Benchmarks' leaky positive class is an un-deduplicated assembly with
77,421 intervals on 40,934 coordinates; GUE's tasks, matching it on sequence type
and length, are clean because they were not built that way. That is a claim about
construction, and it is the one our manipulation experiment (§4.7) directly tests.

Two parts of the reviewer's question we have **not** answered and do not claim to:
**protein sequence modelling** and **long-range regulatory** modelling. Both are
outside this study, and we name them as extensions rather than implying coverage
(§5). We would rather concede those two than stretch DNA-suite evidence to cover
them.

### R1.2 — "The corrected ranking is treated as the true generalization ranking, but de-duplicating the training set also removes learnable signal."

Agreed, and we now **measure** rather than assert it. On `human_nontata_promoters`
the corrected and novel-only rankings diverge (τ = −0.67): the random forest is
demoted under the corrected split yet remains best on truly novel sequences, so its
advantage is partly genuine generalization. Single-linkage **cohesion** supplies the
biological reason (§4.13): ensembl's clusters are 99.6% pairwise with largest-cluster
cohesion 0.96 — clean discrete locus duplication, where correction is a clean fix —
whereas nontata's 420-member largest cluster has cohesion 0.083, i.e. heavy
transitive chaining that over-removes genuine promoter and gene-family signal. We
reframe both the corrected split and novel-only accuracy as explicit
near-duplicate-bounded proxies (Methods §3.5, §3.9) and name leave-one-chromosome-out
as the deployment-relevant control (§4.12). This is the core of the
nontata-as-cautionary-case framing (§4.3, §5).

We also disclose a decomposition that cuts the other way and belongs in the record
(§4.3): on ensembl, evaluated on the as-shipped split's *novel* stratum with no
re-splitting and no retraining, the forest already ranks below the linear SVM, but
by only 1.2 points; after re-splitting and refitting it sits ~10 points below, so the remaining ~9 points are a **retraining penalty**, because
de-leaking removes near-duplicate partners from the training set too. Both are
consequences of leakage, but they are different quantities and we now say so.

### R1.3 — "Validate leakage with an alignment-based measure and give threshold recommendations."

**Alignment:** we re-clustered both leaky datasets with **MMseqs2** alignment
identity, fed the external clusters into the *same* whole-cluster splitter (only the
edge metric changes), and refit all four models. The `human_enhancers_ensembl` effect
is metric-independent (RF drop **0.164** Jaccard ≈ **0.162** alignment; RF still rank
4); nontata is milder under alignment (0.108 → 0.064), reinforcing its partial status.
We describe this as "alignment-scored", not k-mer-independent, since MMseqs2 is
itself k-mer-seeded (**§4.12**, Methods **§3.6**).

**Thresholds:** the clustering-threshold sweep (0.5/0.7/0.9) changes no verdict
(Methods §3.5); the length-robust containment re-measure and the bimodal-vs-graded
similarity structure (cohesion 0.96 vs 0.083) provide the decision basis. The new
Conclusion (§6) states the resulting recommendations plainly: audit at full dataset
scale, report both a length-blind and a length-robust similarity, deduplicate in
coordinate space before splitting, and keep a memorization-prone learner in the
comparison set as a tripwire.

---

## Reviewer 2

### R2.a1 — "Class balance is not characterized, and imbalanced data are not tested."

We now disclose that every balanced dataset is **curated** to a 0.500 positive
fraction in both train and test (positive/negative subfolders); only nontata is a
natural mild imbalance (0.544) (Methods §3.1). Because balance is preprocessed, we
add a **prevalence stress test** (RF, π ∈ {0.5, 0.2, 0.1}) with prevalence-aware
metrics: the leakage inflation is **revealed by AUPRC and MCC on both leaky datasets, and
additionally by minority recall on `human_enhancers_ensembl`, but masked by accuracy on
both** (on `human_nontata_promoters` minority recall is flat, 0.673 -> 0.674, and we
say so in §4.13) (ensembl π=0.2: AUPRC 0.622 → 0.477, MCC 0.422 → 0.182, minority recall
0.258 → 0.070, while accuracy barely moves, 0.844 → 0.808). The splitter is extended
with a per-class realized-fraction check and preserves the target prevalence
(realized 0.5001 / 0.2007 / 0.1001). **§4.13**; Methods §3.9.

### R2.a2 — "The similarity measure and the features are both k-mer based, so the leakage finding may be circular."

Addressed by the alignment re-measure (R1.3): with alignment identity as the edge
metric and the same splitter, the ensembl RF drop is essentially unchanged
(0.164 vs 0.162) and the demotion persists. We add two further
metric-independent lines of evidence. First, ensembl's leakage is **38.0%
byte-identical** duplication — no similarity metric is involved in counting exact
copies. Second, and new in this revision, the **coordinate-space analysis** (§4.6)
identifies the defect in the shipped genomic intervals, which are causally upstream
of and statistically independent from the k-mer clustering: 77,421 positive intervals
sit on only 40,934 distinct coordinates, and every one of the 36,487 duplicated pairs
carries byte-identical sequences and a single label. That is a cross-modal
confirmation, not a restatement. **§4.12**, **§4.6**; Methods §3.4, §3.6, §3.10.

### R2.a3 — "Extend to deep models and to the multiclass setting."

**Deep models:** the CNN (R1.1a, §4.9), now seed-replicated with a paired interval
and shown to move from rank 5 to rank 1 under correction (R1.1c, §4.4); and the
multilayer perceptron in the nine-model roster (§4.4), whose drop also excludes zero
— the inflation reaches a neural learner that we had not designated
memorization-prone in advance.
**Multiclass:** the only multiclass set is clean, so we answer by construction,
injecting label-carrying near-duplicates at f ∈ {0, 0.1, 0.2, 0.4} into the clean
3-class `human_ensembl_regulatory` set. The RF drop grows monotonically with f
(+0.0004, +0.118, +0.179, +0.258) while LR stays flat (≤0.066), and the corrected
accuracy is stable (~0.58) because the split removes the injected leakage. **§4.13**;
Methods §3.9. We state explicitly that leakage is injected because the suite ships no
naturally leaky multiclass set.

One part of this comment we have **not** fully met, and would rather name than let
pass: the reviewer asks for deep models "in both binary and 3-class settings", and our
multiclass arm is **classical only** (random forest against logistic regression). The
CNN evidence is binary. We did not extend the injected-leakage construction to the CNN
because the mechanism there is imposed by us rather than measured, so a deep replicate
would test our injection procedure rather than the suite; the binary CNN result, which
is measured on shipped data, is the stronger evidence and is where we put the compute.
That is a judgement, not a claim of coverage, and a referee who disagrees is entitled
to ask for the run.

### R2.b1 — "Give a threshold-sensitivity analysis and concrete recommendations."

The verdict is robust to the clustering threshold (0.5/0.7/0.9 sweep, no change;
Methods §3.5) and to removal of the single largest component. We report the geometry
that drives the recommendation: the length-robust containment index (§4.1, §4.13) as
the metric to prefer when lengths vary, and single-linkage cohesion as the diagnostic
separating a clean discrete-duplication case (ensembl — one threshold suffices) from
a transitive-chaining case (nontata — treat as partial scope, avoid over-removal).
Report-card guidance and the length-cap disclosure are in Table 2; the consolidated
recommendations are in §6.

---

## Reviewer 3

### R3.1 — "Deep models are needed, specifically attention-based / foundation models (e.g. DNABERT-2, HyenaDNA)."

**(i) Trained deep net:** answered by the from-scratch 1D residual CNN (§4.9),
trained on the training split only, whose regularized reference cell drops on both
leaky datasets with cluster-bootstrap CIs excluding zero — the memorization mechanism
reaches a trained neural network with no pretraining confound. Since the first
revision we have strengthened this considerably: the CNN is now seed-replicated with a
paired cluster-bootstrap interval, and on `human_enhancers_ensembl` it ranks **fifth
on the leaky split and first on the corrected one, on every one of five seeds**
(paired difference +0.0111, CI [0.0063, 0.0161]; R1.1c, §4.4). The leaky split does
not merely fail to detect the deep model — it actively demotes it. We also attempted a
from-scratch attention (Transformer) encoder under the same protocol, and mention it
here only so the record is complete: **it is not in the manuscript and we rest no
claim on it.** It did not yield a usable comparison — on `human_nontata_promoters`
(251 bp) the corrected split slightly *raised* its accuracy (drop −0.016),
consistent with a data-hungry architecture underfitting a short, local-motif task,
and the `human_enhancers_ensembl` arm did not finish within our compute budget. We
therefore report the CNN as the trained-deep-net instrument and leave the
from-scratch-attention question open rather than resting an argument on one
incomplete run. Separately, the pretraining-contamination argument below justifies
excluding *pretrained* attention models, not attention architectures per se.

**(ii)/(iii) Named foundation models:** DNABERT-2, HyenaDNA and the Nucleotide
Transformer are cited and discussed but scoped **out** of the split-effect evidence,
and we now **quantify** the confound rather than assert it. The two leaky datasets'
sequences are extracted from the human reference genome — we recovered their hg38
chromosomal coordinates to build the chromosome-holdout control — so **≈100% of these
test sequences already lie inside HyenaDNA's whole-genome (hg38) pretraining corpus**
and within the human component of DNABERT-2's and the Nucleotide Transformer's
multispecies corpora. That is a second leakage channel no train/test re-split can
close, so fine-tuning them would yield a confounded lower bound rather than
split-effect evidence. We therefore leave the deployed-foundation-model question
**genuinely open** (§5) rather than answer it with a contaminated number. We note
this is not evasion but the same problem one level up: evaluating those models
properly requires a pretraining-overlap-aware protocol, which is itself an instance
of the leakage problem this paper studies.

### R3.2 — "The random forest runs at memorization-maximizing defaults; is the result just an untuned-RF artifact?"

This was the objection we took most seriously, and we answer it empirically.

First, the **regularization path** (min_samples_leaf ∈ {1…500}, max_depth ∈ {None…4},
at full scale, largest settings exceeding the largest near-duplicate cluster). At the
default the forest scores 1.000 on the ≥0.9-similarity bin with the full drop
(0.164 / 0.108) and graded gap (0.230 / 0.121); regularizing drives the drop to ≈0
(0.164 → −0.001; 0.108 → 0.000), collapses the graded gap, and the "RF wins on the
leaky split" phenomenon itself **requires** the unregularized forest — rank 1 flips to
rank 4 once min_samples_leaf ≥ 20. **§4.8**; Methods §3.7.

Second, and this is the direct answer: **a practitioner does not know the split is
leaky, and tuning does not rescue them.** We cross-validated the forest over
min_samples_leaf ∈ {1…500} on the as-shipped *training* set under two schemes —
ordinary stratified 5-fold (what a practitioner would actually run) and 5-fold
holding whole near-duplicate clusters out together (the honest version). We ran both
schemes on **both** leaky datasets.

On `human_enhancers_ensembl`, the dataset carrying the reordering claim, the two
procedures diverge and the consequence is stark. Ordinary cross-validation selects
**min_samples_leaf = 1**, the memorizing default, and the model it selects scores 0.860
as shipped but 0.696 once the leakage is removed — a loss of 16.4 points. Cluster-grouped
cross-validation selects **min_samples_leaf = 20**, and that model loses only 2.2 points
(0.751 → 0.729). So tuning does not rescue the practitioner: ordinary tuning actively
selects the configuration whose apparent advantage is almost entirely leakage. Worse, the
benchmark **punishes the correct methodology** — the leakage-aware tuner's model looks
10.8 points *worse* as shipped while being 3.3 points *better* corrected. On
`human_nontata_promoters` both procedures select leaf = 1, so our pre-registered
expectation P5 holds on `human_enhancers_ensembl` and ties, rather than reverses, on
`human_nontata_promoters`. Discussion §5.

The objection is therefore answered on the dataset where it matters: the untuned default
is not a careless choice a tuner would correct, it is *what ordinary tuning selects*.

Finally, we note the objection does not reach the core finding in any case, because
§4.7 shows the defect is a property of the **data**: editing coordinates alone, with
model code untouched, both removes the reordering and creates it.

### R3.3 — "Near-duplicates violate independence; a cluster/block bootstrap is the correct uncertainty estimate."

Implemented and reported in full. Resampling whole within-test near-duplicate
components as blocks widens the CIs by **1.0–1.8×** but **flips no verdict**: the RF
drop still excludes zero on both datasets (ensembl [0.155, 0.173]; nontata
[0.092, 0.126]) and the ensembl LR drop stays null. We now also report the diagnostics
the reviewer's request implies: the intraclass correlation is high (0.78–0.99) yet the
**design effect is small** (1.06–1.17), because the correlated near-duplicates are
test-to-**train**, not test-to-test — within-test clustering is only 1.06–1.19
sequences per component, a regime in which a sample-wise bootstrap of a fixed trained
model's test correctness is nearly unbiased. An analytic Liang–Zeger cluster-robust
standard error agrees with the block bootstrap, and a **combined-source interval**
folding in the five re-split-seed variance still excludes zero for both leaky RF drops
(ensembl [0.154, 0.174]; nontata [0.088, 0.129]). Every deep-model drop CI (§4.9) uses
this cluster bootstrap. **§4.13**; Methods §3.9.

Relatedly, because we report roughly a dozen leaky-versus-clean delta intervals, we
applied a **Benjamini–Hochberg correction** at q = 0.05: it leaves the two leaky RF
drops and the nontata LR drop significant and no clean or three-class delta
significant, so multiplicity changes no verdict either.

### R3.4 — "'Homology' is the wrong word for near-identical/duplicate leakage."

Agreed and adopted throughout: **near-duplicate leakage** is the primary term, the
splitter is **near-duplicate-aware**, and the title is changed. An exact-duplicate
census settles the matter metric-independently (`human_enhancers_ensembl` = 38.0%
byte-identical copies; `human_nontata_promoters` = 0 exact duplicates yet 22.5%
near-duplicates at Jaccard ≥ 0.9 — the more homology-like but statistically weaker
case). We keep "homology" only where the alignment or containment re-measure earns it
(Methods §3.4), and we downgrade the report card's green verdicts to "no
forward-strand near-duplicate leakage detected (length-cap disclosed)" (Table 2, §5),
since a clean verdict carries a heavier evidential burden than a leaky one.

---

## Associate Editor — "This cannot be addressed in a short time / new experiments are needed."

We agree it could not be addressed quickly, and did not try to. The revision adds the
two hard-gate experiments the panel expected — a trained deep model (§4.9) and
alignment-based validation (§4.12) — plus the genomics-standard **leave-one-chromosome-out
control** (§4.12), which independently reproduces the marquee `human_enhancers_ensembl`
demotion (RF rank 1 → 4, identical corrected order) and independently confirms
`human_nontata_promoters` as the partial case (RF stays rank 1). Beyond those, it adds
the nine-model roster, the construct-and-break manipulation, the coordinate-space
analysis, the cross-suite census with its honest null, the tuning-selection experiment,
and the diagnostic condition — six experiments that did not exist in the prior
submission.

We also closed three internal issues a careful reviewer would have flagged:

- the central claim is honestly rescoped to `human_enhancers_ensembl` (n=1 clean,
  partial on the second);
- the previously inconsistent headline drop numbers are reconciled to a single
  decision rule (`.predict()`/argmax) and estimator (5-seed mean + bootstrap CI),
  removing the earlier three-coexisting-values spread and the incorrect
  "thread-nondeterminism" wording — the true causes were a decision-rule
  inconsistency and a point-vs-mean estimator choice, and we say so;
- every pre-registered prediction is now scored in the text whether or not it held.
  Of the six, P1 **fails as registered** — the corrected-gap result that favours it was
  defined after P1 was committed and we label it post hoc rather than count it — P2 holds
  on one leaky dataset and fails on the other, P3 and P4 hold, P5 holds on
  `human_enhancers_ensembl` and ties on `human_nontata_promoters`, and P6 holds on all ten
  doses. We report the failures as prominently as the successes.
