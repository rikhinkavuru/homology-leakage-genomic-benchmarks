# Response to Reviewers — BIOADV-2026-296 (resubmission)

We thank the reviewers and the Associate Editor for the review; it materially
improved the paper. The decision turned on one question — *is the ranking result
general, or is it an artifact of this particular comparison?* — and the revision
answers it with experiments rather than argument.

**A note on section pointers.** The manuscript has been restructured onto a
construction-defect-first spine and shortened to the Original Article budget, so
its numbering differs from the prior submission's. Pointers below are to the
*current* manuscript: §N.M is the main text, §S-N.M the supplementary. A
concordance with the prior numbering is at the end of this document.

## What is new, and why it answers the decision

Five things carry the revision. Each targets the generality concern directly.

1. **The defect is located upstream of any model, in the shipped coordinates
   (§3.2, Fig. 1).** The leaky positive class ships 77,421 genomic intervals on
   40,934 distinct coordinates; 36,487 coordinates carry exactly two rows each,
   so 94.3% of positive rows sit on a duplicated coordinate. A coordinate screen
   with **no fitted parameter** — the 0.1 cut carried over unchanged from
   sequence space — separates seven of the suite's eight datasets. This evidence
   carries no sampling uncertainty and no model dependence.

2. **The construction step is edited in both directions (§3.3, Table 3).** The
   prior submission argued the cause was a curation defect; it did not test it.
   Applying the omitted merge step to `human_enhancers_ensembl` collapses its
   leak fraction 0.390 → 0.012 and the random-forest inflation +0.165
   [0.157, 0.173] → −0.008 [−0.019, 0.003], and the reordering disappears.
   Imposing the same defect on the clean `human_ocr_ensembl`, at matched size
   *and* balance, raises leakage 0.004 → 0.204 and inflation +0.002
   [−0.019, 0.025] → +0.105 [0.083, 0.126], and manufactures the reordering on
   demand (forest promoted to rank 1 on the contaminated split, demoted to rank 4
   by correction). Model code, split ratio and — in the fix-a-leaky direction —
   the sequences themselves are untouched. All four intervals are two-arm cluster
   bootstraps and the control and intervention intervals are disjoint in both
   directions. A ten-dose sweep of the same manipulation is in §S-2.2.

3. **A pre-registered cross-suite prediction, committed before the target suite
   was examined, and falsified (§3.12).** The prediction was that GUE's short
   fixed-length human regulatory tasks — core promoter 70 bp, promoter 300 bp,
   TF binding 101 bp — would be leaky. At full scale **all eleven are clean**,
   0.009–0.041 by Jaccard, the largest under half the verdict cut; the registered
   predictions score 3 of 15. This refutes the natural reading of the paper's own
   finding, that short human regulatory DNA is inherently at risk, and replaces it
   with the sharper claim the title now carries: leakage requires **an overlapping
   construction step together with a split that ignores genomic position.** The
   condition is conjunctive, and we have scoped it rather than stated it
   exclusively, because our own §4 documents the exception: GUE's *virus_covid*
   measures a near-duplicate fraction of 1.000, the largest we observed anywhere,
   and that is biology — nine SARS-CoV-2 variants differing by a handful of
   mutations, on which no split could be otherwise. Where similarity is a property
   of the sequences, a leak fraction carries no information about curation. The
   eleven are counted conservatively as seven independent test partitions, which
   bounds how much the null carries.

4. **A nine-model roster, and a from-scratch CNN as a tenth entrant (§3.6,
   Table S2).** The sharpest form of the objection is that a claim about
   "rankings" made over four models is a claim about nothing. The comparison now
   spans nine learners end to end in memorization propensity. The leaky split
   **cannot separate its top three at all** — MLP 0.8736, 1-NN 0.8729,
   extremely randomized trees 0.8721, within 0.0015, the paired interval on the
   top-two margin [−0.005, 0.006] including zero — **yet those three span 23
   accuracy points corrected** (0.768 / 0.537 / 0.628). 1-NN falls from rank 2 to
   rank 9, 0.873 → 0.537, which is 3.7 points above this balanced test set's base
   rate. The benchmark is not merely crowning the wrong model; it is blind to
   enormous real differences between models it reports as tied.

5. **An out-of-suite coordinate census at four orders of magnitude of interval
   length (§3.4).** Because the screen needs only shipped coordinates it runs
   outside the suite. Applied unchanged to BEND, it returns 0.000 at ≥50%
   reciprocal overlap on chromatin accessibility (1,410,554 train against 372,153
   test intervals), 0.000 on histone modification, and 0.000 on enhancer
   annotation in all ten folds — the last at **100,096 bp per sample**.

Three structural changes: **near-duplicate leakage** is adopted as the primary
term and the title is changed (R3.4); a Conclusion is added (§5); and the pinned
software environment is reported (§S-1.8).

Three corrections to the prior submission are stated in their own right below —
on what dedup removes (R1.2), on the pretraining-overlap argument (R3.1), and on
the design effect (R3.3). Each replaces a claim that was wrong with a measured
one.

---

## Associate Editor (Prof. Shanfeng Zhu)

- **Limited model scope.** Answered three times over: the nine-model roster
  (§3.6, Table S2); a from-scratch 1D residual CNN with a pre-registered
  dropout×weight-decay grid and a binding refutation condition (§3.9, §S-1.4);
  and a multiclass CNN arm (§3.9, Table S5). The named pretrained models remain
  scoped out, but on a **measured** confound with a stated exception rather than
  an asserted one (R3.1).
- **Validation of homology detection.** MMseqs2 alignment-scored re-clustering
  fed into the same whole-cluster path (§S-1.5, §S-3), plus a length-robust
  containment index that moves one previously-clean verdict to borderline
  (§3.1, Table 1).
- **Bootstrap methodology.** A cluster (block) bootstrap over whole within-test
  near-duplicate components, with intraclass correlation, Kish design effects,
  a directly measured variance inflation, an analytic cluster-robust cross-check
  and a combined-source interval (§3.13). It flips no verdict — and the
  dependence it measures is *larger* than the prior submission reported (R3.3).
- **Dataset characteristics.** Class balance is disclosed as curated, not
  natural (§2.1, Table S1), and a prevalence stress test is added (§3.13, §S-5).
- **Hyperparameter choice.** A regularization path (§3.8) plus the experiment the
  objection really demands: ordinary cross-validation on the as-shipped training
  set **selects** the memorizing configuration (§3.10).

---

## Reviewer 1

### R1.1 — "The evaluation uses only classical models; a trained deep network is needed, and the reach beyond this domain is unclear."

**(a) The trained deep model.** A from-scratch 1D residual CNN in the
Basset/DeepSEA lineage — one-hot input, no *k*-mer features, trained on each
training split only, so no pretraining confound — with a pre-registered
dropout×weight-decay grid and a refutation condition pre-specified in a
document committed alongside the results it predicts
(`results/deep_preregistration.md`). We state plainly, here and in the
manuscript, that this is a binding stated condition rather than an externally
timestamped registration.
The regularized reference cell drops on **both** leaky datasets:
`human_nontata_promoters` +0.043 [0.025, 0.061] and `human_enhancers_ensembl`
+0.014 [0.007, 0.021], both cluster-bootstrap intervals excluding zero, so the
refutation condition — which requires the interval to include zero on both — is
not triggered. Nineteen of the 20 cells drop with intervals excluding zero; the
exception is the nontata dropout-0.6 cell, where heavy dropout underfits and
accuracy *rises* 0.020. §3.9; methods in §S-1.4.

Three disclosures travel with that result and are in the manuscript, not only
here. One `human_enhancers_ensembl` seed does include zero. `human_nontata_promoters`,
which carries the larger drop and is the more dispersed of the two across the
grid, was **not** seed-replicated, so that verdict rests on an unreplicated cell
on the dataset from which §3.7 scopes the mechanism *away*. And one registered
expectation was not met: the pre-registration named this a *dose-response* grid,
and it is not one — the drop is non-monotone in both dropout and weight decay.

**(b) The deep model is not merely affected; it is the model the leaky split
penalises.** Across **five training seeds** the network scores 0.8028–0.8131
corrected (mean 0.8086, sd 0.0040) against the linear SVM's 0.7975, and
0.8119–0.8359 as shipped. **On every seed it ranks fifth of ten on the leaky
split and first on the corrected one.** Both arms of the inversion criterion
clear zero: as-shipped MLP−CNN +0.0500 [0.0320, 0.0679], corrected CNN−MLP
+0.0402 [0.0309, 0.0495]. Against the harder comparator the criterion does not
require, `LinearSVC`, the paired difference is +0.0111 with a combined-source
interval of [0.0019, 0.0204], folding the seed-to-seed sd (0.0040) into the
cluster-bootstrap sd (0.0026) in quadrature, because the quantity of interest is
the advantage of *one* trained network rather than of an average over five.
Bootstrapping seed-averaged per-example correctness instead gives
[0.0063, 0.0161] — narrower than any individual seed's interval, because
averaging removes the training variance rather than estimating it — and we do
not quote that as the result. Four of the five individual seeds' paired intervals
exclude zero and the fifth does not (+0.0053, [−0.0006, 0.0109]), so the
advantage is uniform in rank but not in significance; the CNN was not in the
pre-registered nine-model roster, so the comparison is **post hoc** and labelled
so; and it is one dataset. The reference cell is the *maximum* of the nine-cell
grid, whose corrected accuracies span 0.749–0.813 with 0.7975 inside — five cells
above, four below — so no conclusion is drawn from that cell alone; the claim
rests on seed replication of one pre-specified cell against a paired interval.
§3.6.

The same replication **weakened** a number the prior submission reported: the
reference cell's drop is 0.014 at seed 0 but ranges 0.0049–0.0269 across seeds, a
five-fold spread whose smallest does not exclude zero. §3.9 now reports the range
and treats +0.014 as one draw. Corrected accuracy is stable to ±0.004 while the
drop varies five-fold, so it is the as-shipped score that moves.

**(c) A from-scratch attention encoder was also attempted, and is reported here
because the record should be complete.** It is not in the manuscript and no claim
rests on it. On `human_nontata_promoters` (251 bp) the corrected split *raised*
its accuracy: drop **−0.0163, cluster-bootstrap CI [−0.0292, −0.0041]**, which
**excludes zero** and is therefore a significant effect in the direction contrary
to the CNN's. The `human_enhancers_ensembl` arm did not finish within the compute
budget, so there is no second dataset to read it against. Two things are worth
saying plainly rather than filing it as inconclusive. First, the dataset it ran
on is precisely the one already scoped as the cautionary case, where the
prevalence-corrected graded gap does not separate model families and the
demotion is accuracy-only, so a contrary sign there is the weakest possible
position from which to generalize. Second, the sign is what a data-hungry
architecture underfitting a 251 bp local-motif task would produce, and the
CNN grid contains an instance of the same behaviour — its dropout-0.6 nontata
cell also rises, for the same reason. On one dataset, with one architecture, and
with the paired arm missing, that is a hypothesis rather than a result. The CNN
is therefore the trained-deep-net instrument, and the from-scratch-attention
question is left open. Separately, the argument at R3.1 excludes *pretrained*
attention models, not attention architectures.

**(d) Reach.** Two claims are separated explicitly in the Introduction. **Claim
I** — de-leaking lowers accuracy — is credited to prior work across domains and
reproduced only as a within-suite control, whose contribution is the matched
clean control on which the same procedure moves nothing. **Claim II** — leakage
can *reorder* rankings — is the contribution, scoped to short human regulatory
DNA plus a memorization-prone learner in the comparison set. The title carries
the scope.

Reach was then tested empirically on three further suites. The Nucleotide
Transformer census (§3.11) finds three of eleven independent tasks leaky and six
more borderline, leaving two clean, so the defect is not confined to Genomic
Benchmarks. GUE falsified the pre-registered prediction (§3.12, above). And the
coordinate screen runs on **BEND** (§3.4), which is the direct answer to the
long-range half of this comment: the enhancer-annotation task is **100,096 bp per
sample**, four orders of magnitude beyond anything audited in sequence space
here, and it is clean in all ten folds — 0.000 at ≥50% reciprocal overlap. Gene
finding, the one BEND task partitioned at 80% identity rather than by chromosome,
reproduces the `drosophila_enhancers_stark` pattern: 0.183 of its 597 test
intervals touch a training interval at ≥1 bp, yet only 0.002 reach ≥50% and none
90%. This is the only prospective construction-based call in the paper, and it is
near-tautological on the three chromosome-split partitions — stated so rather
than counted as a strong prediction.

**Protein sequence modelling remains outside this study** and is named as an
extension rather than implied as coverage (§4). We would rather concede it than
stretch DNA-suite evidence over it.

### R1.2 — "The corrected ranking is treated as the true generalization ranking, but de-duplicating the training set also removes learnable signal."

**The reviewer's concern is sound and the prior submission's answer to it was
wrong.** That answer attributed `human_nontata_promoters`'s low cluster cohesion
(0.083) to transitive chaining that "over-removes genuine promoter and gene-family
signal." Measurement contradicts it, and the claim is deleted at all three sites
where it appeared.

What the largest cluster actually is: its 420 members are **420/420 negative
class** — non-promoters — and they are **a single locus**, 251 bp windows tiling
1,421 bp at a median 2 bp step, with 62% of member pairs offset by ≥251 bp and
sharing no sequence at all. The low cohesion is **window geometry, not chained
paralogy**. Dataset-wide the same picture holds: 94.3% of negatives sit in
multi-member components against 0.4% of positives, and 2,808 of 2,811 components
are class-pure — the same fact §3.2 reports as R_self = 0.961 in the negative
class. Nothing here is gene-family signal being over-removed; it is one
over-tiled negative region. §3.13; §S-3.1.

The consequence for the reviewer's question is that the *cautionary* verdict on
`human_nontata_promoters` is unchanged while its *reason* is replaced. It now
rests on five independent signals, all accuracy-only (§4): overlapping corrected
CIs (RF [0.812, 0.828], LR [0.821, 0.837]), the AUROC/F1 reversal (the forest
stays rank 1, corrected AUROC 0.929, the highest of the four), the chromosome
holdout (the forest stays rank 1, 0.9284 → 0.8199), the novel-only rank
correlation, and the milder alignment-scored drop (0.108 → 0.064). Because the
clusters are the geometric case rather than the duplication case, whole-cluster
correction is a **coarser instrument** there, and the certification tool returns
*leaky-but-correction-unsafe* rather than *leaky* below cohesion 0.5, directing
the user to the novel-stratum readout instead of a re-split.

The general point the reviewer raises is met by reporting the decomposition
rather than assuming the corrected split is truth. On
`human_enhancers_ensembl`, evaluated on the as-shipped split's *novel* stratum
with no re-splitting and no retraining, the forest already ranks below the linear
SVM (0.770 against 0.782) — but by only 1.2 points. The remaining 10 points is a
**retraining penalty**: de-leaking removes near-duplicate partners from the
training set too, which costs the memorizer more than the others (residual
−0.074 against +0.012 to +0.016). Both are consequences of leakage, but they are
different quantities and the manuscript now separates them (§3.6). Corrected and
novel-only accuracy are both framed as explicit near-duplicate-bounded proxies
(§2.4, §S-1.6), and leave-one-chromosome-out is named as the deployment-relevant
control (§3.13) — with its own two disclosures, that its nontata test set is
74.33% positive against a 47.20% training prevalence and that it is scored on
accuracy, which the imbalance panel shows understates the correction.

### R1.3 — "Validate leakage with an alignment-based measure and give threshold recommendations."

**Alignment.** Both leaky datasets were re-clustered with **MMseqs2**
`easy-cluster` (`--min-seq-id 0.7`, `-c 0.8`) and the external cluster vector fed
into the *same* whole-cluster assignment path, with all four models refit — only
the edge metric and linkage change. The `human_enhancers_ensembl` effect is
reproduced under an independent clustering engine (RF drop 0.164 under 8-mer
Jaccard against 0.162 under alignment identity; the forest still demoted to rank
4). On `human_nontata_promoters` the drop is milder (0.108 → 0.064), reinforcing
its partial status. Two properties are stated rather than glossed: `mmseqs
cluster` defaults to a cascaded greedy set-cover whereas the main splitter uses
connected components, so the arm varies **linkage alongside metric** — on nontata
MMseqs2 returned a *finer* partition (23,758 clusters against 23,312) despite the
looser identity criterion, which is a linkage signature rather than a metric one
— and `--min-seq-id 0.7` (≈70% identity) is **not** calibrated to an 8-mer
Jaccard of 0.7 (≈97–98% identity) despite the shared numeral. The arm is
described as **alignment-scored**, not *k*-mer-independent, since MMseqs2 is
itself *k*-mer-seeded. §S-1.5, §S-3; summarized at §3.13.

**Thresholds.** The clustering-threshold sweep (0.5 / 0.7 / 0.9) changes no
verdict, but it is reported with its actual structure rather than as a flat null,
because the two datasets behave differently: the drop is flat on
`human_enhancers_ensembl` (0.156 / 0.156 / 0.157; 0.158 under largest-component
removal) and **scales strongly** on `human_nontata_promoters` (0.205 / 0.121 /
0.006), since 0.9 leaves its large [0.5, 0.9) band uncorrected (§2.4). That
contrast is itself the recommendation: threshold choice matters exactly where
redundancy is graded rather than discrete. Detector calibration (§S-4) makes the
same point on the measurement side — at 70 bp the 8-mer Jaccard at 0.7 misses two
point mutations 70% of the time where at ≥300 bp it tolerates five, so it is a
near-exact-duplicate detector on short sequences and a homology detector on long
ones, and the threshold should follow length. The Conclusion (§5) states the four
resulting practices plainly.

---

## Reviewer 2

### R2.1 (a1) — "Class balance is not characterized, and imbalanced data are not tested."

Every balanced dataset is disclosed as **curated** to a 0.500 positive fraction in
both train and test (positive/negative subfolders); only nontata carries a
natural mild imbalance, 0.544 (§2.1, Table S1). Because balance is preprocessed,
a **prevalence stress test** is added (random forest, π ∈ {0.5, 0.2, 0.1}) with
prevalence-aware metrics. The inflation is revealed by AUPRC and MCC on both
leaky datasets and masked by accuracy on both: on `human_enhancers_ensembl` at
π = 0.2, correction lowers AUPRC 0.622 → 0.477, MCC 0.422 → 0.182 and minority
recall 0.258 → 0.070 while accuracy barely moves, 0.844 → 0.808; on
`human_nontata_promoters` AUPRC falls 0.949 → 0.813 and MCC 0.785 → 0.682 against
accuracy's 0.934 → 0.903 — but **minority recall is flat there** (0.673 → 0.674),
so the recall collapse does not replicate and the manuscript says so. The
splitter carries a per-class realized-fraction check and preserves the target
prevalence (realized 0.5001 / 0.2007 / 0.1001). Two limits bound the panel: it is
random-forest-only and its prevalence is induced by downsampling rather than
native, so **it does not test the reordering claim under imbalance**. §3.13;
§S-1.6, §S-5.

### R2.2 (a2) — "The similarity measure and the features are both k-mer based, so the leakage finding may be circular."

Three independent lines answer this, of which the third is new and is the
strongest.

First, the alignment re-measure (R1.3): with alignment identity as the edge
metric and the same splitter, the ensembl RF drop is essentially unchanged
(0.164 against 0.162) and the demotion persists.

Second, exact duplication, counted by string equality on a code path using no
*k*-mers: `human_enhancers_ensembl` has 11,774 byte-identical test sequences,
0.380 of its test set. Counting exact copies involves no similarity metric at
all. The same code path independently confirms the GUE verdicts (exact-duplicate
rate 0.000–0.026 across the eleven) and the Nucleotide Transformer `enhancers`
finding (25.0% byte-identical, 100 of 400 test sequences at label concordance
1.000).

Third, and structurally the answer to the circularity concern, the
**coordinate-space analysis** (§3.2). The defect is identified in the shipped
genomic intervals, which are read on a separate code path from the *k*-mer
clustering and are causally **upstream** of it: 77,421 positive intervals on
40,934 distinct coordinates, with 36,487 coordinates carrying exactly two rows.
The screen thresholds the maximum-over-classes ≥50%-reciprocal-overlap statistic
at 0.1 — the cut already in use in sequence space — so it introduces **no fitted
parameter**, and it separates 7 of 8 datasets. The one miss,
`demo_coding_vs_intergenomic_seqs`, is disclosed as structural rather than
measured: one of its classes is keyed by transcript accession rather than by
interval, so its 0.006 is structurally zero. This is cross-modal confirmation,
not a restatement.

The recovered coordinates are themselves validated rather than trusted (§2.2):
recovered interval width equals sequence length for 100% of rows in all eight
datasets — a check informative only for the four variable-length ones, and stated
as such.

### R2.3 (a3) — "Extend to deep models and to the multiclass setting."

**This comment is now answered in full, in both settings. The prior submission's
concession on the multiclass half is withdrawn and replaced by the run.**

*Binary deep:* the CNN (R1.1), seed-replicated, moving rank 5 → rank 1 under
correction on every one of five seeds, with both inversion arms clearing zero;
plus the multilayer perceptron in the nine-model roster, whose drop also excludes
zero. The MLP was **not** designated memorization-prone in advance, so the
inflation demonstrably reaches a neural learner the pre-registration did not
predict it would.

*Multiclass deep:* the suite ships no naturally leaky multiclass set, so the
question is answered by construction — label-carrying near-duplicates injected at
f ∈ {0, 0.1, 0.2, 0.4} into the clean 3-class `human_ensembl_regulatory` set,
giving realised test-set leak fractions φ = 0.29, 0.44 and 0.62. The CNN arm runs
**three seeds per dose** on the identical construction (§3.9, Table S5).

The result is a **null, and it is an asset rather than a gap.** The CNN shows no
dose-response: mean drops **−0.0386 (sd 0.0239), −0.0182 (0.0296), −0.0249
(0.0325), −0.0089 (0.0362)** at f = 0, 0.1, 0.2, 0.4 — every mean negative, and
the seed standard deviation equalling or exceeding any dose effect. The random
forest on the *identical* construction drops **+0.0004, +0.1177, +0.1791,
+0.2581**, monotone in f. Logistic regression is not untouched either — it drops
0.0158, 0.0323, 0.0661, monotone, its as-shipped accuracy climbing 0.591 → 0.669
as it exploits the injected copies — but the forest's advantage over it is 7.4×
to 3.9×, narrowing as the dose rises.

That contrast is a direct test of the paper's own mechanism claim, and it passes.
The label-carrying copies are available to **every** learner at every dose; the
differential is a property of *which learner exploits them*. A regularized
convolutional network does not, while a default forest exploits them roughly
twentyfold at low dose. This is what "**memorization propensity, not capacity**"
predicts, and the same claim is corroborated in the binary setting by
HistGradientBoosting — nominally higher-capacity than the forest — being barely
inflated where the forest is heavily inflated (§4). The multiclass mechanism
therefore generalizes **for the classical memorizer and not for the regularized
network**, and the manuscript states it in exactly that form.

All three seeds are reported. **Seed 1 underfits** — its as-shipped accuracy
averages 0.6856 against 0.7552 and 0.7534 for the other two — and it is
**retained rather than excluded**, since dropping the seed that disagrees would
make the null a selection artifact. Each cell is a single run, which is stated.

The four practices in the Conclusion transfer unchanged to multiclass, with the
addition that per-class prevalence should be checked after splitting.

### R2.4 (b1) — "Give a threshold-sensitivity analysis and concrete recommendations."

Threshold sensitivity is at R1.3 above, reported with its real structure — flat
on `human_enhancers_ensembl`, strongly scaling on `human_nontata_promoters` —
rather than as a uniform null, plus the length-dependent detector calibration
(§S-4) and robustness to removal of the single largest component.

The recommendations follow the geometry rather than a preference. The
**length-robust containment index** is the metric to prefer when lengths vary: an
8-mer Jaccard is bounded by the length ratio and undercounts leakage — across
`human_enhancers_ensembl`'s test–train pairs, 61% have their Jaccard held below
0.7 by length ratio alone — while unfloored containment runs the other way and is
an *upper* bound on variable-length data. Reporting both is what moved
`demo_coding_vs_intergenomic_seqs` from clean to borderline. **Single-linkage
cohesion** is the diagnostic separating the discrete-duplication case (ensembl,
99.6% pairwise at cohesion 0.96 — whole-cluster correction is a clean fix) from
the geometric tiling case (nontata, cohesion 0.083 — treat as partial scope and
use the novel-stratum readout), and the certification tool returns
*leaky-but-correction-unsafe* below cohesion 0.5.

Across task types, the Conclusion (§5) states four practices: audit at **full
dataset scale**, since subsampling can make a heavily leaky dataset look clean —
a subsample is admissible to confirm cleanliness, never to establish it; report
**both** a length-blind and a length-robust similarity, letting the threshold
follow sequence length; **partition in coordinate space before splitting**, with
the two limits that coordinates may escalate a verdict and never clear one and
that an exact-coordinate collapse is a no-op where redundancy is in overlapping
windows rather than repeated rows; and include at least one **memorization-prone
learner** as a tripwire, which is costlier than the as-shipped screens because it
requires building the corrected split. Report-card guidance and the length-cap
disclosure are in Table 1.

---

## Reviewer 3

### R3.1 — "Deep models are needed, specifically attention-based / foundation models (e.g. DNABERT-2, HyenaDNA)."

**(i) Trained deep net:** the from-scratch 1D residual CNN (§3.9), trained on the
training split only, with cluster-bootstrap intervals excluding zero on both
leaky datasets, and seed-replicated to a rank 5 → rank 1 inversion on every seed
(§3.6). The leaky split does not merely fail to detect the deep model; it
actively demotes it. The three qualifications are repeated at R1.1 rather than
left to a cross-reference. The from-scratch **attention** encoder attempted under
the same protocol is disclosed at R1.1(c), including that its one completed arm
is significant in the *contrary* direction; it is not in the manuscript and no
claim rests on it. Nothing below excludes attention architectures — only
*pretrained* ones.

**(ii)/(iii) The named foundation models: a correction.** DNABERT-2, HyenaDNA and
the Nucleotide Transformer remain scoped **out** of the split-effect evidence.
The prior submission justified that with a claim that pretraining overlap is
"approximately 100%, not a quantity we must estimate." **That claim is false, and
the manuscript now measures what it previously asserted.**

HyenaDNA inherits the Enformer interval set and designates **chromosomes 14 and
X as its held-out pretraining test chromosomes**. Both are present in these
datasets, carrying 6.6% of `human_enhancers_ensembl`'s test set and 15.3% of
`human_nontata_promoters`'s — the latter **unevenly by class**, 24.6% of test
negatives against 7.5% of positives. Pretraining overlap for that model is
therefore bounded at **≤93.4% and ≤84.7%**, not ≈100%. For DNABERT-2 and the
Nucleotide Transformer, whose human component is one part of a multispecies
corpus, the argument is explicitly one from provenance and is labelled as such.

The assembly underlying that argument is likewise now **measured, not asserted**
(§2.2). All 190,973 shipped intervals of the two leaky datasets name a primary
chromosome and none an alternate contig. For 101 test intervals stratified over
both datasets, classes and strands, the reference was fetched at the shipped
coordinates from both GRCh38 and GRCh37 via the UCSC REST API, reading intervals
as 0-based half-open and reverse-complementing minus-strand records: **GRCh38
reproduces the shipped sequence in 101 of 101 cases, GRCh37 in 1 of 48, and a
1-based reading in none.** Independently and without sampling, 915 of the 190,973
intervals (0.48%) end past a GRCh37 chromosome boundary and **none** past a
GRCh38 one.

Two consequences follow, and both are stronger than the original absolute claim.
First, the exclusion is a refusal on **validity** grounds and not a compute
limitation: the pretraining corpus has already seen the great majority of these
test sequences, so a fine-tuned number would carry a confound nameable in advance
and unremovable by re-splitting, and publishing it carefully captioned would be
worse than declining, because it would enter the literature as a measurement.
Second — and this is what the correction buys — **chromosomes 14 and X are a
stratum HyenaDNA provably never saw, and are the correct confound-free protocol
for a future pretrained-model evaluation.** These datasets already contain enough
of that stratum to carry an interval. A second route is priced but not performed:
apply break-a-clean to `drosophila_enhancers_stark` and fine-tune HyenaDNA on
both arms under both splits, manufacturing the contamination inside a genome the
model has never seen — 6,914 sequences of at most 3,237 bp and twelve runs of
`hyenadna-small-32k`.

One logical point cuts in the paper's favour and is stated rather than left
implicit: pretraining is label-free and common to both split arms, so it
*attenuates* the as-shipped-minus-corrected difference rather than inventing it,
and a positive finding under contamination would be conservative evidence.
Whether deployed foundation models inherit this inflation on human benchmarks is
left open (§4, "Pretrained models and the second leakage channel"), which is a
titled paragraph so that a reader checking this specific answer need not hunt for
it inside a limitations list.

### R3.2 — "The random forest runs at memorization-maximizing defaults; is the result just an untuned-RF artifact?"

This was the objection taken most seriously, and it is answered empirically in
two steps.

First, the **regularization path** (`min_samples_leaf` ∈ {1…500}, `max_depth` ∈
{None…4}, at full scale, the largest settings exceeding the largest near-duplicate
cluster, so no leaf can be a pure duplicate block). At the default the forest
scores exactly 1.000 on the ≥0.9-similarity bin with drops of 0.164 and 0.108 and
graded gaps 0.230 and 0.121, at rank 1. At leaf size 500 the drop is ≈0
(0.164 → −0.001; 0.108 → 0.000) and the gap collapses (0.230 → 0.020;
0.121 → 0.061) — monotonically but for one 0.0012 reversal, and non-monotonically
along the depth axis on nontata, where the gap rises to 0.224 at depth 16 before
falling to 0.161. The "forest wins on the leaky split" phenomenon itself
**requires** the unregularized forest: rank 1 has become rank 4 by
`min_samples_leaf` ≥ 20 on both datasets. Regularization removes the inflation
without recovering the accuracy the inflation concealed — corrected nontata
accuracy falls 0.820 → 0.785 along the same path — so the two interventions are
not interchangeable. §3.8.

Second, and this is the direct answer: **a practitioner does not know the split is
leaky, and tuning does not rescue them.** Cross-validating the leaf grid on the
as-shipped *training* set under two schemes — ordinary stratified 5-fold, which is
what a practitioner runs, and 5-fold holding whole near-duplicate clusters out
together — on both leaky datasets: on `human_enhancers_ensembl`, the dataset
carrying the reordering claim, ordinary cross-validation selects
`min_samples_leaf` = 1, the memorizing default, and the model it selects scores
0.860 as shipped but 0.696 corrected, a loss of **16.4 points**. Cluster-grouped
cross-validation selects 20, and that model loses **2.2** (0.751 → 0.729). Worse,
the benchmark **punishes the correct methodology**: the leakage-aware tuner's
model appears 10.9 points *worse* as shipped (0.751 against 0.860) while being
3.3 points *better* corrected (0.729 against 0.696) — a point-estimate comparison
to which no interval is attached. On `human_nontata_promoters` both procedures
select leaf = 1, so registered prediction P5 holds on the first dataset and
**ties, rather than reverses**, on the second. §3.10.

The objection is therefore answered where it matters: the untuned default is not
a careless choice a tuner would correct, it is what ordinary model selection
selects, and the benchmark rewards it. It also does not reach the core finding in
any case, since §3.3 shows the defect is a property of the **data** — editing
coordinates alone, with model code untouched, both removes the reordering and
creates it.

### R3.3 — "Near-duplicates violate independence; a cluster/block bootstrap is the correct uncertainty estimate."

Implemented throughout: whole within-test near-duplicate components are resampled
as blocks, and every deep-model and manipulation interval in the paper uses it.
It widens the intervals 1.2–1.8× and **flips no verdict** — the forest's drop
still excludes zero on both datasets (`human_enhancers_ensembl` [0.155, 0.173];
`human_nontata_promoters` [0.092, 0.126]), a combined-source interval folding in
the five re-split-seed variance still excludes zero for both, an analytic
Liang–Zeger cluster-robust standard error agrees, and Benjamini–Hochberg over the
declared 15-delta family leaves the same set significant.

**A correction, because the prior submission got the diagnostic wrong in the
flattering direction.** It reported the design effect as "small (1.06–1.17)" and
attributed the small movement to the near-duplicates being test-to-**train**
rather than test-to-test. That statistic was computed on the **original arm
only**, which is the wrong arm. The splitter assigns whole components to one
side, so every near-duplicate relation *surviving correction* is test-to-test by
construction, and **the corrected arm is the more clustered one**: 43.4% of
corrected `human_nontata_promoters` test sequences lie in a multi-member
component against 25.2% as shipped, and 47.7% of `human_enhancers_ensembl`'s
against 10.1%.

The corrected figures are these. Component sizes are heavy-tailed, so Kish's m*
is 4.45 and 2.58 where the *mean* size is 1.55 and 1.33, giving design effects of
**2.80 and 2.58** for the forest where the mean-size convention returns 1.29 and
1.33. The inflation is also measured directly rather than modelled: on the
corrected arm the cluster-bootstrap variance of the forest's accuracy exceeds its
sample-bootstrap variance by **3.96× and 2.12×** (3.10× and 1.78× for logistic
regression). The intraclass correlation is high (0.78–0.99) across the four
fits, so the design effect is governed by the block-size distribution rather than
by the within-block correlation.

**Every reported confidence interval is nonetheless correct as computed**, and
the reason is worth stating: the block bootstrap *resamples* those components
rather than modelling them, so the dependence is already inside every interval.
What the prior submission got wrong was the description of the dependence, not
the arithmetic that accommodates it. The dependence is real, larger than a
mean-size design effect suggests, and concentrated in the arm the correction
creates. §3.13.

One source no test-set resampling reaches is the fit, which is held fixed, and
that is stated too.

### R3.4 — "'Homology' is the wrong word for near-identical/duplicate leakage."

Adopted throughout: **near-duplicate leakage** is the primary term, the splitter
is **near-duplicate-aware**, and the title is changed. An exact-duplicate census
settles the matter metric-independently: `human_enhancers_ensembl` has 11,774
byte-identical test sequences, 0.380 of its test set, whereas
`human_nontata_promoters` has **zero** exact duplicates yet 0.225 near-duplicates
at Jaccard ≥ 0.9 — the more homology-like but statistically weaker case, and one
more reason it is the cautionary rather than the demonstrative dataset.
"Homology" is retained only where the alignment or containment re-measure earns
it, and the report card's green verdicts are downgraded to "no forward-strand
near-duplicate leakage detected," subject to the disclosed length-pair cap
(Table 1), since a clean verdict carries a heavier evidential burden than a leaky
one. The Discussion adds the case where the detector is right and the word would
still mislead: GUE's `virus_covid` measures a near-duplicate fraction of 1.000,
the largest observed anywhere, and it is **not** a curation defect — nine-way
SARS-CoV-2 variant classification over 999 bp windows of a ~30 kb genome whose
variants differ by a handful of mutations has a test-to-train similarity
*minimum* of 0.824, so no split could be otherwise (§4).

---

## Associate Editor — "This cannot be addressed in a short time / new experiments are needed."

Agreed, and it was not attempted quickly. The two hard-gate experiments are here
— a trained deep model (§3.9) and alignment-based validation (§S-3) — plus the
genomics-standard leave-one-chromosome-out control (§3.13), which independently
reproduces the `human_enhancers_ensembl` demotion (forest rank 1 → 4, identical
corrected order) and independently confirms `human_nontata_promoters` as the
partial case (forest stays rank 1). Beyond those: the coordinate-space analysis
and its parameter-free screen, the two-directional construct-and-break
manipulation with cluster-bootstrap intervals on all four arms, the ten-dose
sweep, the out-of-suite BEND census at 100,096 bp, the nine-model roster, the
Nucleotide Transformer census, the falsified GUE pre-registration, the
tuning-selection experiment, the multiclass CNN arm, and the assembly
verification against GRCh38 and GRCh37.

Four internal issues were also closed, all in the direction of claiming less:

- The central claim is rescoped: `human_enhancers_ensembl` is an existence proof
  holding under accuracy, AUROC, F1, a bootstrap winner probability, a
  chromosome-holdout control and an alignment-scored re-clustering;
  `human_nontata_promoters` is a cautionary partial case with five independent
  signals showing its demotion is accuracy-only.
- The previously inconsistent headline drop numbers are reconciled to a single
  decision rule (`.predict()`/argmax) and estimator (five-seed mean with a stated
  interval). The earlier "thread-nondeterminism" wording was wrong; the true
  causes were a decision-rule inconsistency and a point-versus-mean estimator
  choice, and the manuscript says so.
- Where a point estimate and its interval are computed on different estimands,
  that is disclosed rather than smoothed — the `human_nontata_promoters` headline
  is the five-seed mean, 11.8 points, while the combined-source interval is
  computed on the seed-0 partition where the delta is 10.8, giving [8.8, 12.9].
- All six registered predictions are scored in the text whether or not they held.
  **P1 fails as registered** — the corrected-gap reading that favours it was
  defined after P1 was committed and is labelled *post hoc* rather than counted;
  P2 holds on `human_enhancers_ensembl` and fails on `human_nontata_promoters`;
  P3 and P4 hold; P5 holds on the first and ties on the second; P6 is right on
  all ten doses, with four limits stated in §S-2.2 including the sharpest one,
  that a much simpler rule — "the ranking inverts exactly when the forest leads
  the as-shipped split" — also scores 10/10 on those doses using no
  novel-stratum accuracies, no graded gaps and no φ* at all. The out-of-sample
  test on the Nucleotide Transformer suite rests on **one** admissible informative
  call, correct, with the aggregate 24/24 score noted as nearly uninformative
  since a constant "nothing swaps" predictor scores 23/24 (§S-2.3).

---

## Concordance: prior submission → current manuscript

The manuscript was restructured onto a construction-defect-first spine, the
Related-work section folded into the Introduction, and extended methods and
several results moved to the supplement. Prior numbering on the left.

| Prior (BIOADV-2026-296) | Current |
| --- | --- |
| §2 Related work | folded into §1 Introduction |
| §3.1 Datasets | §2.1; full table §S-1.1, Table S1 |
| §3.2 Features, models, decision rule | §2.5 |
| §3.3 Expanded nine-model roster | §S-1.2 (Table S2) |
| §3.4 Near-duplicate measurement | §2.4 |
| §3.5 Near-duplicate-aware re-split | §2.4 |
| §3.6 Controls and auxiliary diagnostics | §2.6; MMseqs2 detail §S-1.5 |
| §3.7 Regularization path and graded gap | §2.6 (path); §S-1.3 (graded gap) |
| §3.8 From-scratch 1D CNN grid | §S-1.4 |
| §3.9 Cluster bootstrap; imbalance / tuning / robustness definitions | §2.6 (bootstrap); §S-1.6 (definitions) |
| §3.10 Coordinate-space construction signatures | §2.2 |
| §3.11 Construct-and-break manipulations | §2.3 |
| §3.12 Cross-suite census and re-ranking | §2.6; counting conventions §S-1.7 |
| §3.13 Diagnostic condition φ* | §S-2.1 |
| §3.14 Reproducibility / software environment | §S-1.8 |
| §4.1 Leakage dataset-specific; length-robust re-measure | §3.1 (Table 1) |
| §4.2 Correcting the split lowers accuracy only on leaky datasets | §3.5 |
| §4.3 Leakage reorders the ranking | §3.6 |
| §4.4 Not an artifact of a four-model comparison (nine-model roster) | §3.6, "Nine learners" paragraph; Table S2 |
| §4.5 The mechanism | §3.7 |
| §4.6 The cause is a curation defect in the coordinates | §3.2 (Fig. 1) |
| §4.7 Construct-and-break | §3.3 (Table 3) |
| §4.8 Property of the unregularized forest | §3.8 |
| §4.9 Deep models | §3.9; grid §S-1.4; multiclass deep arm Table S5 |
| §4.10 Second suite: Nucleotide Transformer census | §3.11; splice mechanism §S-7 |
| §4.10 GUE pre-registration | §3.12 |
| §4.11 Dose-response | §S-2.2 (Fig. 2 in main text) |
| §4.12 Alignment-identity control | §S-3 (methods §S-1.5); summary §3.13 |
| §4.12 Chromosome-holdout control | §3.13, "Chromosome holdout" |
| §4.13 Robustness: imbalance, containment, cluster structure | §3.13; full imbalance panel §S-5; cluster structure §S-3.1 |
| §5 Discussion | §4 |
| §6 Conclusion | §5 |
| — (new) | §3.4 Out-of-suite BEND coordinate census |
| — (new; previously inside the Discussion) | §3.10 Ordinary tuning selects the memorizing configuration |
| — (new) | §S-2.3 Out-of-sample test of φ* on the Nucleotide Transformer suite |
| — (new) | §S-4 Detector calibration and the estimator floor |
