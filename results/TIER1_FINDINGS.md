# Tier-1 findings — mechanism, standard, and a second suite

Companion to `results/tier1_preregistration.md`, which is committed at `2edad5c`,
**before** this document and before every result CSV it discusses, so the forward
predictions in its §4 have a tamper-evident position in history. Every number below
names the CSV it comes from; those CSVs are committed alongside this file.

These findings survived an adversarial audit (seven independent hostile lenses, every
finding re-verified against source by a separate agent: 56 raised, 47 confirmed). §7
lists what the audit changed. Several headline claims are weaker here than in the first
draft, and one is materially reframed; the corrections are marked in place rather than
quietly absorbed.

---

## Headline

1. **A predictive law.** A closed-form ranking breakdown point `φ* = δ/Δg` says a
   benchmark reports the wrong winner exactly when its leak fraction exceeds `φ*`. On
   `human_enhancers_ensembl`, φ = 0.380 against φ* ≈ 0.063 — a sixfold exceedance — and
   the predicted inversion is the one observed, with cluster-bootstrap CIs excluding
   zero. Its genuinely out-of-sample tests score 5/6.
2. **A construction rule, validated causally.** Coordinate geometry, computed
   independently of any k-mer, separates leaky from clean 7/8 strictly. Editing only the
   construction step **creates the pathology in a clean dataset and cures it in a leaky
   one** — both directions, ranking inversion included.
3. **An executable standard.** `certify --self-validate` reproduces the paper's eight
   published verdicts and exits non-zero on drift.
4. **A second suite.** Four of thirteen Nucleotide Transformer tasks are leaky, one at
   **25.0 % byte-identical**. On the two tasks where sequence length is unchanged between
   the original and revised releases, leakage falls from 0.244/0.245 to 0.0007/0.0010.

---

## 1. The inversion law (Deepener #2)

```
a_M = n_M + φ·g_M          Δ_M = a_M − a_M^corr = φ·g_M
inversion of pair (A,B)  iff  0 < δ < φ·Δg      φ* = δ/Δg
δ = n_B − n_A     Δg = g_A − g_B     g_M = h_M − n_M
```

### 1.1 What each part actually tests — and what it does not

The first draft of this document called all four parts "predictions". Three of them are
not, and saying so is the difference between a law and a restatement.

| # | claim | honest status | result |
|---|---|---|---|
| P1 | `a_orig ≈ n + φ·g` | **IDENTITY, not a prediction.** `a_orig` *is* the bin-weighted mean of the same graded accuracies (same model, same fit, same test set), so the multi-bin form is exact by construction | 4-bin residual 0.0000 / 0.0001; 2-bin residual 0.0020 / 0.0175 — this measures only mid-bin mass (1.4 % vs 22.4 %), not predictive skill |
| P2 | corrected winner = `argmax(n)` | **genuinely out of sample** — the theory never sees the corrected split, which needs re-splitting *and* retraining | **5/6** scored readouts: ensembl 3/3, nonTATA 2/3 |
| P3 | reg-path rank, 22 cells | **IN-SAMPLE** restatement of P1 as an ordering, on the same as-shipped test set | 21/22, but see the caveat below |
| P4 | injection sweep, *h* fitted at f = 0.1 | **out of sample** on the held-out cells; the f\* value is **not falsifiable** at this grid | held-out cells predicted to 0.0011 and 0.0008 |
| — | observable form `0<μ<Δ_A−Δ_B` | **TAUTOLOGY** — reduces to `a_B^corr > a_A^corr` | 12/12 by algebra; zero content |

Two further deflations we owe the reader. The P2 tally **excludes a fourth readout**
(`novel_only`), because its ground truth, `novelonly_ranking.best_novel_model`, is itself
`argmax(n)` — it is true by construction. And **P3's 21/22 is less impressive than it
looks**: 16 of the 22 cells have observed rank 4 with the nearest competitor 0.012–0.086
away, so the rank call is a blowout; the single miss (nonTATA `min_samples_leaf=5`) is
one of the few genuinely close cells. What P3 does establish is that the identity holds
across a twelve-fold change in *g*.

### 1.2 Anchors and cluster-bootstrap verdicts

`inversion_law_bootstrap.csv` — 10,000 replicates, blocks = within-test Jaccard>0.7
components (G = 29,311 / 7,618, matching the frozen `cluster_bootstrap_full.csv`).

| dataset | pair | δ [95 % CI] | φ* | φ | m_inv [95 % CI] | verdict |
|---|---|---|---|---|---|---|
| ensembl | RF → LinearSVC | +0.0122 [0.0054, 0.0192] | 0.0633 | 0.3802 | 0.0611 [0.0560, 0.0663] | **CLEAN** |
| ensembl | RF → LR | −0.0049 [−0.0116, 0.0019] | — | 0.3802 | 0.0783 [0.0733, 0.0835] | FRAGILE |
| nonTATA | every pair | δ ≤ 0, or m_inv CI ∋ 0 | — | 0.2255 | — | **no CLEAN inversion** |

### 1.3 The law's real operating point

The first draft said the law "predicts the nonTATA non-inversion". That overstated it:
nonTATA's corrected-split accuracy ranking *does* invert (RF 1 → 3). What the law
predicts is that nonTATA has **no inversion on the novel stratum** (δ ≤ 0 for every RF
pair), which is why the paper already grades that dataset cautionary — its accuracy
inversion does not survive AUROC, F₁, or novel-only. Stated as a classifier against
corrected-split accuracy inversions, the law's operating point is **high precision, low
recall**: it fires on one pair, that pair is a true inversion, and it stays silent on
several real ones. That asymmetry is structural, not tuned — see §1.4.

### 1.4 Why the law under-fires: a retraining penalty it does not model

`a_corr ≈ n` is the single assumption, and it fails asymmetrically:

| dataset | LR | LinearSVC | RF | HGB |
|---|---|---|---|---|
| ensembl | +0.0157 | +0.0155 | **−0.0741** | +0.0120 |
| nonTATA | −0.0046 | −0.0463 | **−0.0593** | −0.0113 |

Retraining on a de-leaked training set costs the memorizer far more than the others, so
the law under-predicts how far RF falls and misses inversions rather than inventing them.
(The first draft advertised "zero false positives out of 12" as evidence; it is not —
a false positive is close to algebraically impossible on that readout, so the statistic
carries no information. It has been removed.)

### 1.5 The injection axis

φ(f) = n_inj/(n_test+n_inj) is known by construction. Fitting the leaked-stratum accuracy
at **f = 0.1 only** gives h_RF = **0.9988** — the memorization ceiling, expected *a
priori* to be ≈1 for near-exact copies — and h_LR = 0.7096, both invariant in *f*. The
held-out cells follow: f = 0.2 → 0.7655 predicted vs 0.7644 observed; f = 0.4 → 0.8374 vs
0.8366.

The implied φ* = 0.0408 corresponds to f\* = 0.0106, and the observed RF-over-LR flip
occurs between f = 0 and f = 0.1. **This is a consistency check, not a validated
prediction**: the grid has no cell between 0 and 0.1, so any f\* in that interval would
be "bracketed". Testing it properly needs cells near f = 0.005–0.02, which we have not
run. The defensible statement is the qualitative one — roughly one percent of a clean
benchmark's own training data, injected into its test set, is enough to change which
model wins.

---

## 2. The construction rule (Deepener #3)

Coordinates for all 8 datasets were recovered offline and realigned to loader row order.
Recovered interval width equals sequence length for 100 % of rows in all 8 — though note
this check is **vacuous for the four fixed-length datasets** (200/200/500/251 bp) and is
only informative for the variable-length ones (ensembl, ocr, regulatory, stark).

**Statistic:** max over classes of the share of TEST intervals overlapping a TRAIN
interval at ≥50 % reciprocal overlap. **Threshold 0.1, inherited from
`report_card.py:49`. Zero fitted parameters.**

| dataset | redundant class | R_self | xsplit ≥50 % | exact | coord | sequence |
|---|---|---|---|---|---|---|
| human_enhancers_ensembl | positive | 0.4713 | **0.7552** | 0.7552 | LEAKY | LEAKY |
| human_nontata_promoters | **negative** | 0.9605 | **0.9920** | 0.0000 | LEAKY | LEAKY |
| human_ocr_ensembl | negative | 0.0101 | 0.0062 | 0 | CLEAN | clean |
| human_ensembl_regulatory | enhancer | 0.0000 | 0.0000 | 0 | CLEAN | clean |
| human_enhancers_cohn | negative | 0.0090 | 0.0060 | 0 | CLEAN | clean |
| drosophila_enhancers_stark | negative | 0.3130 | 0.0358 | 0 | CLEAN | clean |
| demo_coding_vs_intergenomic | intergenomic | 0.0085 | 0.0064 | 0 | CLEAN | **borderline** |
| demo_human_or_worm | worm | 0.0966 | 0.0730 | 0.0003 | CLEAN | clean |

**Agreement is 7/8 strict** (counting `borderline` as its own level) **and 8/8 only when
`borderline` is folded into `clean`.** The first draft reported 8/8 without disclosing
that recoding; the miss is `demo_coding_vs_intergenomic_seqs`, and it is the same dataset
§2.3 uses as the example of a mechanism coordinates cannot see. Separation among the
strict calls: min LEAKY 0.7552 vs max CLEAN 0.0730.

**Coverage gap.** `demo_coding`'s `coding_seqs` class is keyed by ENST transcript
accession — 50,000 rows, 50,000 distinct "regions" — so it lives in a per-row coordinate
space where no two intervals can overlap and every coordinate statistic is structurally
zero. That is *absence of measurement*, not evidence of cleanliness. Its verdict rests on
the `intergenomic_seqs` class alone.

**Independence caveat.** `human_ocr_ensembl`'s positive class and
`human_ensembl_regulatory`'s `ocr` class are the *same intervals*. They are not two
independent confirmations of the merged-consensus prediction; effective n is 7, not 8.

### 2.1 The internal-Ensembl contrast

Same provider, same organism, opposite verdicts, so only the merge step differs:
`human_enhancers_ensembl` ships 77,421 positives on **40,934 unique coordinates**, with
94.3 % of positive rows on a duplicated coordinate and 75.5 % of test positives having
their exact coordinate in train; the Regulatory-Build sets sit at 0.0000–0.0101. (The
first draft said "exactly 0.000 in every class" — that is true of
`human_ensembl_regulatory` but not of `human_ocr_ensembl`, whose negative class is
0.0101.) All **36,487** duplicated-coordinate groups carry byte-identical sequences and a
single label.

**On mechanism, we now claim less.** GB documentation attributes this set to the FANTOM5
atlas (808 CAGE libraries), and the first draft asserted the redundancy *is* an un-merged
808-library union. The measured signature does not support that: multiplicity is capped
at exactly 2 with identical boundaries, whereas a union over many independently-called
tracks would give a broad multiplicity spectrum and jagged partial overlaps. What the
data supports is the weaker, sufficient statement: **each of 40,934 intervals is shipped
exactly twice, with no deduplication step.**

### 2.2 Cross-modal prediction — including the case where it fails

The first draft's table omitted `human_nontata_promoters`, the one dataset where this
prediction misses badly. All eight rows, from `construction_rule.csv`:

| dataset | coord-predicted leak | measured sequence leak @0.9 |
|---|---|---|
| human_enhancers_ensembl | **0.3776** | **0.3802** |
| human_nontata_promoters | **0.3434** | **0.2254** |
| human_ocr_ensembl | 0.0001 | 0.001 |
| human_ensembl_regulatory | 0.0000 | 0.001 |
| human_enhancers_cohn | 0.0003 | 0.000 |
| demo_coding_vs_intergenomic | 0.0009 | 0.024 |
| demo_human_or_worm | 0.0077 | 0.003 |
| drosophila_enhancers_stark | 0.0034 | 0.006 |

Coordinates track sequence leakage to **0.26 pp** under the exact-duplication mechanism
and **over-predict by 11.8 pp** under window tiling, where ≥90 % reciprocal overlap of a
251 bp window still leaves enough novel 8-mers to fall below the 0.9 Jaccard cut. The
honest claim is therefore *classification*, not calibration: coordinates say reliably
**whether** a dataset leaks, and only under exact duplication say **how much**.

### 2.3 Three mechanisms

| mechanism | signature | example |
|---|---|---|
| exact coordinate duplication | `xsplit_exact` high | ensembl (0.7552) |
| contiguous window tiling | `xsplit_ge50pct` high, exact = 0 | nonTATA negatives (0.9920 / 0.0000) |
| paralog / repeat homology | coordinates clean, sequence leaky | demo_coding (0.0064 coord, containment-borderline) |

The third is invisible to coordinates. **Coordinates may escalate a verdict but must
never clear one**, and the sequence detectors remain necessary.

### 2.4 Construct-and-break: the defect is causal in both directions

`construction_manipulation.csv`. Each manipulation is paired with an unmanipulated
control run of the identical pipeline at the same scale and split seed.

| condition | n | leak@0.7 | exact | RF acc orig→corr | RF drop | RF rank | inverts |
|---|---|---|---|---|---|---|---|
| **B control** — ensembl as shipped | 154,842 | 0.3898 | 0.3853 | 0.8607 → 0.6957 | **+0.1650** | 1 → 4 | **yes** |
| **B manipulated** — duplicate coordinates collapsed | 118,355 | **0.0110** | 0.0039 | 0.7331 → 0.7291 | **+0.0040** | 4 → 4 | **no** |
| **A control** — ocr as shipped (merged) | 20,000 | 0.0040 | 0.0000 | 0.6448 → 0.6426 | +0.0022 | 4 → 4 | no |
| **A manipulated** — 94.3 % of positives duplicated | 29,426 | **0.5076** | 0.5055 | 0.7594 → 0.7036 | **+0.0558** | **1 → 4** | **yes** |

Applying the omitted merge step to the leaky dataset drops its leak fraction 35-fold and
its random-forest inflation 41-fold, and **the ranking inversion disappears**. Imposing
the same construction defect on a clean dataset raises its leak fraction 127-fold and
**manufactures the inversion**, promoting the random forest from rank 4 to rank 1 on the
contaminated split before it is demoted again by correction. Editing coordinates alone —
no change to sequences, labels, model code, or split ratio — both creates and cures the
pathology.

Two caveats. Manipulation B removes 36,487 rows (23.6 %), so its training set shrinks and
absolute accuracies are not comparable to the control; the *drop* and the *ranking* are
the outcomes, and both are measured within each condition. Manipulation A duplicates
rows, which is leaky by construction — its content is not that duplication causes
leakage, which is obvious, but that it reproduces the full downstream syndrome including
the rank-1 promotion of the memorizer.

---

## 3. `certify` — the executable standard (Deepener #6-A)

`python -m audit.tools.certify --self-validate` reproduces **8/8** published verdicts
(2 LEAKY / 1 borderline / 5 clean) and exits non-zero on drift.

**What that gate is and is not.** It is a *regression gate on the frozen tables*: it
reads three committed CSVs and recomposes the verdicts by rule, re-running no similarity
computation and no model. For seven of the eight datasets it applies `jaccard > 0.1` to
the same column that produced the original verdict, so those are consistency checks
rather than independent confirmations. Its non-trivial content is the composition step —
it derives the `borderline` level for `demo_coding` from full-scale containment, which
`report_card.py` cannot express, and which the paper otherwise produces by hand.

**The full C1–C8 path has been executed end to end on one dataset**,
`drosophila_enhancers_stark`, and every check now reports a state that can move the
verdict: C1 pass (full scale), C2 pass, C3 pass (MMseqs2 agrees), C4/C5 low-confidence
(high-similarity bin holds 10 sequences, below the n ≥ 50 floor), C6
`drop_is_repartition_cost`, C7 `drop_includes_0`, C8 clean. Residual leak after
re-splitting is 0.000; the drop is 0.0235 with two-arm cluster CI **[−0.0072, 0.0533]**,
which includes zero and agrees with the frozen `cluster_bootstrap_full.csv` value
[−0.0067, 0.0539]. Verdict: **clean**.

### 3.1 Two defects the standard caught in itself

- **A single-arm bootstrap.** C7 computed the drop CI by resampling only the
  original-split correctness vector and holding the corrected accuracy fixed, discarding
  half the variance (SE understated by ≈1/√2). It reported stark's drop as
  [0.0019, 0.0434], excluding zero, against the frozen two-arm [−0.0067, 0.0539], which
  includes it — an anticonservative interval published by a tool whose entire pitch is
  statistical rigor, and contradicting the paper's own "all clean-dataset deltas include
  zero". `certify` now calls the frozen `cluster_bootstrap.delta_boot` directly.
- **A tripwire firing on ten sequences.** An early run escalated stark to borderline on a
  memorizer gap of 0.197 measured over a high-similarity bin of 10 sequences, below the
  `MIN_N = 50` floor `run_graded.py` already applies, while the matched random-resplit
  control showed the drop was repartitioning cost. The tripwire now requires a populated
  bin, an excess over that control, and a drop CI excluding zero.

Four of the eight advertised checks could not previously affect the verdict; C1, C3, C7
and C8 are now wired into it, and every check's state is recorded in the report.

---

## 4. Cross-suite census: a second suite (#1-A)

`crosssuite_census.csv`, `crosssuite_exact_verification.csv`. Train sets capped at
20,000; capping only removes near-duplicate partners, so every figure is a lower bound.

| suite | tasks | LEAKY | borderline | clean | max jac@0.7 | max exact |
|---|---|---|---|---|---|---|
| NT-original | 13 | **4** | 3 | 6 | **0.2500** | **0.2500** |
| NT-revised | 13 | 0 | 0 | **13** | 0.0020 | 0.0000 |

**Verified independently of the census code path** (plain string set membership, no
k-mers): NT-original `enhancers` ships 400 test sequences of which **exactly 100 are
byte-identical to a training sequence**, label concordance **1.000**; its training set
holds 14,968 rows on 14,002 unique sequences.

### 4.1 The length confound, and what survives it

The audit caught a real threat to this section. **Sequence length changed between the two
releases on 8 of 13 tasks**, and 8-mer Jaccard falls monotonically as sequences lengthen,
so a naive original-vs-revised Jaccard comparison is confounded — and the confound is
worst on the flagship `enhancers` task, which went from 200 bp to 400 bp.

The comparison that survives is restricted to **length-matched tasks**:

| task | length (both) | jac orig → rev | containment orig → rev | verdict |
|---|---|---|---|---|
| splice_sites_acceptors | 600 bp | **0.2435 → 0.0007** | 0.3724 → 0.0073 | LEAKY → clean |
| splice_sites_donors | 600 bp | **0.2452 → 0.0010** | 0.3699 → 0.0073 | LEAKY → clean |
| promoter_all / _no_tata / _tata | 300 bp | ≤0.0145 → 0.0000 | ≤0.0161 → ≤0.0044 | clean → clean |

So on the two tasks where sequence length is held constant, near-duplicate leakage falls
by roughly 300-fold under both a length-blind and a length-robust metric. That is the
controlled result.

**What we no longer claim.** The `enhancers` 25.0 % → 0 % contrast is *not* a controlled
comparison: the revised task was rebuilt at a different window size, so its cleanliness
cannot be attributed to curation alone. The 25.0 % byte-identical figure remains a
verified property of the original release, and the revised release is verifiably clean;
the causal attribution belongs to the length-matched splice tasks.

### 4.2 A pre-registered prediction that failed

The registered call `NT-original → LEAKY` was **wrong as a suite-level statement**: only
4 of 13 tasks are leaky and 6 are clean, including all three promoter tasks. The
informative consequence is that **`human_nontata_promoters`' leak fraction of 0.406 is
not a property of the promoter task** — NT's equivalent is clean at 0.0059 — but of
Genomic Benchmarks' construction of it. That separates task effect from curation effect
and corroborates §2, while refuting the coarser prediction we registered.

---

## 5. Reviewer wires: closed, and partially closed

| wire | status | evidence |
|---|---|---|
| dataset characteristics / provenance | **closed** | coordinate rule 7/8 strict, plus manipulations in both directions |
| hyperparameter choice | **closed** | the identity reproduces the reg-path outcome across a 12-fold change in *g* |
| bootstrap methodology | **closed** | cluster bootstrap over within-test components, ICC + design effect, two-arm deltas |
| homology-detection validation | **closed** | coordinates are k-mer-independent; MMseqs2 agreement |
| single-suite, n = 1, not systematic | **PARTIALLY closed** | a second suite has 4 leaky tasks, one 25 % byte-identical — but this is a **leak census, not a demonstrated re-ranking**. No model was trained on NT. The precondition for inversion is shown to be widespread; the inversion itself is still n = 1 |
| CNN and Transformer generality | **partially closed** | the identity holds on the from-scratch CNN cells; the foundation-model roster remains Tier 2/3 |
| protein / long-range reach | **open** | Tier 2/3 |

---

## 6. Reproduction

```bash
python -m audit.experiments.exp_inversion_law --bootstrap    # ~9 min
python -m audit.experiments.exp_construction                 # ~1 min
python -m audit.experiments.exp_construction_manip           # ~30 min
python -m audit.experiments.exp_crosssuite_census            # ~5 min + download
python -m audit.experiments.exp_crosssuite_verify            # ~2 min
python -m audit.tools.certify --self-validate                # ~2 s, exit 1 on drift
```

`huggingface_hub==1.24.0` is required by the census modules and is pinned in
`results/requirements.txt`.

---

## 7. What the adversarial audit changed

Recorded because a QA gate that leaves no trace is indistinguishable from one that never
ran. 56 findings raised, 47 confirmed after independent re-verification.

| # | correction |
|---|---|
| 1 | `certify` C7 used a single-arm bootstrap; replaced with the frozen two-arm `delta_boot` |
| 2 | P1 relabelled from "prediction" to arithmetic identity |
| 3 | P2's `novel_only` readout excluded as tautological; score 6/8 → **5/6** |
| 4 | P3 relabelled from "out of sample" to in-sample restatement; low discriminative content disclosed |
| 5 | f\* labelled a consistency check — unfalsifiable at the available grid |
| 6 | "zero false positives out of 12" removed as algebraically vacuous |
| 7 | "predicts the nonTATA non-inversion" reframed as high-precision/low-recall |
| 8 | construction rule reported **7/8 strict** alongside 8/8 lenient; the recoding disclosed |
| 9 | `demo_coding`'s `coding_seqs` flagged unmeasurable in coordinate space |
| 10 | §2.2 cross-modal table now shows all 8 rows including the 11.8 pp nonTATA miss |
| 11 | FANTOM5 union mechanism downgraded to the exact multiplicity-2 duplication the data shows |
| 12 | ocr / regulatory `ocr` identified as the same intervals; effective n = 7 |
| 13 | "self-overlap exactly 0.000 in every class" corrected (ocr negative is 0.0101) |
| 14 | `len_match = 1.0` noted vacuous for the four fixed-length datasets |
| 15 | **NT length confound disclosed; the curation claim moved to the length-matched splice tasks** |
| 16 | NT suite-level pre-registration recorded as failed (4/13, not suite-wide) |
| 17 | `--self-validate` described as a regression gate, circular for 7/8, not end-to-end |
| 18 | C1/C3/C7/C8 wired into the verdict instead of being advertised but inert |
| 19 | `huggingface_hub` pinned in `requirements.txt` |
| 20 | census now writes atomically to a sidecar — an interrupted run had truncated the CSV |
| 21 | pre-registration committed at `2edad5c`, before the results, so anchoring is real |
