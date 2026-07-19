# Pending manuscript edits from the experiment session

The parallel session owns `paper/main.tex`. These are ready-to-paste replacements with
their exact targets. **Merge before your final build** — two disclosures currently in the
paper are true today and become false once these land.

Source of every number: `results/construction_manipulation.csv` (committed).

---

## EDIT 1 — Table 2 (the manipulation table): add the interval column

The construct-and-break manipulation is the paper's only causal experiment and shipped
without uncertainty. It now has cluster-bootstrap intervals on every arm, and **both
directions are non-overlapping between control and intervention.**

**Find** the `tab:manip` table body (rows beginning `\multirow{2}{*}{Fix-a-leaky}`) and
**replace the whole `tabular` block** with:

```latex
\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llrrrll@{}}
\toprule
& Condition & $n$ & Leak & RF drop & 95\% CI & RF rank\\
\midrule
\multirow{2}{*}{Fix-a-leaky} & \textit{ensembl}, as shipped & 154{,}842 & 0.390 & $+0.165$ & $[0.157,0.173]^{*}$ & $1\!\to\!4$\\
& \quad duplicates collapsed & 81{,}868 & \textbf{0.012} & $\mathbf{-0.008}$ & $[-0.019,0.003]$ & $4\!\to\!4$\\
\midrule
\multirow{2}{*}{Break-a-clean} & \textit{ocr}, as shipped & 20{,}000 & 0.004 & $+0.002$ & $[-0.019,0.025]$ & $4\!\to\!4$\\
& \quad positives duplicated & 20{,}000 & \textbf{0.204} & $\mathbf{+0.105}$ & $[0.083,0.126]^{*}$ & $\mathbf{1\!\to\!4}$\\
\botrule
\end{tabular}
```

**Append to that table's caption:**

```latex
Intervals are two-arm cluster bootstraps over whole within-test near-duplicate
components; $^{*}$ marks intervals excluding zero. In both directions the control and
intervention intervals are disjoint.
```

---

## EDIT 2 — §4.7 (construct-and-break): delete the no-interval disclosure

**Find and DELETE** this sentence (it is now false):

> And each condition is a single fit at a single split seed with no interval attached, so
> we read the order-of-magnitude contrast against the control and not small differences
> between arms.

**Replace with:**

```latex
Each arm now carries a two-arm cluster-bootstrap interval, and the two directions are
cleanly separated: the leaky control's random-forest inflation is $+0.165$
$[0.157,0.173]$ while the merged, balance-matched version of the same dataset is
$-0.008$ $[-0.019,0.003]$, a null; imposing the defect on the clean dataset moves it from
$+0.002$ $[-0.019,0.025]$ to $+0.105$ $[0.083,0.126]$. Control and intervention intervals
are disjoint in both directions. Each condition remains a single fit at one split seed,
so we still read the contrast between arms rather than small differences within one.
```

---

## EDIT 3 — §5 Discussion: the tuning limitation (HOLD until the second job lands)

**Do not apply yet.** The ensembl tuning run is still going. When it finishes I will
append EDIT 3 here with the outcome. Until then the paragraph currently in §5 beginning
*"We report this for \textit{human\_nontata\_promoters} only"* is accurate and should
stay exactly as written.

If you need to ship before it lands, leave §5 untouched — it is honest as-is.

---

---

## EDIT 3 — §5 Discussion: the tuning limitation is now RESOLVED (ready to merge)

The ensembl tuning run completed. **Delete** the limitation sentence beginning
*"We report this for \textit{human\_nontata\_promoters} only"* through
*"...does not yet establish it on the specific dataset carrying the reordering claim."*

**Replace the whole tuning passage with:**

```latex
The decisive answer is empirical rather than rhetorical, so we ran it. A practitioner
does not know the split is leaky; they tune the way everyone tunes, by cross-validating
on the training set. We therefore cross-validated the forest over
\texttt{min\_samples\_leaf}$\in\{1,\dots,500\}$ on the as-shipped training set of both
leaky datasets, once with ordinary stratified $5$-fold and once with folds that hold
whole near-duplicate clusters out together---the leakage-aware version of the same
procedure.

On \textit{human\_enhancers\_ensembl}, the dataset carrying the reordering claim, the two
procedures diverge and the consequence is stark. Ordinary cross-validation selects
\texttt{min\_samples\_leaf}$=1$, the memorizing default, and the model it selects scores
$0.860$ on the as-shipped split but $0.696$ once the leakage is removed---a loss of
$16.4$ points. Cluster-grouped cross-validation instead selects
\texttt{min\_samples\_leaf}$=20$, and \emph{that} model loses only $2.2$ points
($0.751\to0.729$). So tuning does not rescue the practitioner: ordinary tuning actively
selects the configuration whose apparent advantage is almost entirely leakage. Worse, the
benchmark punishes the correct methodology---the leakage-aware tuner's model appears
$10.8$ points \emph{worse} on the as-shipped split ($0.751$ against $0.860$) while in fact
being $3.3$ points \emph{better} on the corrected one ($0.729$ against $0.696$). A
practitioner who does the right thing is penalised by the benchmark for doing it.

On \textit{human\_nontata\_promoters} both procedures select
\texttt{min\_samples\_leaf}$=1$, so our pre-registered expectation P5---that the grouped
procedure would always select a more regularized forest---holds on one leaky dataset and
fails on the other, and we report it that way.
```

**Also update the P5 line in Methods §3.3** (currently "P5 fails outright"):

```latex
P5 holds on \textit{human\_enhancers\_ensembl} and fails on \textit{human\_nontata\_promoters}
```

Numbers: `results/tuning_selection.csv` (4 rows, both datasets).

## Status (updated)

| edit | ready | blocks final build? |
|---|---|---|
| 1 — Table 2 intervals | **yes** | yes |
| 2 — §4.7 disclosure | **yes** | yes |
| 3 — §5 tuning | **yes** | yes — the current limitation text is now false |

---

## EDIT 4 — NEW: GUE results (scope narrowing + a detector false positive)

The pre-registration's binding forward predictions on GUE are now executed
(`results/gue_census.csv`). They mostly **failed**, and the failure narrows the paper's
scope claim in a way that must be reflected. Two changes.

**(a) Add to §4.9 (cross-suite), after the NT paragraphs:**

```latex
The pre-registration also carried binding predictions for GUE \citep{zhou2023dnabert2},
registered before any of that suite was examined: that its short fixed-length human
regulatory tasks---core promoter at $70$\,bp, promoter at $300$\,bp, transcription-factor
binding at $101$\,bp---would be leaky, and its multi-species tasks clean. Executed, the
predictions score $5$ of $17$, and \textbf{all eleven predicted-leaky tasks are clean},
with near-duplicate fractions spanning $0.005$--$0.041$ (median $0.009$). This refutes the
scope hypothesis the prediction encoded. Leakage does not track task type: eleven
independently built benchmarks in precisely the regime we flagged carry none of it. What
it tracks is the construction step of \S\ref{sec:r-coord}, and Genomic Benchmarks' two
leaky datasets have an un-deduplicated assembly where GUE's comparable tasks do not.
```

**(b) Add to the Discussion limitations, and this one is important:**

```latex
The pre-registered GUE run also exposed a limitation of our primary detector. The
\textit{virus\_covid} task, which we registered as clean, measures a near-duplicate
fraction of $1.000$---every test sequence has a near-duplicate in training, the largest
value we observed anywhere. It is not a curation defect. The task is nine-way SARS-CoV-2
variant classification over $999$\,bp windows of a ${\sim}30$\,kb genome whose variants
differ by a handful of mutations, so the test-to-train similarity has a \emph{minimum} of
$0.707$ and a median of $0.968$: the corpus is near-identical by biology. A near-duplicate
leak fraction carries no information there, and our detector cannot distinguish a curator
who omitted deduplication from an organism that is simply conserved. Any application of
this audit to viral, organellar, or other highly conserved sequence sets must establish
that distinction by other means---the coordinate-space signature of \S\ref{sec:m-coord} is
one, since biological conservation does not produce duplicated interval coordinates.
```

**(c) Scope sentence.** Wherever the paper says Claim II "requires short, fixed- or
near-fixed-length human regulatory DNA" (Introduction, and the Discussion scope
paragraph), that condition is now **refuted** and should be replaced by the construction
condition: an un-deduplicated assembly step, plus a memorization-prone model in the
comparison set.

---

## EDIT 5 — §3.3, P1 is scored on a statistic that post-dates it (please read carefully)

This one is an integrity issue, not a polish item, and it is the kind a referee on a
methodology paper will look for specifically. Flagging rather than fixing, since
`main.tex` is yours.

**Current text (line 124, end of the paragraph):**

> Of the five, P1 holds under the corrected gap and fails under the raw one, P2 holds on
> one leaky dataset and fails on the other, P3 and P4 hold, and P5 fails outright.

Two things are wrong with the P1 clause.

**(a) The corrected gap did not exist when P1 was registered.** P1 was committed in
`exp_roster.py` as "kNN-1 has the largest graded memorization gap $g$", where $g$ was the
*raw* `acc[>=0.9] - acc[<0.5]`. That is what the scoring code reads (`gg.graded_gap`) and
`results/roster_predictions.csv` records `holds=False`. The balanced-accuracy correction
was introduced **afterwards**, as the round-6 response to discovering the prevalence
confound (commit "Round-6 blockers: ... the graded gap was confounded"). Scoring a
pre-registered prediction on an outcome measure defined after the result was seen is
outcome switching. Leading with "holds under the corrected gap" invites exactly that
reading.

**(b) Even on the corrected gap, P1 holds on only one of the two datasets it covers.**
P1 was registered over "the leaky sets", plural. Corrected-gap winners:

| dataset | raw-gap top | corrected-gap top | P1 |
|---|---|---|---|
| `human_enhancers_ensembl` | RF | **kNN-1** | fails raw, holds corrected |
| `human_nontata_promoters` | kNN-15 | MLP | **fails both** |

So P1 has the same shape as P2 — holds on one leaky dataset, fails on the other — and the
sentence currently grants it more than P2 while P2 is the one that actually passed as
registered.

**Suggested replacement for the final sentence:**

```latex
Of the five, P1 fails as registered: on the raw gap it was committed to, the largest
value belongs to the forest on \textit{human\_enhancers\_ensembl} and to
$15$-nearest-neighbour on \textit{human\_nontata\_promoters}. We note, and label as
post hoc, that under the prevalence-corrected gap of \S\ref{sec:m-graded}---a statistic
we defined after P1 was committed, in response to the confound described there---$1$-NN
does have the largest gap on \textit{human\_enhancers\_ensembl}, though still not on
\textit{human\_nontata\_promoters}; we do not count this as a confirmation. P2 holds on
one leaky dataset and fails on the other, P3 and P4 hold, and P5 holds on
\textit{human\_enhancers\_ensembl} and fails on \textit{human\_nontata\_promoters}.
```

Note the P5 clause also needs the update from EDIT 3 ("fails outright" is now stale).

**Nothing in \S4.5 needs changing** — it already scopes the corrected-gap ordering claim
to `human_enhancers_ensembl`, and \S4.4 already says the corrected gap fails to separate
the families on `human_nontata_promoters`. It is only the \S3.3 summary sentence that
overstates.

Paying this cost is worth it: a pre-registration whose author reports "our prediction
failed" is far more persuasive than one where every prediction somehow held, and this
paper's whole argument is that people should hold themselves to measurements they
committed to in advance.

---

## EDIT 6 — the certification tool now has a ninth check (cohesion), and the paper should state its cut

Round 7's practitioner lens found two defects in `audit/tools/certify.py`, both now fixed
in `audit/**`. One of them changes what the paper can claim for the tool.

**What was wrong.** The tool was offered as the operational deliverable for any new
dataset, but it did not compute cluster cohesion — the one diagnostic the paper itself
uses to decide whether *correcting* a split is legitimate rather than destructive. Run on
`human_nontata_promoters` it returned a bare `LEAKY`, on the very dataset where the paper
concludes single-linkage de-duplication over-removes genuine promoter signal. A
practitioner following the tool would have re-split and destroyed signal.

**What it does now.** C9 reports largest-cluster cohesion (fraction of member pairs that
are real edges) and the fraction of clusters that are simple pairs, and below a cohesion
of $0.5$ it downgrades `LEAKY` to `leaky-but-correction-unsafe`, directing the user to
the novel-stratum readout on the as-shipped split instead of re-splitting. Verified
full-scale: `human_nontata_promoters` → cohesion $0.084$, verdict
`leaky-but-correction-unsafe` (`results/certify_nontata.json`). The cut is calibrated on
this paper's two leaky datasets, which sit either side of it by an order of magnitude:

| dataset | largest cluster | cohesion | clusters that are pairs | correction |
|---|---|---|---|---|
| `human_enhancers_ensembl` | 143 | **0.959** | 0.996 | safe |
| `human_nontata_promoters` | 420 | **0.084** | 0.240 | unsafe |

**(a) If the paper describes the tool's checks by number, it is now nine, not eight.**

**(b) Suggested sentence for wherever the tool is introduced (\S6 or Methods):**

```latex
The tool additionally reports cluster cohesion---the fraction of member pairs in the
largest component that are genuine near-duplicate edges---because whole-cluster
assignment is only a legitimate correction when components are near-duplicate sets
rather than single-linkage chains. Below a cohesion of $0.5$ it returns
\emph{leaky-but-correction-unsafe} rather than \emph{leaky}, and directs the user to the
novel-stratum readout on the as-shipped split instead of a re-split. The cut is
calibrated on the two leaky datasets studied here, which fall an order of magnitude
either side of it: \textit{human\_enhancers\_ensembl} has cohesion $0.96$ and
\textit{human\_nontata\_promoters} $0.08$, which is the quantitative form of the
distinction we draw between the clean case and the cautionary one.
```

This is worth having in the paper: it converts the ensembl/nonTATA split from a judgement
call the reader has to trust into a stated, reproducible threshold, which is the single
most common referee complaint about the current cautionary framing.

**(c) The other fixed defect needs no paper change** but is worth knowing: the two shipped
tools did not compose — `homology_split.py` writes `{train_idx, test_idx}` and `certify.py`
read `{train, test}`, so certifying the repo's own `demo_splits.json` died on a raw
`KeyError`. `certify.py` now accepts both spellings and gained `--emit-splits`, which
writes back the corrected partition it already computed (in both spellings), so
"certify then retrain" no longer requires a second run with independently chosen
parameters.
