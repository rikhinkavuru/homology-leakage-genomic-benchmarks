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
