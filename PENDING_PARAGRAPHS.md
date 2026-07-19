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

## Status

| edit | ready | blocks final build? |
|---|---|---|
| 1 — Table 2 intervals | **yes** | yes, merge it |
| 2 — §4.7 disclosure | **yes** | yes, merge it |
| 3 — §5 tuning | no, job running | no — current text is correct |
