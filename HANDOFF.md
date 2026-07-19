# Handoff — parallel-session work while two experiments run

## Do not touch (two jobs are writing these)

Two CPU-bound experiments are running on full-scale `human_enhancers_ensembl`:

| job | writes | why it matters |
|---|---|---|
| `exp_tuning --datasets human_enhancers_ensembl` | `results/tuning_selection.csv` | whether leaky cross-validation selects the memorizing forest on the dataset the reordering claim lives on |
| `exp_construction_manip --only B` | `results/construction_manipulation.csv` | cluster-bootstrap intervals on the fix-a-leaky direction |

**Rules for a parallel session:**
- Do **not** edit or delete those two CSVs.
- Do **not** run anything in `audit/experiments/` — they are CPU-bound and will contend
  (the machine is already saturated; adding load roughly doubles everyone's wall clock).
- Do **not** rewrite git history. Normal commits on top are fine.
- Everything below is pure writing/figure work: no CPU, no conflict.


## File ownership (agreed 2026-07-19)

| owner | files | note |
|---|---|---|
| **the parallel session** | `paper/main.tex`, `paper/references.bib`, `paper/*.md` letters, `paper/Fig/*` | **sole owner.** The original session will not edit these. |
| the original session | `results/*.csv`, `results/TIER1_FINDINGS.md`, `audit/**` | owns the two running experiments and their integration |

When the two experiments finish, the original session will **not** patch `main.tex`.
It will append ready-to-paste LaTeX to `PENDING_PARAGRAPHS.md` at the repo root, with the
exact target location for each snippet. Check that file before your final build.

## Where the work stands

Independent blind scoring: **science 7.5/10, manuscript 7.0/10** (from 6.5/6.5).
Assessed ceilings: science **8.5** (structurally capped — the suite has only two leaky
datasets and one is honestly disclaimed), manuscript **9.5** (all editing).
AE recommendation: major revision, low end.

Seven adversarial audit rounds: 272 findings raised, 222 confirmed. Round 7 used four
fresh perspectives (PDF-only referee, benchmark curator, practitioner, replication
specialist). Full record: `results/audit_findings.csv`. Narrative: `results/TIER1_FINDINGS.md`.

Build: `paper/build.sh` → 15 pages, 0 errors, 0 undefined refs, 45 overfull hboxes.

## Highest-value parallel work, ranked

**All of the remaining points are on the manuscript axis, and none of it needs compute.**

1. **Cut hedge density.** The single most-cited manuscript problem. After seven audit
   rounds the paper has absorbed so much self-criticism that several sections spend more
   words limiting a finding than stating it. The fix is not to delete the caveats — they
   are load-bearing and hard-won — but to move them: state the result in the first
   sentence of each subsection, put the qualification after it, and never open a
   subsection with a limitation. Worst offenders: §4.4 (roster), §4.10/§4.11
   (cross-suite and the φ* discussion, which opens by calling most of its own evidence
   "vacuous by construction"), and the Discussion close.

2. **Add a Conclusion.** There is none (`grep 'section{Conclusion' paper/main.tex` → 0).
   A referee reading a 15-page paper with no closing synthesis has to build it themselves.

3. **Fix Figure 1's caption.** It asserts "Only the near-duplicate-aware split lowers
   accuracy, and only on leaky datasets", but the figure has **no clean-dataset panel**
   — the second half is unsupported by the image. Either add a clean panel (matplotlib,
   data already in `results/`) or narrow the caption. Also drop the apologetic
   "the y-axis starts at 0" sentence and just fix the axis.

4. **Report software versions.** Nothing anywhere pins Python, scikit-learn, numpy.
   They are in `results/requirements.txt` (scikit-learn 1.8.0, numpy 2.4.6, pandas 3.0.3,
   Python 3.13.3). A reproducibility paper that does not state its own versions is an
   easy referee hit.

5. **References are thin** — 15 for a benchmarking-methodology paper. Gaps a referee
   would notice: nothing on benchmark contamination in LLMs (the closest large
   literature), nothing on GraphPart/CD-HIT alternatives beyond a passing cite, no
   DeepSTARR/BEND/GUE citations even though they are discussed as future work.

6. **Rewrite `paper/response_to_reviewers.md` and `paper/cover_letter.md`.** Both predate
   every change made in this session and now describe a different paper. The response
   letter in particular should lead with the three things that answer the rejection:
   the nine-model roster, the construct-and-break manipulation, and the cross-suite
   census — plus the honest null.

7. **Terminology sweep.** "perceptron" vs "MLP" vs "multilayer perceptron" are used
   interchangeably, including in the abstract where "perceptron" sits against "both
   linear models" and reads as if it were one.

## What to tell a new session

> Read `HANDOFF.md` at the repo root first. Two experiments are running — do not run
> anything in `audit/experiments/` and do not touch `results/tuning_selection.csv` or
> `results/construction_manipulation.csv`. Work only on the manuscript
> (`paper/main.tex`, build with `paper/build.sh`) and the letters. Items 1–7 in the
> handoff, in order.

## Notes back from the parallel session (items 1-7 done)

Items 1-7 are complete; see the four commits after `817c395`. Two things the
original session should know, both touching files it owns:

1. **`audit/figures/make_paper_figures.py` no longer reproduces Figure 1.**
   Fig. 1 needed a clean-dataset panel (its caption claimed "only on leaky
   datasets" with no clean panel in the image), and the bar chart was replaced
   with a dumbbell plot so the cropped y-axis is legitimate and the apologetic
   caption sentence could go. Rather than edit `audit/**`, which this session
   does not own, the new figure is built by **`paper/Fig/make_fig_controls.py`**
   (reads `results/summary_final.csv` only; no fitting, no CPU load):

       ./venv/bin/python paper/Fig/make_fig_controls.py

   `make_paper_figures.py` still emits the old single-panel `fig_controls` into
   `results/figures/`, which is now superseded. Whoever owns `audit/**` should
   either point it at the new script or drop its `fig_controls` block, and add
   the new script to `REPRODUCE.md`. Left alone so as not to collide.

2. **Both soon-to-be-obsolete disclosures are now cleanly deletable.** Exact
   recipes, so neither is missed on the final merge:

   - **Discussion §5, tuning limitation.** The tuning argument was reordered so
     the result comes first and the caveat is the **last two sentences** of the
     paragraph, beginning "One scope limit belongs with this argument." If the
     ensembl job also selects `min_samples_leaf=1`, delete those two sentences
     wholesale and restate the result for both datasets. No other surgery.
   - **§4.7, manipulation intervals.** The no-interval disclosure is the **last
     sentence** of the "Three caveats belong with this result" paragraph
     ("And each condition is a single fit at a single split seed with no
     interval attached…"). When Manipulation B's intervals land, delete that
     sentence **and change "Three caveats" to "Two caveats"** in the same
     paragraph — the count is baked into the opener.

   As of this session's last build, both disclosures are still true and
   correctly stated, and `PENDING_PARAGRAPHS.md` did not yet exist.

Section numbering changed (a Conclusion was added as §6, and Related work grew),
so check `main.aux` rather than assuming old numbers when writing
`PENDING_PARAGRAPHS.md`. Current: manipulation table is **Table 4**, roster is
**Table 3**, report card is **Table 2**.

## When the two jobs finish

Check `exp_tuning_ens.log` and `exp_manip_ci_B.log` for `EXP_TUNING_DONE` /
`EXP_CONSTRUCTION_MANIP_DONE`, then:
- Tuning: if ensembl also selects `min_samples_leaf=1`, remove the limitation paragraph
  currently flagged in the Discussion (§5) and state the result for both datasets.
- Manipulation B: add the intervals to Table 2 (the manipulation table) and delete the
  disclosure that the manipulation deltas carry no uncertainty interval.
