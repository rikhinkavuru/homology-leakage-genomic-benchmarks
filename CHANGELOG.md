# CHANGELOG — paper cleanup pass

Scope: `paper/main.tex`, `paper/references.bib`, figure scripts (`make_paper_figures.py`, `make_part_b_figures.py`, `make_graded_figure.py`) + regenerated figures. Coordinated via subagents (A & E parallel on disjoint files; B→C→D serial on `main.tex`, rebuild between each).

**Build command** (manual; no latexmk/Makefile):
```bash
cd paper && export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH" && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && \
  pdflatex ... main.tex && pdflatex ... main.tex
```
**Engine:** pdflatex (OUP `oup-authoring-template`, `[webpdf,modern,large,namedate]`); no fontenc/inputenc/fontspec overrides — diacritics via LaTeX accent macros.

**Final build health:** 6 pages, 0 errors, 0 undefined refs/citations, no `??`.
**Page count diff:** started 6 → 5 after Fig 3 compression (Agent E) → 6 again after the Fig 1 caption note was added. **Net: 6 pages (unchanged from start).** No content lost (clean refs/cites throughout).

Every line item below is labeled **[verified — already correct]** (no edit) or **[changed]** (edited this run).

---

## Agent A — `paper/references.bib` (owns: .bib only)

| Item | Status | Detail / rationale |
|---|---|---|
| Diacritics Grešová, Katarína, Čechák, Šimeček (`gresova2023`), Söding (`steinegger2017mmseqs2`) | **[verified — already correct]** | Already use LaTeX macros (`\v{s}`,`\'a`,`\'\i`,`\v{C}`,`\v{S}`,`\v{c}`,`\"o`). Confirmed rendering in the compiled PDF (References + Data Availability). No change; engine stays pdflatex (no UTF-8 switch). |
| arXiv entry `bushuiev2024` | **[changed]** | `journal={arXiv preprint arXiv:2404.10457}` (no doi) → `journal={arXiv preprint}` + `doi={10.48550/arXiv.2404.10457}`. Rationale: give every entry a `doi` for consistent identifier formatting; arXiv id stays visible via its canonical DOI. |
| arXiv entry `huang2025` | **[changed]** | Same transformation → `doi={10.48550/arXiv.2507.21404}`. |
| bioRxiv entries `rafi2025`, `kovtun2024` | **[verified — already correct]** | `journal={bioRxiv}` + `10.1101/...`; mutually consistent. |
| Published entries `gresova2023`, `li2006cdhit`, `steinegger2017mmseqs2` | **[verified — already correct]** | journal + real DOI; unchanged. |

Result: all 7 entries now carry a `doi`; no real DOI invented/altered; no `eprint`/`archivePrefix` added (this natbib `.bst` may silently drop them). bibtex runs clean.

## Agent B — `main.tex` front-matter metadata (owns: preamble/metadata)

> Boundary note: the task assigned "preamble (above `\begin{document}`)," but in the OUP class the journal-metadata commands live just *below* `\begin{document}`. Agent B correctly **stopped at its stated boundary** (made zero edits) and flagged the mismatch rather than crossing it. The orchestrator then applied the flags (trivial comment-adding) as the Agent-B workstream.

| Item | Status | Detail |
|---|---|---|
| `\DOI{DOI added during production}` | **[changed]** | appended `% TODO: pre-acceptance stub -- DOI assigned by OUP at production`. |
| `\copyrightyear{2026}`, `\pubyear{2026}` | **[changed]** | appended `% TODO: confirm ... at acceptance`; `\pubyear` TODO also notes Volume/Issue stay blank until production (no `\volume`/`\issue` command exists; running header shows them empty). |
| `\access{Advance Access Publication Date: Day Month Year}` | **[changed]** | appended `% TODO: pre-acceptance stub -- publication date set at production`. |
| `\firstpage{1}` | **[changed]** | appended `% TODO: pre-acceptance stub -- first page assigned at production`. |
| commented `\received/\revised/\accepted` | **[changed]** | added a `% TODO: uncomment and fill ... at acceptance` line above them. |
| values themselves | **[verified — intentional]** | No production value was filled in; only `% TODO` comments added. |

## Agent C — `main.tex` body sections (owns: \section prose)

| Item | Status | Detail |
|---|---|---|
| §4.5 "two independent honest evaluations" | **[changed]** | → "two independent **evaluation routes**". |
| §1 `regulatory-genomics` | **[verified — already correct]** | Hyphenated with following space; no run-together `regulatorygenomics` anywhere. |
| §3.5 / §4.4 `label-concordance` | **[verified — already correct]** | Real hyphen in source, not a line-break artifact. |
| run-together words / doubled spaces in prose | **[verified — none]** | All `[a-z][A-Z]` adjacencies are legit proper nouns/acronyms (AlphaFold-Multimer, LIT-PCBA, MinHash, AUROC, HistGradientBoosting, LinearSVC). Doubled spaces only inside `table*` alignment (D's region), untouched. |

## Agent D — `main.tex` tables (owns: the two table environments)

| Item | Status | Detail |
|---|---|---|
| Table 1 / Table 2 size header | **[changed]** | Table 2 `$n$` → `$n$ (full)` to match Table 1 (both report full dataset size). |
| Table 2 `Acc.\ drop` | **[changed]** | → `Acc.\ lost` (header); caption's ``Acc.\ drop'' → ``Acc.\ lost'' to match; sign clarification "original − corrected; positive = accuracy lost" kept. |
| booktabs / no vertical rules | **[verified — already correct]** | `\toprule/\midrule/\botrule`; specs `@{}lrlll@{}` and `@{}lrrrllrll@{}` have no `|`. |

## Agent E — figure scripts + outputs (owns: plotting scripts + figures)

| Item | Status | Detail / rationale |
|---|---|---|
| Fig 1 `fig_controls` y-floor | **[verified — kept by design]** | `ylim(50,100)` intentionally KEPT (per author: a 0-baseline flattens the drop the figure exists to show). Disclosure added to caption instead (see orchestrator row). |
| Fig 1 title/top-tick crowding | **[changed]** | `make_paper_figures.py`: `set_title(..., pad=14)` so the title clears the top frame / 100 tick. Verified in rendered PDF. |
| Fig 1 caption truncation note | **[changed]** (main.tex, orchestrator-applied from E's spec) | Appended to `fig:controls` caption: "The $y$-axis is truncated at 50\% (not 0\%) to make the homology-aware accuracy drop visible; bars span best-model accuracies of roughly 70--93\%." |
| Fig 2 `fig_ranking_inversion` palette | **[verified — already correct, no change]** | Per constraint: only restyle if LR/LinearSVC are hard to tell apart in grayscale. Rendered + converted to grayscale and inspected: LR = darker gray, **solid**, **circle**; LinearSVC = lighter gray, **dashed**, **triangle** — three independent cues. Colors LEFT UNCHANGED. |
| Fig 3 `fig_graded_performance` wasted space | **[changed]** | `make_graded_figure.py`: `subplots(2,2, figsize=(11,6.4), gridspec_kw={height_ratios:[1.7,1.0]})` so the sparse clean-control bottom row is shorter. Verified n-count labels (n=4966, n=28, n=4, n=2, …) remain legible. |
| Vector export | **[verified — already correct]** | All three paper figures are vector PDF (font objects, 0 image XObjects). Regenerated and copied to `paper/Fig/`. |

> Figure-output housekeeping: `fig_ranking_inversion` and the unused `fig_capacity_scaling`/`fig_capacity_extended` were regenerated but their content was unchanged (only PDF CreationDate metadata churned), so they were reverted to keep the commit to genuinely-changed files (`fig_controls`, `fig_graded_performance`).

## Phase 2 — `REVISIONS.md` (proposals only; NOT applied to main.tex)

- Abstract softening: **already satisfies — no change proposed**.
- §4.3 τ paragraph: **already satisfies — no change proposed** (already leads with 1→3 / 1→4 rank changes; τ demoted to descriptive support).
- Methods bootstrap sentence: **proposed** (1,000 resamples, percentile, resampling unit = individual test sequences, no refit, fixed seed).
- Reviewer-risk experiments (flagged-only, not attempted): (a) k-mer-Jaccard circularity → MMseqs2/alignment re-split plan; (b) classical-only vs neural → one 1D-CNN plan.

---

## Could NOT verify

- **GitHub repo vs paper.** Data Availability cites `https://github.com/rikhinkavuru/homology-leakage-genomic-benchmarks`, which matches the `origin` push remote (confirmed). I could **not** independently verify external/anonymous accessibility of the repo, nor re-derive every paper number from the pushed state without re-running the full pipeline.
- **arXiv DOIs not resolved.** `10.48550/arXiv.2404.10457` and `...2507.21404` follow arXiv's algorithmic DOI scheme but were **not** resolved against doi.org to confirm registration. Worth a one-click check before submission.
- **Production stubs.** `\pubyear`/`\copyrightyear` 2026 and the empty Volume/Issue are author/journal decisions (flagged with `% TODO`), not verifiable here.
- **`.DS_Store`** is present untracked in the repo root (macOS junk); not committed. Consider adding to `.gitignore` (out of scope).
