#!/usr/bin/env python3
"""
Check the documents' numbers against the CSVs they come from.

Thirteen adversarial audit rounds established that the dominant defect in this project is
not a wrong computation but a wrong TRANSCRIPTION: a value corrected in one place and left
stale in another. Rounds 10 through 12 were almost entirely that.

The first version of this module was mutation-tested by round 13 and largely failed. It
searched the WHOLE of main.tex for each numeral, so a value quoted at four sites still
passed when one was corrupted -- precisely the failure mode it existed to catch. It also
contained a check whose predicate could not fail, and a docstring claiming coverage it did
not have. This version fixes all three, and the design rules that follow exist because
each was violated once:

  1. ANCHOR every check to the passage that makes the claim, never to the document. A
     check takes a context regex; the value must appear inside the matched window.
  2. ASSERT OCCURRENCE COUNTS where a value should appear a known number of times, so
     corrupting one of several sites drops the count and fails.
  3. NEVER write a predicate that cannot fail. Every check must be shown to fail on a
     deliberately broken input -- `--self-test` does exactly that and is part of the gate.
  4. Check the RESULTS DOCUMENTS too, not just the manuscript. Half the round-11 and
     round-12 findings were paper/results-document divergence, which a manuscript-only
     gate cannot see by construction.

It remains a gate, not a proof: it checks only claims registered in it. Adding a check
when a number enters a document is the discipline it is meant to enforce.

Run:  python -m audit.tools.check_numbers [--self-test]
Exit: 0 if every registered claim matches its source, 1 otherwise.
"""
from __future__ import annotations
import argparse
import math
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(HERE, "results")
# Every document a reader can receive. The letters were outside this map until round 17,
# and that is exactly how a retired claim survived into the submission packet: the guard
# reported "absent everywhere" while "exceeded our compute budget" was live in the
# response to reviewers, describing a run that had completed. A guard is only as wide as
# its document list, so anything a referee or editor reads belongs here.
DOCS = {
    # The submission artifact is not always main.tex. During the restructure the manuscript
    # a referee will actually read is paper/main_resubmission.tex, and this guard went on validating
    # the reference copy -- the exact failure the comment above warns about, committed by
    # the guard itself. Set PAPER_TEX to repoint it at whichever file is being submitted.
    # The DEFAULT is now the submitted file, not the reference copy: a gate whose default
    # target is the frozen prior submission passes cleanly while saying nothing about the
    # manuscript anyone is editing, and defaults are what get run.
    "paper": os.path.join(HERE, "paper", os.environ.get("PAPER_TEX", "main_resubmission.tex")),
    "findings": os.path.join(R, "TIER1_FINDINGS.md"),
    "reproduce": os.path.join(HERE, "REPRODUCE.md"),
    "response": os.path.join(HERE, "paper", "response_to_reviewers.md"),
    "cover": os.path.join(HERE, "paper", "cover_letter.md"),
    # PAPER_NUMBERS calls itself the single authoritative source, so it is the
    # last document that should be outside the guard. Round 22 found it frozen
    # two rounds behind, contradicting Table 2 on the borderline verdict.
    "numbers": os.path.join(R, "PAPER_NUMBERS.md"),
}

_cache = {}


def doc(name):
    r"""Read a document, resolving \input{} so every check sees the whole manuscript.

    This matters more than it looks. When the table bodies moved out of main.tex into
    generated \input files, every numeric check instantly went blind to the values in
    them -- and the tables are precisely where the numbers live. Two corrected-gap
    checks failed loudly and revealed it; the rest would have kept passing on reduced
    coverage without saying so, which is the worst possible failure mode for a gate.
    Inlining here restores full coverage and keeps it as tables move.
    """
    if name not in _cache:
        with open(DOCS[name]) as fh:
            body = fh.read()
        base = os.path.dirname(DOCS[name])

        def _inline(m):
            path = os.path.join(base, m.group(1))
            if not path.endswith(".tex"):
                path += ".tex"
            try:
                with open(path) as fh:
                    return fh.read()
            except OSError:
                return m.group(0)      # leave unresolved rather than hide the reference

        body = re.sub(r"\\input\{([^}]+)\}", _inline, body)

        # A manuscript split across main text and supplementary is ONE document as far as
        # a referee is concerned, and the same blindness the docstring describes recurs
        # when content moves out to a supplement: every check that scans "paper" starts
        # reporting "context not found" for material that is present, correct, and cited.
        # Append the supplement so coverage follows the content, exactly as \input
        # inlining does. Absent supplement, nothing changes.
        # Scoped to the restructured submission only: main.tex is the prior submission and
        # ships no supplement, so appending one to it double-counts every value and turns
        # "expected 2 sites" into "found 4". Running the reference copy as a control is
        # what caught that.
        if name == "paper" and os.path.basename(DOCS[name]) != "main.tex":
            supp = os.path.join(base, "supplementary.tex")
            if os.path.exists(supp):
                with open(supp) as fh:
                    body += "\n" + re.sub(r"\\input\{([^}]+)\}", _inline, fh.read())

        _cache[name] = body
    return _cache[name]


def csv(name):
    return pd.read_csv(os.path.join(R, name))


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def _tokens(value, places, signed):
    """Every rendering of `value` we accept at `places` decimals.

    We accept the two neighbouring roundings only when the source sits on an exact half
    (0.1215, 0.1055 both do), where round-half-up and float-repr rounding disagree.
    Anything further from the source than one unit in the last place still fails.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None                      # non-finite never matches; see rule 3
    out = set()
    for v in (value, math.nextafter(value, math.inf), math.nextafter(value, -math.inf)):
        t = f"{v:+.{places}f}" if signed else f"{v:.{places}f}"
        out.add(t)
        if not signed and t.startswith("0."):
            out.add(t[1:])               # ".009" as well as "0.009"
    return out


def near(where, context, value, places=3, signed=False, label="", span=420):
    """The value must appear INSIDE the window matched by `context`.

    This is the rule-1 check. `context` is a regex identifying the passage that makes the
    claim; we take the matched span plus `span` characters after it and look for the value
    only there, so a correct copy of the number elsewhere in the file cannot rescue a
    corrupted one here. `span` must be wide enough to reach the value from the anchor and
    narrow enough to exclude the next claim -- a sentence or two.
    """
    s = doc(where)
    toks = _tokens(value, places, signed)
    if toks is None:
        return (False, label or context, "value is not finite")
    m = re.search(context, s, re.S)
    if not m:
        return (False, label or context, f"context not found in {where}")
    lo = max(0, m.start() - 40)
    window = s[lo:m.end() + span]
    # A bare substring test lets a corrupted value pass whenever the expected rendering
    # is a PREFIX of it -- 0.08 matches 0.089, +0.122 matches +0.1229. Round 14 proved
    # five checks blind that way. Require a non-digit boundary on both sides.
    ok = any(re.search(r"(?<!\d)" + re.escape(t) + r"(?!\d)", window) for t in toks)
    shown = sorted(toks)[0]
    return (ok, label or context,
            f"{where}: {shown} present in the passage" if ok
            else f"{where}: {shown} NOT in the passage that claims it")


def absent(where, value, places=3, signed=False, label="", context=None):
    """A value the documents must NOT contain -- a specific regression, killed.

    `context` restricts the search to the passage that would carry the regression. Without
    it a bare numeral collides with unrelated quantities: "0.005" appears in this project
    as a confidence bound, a grid endpoint and a margin, none of which is the stale GUE
    minimum we are trying to keep out.
    """
    s = doc(where)
    if context is not None:
        m = re.search(context, s, re.S)
        if not m:
            return (True, label, "context absent, so the regression cannot be present")
        s = s[m.start():m.end() + 260]
    toks = _tokens(value, places, signed)
    if toks is None:
        return (True, label, "not applicable")
    hit = [t for t in toks
           if re.search(r"(?<!\d)" + re.escape(t) + r"(?!\d)", s)]
    return (not hit, label, "absent" if not hit else f"{where}: found {hit[0]}")


def occurs(where, text, n, label=""):
    """`text` must appear exactly `n` times. Rule 2: corrupting one site drops the count.

    Counted with a digit boundary, because a plain substring count silently conflates a
    value with any longer decimal that starts the same way: "+0.179" matched "+0.1791"
    at two sites, so the check reported four occurrences of a value stated twice and its
    expected count had been tuned to the wrong number. A count that can be satisfied by
    a different value is not a count of the value.
    """
    pat = re.escape(text) + r"(?!\d)" if text and text[-1].isdigit() else re.escape(text)
    got = len(re.findall(pat, doc(where)))
    return (got == n, label or f"{text!r} appears {n}x",
            f"{where}: {got} occurrence(s)" if got == n
            else f"{where}: expected {n}, found {got}")


def says(where, text, label="", want=True):
    """Substring test with whitespace normalised.

    Without normalisation the check is fragile to line wrapping: TIER1_FINDINGS wraps
    "tasks of eleven are\n   leaky", so a literal "twelve are leaky" could never match a
    revert of that sentence and the check was vacuous. Round 14's self-test caught it.
    """
    norm = " ".join(doc(where).split())
    got = " ".join(text.split()) in norm
    return (got == want, label or f"{where} says {text!r}",
            "present" if got else "absent")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_corrected_gaps():
    g = csv("graded_gap_corrected.csv")
    MEM = {"kNN-1", "kNN-15", "RF", "ExtraTrees"}
    out = []
    for dset, tag in (("human_enhancers_ensembl", "ensembl"),
                      ("human_nontata_promoters", "nontata")):
        d = g[g.dataset == dset]
        mem = d[d.model.isin(MEM)].graded_gap_balanced.mean()
        rest = d[~d.model.isin(MEM)].graded_gap_balanced.mean()
        bad = d[~d.model.isin(MEM | {"MLP"})].graded_gap_balanced.mean()
        ctx = (r"memorization-prone learners.{0,120}average" if tag == "ensembl"
               else r"memorizers average")
        out.append(near("paper", ctx, mem, signed=True,
                        label=f"{tag} memorizer mean {mem:+.3f}"))
        out.append(near("paper", ctx, rest, signed=True,
                        label=f"{tag} non-memorizer mean {rest:+.3f}"))
        # the MLP-excluded mean is the specific round-10/11 regression: it must be gone
        out.append(absent("paper", bad, signed=True,
                          label=f"{tag} MLP-excluded mean {bad:+.3f} absent from paper"))
        out.append(absent("findings", bad, signed=True,
                          label=f"{tag} MLP-excluded mean {bad:+.3f} absent from findings"))
    # 1-NN's corrected gap, the section's headline
    knn = float(g[(g.dataset == "human_enhancers_ensembl") &
                  (g.model == "kNN-1")].graded_gap_balanced.iloc[0])
    # Anchored on the clause that STATES the value, not on "definitional memorizer":
    # that phrase now occurs twice in the bundle (the roster spec introduces the term,
    # the exploratory section uses it), and near() takes the first match, which was the
    # occurrence that never carried the number.
    out.append(near("paper", r"has by far the largest gap", knn, signed=True,
                    label=f"1-NN corrected gap {knn:+.3f}"))
    # Rule 2, applied where round 14 proved a single anchored check was not enough. Each
    # of these values is stated at several sites; an anchored check protects one of them
    # and the count protects the rest, so a stale site drops the count and fails.
    for tok, n, what in (("0.461", 3, "1-NN corrected gap"),
                         # 2, not 3: the third site was a substring match on "+0.1791",
                         # a different quantity, which the boundary-aware count above
                         # now rejects. Both surviving sites verified against
                         # roster_nontata_mechanism_recheck.csv mean_gap_others=0.1786.
                         ("+0.179", 2, "nonTATA non-memorizer mean"),
                         ("0.370", 2, "ExtraTrees corrected gap"),
                         ("$3$ of $15$", 2, "GUE registered score"),
                         ("$0.96$", 4, "ensembl cohesion")):
        out.append(occurs("paper", tok, n, label=f"{what} {tok} at all {n} sites"))
    return out


def check_gue():
    g = csv("gue_census.csv")
    reg = g[g.registered]
    leaky = reg[reg.preregistered == "LEAKY"]
    correct = int(reg.prereg_correct.fillna(False).astype(bool).sum())
    lo, hi = leaky.leak_jaccard_0p7.min(), leaky.leak_jaccard_0p7.max()
    clo, chi = leaky.leak_containment_0p7.min(), leaky.leak_containment_0p7.max()
    ctx = r"predicted-leaky tasks are clean"
    return [
        (len(reg) == 15, f"15 registered GUE tasks (found {len(reg)})", ""),
        (correct == 3, f"registered score {correct}/15", "must be 3/15"),
        (int(leaky.prereg_correct.fillna(False).astype(bool).sum()) == 0,
         "0 of 11 predicted-LEAKY correct", ""),
        (bool(g[~g.registered].prereg_correct.isna().all()),
         "unregistered tasks carry no score", "mouse_0/mouse_1 unscored"),
        (bool(g.train_frac_used.eq(1.0).all()),
         "GUE census full scale on every row",
         "all train_frac_used=1.0" if g.train_frac_used.eq(1.0).all()
         else "SOME ROWS CAPPED -- verdicts are lower bounds"),
        near("paper", ctx, lo, label=f"GUE Jaccard min {lo:.3f}"),
        near("paper", ctx, hi, label=f"GUE Jaccard max {hi:.3f}"),
        near("paper", ctx, clo, label=f"GUE containment min {clo:.3f}"),
        near("paper", ctx, chi, label=f"GUE containment max {chi:.3f}"),
        # The virus_covid false-positive argument's two figures. Until round 26 these had
        # no generator: exp_gue kept only thresholded fractions, so the minimum and median
        # of the similarity vector existed nowhere but a hand-written notes file.
        near("paper", r"test-to-train similarity has a", float(g[g.task == "virus_covid"].sim_min.iloc[0]),
             label="virus_covid similarity minimum"),
        near("paper", r"test-to-train similarity has a", float(g[g.task == "virus_covid"].sim_median.iloc[0]),
             label="virus_covid similarity median"),
        near("paper", r"predictions score", float(correct), places=0,
             label=f"paper quotes {correct} of 15"),
        # the capped-run range is the round-12 regression in the findings doc
        absent("findings", 0.0051, context=r"jac@0\.7 spans",
               label="capped GUE minimum 0.005 absent from the findings GUE table"),
        near("findings", r"jac@0\.7 spans", lo, label=f"findings quote full-scale min {lo:.3f}"),
    ]


def check_manipulation():
    m = csv("construction_manipulation.csv").set_index("condition")

    def ci(cond):
        return [float(x) for x in
                re.findall(r"-?\d+\.\d+", str(m.loc[cond, "rf_drop_ci"]))[:2]]
    b_ctl, b_int = ci("B_control_unmerged_union"), ci("B_manipulated_merged_balanced")
    a_ctl, a_int = ci("A_control_merged_consensus"), ci("A_manipulated_matched")
    s = doc("paper").replace(" ", "")
    out = []
    for lo, hi, label, n in ((b_ctl[0], b_ctl[1], "fix-a-leaky control", 2),
                             (b_int[0], b_int[1], "fix-a-leaky intervention", 2),
                             (a_ctl[0], a_ctl[1], "break-a-clean control", 2),
                             (a_int[0], a_int[1], "break-a-clean intervention", 2)):
        tok = f"[{lo:.3f},{hi:.3f}]"
        got = s.count(tok)
        # each interval appears twice: once in Table 4, once in the section-4.7 prose.
        # Rule 2: corrupting either site drops the count and fails.
        out.append((got == n, f"{label} CI {tok} at {n} sites",
                    f"{got} occurrence(s)"))
    out.append((b_ctl[0] > b_int[1], "fix-a-leaky intervals disjoint",
                f"{b_ctl[0]:.4f} > {b_int[1]:.4f}"))
    out.append((a_int[0] > a_ctl[1], "break-a-clean intervals disjoint",
                f"{a_int[0]:.4f} > {a_ctl[1]:.4f}"))
    return out


def check_dose_response():
    d = csv("dose_response.csv")
    ok = int(d.prediction_correct.sum())
    triv = int(((d.rf_rank_orig == 1) == d.ranking_inverts).sum())
    flip = d[d.ranking_inverts].iloc[0]
    lo, hi = d.mid_band_frac.min() * 100, d.mid_band_frac.max() * 100
    return [
        (ok == len(d), f"P6 holds on {ok}/{len(d)} doses", ""),
        (len(d) == 10, f"ten doses committed (found {len(d)})", ""),
        near("paper", r"first inverts at dose", float(flip.phi_used),
             label=f"flip-point phi {flip.phi_used:.3f}"),
        near("paper", r"first inverts at dose", float(flip.phi_star),
             label=f"flip-point phi* {flip.phi_star:.3f}"),
        near("paper", r"we report the\s+intermediate", lo, places=2,
             label=f"band minimum {lo:.2f}%"),
        near("paper", r"we report the\s+intermediate", hi, places=2,
             label=f"band maximum {hi:.2f}%"),
        # rule 3: assert the COMPUTED competing-rule score, not the mere phrase
        says("paper", f"also scores ${triv}/{len(d)}$",
             label=f"competing rule {triv}/{len(d)} disclosed with its true score"),
    ]


def _pct(x):
    """Percent, rounded half-up. 0.0875*100 is 8.749999... in binary, which formats as
    8.7 while the paper correctly writes 8.8; the gate must not fail on that."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(x)) * 100)


def check_intervals_are_labelled():
    """The round-12 defect: an interval labelled as the cluster bootstrap when it was the
    combined-source one. Both must be quoted, each under its own name."""
    cb = csv("cluster_bootstrap_full.csv")
    row = cb[(cb.dataset == "human_nontata_promoters") & (cb.model == "RF_k6")].iloc[0]
    clu = [_pct(x) for x in re.findall(r"-?\d+\.\d+", str(row.delta_ci_cluster))[:2]]
    com = [_pct(x) for x in re.findall(r"-?\d+\.\d+", str(row.delta_ci_combined))[:2]]
    s = doc("paper")
    return [
        (f"[{clu[0]:.1f},{clu[1]:.1f}]".replace(" ", "") in s.replace(" ", "")
         or f"${clu[0]:.1f},{clu[1]:.1f}$" in s
         or f"[{clu[0]:.1f},{clu[1]:.1f}]" in s,
         f"nontata cluster-bootstrap CI [{clu[0]:.1f},{clu[1]:.1f}] quoted", ""),
        (f"[{com[0]:.1f},{com[1]:.1f}]" in s,
         f"nontata combined-source CI [{com[0]:.1f},{com[1]:.1f}] quoted", ""),
        (abs(clu[0] - com[0]) > 1e-6,
         "the two intervals are genuinely different numbers",
         f"cluster {clu} vs combined {com}"),
        near("paper", r"combined-source interval---which additionally folds", com[0],
             places=1, label="nontata combined-source interval named where it is used"),
    ] + _check_interval_robustness(cb)


def _check_interval_robustness(cb):
    """E1/E2/E4: the BCa-vs-percentile agreement and the Welch df, which are reported
    as agreement rather than as corrections, so the numbers backing that claim are
    guarded. If BCa ever diverged from percentile the paper's 'they agree' would be
    false, and this catches it."""
    leaky = cb[cb.dataset.isin(["human_enhancers_ensembl", "human_nontata_promoters"])]
    max_shift = float(leaky.bca_vs_pct_max_shift.abs().max())
    min_df = float(leaky.combined_df.min())
    out = [
        (max_shift <= 0.0006 + 1e-9,
         "BCa differs from percentile by at most 0.0006 on every leaky delta",
         f"max shift {max_shift}"),
        (min_df > 4.0,
         "every combined-source effective df exceeds the 4 a flat t_4 would assume",
         f"min df {min_df}"),
        says("paper", "differs from the percentile interval by at most $0.0006$",
             label="the BCa agreement is stated in the supplement"),
    ]
    # the two interval values the supplement quotes must match the CSV
    ens = leaky[(leaky.dataset == "human_enhancers_ensembl") & (leaky.model == "RF_k6")].iloc[0]
    e = [_pct(x) / 100 for x in re.findall(r"-?\d+\.\d+", str(ens.delta_ci_combined))[:2]]
    out.append(near("paper", r"forest interval is unchanged at", e[0], places=3,
                    label="ensembl combined interval quoted in the supplement", span=200))
    return out


def check_cohesion():
    g = csv("exp_g11_cohesion.csv").set_index("dataset")
    ens = g.loc["human_enhancers_ensembl", "largest_cluster_cohesion"]
    ntp = g.loc["human_nontata_promoters", "largest_cluster_cohesion"]
    return [
        near("paper", r"magnitude either side of it", ens, places=2,
             label=f"ensembl cohesion {ens:.2f}"),
        near("paper", r"magnitude either side of it", ntp, places=3,
             label=f"nontata cohesion {ntp:.3f}"),
        (ens >= 0.5 > ntp, "the two datasets straddle the 0.5 cut",
         f"{ens:.3f} >= 0.5 > {ntp:.3f}"),
    ]


def check_crosssuite_counts():
    c = csv("crosssuite_census.csv")
    nt = c[c.suite == "NT-original"]
    independent = len(nt) - 2          # enhancers_types and promoter_all are nested
    leaky = nt[(nt.verdict == "LEAKY") & (nt.task != "enhancers_types")]
    borderline = nt[(nt.verdict == "borderline")]
    clean = independent - len(leaky) - len(borderline)
    return [
        (len(nt) == 13, f"13 NT-original tasks shipped (found {len(nt)})", ""),
        # Round 14 found the NT census had been left capped at 20,000 while the GUE one
        # was re-run in full; three verdicts flipped when it was finally run uncapped.
        # This mirrors the GUE full-scale assertion so it cannot recur silently.
        (not bool(c.subsampled.any()),
         "cross-suite census is full scale on every row",
         "no row subsampled" if not c.subsampled.any()
         else "SOME ROWS CAPPED -- clean verdicts there are provisional"),
        (len(leaky) == 3, f"{len(leaky)} leaky independent tasks", ""),
        (len(borderline) == 6, f"{len(borderline)} borderline", ""),
        (clean == 2, f"{clean} clean independent tasks", "3+6+2 = 11"),
        says("paper", "three of eleven", label="paper says three of eleven"),
        says("paper", "three of twelve", want=False,
             label="paper no longer says three of twelve"),
        # the round-12/13 regression lived in the findings document, not the paper
        says("findings", "of 12 independent", want=False,
             label="findings no longer says 12 independent"),
        says("findings", "twelve are leaky", want=False,
             label="findings no longer says twelve"),
        occurs("paper", "three of eleven", 3,
               label="paper states the eleven-task count at all 3 sites"),
    ]


def check_retired_claims():
    """Statements the project has corrected, asserted absent from EVERY document.

    This is the systemic answer to the failure that produced the blocker in rounds 14, 15
    and 16 and most of the majors in 10 through 13: a claim gets corrected where it was
    found and survives in its counterpart. Each entry below is a phrase that was true once
    and is now false; if any of them reappears anywhere, a fix has been half-applied again.
    Add to this list whenever a claim is retired, not just where it was found.
    """
    retired = [
        ("capped at 20,000",        "the cross-suite census cap, lifted in round 14"),
        ("capped at $20{,}000$ sequences for the census",
                                    "the same cap, in the manuscript's phrasing"),
        ("roughly threefold",       "the overstated GUE capped-to-full ratio (true max 2.8x)"),
        ("roughly tripled",         "the same overstatement, manuscript phrasing"),
        ("three of twelve",         "the pre-collapse Nucleotide Transformer task count"),
        ("of 12 independent",       "the same count, findings phrasing"),
        ("same duplicated-coordinate defect",
                                    "an NT mechanism claim unsupported by that release"),
        ("short, fixed-length human regulatory",
                                    "false of human_enhancers_ensembl (2-573 bp)"),
        ("5/17",                    "the GUE tally inflated by two unregistered tasks"),
        ("no interval attached",    "the manipulation disclosure, obsolete since 5d2b5dd"),
        ("exceeded our compute budget",
                                    "the ensembl tuning run that in fact completed"),
        ("Both select min_samples_leaf = 1",
                                    "true of nontata only; ensembl grouped CV selects 20"),
        ("cluster-grouped cross-validation select the memorizing default",
                                    "the same claim in paraphrase -- round 18 found the "
                                    "literal-string guard blind to it"),
        ("expectation to the contrary was wrong",
                                    "P5 holds on ensembl; it did not fail"),
        ("P5 fails outright",       "P5 holds on ensembl and ties on nontata"),
        ("P5) failing outright",    "the same, in the cover letter's phrasing"),
        ("Of the five,",            "there are six pre-registered predictions, not five"),
        ("Three limits bound",      "the Discussion limitations count: the foundation-model "
                                    "exclusion was split into its own titled paragraph, so "
                                    "the remaining paragraph carries two, not three"),
        ("do not claim the network would win",
                                    "superseded: the seed-replicated paired comparison now "
                                    "supports the ranking claim 4.4 previously declined"),
        ("single most useful experiment",
                                    "4.4 nominated the seed-replicated CNN comparison as "
                                    "future work; it has been run"),
        ("Whether a CNN-based leaderboard",
                                    "the Discussion's open question, now answered: the CNN "
                                    "goes rank 5 -> rank 1 under correction"),
        ("interval-bearing CNN comparison that we have not run",
                                    "we have now run it"),
    ]
    out = []
    for phrase, why in retired:
        # Case-folded. Round 18 found "Three of twelve" alive in the response letter while
        # this check reported the lowercase phrase absent: a sentence-initial capital was
        # enough to hide a retired claim in a shipped PDF.
        needle = " ".join(phrase.split()).lower()
        hits = [w for w in DOCS if needle in " ".join(doc(w).split()).lower()]
        out.append((not hits, f"retired: {phrase!r} ({why})",
                    "absent everywhere" if not hits
                    else f"STILL PRESENT in {', '.join(hits)}"))
    return out


def check_letter_section_refs():
    """The letters are markdown and cannot use \\ref, so their section pointers are typed
    by hand and drift whenever a subsection is inserted. Round 18 found all ten of them off
    by one, because the dose-response section became 4.11 and everything after it shifted --
    so the response letter routed reviewers to the wrong section for the two items the
    Associate Editor had gated the decision on. This resolves each pointer against the
    manuscript's own numbering and checks the topic matches.
    """
    tex = doc("paper")
    # DERIVE the Results section number rather than assuming it. The restructure dropped
    # a preceding section, so Results went from 4 to 3, and a hard-coded "4." resolved
    # every pointer against the wrong section -- reporting the letters as broken while
    # they were right, which is worse than the drift the check exists to catch.
    marker = "\\section{Results}"
    if marker in tex:
        sec_no = len(re.findall(r"\\section\{", tex[:tex.index(marker)])) + 1
        body = tex[tex.index(marker):]
    else:
        sec_no, body = 4, tex
    end = body.find("\\section{Discussion}")
    if end > 0:
        body = body[:end]
    subs = re.findall(r"\\subsection\{(.+?)\}", body)
    numbering = {f"{sec_no}.{i+1}": t for i, t in enumerate(subs)}
    # Only pointers written in the CURRENT numbering are validated. The response letter's
    # crosswalk table also carries prior-submission numbers in its left column, which are
    # historical by design and must not be resolved against the current manuscript.
    expect = [
        (f"§{sec_no}.13", ("robust",)),
    ]
    out = []
    for ref, keywords in expect:
        num = ref.lstrip("§")
        title = numbering.get(num, "")
        ok = any(k.lower() in title.lower() for k in keywords)
        out.append((ok, f"{ref} in the letters resolves to a section about {keywords[0]}",
                    f"{num} = {title[:56]!r}" if title else f"{num} not found"))
    # and nothing may point past the last Results subsection
    highest = max(int(k.split(".")[1]) for k in numbering) if numbering else 0
    for where in ("response", "cover"):
        refs = [int(m) for m in re.findall(r"§4\.(\d+)", doc(where))]
        bad = [r for r in refs if r > highest]
        out.append((not bad, f"{where} letter has no pointer past §4.{highest}",
                    "all in range" if not bad else f"dangling: {bad}"))
    return out


def check_shared_numbers():
    """Quantities the manuscript and the letters both state, checked against the CSV AND
    against each other.

    The retired-claims group catches a stale *phrase*; it cannot see a stale *numeral*.
    Round 19 proved the gap: the ICC range was corrected in main.tex and left at its old
    value in the response letter, and the gate passed 77/77 while the two documents in one
    submission packet quoted different ranges for the same four fits. Any figure that
    appears in more than one document belongs here.
    """
    cb = csv("cluster_bootstrap_full.csv")
    icc = cb[cb.icc.notna()].icc
    lo, hi = icc.min(), icc.max()
    out = []
    for where, ctx in (("paper", r"intraclass correlation is high"),
                       ("response", r"intraclass correlation is high")):
        out.append(near(where, ctx, float(lo), places=2,
                        label=f"{where}: ICC lower bound {lo:.2f}"))
        out.append(near(where, ctx, float(hi), places=2,
                        label=f"{where}: ICC upper bound {hi:.2f}"))
    # and the two documents must agree with each other, not merely each with the CSV
    pat = re.compile(r"intraclass correlation is high \(?\$?([\d.]+)\$?[^\d]{1,12}\$?([\d.]+)")
    seen = {}
    for where in ("paper", "response"):
        m = pat.search(" ".join(doc(where).split()).replace("--", "-").replace("–", "-"))
        seen[where] = (m.group(1), m.group(2)) if m else None
    agree = seen["paper"] is not None and seen["paper"] == seen["response"]
    out.append((agree, "manuscript and response letter quote the same ICC range",
                f"{seen['paper']} vs {seen['response']}"))
    return out


def check_imbalance_panel():
    """The prevalence stress test. No check covered exp_imbalance_panel.csv until round 27,
    which is how two successive over-broad readings of it reached the manuscript: first
    "nontata shows the same pattern" and then, as its replacement, "accuracy is the metric
    least disturbed". Both were false on nontata. What IS true on both datasets is that
    AUPRC and MCC fall further than accuracy, so that is what this asserts.
    """
    d = csv("exp_imbalance_panel.csv")
    out = []
    for ds, tag in (("human_enhancers_ensembl", "ensembl"),
                    ("human_nontata_promoters", "nontata")):
        sub = d[(d.dataset == ds) & (d.target_pi == 0.2)] if "target_pi" in d.columns else \
              d[d.dataset == ds]
        if len(sub) < 2:
            continue
        o = sub[sub.split == "original"].iloc[0]
        c = sub[sub.split == "corrected"].iloc[0]
        d_acc, d_auprc, d_mcc = (abs(o.acc - c.acc), abs(o.auprc - c.auprc),
                                 abs(o.mcc - c.mcc))
        out.append((d_auprc > d_acc and d_mcc > d_acc,
                    f"{tag}: AUPRC and MCC both move more than accuracy",
                    f"auprc {d_auprc:.3f}, mcc {d_mcc:.3f} vs acc {d_acc:.3f}"))
    # and the specific superlative that was false must not reappear
    out.append(absent("paper", 0.0, label="(placeholder)") if False else
               (("accuracy is the metric least disturbed" not in " ".join(doc("paper").split())),
                "the false 'accuracy least disturbed' generalization is absent",
                "absent"))
    return out


def check_cnn_seed_replication():
    """The seed-replicated CNN comparison (4.4, 4.9) and its paired interval.

    This one is guarded tightly because it is the newest claim in the paper and the most
    flattering, which is the combination that has historically gone stale first. Three
    things are asserted, not one: the corrected-accuracy range, the paired difference and
    its interval, and -- crucially -- the 4-of-5 count. That last is the number a summary
    would round away, since the seed-averaged interval excludes zero and it is tempting to
    report only that. The manuscript must keep saying that one seed does not.
    """
    seeds = csv("exp_deep_cnn_seeds.csv")
    paired = csv("exp_deep_cnn_paired.csv")

    corr = seeds["acc_corr"]
    lo_acc, hi_acc = float(corr.min()), float(corr.max())
    drops = seeds["drop"]
    lo_drop, hi_drop = float(drops.min()), float(drops.max())

    per_seed = paired[paired["seed"].astype(str) != "mean"]
    mean_row = paired[paired["seed"].astype(str) == "mean"].iloc[0]
    n_excl = int(per_seed["paired_excl0"].astype(str).str.lower().eq("true").sum())
    n_tot = len(per_seed)

    out = []
    # 4.4: the corrected-accuracy range and the paired interval
    for where, ctx in (("paper", r"Replicating the pre-registered reference cell"),
                       ("response", r"Across\s+\*\*five training seeds\*\*")):
        out.append(near(where, ctx, lo_acc, places=4, label=f"{where}: CNN corrected min {lo_acc:.4f}"))
        out.append(near(where, ctx, hi_acc, places=4, label=f"{where}: CNN corrected max {hi_acc:.4f}"))
    # The COMBINED-SOURCE interval is the one the documents must quote. The narrow
    # seed-averaged interval is also asserted, but only as the figure each document
    # explicitly disowns -- an audit found it reported as the headline, where it
    # understated uncertainty precisely in the flattering direction, so the guard now
    # pins which of the two plays which role.
    for where, ctx in (("paper", r"the paired difference is"),
                       ("response", r"the paired difference is")):
        out.append(near(where, ctx, float(mean_row["paired_diff"]), places=4,
                        label=f"{where}: paired difference {float(mean_row['paired_diff']):.4f}"))
        out.append(near(where, ctx, float(mean_row["combined_ci_lo"]), places=4,
                        label=f"{where}: combined CI lower {float(mean_row['combined_ci_lo']):.4f}"))
        out.append(near(where, ctx, float(mean_row["combined_ci_hi"]), places=4,
                        label=f"{where}: combined CI upper {float(mean_row['combined_ci_hi']):.4f}"))
    out.append(near("cover", r"combined-source CI", float(mean_row["combined_ci_lo"]),
                    places=4, label="cover: combined CI lower"))
    out.append(near("cover", r"combined-source CI", float(mean_row["combined_ci_hi"]),
                    places=4, label="cover: combined CI upper"))
    # the narrow interval must appear ONLY as the disowned alternative, in both docs
    for where in ("paper", "response"):
        body = " ".join(doc(where).split())
        narrow = f"{float(mean_row['paired_ci_lo']):.4f}" in body.replace("$", "")
        disowned = ("do not quote that as the result" in body
                    or "do not report that figure as the result" in body)
        out.append((not narrow or disowned,
                    f"{where}: the seed-averaged interval is disowned where it appears",
                    "" if (not narrow or disowned) else
                    "the narrow interval appears without being disowned"))
    # 4.9: the drop range, which weakens a previously single-valued claim
    out.append(near("paper", r"gives drops of", lo_drop, places=4,
                    label=f"paper: CNN drop min {lo_drop:.4f}"))
    out.append(near("paper", r"gives drops of", hi_drop, places=4,
                    label=f"paper: CNN drop max {hi_drop:.4f}"))

    # the non-uniformity must survive in every document that states the result
    assert n_excl == 4 and n_tot == 5, f"CSV says {n_excl}/{n_tot}; update the prose"
    for where in ("paper", "response", "cover"):
        body = " ".join(doc(where).split())
        states_split = ("four of the five" in body.lower()
                        or "four of five" in body.lower()
                        or "one of the five seeds" in body.lower()
                        or "one of the five" in body.lower())
        out.append((states_split,
                    f"{where}: discloses that one of five seeds does not clear zero",
                    "" if states_split else "the 4/5 split has been summarised away"))
    return out


def check_generated_tables():
    """The manuscript's tables must be the generated ones, and must be current.

    Two failure modes, both of which have precedent in this project. Someone edits a
    generated .tex by hand and the next build silently reverts it; or a source CSV moves
    and the committed .tex is never regenerated, so the paper ships a table that
    disagrees with its own data. The first is caught by the header assertion, the second
    by re-deriving both files in memory and comparing.
    """
    out = []
    # deliberately the RAW file: doc() inlines \input, which would erase the very
    # directive this check exists to assert
    # Both files, not just the main text: the roster table now lives in the supplement,
    # and a check that reads only DOCS["paper"] reported a hand-typed table for a table
    # that is correctly \input one file over.
    raw = ""
    for path in (DOCS["paper"], os.path.join(HERE, "paper", "supplementary.tex")):
        if os.path.exists(path):
            with open(path) as fh:
                raw += fh.read()
    for name in ("tab_reportcard", "tab_roster"):
        out.append((("\\input{%s}" % name) in raw,
                    f"bundle: table body {name} is \\input, not hand-typed", ""))
    try:
        from audit.pipeline.emit_tables import build_reportcard, build_roster
        for name, fn in (("tab_reportcard", build_reportcard),
                         ("tab_roster", build_roster)):
            path = os.path.join(HERE, "paper", f"{name}.tex")
            cur = open(path).read() if os.path.exists(path) else None
            fresh = fn()
            out.append((cur == fresh, f"paper: {name}.tex is current with its source CSVs",
                        "" if cur == fresh else
                        "regenerate: python -m audit.pipeline.emit_tables"))
            out.append((cur is not None and cur.startswith("% GENERATED"),
                        f"paper: {name}.tex still carries its do-not-edit header", ""))
    except Exception as ex:                                        # noqa: BLE001
        out.append((False, "emit_tables importable and runnable", f"{type(ex).__name__}: {ex}"))
    return out


def check_no_selective_quotation():
    """Universal claims must hold on ALL rows of their slice, not the quoted subset.

    This is the second-largest late defect class: 21 confirmed findings, and rounds 25,
    26 and 27 each surfaced one. The shape is always the same -- a passage quotes the
    rows of a CSV that agree with it and generalises over the rest. Examples it is built
    from: four learners were run on the NT `enhancers` task and three were quoted as
    "all excluding zero" while the omitted fourth was null; and a threshold-robustness
    claim that is true on human_enhancers_ensembl (drop flat at 0.156/0.156/0.157) and
    false on human_nontata_promoters (0.205/0.121/0.006).

    Two mechanisms, because the class has two mechanisms:

    DENOMINATOR -- when the manuscript reports a count of rows satisfying some property,
    the slice's TOTAL must appear near it. "three of four excluded zero" is honest;
    "three excluded zero" is where the fourth disappears. Asserting the denominator is
    present is what converts this from a lens someone has to think to write into a gate.

    SCOPE -- when a property holds on one dataset in a slice and not another, any
    passage asserting it must name the dataset. A bare universal is a defect even if
    every number it quotes is correct.
    """
    out = []

    # -- DENOMINATOR: NT cross-suite per-task drops -------------------------------
    cs = csv("crosssuite_ranking.csv")
    for task in ("enhancers",):
        sl = cs[(cs.suite == "NT-original") & (cs.task == task)]
        if sl.empty:
            out.append((False, f"crosssuite slice {task} present", "task missing from CSV"))
            continue
        n_tot = len(sl)
        n_excl = int(sl["drop_excl0"].astype(str).str.lower().eq("true").sum())
        body = " ".join(doc("paper").split())
        # find the passage; require BOTH the count and the denominator to be stated
        anchor = body.find(r"on \textit{" + task + "}")
        window = body[anchor:anchor + 700] if anchor >= 0 else ""
        # Require an explicit "<n_excl> of (the) <n_tot>" construction. Testing for the
        # bare digits is useless -- "4" occurs inside 0.0413 and "3" inside 0.0366, so a
        # substring test passes on a passage that states no denominator at all. The
        # self-test caught exactly that and reported this check as BLIND.
        WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine"}
        a = f"(?:{n_excl}|{WORD.get(n_excl, n_excl)})"
        b = f"(?:{n_tot}|{WORD.get(n_tot, n_tot)})"
        stated = re.search(rf"\b{a}\s+of\s+(?:the\s+)?{b}\b", window, re.I) is not None
        ok = (n_excl == n_tot) or stated
        out.append((ok,
                    f"paper: NT {task} drops state the denominator "
                    f"({n_excl} of {n_tot} exclude zero)",
                    "" if ok else
                    f"{n_tot - n_excl} model(s) omitted; no '{n_excl} of {n_tot}' construction"))

    # -- SCOPE: threshold sensitivity is NOT uniform across datasets --------------
    ts = csv("threshold_sensitivity.csv")
    rf6 = ts[(ts.model == "RF") & (ts.k == 6)]
    spread = {}
    for d, g in rf6.groupby("dataset"):
        v = g.sort_values("threshold")["delta_acc"].to_numpy()
        spread[d] = float(v.max() - v.min())
    # the claim "the sweep changes no verdict" is only safe where the spread is small;
    # assert the manuscript does not make it unscoped where a dataset disagrees
    uneven = {d: s for d, s in spread.items() if s > 0.05}
    body = " ".join(doc("paper").split())
    if uneven:
        named = all(tex_dataset(d) in body for d in uneven)
        out.append((named,
                    "paper: datasets whose threshold sweep is uneven are named, not "
                    "absorbed into a blanket robustness claim",
                    "" if named else f"unnamed: {sorted(uneven)}"))
        # and the blanket phrasing must not appear without a scope word nearby
        low = body.lower()
        for phrase in ("changes no verdict", "robust to the clustering threshold"):
            # Every occurrence, not just the first: a scoped use early in the paper
            # does not license an unscoped one later.
            for m in re.finditer(re.escape(phrase), low):
                i, j = m.start(), m.end()
                # The window must EXCLUDE the matched phrase. Centring a window on the
                # match and then searching it for a scope word is tautological whenever
                # the phrase contains one -- "changes no verdict" contains "verdict", so
                # this assertion passed unconditionally and tested nothing.
                win = low[max(0, i - 300):i] + " " + low[j:j + 300]
                scoped = any(w in win for w in ("verdict", "leaky", "clean"))
                out.append((scoped,
                            f"paper: '{phrase}' is scoped to what the sweep actually preserves",
                            "" if scoped else "reads as a blanket robustness claim"))
    return out


def tex_dataset(d):
    return d.replace("_", r"\_")


def check_detector_specificity():
    """The measured operating characteristic, and the hashFrag head-to-head.

    These replaced an argument that specificity could not be measured, so they are the
    numbers most likely to be read closely by a referee and the ones with no prior
    guard at all.
    """
    m = csv("detector_specificity_measured.csv")
    pooled = m[(m.band == "POOLED") & (m.ground_truth == "homolog")].set_index("dataset")
    out = []
    for d, short in (("human_enhancers_ensembl", "ensembl"),
                     ("human_nontata_promoters", "nontata")):
        r = pooled.loc[d]
        out.append(near("paper", r"Clopper--Pearson lower bounds", float(r.precision_lo),
                        places=3, label=f"{short} precision lower bound"))
        out.append((abs(float(r.precision) - 1.0) < 1e-9,
                    f"{short}: measured precision is exactly 1.000",
                    f"{r.precision}"))
        out.append((float(r.fpr) == 0.0, f"{short}: measured per-pair FPR is 0",
                    f"{r.fpr}"))
    # the recall pair, which is the number that bounds the paper's own scope claim
    out.append(near("paper", r"Recall runs the other way",
                    float(pooled.loc["human_enhancers_ensembl"].recall), places=3,
                    label="ensembl measured recall"))

    s = csv("hashfrag_sweep.csv")
    w = s[s.threshold == 220].set_index("dataset")
    agree = int(w.verdicts_agree.sum())
    out.append((agree == 7, "hashFrag reproduces 7 of 8 verdicts in the agreement window",
                f"{agree}/8 at threshold 220"))
    out.append((w.loc["drosophila_enhancers_stark", "verdicts_agree"] == 0,
                "the sole disagreement is drosophila_enhancers_stark", ""))
    for d, short in (("human_enhancers_ensembl", "ensembl"),
                     ("human_nontata_promoters", "nontata")):
        out.append(near("paper", r"a window at \$220\$--\$240\$", float(w.loc[d, "kappa"]),
                        places=3, label=f"{short} hashFrag kappa", span=620))
        out.append(near("paper", r"a window at \$220\$--\$240\$",
                        float(w.loc[d, "agreement"]), places=3,
                        label=f"{short} hashFrag agreement", span=620))
    # the strand bound, stated in the supplement
    out.append((float(w.rev_only_frac.max()) <= 0.0100 + 1e-9,
                "reverse-strand-only leakage is at most 0.0100 on every dataset",
                f"max {w.rev_only_frac.max():.4f}"))
    out.append(near("paper", r"qualifying hit is on the reverse strand",
                    float(w.loc["human_enhancers_ensembl", "rev_only_frac"]), places=4,
                    label="ensembl reverse-strand-only share"))

    # canonical census, suite-wide: the claim is that NO verdict moves, so assert the
    # property rather than the sentence -- a rewrite of the prose must not silently
    # retire it.
    cc = csv("canonical_census_suite.csv")
    out.append((int(cc.verdict_moved.sum()) == 0,
                "canonicalization moves no verdict on any of the eight datasets",
                f"{int(cc.verdict_moved.sum())} moved"))
    out.append((cc.dataset.nunique() == 8,
                "the canonical census covers the whole suite, not just the leaky pair",
                f"{cc.dataset.nunique()} datasets"))
    jac = cc[(cc.metric == "jaccard") & (cc.threshold == 0.7)]
    out.append(near("paper", r"largest Jaccard movement anywhere", float(jac.delta.max()),
                    places=4, label="largest canonical Jaccard movement"))
    con = cc[(cc.metric == "containment") & (cc.threshold == 0.7)]
    out.append(near("paper", r"largest containment movement", float(con.delta.max()),
                    places=4, label="largest canonical containment movement"))
    return out


def check_decomposition_and_spread():
    """B1's three-arm decomposition and B3's two spread controls.

    Both replaced load-bearing claims, and B3's difficulty control WEAKENED one, so
    these are guarded as properties as well as values: a later edit must not quietly
    restore "the benchmark is blind to enormous real differences", which the
    difficulty-matched arm does not support.
    """
    out = []
    d = pd.read_csv(os.path.join(R, "eval_side_inflation.csv"))
    ens = d[(d.dataset == "human_enhancers_ensembl") & (d.model == "RF")].iloc[0]
    ntp = d[(d.dataset == "human_nontata_promoters") & (d.model == "RF")].iloc[0]
    # the text states these in PERCENTAGE POINTS ("8.98 of the forest's 16.4 points"),
    # so the check has to compare in the same units the sentence uses
    out.append(near("paper", r"accounts for", 100 * float(ens.eval_side_inflation),
                    places=2, label="ensembl RF evaluation-side inflation", span=520))
    out.append(near("paper", r"accounts for the remaining", 100 * float(ens.split_side_effect),
                    places=2, label="ensembl RF split-side effect", span=300))
    out.append((abs(float(ens.eval_side_inflation) + float(ens.split_side_effect)
                    - float(ens.total_drop)) < 5e-4,
                "the two arms of the decomposition sum to the total drop", ""),)
    out.append(near("paper", r"the same decomposition gives", 100 * float(ntp.eval_side_inflation),
                    places=2, label="nontata RF evaluation-side inflation", span=200))
    # the memorizer is the only model paying a retraining penalty on ensembl
    e = d[d.dataset == "human_enhancers_ensembl"].set_index("model")
    out.append((float(e.loc["RF", "split_side_effect"]) > 0
                and all(float(e.loc[m, "split_side_effect"]) < 0
                        for m in ("LR", "LinearSVC", "HGB")),
                "only the forest pays a split-side penalty on ensembl", ""))
    # the raw evaluation-side figure is prevalence-confounded; the balanced one is not
    out.append(near("paper", r"From balanced accuracy, for which a constant classifier",
                    float(ens.eval_side_inflation_balanced), places=4, signed=True,
                    label="ensembl RF balanced evaluation-side inflation"))

    s = pd.read_csv(os.path.join(R, "spread_control.csv"))
    clean = s[(s.experiment == "roster_spread") & (~s.leaky)]
    for dset in ("human_enhancers_cohn", "demo_human_or_worm"):
        g = clean[clean.dataset == dset]
        a = g[g.arm.str.startswith("as_shipped_split/as_shipped")].iloc[0]
        c = g[g.arm.str.startswith("corrected_split/as_shipped")].iloc[0]
        out.append((float(c.top3_spread) <= float(a.top3_spread) + 1e-9,
                    f"{dset}: re-splitting does not expand top-3 spread",
                    f"{a.top3_spread} -> {c.top3_spread}"))
    m = s[s.experiment == "difficulty_matched"]
    if len(m):
        dm = m[m.arm.str.contains("difficulty_matched_to")].iloc[0]
        out.append(near("paper", r"their spread expands from \$0\.0015\$",
                        float(dm.top3_spread), places=4,
                        label="difficulty-matched top-3 spread"))
        out.append((float(dm.top3_spread) > 0.5 * 0.2314,
                    "the difficulty control explains most of the spread expansion, "
                    "so the strong reading is not claimed",
                    f"matched {dm.top3_spread} vs corrected 0.2314"))
    # says(), not a literal `in`: the response letter wraps this phrase across a line
    # ("blind to\n   enormous real differences"), so the substring form reported ABSENT
    # for a sentence that was present -- the exact vacuity this module's docstring warns
    # about, reintroduced by a check written to catch a retired claim.
    for k in ("paper", "response", "cover"):
        out.append(says(k, "blind to enormous real differences", want=False,
                        label=f"retired in {k}: 'blind to enormous real differences' "
                              "(the difficulty-matched control does not support it)"))
    return out


def check_dose_replication():
    """C6: the inversion threshold under replication, and P6's refutation.

    This one is guarded hard because it runs AGAINST the paper. The single-fit sweep
    reported a three-digit threshold and "none below it"; 9 of 15 replicated trials
    below that value inverted. A later edit must not restore the sharp reading.
    """
    d = csv("dose_replication_summary.csv")
    g = d[d.dose_target_share != "THRESHOLD_SUMMARY"].copy()
    g["phi_mean"] = g.phi_mean.astype(float)
    out = []
    for _, r in g.sort_values("phi_mean").iterrows():
        out.append(near("paper", r"rate rises from", float(r.inversion_prob), places=2,
                        label=f"inversion probability at phi={r.phi_mean:.3f}", span=420))
    # the intervals overlap, so no breakpoint is identifiable
    los = g.split_cluster_ci_lo.astype(float)
    his = g.split_cluster_ci_hi.astype(float)
    out.append((float(los.max()) < float(his.min()),
                "the split-clustered intervals overlap across all three doses, "
                "so no threshold is identifiable",
                f"max lo {los.max():.2f} < min hi {his.min():.2f}"))
    summ = d[d.dose_target_share == "THRESHOLD_SUMMARY"]
    if len(summ):
        note = str(summ.iloc[0].per_split_seed_rates)
        n_below = int(note.split("censored_below=")[1].split(";")[0])
        out.append((n_below > 0,
                    "trials inverted BELOW the committed single-fit threshold",
                    f"censored_below={n_below}"))
        out.append(says("paper", "Nine of the fifteen trials below the committed threshold",
                        label="the paper states how many inverted below the threshold"))
    out.append(says("paper", "is therefore refuted on replication",
                    label="P6's refutation on replication is stated"))
    # and the retired sharp reading must not come back
    out.append(says("paper", "and finds none below it", want=False,
                    label="retired: the single-fit 'none below it' claim"))
    return out


def check_supplement_pointers():
    r"""Every "Supplementary \S SN" in the main text must resolve to a real section,
    and to one whose title is plausibly what the sentence is pointing at.

    Cross-document \ref cannot resolve between two separate LaTeX documents, so these
    pointers are typed by hand and silently rot whenever a supplement section is added,
    removed or reordered. Moving seven blocks out of the main text renumbered the
    supplement and left six pointers aimed at the wrong section, with the build
    reporting zero undefined references throughout -- there is nothing for LaTeX to
    catch. This is the only guard on them.
    """
    main = open(DOCS["paper"]).read()
    supp_path = os.path.join(HERE, "paper", "supplementary.tex")
    if not os.path.exists(supp_path):
        return [(True, "no supplement to check", "")]
    titles = re.findall(r"\\section\{(.+?)\}", open(supp_path).read())
    out = [(len(titles) > 0, f"supplement has {len(titles)} numbered sections", "")]
    # keywords that must appear in the target section's title, by section number
    expect = {
        5: ("bend",), 6: ("manipulation", "replication"), 7: ("robustness",),
        10: ("exploratory",), 11: ("pretrained", "overlap-free"),
        4: ("detector", "calibration"), 3: ("alignment",), 2: ("dose", "varphi"),
    }
    seen = set()
    for m in re.finditer(r"Supplementary \\S ?S(\d+)", main):
        n = int(m.group(1))
        seen.add(n)
        ok = 1 <= n <= len(titles)
        out.append((ok, f"Supplementary \\S S{n} resolves to a real section",
                    titles[n - 1][:52] if ok else f"only {len(titles)} sections exist"))
        if ok and n in expect:
            t = titles[n - 1].lower()
            hit = any(k in t for k in expect[n])
            out.append((hit, f"Supplementary \\S S{n} points at the right topic",
                        f"S{n} = {titles[n-1][:46]!r}"))
    return out


CHECKS = [
    ("supplement pointers", check_supplement_pointers),
    ("dose replication / P6", check_dose_replication),
    ("decomposition / spread controls", check_decomposition_and_spread),
    ("detector specificity / hashFrag", check_detector_specificity),
    ("retired claims", check_retired_claims),
    ("generated tables", check_generated_tables),
    ("selective quotation", check_no_selective_quotation),
    ("CNN seed replication / paired interval", check_cnn_seed_replication),
    ("imbalance panel", check_imbalance_panel),
    ("shared numbers across documents", check_shared_numbers),
    ("letter section pointers", check_letter_section_refs),
    ("corrected graded gaps", check_corrected_gaps),
    ("GUE census and pre-registered score", check_gue),
    ("manipulation intervals", check_manipulation),
    ("dose-response / P6", check_dose_response),
    ("interval labelling", check_intervals_are_labelled),
    ("cluster cohesion / C9", check_cohesion),
    ("cross-suite counts", check_crosssuite_counts),
]


def run(verbose=True):
    failed = total = 0
    for group, fn in CHECKS:
        if verbose:
            print(group)
        try:
            results = fn()
        except Exception as ex:                                    # noqa: BLE001
            if verbose:
                print(f"  !! check raised {type(ex).__name__}: {ex}")
            failed += 1
            continue
        for ok, label, detail in results:
            total += 1
            if not ok:
                failed += 1
            if verbose:
                print(f"  [{'ok  ' if ok else 'FAIL'}] {label}"
                      + (f"  -- {detail}" if detail else ""))
        if verbose:
            print()
    return total, failed


def self_test():
    """Rule 3: demonstrate that the gate can actually fail.

    Each mutation corrupts ONE site of a value the manuscript states at several. Under the
    old whole-document matcher every one of these passed. A mutation that does not make
    the gate fail is a check that cannot fail, which is a defect in the gate itself.
    """
    import copy
    # Rule 4 said the gate reads the results documents too; rule 3 says every check must
    # be shown to fail. Until round 14 this function mutated only the manuscript, so no
    # non-manuscript check was ever proved capable of failing. Mutations now name their
    # document.
    base = {k: doc(k) for k in DOCS}
    mutations = [
        ("findings: the NT independent-task count",
         "findings", "tasks of eleven are", "tasks of twelve are"),
        ("findings: the full-scale GUE minimum in the outcome table",
         "findings", "jac@0.7 spans 0.009", "jac@0.7 spans 0.005"),
        ("paper: nonTATA non-memorizer mean at its single site",
         "paper", "memorizers average $+0.122$ against $+0.179$",
         "memorizers average $+0.122$ against $+0.155$"),
        ("paper: ensembl non-memorizer mean where section 4.5 states it",
         "paper", "average $+0.292$ against $+0.165$", "average $+0.292$ against $+0.155$"),
        ("paper: one of the two sites quoting the break-a-clean interval",
         "paper", "$[0.083,0.126]^{*}$", "$[0.084,0.127]^{*}$"),
        ("paper: the competing-rule score in the dose-response caveat",
         "paper", "also scores $10/10$", "also scores $7/10$"),
        ("paper: the GUE Jaccard minimum", "paper", "$0.009$ to $0.041$",
         "$0.011$ to $0.041$"),
        ("paper: the CNN combined-source interval lower bound",
         "paper", "$[0.0019,0.0204]$", "$[0.0031,0.0204]$"),
        ("response: the CNN corrected-accuracy floor, letter side",
         "response", "0.8028–0.8131", "0.8044–0.8131"),
        ("cover: the 4-of-5 disclosure summarised away",
         "cover", "four of the five seeds' paired intervals exclude zero",
         "all five seeds' paired intervals exclude zero"),
        # the selective-quotation class: drop the denominator and the omitted fourth
        # learner disappears, which is exactly the round-26 defect
        ("paper: the NT enhancers denominator removed",
         "paper", "three of the four learners excluding zero",
         "all excluding zero"),
    ]
    print("== self-test: each mutation must make the gate FAIL ==\n")
    bad = 0
    for label, where, old, new in mutations:
        if old not in base[where]:
            print(f"  [SKIP] {label} -- target text not present; check needs updating")
            bad += 1
            continue
        _cache.update(base)
        _cache[where] = base[where].replace(old, new, 1)
        _, failed = run(verbose=False)
        print(f"  [{'ok  ' if failed else 'BLIND'}] {label}"
              + ("" if failed else "  <-- gate did NOT fail; this check is vacuous"))
        if not failed:
            bad += 1
    _cache.update(base)
    print(f"\n{len(mutations) - bad}/{len(mutations)} mutations detected")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="verify the gate can fail, by breaking the manuscript on purpose")
    a = ap.parse_args()

    print("== check_numbers: document claims against committed CSVs ==\n")
    total, failed = run()
    print(f"{total - failed}/{total} checks pass")

    bad = 0
    if a.self_test:
        print()
        bad = self_test()

    if failed or bad:
        print("\nRESULT: FAIL" + ("" if not failed else
              " -- a document number disagrees with its source CSV")
              + ("" if not bad else " -- and the gate has a check that cannot fail"))
        sys.exit(1)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
