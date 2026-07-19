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
    "paper": os.path.join(HERE, "paper", "main.tex"),
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
    if name not in _cache:
        with open(DOCS[name]) as fh:
            _cache[name] = fh.read()
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
    """`text` must appear exactly `n` times. Rule 2: corrupting one site drops the count."""
    got = doc(where).count(text)
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
    out.append(near("paper", r"definitional memorizer", knn, signed=True,
                    label=f"1-NN corrected gap {knn:+.3f}"))
    # Rule 2, applied where round 14 proved a single anchored check was not enough. Each
    # of these values is stated at several sites; an anchored check protects one of them
    # and the count protects the rest, so a stale site drops the count and fails.
    for tok, n, what in (("0.461", 3, "1-NN corrected gap"),
                         ("+0.179", 3, "nonTATA non-memorizer mean"),
                         ("0.370", 2, "ExtraTrees corrected gap"),
                         ("$3$ of $15$", 2, "GUE registered score"),
                         ("$0.96$", 2, "ensembl cohesion")):
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
        near("paper", r"we report the intermediate", lo, places=1,
             label=f"band minimum {lo:.1f}%"),
        near("paper", r"we report the intermediate", hi, places=1,
             label=f"band maximum {hi:.1f}%"),
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
    ]


def check_cohesion():
    g = csv("exp_g11_cohesion.csv").set_index("dataset")
    ens = g.loc["human_enhancers_ensembl", "largest_cluster_cohesion"]
    ntp = g.loc["human_nontata_promoters", "largest_cluster_cohesion"]
    return [
        near("paper", r"cohesion \$0\.\d\d\$ and", ens, places=2,
             label=f"ensembl cohesion {ens:.2f}"),
        near("paper", r"cohesion \$0\.\d\d\$ and", ntp, places=2,
             label=f"nontata cohesion {ntp:.2f}"),
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
    # Results subsections in source order; index+1 is the number after "4."
    marker = "\\section{Results}"
    body = tex[tex.index(marker):] if marker in tex else tex
    end = body.find("\\section{Discussion}")
    if end > 0:
        body = body[:end]
    subs = re.findall(r"\\subsection\{(.+?)\}", body)
    numbering = {f"4.{i+1}": t for i, t in enumerate(subs)}
    expect = [
        ("§4.12", ("alignment", "chromosome", "control")),
        ("§4.13", ("robust",)),
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


CHECKS = [
    ("retired claims", check_retired_claims),
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
        ("paper: the GUE Jaccard minimum", "paper", "from $0.009$ to $0.041$",
         "from $0.011$ to $0.041$"),
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
