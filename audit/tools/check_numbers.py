#!/usr/bin/env python3
"""
Check the manuscript's numbers against the CSVs they come from.

Twelve adversarial audit rounds found that by far the most common defect in this project
is not a wrong computation but a wrong TRANSCRIPTION: a value corrected in one place and
left stale in another. Rounds 10, 11 and 12 were dominated by it -- a gap mean fixed in
one paragraph and not the next, a task count fixed in the paper and not in the results
document, an interval labelled as the cluster bootstrap when it was the combined-source
one. Each was found by an expensive audit agent re-deriving the number by hand.

This module makes that class of defect cheap to find. Every check below states a claim
the manuscript makes, recomputes it from the committed CSV, and fails if they disagree.
It is a regression gate, not a proof of correctness: it can only check claims someone has
registered here. Adding a check when a number enters the paper is the point.

Run:  python -m audit.tools.check_numbers
Exit: 0 if every registered claim matches its source, 1 otherwise.
"""
from __future__ import annotations
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "paper", "main.tex")



def quoted(s, value, places=3, signed=False):
    """True if `value` appears in the TeX at the given precision.

    Two tolerances, both necessary and neither a loosening of the check. First we match
    the bare numeral rather than requiring `$N$`, because the paper legitimately writes
    it inside a larger math group (`$\\varphi^{*}=0.060$`). Second we accept either
    neighbouring rounding when the source value sits exactly on a half -- 0.1215 and
    0.1055 are both exact halves, and round-half-up (0.122, 0.106) and Python's
    float-repr rounding (0.121, 0.105) disagree there. Anything further from the source
    than one unit in the last place still fails.
    """
    import math
    cands = set()
    for v in (value, math.nextafter(value, math.inf), math.nextafter(value, -math.inf)):
        t = f"{v:+.{places}f}" if signed else f"{v:.{places}f}"
        cands.add(t)
        if not signed:
            cands.add(t.lstrip("0") if t.startswith("0.") else t)
    return any(c in s for c in cands)

def tex():
    with open(PAPER) as fh:
        return fh.read()


def csv(name):
    return pd.read_csv(os.path.join(R, name))


# ---------------------------------------------------------------------------
# Each check returns (ok, label, detail).
# ---------------------------------------------------------------------------
def check_corrected_gaps():
    """The prevalence-corrected graded gaps quoted in section 4.5."""
    g = csv("graded_gap_corrected.csv")
    s = tex()
    out = []
    MEM = {"kNN-1", "kNN-15", "RF", "ExtraTrees"}
    for dset, tag in (("human_enhancers_ensembl", "ensembl"),
                      ("human_nontata_promoters", "nontata")):
        d = g[g.dataset == dset]
        mem = d[d.model.isin(MEM)].graded_gap_balanced.mean()
        rest = d[~d.model.isin(MEM)].graded_gap_balanced.mean()
        # the paper must quote the mean over ALL non-memorizers, not a subset
        out.append((quoted(s, mem, signed=True), f"{tag} memorizer mean {mem:+.3f}",
                    "quoted in main.tex" if quoted(s, mem, signed=True) else "NOT FOUND in main.tex"))
        out.append((quoted(s, rest, signed=True), f"{tag} non-memorizer mean {rest:+.3f}",
                    "quoted in main.tex" if quoted(s, rest, signed=True) else "NOT FOUND in main.tex"))
        # and the subset-excluding-MLP value must NOT appear, since that was the old bug
        bad = d[~d.model.isin(MEM | {"MLP"})].graded_gap_balanced.mean()
        out.append((not quoted(s, bad, signed=True),
                    f"{tag} MLP-excluded mean {bad:+.3f} absent",
                    "absent, correct" if not quoted(s, bad, signed=True)
                    else "PRESENT -- this is the mean computed by dropping the MLP"))
    return out


def check_gue_range():
    """The GUE leak-fraction range quoted in the cross-suite section."""
    g = csv("gue_census.csv")
    r = g[(g.registered) & (g.preregistered == "LEAKY")]
    lo, hi = r.leak_jaccard_0p7.min(), r.leak_jaccard_0p7.max()
    clo, chi = r.leak_containment_0p7.min(), r.leak_containment_0p7.max()
    s = tex()
    return [
        (f"${lo:.3f}$" in s, f"GUE Jaccard minimum {lo:.3f}",
         "quoted" if f"${lo:.3f}$" in s else "NOT FOUND -- an earlier draft quoted 0.011"),
        (f"${hi:.3f}$" in s, f"GUE Jaccard maximum {hi:.3f}", "quoted"),
        (f"${clo:.3f}$" in s, f"GUE containment minimum {clo:.3f}", "quoted"),
        (f"${chi:.3f}$" in s, f"GUE containment maximum {chi:.3f}", "quoted"),
        (bool(g.train_frac_used.eq(1.0).all()),
         "GUE census is full scale on every row",
         "all rows train_frac_used=1.0" if g.train_frac_used.eq(1.0).all()
         else "SOME ROWS ARE CAPPED -- their verdicts are lower bounds"),
    ]


def check_gue_score():
    """The pre-registered GUE tally. Unregistered tasks must never enter it."""
    g = csv("gue_census.csv")
    reg = g[g.registered]
    correct = int(reg.prereg_correct.fillna(False).astype(bool).sum())
    leaky = reg[reg.preregistered == "LEAKY"]
    s = tex()
    return [
        (len(reg) == 15, f"15 registered GUE tasks (found {len(reg)})", ""),
        (correct == 3, f"registered score {correct}/15", "must be 3/15"),
        (int(leaky.prereg_correct.fillna(False).astype(bool).sum()) == 0,
         "0 of 11 predicted-LEAKY correct", ""),
        (bool(g[~g.registered].prereg_correct.isna().all()),
         "unregistered tasks carry no score",
         "mouse_0/mouse_1 must have empty prereg_correct"),
        (f"${correct}$ of ${len(reg)}$" in s or f"{correct} of {len(reg)}" in s,
         f"main.tex quotes {correct} of {len(reg)}", ""),
    ]


def check_manipulation_intervals():
    """Table 4's intervals, and the disjointness the text claims."""
    m = csv("construction_manipulation.csv").set_index("condition")
    s = tex()
    out = []
    for cond, label in (("B_control_unmerged_union", "fix-a-leaky control"),
                        ("B_manipulated_merged_balanced", "fix-a-leaky intervention"),
                        ("A_control_merged_consensus", "break-a-clean control"),
                        ("A_manipulated_matched", "break-a-clean intervention")):
        ci = str(m.loc[cond, "rf_drop_ci"])
        lo, hi = [float(x) for x in re.findall(r"-?\d+\.\d+", ci)[:2]]
        out.append((f"[{lo:.3f},{hi:.3f}]" in s.replace(" ", ""),
                    f"{label} CI [{lo:.3f},{hi:.3f}]",
                    "quoted" if f"[{lo:.3f},{hi:.3f}]" in s.replace(" ", "") else "NOT FOUND"))
    b_lo = float(re.findall(r"-?\d+\.\d+",
                            str(m.loc["B_control_unmerged_union", "rf_drop_ci"]))[0])
    b_hi = float(re.findall(r"-?\d+\.\d+",
                            str(m.loc["B_manipulated_merged_balanced", "rf_drop_ci"]))[1])
    a_hi = float(re.findall(r"-?\d+\.\d+",
                            str(m.loc["A_control_merged_consensus", "rf_drop_ci"]))[1])
    a_lo = float(re.findall(r"-?\d+\.\d+",
                            str(m.loc["A_manipulated_matched", "rf_drop_ci"]))[0])
    out.append((b_lo > b_hi, "fix-a-leaky intervals disjoint",
                f"{b_lo:.4f} > {b_hi:.4f}" if b_lo > b_hi else "OVERLAP"))
    out.append((a_lo > a_hi, "break-a-clean intervals disjoint",
                f"{a_lo:.4f} > {a_hi:.4f}" if a_lo > a_hi else "OVERLAP"))
    return out


def check_dose_response():
    """P6, and the band size the Methods promises to report wherever eq. (2) is used."""
    d = csv("dose_response.csv")
    s = tex()
    ok = int(d.prediction_correct.sum())
    flip = d[d.ranking_inverts]
    out = [
        (ok == len(d), f"P6 holds on {ok}/{len(d)} doses", ""),
        (len(d) == 10, f"ten doses committed (found {len(d)})", ""),
    ]
    if len(flip):
        phi, ps = flip.iloc[0].phi_used, flip.iloc[0].phi_star
        out.append((quoted(s, phi), f"flip-point phi {phi:.3f} quoted", ""))
        out.append((quoted(s, ps), f"flip-point phi* {ps:.3f} quoted", ""))
    if "mid_band_frac" in d.columns:
        lo, hi = d.mid_band_frac.min() * 100, d.mid_band_frac.max() * 100
        out.append((f"${lo:.1f}$--${hi:.1f}\\%$" in s,
                    f"intermediate band {lo:.1f}-{hi:.1f}% reported", ""))
    # the trivial competing rule must still be disclosed
    triv = int(((d.rf_rank_orig == 1) == d.ranking_inverts).sum())
    out.append(("also scores $10/10$" in s or "also scores" in s,
                f"competing rule (RF leads) scores {triv}/{len(d)}, disclosed in text",
                "disclosed" if "also scores" in s else "NOT DISCLOSED"))
    return out


def check_cohesion():
    """The C9 cut and the two datasets it is calibrated on."""
    g = csv("exp_g11_cohesion.csv").set_index("dataset")
    s = tex()
    ens = g.loc["human_enhancers_ensembl", "largest_cluster_cohesion"]
    ntp = g.loc["human_nontata_promoters", "largest_cluster_cohesion"]
    return [
        (f"${ens:.2f}$" in s, f"ensembl cohesion {ens:.2f} quoted", ""),
        (f"${ntp:.2f}$" in s, f"nontata cohesion {ntp:.2f} quoted", ""),
        (ens >= 0.5 > ntp, "the two datasets straddle the 0.5 cut",
         f"{ens:.3f} >= 0.5 > {ntp:.3f}"),
    ]


def check_crosssuite_counts():
    """The Nucleotide Transformer independent-task count, after both nestings collapse."""
    c = csv("crosssuite_census.csv")
    nt = c[c.suite == "NT-original"]
    # enhancers_types duplicates enhancers; promoter_all is the union of the other two
    independent = len(nt) - 2
    leaky = nt[(nt.verdict == "LEAKY") & (nt.task != "enhancers_types")]
    s = tex()
    return [
        (len(nt) == 13, f"13 NT-original tasks shipped (found {len(nt)})", ""),
        (independent == 11, f"{independent} independent after collapsing both nestings", ""),
        (len(leaky) == 3, f"{len(leaky)} leaky independent tasks", ""),
        ("three of eleven" in s, "main.tex says three of eleven",
         "correct" if "three of eleven" in s else "STALE -- says twelve somewhere"),
        ("three of twelve" not in s, "main.tex no longer says three of twelve", ""),
    ]


CHECKS = [
    ("corrected graded gaps", check_corrected_gaps),
    ("GUE leak range", check_gue_range),
    ("GUE pre-registered score", check_gue_score),
    ("manipulation intervals", check_manipulation_intervals),
    ("dose-response / P6", check_dose_response),
    ("cluster cohesion / C9", check_cohesion),
    ("cross-suite counts", check_crosssuite_counts),
]


def main():
    print("== check_numbers: manuscript claims against committed CSVs ==\n")
    failed = 0
    total = 0
    for group, fn in CHECKS:
        print(f"{group}")
        try:
            results = fn()
        except Exception as ex:                                    # noqa: BLE001
            print(f"  !! check raised {type(ex).__name__}: {ex}")
            failed += 1
            continue
        for ok, label, detail in results:
            total += 1
            mark = "ok  " if ok else "FAIL"
            if not ok:
                failed += 1
            print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
        print()
    print(f"{total - failed}/{total} checks pass")
    if failed:
        print("\nRESULT: FAIL -- a manuscript number disagrees with its source CSV, or a "
              "value the manuscript should quote was not found in it.")
        sys.exit(1)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
