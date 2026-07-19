#!/usr/bin/env python3
"""
`certify` -- an executable homology-leakage certification standard (Deepener #6-A).

Turns the paper's one-suite audit into a reusable BUILD-and-CERTIFY protocol that any
benchmark author can run before shipping a split. It wraps modules already in this
repo, so a certification is exactly the audit the paper performed.

THE NINE CERTIFY CHECKS (each maps to a repo module and to a failure this audit hit)
------------------------------------------------------------------------------------
 C1 full scale        Run on the FULL dataset, never a subsample. demo_coding flips
                      clean->borderline ONLY at full scale.      [full_scale_containment]
 C2 two metrics       Report BOTH length-blind k-mer Jaccard AND length-robust
                      containment. Ensembl is 0.38 by Jaccard but 0.85 by containment
                      because its intervals vary 2-573 bp.        [exp_geometry]
 C3 alignment check   Re-cluster with an alignment engine (MMseqs2 / BLASTn) and
                      confirm the verdict.                        [exp_alignment]
 C4 memorizer tripwire Include a memorization-prone model (default RF). A benchmark
                      whose ranking is stable ONLY for regularized models is not clean.
                                                                  [run_extended_models]
 C5 graded gap        Certify on the confound-immune graded memorization gap
                      acc[>=0.9]-acc[<0.5] + nearest-neighbour label concordance,
                      NOT on accuracy alone.        [run_graded, check1_label_concordance]
 C6 random control    A matched random re-split must reproduce the original score;
                      otherwise the "drop" is a split-size artefact.  [run_robustness_full]
 C7 cluster CIs       Every delta carries a cluster/block-bootstrap CI over whole
                      within-test near-duplicate components, with ICC + design effect.
                                                                  [cluster_bootstrap]
 C8 coordinate check  Where genomic coordinates ship, the max-over-class share of TEST
                      intervals overlapping a TRAIN interval at >=50% reciprocal overlap
                      must be < 0.1. Catches the cause upstream of sequence space and
                      is independent of any k-mer choice.          [exp_construction]
 C9 cluster cohesion  Whether whole-cluster re-splitting is a legitimate correction at
                      all. Single linkage chains -- A~B, B~C puts unrelated A and C in
                      one component -- so on a chained dataset "correcting" the split
                      quarantines a gene family and removes real signal. Below a cohesion
                      of 0.5 the verdict becomes leaky-but-correction-unsafe and the tool
                      directs the user to the novel-stratum readout on the as-shipped
                      split instead.                                    [exp_geometry G11]

Green verdicts carry the heavier evidential burden: a dataset is only certified CLEAN
when every applicable check passes, whereas ONE failing check is enough to flag it.

USAGE
-----
  python -m audit.tools.certify --self-validate
      Reproduce the paper's 8-dataset verdicts from committed CSVs (seconds, no
      refitting). This is the repo's regression gate: it FAILS loudly on drift.

  python -m audit.tools.certify --fasta X.fa --labels y.txt [--splits s.json]
      Certify an arbitrary dataset end to end. --splits accepts either key spelling,
      so a split written by audit/core/homology_split.py loads directly.

  python -m audit.tools.certify --dataset D --emit-splits corrected.json
      Certify, and hand back the corrected near-duplicate-aware partition the
      certification itself used, so retraining does not re-derive it with different
      parameters.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(HERE, "results")

# The verdicts the paper reports (paper/main.tex:122,204). --self-validate must
# reproduce these exactly, by rule, from committed CSVs.
EXPECTED = {
    "human_nontata_promoters": "LEAKY",
    "human_enhancers_ensembl": "LEAKY",
    "demo_coding_vs_intergenomic_seqs": "borderline",
    "human_ocr_ensembl": "clean",
    "demo_human_or_worm": "clean",
    "drosophila_enhancers_stark": "clean",
    "human_enhancers_cohn": "clean",
    "human_ensembl_regulatory": "clean",
}
LEAK_CUT = 0.1          # the repo's existing cut (report_card.py:49); not re-tuned


def verdict_from(jac, con, coord=None):
    """The composed rule. Jaccard is a LOWER BOUND (it penalises length mismatch), so a
    dataset clean by Jaccard but leaky by length-robust containment is `borderline`,
    not clean -- this is the demo_coding case that a Jaccard-only rule misses.
    A coordinate signal, when available, can only ESCALATE (never clear) a verdict."""
    v = "LEAKY" if jac > LEAK_CUT else ("borderline" if (con is not None and con > LEAK_CUT)
                                        else "clean")
    if coord is not None and coord > LEAK_CUT and v == "clean":
        v = "borderline"
    return v


# ---------------------------------------------------------------------------
# self-validation: recompose the published verdicts from committed CSVs
# ---------------------------------------------------------------------------
def self_validate(verbose=True):
    rc = pd.read_csv(f"{R}/leakage_report_card.csv")
    rc["dataset"] = rc["dataset"].str.replace(" (3-class)", "", regex=False)
    rc = rc.set_index("dataset")
    # C1: containment MUST come from the FULL-scale run. exp_gb1_containment.csv is a
    # 20k subsample for non-leaky sets and reports demo_coding at 0.043 -> would
    # wrongly certify it clean. This is CERTIFY check C1 in action.
    fsc = pd.read_csv(f"{R}/full_scale_containment.csv").set_index("dataset")
    gb1 = pd.read_csv(f"{R}/exp_gb1_containment.csv").set_index("dataset")
    coord = None
    cpath = f"{R}/construction_rule.csv"
    if os.path.exists(cpath):
        coord = pd.read_csv(cpath).set_index("dataset")

    rows, ok = [], True
    for d, want in EXPECTED.items():
        jac = float(rc.loc[d, "leak_at_0p7"])
        if d in fsc.index:
            con, src = float(fsc.loc[d, "leak_con_0p7"]), "full_scale_containment"
        elif d in gb1.index:
            con, src = float(gb1.loc[d, "leak_contain_0p7"]), "exp_gb1_containment(full)"
        else:
            con, src = None, "unavailable"
        co = float(coord.loc[d, "xsplit_ge50pct_max"]) if (
            coord is not None and d in coord.index) else None
        got = verdict_from(jac, con, co)
        rows.append(dict(dataset=d, leak_jaccard_0p7=jac, leak_containment_0p7=con,
                         containment_source=src, coord_xsplit_ge50pct=co,
                         verdict=got, expected=want, match=bool(got == want)))
        ok &= got == want
    df = pd.DataFrame(rows)
    if verbose:
        print("== certify --self-validate ==")
        print(df.to_string(index=False))
        n = int(df.match.sum())
        print(f"\nreproduced {n}/{len(df)} published verdicts "
              f"({(df.verdict=='LEAKY').sum()} LEAKY / "
              f"{(df.verdict=='borderline').sum()} borderline / "
              f"{(df.verdict=='clean').sum()} clean)")
        print("RESULT:", "PASS" if ok else "FAIL")
        # C9 is computed only by a full certify_dataset() run, not from the committed
        # CSVs, so this gate cannot see it. Say so, or a reader comparing the two paths
        # will read the refinement as drift.
        print("\nnote: --self-validate reproduces the published report-card verdicts "
              "(LEAKY / borderline / clean) by rule from committed CSVs. A full "
              "certify run additionally applies C9 (cluster cohesion), which refines "
              "human_nontata_promoters from LEAKY to leaky-but-correction-unsafe "
              "(cohesion 0.084). That is a refinement of the same verdict, not a "
              "disagreement with it.")
    df.to_csv(f"{R}/certify_self_validation.csv", index=False)
    return ok, df


# ---------------------------------------------------------------------------
# engine agreement (C3)
# ---------------------------------------------------------------------------
def _have(tool):
    return shutil.which(tool) is not None


def mmseqs_clusters(seqs, min_seq_id=0.7, cov=0.8, threads=4):
    """Alignment-based clustering via MMseqs2 easy-cluster. Returns per-sequence
    cluster ids, or None when the binary is absent (certify degrades, never crashes --
    exp_alignment.py:29 crashes instead, which is why this wrapper exists)."""
    if not _have("mmseqs"):
        return None
    tmp = tempfile.mkdtemp(prefix="certify_mmseqs_")
    try:
        fa = os.path.join(tmp, "in.fasta")
        with open(fa, "w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">s{i}\n{s}\n")
        pref = os.path.join(tmp, "clu")
        cmd = ["mmseqs", "easy-cluster", fa, pref, os.path.join(tmp, "tmp"),
               "--min-seq-id", str(min_seq_id), "-c", str(cov),
               "--threads", str(threads), "-v", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        tsv = pref + "_cluster.tsv"
        if p.returncode != 0 or not os.path.exists(tsv):
            print(f"  [C3] mmseqs failed (rc={p.returncode}); skipping alignment check")
            return None
        rep = {}
        out = np.full(len(seqs), -1, int)
        with open(tsv) as fh:
            for line in fh:
                a, b = line.split()
                out[int(b[1:])] = rep.setdefault(a, len(rep))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# full certification of an arbitrary dataset
# ---------------------------------------------------------------------------
def certify_dataset(seqs, y, tr, te, name="dataset", boot=2000, seed=0,
                    full_scale_n=None):
    """Certify one dataset. `full_scale_n` is the size of the UNSUBSAMPLED dataset when
    the caller has capped it; C1 then reports that the run is not full scale and any
    clean verdict is provisional (subsampling removes near-duplicate partners, so it can
    only under-report leakage -- this repo's demo_coding case)."""
    from audit.core import expkit as E
    from audit.core import homology_split as H
    t0 = time.time()
    y = np.asarray(y)
    # full_scale_n=None means the caller could not tell us the true dataset size, so C1
    # is UNVERIFIABLE -- not "pass". Defaulting it to len(seqs) would compare the input
    # to itself and affirmatively certify full scale having checked nothing, which is
    # exactly the inert-check failure this tool exists to catch.
    c1_unverifiable = full_scale_n is None
    rep = dict(dataset=name, n=len(seqs), n_train=len(tr), n_test=len(te),
               len_min=int(min(map(len, seqs))), len_med=int(np.median([len(s) for s in seqs])),
               len_max=int(max(map(len, seqs))),
               class_balance=round(float(np.mean(y == y.max())), 4))

    # C2 -- both metrics, on the AS-SHIPPED split (this is what the benchmark ships with)
    sim_j = E.max_sim_to_train(seqs, tr, te, k=8, mode="jaccard")
    sim_c = E.max_sim_to_train(seqs, tr, te, k=8, mode="containment")
    for tag, s in (("jaccard", sim_j), ("containment", sim_c)):
        rep[f"leak_{tag}_0p7"] = round(float((s > 0.7).mean()), 4)
        rep[f"leak_{tag}_0p9"] = round(float((s >= 0.9).mean()), 4)
    exact = len(seqs) - len(set(seqs))
    rep["exact_duplicate_rows"] = int(exact)
    rep["exact_dup_frac"] = round(exact / len(seqs), 4)

    # C5 -- graded memorization gap + label concordance (confound-immune)
    X = E.featurize(seqs, 6)
    novel, high = sim_j < 0.5, sim_j >= 0.9
    rep["phi_leak_fraction"] = round(float(high.mean()), 4)
    # The graded gap is only interpretable when the high-similarity bin is populated.
    # run_graded.py flags bins with n < 50 as low_confidence; the tripwire must respect
    # that or it fires on a handful of sequences (drosophila_stark has ~10 there, and a
    # spurious 0.20 "gap" escalated a genuinely clean set before this guard existed).
    MIN_N = 50
    rep["n_high_bin"], rep["n_novel_bin"] = int(high.sum()), int(novel.sum())
    rep["graded_gap_low_confidence"] = bool(high.sum() < MIN_N)
    gaps = {}
    for m in ("RF", "LR"):                      # C4: RF is the memorizer tripwire
        c = E.correctness(E.models()[m], X, y, tr, te)
        n_acc = float(c[novel].mean()) if novel.any() else np.nan
        h_acc = float(c[high].mean()) if high.any() else np.nan
        gaps[m] = h_acc - n_acc
        rep[f"{m}_acc_orig"] = round(float(c.mean()), 4)
        rep[f"{m}_novel_acc"] = round(n_acc, 4)
        rep[f"{m}_graded_gap"] = round(gaps[m], 4) if not np.isnan(gaps[m]) else None
    rep["memorizer_tripwire_gap"] = rep["RF_graded_gap"]

    # C7 -- corrected split + cluster-bootstrap CI on the drop
    comp = E.clusters(seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, seed, len(te) / len(seqs))
    ver = H.verify_split(seqs, ctr, cte, 0.7, 8)
    rep["residual_leak_after_split"] = round(float(ver["residual_leak_fraction"]), 6)
    # The corrected split has a DIFFERENT test set, so it needs its own blocks and must
    # be resampled too. Holding the corrected accuracy fixed and bootstrapping only the
    # original arm drops half the variance (SE understated by ~1/sqrt(2)) and turns null
    # drops significant -- an earlier version of this function reported drosophila_stark
    # as [0.0019, 0.0434] against the frozen [-0.0067, 0.0539]. We therefore call the
    # repo's frozen two-arm estimator rather than reimplementing it.
    from audit.experiments.cluster_bootstrap import (test_internal_clusters, delta_boot,
                                                  BOOT as _FROZEN_BOOT)
    blocks, _ = test_internal_clusters([seqs[i] for i in te], 0.7)
    blocks_c, _ = test_internal_clusters([seqs[i] for i in cte], 0.7)
    for m in ("RF", "LR"):
        c_o = E.correctness(E.models()[m], X, y, tr, te)
        c_c = E.correctness(E.models()[m], X, y, ctr, cte)
        d_pt, lo, hi, excl0 = delta_boot(c_o, blocks, c_c, blocks_c, mode="cluster")
        # delta_boot's replicate count is the frozen module constant cluster_bootstrap.BOOT,
        # deliberately: it is the estimator the paper's intervals were computed with, and
        # letting a CLI flag vary it would mean two datasets certified by this tool could
        # carry intervals from different estimators. `boot` is therefore recorded, not used.
        rep["boot_requested"] = int(boot)
        rep["boot_actual"] = int(_FROZEN_BOOT)
        st = E.icc_oneway(c_o, blocks)
        rep[f"{m}_acc_corrected"] = round(float(c_c.mean()), 4)
        rep[f"{m}_drop"] = round(d_pt, 4)
        rep[f"{m}_drop_ci_cluster"] = f"[{lo:.4f}, {hi:.4f}]"
        rep[f"{m}_drop_excl0"] = bool(excl0)
        rep[f"{m}_icc"] = round(st["icc"], 4)
        rep[f"{m}_design_effect"] = round(st["deff"], 4)

    # C6 -- matched random re-split negative control (same sizes, ignores homology)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(seqs))
    rtr, rte = perm[: len(ctr)], perm[len(ctr): len(ctr) + len(cte)]
    a_r = float(E.correctness(E.models()["RF"], X, y, rtr, rte).mean())
    rep["RF_acc_random_resplit"] = round(a_r, 4)
    rep["random_control_drop"] = round(rep["RF_acc_orig"] - a_r, 4)

    # C3 -- alignment engine agreement
    mm = mmseqs_clusters(seqs)
    if mm is not None:
        same = np.zeros(len(te), bool)
        trset = set(mm[tr])
        for i, t in enumerate(te):
            same[i] = mm[t] in trset
        rep["leak_mmseqs_cluster_share"] = round(float(same.mean()), 4)
        rep["engine_agreement"] = bool(
            (rep["leak_mmseqs_cluster_share"] > LEAK_CUT) ==
            (rep["leak_jaccard_0p7"] > LEAK_CUT))
    else:
        rep["leak_mmseqs_cluster_share"] = None
        rep["engine_agreement"] = None

    # ---- C9: cluster cohesion (single-linkage chaining diagnostic).
    # Whole-cluster assignment is only a legitimate correction when the clusters are
    # actually near-duplicate sets. Single linkage chains: A~B and B~C put A and C in one
    # component even when A and C are unrelated, so a "correction" can quarantine an
    # entire gene family and remove genuine signal rather than leakage. This is not
    # hypothetical -- it is the difference between this repo's two leaky datasets, and
    # without this check the tool returns a bare LEAKY on the one where correcting is
    # destructive. Two metrics, defined exactly as exp_geometry.py's G11 so the numbers
    # reconcile with results/exp_g11_cohesion.csv:
    #   cohesion      fraction of member PAIRS in the largest cluster that are real edges
    #   frac_pairs    fraction of multi-member clusters that are simple size-2 pairs
    # Calibration on the two leaky datasets: ensembl 0.959 / 0.996 (clusters are duplicate
    # pairs; correction is safe) vs nontata 0.083 / 0.240 (large chained blobs; the paper
    # finds correction there over-removes promoter/gene-family signal).
    COHESION_CUT = 0.5
    sizes = np.bincount(comp)
    multi = sizes[sizes > 1]
    biggest = int(sizes.argmax())
    members = np.where(comp == biggest)[0]
    nm = len(members)
    cohesion = 1.0
    if nm > 1:
        # Enumerating all pairs is O(nm^2); cap it with a deterministic subsample so a
        # pathological component cannot make certification unbounded. The estimate is
        # unbiased -- it is a mean over a uniform sample of the same pair population.
        MAX_M = 300
        mem = members
        if nm > MAX_M:
            mem = members[np.random.RandomState(seed).choice(nm, MAX_M, replace=False)]
        sub = [seqs[i] for i in mem]
        M_sub = H.kmer_binary_matrix(sub, 8)
        er, ec = H._edges(M_sub, 0.7)
        present = len({(min(a, b), max(a, b)) for a, b in zip(er.tolist(), ec.tolist())
                       if a != b})
        possible = len(mem) * (len(mem) - 1) // 2
        cohesion = present / possible if possible else 1.0
        rep["cohesion_estimated_on_subsample"] = bool(nm > MAX_M)
    rep["max_cluster"] = int(sizes.max())
    rep["largest_cluster_cohesion"] = round(float(cohesion), 4)
    rep["frac_clusters_that_are_pairs"] = (round(float((multi == 2).mean()), 4)
                                           if len(multi) else None)
    rep["correction_safe"] = bool(cohesion >= COHESION_CUT)

    # ---- C8: coordinate geometry, when the dataset ships intervals we have measured
    coord = None
    cpath = f"{R}/construction_rule.csv"
    if os.path.exists(cpath):
        cr = pd.read_csv(cpath).set_index("dataset")
        if name in cr.index:
            coord = float(cr.loc[name, "xsplit_ge50pct_max"])
    rep["coord_xsplit_ge50pct"] = coord

    # ---- compose the verdict from the checks, and record what each one could decide.
    # Every advertised check must be able to move the outcome or be declared inert;
    # otherwise the badge overstates what was actually verified.
    excess = rep["RF_drop"] - rep["random_control_drop"]
    rep["drop_excess_over_random_control"] = round(float(excess), 4)
    # The tripwire may escalate only when BOTH (a) the high-similarity bin clears the
    # n>=50 floor and (b) C6 says the drop is homology-driven rather than the cost of
    # repartitioning, which every dataset pays, and (c) C7 says the drop excludes zero.
    rep["tripwire_fired"] = bool(
        (rep["memorizer_tripwire_gap"] or 0) > 0.05
        and not rep["graded_gap_low_confidence"]
        and excess > 0
        and rep["RF_drop_excl0"])

    # C2 is a genuine check, not a constant: a dataset whose length-robust containment
    # exceeds its length-blind Jaccard by more than the cut is one where reporting only
    # the length-blind metric would have under-called it (the demo_coding case).
    c2 = ("DISAGREE_containment_higher"
          if rep["leak_containment_0p7"] > LEAK_CUT >= rep["leak_jaccard_0p7"]
          else "pass")
    checks = {
        "C1_full_scale": ("UNVERIFIABLE" if c1_unverifiable
                          else "pass" if rep["n"] >= full_scale_n else "NOT_FULL_SCALE"),
        "C2_two_metrics": c2,
        "C3_alignment_engine": ("not_applicable" if rep["engine_agreement"] is None
                                else "pass" if rep["engine_agreement"] else "DISAGREE"),
        "C4_C5_tripwire": ("low_confidence" if rep["graded_gap_low_confidence"]
                           else "fired" if rep["tripwire_fired"] else "quiet"),
        "C6_random_control": "pass" if excess > 0 else "drop_is_repartition_cost",
        "C7_cluster_ci": "drop_excludes_0" if rep["RF_drop_excl0"] else "drop_includes_0",
        "C8_coordinates": ("not_applicable" if coord is None
                           else "LEAKY" if coord > LEAK_CUT else "clean"),
        "C9_cluster_cohesion": ("correction_safe" if rep["correction_safe"]
                                else "CHAINED_correction_unsafe"),
    }
    rep["checks"] = checks

    # verdict_from already folds in C2 (containment) and C8 (coordinates), so those two
    # cannot escalate again here; the remaining checks each get one chance to do so.
    v = verdict_from(rep["leak_jaccard_0p7"], rep["leak_containment_0p7"], coord)
    escalations = []
    if checks["C8_coordinates"] == "LEAKY" and coord is not None and coord > LEAK_CUT:
        escalations.append("C8 coordinate geometry")
    if v == "clean" and rep["tripwire_fired"]:
        v = "borderline"
        escalations.append("C4/C5 memorizer tripwire, confirmed by C6+C7")
    if v == "clean" and checks["C3_alignment_engine"] == "DISAGREE":
        v = "borderline"
        escalations.append("C3 alignment engine disagrees")
    # C1 must be able to MOVE the verdict, not merely warn: a clean badge earned on a
    # subsample is not a clean badge, because subsampling removes near-duplicate
    # partners and can only under-report leakage.
    if v == "clean" and checks["C1_full_scale"] in ("NOT_FULL_SCALE", "UNVERIFIABLE"):
        v = "provisional-clean"
        escalations.append(
            "C1: not full scale; clean verdict is provisional"
            if checks["C1_full_scale"] == "NOT_FULL_SCALE"
            else "C1: full-scale status unverifiable from the input; pass --full-n to "
                 "assert the true dataset size. Clean verdict is provisional")
    # C9 does not change WHETHER the dataset leaks -- it changes what the user should do
    # about it. A leaky dataset whose clusters are chained cannot be repaired by
    # whole-cluster re-splitting without removing real signal, so the tool must say so
    # instead of handing back a bare LEAKY that implies "re-split and proceed".
    if v in ("LEAKY", "borderline") and not rep["correction_safe"]:
        v = "leaky-but-correction-unsafe"
        escalations.append(
            f"C9: largest-cluster cohesion {rep['largest_cluster_cohesion']:.3f} < "
            f"{COHESION_CUT}, so components are single-linkage chains rather than "
            f"near-duplicate sets. Whole-cluster re-splitting will quarantine related "
            f"but non-duplicate sequences and remove genuine signal. Report the "
            f"novel-stratum accuracy (RF_novel_acc / LR_novel_acc above) on the "
            f"as-shipped split instead of re-splitting.")
    rep["verdict"] = v
    rep["escalated_by"] = escalations or None
    # Carried out for --emit-splits and popped before the report is written, so
    # "certify then retrain" does not require a second tool run whose parameters
    # are chosen independently of the ones the certification actually used.
    rep["_corrected_split"] = (ctr, cte)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _load_splits(path):
    """Read a split JSON, accepting either key spelling.

    homology_split.py writes {'train_idx', 'test_idx'}; earlier versions of this tool
    read only {'train', 'test'}, so certifying the repo's own shipped demo_splits.json --
    the obvious next step after producing a split -- died on a bare KeyError. The two
    tools are advertised together, so they have to compose.
    """
    sp = json.load(open(path))
    tr = sp.get("train_idx", sp.get("train"))
    te = sp.get("test_idx", sp.get("test"))
    if tr is None or te is None:
        raise SystemExit(
            f"[certify] {path}: expected train/test indices under either "
            f"'train_idx'/'test_idx' (what homology_split.py writes) or 'train'/'test'. "
            f"Found keys: {sorted(sp)}")
    return np.asarray(tr), np.asarray(te)


def _read_fasta(path):
    ids, seqs, cur = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
                ids.append(line[1:].split()[0])
            elif line:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return ids, seqs


def main():
    ap = argparse.ArgumentParser(
        prog="certify",
        description="Certify a sequence benchmark against near-duplicate leakage.")
    ap.add_argument("--self-validate", action="store_true",
                    help="reproduce the paper's 8-dataset verdicts from committed CSVs")
    ap.add_argument("--dataset", help="a Genomic Benchmarks dataset name to certify")
    ap.add_argument("--fasta"), ap.add_argument("--labels")
    ap.add_argument("--splits", help="JSON of split indices; accepts either "
                    "{'train_idx','test_idx'} (what homology_split.py writes) or "
                    "{'train','test'}")
    ap.add_argument("--emit-splits", dest="emit_splits", metavar="PATH",
                    help="also write the corrected near-duplicate-aware split this tool "
                         "computes internally, so 'certify then retrain' does not require "
                         "a second tool run with independently chosen parameters. Written "
                         "with both key spellings.")
    ap.add_argument("--out", default=os.path.join(R, "certify_report.json"))
    ap.add_argument("--boot", type=int, default=2000,
                    help="recorded in the report for provenance; the cluster-bootstrap\n"
                         "replicate count is fixed by the frozen estimator so that every\n"
                         "certified dataset carries an interval from the same estimator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int,
                    help="certify a subsample of --dataset; C1 then reports the run as "
                         "not full scale and any clean verdict becomes provisional")
    ap.add_argument("--full-n", type=int, dest="full_n",
                    help="size of the UNSUBSAMPLED dataset, for the --fasta path. "
                         "Without it C1 is UNVERIFIABLE and a clean verdict is "
                         "downgraded to provisional-clean, since the tool cannot tell a "
                         "full dataset from a subsample by looking at it")
    a = ap.parse_args()

    if a.self_validate:
        ok, _ = self_validate()
        sys.exit(0 if ok else 1)

    if a.dataset:
        from audit.core import expkit as E
        if a.cap:
            # Certifying a subsample: record the TRUE size so C1 can fail. Without this
            # the check compares len(seqs) to itself and can never fire.
            full_n = len(E.load(a.dataset, full=True)[0])
            seqs, y, tr, te, _ = E.load(a.dataset, full=False, cap=a.cap)
        else:
            seqs, y, tr, te, _ = E.load(a.dataset, full=True)
            full_n = len(seqs)
        rep = certify_dataset(seqs, y, tr, te, name=a.dataset,
                              boot=a.boot, seed=a.seed, full_scale_n=full_n)
    elif a.fasta and a.labels:
        _ids, seqs = _read_fasta(a.fasta)
        y = np.array([int(x) for x in open(a.labels).read().split()])
        if a.splits:
            tr, te = _load_splits(a.splits)
        else:
            rng = np.random.RandomState(a.seed)
            perm = rng.permutation(len(seqs))
            cut = int(0.8 * len(seqs))
            tr, te = perm[:cut], perm[cut:]
            print("[certify] no --splits given: using a random 80/20 split, which is "
                  "exactly the construction this tool exists to test")
        rep = certify_dataset(seqs, y, tr, te, name=os.path.basename(a.fasta),
                              boot=a.boot, seed=a.seed, full_scale_n=a.full_n)
    else:
        ap.error("give --self-validate, or --dataset, or --fasta with --labels")

    corrected = rep.pop("_corrected_split", None)
    if a.emit_splits:
        if corrected is None:
            print("[certify] --emit-splits: no corrected split was computed")
        else:
            ctr, cte = corrected
            payload = {
                "source": "audit.tools.certify",
                "dataset": rep["dataset"],
                "threshold": 0.7, "k": 8, "seed": a.seed,
                "residual_leak_after_split": rep["residual_leak_after_split"],
                "correction_safe": rep["correction_safe"],
                "largest_cluster_cohesion": rep["largest_cluster_cohesion"],
                # both spellings, so this file loads into either tool without a shim
                "train_idx": [int(i) for i in ctr], "test_idx": [int(i) for i in cte],
                "train": [int(i) for i in ctr], "test": [int(i) for i in cte],
            }
            with open(a.emit_splits, "w") as fh:
                json.dump(payload, fh)
            note = ("" if rep["correction_safe"] else
                    "  WARNING: C9 says this dataset's clusters are chained "
                    "(cohesion %.3f); this corrected split removes genuine signal. "
                    "Prefer the novel-stratum readout on the as-shipped split."
                    % rep["largest_cluster_cohesion"])
            print(f"[certify] corrected split -> {a.emit_splits} "
                  f"({len(ctr)} train / {len(cte)} test){note}")

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nVERDICT: {rep['verdict']}  ->  {a.out}")


if __name__ == "__main__":
    main()
