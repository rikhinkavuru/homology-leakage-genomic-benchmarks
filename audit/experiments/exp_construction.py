#!/usr/bin/env python3
"""
Deepener #3 -- the construction rule: which benchmarks leak, predicted from
provenance + coordinate geometry (CPU-only, no model fitting in this module).

THESIS
------
Near-duplicate leakage is NOT caused by "multi-cell-type data". It is caused by
whether the positive class is an UN-MERGED UNION of experimentally-called regions
that tiles recurring loci (leaky), versus a MERGED non-overlapping segmentation, a
deduplicated curated set, or a random genome draw (clean). The decisive evidence is
INTERNAL TO ENSEMBL: human_enhancers_ensembl is the raw FANTOM5 enhancer union over
808 CAGE libraries (LEAKY), while human_ocr_ensembl and human_ensembl_regulatory come
from the Ensembl Regulatory Build -- an explicitly merged, cell-type-collapsed
"MultiCell" consensus (CLEAN). Same provider, same organism, opposite verdicts.

WHY COORDINATES ARE THE RIGHT INDEPENDENT VARIABLE
--------------------------------------------------
The coordinate signatures below are computed from the shipped interval CSVs, which are
causally UPSTREAM of and statistically INDEPENDENT from the sequence-space k-mer
clustering used to measure leakage. So "coordinates predict the sequence-level leak
fraction" is a real cross-modal prediction, not a restatement.

LAYER 2 SIGNATURES (per dataset, per class)
    R_self          1 - n_merged/n_raw  -- share of intervals absorbed by merging
                    overlapping intervals within a class. 0 for a disjoint set.
    dup_frac_X      share of intervals having >=X reciprocal overlap with another
                    interval of the same class (X in 0.5, 0.9, 1.0)
    xsplit_>=1bp    share of TEST intervals overlapping ANY TRAIN interval at all
    xsplit_>=50/90  same, at >=50% / >=90% reciprocal overlap

Coordinates are recovered offline from datacache/intervals/*.csv.gz and aligned to
expkit.load() row order. Overlap is strand-AGNOSTIC throughout (strand is NaN for
human_ocr_ensembl and human_ensembl_regulatory, so a strand-aware rule would not be
uniformly applicable).

Run:  python -m audit.experiments.exp_construction
Out:  results/construction_signatures.csv, construction_provenance.csv,
      construction_rule.csv
"""
from __future__ import annotations
import os, glob, sys, time, urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

GBROOT = os.path.join(os.path.expanduser("~"), ".genomic_benchmarks")
IV_BASE = "https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/datasets"
IV_CACHE = os.path.join(HERE, "datacache", "intervals")
R = os.path.join(HERE, "results")
BIN = 200_000            # > max interval length in every GB dataset (max 3237 bp)

DATASETS = ["human_enhancers_ensembl", "human_nontata_promoters",
            "human_ocr_ensembl", "human_ensembl_regulatory",
            "human_enhancers_cohn", "demo_coding_vs_intergenomic_seqs",
            "demo_human_or_worm", "drosophila_enhancers_stark"]

# ---------------------------------------------------------------------------
# Layer 1 -- provenance taxonomy, read from the GB construction notebooks and the
# primary sources. `merge_step` records whether ANY merge / redundancy-removal /
# identity-filter was applied to the positive class before splitting.
# ---------------------------------------------------------------------------
PROVENANCE = {
    # NOTE ON MECHANISM. The GB documentation attributes this set to the FANTOM5
    # enhancer atlas (Andersson 2014, 808 CAGE libraries). We deliberately do NOT claim
    # the observed redundancy IS an 808-library union, because the measured signature is
    # not what such a union would produce: multiplicity is capped at exactly 2 with
    # byte-identical sequences and identical interval boundaries, whereas a union over
    # many independently-called tracks would give a broad multiplicity spectrum and
    # jagged, only-partially-overlapping boundaries. What the data supports is the
    # weaker, sufficient statement recorded here: the positive class ships each of
    # 40,934 enhancers exactly twice with no deduplication step.
    "human_enhancers_ensembl": dict(
        positive_source="FANTOM5-derived enhancer set (per GB docs); observed as exact "
                        "2x row duplication of 40,934 unique intervals",
        category="un-deduplicated assembly (exact multiplicity-2 duplication)",
        merge_step="none", predicted="LEAKY"),
    "human_nontata_promoters": dict(
        positive_source="EPDnew fixed 251 bp TSS windows (Dreos 2013/2015; Umarov 2017)",
        category="fixed window over recurring loci, redundancy filter omitted",
        merge_step="none (community-standard CD-HIT-EST 0.8 step not applied)",
        predicted="LEAKY"),
    "human_ocr_ensembl": dict(
        positive_source="Ensembl Regulatory Build open chromatin regions (Zerbino 2015)",
        category="merged segmentation consensus", merge_step="MultiCell merge",
        predicted="CLEAN"),
    "human_ensembl_regulatory": dict(
        positive_source="Ensembl Regulatory Build 3-class segmentation (Zerbino 2015)",
        category="merged segmentation consensus", merge_step="MultiCell merge",
        predicted="CLEAN"),
    "human_enhancers_cohn": dict(
        positive_source="Cohn et al. 2018 curated enhancer set",
        category="deduplicated curated set", merge_step="curation/dedup",
        predicted="CLEAN"),
    "drosophila_enhancers_stark": dict(
        positive_source="Stark lab Drosophila enhancers",
        category="deduplicated curated set", merge_step="curation/dedup",
        predicted="CLEAN"),
    "demo_coding_vs_intergenomic_seqs": dict(
        positive_source="random genome draw (coding vs intergenomic)",
        category="random genome draw", merge_step="n/a (disjoint draw)",
        predicted="CLEAN"),
    "demo_human_or_worm": dict(
        positive_source="random genome draw (human vs C. elegans)",
        category="random genome draw", merge_step="n/a (disjoint draw)",
        predicted="CLEAN"),
}


# ---------------------------------------------------------------------------
# coordinate recovery (offline; aligned to expkit.load row order)
# ---------------------------------------------------------------------------
def fetch_intervals(dset):
    """Per-(split,class) interval CSVs, cached under datacache/intervals/<dset>/."""
    dd = os.path.join(IV_CACHE, dset)
    os.makedirs(dd, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(dd, "*.csv.gz"))}
    classes = _classes(dset)
    out = {}
    for split in ("train", "test"):
        for cls in classes:
            fn = f"{split}_{cls}.csv.gz"
            path = os.path.join(dd, fn)
            if fn not in have and not os.path.exists(path):
                urllib.request.urlretrieve(f"{IV_BASE}/{dset}/{split}/{cls}.csv.gz", path)
            df = pd.read_csv(path, compression="gzip")
            df["id"] = df["id"].astype(str)
            out[(split, cls)] = df
    return out


def _classes(dset):
    """Class directory names, from the local extraction if present else the known
    GB convention (positive/negative for binary; the 3-class set names its own)."""
    base = os.path.join(GBROOT, dset)
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        if os.path.isdir(sd):
            cs = sorted(c for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c)))
            if cs:
                return cs
    if dset == "human_ensembl_regulatory":
        return ["enhancer", "ocr", "promoter"]
    return ["negative", "positive"]


def recover_coords(dset):
    """Interval table aligned 1:1 with expkit.load(dset, full=True) row order.

    expkit -> run_suite.discover_and_subsample builds rows by
        for split in (train, test): for cls in sorted(listdir): for p in sorted(glob('*.txt'))
    and the .txt stem IS the interval-CSV `id`. So sorting each (split,cls) block's ids
    by `id + '.txt'` reproduces the glob order EXACTLY, with no need for the extracted
    .txt folders (two datasets ship corrupt zips). Where the folders DO exist we verify
    the recovered order against the real filenames and against the sequence lengths."""
    csv = fetch_intervals(dset)
    classes = _classes(dset)
    label_of = {c: i for i, c in enumerate(classes)}
    frames, verify = [], dict(dataset=dset, order_check="derived", glob_match=None,
                              len_match=None, n_rows=None)
    for split in ("train", "test"):
        for cls in classes:
            df = csv[(split, cls)].copy()
            df = df.iloc[np.argsort([i + ".txt" for i in df["id"].tolist()], kind="stable")]
            cd = os.path.join(GBROOT, dset, split, cls)
            if os.path.isdir(cd):
                paths = sorted(glob.glob(os.path.join(cd, "*.txt")))
                stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
                ok = stems == df["id"].tolist()
                verify["glob_match"] = ok if verify["glob_match"] is None else (
                    verify["glob_match"] and ok)
            df["split"], df["cls"], df["label"] = split, cls, label_of[cls]
            frames.append(df)
    iv = pd.concat(frames, ignore_index=True)
    iv["start"] = iv["start"].astype(np.int64)
    iv["end"] = iv["end"].astype(np.int64)
    iv["region"] = iv["region"].astype(str)
    verify["n_rows"] = len(iv)

    # independent alignment check: recovered interval length vs actual sequence length
    try:
        seqs, y, otr, ote, _ = E.load(dset, full=True)
        if len(seqs) == len(iv):
            L = np.array([len(s) for s in seqs])
            W = (iv["end"] - iv["start"]).to_numpy()
            verify["len_match"] = float((L == W).mean())
            verify["label_match"] = float((np.asarray(y) == iv["label"].to_numpy()).mean())
    except Exception as ex:                                    # noqa: BLE001
        verify["len_match"] = f"unavailable ({type(ex).__name__})"
    return iv, verify


# ---------------------------------------------------------------------------
# interval geometry (binned sweep; exact because BIN >> max interval length)
# ---------------------------------------------------------------------------
def _index(df):
    """(region, bin) -> row positions, each interval registered in every bin it spans."""
    idx = defaultdict(list)
    reg = df["region"].to_numpy()
    s = df["start"].to_numpy(); e = df["end"].to_numpy()
    for i in range(len(df)):
        for b in range(s[i] // BIN, e[i] // BIN + 1):
            idx[(reg[i], b)].append(i)
    return idx, s, e, reg


def merged_count(df):
    """Number of connected intervals after merging overlaps, per class-level table."""
    n = 0
    for _, g in df.groupby("region", sort=False):
        s = np.sort(g["start"].to_numpy())
        order = np.argsort(g["start"].to_numpy(), kind="stable")
        e = g["end"].to_numpy()[order]
        cur_end = -1
        for i in range(len(s)):
            if s[i] >= cur_end:          # half-open [start, end)
                n += 1
                cur_end = e[i]
            else:
                cur_end = max(cur_end, e[i])
    return n


def self_overlap(df, thresholds=(0.5, 0.9, 1.0)):
    """Share of intervals having >= t reciprocal overlap with ANOTHER interval."""
    idx, s, e, reg = _index(df)
    n = len(df)
    best = np.zeros(n)
    for i in range(n):
        cands = set()
        for b in range(s[i] // BIN, e[i] // BIN + 1):
            cands.update(idx[(reg[i], b)])
        cands.discard(i)
        if not cands:
            continue
        c = np.fromiter(cands, int, len(cands))
        ov = np.minimum(e[i], e[c]) - np.maximum(s[i], s[c])
        ov = np.maximum(ov, 0)
        rec = ov / np.maximum(np.maximum(e[i] - s[i], e[c] - s[c]), 1)
        best[i] = rec.max()
    return {f"dup_frac_{t}": float((best >= t).mean()) for t in thresholds}, best


def exact_dup_stats(df):
    """Exact-coordinate duplication census: how often does the SAME interval ship
    more than once, and how often does a TEST interval's exact coordinate also
    appear in TRAIN? This is the un-merged-union fingerprint in its purest form."""
    key = df["region"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    vc = key.value_counts()
    tr = set(key[df.split.to_numpy() == "train"])
    te = key[df.split.to_numpy() == "test"]
    return dict(n_unique_coords=int(len(vc)),
                frac_rows_on_dup_coord=float(vc[vc > 1].sum() / len(df)) if len(df) else 0.0,
                max_coord_multiplicity=int(vc.max()) if len(vc) else 0,
                xsplit_exact=float(te.isin(tr).mean()) if len(te) else 0.0)


def cross_split_overlap(test_df, train_df, thresholds=(0.0, 0.5, 0.9)):
    """Share of TEST intervals overlapping ANY TRAIN interval at >= t reciprocal
    overlap (t=0.0 means >= 1 bp)."""
    idx, s_tr, e_tr, reg_tr = _index(train_df)
    s = test_df["start"].to_numpy(); e = test_df["end"].to_numpy()
    reg = test_df["region"].to_numpy()
    best = np.zeros(len(test_df))
    for i in range(len(test_df)):
        cands = set()
        for b in range(s[i] // BIN, e[i] // BIN + 1):
            cands.update(idx[(reg[i], b)])
        if not cands:
            continue
        c = np.fromiter(cands, int, len(cands))
        ov = np.minimum(e[i], e_tr[c]) - np.maximum(s[i], s_tr[c])
        ov = np.maximum(ov, 0)
        rec = ov / np.maximum(np.maximum(e[i] - s[i], e_tr[c] - s_tr[c]), 1)
        best[i] = rec.max()
    out = {}
    for t in thresholds:
        key = "xsplit_ge1bp" if t == 0.0 else f"xsplit_ge{int(t*100)}pct"
        out[key] = float((best > 0).mean()) if t == 0.0 else float((best >= t).mean())
    return out, best


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(R, exist_ok=True)
    sig_rows, ver_rows = [], []
    for d in DATASETS:
        t1 = time.time()
        iv, ver = recover_coords(d)
        ver_rows.append(ver)
        pos_cls = _classes(d)[-1]        # 'positive' for binary; last name for 3-class
        for cls in _classes(d) + ["ALL"]:
            sub = iv if cls == "ALL" else iv[iv.cls == cls]
            if not len(sub):
                continue
            raw, mer = len(sub), merged_count(sub)
            dup, _ = self_overlap(sub)
            te, tr = sub[sub.split == "test"], sub[sub.split == "train"]
            xs, _ = cross_split_overlap(te, tr) if len(te) and len(tr) else ({}, None)
            # A class whose "region" is a per-row accession (e.g. demo_coding's
            # coding_seqs ships ENST transcript ids, one distinct region per sequence)
            # lives in a per-row coordinate space, so no two intervals can ever overlap
            # and every coordinate statistic is structurally 0. That is NOT evidence of
            # cleanliness and must not be reported as such.
            measurable = bool(sub.region.nunique() < len(sub))
            sig_rows.append(dict(
                dataset=d, cls=cls, is_positive=bool(cls == pos_cls),
                coord_measurable=measurable,
                n_raw=raw, n_test=int(len(te)), n_merged=mer,
                R_self=round(1 - mer / raw, 4),
                median_len=int(np.median((sub.end - sub.start))),
                n_regions=int(sub.region.nunique()),
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in exact_dup_stats(sub).items()},
                **{k: round(v, 4) for k, v in dup.items()},
                **{k: round(v, 4) for k, v in xs.items()}))
        print(f"[{d}] n={len(iv)} recovered ({time.time()-t1:.0f}s) "
              f"len_match={ver.get('len_match')} glob_match={ver.get('glob_match')}",
              flush=True)

    sig = pd.DataFrame(sig_rows)
    sig.to_csv(f"{R}/construction_signatures.csv", index=False)
    pd.DataFrame(ver_rows).to_csv(f"{R}/construction_alignment_check.csv", index=False)

    prov = pd.DataFrame([dict(dataset=k, **v) for k, v in PROVENANCE.items()])
    # observed verdict from the frozen report card (+ the full-scale containment
    # refinement that makes demo_coding borderline)
    rc = pd.read_csv(f"{R}/leakage_report_card.csv")
    rc["dataset"] = rc["dataset"].str.replace(" (3-class)", "", regex=False)
    fsc = pd.read_csv(f"{R}/full_scale_containment.csv").set_index("dataset")
    gb1 = pd.read_csv(f"{R}/exp_gb1_containment.csv").set_index("dataset")

    def observed(d):
        row = rc[rc.dataset == d]
        jac = float(row.leak_at_0p7.iloc[0])
        con = (float(fsc.loc[d, "leak_con_0p7"]) if d in fsc.index
               else float(gb1.loc[d, "leak_contain_0p7"]) if d in gb1.index else np.nan)
        if jac > 0.1:
            return "LEAKY", jac, con
        if not np.isnan(con) and con > 0.1:
            return "borderline", jac, con
        return "CLEAN", jac, con

    obs = [observed(d) for d in prov.dataset]
    prov["observed_verdict"] = [o[0] for o in obs]
    prov["leak_jaccard_0p7"] = [round(o[1], 4) for o in obs]
    prov["leak_containment_0p7_fullscale"] = [round(o[2], 4) for o in obs]

    # ---- the decision rule -------------------------------------------------
    # Two corrections to the pre-registered thesis, both forced by the data:
    #  (i) redundancy is NOT always in the positive class -- nonTATA's is entirely in
    #      the NEGATIVE class -- so the statistic must be a MAX OVER CLASSES;
    # (ii) ">=1 bp overlap" over-calls on long intervals (drosophila_stark tiles at
    #      2142 bp median with only 1 bp-scale touches), so the statistic must be
    #      RECIPROCAL overlap >= 50%, not mere adjacency.
    # Threshold is NOT tuned: we reuse the repo's existing sequence-leak cut of 0.1
    # (report_card.py:49), so the rule has zero free parameters.
    THRESH = 0.1
    per = sig[sig.cls != "ALL"]
    rule_rows = []
    for d in DATASETS:
        s = per[per.dataset == d]
        w = s.n_test / s.n_test.sum()
        rule_rows.append(dict(
            dataset=d,
            category=PROVENANCE[d]["category"], merge_step=PROVENANCE[d]["merge_step"],
            redundant_class=str(s.loc[s.xsplit_ge50pct.idxmax(), "cls"]),
            R_self_max=round(float(s.R_self.max()), 4),
            xsplit_ge1bp_max=round(float(s.xsplit_ge1bp.max()), 4),
            xsplit_ge50pct_max=round(float(s.xsplit_ge50pct.max()), 4),
            xsplit_exact_max=round(float(s.xsplit_exact.max()), 4),
            frac_rows_on_dup_coord_max=round(float(s.frac_rows_on_dup_coord.max()), 4),
            # cross-modal prediction: test-weighted coordinate overlap vs the
            # independently measured SEQUENCE leak fraction at the same stringency
            coord_pred_leak_ge90=round(float((w * s.xsplit_ge90pct).sum()), 4),
            coord_pred_leak_exact=round(float((w * s.xsplit_exact).sum()), 4)))
    rule = pd.DataFrame(rule_rows)
    lk = pd.read_csv(f"{R}/leakage_report_card.csv")
    lk["dataset"] = lk["dataset"].str.replace(" (3-class)", "", regex=False)
    lk = lk.set_index("dataset")
    rule["seq_leak_0p9_measured"] = [float(lk.loc[d, "leak_at_0p9"]) for d in rule.dataset]
    rule["seq_leak_0p7_measured"] = [float(lk.loc[d, "leak_at_0p7"]) for d in rule.dataset]
    rule["coord_verdict"] = np.where(rule.xsplit_ge50pct_max > THRESH, "LEAKY", "CLEAN")
    rule["seq_verdict"] = [prov.set_index("dataset").loc[d, "observed_verdict"]
                           for d in rule.dataset]
    rule["provenance_verdict"] = [PROVENANCE[d]["predicted"] for d in rule.dataset]
    # `borderline` is a real third level, and how it is folded decides the headline
    # score. Report BOTH foldings rather than silently picking the flattering one:
    # LENIENT (borderline->CLEAN) gives 8/8; STRICT (borderline->LEAKY) gives 7/8,
    # the miss being demo_coding, whose leakage is coding-paralog homology with no
    # coordinate signature at all (and whose coding_seqs class is not even measurable
    # in coordinate space -- see coord_measurable).
    lenient = rule.seq_verdict.replace({"borderline": "CLEAN"})
    strict = rule.seq_verdict.replace({"borderline": "LEAKY"})
    rule["coord_agrees_seq_lenient"] = rule.coord_verdict == lenient
    rule["coord_agrees_seq_strict"] = rule.coord_verdict == strict
    rule["coord_agrees_seq"] = rule["coord_agrees_seq_lenient"]
    rule["provenance_agrees_seq"] = rule.provenance_verdict == lenient
    unmeasurable = per[~per.coord_measurable].groupby("dataset").size()
    rule["classes_not_measurable_in_coord_space"] = [
        int(unmeasurable.get(d, 0)) for d in rule.dataset]
    # leave-one-dataset-out: the rule is a fixed threshold with no fitted parameter,
    # so holding out any dataset leaves the other seven's verdicts unchanged; the
    # meaningful LODO statement is the separation margin.
    lo = float(rule[rule.coord_verdict == "LEAKY"].xsplit_ge50pct_max.min())
    hi = float(rule[rule.coord_verdict == "CLEAN"].xsplit_ge50pct_max.max())
    rule.to_csv(f"{R}/construction_rule.csv", index=False)
    prov["R_self_max"] = rule.set_index("dataset").loc[prov.dataset, "R_self_max"].values
    prov["rule_hit"] = rule.set_index("dataset").loc[
        prov.dataset, "provenance_agrees_seq"].values
    prov.to_csv(f"{R}/construction_provenance.csv", index=False)

    print("\n== construction rule (coordinate geometry -> leakage) ==")
    print(rule[["dataset", "category", "redundant_class", "R_self_max",
                "xsplit_ge1bp_max", "xsplit_ge50pct_max", "xsplit_exact_max",
                "coord_verdict", "seq_verdict"]].to_string(index=False))
    print(f"\ncoordinate rule (xsplit_ge50pct_max > {THRESH}) vs the sequence verdict:")
    print(f"  {int(rule.coord_agrees_seq_strict.sum())}/{len(rule)} STRICT "
          f"(borderline counted as LEAKY)  -- miss: "
          f"{list(rule[~rule.coord_agrees_seq_strict].dataset)}")
    print(f"  {int(rule.coord_agrees_seq_lenient.sum())}/{len(rule)} LENIENT "
          f"(borderline counted as CLEAN)")
    nm = rule[rule.classes_not_measurable_in_coord_space > 0]
    if len(nm):
        print(f"  classes with per-row coordinate spaces (rule has NO coverage): "
              f"{list(nm.dataset)}")
    print(f"provenance rule agrees on {int(rule.provenance_agrees_seq.sum())}/{len(rule)}")
    print(f"separation margin: min LEAKY = {lo:.4f}  vs  max CLEAN = {hi:.4f}  "
          f"({lo/max(hi,1e-9):.1f}x gap, threshold {THRESH} sits inside it)")
    print("\n== cross-modal check: coordinates predict the SEQUENCE leak fraction ==")
    print(rule[["dataset", "coord_pred_leak_exact", "coord_pred_leak_ge90",
                "seq_leak_0p9_measured"]].to_string(index=False))
    print(f"\nEXP_CONSTRUCTION_DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
