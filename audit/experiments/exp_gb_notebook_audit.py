#!/usr/bin/env python3
"""
A4: ground the construction claim in Genomic Benchmarks' OWN published methods.

The manuscript currently argues backwards -- it infers how the benchmark datasets were
built from the geometry of the shipped coordinates. A reviewer can fairly call that
retrospective inference. This experiment replaces the inference with the curators'
primary source: the construction notebooks the Genomic Benchmarks authors published at
ML-Bioinfo-CEITEC/genomic_benchmarks docs/<dataset>/create_datasets.ipynb.

WHAT IS FETCHED
  1. The four create_datasets.ipynb notebooks, pinned to the last commit that touched
     each file (so the emitted URL is a stable citation, not a moving `main` ref).
  2. The code those notebooks DELEGATE to. This is the crux. Two of the four notebooks
     contain no construction logic at all -- they shell out to an external scraper. A
     search that stopped at the notebook would report "no deduplication step found" and
     the manuscript would be overclaiming on the strength of a file that simply does not
     contain the relevant code. So for each dataset we also fetch and search whatever it
     calls:
       - genomic_benchmarks/src/genomic_benchmarks/seq2loc/seq2loc.py  (cohn, nontata)
       - katarinagresova/ensembl_scraper scraper/*.py                  (ensembl x2)

THE DEDUPLICATION SEARCH IS DELIBERATELY GENEROUS
  The manuscript's claim is that an overlap-removal step is ABSENT. That is a negative
  claim, so a false negative here is the failure mode that hurts us: it would let the
  paper assert absence on the basis of a search that was too narrow. The search is
  therefore run with a wide token list (DEDUP_PATTERNS) over BOTH notebook cells and
  delegated sources, EVERY hit is written verbatim to the CSV, and the exact pattern
  list is written to the CSV too so a reviewer can see what was and was not looked for.
  Hits are reported raw. Where a hit is a token match that does not actually remove
  duplicate or overlapping sequences, that is recorded as an adjudication string beside
  the verbatim line rather than silently dropped.

Run: PYTHONPATH=. ./venv/bin/python audit/experiments/exp_gb_notebook_audit.py
  -> results/gb_notebook_audit.csv
"""
import os
import re
import json
import datetime
import urllib.request
import urllib.error

import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")
os.makedirs(RESULTS, exist_ok=True)

GB_REPO = "ML-Bioinfo-CEITEC/genomic_benchmarks"
SCRAPER_REPO = "katarinagresova/ensembl_scraper"
RAW = "https://raw.githubusercontent.com"
API = "https://api.github.com"

DATASETS = ["human_enhancers_ensembl", "human_ocr_ensembl",
            "human_enhancers_cohn", "human_nontata_promoters"]

# The ensembl_scraper commit that docs/human_ocr_ensembl/create_datasets.ipynb pins in
# its own `pip install git+...@<sha>` line. Extracted from the notebook at run time;
# this is only the fallback if the extraction fails.
SCRAPER_SHA_FALLBACK = "6d3bba8e6be7f5ead58a3bbaed6a4e8cd35e62fd"

# ---------------------------------------------------------------------------
# the generous dedup/overlap search
# ---------------------------------------------------------------------------
# Every token the task specified, plus the bedtools/pybedtools/pyranges vocabulary a
# genomics pipeline would plausibly use to collapse overlapping intervals, plus the
# pandas idioms for the same. Case-insensitive.
DEDUP_PATTERNS = [
    r"drop_duplicates",
    r"\bdedup\w*",
    r"\.duplicated\b",
    r"\bduplicate\w*",
    r"bedtools\s+merge",
    r"\bpybedtools\b",
    r"\bpyranges\b",
    r"\bBedTool\b",
    r"\.merge\b",
    r"\bmerge\b",
    r"\boverlap\w*",
    r"\bintersect\w*",
    r"\bsubtract\b",
    r"\bunique\b",
    r"\.nunique\b",
    r"\bset\(",
    r"\bnp\.unique\b",
    r"\bcollaps\w*",
    r"\bredundan\w*",
    r"\bcd-?hit\b",
    r"\bblast\w*",
    r"\bmmseqs\b",
    r"\bcluster\w*",
]
DEDUP_RE = re.compile("|".join(DEDUP_PATTERNS), re.IGNORECASE)

SEQ2LOC_RE = re.compile(r"seq2loc|fasta2loc", re.IGNORECASE)

# ---------------------------------------------------------------------------
# adjudication of the generous search's hits
# ---------------------------------------------------------------------------
# The generous search is tuned to over-fire, so a raw hit count is not the claim the
# manuscript makes. The claim is narrower: is there a step that REMOVES duplicate or
# overlapping sequences from the shipped dataset? Each raw hit is matched against these
# rules and labelled. Anything unmatched is labelled UNADJUDICATED rather than assumed
# benign -- the search must fail loud, not fail clean.
#   (regex on the hit line, removes_dup_or_overlap?, verdict)
ADJUDICATION = [
    (r"feature_types = seqs\[.*\]\.unique\(\)", False,
     "FALSE POSITIVE: .unique() is applied to feature-TYPE NAMES (e.g. 'enhancer', "
     "'promoter') to loop over them, not to sequences or loci."),
    (r"def is_intersecting|intersecting = \(df_forbidden|return intersecting\.any\(\)", False,
     "NOT a deduplication step: a rejection-sampling guard used WHILE DRAWING negatives. "
     "It tests one scalar start position against the positive intervals. It never runs "
     "over the assembled dataset, never compares negatives to each other, and removes "
     "nothing after the fact."),
    (r"cannot\s+intersect with original sequence|intersect with original sequence", False,
     "Docstring prose describing the negative sampler, not executable dedup code."),
    (r"while is_intersecting", False,
     "Rejection-sampling loop condition; redraws a start position. Not a removal step."),
]


def adjudicate(hits):
    """Label every raw hit. Returns (per_hit_labels, any_genuine_removal_step)."""
    labels, genuine = [], False
    for h in hits:
        line = h.split(": ", 1)[-1]
        for rx, removes, verdict in ADJUDICATION:
            if re.search(rx, line):
                labels.append(f"[{h.split(':L')[0]}] {verdict}")
                genuine = genuine or removes
                break
        else:
            labels.append(f"[{h}] UNADJUDICATED -- review manually")
            genuine = True  # fail loud: an unrecognised hit counts against the claim
    return labels, genuine

ACCESS_DATE = datetime.date.today().isoformat()


def fetch(url):
    """Return (text, http_status). Never raises on HTTP error -- an unreachable
    artifact must be reported as such, not silently replaced by prose."""
    req = urllib.request.Request(url, headers={"User-Agent": "homology-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        return "", e.code
    except Exception as e:  # network down, DNS, timeout
        return "", f"ERROR:{type(e).__name__}"


def last_commit_for_path(repo, path):
    """SHA of the most recent commit touching `path`. Pinning to this rather than to
    `main` is what makes the emitted URL a stable citation."""
    txt, st = fetch(f"{API}/repos/{repo}/commits?path={path}&per_page=1")
    if st != 200 or not txt:
        return "", st
    try:
        js = json.loads(txt)
        return (js[0]["sha"], js[0]["commit"]["committer"]["date"]) if js else ("", "")
    except Exception:
        return "", ""


def cell_lines(nb):
    """Yield (cell_index, cell_type, line) for every source line in the notebook."""
    for i, c in enumerate(nb.get("cells", [])):
        for line in "".join(c.get("source", [])).splitlines():
            if line.strip():
                yield i, c.get("cell_type", "?"), line.rstrip()


def outputs_text(nb):
    """Flatten every rendered output in the notebook. The shipped notebooks retain
    their execution outputs, which carry hard counts (`N sequences read and parsed.`)
    and, for nontata, the actual negative coordinates -- primary evidence that does not
    depend on our re-running anything."""
    out = []
    for c in nb.get("cells", []):
        for o in c.get("outputs", []) or []:
            if "text" in o:
                out.append("".join(o["text"]))
            data = o.get("data") or {}
            if "text/plain" in data:
                out.append("".join(data["text/plain"]))
    return "\n".join(out)


def scan_dedup(label, text):
    """Run the generous search over `text`. Returns a list of 'label:line N: <line>'
    strings, one per matching line, verbatim."""
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if DEDUP_RE.search(line):
            hits.append(f"{label}:L{n}: {line.strip()}")
    return hits


def grab(pattern, text, flags=re.DOTALL):
    m = re.search(pattern, text, flags)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else ""


def tiling_evidence(out_text):
    """Pull (start, end) coordinate pairs out of the notebook's own rendered output and
    report the window length and the step between consecutive windows. For nontata this
    is the manuscript's tiling geometry appearing in the curators' shipped output, which
    is a much stronger citation than our re-derivation of it."""
    pairs = []
    for a, b in re.findall(r"(\d{6,})\s+(\d{6,})", out_text):
        a, b = int(a), int(b)
        if 0 < b - a < 100000:
            pairs.append((a, b))
    if len(pairs) < 3:
        return "", ""
    lens = sorted({b - a for a, b in pairs})
    starts = sorted(a for a, _ in pairs)
    steps = [j - i for i, j in zip(starts, starts[1:]) if j != i]
    # A rendered head() mixes the positive and the negative tables, and the positives are
    # scattered across the genome. What matters is whether ANY consecutive windows are
    # closer together than one window length -- i.e. whether they overlap each other.
    wl = max(lens)
    overlapping = sorted(s for s in steps if s < wl)
    med = overlapping[len(overlapping) // 2] if overlapping else None
    ev = "; ".join(f"{a}-{b}" for a, b in sorted(pairs)[:8])
    geom = (f"window_len={lens if len(lens) < 4 else lens[:4]}; "
            f"n_consecutive_pairs={len(steps)}; "
            f"n_pairs_overlapping(step<{wl})={len(overlapping)}; "
            f"overlapping_steps={overlapping[:8]}; median_overlapping_step={med}")
    return ev, geom


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"access date {ACCESS_DATE}")

    # --- delegated sources, fetched once and shared across the datasets that use them
    seq2loc_path = "src/genomic_benchmarks/seq2loc/seq2loc.py"
    seq2loc_sha, _ = last_commit_for_path(GB_REPO, seq2loc_path)
    seq2loc_url = f"{RAW}/{GB_REPO}/{seq2loc_sha or 'main'}/{seq2loc_path}"
    seq2loc_src, seq2loc_st = fetch(seq2loc_url)
    print(f"  seq2loc.py  {seq2loc_st}  {seq2loc_url}")

    rows = []
    for ds in DATASETS:
        path = f"docs/{ds}/create_datasets.ipynb"
        sha, cdate = last_commit_for_path(GB_REPO, path)
        url = f"{RAW}/{GB_REPO}/{sha or 'main'}/{path}"
        txt, status = fetch(url)
        print(f"  {ds:26s} HTTP {status}  sha {sha[:12]}")

        row = {
            "dataset": ds,
            "notebook_url": url,
            "notebook_repo_path": path,
            "notebook_commit_sha": sha,
            "notebook_commit_date": cdate,
            "http_status": status,
            "access_date": ACCESS_DATE,
            "dedup_patterns_searched": " | ".join(DEDUP_PATTERNS),
        }

        if status != 200 or not txt:
            # Unreachable artifact. Say so; do not invent cell contents.
            row.update({"n_cells": "", "parse_ok": False,
                        "notes": "notebook UNREACHABLE -- no content parsed"})
            rows.append(row)
            continue

        try:
            nb = json.loads(txt)
        except Exception as e:
            row.update({"n_cells": "", "parse_ok": False,
                        "notes": f"notebook fetched but JSON parse failed: {e}"})
            rows.append(row)
            continue

        cells = nb.get("cells", [])
        src_all = "\n".join(l for _, _, l in cell_lines(nb))
        outs = outputs_text(nb)

        row["n_cells"] = len(cells)
        row["n_code_cells"] = sum(1 for c in cells if c.get("cell_type") == "code")
        row["parse_ok"] = True

        # ---- seq2loc / sequence->locus remapping
        s2l_lines = [f"cell{i}: {l}" for i, _, l in cell_lines(nb) if SEQ2LOC_RE.search(l)]
        row["uses_seq2loc"] = bool(s2l_lines)
        row["seq2loc_evidence"] = " || ".join(s2l_lines)

        # ---- what the notebook delegates to
        delegated = []
        deleg_urls = []
        scraper_sha = ""
        m = re.search(r"ensembl_scraper\.git@([0-9a-f]{7,40})", src_all)
        if m:
            scraper_sha = m.group(1)
        if re.search(r"ensembl_scraper", src_all):
            scraper_sha = scraper_sha or SCRAPER_SHA_FALLBACK
            for f in ["random_negatives.py", "ensembl_scraper.py", "preprocessing.py",
                      "utils.py", "config.py", "cli.py"]:
                u = f"{RAW}/{SCRAPER_REPO}/{scraper_sha}/scraper/{f}"
                s, st = fetch(u)
                deleg_urls.append(f"{u} [HTTP {st}]")
                if st == 200:
                    delegated.append((f"ensembl_scraper/{f}", s))
        if row["uses_seq2loc"] and seq2loc_st == 200:
            delegated.append(("genomic_benchmarks/seq2loc.py", seq2loc_src))
            deleg_urls.append(f"{seq2loc_url} [HTTP {seq2loc_st}]")

        row["delegates_construction_to_external_code"] = bool(delegated)
        row["delegated_source_urls"] = " || ".join(deleg_urls)
        row["ensembl_scraper_sha_pinned_by_notebook"] = (
            m.group(1) if m else ("NOT PINNED -- notebook installs from HEAD"
                                  if re.search(r"ensembl_scraper", src_all) else ""))

        # ---- the generous dedup search, notebook AND delegated code
        nb_hits = scan_dedup("notebook", src_all)
        dl_hits = []
        for name, s in delegated:
            dl_hits += scan_dedup(name, s)

        row["dedup_hits_notebook"] = len(nb_hits)
        row["dedup_hits_delegated"] = len(dl_hits)
        # Raw, unadjudicated result of the generous token search.
        row["dedup_step_found"] = bool(nb_hits or dl_hits)
        row["dedup_evidence"] = " || ".join(nb_hits + dl_hits)

        # Adjudicated, and this is the boolean the manuscript's claim actually rests on.
        labels, genuine = adjudicate(nb_hits + dl_hits)
        row["dedup_removes_duplicate_or_overlapping_seqs"] = genuine
        row["dedup_adjudication"] = " || ".join(labels)

        # ---- incidental, non-explicit collapse of EXACT duplicates
        # fasta2loc stores results in a dict. With use_seq_ids=False the key is the
        # SEQUENCE ITSELF, so two byte-identical input records silently become one row.
        # That is not an overlap-removal step, but it IS a de-facto exact-duplicate
        # filter, and the manuscript must not claim "no deduplication of any kind".
        s2l = dict(delegated).get("genomic_benchmarks/seq2loc.py", "")
        incidental = []
        if re.search(r"use_seq_ids\s*=\s*False", src_all) and re.search(r"sname = s\b", s2l):
            incidental.append(
                "fasta2loc(..., use_seq_ids=False) keys its results dict by the SEQUENCE "
                "string (seq2loc.py: `sname = s`), so byte-identical input records "
                "collapse to one row. Incidental exact-duplicate collapse, NOT overlap "
                "removal.")
        if re.search(r'position\["terminal"\] = ', s2l):
            incidental.append(
                "seq2loc._update_tree assigns `position[\"terminal\"] = (seq_name, ...)` "
                "unconditionally, so identical sequences (or a sequence and another's "
                "reverse complement) overwrite each other in the trie regardless of "
                "use_seq_ids. Again exact-match only.")
        if re.search(r"results\[pos\[.terminal.\]\[0\]\] = ", s2l):
            incidental.append(
                "seq2loc writes `results[name] = (chrom, start, end, strand)`, so a "
                "sequence occurring at several genomic loci keeps only the LAST locus "
                "found; multi-mapping is silently collapsed.")
        row["incidental_exact_dup_collapse"] = bool(incidental)
        row["incidental_collapse_note"] = " || ".join(incidental)

        # ---- counts the curators' own execution output reports
        read = re.findall(r"(\d+) sequences read and parsed", outs)
        found = re.findall(r"(\d+) sequences found in the reference", outs)
        row["n_seqs_read_reported"] = ";".join(read)
        row["n_seqs_found_reported"] = ";".join(found)

        ev, geom = tiling_evidence(outs)
        row["shipped_output_coord_evidence"] = ev
        row["shipped_output_geometry"] = geom

        # ---- quantify the incidental collapse against the actual upstream FASTA
        # Only for plain-FASTA inputs served over raw.githubusercontent (the cohn input
        # is a .tgz behind a university host and is deliberately not fetched here).
        fa_stats = []
        for u in re.findall(r"https://raw\.githubusercontent\.com/\S+\.fa\b", src_all):
            fa, st = fetch(u)
            if st != 200:
                fa_stats.append(f"{u.rsplit('/', 1)[-1]}: HTTP {st}, not counted")
                continue
            seqs, cur = [], []
            for line in fa.splitlines():
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                    cur = []
                elif line.strip():
                    cur.append(line.strip())
            if cur:
                seqs.append("".join(cur))
            lens = sorted({len(s) for s in seqs})
            fa_stats.append(
                f"{u.rsplit('/', 1)[-1]}: records={len(seqs)} unique_seqs={len(set(seqs))} "
                f"exact_dup_records={len(seqs) - len(set(seqs))} "
                f"seq_lengths={lens if len(lens) < 4 else lens[:4]}")
        row["upstream_fasta_dup_census"] = " || ".join(fa_stats)

        # ---- positives / negatives provenance, quoted from what was fetched
        wgets = [l.strip() for _, _, l in cell_lines(nb) if re.search(r"wget|pip install git\+", l)]
        row["download_lines_verbatim"] = " || ".join(wgets)

        deleg_map = dict(delegated)
        if "ensembl_scraper/ensembl_scraper.py" in deleg_map:
            body = deleg_map["ensembl_scraper/ensembl_scraper.py"]
            row["negatives_method"] = grab(
                r"4\) For each DNA sequence.*?in this way\.", body)
            row["positives_source"] = (
                "Ensembl regulatory/external-feature table, downloaded and parsed by "
                "ensembl_scraper (get_feature_class_loci -> parse_feature_file); "
                "notebook itself contains no locus logic. "
                f"notebook invocation: {grab(r'!python -m scraper[.]ensembl_scraper.*', src_all, 0)}")
            rn = deleg_map.get("ensembl_scraper/random_negatives.py", "")
            row["negatives_method_code_verbatim"] = " || ".join(
                x for x in [grab(r"def is_intersecting.*?return intersecting\.any\(\)", rn),
                            grab(r"def get_random_pos.*?return c, pos", rn),
                            grab(r"seq_length = int\(excluded_seqs.*?break", rn)] if x)
        elif row["uses_seq2loc"]:
            # Prefer a call that is unambiguously the negative one; cohn concatenates
            # positives and negatives into a single FASTA and makes ONE call, so fall
            # back to whatever fasta2loc call exists and say so.
            neg_call = (grab(r".*fasta2loc\(.*nonprom.*", src_all, 0) or
                        grab(r".*negative.*fasta2loc.*", src_all, 0) or
                        (grab(r".*fasta2loc\(.*", src_all, 0) +
                         "  [single call covering BOTH classes -- this notebook writes "
                         "positives and negatives into one FASTA first]"))
            row["negatives_method"] = (
                "Negatives are NOT generated by this notebook. They are taken verbatim "
                "from a third-party file downloaded by wget, then only remapped to "
                "GRCh38 coordinates by exact string search (fasta2loc). "
                f"remap call: {neg_call or 'see seq2loc_evidence'}")
            row["positives_source"] = (
                "Third-party FASTA downloaded by wget and remapped to GRCh38 by "
                f"fasta2loc; see download_lines_verbatim. {', '.join(wgets)[:300]}")
            row["negatives_method_code_verbatim"] = grab(
                r"def fasta2loc.*?return results", deleg_map.get(
                    "genomic_benchmarks/seq2loc.py", ""))
        else:
            row["negatives_method"] = "NOT DETERMINED from fetched sources"
            row["positives_source"] = "NOT DETERMINED from fetched sources"
            row["negatives_method_code_verbatim"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "gb_notebook_audit.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, {len(df.columns)} cols)")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:26s} HTTP {r['http_status']}  cells={r.get('n_cells')}  "
              f"seq2loc={r.get('uses_seq2loc')}  dedup_found={r.get('dedup_step_found')} "
              f"(nb={r.get('dedup_hits_notebook')}, delegated={r.get('dedup_hits_delegated')})")


if __name__ == "__main__":
    main()
