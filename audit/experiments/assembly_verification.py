#!/usr/bin/env python3
"""Empirically determine which human assembly the Genomic Benchmarks intervals
are stated in, by fetching reference sequence at the shipped coordinates from
BOTH hg38 and hg19 via the UCSC REST API and comparing to the shipped sequence.

Conventions tested:
  - 0-based half-open (BED/UCSC API native): start=start, end=end
  - 1-based inclusive interpretation of the CSV: start=start-1, end=end
Strand '-' -> reverse complement of the fetched + strand sequence.
"""
import os, glob, json, time, urllib.request, urllib.parse, random
import pandas as pd

HERE = "/Users/rikhinkavuru/homology_audit"
GBROOT = os.path.expanduser("~/.genomic_benchmarks")
IV = os.path.join(HERE, "datacache", "intervals")
API = "https://api.genome.ucsc.edu/getData/sequence"
CACHE = "/private/tmp/claude-502/-Users-rikhinkavuru-homology-audit/79894bc5-542d-4ec4-b08a-3a6f54a35af6/scratchpad/ucsc_cache.json"

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def fetch(genome, chrom, start, end):
    key = f"{genome}:{chrom}:{start}-{end}"
    if key in _cache:
        return _cache[key]
    url = API + "?" + urllib.parse.urlencode(
        dict(genome=genome, chrom=chrom, start=start, end=end))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            seq = d.get("dna", "").upper()
            _cache[key] = seq
            json.dump(_cache, open(CACHE, "w"))
            return seq
        except Exception as e:
            if attempt == 3:
                print("  FETCH FAIL", key, e)
                return ""
            time.sleep(2 * (attempt + 1))
    return ""

def shipped(dset, split, cls, _id):
    p = os.path.join(GBROOT, dset, split, cls, f"{_id}.txt")
    with open(p) as f:
        return f.read().strip().upper()

def run(dset, n_per_class=13, seed=20260719):
    rows = []
    rng = random.Random(seed)
    for cls in sorted(os.listdir(os.path.join(GBROOT, dset, "test"))):
        if not os.path.isdir(os.path.join(GBROOT, dset, "test", cls)):
            continue
        df = pd.read_csv(os.path.join(IV, dset, f"test_{cls}.csv.gz"))
        df["id"] = df["id"].astype(str)
        # stratify by strand so both are represented
        idx = []
        for st in df["strand"].unique():
            sub = df[df["strand"] == st]
            k = min(len(sub), max(1, n_per_class // max(1, df["strand"].nunique())))
            idx += rng.sample(list(sub.index), k)
        for i in idx:
            r = df.loc[i]
            obs = shipped(dset, "test", cls, r["id"])
            rec = dict(dataset=dset, cls=cls, id=r["id"], chrom=r["region"],
                       start=int(r["start"]), end=int(r["end"]), strand=r["strand"],
                       len_shipped=len(obs), len_interval=int(r["end"]) - int(r["start"]))
            for genome in ("hg38", "hg19"):
                for conv, (s, e) in dict(
                        halfopen=(int(r["start"]), int(r["end"])),
                        onebased=(int(r["start"]) - 1, int(r["end"]))).items():
                    ref = fetch(genome, r["region"], s, e)
                    if not ref:
                        rec[f"{genome}_{conv}"] = None
                        rec[f"{genome}_{conv}_unevaluable"] = "fetch_failed"
                        continue
                    cand = rc(ref) if r["strand"] == "-" else ref
                    rec[f"{genome}_{conv}"] = (cand == obs)
                    # also record fwd-strand-ignoring-strand match
                    if conv == "halfopen":
                        rec[f"{genome}_fwd_ignorestrand"] = (ref == obs)
            rows.append(rec)
            print(".", end="", flush=True)
    print()
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for d in ["human_enhancers_ensembl", "human_nontata_promoters"]:
        df = run(d)
        out.append(df)
        print(f"\n=== {d} (n={len(df)}) ===")
        for c in ["hg38_halfopen", "hg38_onebased", "hg19_halfopen", "hg19_onebased",
                  "hg38_fwd_ignorestrand", "hg19_fwd_ignorestrand"]:
            v = df[c].dropna()
            print(f"  {c:26s} {v.sum():3d}/{len(v):3d} = {v.mean():.3f}")
        print("  len_shipped == len_interval:",
              (df.len_shipped == df.len_interval).sum(), "/", len(df))
        print("  strand counts:", df.strand.value_counts().to_dict())
    full = pd.concat(out, ignore_index=True)
    full["access_date"] = "2026-07-20"
    full["api"] = API
    full.to_csv(os.path.join(HERE, "results", "assembly_verification.csv"), index=False)
    print("\n=== POOLED (n=%d) ===" % len(full))
    for c in ["hg38_halfopen", "hg38_onebased", "hg19_halfopen", "hg19_onebased"]:
        v = full[c].dropna()
        print(f"  {c:20s} {int(v.sum()):3d}/{len(v):3d}")
    print("  rows where the GRCh37 fetch is unevaluable:",
          int(full.get("hg19_halfopen_unevaluable", pd.Series(dtype=object)).notna().sum()))
