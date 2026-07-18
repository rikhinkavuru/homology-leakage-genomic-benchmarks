#!/usr/bin/env python3
"""Serial, single-process prefetch of every (dataset, cap, seed) the revision needs.
Builds datacache/*.pkl so all downstream parallel experiments read the pickle and
NEVER call genomic_benchmarks download_dataset (which force-re-extracts and races)."""
import os, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import expkit as E

# remove any corrupt partial zip so nontata re-downloads cleanly, once
import glob
for z in glob.glob(os.path.expanduser("~/.genomic_benchmarks/*.zip")):
    try: os.remove(z); print("removed stale zip", z)
    except OSError: pass

BIG = 10**12
COMBOS = [
    ("human_nontata_promoters", BIG, 0),
    ("human_enhancers_ensembl", BIG, 0),
    ("human_enhancers_cohn", 20000, 0),
    ("human_ocr_ensembl", 20000, 0),
    ("demo_human_or_worm", 20000, 0),
    ("demo_coding_vs_intergenomic_seqs", 20000, 0),
    ("drosophila_enhancers_stark", 20000, 0),
    ("human_ensembl_regulatory", BIG, 0),
    ("human_ensembl_regulatory", 20000, 0),
]
t0 = time.time()
for dset, cap, seed in COMBOS:
    df = E._load_df(dset, cap, seed)
    print(f"cached {dset} cap={cap} seed={seed}: n={len(df)} "
          f"classes={sorted(df['label'].unique())} ({time.time()-t0:.0f}s)", flush=True)
print("PREFETCH_DONE", round(time.time()-t0), "s")
