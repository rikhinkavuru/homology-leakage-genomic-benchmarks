#!/usr/bin/env python3
"""Download every Genomic Benchmarks dataset in the suite and report its
structure (class-folder names vary across datasets), counts, and a sample
sequence length. De-risks the slow/large downloads before the heavy run."""
import os, glob, sys, time
from genomic_benchmarks.loc2seq import download_dataset

DSETS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "demo_human_or_worm",
    "demo_coding_vs_intergenomic_seqs",
    "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl",   # smoke test
    "human_ensembl_regulatory",        # optional 3-class
]

for d in DSETS:
    t = time.time()
    try:
        p = download_dataset(d, version=0)
        info = {}
        for split in ("train", "test"):
            sd = os.path.join(p, split)
            if not os.path.isdir(sd):
                continue
            for cls in sorted(os.listdir(sd)):
                cd = os.path.join(sd, cls)
                if os.path.isdir(cd):
                    info[f"{split}/{cls}"] = len(glob.glob(os.path.join(cd, "*.txt")))
        anyf = next(iter(glob.glob(os.path.join(p, "train", "*", "*.txt"))), None)
        samp = ""
        if anyf:
            with open(anyf) as fh:
                samp = fh.read().strip()
        total = sum(info.values())
        print(f"OK   {d} :: total={total} :: {info} :: samplelen={len(samp)} :: ({time.time()-t:.0f}s)")
    except Exception as e:
        print(f"FAIL {d} :: {type(e).__name__}: {e} :: ({time.time()-t:.0f}s)")
    sys.stdout.flush()
print("PREP_DONE")
