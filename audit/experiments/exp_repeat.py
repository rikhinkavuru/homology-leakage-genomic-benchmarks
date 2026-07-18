#!/usr/bin/env python3
"""
GB3: characterize the leaked near-duplicate pairs — are they low-complexity /
repeat-driven (dustmasker) or unique sequence? Determines whether the homology-aware
removal is a fix (locus duplication) or an over-correction (genuine repeat-family biology).
Reports the low-complexity-masked fraction of leaked vs non-leaked test sequences.
"""
import os, subprocess, time
import numpy as np, pandas as pd
from audit.core import expkit as E

t0 = time.time()
TMP = "/private/tmp/claude-502/-Users-rikhinkavuru-homology-audit/a0a824c4-7be0-4f3c-ad5a-164c624bc48e/scratchpad/dust"
os.makedirs(TMP, exist_ok=True)

def dust_masked_fraction(seqs, tag):
    fa = os.path.join(TMP, f"{tag}.fasta")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    out = os.path.join(TMP, f"{tag}.masked")
    subprocess.run(["dustmasker", "-in", fa, "-out", out, "-outfmt", "fasta"], check=True, capture_output=True)
    # parse: lowercase = masked low-complexity
    frac, cur = [], []
    def flush():
        if cur:
            s = "".join(cur); L = len(s)
            frac.append(sum(1 for ch in s if ch.islower()) / L if L else 0.0)
    with open(out) as fh:
        for line in fh:
            if line.startswith(">"):
                flush(); cur = []
            else:
                cur.append(line.strip())
    flush()
    return np.array(frac)

rows = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    sim = E.cached_max_sim(d, seqs, otr, ote, 8)
    leaked = ote[sim >= 0.9]
    nonleaked = ote[sim < 0.5]
    # sample to keep dustmasker fast
    rng = np.random.RandomState(0)
    lk = leaked[rng.choice(len(leaked), min(3000, len(leaked)), replace=False)] if len(leaked) else leaked
    nl = nonleaked[rng.choice(len(nonleaked), min(3000, len(nonleaked)), replace=False)] if len(nonleaked) else nonleaked
    fl = dust_masked_fraction([seqs[i] for i in lk], f"{d}_leaked")
    fn = dust_masked_fraction([seqs[i] for i in nl], f"{d}_nonleaked")
    rows.append(dict(dataset=d, n_leaked=len(leaked), n_novel=len(nonleaked),
                     leaked_lowcomplex_frac=round(float(fl.mean()), 4),
                     novel_lowcomplex_frac=round(float(fn.mean()), 4),
                     leaked_frac_mostly_masked=round(float((fl > 0.5).mean()), 4)))
    print(f"[{d}] leaked low-complexity mask={rows[-1]['leaked_lowcomplex_frac']} "
          f"vs novel={rows[-1]['novel_lowcomplex_frac']} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/exp_repeat.csv", index=False)
print("\nEXP_REPEAT_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
print("READING: if leaked low-complexity fraction ~= novel, the leakage is NOT repeat/low-complexity "
      "artifact but genuine sequence duplication (consistent with the mostly-pairwise cluster topology "
      "= discrete locus duplication, not TE-family expansion).")
