#!/usr/bin/env python3
"""Scale-consistent Jaccard + containment leak fractions for the clean datasets, so
the report card's containment column is never mixed-scale (the audit caught full-scale
Jaccard paired with 20k containment -> impossible con<Jac). Computes BOTH metrics on the
SAME loaded data (so con>=Jac always holds), trying full scale first and falling back to
the cached 20k subsample if the full-scale download is unavailable; the scale used is
reported per dataset. Serial, memory-safe."""
import time, urllib.error
import numpy as np, pandas as pd
import expkit as E
import homology_split as H

t0 = time.time()
CLEAN = ["demo_coding_vs_intergenomic_seqs", "human_ocr_ensembl",
         "demo_human_or_worm", "human_enhancers_cohn", "drosophila_enhancers_stark"]

def load_best(d):
    for attempt in range(2):
        try:
            seqs, y, otr, ote, tf = E.load(d, full=True)
            return seqs, otr, ote, "full"
        except (urllib.error.ContentTooShortError, urllib.error.URLError, OSError) as e:
            print(f"  [{d}] full-scale load attempt {attempt+1} failed ({type(e).__name__}); retrying/falling back", flush=True)
    seqs, y, otr, ote, tf = E.load(d, full=False, cap=20000)   # cached 20k fallback
    return seqs, otr, ote, "20k"

def leak_fracs(seqs, tr, te):
    M = H.kmer_binary_matrix(seqs, 8)
    sq = np.asarray(M[te].sum(1)).ravel(); sr = np.asarray(M[tr].sum(1)).ravel()
    RT = M[tr].T.tocsr()
    jac = np.zeros(len(te)); con = np.zeros(len(te))
    for s in range(0, len(te), 500):
        e = min(s + 500, len(te))
        inter = (M[te][s:e] @ RT).toarray()
        union = sq[s:e, None] + sr[None, :] - inter
        mins = np.minimum(sq[s:e, None], sr[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            jj = np.where(union > 0, inter / union, 0.0)
            cc = np.where(mins > 0, inter / mins, 0.0)
        jac[s:e] = jj.max(1); con[s:e] = cc.max(1)
    return jac, con

rows = []
for d in CLEAN:
    seqs, otr, ote, scale = load_best(d)
    jac, con = leak_fracs(seqs, otr, ote)
    r = dict(dataset=d, scale=scale, n=len(seqs),
             leak_jac_0p7=round(float((jac > 0.7).mean()), 4),
             leak_con_0p7=round(float((con > 0.7).mean()), 4),
             leak_con_0p9=round(float((con > 0.9).mean()), 4),
             verdict_con=("LEAKY" if (con > 0.7).mean() > 0.1 else "clean"))
    rows.append(r)
    print(f"[{d}] scale={scale} n={len(seqs)} jac={r['leak_jac_0p7']} con={r['leak_con_0p7']} "
          f"verdict_con={r['verdict_con']} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/full_scale_containment.csv", index=False)
print("\nFULL_SCALE_CONTAINMENT_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
