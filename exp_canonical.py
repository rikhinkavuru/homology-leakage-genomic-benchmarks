#!/usr/bin/env python3
"""
GB7/G5: reverse-complement canonicalization, both as leakage metric AND as features.

DNA is double-stranded and enhancers/OCRs are strand-symmetric, so forward-strand-only
k-mers (a) undercount leakage (revcomp near-duplicates invisible) and (b) are a
biologically wrong feature space that may inflate the RF's apparent memorization edge.
Re-measure leakage with canonical k-mers, and re-run the 4-model comparison with
canonical (revcomp-collapsed) features; does the RF demotion persist strand-symmetrically?
"""
import time
import numpy as np, pandas as pd
import expkit as E

t0 = time.time()
M4 = ["LR", "LinearSVC", "RF", "HGB"]
rows, leakrows = [], []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    # leakage: forward vs canonical
    jf = E.cached_max_sim(d, seqs, otr, ote, 8, canonical=False)
    jc = E.cached_max_sim(d, seqs, otr, ote, 8, canonical=True)
    leakrows.append(dict(dataset=d, leak_fwd_0p7=round(float((jf > 0.7).mean()), 4),
                         leak_canon_0p7=round(float((jc > 0.7).mean()), 4),
                         leak_fwd_0p9=round(float((jf > 0.9).mean()), 4),
                         leak_canon_0p9=round(float((jc > 0.9).mean()), 4)))
    print(f"[{d}] leak fwd0.7={leakrows[-1]['leak_fwd_0p7']} canon0.7={leakrows[-1]['leak_canon_0p7']} ({time.time()-t0:.0f}s)", flush=True)

    # canonical FEATURES, 4-model comparison, orig vs corrected split
    Xc = E.featurize_canonical(seqs, 6)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    acc_o, acc_c = {}, {}
    for m in M4:
        acc_o[m] = float((E.models()[m].fit(Xc[otr], y[otr]).predict(Xc[ote]) == y[ote]).mean())
        acc_c[m] = float((E.models()[m].fit(Xc[ctr], y[ctr]).predict(Xc[cte]) == y[cte]).mean())
    order_o = sorted(M4, key=lambda m: -acc_o[m]); order_c = sorted(M4, key=lambda m: -acc_c[m])
    for m in M4:
        rows.append(dict(dataset=d, features="canonical_k6", model=m,
                         acc_orig=round(acc_o[m], 4), acc_corr=round(acc_c[m], 4),
                         drop=round(acc_o[m] - acc_c[m], 4),
                         rank_orig=order_o.index(m) + 1, rank_corr=order_c.index(m) + 1))
    print(f"  [{d}] canonical-feat: RF rank {order_o.index('RF')+1}->{order_c.index('RF')+1}; "
          f"RF drop={round(acc_o['RF']-acc_c['RF'],4)} ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(leakrows).to_csv("results/exp_canonical_leak.csv", index=False)
pd.DataFrame(rows).to_csv("results/exp_canonical_models.csv", index=False)
print("\nEXP_CANONICAL_DONE", round(time.time()-t0), "s")
print("LEAK fwd vs canonical:\n", pd.DataFrame(leakrows).to_string(index=False))
print("MODELS (canonical features):\n", pd.DataFrame(rows).to_string(index=False))
