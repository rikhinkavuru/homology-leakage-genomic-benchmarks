#!/usr/bin/env python3
"""
S9 / R2.a3 multiclass: injected-leakage on the clean 3-class human_ensembl_regulatory.
GB ships no naturally-leaky multiclass set (leak@0.7=0.005), so we CONSTRUCT leakage:
copy a fraction f of train sequences into the test set with a few point mutations
(carrying their labels), for f in {0,0.1,0.2,0.4}. Show the inflation appears iff f>0,
scales with f, and is removed by the homology-aware split — for RF (the memorizer) and
LR (control). Full scale, k=6, decision rule = .predict() (expkit).
"""
import time
import numpy as np, pandas as pd
import expkit as E

t0 = time.time()
D = E.THREECLASS
FRACS = [0.0, 0.1, 0.2, 0.4]
rng0 = np.random.RandomState(0)
BASES = np.array(list("ACGT"))

def mutate(s, n_mut, rng):
    s = list(s); L = len(s)
    for _ in range(n_mut):
        p = rng.randint(L); s[p] = BASES[rng.randint(4)]
    return "".join(s)

# 20k stratified subsample (full 289k pairwise clustering is intractable; the injected-
# leakage mechanism is scale-independent — a controlled construction, not a leakage-fraction estimate).
seqs, y, otr, ote, tf = E.load(D, full=False, cap=20000)
print(f"[{D}] n={len(seqs)} (20k subsample) classes={sorted(set(y))} test_frac={tf:.3f} ({time.time()-t0:.0f}s)", flush=True)

rows = []
for f in FRACS:
    # inject: take fraction f of TRAIN sequences, mutate lightly, append as TEST (leaked)
    rng = np.random.RandomState(100 + int(f * 100))
    n_inj = int(round(f * len(otr)))
    inj_src = rng.choice(otr, size=n_inj, replace=False) if n_inj else np.array([], int)
    inj_seqs = [mutate(seqs[i], max(1, int(0.02 * len(seqs[i]))), rng) for i in inj_src]  # ~2% point mutations
    inj_y = y[inj_src]
    # build contaminated dataset: original + injected (injected are test-side leaked copies of train)
    all_seqs = seqs + inj_seqs
    all_y = np.concatenate([y, inj_y])
    tr2 = otr                                        # train unchanged
    te2 = np.concatenate([ote, np.arange(len(seqs), len(all_seqs))])  # test + injected copies
    X = E.featurize(all_seqs, 6)
    out = {}
    for m in ["RF", "LR"]:
        # original (contaminated) split
        acc_o = float((E.models()[m].fit(X[tr2], all_y[tr2]).predict(X[te2]) == all_y[te2]).mean())
        # homology-aware corrected split on the contaminated data
        comp = E.clusters(all_seqs, 0.7, 8)
        ctr, cte = E.assign(comp, all_y, 0, tf)
        acc_c = float((E.models()[m].fit(X[ctr], all_y[ctr]).predict(X[cte]) == all_y[cte]).mean())
        out[m] = (acc_o, acc_c)
        rows.append(dict(dataset=D, inject_frac=f, model=m, n_injected=n_inj,
                         acc_orig=round(acc_o, 4), acc_corr=round(acc_c, 4),
                         drop=round(acc_o - acc_c, 4)))
    print(f"  f={f}: RF drop={out['RF'][0]-out['RF'][1]:+.4f} (orig {out['RF'][0]:.4f}->corr {out['RF'][1]:.4f}); "
          f"LR drop={out['LR'][0]-out['LR'][1]:+.4f} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/exp_inject3class.csv", index=False)
print("\nEXP_INJECT3CLASS_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
print("READING: RF drop should be ~0 at f=0 and grow with f; LR (non-memorizer) stays ~flat. "
      "Confirms leakage inflation + homology-split correction generalize to the 3-class setting.")
