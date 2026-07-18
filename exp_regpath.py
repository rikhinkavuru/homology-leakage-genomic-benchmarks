#!/usr/bin/env python3
"""
T1.6 / C-2: is the leakage effect an untuned-RandomForest-defaults artifact?

Drive an RF regularization PATH on both leaky datasets (full scale, k=6) and ask,
at each setting: does it still MEMORIZE (graded gap acc[>=0.9]-acc[<0.5]) and does it
still get DEMOTED after the homology split? The graded gap is the confound-immune
headline (immune to 'a regularized RF just got worse everywhere'). We also record the
homology drop, novel-only accuracy (generalization), and RF's rank vs the 3 fixed
baselines on the corrected split.

Axes: min_samples_leaf {1,5,20,50,100,200,500} (200/500 EXCEED the max near-dup
cluster: 143 ensembl, 420 nontata -> no single leaf can be a pure duplicate block),
plus max_depth {None,16,8,4}. n_estimators=150, seed=0 (match frozen RF).
"""
import time, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import expkit as E

t0 = time.time()
MSL = [1, 5, 20, 50, 100, 200, 500]
MD = [None, 16, 8, 4]
M3 = ["LR", "LinearSVC", "HGB"]        # fixed baselines for the ranking question

def graded_gap(corr, sim):
    hi = corr[sim >= 0.9]; lo = corr[sim < 0.5]
    g = (hi.mean() - lo.mean()) if (len(hi) and len(lo)) else np.nan
    return float(g), float(hi.mean()) if len(hi) else np.nan, float(lo.mean()) if len(lo) else np.nan, int(len(hi)), int(len(lo))

rows = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    X = E.featurize(seqs, 6)
    sim = E.cached_max_sim(d, seqs, otr, ote, 8)         # test->train max sim (orig split)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    # fixed-baseline accuracies (orig + corrected) for the ranking question
    base_o, base_c = {}, {}
    for m in M3:
        base_o[m] = float((E.models()[m].fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).mean())
        base_c[m] = float((E.models()[m].fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).mean())
    print(f"[{d}] baselines done ({time.time()-t0:.0f}s)", flush=True)

    def run_rf(**kw):
        rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0, **kw)
        rf.fit(X[otr], y[otr])
        co = (rf.predict(X[ote]) == y[ote]).astype(float)
        rf2 = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0, **kw)
        rf2.fit(X[ctr], y[ctr])
        cc = (rf2.predict(X[cte]) == y[cte]).astype(float)
        gap, acchi, acclo, nhi, nlo = graded_gap(co, sim)
        acc_o = float(co.mean()); acc_c = float(cc.mean())
        novel = acclo                                       # orig-trained acc on <0.5 bin
        # RF rank on corrected split among {RF_this, LR, SVC, HGB}
        accs = dict(RF=acc_c, **base_c); order = sorted(accs, key=lambda k: -accs[k])
        rf_rank_c = order.index("RF") + 1
        accs_o = dict(RF=acc_o, **base_o); order_o = sorted(accs_o, key=lambda k: -accs_o[k])
        rf_rank_o = order_o.index("RF") + 1
        return dict(acc_orig=round(acc_o, 4), acc_corr=round(acc_c, 4),
                    drop=round(acc_o - acc_c, 4), graded_gap=round(gap, 4),
                    acc_hi=round(acchi, 4), novel_acc=round(novel, 4),
                    rf_rank_orig=rf_rank_o, rf_rank_corr=rf_rank_c)

    for msl in MSL:
        r = run_rf(min_samples_leaf=msl)
        rows.append(dict(dataset=d, axis="min_samples_leaf", value=msl, **r))
        print(f"  msl={msl}: gap={r['graded_gap']} drop={r['drop']} rf_rank_corr={r['rf_rank_corr']} "
              f"novel={r['novel_acc']} ({time.time()-t0:.0f}s)", flush=True)
    for md in MD:
        r = run_rf(max_depth=md)
        rows.append(dict(dataset=d, axis="max_depth", value=str(md), **r))
        print(f"  max_depth={md}: gap={r['graded_gap']} drop={r['drop']} rf_rank_corr={r['rf_rank_corr']} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/exp_regpath.csv", index=False)
print("\nEXP_REGPATH_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
