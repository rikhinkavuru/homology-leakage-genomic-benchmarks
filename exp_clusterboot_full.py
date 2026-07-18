#!/usr/bin/env python3
"""
T1.5: complete the R3.3 cluster/block bootstrap answer.
- Leaky RF/LR k6: orig-acc + delta CI, sample vs cluster bootstrap (BOOT=10000),
  ICC + design effect (deff) via expkit.icc_oneway (replaces the informal 'seq/cluster'),
  analytic Liang-Zeger CRVE cross-check, and a COMBINED-SOURCE delta CI folding the
  5 re-split-seed variance in quadrature.
- Clean (5) LR+RF k6 delta cluster CI (does 'all clean deltas include 0' survive?).
- 3-class best-model delta cluster CI.
"""
import time
import numpy as np, pandas as pd
import expkit as E

t0 = time.time(); BOOT = 10000
def crve_mean_ci(c, cl):
    """Liang-Zeger cluster-robust 95% CI for a mean, clustered by cl."""
    c = np.asarray(c, float); n = len(c); cbar = c.mean()
    r = c - cbar
    uniq = np.unique(cl)
    meat = sum((r[cl == u].sum()) ** 2 for u in uniq)
    se = np.sqrt(meat) / n
    return cbar, cbar - 1.96 * se, cbar + 1.96 * se, se

def within_test_clusters(seqs, te):
    return E.clusters([seqs[i] for i in te], 0.7, 8)

rows = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    X = E.featurize(seqs, 6)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    # 5 re-split seeds corrected accuracy (combined-source variance)
    seed_acc = {m: [] for m in ["RF", "LR"]}
    corr0 = {}
    for s in range(5):
        ctr, cte = E.assign(comp, y, s, tf)
        for m in ["RF", "LR"]:
            cc = (E.models()[m].fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).astype(float)
            seed_acc[m].append(float(cc.mean()))
            if s == 0:
                corr0[m] = (cc, cte)
    ocl = within_test_clusters(seqs, ote)
    for m in ["RF", "LR"]:
        c_o = (E.models()[m].fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).astype(float)
        cc, cte = corr0[m]
        ccl = within_test_clusters(seqs, cte)
        # ICC / deff on original leaky test (where near-dup clusters live)
        dd = E.icc_oneway(c_o, ocl)
        # bootstrap delta (orig - corrected), sample vs cluster
        rng = np.random.RandomState(E.BSEED)
        do_s = E.sample_boot(c_o, BOOT); dc_s = E.sample_boot(cc, BOOT)
        d_s = do_s - dc_s; lo_s, hi_s = E.ci(d_s)
        do_c = E.cluster_boot(c_o, ocl, BOOT); dc_c = E.cluster_boot(cc, ccl, BOOT)
        d_c = do_c - dc_c; lo_c, hi_c = E.ci(d_c)
        # combined-source: fold 5-seed corrected-acc SD into the delta variance
        seed_sd = np.std(seed_acc[m], ddof=1)
        delta = float(c_o.mean() - cc.mean())
        boot_sd = np.std(d_c)                      # cluster-bootstrap SD of the delta
        comb_sd = np.sqrt(boot_sd ** 2 + seed_sd ** 2)
        cb_lo, cb_hi = delta - 1.96 * comb_sd, delta + 1.96 * comb_sd
        # CRVE cross-check on original acc
        _, crlo, crhi, crse = crve_mean_ci(c_o, ocl)
        rows.append(dict(dataset=d, model=f"{m}_k6", delta=round(delta, 4),
                         icc=round(dd["icc"], 4), deff=round(dd["deff"], 3), n_clusters=dd["n_clusters"],
                         delta_ci_sample=[round(lo_s, 4), round(hi_s, 4)], excl0_sample=bool(lo_s > 0 or hi_s < 0),
                         delta_ci_cluster=[round(lo_c, 4), round(hi_c, 4)], excl0_cluster=bool(lo_c > 0 or hi_c < 0),
                         seed_sd=round(float(seed_sd), 4),
                         delta_ci_combined=[round(cb_lo, 4), round(cb_hi, 4)], excl0_combined=bool(cb_lo > 0 or cb_hi < 0),
                         orig_acc_CRVE_ci=[round(crlo, 4), round(crhi, 4)]))
        print(f"[{d}] {m}: delta={delta:+.4f} icc={dd['icc']:.4f} deff={dd['deff']:.2f} "
              f"sampleCI={rows[-1]['delta_ci_sample']} clusterCI={rows[-1]['delta_ci_cluster']} "
              f"combinedCI={rows[-1]['delta_ci_combined']} ({time.time()-t0:.0f}s)", flush=True)

# clean 5 + 3-class deltas, cluster bootstrap
for d in E.CLEAN + [E.THREECLASS]:
    seqs, y, otr, ote, tf = E.load(d, full=(d == E.THREECLASS))
    X = E.featurize(seqs, 6)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    ocl = within_test_clusters(seqs, ote); ccl = within_test_clusters(seqs, cte)
    ms = ["RF", "LR"] if d != E.THREECLASS else ["RF"]
    for m in ms:
        c_o = (E.models()[m].fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).astype(float)
        cc = (E.models()[m].fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).astype(float)
        do_c = E.cluster_boot(c_o, ocl, BOOT); dc_c = E.cluster_boot(cc, ccl, BOOT)
        d_c = do_c - dc_c; lo, hi = E.ci(d_c)
        rows.append(dict(dataset=d, model=f"{m}_k6", delta=round(float(c_o.mean()-cc.mean()), 4),
                         icc=None, deff=None, n_clusters=None,
                         delta_ci_cluster=[round(lo, 4), round(hi, 4)], excl0_cluster=bool(lo > 0 or hi < 0)))
    print(f"[clean {d}] done ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/cluster_bootstrap_full.csv", index=False)
print("\nEXP_CLUSTERBOOT_FULL_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
