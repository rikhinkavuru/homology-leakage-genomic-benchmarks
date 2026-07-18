#!/usr/bin/env python3
"""
T1.1 evidence + G6 + G8 + G10, all via expkit (consistent with frozen pipeline).

1. RANKINGS under acc / AUROC / F1 on original vs corrected split, both leaky
   datasets, 4 models -> does the RF demotion survive a threshold-free metric?
2. P(model ranks 1st) bootstrap (sample-wise AND cluster/block) on each split
   -- the correct 'which model wins' estimator (G6).
3. CI-overlap test applied to the leaky demotions (the pairs the paper never tested).
4. G8: clean-dataset RF delta CIs (5 clean sets, k6+k4).
5. G10: sub-8bp / sub-20bp sequence counts per dataset + their share of the
   <0.5-similarity 'novel' bin on the leaky sets.
"""
import sys, time, json
import numpy as np, pandas as pd
import expkit as E

M4 = ["LR", "LinearSVC", "RF", "HGB"]
t0 = time.time()

def rank_of(accs, model):
    order = sorted(accs, key=lambda m: -accs[m])
    return order.index(model) + 1, order

def eval_all(seqs, y, tr, te, ks=(6,)):
    """Fit all 4 models (k6) on (tr->te); return per-model correctness + metrics."""
    out = {}
    for k in ks:
        X = E.featurize(seqs, k)
        for m in M4:
            r = E.metrics(E.models()[m], X, y, tr, te)
            c = (r["pred"] == y[te]).astype(float)
            out[f"{m}_k{k}"] = dict(acc=r["acc"], auroc=r["auroc"], f1=r["f1"], corr=c)
    return out

def p_rank1(corr_by_model, cl=None, boot=2000, seed=E.BSEED):
    """Bootstrap P(model has top accuracy). If cl given, cluster/block bootstrap."""
    models = list(corr_by_model)
    C = np.vstack([corr_by_model[m] for m in models])      # (nmodels, ntest)
    n = C.shape[1]; rng = np.random.RandomState(seed)
    wins = {m: 0 for m in models}
    if cl is None:
        for _ in range(boot):
            idx = rng.randint(0, n, size=n)
            acc = C[:, idx].mean(1)
            wins[models[int(acc.argmax())]] += 1
    else:
        uniq = np.unique(cl); G = len(uniq)
        members = [np.where(cl == u)[0] for u in uniq]
        for _ in range(boot):
            pick = rng.randint(0, G, size=G)
            idx = np.concatenate([members[p] for p in pick])
            acc = C[:, idx].mean(1)
            wins[models[int(acc.argmax())]] += 1
    return {m: wins[m] / boot for m in models}

# ---------------- 1-3: leaky-dataset rankings + rank-prob ----------------
rank_rows, prob_rows, overlap_rows = [], [], []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    o = eval_all(seqs, y, otr, ote)     # original
    c = eval_all(seqs, y, ctr, cte)     # corrected
    for metric in ["acc", "auroc", "f1"]:
        for split, res in [("original", o), ("corrected", c)]:
            vals = {m: res[f"{m}_k6"][metric] for m in M4}
            rrf, order = rank_of(vals, "RF")
            rank_rows.append(dict(dataset=d, metric=metric, split=split,
                                  rf_rank=rrf, best=order[0], order=">".join(order),
                                  **{f"{m}": round(vals[m], 4) for m in M4}))
    # P(rank1) on each split, sample + cluster bootstrap (test-internal clusters)
    for split, res, tr, te in [("original", o, otr, ote), ("corrected", c, ctr, cte)]:
        cl = E.clusters([seqs[i] for i in te], 0.7, 8)
        corr = {m: res[f"{m}_k6"]["corr"] for m in M4}
        ps = p_rank1(corr); pc = p_rank1(corr, cl=cl)
        for m in M4:
            prob_rows.append(dict(dataset=d, split=split, model=m,
                                  p_rank1_sample=round(ps[m], 3), p_rank1_cluster=round(pc[m], 3)))
    # CI overlap on the corrected-split demotion pairs (RF vs each model)
    from expkit import sample_boot, ci
    for m in M4:
        cc = c[f"{m}_k6"]["corr"]
        lo, hi = ci(sample_boot(cc))
        overlap_rows.append(dict(dataset=d, split="corrected", model=m,
                                 acc=round(float(cc.mean()), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    print(f"[{d}] rankings+probs done ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rank_rows).to_csv("results/exp_rankings.csv", index=False)
pd.DataFrame(prob_rows).to_csv("results/exp_rank_prob.csv", index=False)
pd.DataFrame(overlap_rows).to_csv("results/exp_corrected_ci.csv", index=False)

# ---------------- 4: G8 clean-dataset RF delta CIs ----------------
g8 = []
for d in E.CLEAN:
    seqs, y, otr, ote, tf = E.load(d)          # 20k subsample (frozen scale)
    comp = E.cached_clusters(d, seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    for k in (6, 4):
        X = E.featurize(seqs, k)
        co = (E.models()["RF"].fit(X[otr], y[otr]).predict(X[ote]) == y[ote]).astype(float)
        cc = (E.models()["RF"].fit(X[ctr], y[ctr]).predict(X[cte]) == y[cte]).astype(float)
        from expkit import sample_boot, ci
        rng = np.random.RandomState(E.BSEED)
        do = co[rng.randint(0, len(co), (1000, len(co)))].mean(1)
        dc = cc[rng.randint(0, len(cc), (1000, len(cc)))].mean(1)
        dd = do - dc; lo, hi = ci(dd)
        g8.append(dict(dataset=d, model=f"RF_k{k}", delta=round(float(co.mean()-cc.mean()), 4),
                       ci_lo=round(lo, 4), ci_hi=round(hi, 4), excludes0=bool(lo > 0 or hi < 0)))
    print(f"[clean {d}] RF delta CI done ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(g8).to_csv("results/exp_clean_rf_ci.csv", index=False)

# ---------------- 5: G10 short-sequence census ----------------
g10 = []
for d in E.LEAKY + E.CLEAN:
    seqs, y, otr, ote, tf = E.load(d)
    L = np.array([len(s) for s in seqs])
    sub8 = int((L < 8).sum()); sub20 = int((L < 20).sum())
    # share of the <0.5-sim novel test bin that is sub-8bp (leaky only meaningful)
    share = None
    if d in E.LEAKY:
        sim = E.cached_max_sim(d, seqs, otr, ote, 8)
        novel = ote[sim < 0.5]
        Lnov = np.array([len(seqs[i]) for i in novel])
        share = round(float((Lnov < 8).mean()), 4) if len(novel) else None
    g10.append(dict(dataset=d, n=len(seqs), sub8bp=sub8, sub20bp=sub20,
                    frac_sub8=round(sub8/len(seqs), 5), novelbin_sub8_share=share))
pd.DataFrame(g10).to_csv("results/exp_shortseq.csv", index=False)

print("\nEXP_STATS_DONE", time.time()-t0, "s")
print("RANKINGS(leaky,k6):")
print(pd.DataFrame(rank_rows)[["dataset","metric","split","rf_rank","best"]].to_string(index=False))
print("\nP(rank1):"); print(pd.DataFrame(prob_rows).to_string(index=False))
print("\nG10 shortseq:"); print(pd.DataFrame(g10).to_string(index=False))
