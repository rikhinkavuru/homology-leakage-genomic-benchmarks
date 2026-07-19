#!/usr/bin/env python3
"""
Uncertainty quantification on the frozen results (adds variance/CIs; changes nothing).

STEP 2  re-split seed variance: homology-aware corrected accuracy over 5 cluster->side
        seeds, LR k6 (RF k6 5-seed already in splitter_validation.csv -> reused; this
        run also refits RF k6 seed 0 as a cross-check).
STEP 3  bootstrap 95%% CIs over the TEST set (resample stored per-example correctness,
        1000 draws; NO model refit beyond the single fit needed to obtain predictions,
        since predictions were never stored). Original & homology-aware accuracy + their
        delta, both leaky datasets (RF k6 best, LR k6 other) and the 5 clean datasets
        (best model LR k6).
STEP 4  'statistically tied': for the clean-dataset rank swaps, do the swapped models'
        accuracy CIs overlap?

Scale matches the frozen results: FULL for leaky, 20k subsample (seed 0) for clean.
Split method matches splitter_validation (k=8 Jaccard > 0.7 components, whole-cluster
per-class assignment). Deterministic; bootstrap seed reported.
"""
import os, time, gc, itertools
import numpy as np
import pandas as pd


def _group_of(dataset, _cache={}):
    """The dataset's verdict from the released report card; 'clean' if absent."""
    if not _cache:
        try:
            import os
            _rc = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'results',
                'leakage_report_card.csv'))
            _cache.update(dict(zip(_rc.dataset, _rc.verdict)))
        except Exception:
            _cache['__none__'] = True
    v = _cache.get(dataset, 'clean')
    return 'clean' if v == 'LEAKY' and False else (v if v in ('clean', 'borderline') else 'clean')
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S
from audit.core import homology_split as H
from audit.pipeline.run_extended_models import make_models

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
CLEAN = ["human_enhancers_cohn", "human_ocr_ensembl", "demo_human_or_worm",
         "demo_coding_vs_intergenomic_seqs", "drosophila_enhancers_stark"]
SEEDS = [0, 1, 2, 3, 4]
BOOT = 1000
BSEED = 20240524                     # bootstrap RNG seed (reported)
M4 = ["LR", "LinearSVC", "RF", "HGB"]
R = RA.RESULTS_DIR


def load(d):
    cap = 10**12 if d in LEAKY else 20000
    df, info = S.discover_and_subsample(d, cap=cap, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    otr = np.where(df["split"].to_numpy() == "train")[0]
    ote = np.where(df["split"].to_numpy() == "test")[0]
    return seqs, y, otr, ote, len(ote) / len(df)


def components(seqs):
    M = H.kmer_binary_matrix(seqs, 8)
    er, ec = H._edges(M, 0.7)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    _, comp = connected_components(g + g.T, directed=False)
    return comp


def correct(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    return (model.predict(X[te]) == y[te])


def acc_ci(c, seed=BSEED):
    rng = np.random.RandomState(seed)
    c = np.asarray(c, dtype=np.float64); n = len(c)
    draws = c[rng.randint(0, n, size=(BOOT, n))].mean(axis=1)
    return float(c.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def delta_ci(c_orig, c_corr, seed=BSEED):
    rng = np.random.RandomState(seed)
    co = np.asarray(c_orig, float); cc = np.asarray(c_corr, float)
    do = co[rng.randint(0, len(co), size=(BOOT, len(co)))].mean(1)
    dc = cc[rng.randint(0, len(cc), size=(BOOT, len(cc)))].mean(1)
    d = do - dc
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return float(co.mean() - cc.mean()), lo, hi, bool(lo > 0 or hi < 0)


def main():
    t0 = time.time()
    step2, acc_rows, delta_rows, tie_rows = [], [], [], []
    rank = pd.read_csv(f"{R}/ranking_inversion.csv")

    # ---------------- LEAKY: Step 2 (LR k6 x5) + Step 3 (CIs, RF/LR k6) ----------------
    for d in LEAKY:
        seqs, y, otr, ote, tf = load(d)
        X = RA.featurize(seqs, 6)
        comp = components(seqs)
        splits = {s: H._assign(comp, y, s, tf) for s in SEEDS}
        print(f"[{d}] loaded n={len(y)} test_frac={tf:.3f} ({time.time()-t0:.0f}s)", flush=True)

        # Step 2: LR k6 over 5 seeds; RF k6 seed0 cross-check
        lr_acc, corr0 = [], {}
        for s in SEEDS:
            tr, te = splits[s]
            c = correct(make_models()["LR"], X, y, tr, te)
            lr_acc.append(float(c.mean()))
            if s == 0:
                corr0["LR"] = c
        a = np.array(lr_acc)
        step2.append(dict(dataset=d, model="LR_k6", n_seeds=5, per_seed=str([round(x, 4) for x in lr_acc]),
                          mean=round(a.mean(), 4), sd=round(a.std(ddof=0), 4),
                          min=round(a.min(), 4), max=round(a.max(), 4), range=round(a.max() - a.min(), 4)))
        corr0["RF"] = correct(make_models()["RF"], X, y, *splits[0])   # RF k6 corrected seed0
        print(f"  Step2 LR k6 5-seed mean={a.mean():.4f} sd={a.std(ddof=0):.4f}; "
              f"RF k6 seed0 corrected acc={corr0['RF'].mean():.4f} (cross-check) ({time.time()-t0:.0f}s)", flush=True)

        # Step 3: original-split CIs + delta (RF k6 best, LR k6 other)
        for m in ["RF", "LR"]:
            c_orig = correct(make_models()[m], X, y, otr, ote)
            ao, lo, hi = acc_ci(c_orig)
            ac, clo, chi = acc_ci(corr0[m])
            acc_rows.append(dict(dataset=d, scale="full", model=f"{m}_k6", split="original",
                                 n_test=len(ote), accuracy=round(ao, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
            acc_rows.append(dict(dataset=d, scale="full", model=f"{m}_k6", split="homology@0.7 (seed0)",
                                 n_test=int(corr0[m].shape[0]), accuracy=round(ac, 4), ci_lo=round(clo, 4), ci_hi=round(chi, 4)))
            dm, dlo, dhi, exz = delta_ci(c_orig, corr0[m])
            delta_rows.append(dict(dataset=d, group="leaky", model=f"{m}_k6",
                                   delta_orig_minus_corr=round(dm, 4), ci_lo=round(dlo, 4),
                                   ci_hi=round(dhi, 4), excludes_zero=exz))
        del X; gc.collect()

    # ---------------- CLEAN: Step 3 delta (LR k6) + Step 4 (tie overlap) ----------------
    for d in CLEAN:
        seqs, y, otr, ote, tf = load(d)
        X = RA.featurize(seqs, 6)
        comp = components(seqs)
        tr0, te0 = H._assign(comp, y, 0, tf)
        cob, ci = {}, {}
        for m in M4:
            cob[m] = correct(make_models()[m], X, y, otr, ote)
            ci[m] = acc_ci(cob[m])           # (acc, lo, hi) on ORIGINAL split
            acc_rows.append(dict(dataset=d, scale="subsample20k", model=f"{m}_k6", split="original",
                                 n_test=len(ote), accuracy=round(ci[m][0], 4),
                                 ci_lo=round(ci[m][1], 4), ci_hi=round(ci[m][2], 4)))
        # delta (best model LR k6): original vs corrected seed0
        c_corr_LR = correct(make_models()["LR"], X, y, tr0, te0)
        dm, dlo, dhi, exz = delta_ci(cob["LR"], c_corr_LR)
        # Group label comes from the report card, not a hardcoded "clean": the
        # verdict went three-way in round 20 and this line kept labelling
        # demo_coding clean, contradicting section 7 and section 13 of the same
        # generated document.
        delta_rows.append(dict(dataset=d, group=_group_of(d), model="LR_k6",
                               delta_orig_minus_corr=round(dm, 4), ci_lo=round(dlo, 4),
                               ci_hi=round(dhi, 4), excludes_zero=exz))
        # Step 4: swapped pairs (from ranking_inversion k6 accuracy) -> CI overlap on original split
        rr = rank[(rank.dataset == d) & (rank.k == 6) & (rank.metric == "accuracy")].iloc[0]
        ro = rr["original_ranking"].split(">"); rc = rr["corrected_ranking"].split(">")
        poso = {m: i for i, m in enumerate(ro)}; posc = {m: i for i, m in enumerate(rc)}
        swaps = set()
        for a_, b_ in itertools.combinations(M4, 2):
            if (poso[a_] - poso[b_]) * (posc[a_] - posc[b_]) < 0:   # order flipped
                swaps.add(tuple(sorted((a_, b_))))
        for a_, b_ in sorted(swaps):
            (aa, alo, ahi), (ba, blo, bhi) = ci[a_], ci[b_]
            overlap = not (ahi < blo or bhi < alo)
            tie_rows.append(dict(dataset=d, model_a=a_, acc_a=round(aa, 4), ci_a=f"[{alo:.4f},{ahi:.4f}]",
                                 model_b=b_, acc_b=round(ba, 4), ci_b=f"[{blo:.4f},{bhi:.4f}]",
                                 ci_overlap=overlap, gap=round(abs(aa - ba), 4)))
        print(f"[{d}] clean delta LR k6 = {dm:+.4f} CI[{dlo:+.4f},{dhi:+.4f}] excl0={exz}; "
              f"swaps={sorted(swaps)} ({time.time()-t0:.0f}s)", flush=True)
        del X; gc.collect()

    pd.DataFrame(step2).to_csv(f"{R}/step2_seed_variance.csv", index=False)
    pd.DataFrame(acc_rows).to_csv(f"{R}/step3_accuracy_ci.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(f"{R}/step3_delta_ci.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(f"{R}/step4_tie_overlap.csv", index=False)
    print("\n=== STEP 2 (LR k6 5-seed) ==="); print(pd.DataFrame(step2).to_string(index=False), flush=True)
    print("\n=== STEP 3 delta CIs ==="); print(pd.DataFrame(delta_rows).to_string(index=False), flush=True)
    print("\n=== STEP 4 tie overlap ==="); print(pd.DataFrame(tie_rows).to_string(index=False), flush=True)
    print(f"\nVARCI_DONE bootstrap_seed={BSEED} draws={BOOT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
