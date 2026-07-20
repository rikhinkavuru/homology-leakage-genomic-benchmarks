#!/usr/bin/env python3
"""
Decompose the headline accuracy drop into an EVALUATION-side and a SPLIT-side part.

THE CONFOUND (reviewer item B1)
-------------------------------
The paper's headline is "correcting the split costs the best model 16.5 points". That
number confounds two different things:

  (i)  EVALUATION-side inflation. The as-shipped test set contains near-duplicates of
       training sequences. Simply reporting accuracy on the novel stratum instead of on
       the whole test set lowers the score, with NO retraining and NO re-splitting. This
       is what the benchmark's headline number overstates about a model that already
       exists.
  (ii) SPLIT-side effect. A near-duplicate-aware re-split additionally strips the
       near-duplicate partners out of TRAINING, so the model is refit with less
       (and less redundant) data. That is a retraining penalty, not a measurement error.

This script supplies the third arm B1 asks for: ONE fit on the as-shipped training set,
evaluated twice -- (a) the full as-shipped test set, (b) the novel-only stratum
(max 8-mer Jaccard to training < 0.5). Their difference is (i), uncontaminated by any
retraining. The corrected re-split accuracy (frozen, homology@0.7, from
extended_models_long.csv) supplies (ii) as the remainder.

THE SECOND CONFOUND, which we do NOT hide
-----------------------------------------
The novel stratum has a different class composition from the whole test set (on
human_enhancers_ensembl the >=0.9 stratum is 99.97% positive and the novel stratum
19.68% positive), so part of the raw evaluation-side gap is a prevalence shift rather
than lost memorization. Exactly as in exp_graded_corrected.py we therefore report the
BALANCED-accuracy version alongside the raw one, plus the constant-classifier bound --
the gap a model that memorizes nothing gets for free from the prevalence shift alone.

Uncertainty is a cluster bootstrap over whole within-test near-duplicate components
(the same resampling unit as cluster_bootstrap.py); full-test and novel-only accuracy
are recomputed inside each resample, so the interval is on the paired gap.

Run:  PYTHONPATH=. ./venv/bin/python -m audit.experiments.exp_eval_side_inflation
Out:  results/eval_side_inflation.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
LEAKY = ["human_enhancers_ensembl", "human_nontata_promoters"]
M4 = ["LR", "LinearSVC", "RF", "HGB"]
SIM_K, FEAT_K, NOVEL_CUT, THR = 8, 6, 0.5, 0.7
BOOT, BSEED = 1000, 12345


def balanced_acc(y_true, y_pred):
    """Unweighted mean of per-class recall; a constant predictor scores 0.5 in any
    stratum regardless of prevalence."""
    out = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum():
            out.append(float((y_pred[m] == c).mean()))
    return float(np.mean(out)) if out else np.nan


def gap_cluster_boot(correct, novel_mask, cl, boot=BOOT, seed=BSEED):
    """Cluster bootstrap of (full-test acc - novel-only acc). Resample whole test
    near-duplicate components; recompute BOTH accuracies inside each resample so the
    interval is on the paired difference, not on two marginals."""
    rng = np.random.RandomState(seed)
    correct = np.asarray(correct, float)
    uniq = np.unique(cl)
    n_all = np.array([(cl == u).sum() for u in uniq], float)
    s_all = np.array([correct[cl == u].sum() for u in uniq], float)
    n_nov = np.array([((cl == u) & novel_mask).sum() for u in uniq], float)
    s_nov = np.array([correct[(cl == u) & novel_mask].sum() for u in uniq], float)
    G = len(uniq)
    out = np.empty(boot)
    for b in range(boot):
        p = rng.randint(0, G, size=G)
        dn = n_nov[p].sum()
        out[b] = (s_all[p].sum() / n_all[p].sum()) - (s_nov[p].sum() / dn if dn else np.nan)
    return out[~np.isnan(out)]


def corrected_acc(dset, model):
    """Frozen corrected-split accuracy: mean over the three homology@0.7 split seeds in
    extended_models_long.csv at k=6. Read, not recomputed, so this column stays
    identical to the number the rest of the paper quotes."""
    ext = pd.read_csv(os.path.join(R, "extended_models_long.csv"))
    q = ext[(ext.dataset == dset) & (ext.k == FEAT_K) &
            (ext.split_type == "homology@0.7") & (ext.model == model)]
    return float(q["accuracy"].mean()), int(len(q))


def main():
    os.makedirs(R, exist_ok=True)
    rows = []
    for d in LEAKY:
        t0 = time.time()
        seqs, y, tr, te, _ = E.load(d, full=True)
        y = np.asarray(y)
        X = E.featurize(seqs, FEAT_K)
        sim = E.cached_max_sim(d, seqs, tr, te, k=SIM_K, mode="jaccard")
        comp = E.cached_clusters(d, seqs, threshold=THR, k=SIM_K, mode="jaccard")
        cl = np.asarray(comp)[te]
        novel = sim < NOVEL_CUT
        yt = y[te]
        pos = yt.max()
        p_all = float((yt == pos).mean())
        p_nov = float((yt[novel] == pos).mean())
        # A constant "always predict the majority class of the full test set" model
        # memorizes nothing, yet moves between the two evaluations purely on prevalence.
        const_full = max(p_all, 1 - p_all)
        const_nov = p_nov if p_all >= 0.5 else 1 - p_nov
        print(f"[{d}] n_test={len(te)} novel n={int(novel.sum())} "
              f"({novel.mean():.3f}) | pos_rate full={p_all:.4f} novel={p_nov:.4f} | "
              f"constant-classifier eval-side gap = {const_full - const_nov:+.4f}",
              flush=True)

        # the exact four-model set the frozen four-model comparison uses
        from audit.pipeline.run_extended_models import make_models
        for m in M4:
            mdl = make_models()[m]
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            correct = (pred == yt).astype(float)
            a_all = float(correct.mean())
            a_nov = float(correct[novel].mean())
            b_all = balanced_acc(yt, pred)
            b_nov = balanced_acc(yt[novel], pred[novel])
            draws = gap_cluster_boot(correct, novel, cl)
            lo, hi = E.ci(draws)
            a_cor, n_cor = corrected_acc(d, m)
            rows.append(dict(
                dataset=d, model=m, n_test=len(te), n_novel=int(novel.sum()),
                novel_frac=round(float(novel.mean()), 4),
                pos_rate_full=round(p_all, 4), pos_rate_novel=round(p_nov, 4),
                acc_asshipped_full=round(a_all, 4),
                acc_asshipped_novelonly=round(a_nov, 4),
                acc_corrected_resplit=round(a_cor, 4), n_corrected_seeds=n_cor,
                eval_side_inflation=round(a_all - a_nov, 4),
                eval_side_ci_lo=round(lo, 4), eval_side_ci_hi=round(hi, 4),
                split_side_effect=round(a_nov - a_cor, 4),
                total_drop=round(a_all - a_cor, 4),
                eval_side_share=round((a_all - a_nov) / (a_all - a_cor), 4)
                if abs(a_all - a_cor) > 1e-9 else None,
                bal_asshipped_full=round(b_all, 4),
                bal_asshipped_novelonly=round(b_nov, 4),
                eval_side_inflation_balanced=round(b_all - b_nov, 4),
                constant_classifier_eval_side_gap=round(const_full - const_nov, 4),
            ))
            r = rows[-1]
            print(f"   {m:9s} full={a_all:.4f} novel={a_nov:.4f} corrected={a_cor:.4f}"
                  f" | eval-side {r['eval_side_inflation']:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"  split-side {r['split_side_effect']:+.4f}"
                  f" | balanced eval-side {r['eval_side_inflation_balanced']:+.4f}",
                  flush=True)
        print(f"  [{d} done {time.time()-t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "eval_side_inflation.csv"), index=False)

    # rankings on each of the three arms, so the "does the ranking still change?"
    # question is answered from this file rather than by eye
    print("\nrankings (best first):")
    for d in LEAKY:
        s = out[out.dataset == d]
        for col, lab in (("acc_asshipped_full", "as-shipped, full test"),
                         ("acc_asshipped_novelonly", "as-shipped fit, novel-only"),
                         ("acc_corrected_resplit", "corrected re-split")):
            o = s.sort_values(col, ascending=False)
            print(f"  {d:26s} {lab:28s} " +
                  " > ".join(f"{r.model}({getattr(r, col):.4f})" for r in o.itertuples()))
    print("\nwrote results/eval_side_inflation.csv")


if __name__ == "__main__":
    main()
