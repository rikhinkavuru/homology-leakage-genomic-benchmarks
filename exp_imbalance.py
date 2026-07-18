#!/usr/bin/env python3
"""
T1.7 / R2.a1: imbalanced-data evaluation + splitter balance behaviour.

Part A  Exact per-split class counts (curated-balance disclosure).
Part B  Prevalence stress test (semi-synthetic, clearly labelled): downsample the
        positive class of each leaky dataset to target prevalence pi in {0.5,0.2,0.1}
        within BOTH train and test, then compare the ORIGINAL split vs the homology-
        aware corrected split under prevalence-aware metrics (AUPRC vs baseline, AUROC,
        MCC, balanced accuracy, minority precision/recall/F1). Tests whether the
        leakage inflation shows up beyond accuracy, and whether the whole-cluster
        majority-vote splitter PRESERVES the target prevalence (G9).
"""
import time
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, matthews_corrcoef,
                             balanced_accuracy_score, precision_recall_fscore_support, accuracy_score)
import expkit as E

t0 = time.time()

# ---------------- Part A: class counts ----------------
A = []
for d in E.LEAKY + E.CLEAN + [E.THREECLASS]:
    seqs, y, otr, ote, tf = E.load(d, full=(d in E.LEAKY or d == E.THREECLASS))
    for split, idx in [("train", otr), ("test", ote)]:
        vals, cnts = np.unique(y[idx], return_counts=True)
        A.append(dict(dataset=d, split=split, counts=dict(zip(vals.tolist(), cnts.tolist())),
                      pos_frac=round(float((y[idx] == 1).mean()), 4) if set(np.unique(y)) <= {0, 1} else None))
pd.DataFrame(A).to_csv("results/exp_imbalance_counts.csv", index=False)
print("Part A counts done", flush=True)

# ---------------- Part B: prevalence stress test ----------------
def imbalance_idx(y, idx, pi, rng):
    """Downsample the positive class within idx to target positive prevalence pi."""
    pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    n_pos_target = int(round(pi / (1 - pi) * len(neg)))
    n_pos_target = min(n_pos_target, len(pos))
    keep_pos = rng.choice(pos, size=n_pos_target, replace=False)
    return np.sort(np.concatenate([neg, keep_pos]))

def panel(model, X, y, tr, te):
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]; pred = (p >= 0.5).astype(int)
    yte = y[te]
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, labels=[1], zero_division=0, average=None)
    return dict(n_test=len(te), pos_frac=round(float((yte == 1).mean()), 4),
                acc=round(accuracy_score(yte, pred), 4),
                bal_acc=round(balanced_accuracy_score(yte, pred), 4),
                mcc=round(matthews_corrcoef(yte, pred), 4) if len(np.unique(yte)) > 1 else np.nan,
                auroc=round(roc_auc_score(yte, p), 4) if len(np.unique(yte)) > 1 else np.nan,
                auprc=round(average_precision_score(yte, p), 4) if len(np.unique(yte)) > 1 else np.nan,
                auprc_baseline=round(float((yte == 1).mean()), 4),
                minor_prec=round(float(pr[0]), 4), minor_rec=round(float(rc[0]), 4), minor_f1=round(float(f1[0]), 4))

B = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    X = E.featurize(seqs, 6)
    rng = np.random.RandomState(0)
    for pi in [0.5, 0.2, 0.1]:
        tr_i = imbalance_idx(y, otr, pi, rng); te_i = imbalance_idx(y, ote, pi, rng)
        keep = np.sort(np.concatenate([tr_i, te_i]))
        # original (imbalanced) split
        po = panel(E.models()["RF"], X, y, tr_i, te_i)
        # homology-aware corrected split on the imbalanced subset
        sub_seqs = [seqs[i] for i in keep]; sub_y = y[keep]
        comp = E.clusters(sub_seqs, 0.7, 8)
        ctr_l, cte_l = E.assign(comp, sub_y, 0, tf)
        ctr, cte = keep[ctr_l], keep[cte_l]
        pc = panel(E.models()["RF"], X, y, ctr, cte)
        for split, pnl in [("original", po), ("corrected", pc)]:
            B.append(dict(dataset=d, target_pi=pi, split=split, **pnl))
        print(f"[{d}] pi={pi}: orig AUPRC={po['auprc']}(base {po['auprc_baseline']}) MCC={po['mcc']} -> "
              f"corr AUPRC={pc['auprc']} MCC={pc['mcc']}; corr realized pos_frac={pc['pos_frac']} "
              f"(splitter-preserved? target {pi}) ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(B).to_csv("results/exp_imbalance_panel.csv", index=False)
print("\nEXP_IMBALANCE_DONE", round(time.time()-t0), "s")
print(pd.DataFrame(B).to_string(index=False))
