#!/usr/bin/env python3
"""
R2.a3, the half we had been conceding: the injected-leakage MULTICLASS experiment,
run with the from-scratch CNN instead of only RF and LR.

Reviewer 2 asked for deep models "in both binary and 3-class settings". Our CNN evidence
was binary; the multiclass arm (exp_inject3class.py) used only a random forest and
logistic regression, so the answer to that comment was a classical result plus a
concession. This closes it.

CONSTRUCTION -- IDENTICAL TO THE CLASSICAL ARM
----------------------------------------------
Same 20k stratified subsample, same injection fractions {0, 0.1, 0.2, 0.4}, same RNG
seeds, same ~2% point-mutation rate, same near-duplicate-aware corrected split. Only the
model changes, so the comparison against the RF/LR numbers in results/exp_inject3class.csv
is like-for-like. Anything that differs between the two CSVs is the learner, not the setup.

The CNN uses the same pre-registered reference cell as the binary work (dropout 0.3,
weight decay 1e-4) with a three-way softmax head; ResCNN's n_out parameter defaults to
the binary head, so the committed binary results are untouched by its existence.

WHAT TO EXPECT, AND WHAT WOULD BE INFORMATIVE
---------------------------------------------
The mechanism is IMPOSED here rather than measured -- we inject the leakage ourselves --
so a positive result confirms that a memorization-capable deep model exploits injected
near-duplicates in a 3-class setting, which is what the reviewer asked. It is NOT
independent evidence that the suite is leaky; that evidence is the binary work on shipped
data. The honest reading of a null would be equally publishable: it would say the CNN at
this regularization does not exploit the injected copies the way the forest does.

Writes results/exp_inject3class_deep.csv. Does not touch exp_inject3class.csv.

USAGE
    ./venv/bin/python -m audit.experiments.exp_inject3class_deep
    EXP_I3D_SMOKE=1 ./venv/bin/python -m audit.experiments.exp_inject3class_deep
"""
from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from audit.core import expkit as E
from audit.experiments.exp_deep import (
    ResCNN, onehot, _batches, DEVICE, LR as ADAM_LR, BATCH, PATIENCE, MAX_EPOCHS,
)

SMOKE = os.environ.get("EXP_I3D_SMOKE") == "1"
D = E.THREECLASS
FRACS = [0.0, 0.1, 0.2, 0.4]
REF_DROPOUT, REF_WD = 0.3, 1e-4
SEQ_LEN = 401                      # median length of the three-class set (89-802)
BASES = np.array(list("ACGT"))

OUT = "results/exp_inject3class_deep.csv"
if SMOKE:
    OUT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "exp_inject3class_deep_SMOKE.csv")


def mutate(s, n_mut, rng):
    """Byte-for-byte the classical arm's mutator, so the constructions match."""
    s = list(s)
    L = len(s)
    for _ in range(n_mut):
        p = rng.randint(L)
        s[p] = BASES[rng.randint(4)]
    return "".join(s)


def train_multiclass(X, y, train_idx, test_idx, n_class, dropout, wd, seed=0,
                     max_epochs=MAX_EPOCHS):
    """Three-way analogue of exp_deep.train_and_correct: cross-entropy + argmax rather
    than BCE + threshold. Early stopping on the same fixed internal 10% slice, and the
    same seed discipline -- seed varies init/dropout/batch order, the val slice does not.
    Returns the per-test-example correctness vector."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rs = np.random.RandomState(0)
    perm = rs.permutation(len(train_idx))
    n_val = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[perm[:n_val]]
    fit_idx = train_idx[perm[n_val:]]

    model = ResCNN(dropout, n_out=n_class).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y.astype(np.int64))

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        er = np.random.RandomState(1000 + epoch + 1000 * seed)
        order = fit_idx[er.permutation(len(fit_idx))]
        for bi in _batches(order, BATCH):
            xb, yb = X[bi].to(DEVICE), yt[bi].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for bi in _batches(val_idx, 1024):
                xb, yb = X[bi].to(DEVICE), yt[bi].to(DEVICE)
                vtot += float(lossf(model(xb), yb)) * len(bi)
                vn += len(bi)
            vloss = vtot / max(vn, 1)

        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = np.empty(len(test_idx), dtype=np.int64)
    with torch.no_grad():
        off = 0
        for bi in _batches(test_idx, 1024):
            p = model(X[bi].to(DEVICE)).argmax(dim=1).cpu().numpy()
            preds[off:off + len(bi)] = p
            off += len(bi)
    return (preds == y[test_idx]).astype(float), epoch + 1


def main():
    t0 = time.time()
    max_epochs = 2 if SMOKE else MAX_EPOCHS
    fracs = [0.0, 0.2] if SMOKE else FRACS

    seqs, y, otr, ote, tf = E.load(D, full=False, cap=20000)
    classes = sorted(set(int(v) for v in y))
    print(f"device={DEVICE} [{D}] n={len(seqs)} classes={classes} "
          f"test_frac={tf:.3f}{' [SMOKE]' if SMOKE else ''} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for f in fracs:
        rng = np.random.RandomState(100 + int(f * 100))      # same seed rule as the classical arm
        n_inj = int(round(f * len(otr)))
        inj_src = rng.choice(otr, size=n_inj, replace=False) if n_inj else np.array([], int)
        inj_seqs = [mutate(seqs[i], max(1, int(0.02 * len(seqs[i]))), rng) for i in inj_src]
        all_seqs = seqs + inj_seqs
        all_y = np.concatenate([y, y[inj_src]])

        tr2 = otr
        te2 = np.concatenate([ote, np.arange(len(seqs), len(all_seqs))])

        comp = E.clusters(all_seqs, 0.7, 8)                  # computed once, not per model
        ctr, cte = E.assign(comp, all_y, 0, tf)

        X = torch.from_numpy(onehot(all_seqs, SEQ_LEN))
        n_class = len(classes)
        co, ep_o = train_multiclass(X, all_y, tr2, te2, n_class, REF_DROPOUT, REF_WD,
                                    max_epochs=max_epochs)
        cc, ep_c = train_multiclass(X, all_y, ctr, cte, n_class, REF_DROPOUT, REF_WD,
                                    max_epochs=max_epochs)
        acc_o, acc_c = float(co.mean()), float(cc.mean())
        rows.append(dict(dataset=D, inject_frac=f, model="CNN", n_injected=n_inj,
                         acc_orig=round(acc_o, 4), acc_corr=round(acc_c, 4),
                         drop=round(acc_o - acc_c, 4),
                         epochs_orig=ep_o, epochs_corr=ep_c))
        print(f"  f={f}: CNN drop={acc_o-acc_c:+.4f} "
              f"(orig {acc_o:.4f} -> corr {acc_c:.4f}) n_inj={n_inj} "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\n" + df.to_string(index=False), flush=True)
    print(f"\nEXP_INJECT3CLASS_DEEP_DONE {round(time.time()-t0)}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
