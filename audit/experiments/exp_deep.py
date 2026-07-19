#!/usr/bin/env python3
"""
Task T1.2 -- the load-bearing DEEP-MODEL answer to reviewers R1.1a / R2.a3 / R3.1(i).

Pre-registered, falsifiable dose-response experiment (see results/deep_preregistration.md,
written BEFORE this ran). A from-scratch 1D residual CNN (Basset/DeepSEA-style) is trained
on the two LEAKY datasets under a dropout x weight_decay grid, on the ORIGINAL split and on
the homology-CORRECTED split, to test whether near-duplicate leakage inflates high-capacity,
insufficiently-regularized deep nets and demotes them under a homology-aware split.

NO HG38-pretrained model is used (DNABERT-2 / HyenaDNA are pretrained on the exact test
regions -> pretraining-contamination confound; explicitly out of scope). This net sees only
each dataset's training split.

Reuses expkit for splits / similarity / bootstrap. Deterministic (torch.manual_seed(0)).
Does NOT modify any frozen script or existing result file; only writes:
  results/exp_deep_cnn.csv   (one row per dataset x config, plus labelled manip-check rows)
Interpretation is written separately to results/exp_deep_cnn.md.

Env flags:
  EXP_DEEP_SMOKE=1  -> tiny crash-test (subsample, 2 epochs, 1 config) -- writes nothing.
  EXP_DEEP_GRID=reduced -> 6-cell grid (drop the wd=1e-3 column) + manip check.
"""
from __future__ import annotations
import os, sys, time, copy, json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from audit.core import expkit as E

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
SMOKE = os.environ.get("EXP_DEEP_SMOKE") == "1"
GRID_MODE = os.environ.get("EXP_DEEP_GRID", "full")
# See exp_alignment.py: scratch dir is configurable and created lazily, not at import.
SCRATCH = os.path.join(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir()),
                       "homology_audit")
PREP_DIR = os.path.join(SCRATCH, "deep_prep")

FIXED_LEN = {"human_nontata_promoters": 251, "human_enhancers_ensembl": 269}
BOOT = 1000
BSEED = E.BSEED           # 20240524
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 256
LR = 1e-3

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

_TRANS = np.full(256, -1, dtype=np.int64)
for _b, _c in zip(b"ACGT", range(4)):
    _TRANS[_b] = _c


# ----------------------------------------------------------------------------
# one-hot encoding (A,C,G,T -> 4 channels; N/other -> all-zero column; fixed length)
# ----------------------------------------------------------------------------
def onehot(seqs, L):
    """(n, 4, L) float32. Non-ACGT -> all-zero column. Left-align crop/zero-pad to L."""
    n = len(seqs)
    X = np.zeros((n, 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        v = _TRANS[np.frombuffer(s.upper().encode("ascii", "ignore"), dtype=np.uint8)]
        m = min(len(v), L)
        if m == 0:
            continue
        vv = v[:m]
        pos = np.arange(m)
        ok = vv >= 0
        X[i, vv[ok], pos[ok]] = 1.0
    return X


# ----------------------------------------------------------------------------
# 1D residual CNN (Basset/DeepSEA-style)
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU with a residual add (1x1 projection when the
    channel count changes, i.e. residual 'where shapes allow')."""
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU()
        self.proj = None if cin == cout else nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        h = self.relu(self.bn(self.conv(x)))
        res = x if self.proj is None else self.proj(x)
        return h + res


class ResCNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.b1 = ResBlock(4, 64, 9)
        self.b2 = ResBlock(64, 128, 5)
        self.b3 = ResBlock(128, 128, 3)
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = x.mean(dim=2)                 # global average pool over length
        return self.head(x)              # (B, 1) logit


# ----------------------------------------------------------------------------
# training with early stopping on a fixed 10%-of-train internal validation slice
# ----------------------------------------------------------------------------
def _batches(idx, bs):
    for b in range(0, len(idx), bs):
        yield idx[b:b + bs]


def train_and_correct(X, y, train_idx, test_idx, dropout, wd, converge,
                      max_epochs=MAX_EPOCHS, seed=0):
    """Train a ResCNN on X[train_idx] (early stopping on a fixed internal 10% val slice,
    unless converge=True) and return the per-test-example correctness vector on test_idx
    (pred>0.5 == y). Deterministic: torch.manual_seed(seed) so weight init is identical
    across configs -- only dropout/wd/split/convergence vary.

    `seed` varies TRAINING stochasticity only: weight init, dropout masks and batch
    order. The internal validation slice stays at RandomState(0) for every seed, because
    that slice is part of the split definition rather than of training -- varying it too
    would confound seed replication with re-partitioning. seed=0 reproduces the original
    single-seed grid exactly, so results/exp_deep_cnn.csv is unaffected by this parameter
    existing; exp_deep_seeds.py is the only caller that passes anything else."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rs = np.random.RandomState(0)
    perm = rs.permutation(len(train_idx))
    n_val = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[perm[:n_val]]
    fit_idx = train_idx[perm[n_val:]]

    model = ResCNN(dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()

    best_val, best_state, bad = float("inf"), None, 0
    yt = torch.from_numpy(y.astype(np.float32))

    for epoch in range(max_epochs):
        model.train()
        er = np.random.RandomState(1000 + epoch + 1000 * seed)
        order = fit_idx[er.permutation(len(fit_idx))]
        for bi in _batches(order, BATCH):
            xb = X[bi].to(DEVICE)
            yb = yt[bi].to(DEVICE)
            opt.zero_grad()
            out = model(xb).squeeze(1)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()

        # validation loss
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for bi in _batches(val_idx, 1024):
                xb = X[bi].to(DEVICE)
                yb = yt[bi].to(DEVICE)
                out = model(xb).squeeze(1)
                vtot += float(lossf(out, yb)) * len(bi)
                vn += len(bi)
            vloss = vtot / max(vn, 1)

        if converge:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test correctness
    model.eval()
    preds = np.empty(len(test_idx), dtype=np.float32)
    with torch.no_grad():
        off = 0
        for bi in _batches(test_idx, 1024):
            xb = X[bi].to(DEVICE)
            prob = torch.sigmoid(model(xb).squeeze(1)).cpu().numpy()
            preds[off:off + len(bi)] = prob
            off += len(bi)
    yte = y[test_idx]
    correct = ((preds > 0.5).astype(int) == yte).astype(float)
    n_epochs_ran = epoch + 1
    return correct, float(correct.mean()), n_epochs_ran


# ----------------------------------------------------------------------------
# per-dataset prep (expensive; cached to scratchpad): splits, sim, within-test clusters
# ----------------------------------------------------------------------------
def prep_dataset(d, t0):
    os.makedirs(PREP_DIR, exist_ok=True)
    cache = os.path.join(PREP_DIR, f"{d}__prep.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[{d}] prep loaded from cache ({time.time()-t0:.0f}s)", flush=True)
        return {k: z[k] for k in z.files}

    seqs, y, otr, ote, tf = E.load(d)
    print(f"[{d}] loaded n={len(y)} test_frac={tf:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # homology-corrected split (frozen construction)
    comp = E.clusters(seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, 0, tf)
    print(f"[{d}] corrected split built: n_train={len(ctr)} n_test={len(cte)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # test->train max 8-mer Jaccard on the ORIGINAL split (for the graded gap)
    sim = E.max_sim_to_train(seqs, otr, ote, 8)
    print(f"[{d}] max_sim_to_train done ({time.time()-t0:.0f}s)", flush=True)

    # within-test near-duplicate clusters (for the cluster/block bootstrap)
    ocl = E.clusters([seqs[i] for i in ote], 0.7, 8)
    ccl = E.clusters([seqs[i] for i in cte], 0.7, 8)
    print(f"[{d}] within-test clusters: orig {len(np.unique(ocl))}/{len(ote)}  "
          f"corr {len(np.unique(ccl))}/{len(cte)} ({time.time()-t0:.0f}s)", flush=True)

    L = FIXED_LEN[d]
    X = onehot(seqs, L)
    print(f"[{d}] one-hot {X.shape} ({time.time()-t0:.0f}s)", flush=True)

    out = dict(y=y.astype(np.int64), otr=otr, ote=ote, ctr=ctr, cte=cte,
               sim=sim.astype(np.float32), ocl=ocl.astype(np.int64),
               ccl=ccl.astype(np.int64), X=X, tf=np.float32(tf))
    np.savez(cache, **out)
    print(f"[{d}] prep cached -> {cache} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------
# metrics / bootstrap
# ----------------------------------------------------------------------------
def graded_gap(correct, sim):
    hi = correct[sim >= 0.9]
    lo = correct[sim < 0.5]
    g = (hi.mean() - lo.mean()) if (len(hi) and len(lo)) else np.nan
    return (float(g), float(hi.mean()) if len(hi) else np.nan,
            float(lo.mean()) if len(lo) else np.nan, int(len(hi)), int(len(lo)))


def drop_ci(co, cc, ocl, ccl):
    """95% CI for drop = acc_orig - acc_corr, via sample-wise and cluster bootstrap.
    Orig and corrected test sets are disjoint -> resampled independently (different seeds)."""
    so = E.sample_boot(co, BOOT, seed=BSEED)
    sc = E.sample_boot(cc, BOOT, seed=BSEED + 1)
    slo, shi = E.ci(so - sc)
    ko = E.cluster_boot(co, ocl, BOOT, seed=BSEED)
    kc = E.cluster_boot(cc, ccl, BOOT, seed=BSEED + 1)
    klo, khi = E.ci(ko - kc)
    return (slo, shi, bool(slo > 0 or shi < 0)), (klo, khi, bool(klo > 0 or khi < 0))


# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------
def build_grid():
    dropouts = [0.0, 0.3, 0.6]
    wds = [0.0, 1e-4, 1e-3] if GRID_MODE != "reduced" else [0.0, 1e-4]
    grid = []
    for do in dropouts:
        for wd in wds:
            grid.append(dict(dropout=do, wd=wd, converge=False,
                             variant="grid",
                             is_reference=(do == 0.3 and wd == 1e-4),
                             is_manip_check=False))
    # labelled manipulation check: unregularized, trained to convergence (no early stop)
    grid.append(dict(dropout=0.0, wd=0.0, converge=True, variant="manip_check",
                     is_reference=False, is_manip_check=True))
    return grid


def run_smoke():
    print("SMOKE TEST (writes nothing)", flush=True)
    t0 = time.time()
    seqs, y, otr, ote, tf = E.load("human_nontata_promoters")
    rs = np.random.RandomState(0)
    keep = rs.choice(len(y), 3000, replace=False)
    seqs = [seqs[i] for i in keep]; y = y[keep]
    L = FIXED_LEN["human_nontata_promoters"]
    X = torch.from_numpy(onehot(seqs, L))
    n = len(y); tr = np.arange(int(0.8 * n)); te = np.arange(int(0.8 * n), n)
    corr, acc, ep = train_and_correct(X, y, tr, te, 0.3, 1e-4, False, max_epochs=2)
    print(f"smoke ok: acc={acc:.4f} epochs={ep} device={DEVICE} ({time.time()-t0:.0f}s)", flush=True)


def main():
    if SMOKE:
        run_smoke()
        return

    t0 = time.time()
    grid = build_grid()
    print(f"device={DEVICE} grid={len(grid)} configs x {len(E.LEAKY)} datasets x 2 splits "
          f"= {len(grid)*len(E.LEAKY)*2} trainings (GRID_MODE={GRID_MODE})", flush=True)

    rows = []
    for d in E.LEAKY:
        P = prep_dataset(d, t0)
        X = torch.from_numpy(P["X"])                 # (n,4,L) CPU tensor
        y = P["y"]; sim = P["sim"]
        otr, ote, ctr, cte = P["otr"], P["ote"], P["ctr"], P["cte"]
        ocl, ccl = P["ocl"], P["ccl"]

        for cfg in grid:
            tc = time.time()
            do, wd, conv = cfg["dropout"], cfg["wd"], cfg["converge"]
            co, acc_o, ep_o = train_and_correct(X, y, otr, ote, do, wd, conv)
            cc, acc_c, ep_c = train_and_correct(X, y, ctr, cte, do, wd, conv)
            drop = acc_o - acc_c
            gap, acc_hi, acc_lo, n_hi, n_lo = graded_gap(co, sim)
            (slo, shi, sx), (klo, khi, kx) = drop_ci(co, cc, ocl, ccl)
            row = dict(
                dataset=d, dropout=do, weight_decay=wd, variant=cfg["variant"],
                is_reference=cfg["is_reference"], is_manip_check=cfg["is_manip_check"],
                acc_orig=round(acc_o, 4), acc_corr=round(acc_c, 4), drop=round(drop, 4),
                drop_ci_sample_lo=round(slo, 4), drop_ci_sample_hi=round(shi, 4),
                drop_excl0_sample=sx,
                drop_ci_cluster_lo=round(klo, 4), drop_ci_cluster_hi=round(khi, 4),
                drop_excl0_cluster=kx,
                graded_gap=round(gap, 4), acc_hi_sim=round(acc_hi, 4),
                novel_acc=round(acc_lo, 4), n_hi=n_hi, n_lo=n_lo,
                epochs_orig=ep_o, epochs_corr=ep_c)
            rows.append(row)
            tag = " [REF]" if cfg["is_reference"] else (" [MANIP]" if cfg["is_manip_check"] else "")
            print(f"  [{d}] do={do} wd={wd} conv={conv}{tag}: "
                  f"acc_o={acc_o:.4f} acc_c={acc_c:.4f} drop={drop:+.4f} "
                  f"clCI=[{klo:+.4f},{khi:+.4f}] excl0={kx} gap={gap:+.4f} "
                  f"novel={acc_lo:.4f} ({time.time()-tc:.0f}s, tot {time.time()-t0:.0f}s)",
                  flush=True)

        # checkpoint after each dataset
        pd.DataFrame(rows).to_csv("results/exp_deep_cnn.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv("results/exp_deep_cnn.csv", index=False)
    print("\nEXP_DEEP_DONE", round(time.time() - t0), "s -> results/exp_deep_cnn.csv", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
