#!/usr/bin/env python3
"""
exp_published_baseline.py -- C8: re-evaluate the suite's OWN published baseline on the
corrected split.

WHY THIS AND NOT THE EXISTING CNN
---------------------------------
The manuscript already reports a from-scratch residual CNN in the Basset/DeepSEA lineage,
and shows it ranking fifth of ten as shipped and first corrected. That answers "does the
effect reach a trained neural network", but it does not answer the question a benchmark
maintainer would ask, which is what happens to *their* number. Genomic Benchmarks ships a
baseline CNN with the package, that CNN is the model its leaderboard reports, and until
now nothing in this audit had imported it: `genomic_benchmarks.models` appears nowhere
under audit/. The leaderboard claim was therefore being made by proxy.

This runs the published class itself --

    from genomic_benchmarks.models.torch import CNN

-- unmodified, with the package's own optimizer (Adam at library defaults) and its own
tokenization convention, on both arms of the same partitions every other experiment uses.
Nothing about the architecture is retuned; the point is precisely that it is theirs.

WHAT IS AND IS NOT MATCHED TO THE PUBLISHED NUMBER
--------------------------------------------------
The as-shipped arm is a reproduction of the published setup, not a re-run of the published
artifact: the paper reports its own preprocessing and epoch budget, and we fix the epoch
budget here and hold it identical across arms. That is the property the comparison needs.
An absolute disagreement with the published leaderboard value is therefore not evidence of
anything, and we report our own as-shipped number beside the corrected one rather than
against theirs.

Both arms share: the same tokenizer, the same vocabulary, the same padded length, the same
epoch budget, the same optimizer, the same seeds. They differ only in the split.

Out: results/exp_published_cnn.csv       one row per (dataset, seed) plus a mean row
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import numpy as np
import pandas as pd

from audit.core import expkit as E

SEEDS = [0, 1, 2]
EPOCHS = 12
BATCH = 128
EMBED = 100
DATASETS = ["human_enhancers_ensembl"]
# 20k subsample. Full scale is ~3 GPU-free hours per seed on this CPU, and a subsample
# is admissible here for the same reason it is elsewhere in this paper: subsampling can
# only HIDE leakage, so it makes the demotion CONSERVATIVE. The leak in the as-shipped
# split survives subsampling because near-duplicate partners are drawn together.
SUBSAMPLE = 20000
# The package tokenizes at character level over the observed alphabet, padding to a fixed
# length. We rebuild that here rather than importing the loader, because our splits are
# index-based over the cached frame and the package's loader returns its own ordering.
PAD, UNK = 0, 1


def encode(seqs, max_len, vocab=None):
    if vocab is None:
        alphabet = sorted({c for s in seqs for c in s})
        vocab = {c: i + 2 for i, c in enumerate(alphabet)}      # 0 pad, 1 unk
    out = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, :len(s)] = [vocab.get(c, UNK) for c in s]
    return out, vocab


def run_arm(X, y, tr, te, seed, n_classes, vocab_size, max_len, device):
    """Train the published CNN architecture and return its test-set predictions.

    A DEFECT IN THE PUBLISHED CLASS, and how it is handled. For a binary task
    `genomic_benchmarks.models.torch.CNN` sets `output_activation = nn.Sigmoid()`, so
    `forward` returns a probability, and then `train_loop` computes
    `binary_cross_entropy_with_logits(pred, y)` on it -- applying the sigmoid a SECOND
    time inside the loss. The effective probability is sigmoid(sigmoid(logit)), the
    gradient is vanishing, and the model does not train: on our data the loss sticks at
    ln 2 and every logit collapses to ~0, giving exactly chance accuracy. We verified
    this is the shipped behaviour, not our harness. To run the published ARCHITECTURE we
    therefore replace the output activation with the identity so `forward` returns raw
    logits, which is what `binary_cross_entropy_with_logits` expects; the convolutional
    stack, dimensions, optimizer (Adam at library defaults) and tokenization are
    otherwise the package's own. This is the smallest change that makes the published
    model trainable, and it is disclosed rather than silent.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from genomic_benchmarks.models.torch import CNN

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN(number_of_classes=n_classes, vocab_size=vocab_size,
                embedding_dim=EMBED, input_len=max_len, device=device).to(device)
    model.output_activation = nn.Identity()          # see docstring: undo double-sigmoid
    ds = TensorDataset(torch.from_numpy(X[tr]),
                       torch.from_numpy(y[tr].astype(np.float32)).unsqueeze(1))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters())       # package's own optimizer choice
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = model.loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(te), 512):
            xb = torch.from_numpy(X[te[s:s + 512]]).to(device)
            preds.append((model(xb).cpu().numpy().ravel() > 0).astype(int))  # logit>0
    return np.concatenate(preds)


def main():
    import torch
    device = "cpu"          # deterministic and sufficient at this size; MPS is not
    rows = []
    t0 = time.time()
    for d in DATASETS:
        seqs, y, otr, ote, tf = E.load(d, full=False, cap=SUBSAMPLE)
        y = np.asarray(y)
        max_len = int(np.percentile([len(s) for s in seqs], 99))
        X, vocab = encode(seqs, max_len)
        comp = E.cached_clusters(d, seqs, 0.7, 8)
        if len(comp) != len(seqs):        # cache is full-scale; recompute on the subsample
            comp = E.clusters(seqs, 0.7, 8)
        n_classes = int(len(np.unique(y)))
        print(f"[{d}] n={len(seqs)} max_len={max_len} vocab={len(vocab)+2} "
              f"classes={n_classes}", flush=True)

        for seed in SEEDS:
            ctr, cte = E.assign(comp, y, seed, tf)
            po = run_arm(X, y, otr, ote, seed, n_classes, len(vocab) + 2, max_len, device)
            acc_o = float((po == y[ote]).mean())
            pc = run_arm(X, y, ctr, cte, seed, n_classes, len(vocab) + 2, max_len, device)
            acc_c = float((pc == y[cte]).mean())

            corr_o = (po == y[ote]).astype(float)
            corr_c = (pc == y[cte]).astype(float)
            ocl = E.clusters([seqs[i] for i in ote], 0.7, 8)
            ccl = E.clusters([seqs[i] for i in cte], 0.7, 8)
            db = E.cluster_boot(corr_o, ocl, 1000) - E.cluster_boot(corr_c, ccl, 1000)
            lo, hi = E.ci(db)
            rows.append(dict(dataset=d, seed=seed, acc_orig=round(acc_o, 4),
                             acc_corr=round(acc_c, 4), drop=round(acc_o - acc_c, 4),
                             drop_ci_cluster_lo=round(lo, 4),
                             drop_ci_cluster_hi=round(hi, 4),
                             drop_excl0_cluster=bool(lo > 0 or hi < 0),
                             epochs=EPOCHS, n_test_orig=len(ote), n_test_corr=len(cte)))
            print(f"  [{d} seed {seed}] orig={acc_o:.4f} corr={acc_c:.4f} "
                  f"drop={acc_o-acc_c:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"{time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(E.RESULTS, "exp_published_cnn.csv"), index=False)

    df = pd.DataFrame(rows)
    for d, g in df.groupby("dataset"):
        rows.append(dict(dataset=d, seed="mean",
                         acc_orig=round(g.acc_orig.mean(), 4),
                         acc_corr=round(g.acc_corr.mean(), 4),
                         drop=round(g.drop.mean(), 4),
                         drop_ci_cluster_lo=np.nan, drop_ci_cluster_hi=np.nan,
                         drop_excl0_cluster=bool(g.drop_excl0_cluster.all()),
                         epochs=EPOCHS, n_test_orig=np.nan, n_test_corr=np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(E.RESULTS, "exp_published_cnn.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("EXP_PUBLISHED_BASELINE_DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
