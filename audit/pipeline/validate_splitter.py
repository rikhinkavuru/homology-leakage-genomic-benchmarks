#!/usr/bin/env python3
"""
PART C validation: exercise the packaged tool `homology_split.homology_aware_split`
on the two leaky datasets at FULL scale, over 5 seeds. For each seed: split with the
tool, train the frozen best model (RandomForest, k=6), measure corrected test accuracy,
and verify (via the tool's own `verify_split`) that residual cross-split leakage at
Jaccard > 0.7 is exactly 0. Reports corrected accuracy mean +/- std over 5 seeds, i.e.
the tool yields honest, reproducible evaluation.

Uses the packaged tool through its public API. Featurization/model reuse run_audit so
the numbers are comparable to the frozen results.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from audit.core import homology_split as H
from audit.pipeline import run_audit as RA
from audit.pipeline import run_suite as S

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
SEEDS = [0, 1, 2, 3, 4]

rows = []
for d in LEAKY:
    t0 = time.time()
    df, info = S.discover_and_subsample(d, cap=10**12, seed=0)        # full scale
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    test_frac = float((df["split"] == "test").mean())
    X6 = RA.featurize(seqs, 6)
    accs, resids, tfracs = [], [], []
    for s in SEEDS:
        tr, te = H.homology_aware_split(seqs, y, test_frac=test_frac, threshold=0.7, seed=s)
        v = H.verify_split(seqs, tr, te, threshold=0.7)
        rf = RA.make_models()["RF"]
        rf.fit(X6[tr], y[tr])
        acc = accuracy_score(y[te], rf.predict(X6[te]))
        accs.append(acc); resids.append(v["residual_leak_fraction"]); tfracs.append(v["test_fraction"])
        print(f"  {d} seed={s}: acc={acc:.4f} resid_leak>0.7={v['residual_leak_fraction']:.4f} "
              f"test_frac={v['test_fraction']:.3f}", flush=True)
    rows.append(dict(dataset=d, model="RF_k6", n_seeds=len(SEEDS),
                     corrected_acc_mean=round(float(np.mean(accs)), 4),
                     corrected_acc_std=round(float(np.std(accs)), 4),
                     max_residual_leak=round(float(np.max(resids)), 6),
                     test_frac_mean=round(float(np.mean(tfracs)), 4)))
    print(f"[{d}] RF k6 corrected acc {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
          f"over {len(SEEDS)} seeds; max residual leak={max(resids):.6f} ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(RA.RESULTS_DIR, "splitter_validation.csv"), index=False)
print("\nVALIDATION_DONE -> results/splitter_validation.csv")
print(pd.DataFrame(rows).to_string())
"""NOTE: every residual_leak_fraction must be 0.0 -- that is the tool's guarantee."""
