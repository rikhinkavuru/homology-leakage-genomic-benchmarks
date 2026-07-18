#!/usr/bin/env python3
"""Complete STEP 2: compute RF k6 homology-aware corrected accuracy over the 5
cluster->side seeds for both leaky datasets (reproduces splitter_validation.csv,
but adds per-seed + min/max to a committed CSV) and merge with the LR k6 rows in
step2_seed_variance.csv. Same split method as splitter_validation (k=8 Jaccard>0.7
components, whole-cluster per-class assignment). Full scale, deterministic."""
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from audit.pipeline import run_audit as RA, run_suite as S
from audit.core import homology_split as H
from audit.pipeline.run_extended_models import make_models

LEAKY = ["human_nontata_promoters", "human_enhancers_ensembl"]
SEEDS = [0, 1, 2, 3, 4]
R = RA.RESULTS_DIR


def comps(seqs):
    M = H.kmer_binary_matrix(seqs, 8)
    er, ec = H._edges(M, 0.7)
    g = sparse.coo_matrix((np.ones(len(er)), (er, ec)), shape=(M.shape[0], M.shape[0]))
    _, c = connected_components(g + g.T, directed=False)
    return c


rows = []
for d in LEAKY:
    df, info = S.discover_and_subsample(d, cap=10**12, seed=S.SUBSAMPLE_SEED)
    y = df["label"].to_numpy(); seqs = df["seq"].tolist()
    X = RA.featurize(seqs, 6); comp = comps(seqs)
    tf = float((df["split"] == "test").mean())
    accs = []
    for s in SEEDS:
        tr, te = H._assign(comp, y, s, tf)
        m = make_models()["RF"]; m.fit(X[tr], y[tr])
        accs.append(float((m.predict(X[te]) == y[te]).mean()))
    a = np.array(accs)
    rows.append(dict(dataset=d, model="RF_k6", n_seeds=5,
                     per_seed=str([round(x, 4) for x in accs]), mean=round(a.mean(), 4),
                     sd=round(a.std(ddof=0), 4), min=round(a.min(), 4),
                     max=round(a.max(), 4), range=round(a.max() - a.min(), 4)))
    print(f"{d} RF k6 5-seed mean={a.mean():.4f} sd={a.std(ddof=0):.4f} "
          f"min={a.min():.4f} max={a.max():.4f}", flush=True)

old = pd.read_csv(f"{R}/step2_seed_variance.csv")        # existing LR k6 rows
old = old[old["model"] != "RF_k6"]                        # idempotent
combined = pd.concat([pd.DataFrame(rows), old], ignore_index=True)
combined = combined.sort_values(["dataset", "model"]).reset_index(drop=True)
combined.to_csv(f"{R}/step2_seed_variance.csv", index=False)
print("STEP2RF_DONE", flush=True)
print(combined.to_string(index=False), flush=True)
