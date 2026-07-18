#!/usr/bin/env python3
"""
Deepener #1 phase A -- cross-suite near-duplicate leak census (CPU only, no training).

Ports the paper's leak-census step, UNCHANGED, onto benchmark suites built independently
of Genomic Benchmarks, to test whether near-duplicate contamination is a property of one
suite or of the field's construction habits.

THE NATURAL EXPERIMENT
----------------------
The Nucleotide Transformer downstream tasks exist in two versions by the same authors:
  ORIGINAL  InstaDeepAI/nucleotide_transformer_downstream_tasks
  REVISED   InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
whose dataset card states the revision "replaced randomly generated negative samples by
existing chunks of genomes" and gave "all datasets proper chromosome held-out test sets".
The two ship the SAME TASK NAMES, so the pair isolates CURATION from BIOLOGY: any change
in leak fraction between them is attributable to how the split was built, not to what is
being predicted. `promoter_no_tata` is the direct analogue of the paper's
`human_nontata_promoters`, so this is a near-replication of the paper's own anchor on an
independently constructed suite.

ESTIMATOR -- identical to the paper's: for each TEST sequence, the maximum 8-mer Jaccard
(length-blind) and containment (length-robust) similarity to ANY TRAIN sequence, on the
AS-SHIPPED split; leak fraction = share of test above threshold. Plus an exact-duplicate
census. No model is trained here; this is the census that PRECEDES any ranking claim.

DIRECTION OF ERROR (important, and in our favour)
-------------------------------------------------
Where a task exceeds --cap we stratify-subsample it. Subsampling can only REMOVE
potential near-duplicate partners from train, so a censused leak fraction is a LOWER
BOUND on the full-scale value. This repo already demonstrated the direction:
demo_coding_vs_intergenomic_seqs censuses clean at 20k but borderline at full scale
(results/full_scale_containment.csv). So "LEAKY at the cap" is safe; "clean at the cap"
is provisional and is reported as such.

Run:  python -m audit.experiments.exp_crosssuite_census [--cap 20000] [--tasks ...]
Out:  results/crosssuite_census.csv
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "datacache", "crosssuite")

SUITES = {
    "NT-original": "InstaDeepAI/nucleotide_transformer_downstream_tasks",
    "NT-revised": "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
}
# Tasks present in BOTH versions -> the like-for-like curation contrast.
SHARED_TASKS = ["promoter_no_tata", "promoter_all", "promoter_tata", "enhancers",
                "enhancers_types", "splice_sites_all", "splice_sites_acceptors",
                "splice_sites_donors", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac",
                "H3K36me3"]

# Pre-registered predictions (see results/tier1_preregistration.md). Recorded here so the
# code itself carries the prediction that the census then tests.
PREREGISTERED = {
    "NT-original": "LEAKY",     # random-draw negatives + non-chromosome-aware split
    "NT-revised": "CLEAN",      # genomic-chunk negatives + chromosome held-out test
}
LEAK_CUT = 0.1


def read_fna(path):
    """NT .fna headers are `>NAME|task|LABEL`; returns (seqs, labels)."""
    seqs, labels, cur, lab = [], [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line[0] == ">":
                if cur:
                    seqs.append("".join(cur)); labels.append(lab); cur = []
                parts = line[1:].split("|")
                try:
                    lab = int(parts[-1])
                except ValueError:
                    lab = parts[-1]
            else:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur)); labels.append(lab)
    return seqs, np.array(labels)


def fetch(repo, task, split):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f"{task}/{split}.fna", repo_type="dataset",
                           cache_dir=CACHE)


def census_one(suite, repo, task, cap, seed=0):
    t0 = time.time()
    try:
        tr_s, tr_y = read_fna(fetch(repo, task, "train"))
        te_s, te_y = read_fna(fetch(repo, task, "test"))
    except Exception as ex:                                     # noqa: BLE001
        print(f"  [{suite}/{task}] unavailable: {type(ex).__name__}", flush=True)
        return None

    n_tr_full, n_te_full = len(tr_s), len(te_s)
    rng = np.random.RandomState(seed)
    subsampled = False
    # Cap TRAIN only: capping test would change the estimand; capping train makes the
    # result a lower bound (fewer potential near-duplicate partners).
    if n_tr_full > cap:
        keep = rng.choice(n_tr_full, cap, replace=False)
        tr_s = [tr_s[i] for i in keep]; tr_y = tr_y[keep]; subsampled = True
    if len(te_s) > cap:
        keep = rng.choice(len(te_s), cap, replace=False)
        te_s = [te_s[i] for i in keep]; te_y = te_y[keep]; subsampled = True

    seqs = tr_s + te_s
    tr = np.arange(len(tr_s)); te = np.arange(len(tr_s), len(seqs))
    sim_j = E.max_sim_to_train(seqs, tr, te, k=8, mode="jaccard")
    sim_c = E.max_sim_to_train(seqs, tr, te, k=8, mode="containment")
    trset = set(tr_s)
    exact = float(np.mean([s in trset for s in te_s]))
    L = np.array([len(s) for s in seqs])
    yb = np.asarray(te_y)
    bal = (float(np.mean(yb == yb.max())) if yb.dtype.kind in "iu"
           else float(pd.Series(yb).value_counts(normalize=True).max()))

    row = dict(
        suite=suite, task=task, preregistered=PREREGISTERED.get(suite),
        n_train_full=n_tr_full, n_test_full=n_te_full,
        n_train_used=len(tr_s), n_test_used=len(te_s), subsampled=subsampled,
        len_min=int(L.min()), len_med=int(np.median(L)), len_max=int(L.max()),
        class_balance_majority=round(bal, 4),
        exact_dup_test_in_train=round(exact, 4),
        leak_jaccard_0p7=round(float((sim_j > 0.7).mean()), 4),
        leak_jaccard_0p9=round(float((sim_j >= 0.9).mean()), 4),
        leak_containment_0p7=round(float((sim_c > 0.7).mean()), 4),
        leak_containment_0p9=round(float((sim_c >= 0.9).mean()), 4),
        seconds=round(time.time() - t0, 1))
    row["verdict"] = ("LEAKY" if row["leak_jaccard_0p7"] > LEAK_CUT
                      else "borderline" if row["leak_containment_0p7"] > LEAK_CUT
                      else "clean")
    row["verdict_is_lower_bound"] = subsampled
    print(f"  [{suite}/{task}] n={n_tr_full}/{n_te_full} "
          f"jac@0.7={row['leak_jaccard_0p7']:.4f} con@0.7={row['leak_containment_0p7']:.4f} "
          f"exact={exact:.4f} -> {row['verdict']} ({row['seconds']}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--tasks", nargs="*", default=SHARED_TASKS)
    ap.add_argument("--out", default=os.path.join(R, "crosssuite_census.csv"))
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True); os.makedirs(R, exist_ok=True)

    # Progress is written to a PARTIAL sidecar, never to a.out, so an interrupted or
    # task-subset run cannot truncate a previously complete census. a.out is written
    # once, atomically, only after every requested task has been attempted.
    partial = a.out + ".partial"
    rows = []
    for suite, repo in SUITES.items():
        print(f"=== {suite} ({repo}) ===", flush=True)
        for task in a.tasks:
            r = census_one(suite, repo, task, a.cap)
            if r:
                rows.append(r)
                pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    if not df.empty:
        tmp = a.out + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, a.out)
        if os.path.exists(partial):
            os.remove(partial)
    if df.empty:
        print("no tasks censused"); return

    print("\n== cross-suite leak census ==")
    print(df[["suite", "task", "n_train_full", "n_test_full", "exact_dup_test_in_train",
              "leak_jaccard_0p7", "leak_containment_0p7", "verdict"]].to_string(index=False))

    print("\n== NT original vs revised (same task names; curation isolated) ==")
    piv = df.pivot_table(index="task", columns="suite",
                         values=["leak_jaccard_0p7", "leak_containment_0p7",
                                 "exact_dup_test_in_train"])
    print(piv.to_string())
    for suite in SUITES:
        s = df[df.suite == suite]
        if len(s):
            n_leaky = int((s.verdict != "clean").sum())
            print(f"{suite}: {n_leaky}/{len(s)} tasks non-clean "
                  f"(pre-registered {PREREGISTERED[suite]}); "
                  f"median jac@0.7 = {s.leak_jaccard_0p7.median():.4f}")
    print(f"\nEXP_CROSSSUITE_CENSUS_DONE -> {a.out}")


if __name__ == "__main__":
    main()
