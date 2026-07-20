#!/usr/bin/env python3
"""
Census two further public benchmark suites, to test how widespread the defect is.

The paper's census has covered Genomic Benchmarks, the Nucleotide Transformer downstream
tasks and GUE in sequence space, plus BEND in coordinate space. It names DeepSTARR, BEND
and DART-Eval as the obvious next targets. This module adds:

  PGB        the Plant Genomic Benchmark shipped with AgroNT (InstaDeepAI/
             plant-genomic-benchmark). Seventeen dataset-tasks over twelve plant species
             in five task families that fit the cheap-download budget: lncRNA coding
             potential, poly(A) site, PRO-seq occupancy, promoter strength, terminator
             strength. Three families are EXCLUDED for download cost, not for result:
             chromatin_access (39.9 GB), splicing (2.1 GB) and gene_exp (1.3 GB). That
             exclusion is by file size alone and was fixed BEFORE any leak fraction was
             computed, so it cannot be a selection effect on the outcome.

  DeepSTARR  Drosophila STARR-seq enhancer activity (GenerTeam mirror of the DeepSTARR
             Zenodo record 5502060). One dataset-task, 402,296 / 41,186 at 249 bp.

Both are censused with the SAME estimator as every other suite in the paper: exact 8-mer
Jaccard and containment, max over the training set, leak fraction = share of test
sequences above 0.7, verdict LEAKY above 0.1.

Efficiency note, and why it does not change the number. `max_sim_to_train` is called twice
in the other census modules, once per mode, and each call recomputes the same sparse
product M_test @ M_train.T -- the dominant cost. Here that product is formed once per
batch and BOTH metrics are derived from it. The arithmetic is identical: Jaccard is
inter/(sq+sr-inter) and containment is inter/min(sq,sr), exactly as in
homology_split.max_jaccard_to_reference and expkit.max_sim_to_train. `--verify-gue`
re-censuses two GUE tasks through this code path and checks the published
results/gue_census.csv numbers come back, so the shortcut is tested, not asserted.

Regression tasks (promoter/terminator strength, DeepSTARR activity) have continuous
labels. The census statistic is a function of SEQUENCES only and never touches the label,
so it is defined on regression tasks exactly as on classification ones; n_classes is
recorded as empty for them rather than faked.

Splits are as-shipped. Where a suite ships a validation split it is ignored, so the audited
split is the one the leaderboard reports on -- the same convention exp_gue.py uses for
GUE's dev split.

Run:  PYTHONPATH=. venv/bin/python -m audit.experiments.exp_newsuites_census
Out:  results/newsuites_census.csv
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import homology_split as H

R = os.path.join(HERE, "results")
OUT = os.path.join(R, "newsuites_census.csv")
SIM_K = 8
LEAK_CUT = 0.1

# Families downloaded. The three omitted ones are named in the docstring with their sizes.
PGB_FAMILIES = ["lncrna", "poly_a", "pro_seq", "promoter_strength", "terminator_strength"]
PGB_REGRESSION = {"promoter_strength", "terminator_strength"}

# Task-family -> the organism each species name denotes, for the census figure's colour
# axis. PGB is the first non-human, non-fly, non-worm, non-yeast, non-virus suite here.
PGB_ORGANISM = {
    "arabidopsis_thaliana": "Arabidopsis thaliana", "arabidopis_thaliana": "Arabidopsis thaliana",
    "g_max": "Glycine max", "glycine_max": "Glycine max",
    "m_esculenta": "Manihot esculenta",
    "s_bicolor": "Sorghum bicolor", "sorghum_bicolor": "Sorghum bicolor",
    "s_lycopersicum": "Solanum lycopersicum", "solanum_lycopersicum": "Solanum lycopersicum",
    "t_aestivum": "Triticum aestivum",
    "z_mays": "Zea mays", "zea_mays": "Zea mays",
    "chlamydomonas_reinhardtii": "Chlamydomonas reinhardtii",
    "medicago_truncatula": "Medicago truncatula",
    "oryza_sativa_indica_group": "Oryza sativa indica",
    "oryza_sativa_japonica_group": "Oryza sativa japonica",
    "trifolium_pratense": "Trifolium pratense",
    "leaf": "multi-species (leaf assay)", "protoplast": "multi-species (protoplast assay)",
}


def pgb_root():
    hits = glob.glob(os.path.join(
        HERE, "datacache", "pgb",
        "datasets--InstaDeepAI--plant-genomic-benchmark", "snapshots", "*"))
    if not hits:
        raise FileNotFoundError("PGB not cached under datacache/pgb")
    return hits[0]


def deepstarr_root():
    hits = glob.glob(os.path.join(
        HERE, "datacache", "deepstarr",
        "datasets--GenerTeam--DeepSTARR-enhancer-activity", "snapshots", "*"))
    if not hits:
        raise FileNotFoundError("DeepSTARR not cached under datacache/deepstarr")
    return hits[0]


def read_fa(path):
    """PGB ships strict two-line FASTA with header '>id|label'. Anything else is a
    format surprise and should raise rather than be silently coerced."""
    lines = open(path).read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) % 2:
        raise ValueError(f"{path}: odd line count, not two-line FASTA")
    seqs, labs = [], []
    for i in range(0, len(lines), 2):
        if not lines[i].startswith(">"):
            raise ValueError(f"{path}: line {i} is not a header")
        seqs.append(lines[i + 1].upper())
        labs.append(lines[i].rsplit("|", 1)[-1])
    return seqs, labs


def census(seqs_tr, seqs_te, batch_cells=4e7):
    """Leak statistics of a test set against a training set, both metrics from one pass.

    batch_cells bounds the dense intersection block at ~batch_cells float32 entries
    (~160 MB), so memory stays flat as the training set grows instead of scaling with it.
    """
    M = H.kmer_binary_matrix(seqs_tr + seqs_te, k=SIM_K)
    ntr = len(seqs_tr)
    Mr, Mq = M[:ntr], M[ntr:]
    sr = np.asarray(Mr.sum(1)).ravel()
    sq = np.asarray(Mq.sum(1)).ravel()
    RT = Mr.T.tocsr()
    b = max(1, int(batch_cells // max(ntr, 1)))
    jac = np.zeros(Mq.shape[0], np.float32)
    con = np.zeros(Mq.shape[0], np.float32)
    for s in range(0, Mq.shape[0], b):
        e = min(s + b, Mq.shape[0])
        inter = (Mq[s:e] @ RT).toarray()
        union = sq[s:e, None] + sr[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(union > 0, inter / union, 0.0)
            c = np.where(np.minimum(sq[s:e, None], sr[None, :]) > 0,
                         inter / np.minimum(sq[s:e, None], sr[None, :]), 0.0)
        jac[s:e] = j.max(1)
        con[s:e] = c.max(1)
    return jac, con


def row_for(suite, task, family, organism, seqs_tr, seqs_te, labs_te, regression, t0):
    jac, con = census(seqs_tr, seqs_te)
    L = np.array([len(s) for s in seqs_tr + seqs_te])
    trset = set(seqs_tr)
    exact = float(np.mean([s in trset for s in seqs_te]))
    j07, c07 = float((jac > 0.7).mean()), float((con > 0.7).mean())
    j09, c09 = float((jac >= 0.9).mean()), float((con >= 0.9).mean())
    verdict = ("LEAKY" if j07 > LEAK_CUT else
               "borderline" if c07 > LEAK_CUT else "clean")
    row = dict(
        suite=suite, task=task, family=family, organism=organism,
        label_type="regression" if regression else "classification",
        n_train=len(seqs_tr), n_test=len(seqs_te),
        n_train_full=len(seqs_tr), n_test_full=len(seqs_te),
        subsampled=False, verdict_is_lower_bound=False,   # every row is full-scale
        len_min=int(L.min()), len_med=int(np.median(L)), len_max=int(L.max()),
        n_classes=("" if regression else int(len(set(labs_te)))),
        exact_dup_test_in_train=round(exact, 4),
        leak_jaccard_0p7=round(j07, 4), leak_jaccard_0p9=round(j09, 4),
        leak_containment_0p7=round(c07, 4), leak_containment_0p9=round(c09, 4),
        sim_min=round(float(jac.min()), 4), sim_median=round(float(np.median(jac)), 4),
        verdict=verdict, seconds=round(time.time() - t0, 1))
    print(f"  [{suite}/{task:28s}] n={len(seqs_tr)}/{len(seqs_te)} "
          f"len={row['len_med']} jac@0.7={j07:.4f} con@0.7={c07:.4f} "
          f"exact={exact:.4f} -> {verdict} [{row['seconds']}s]", flush=True)
    return row


def run_pgb():
    root = pgb_root()
    rows = []
    for fam in PGB_FAMILIES:
        for trp in sorted(glob.glob(os.path.join(root, fam, "*_train.fa"))):
            task = os.path.basename(trp)[:-len("_train.fa")]
            tep = os.path.join(root, fam, f"{task}_test.fa")
            if not os.path.exists(tep):
                print(f"  [PGB/{fam}/{task}] no test split, skipped", flush=True)
                continue
            t0 = time.time()
            str_, _ = read_fa(trp)
            ste, lte = read_fa(tep)
            rows.append(row_for("PGB", f"{fam}/{task}", fam,
                                PGB_ORGANISM.get(task, task), str_, ste, lte,
                                fam in PGB_REGRESSION, t0))
            pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint after each task
    return rows


def run_deepstarr():
    root = deepstarr_root()
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(root, "train.parquet"))
    te = pd.read_parquet(os.path.join(root, "test.parquet"))
    str_ = tr["sequence"].astype(str).str.upper().tolist()
    ste = te["sequence"].astype(str).str.upper().tolist()
    return [row_for("DeepSTARR", "enhancer_activity", "enhancer_activity",
                    "Drosophila melanogaster", str_, ste, None, True, t0)]


def verify_gue(tasks=("emp_H3", "prom_core_tata")):
    """Re-census two GUE tasks through THIS code path and compare to the committed
    results/gue_census.csv. If the one-pass shortcut altered any number, this fails."""
    from huggingface_hub import hf_hub_download
    ref = pd.read_csv(os.path.join(R, "gue_census.csv")).set_index("task")
    cache = os.path.join(HERE, "datacache", "gue")
    ok = True
    for task in tasks:
        fr = {}
        for split in ("train", "test"):
            p = hf_hub_download("leannmlindsey/GUE", f"GUE/{task}/{split}.csv",
                                repo_type="dataset", cache_dir=cache)
            fr[split] = pd.read_csv(p)
        sc = next(c for c in fr["train"].columns if c.lower() in ("sequence", "seq"))
        a = fr["train"][sc].astype(str).str.upper().tolist()
        b = fr["test"][sc].astype(str).str.upper().tolist()
        jac, con = census(a, b)
        got = (round(float((jac > 0.7).mean()), 4), round(float((con > 0.7).mean()), 4),
               round(float((jac >= 0.9).mean()), 4))
        want = (ref.loc[task, "leak_jaccard_0p7"], ref.loc[task, "leak_containment_0p7"],
                ref.loc[task, "phi_ge0p9"])
        good = all(abs(g - w) < 1e-4 for g, w in zip(got, want))
        ok &= good
        print(f"  verify {task:16s} got={got} want={want} "
              f"{'MATCH' if good else 'MISMATCH'}", flush=True)
    print("GUE reproduction:", "PASS" if ok else "FAIL", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=["pgb", "deepstarr"])
    ap.add_argument("--verify-gue", action="store_true")
    a = ap.parse_args()
    if a.verify_gue:
        sys.exit(0 if verify_gue() else 1)
    rows = []
    if "pgb" in a.suites:
        rows += run_pgb()
    if "deepstarr" in a.suites:
        rows += run_deepstarr()
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} dataset-task censuses)")
    print(df["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
