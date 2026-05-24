#!/usr/bin/env python3
"""
FULL-SCALE leakage measurement (decoupled from the subsampled model-fitting).

The suite (run_suite.py) subsamples large datasets to 20k for the homology
re-split + model retraining, which is fine for the DELTAS but biases the leakage
FRACTION downward (random subsampling breaks near-duplicate pairs). Leakage is a
cheap descriptive statistic -- test->train max k-mer Jaccard is one batched sparse
product, no clustering or fitting -- so we measure it on the FULL dataset here to
get authoritative numbers and resolve whether the large datasets are truly clean
or whether subsampling merely hid their leakage.

Reuses RA.kmer_binary_matrix and RA.max_jaccard_to_reference unchanged.
Run:   python measure_leakage_full.py   -> results/leakage_full.csv + histograms
"""
import os, glob, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_audit as RA
from run_suite import SHORT, THRESHOLDS, FIG

DSETS = [
    "human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl",
    "human_ocr_ensembl", "demo_human_or_worm", "demo_coding_vs_intergenomic_seqs",
    "drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl",
    "human_ensembl_regulatory",
]


def load_split(base, split):
    seqs = []
    sd = os.path.join(base, split)
    for cls in sorted(os.listdir(sd)):
        cd = os.path.join(sd, cls)
        if os.path.isdir(cd):
            for fp in sorted(glob.glob(os.path.join(cd, "*.txt"))):
                with open(fp) as fh:
                    seqs.append(fh.read().strip().upper())
    return seqs


def main():
    from genomic_benchmarks.loc2seq import download_dataset
    rows = []
    for d in DSETS:
        t = time.time()
        try:
            base = download_dataset(d, version=0)
            tr = load_split(base, "train")
            te = load_split(base, "test")
            Mtr = RA.kmer_binary_matrix(tr, RA.SIM_K)
            Mte = RA.kmer_binary_matrix(te, RA.SIM_K)
            max_sim = RA.max_jaccard_to_reference(Mte, Mtr, batch=400)
            row = dict(dataset=d, n_train=len(tr), n_test=len(te),
                       median_maxsim=round(float(np.median(max_sim)), 4),
                       p99_maxsim=round(float(np.percentile(max_sim, 99)), 4))
            for thr in THRESHOLDS:
                row[f"leak_full_{thr}"] = round(float((max_sim > thr).mean()), 4)
            rows.append(row)
            print(f"OK   {d}: ntr={len(tr)} nte={len(te)} "
                  + " ".join(f">{thr}:{row[f'leak_full_{thr}']:.3f}" for thr in THRESHOLDS)
                  + f"  median={row['median_maxsim']:.3f}  ({time.time()-t:.0f}s)", flush=True)
            plt.figure(figsize=(7, 4.2))
            plt.hist(max_sim, bins=60, range=(0, 1), color="#2a7a48", edgecolor="white")
            for thr in THRESHOLDS:
                plt.axvline(thr, color="crimson", ls="--", lw=1)
            plt.xlabel(f"max {RA.SIM_K}-mer Jaccard, test seq -> any train seq (FULL dataset)")
            plt.ylabel("# test sequences")
            plt.title(f"{d}  (FULL: ntrain={len(tr)}, ntest={len(te)})")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, f"sim_hist_FULL_{SHORT[d]}.png"), dpi=120)
            plt.close()
            del Mtr, Mte, tr, te, max_sim
            gc.collect()
        except Exception as e:
            print(f"FAIL {d}: {type(e).__name__}: {e}  ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RA.RESULTS_DIR, "leakage_full.csv"), index=False)
    print("LEAKAGE_FULL_DONE -> results/leakage_full.csv", flush=True)


if __name__ == "__main__":
    main()
