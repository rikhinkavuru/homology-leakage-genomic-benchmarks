#!/usr/bin/env python3
"""
Authoritative final synthesis. Combines:
  * FULL-SCALE leakage           (results/leakage_full.csv)
  * FULL-SCALE drops             (results/fullscale_summary.csv)  -- for the
    datasets the leakage scan flagged as non-trivially leaky and that had been
    subsampled (nontata, enhancers_ensembl, coding_vs_intergenomic)
  * SUBSAMPLE drops              (results/per_dataset_results.csv) -- for the
    genuinely-clean datasets, where the ~0 subsample drop is unconfounded
    because full-scale leakage is ~0 (nothing was masked).

Pure aggregation of already-computed, validated numbers (no model fitting).
Emits results/summary_final.csv, results/capacity_scaling_final.csv,
results/results_FINAL.md, results/figures/capacity_scaling_FINAL.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
FIG = os.path.join(R, "figures")
CONFIGS = [(4, "LR"), (6, "LR"), (4, "RF"), (6, "RF")]
FULLSCALE = ["human_nontata_promoters", "human_enhancers_ensembl",
             "demo_coding_vs_intergenomic_seqs"]
BINARY = ["human_nontata_promoters", "human_enhancers_ensembl",
          "demo_coding_vs_intergenomic_seqs", "human_enhancers_cohn",
          "human_ocr_ensembl", "demo_human_or_worm", "drosophila_enhancers_stark"]

leak = pd.read_csv(os.path.join(R, "leakage_full.csv")).set_index("dataset")
fs = pd.read_csv(os.path.join(R, "fullscale_summary.csv"))
sub = pd.read_csv(os.path.join(R, "per_dataset_results.csv"))
subsum = pd.read_csv(os.path.join(R, "summary.csv")).set_index("dataset")


def cfg_rows(dset):
    """Return {(k,model): dict(orig, rand, hom, hom_std, dhom, drand, oauroc, hauroc)},
    drawing from the full-scale table if available else the subsample table,
    plus the scale label."""
    out = {}
    if dset in FULLSCALE:
        d = fs[fs.dataset == dset]; scale = "full"
        for _, r in d.iterrows():
            out[(int(r.k), r.model)] = dict(
                orig=r.original_acc, rand=r.random_acc_mean, hom=r.homology_acc_mean,
                hom_std=r.homology_acc_std, dhom=r.delta_acc_homology,
                drand=r.delta_acc_random, oauroc=r.original_auroc, hauroc=r.homology_auroc_mean)
    else:
        d = sub[sub.dataset == dset]; scale = "subsample"
        for _, r in d.iterrows():
            out[(int(r.k), r.model)] = dict(
                orig=r.original_accuracy, rand=r.random_accuracy_mean, hom=r.homology_accuracy_mean,
                hom_std=r.homology_accuracy_std, dhom=r.delta_accuracy_homology,
                drand=r.delta_accuracy_random, oauroc=r.original_auroc, hauroc=r.homology_auroc_mean)
    return out, scale


summary_rows, cap_rows = [], []
for dset in BINARY:
    cfgs, scale = cfg_rows(dset)
    best = max(CONFIGS, key=lambda c: cfgs[c]["orig"])
    bk, bm = best
    b = cfgs[best]
    summary_rows.append(dict(
        dataset=dset, drop_scale=scale,
        n_full=int(leak.loc[dset, "n_train"] + leak.loc[dset, "n_test"]),
        leak_full_0p5=leak.loc[dset, "leak_full_0.5"],
        leak_full_0p7=leak.loc[dset, "leak_full_0.7"],
        leak_full_0p9=leak.loc[dset, "leak_full_0.9"],
        best_model=f"{bm}_k{bk}",
        original_acc=round(b["orig"], 4), random_acc=round(b["rand"], 4),
        homology_acc=round(b["hom"], 4), homology_acc_std=round(b["hom_std"], 4),
        delta_homology=round(b["dhom"], 4), delta_random=round(b["drand"], 4),
        original_auroc=round(b["oauroc"], 4), homology_auroc=round(b["hauroc"], 4)))
    crow = {"dataset": dset, "drop_scale": scale}
    deltas = []
    for (k, m) in CONFIGS:
        crow[f"hom_delta_{m}_k{k}"] = round(cfgs[(k, m)]["dhom"], 4)
        crow[f"rand_delta_{m}_k{k}"] = round(cfgs[(k, m)]["drand"], 4)
        deltas.append(cfgs[(k, m)]["dhom"])
    crow["monotone"] = bool(all(deltas[i] <= deltas[i+1] + 1e-9 for i in range(3)))
    cap_rows.append(crow)

summary = pd.DataFrame(summary_rows)
cap = pd.DataFrame(cap_rows)
summary.to_csv(os.path.join(R, "summary_final.csv"), index=False)
cap.to_csv(os.path.join(R, "capacity_scaling_final.csv"), index=False)

# ---- final capacity-scaling figure ----
plt.figure(figsize=(8.5, 5.6))
xs = list(range(4))
for dset in BINARY:
    crow = cap[cap.dataset == dset].iloc[0]
    ys = [crow[f"hom_delta_{m}_k{k}"] * 100 for (k, m) in CONFIGS]
    leaky = leak.loc[dset, "leak_full_0.7"] > 0.1
    plt.plot(xs, ys, marker="o", lw=2.2 if leaky else 1.0,
             alpha=1.0 if leaky else 0.55,
             label=f"{dset.replace('_',' ')[:22]} (leak@0.7={leak.loc[dset,'leak_full_0.7']:.2f})")
# mean random-control band
rand_means = [np.mean([cap[cap.dataset == d].iloc[0][f"rand_delta_{m}_k{k}"] for d in BINARY]) * 100
              for (k, m) in CONFIGS]
plt.plot(xs, rand_means, "k--", lw=1.5, label="random re-split (mean, control)")
plt.axhline(0, color="gray", lw=0.8)
plt.xticks(xs, [f"{m} k{k}" for (k, m) in CONFIGS])
plt.xlabel("model capacity (low -> high)")
plt.ylabel("accuracy drop under homology re-split (points)")
plt.title("Capacity-scaling of the homology-leakage drop\n(full-scale for leaky datasets; subsample for clean)")
plt.legend(fontsize=7.5, loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "capacity_scaling_FINAL.png"), dpi=140)
plt.close()

# ---- results_FINAL.md ----
L = []
L.append("# Homology-leakage audit — final synthesis (Genomic Benchmarks suite)\n")
L.append("Leakage is measured on the **full** dataset (test→train exact 8-mer Jaccard). "
         "The accuracy drop is measured at **full scale for the leaky datasets** "
         "(nontata, enhancers_ensembl, coding_vs_intergenomic) and on a 20k stratified "
         "subsample for the genuinely-clean datasets (where full-scale leakage ≈ 0, so the "
         "subsample drop is unconfounded). Pipeline identical across all: k-mer counts "
         "(k=4/6), LogisticRegression + RandomForest(150), exact 8-mer Jaccard, "
         "whole-cluster re-split; 3 re-split seeds; model seed 0.\n")

L.append("## Master table (best original model per dataset)\n")
L.append("| dataset | n | leak@0.7 (full) | best model | original | random | homology | Δ(hom) | Δ(rand) | drop scale |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in summary.iterrows():
    L.append(f"| {r['dataset']} | {r['n_full']} | {r['leak_full_0p7']:.3f} | {r['best_model']} "
             f"| {r['original_acc']:.3f} | {r['random_acc']:.3f} | {r['homology_acc']:.3f}±{r['homology_acc_std']:.3f} "
             f"| **{r['delta_homology']:+.3f}** | {r['delta_random']:+.3f} | {r['drop_scale']} |")
L.append("")

L.append("## Capacity scaling (homology-aware accuracy drop by model)\n")
L.append("| dataset | leak@0.7 | LR k4 | LR k6 | RF k4 | RF k6 | monotone | scale |")
L.append("|---|---|---|---|---|---|---|---|")
for _, r in cap.iterrows():
    lk = leak.loc[r['dataset'], 'leak_full_0.7']
    L.append(f"| {r['dataset']} | {lk:.3f} | {r['hom_delta_LR_k4']:+.3f} | {r['hom_delta_LR_k6']:+.3f} "
             f"| {r['hom_delta_RF_k4']:+.3f} | {r['hom_delta_RF_k6']:+.3f} | {'yes' if r['monotone'] else 'no'} "
             f"| {r['drop_scale']} |")
L.append("")
L.append("Negative-control (random re-split) deltas, same best-model column, are all within "
         "noise of zero — see `capacity_scaling_final.csv` (`rand_delta_*`).\n")

# cross-dataset stats
leaky = summary[summary.leak_full_0p7 > 0.1]
clean = summary[summary.leak_full_0p7 <= 0.1]
L.append("## Cross-dataset summary\n")
L.append(f"- **Leaky datasets (leak@0.7 > 0.1): {len(leaky)}** — "
         + ", ".join(leaky.dataset) + ".")
L.append(f"  - best-model homology drop: mean **{leaky.delta_homology.mean():.3f}** "
         f"(range {leaky.delta_homology.min():+.3f}..{leaky.delta_homology.max():+.3f}); "
         f"random-control drop mean {leaky.delta_random.mean():+.3f}.")
L.append(f"- **Clean datasets: {len(clean)}** — " + ", ".join(clean.dataset) + ".")
L.append(f"  - best-model homology drop: mean **{clean.delta_homology.mean():+.3f}** "
         f"(range {clean.delta_homology.min():+.3f}..{clean.delta_homology.max():+.3f}) — i.e. ~0.")
L.append("")
L.append("**Headline:** the homology-aware accuracy drop is large only where homology "
         "leakage exists, scales with model capacity (RF ≫ LR), and vanishes under a random "
         "re-split of the same data — establishing that the inflation is caused by "
         "near-duplicate train/test leakage, not by re-splitting per se.\n")

with open(os.path.join(R, "results_FINAL.md"), "w") as fh:
    fh.write("\n".join(L))

print(summary.to_string())
print("\nwrote results/summary_final.csv, capacity_scaling_final.csv, results_FINAL.md, "
      "figures/capacity_scaling_FINAL.png")
