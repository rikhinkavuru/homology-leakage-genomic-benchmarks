#!/usr/bin/env python3
"""
GUE: the pre-registration's binding forward predictions, finally executed.

results/tier1_preregistration.md §4.1 registered these BEFORE any GUE data was touched:

    GUE core-promoter (70 bp), TF-binding (100 bp), promoter (300 bp), human  -> LEAKY
    GUE yeast EMP, virus CVC (multi-species)                                  -> CLEAN

They were the strongest-risk claims in the document and they went unexecuted. This runs
them. Two stages:

  STAGE 1 (census)  leak fraction on the as-shipped split, same estimator as the paper.
  STAGE 2 (screen)  for every task the census flags, compute the diagnostic condition of
                    eq. (1) from AS-SHIPPED measurements only -- phi, per-model novel
                    accuracy n and graded gap g -- and record where an inversion is even
                    POSSIBLE (some challenger with delta = n_B - n_A > 0) and where the
                    condition predicts one (phi > phi* = delta/Dg).

Stage 2 matters because the paper's open question is whether a second, independent leaky
dataset shows a ranking inversion. The Nucleotide Transformer suite did not, and the
condition explained why: delta <= 0 almost everywhere, so no inversion was possible at any
leak fraction. GUE is the remaining registered candidate pool. Any task with delta > 0 AND
phi > phi* is a genuine candidate and is worth the full ranking run; tasks without it are
predicted null in advance, which is the screen doing its job.

Counting note: GUE ships train/dev/test. We use train as train and test as test, and
ignore dev, so the split we audit is the one the leaderboard reports on.

Run:  python -m audit.experiments.exp_gue [--tasks ...] [--cap N]
Out:  results/gue_census.csv (always); results/gue_screen.csv only if some task
      is non-clean -- the inversion screen has nothing to run on a clean task, so a
      fully clean census writes no screen file. The full-scale run writes none.
"""
from __future__ import annotations
import argparse, io, itertools, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E

R = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "datacache", "gue")
REPO = "leannmlindsey/GUE"
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
SIM_K, FEAT_K = 8, 6
LEAK_CUT = 0.1

# Pre-registered, from results/tier1_preregistration.md 4.1. Recorded here so the code
# carries the prediction it tests. These fifteen tasks, and ONLY these, are scored.
#
# The registered text names exactly two families: "GUE core-promoter (70 bp), TF-binding
# (100 bp), promoter (300 bp), human -> LEAKY" and "GUE yeast EMP, virus CVC
# (multi-species) -> CLEAN".
PREREG = {
    "prom_core_all": "LEAKY", "prom_core_notata": "LEAKY", "prom_core_tata": "LEAKY",
    "prom_300_all": "LEAKY", "prom_300_notata": "LEAKY", "prom_300_tata": "LEAKY",
    "human_tf_0": "LEAKY", "human_tf_1": "LEAKY", "human_tf_2": "LEAKY",
    "human_tf_3": "LEAKY", "human_tf_4": "LEAKY",
    "emp_H3": "CLEAN", "emp_H3K4me3": "CLEAN", "emp_H4": "CLEAN",
    "virus_covid": "CLEAN",
}

# NOT pre-registered. An earlier version of this module listed mouse_0/mouse_1 inside
# PREREG under the comment "from tier1_preregistration.md 4.1", but the pre-registration
# names neither: mouse TF-binding is not human, not yeast EMP, and not virus CVC. Both
# happened to come out as this module expected, so scoring them inflated the headline
# from 3/15 to 5/17 -- an unregistered task that agrees with you is not evidence, and
# quietly adding two of them to a pre-registered tally is the exact failure this whole
# paper argues against. They are reported separately and unscored whenever they are
# run. Note the committed full-scale census covers the fifteen REGISTERED tasks only,
# so results/gue_census.csv has no mouse rows; they appeared in the earlier capped
# run. Pass them explicitly via --tasks to re-census them.
EXPLORATORY = {"mouse_0": "expected CLEAN (multi-species, not registered)",
               "mouse_1": "expected CLEAN (multi-species, not registered)"}
DEFAULT = list(PREREG) + list(EXPLORATORY)

# Redundancy among the registered tasks, verified directly against the shipped files.
# The "_all" promoter tasks are the EXACT union of their own notata and tata variants:
# set(prom_core_all.test) == set(prom_core_notata.test) | set(prom_core_tata.test), with
# zero symmetric difference, and likewise at 300 bp. So the eleven predicted-leaky tasks
# are not eleven independent benchmarks -- two of them are unions of two others. The
# independent count is at most NINE test partitions (core notata+tata, 300 notata+tata,
# and the five TF tasks). Any statement of the form "N independent benchmarks are clean"
# must use nine, not eleven.
#
# Nine is an UPPER bound on independence, not a demonstration of it. The 70 bp and the
# 300 bp promoter families ship identical row counts (47,356 train / 5,920 test), which
# is consistent with one underlying promoter set windowed at two widths; only 8% of the
# 70 bp test windows occur as literal substrings of the 300 bp corpus, but that is weak
# evidence either way, since a re-extracted window need not be a substring of a wider
# one. We therefore do NOT claim the two families are independent of each other. If they
# share loci, the independent count is nearer seven.
REDUNDANT_UNIONS = {"prom_core_all": ("prom_core_notata", "prom_core_tata"),
                    "prom_300_all": ("prom_300_notata", "prom_300_tata")}
N_INDEPENDENT_LEAKY_PREDICTIONS = 9


def load_task(task, cap=None, seed=0, sizes=None):
    """Load a GUE task. When `sizes` is a dict it is filled with the FULL, pre-cap row
    counts, so the caller can record whether a verdict rests on a truncated training
    set. Truncating train can only lower a max-similarity-to-train statistic, so a
    capped clean verdict is a lower bound, never a clean bill of health -- this is the
    same C1 discipline the paper applies to its own suite."""
    from huggingface_hub import hf_hub_download
    frames = {}
    for split in ("train", "test"):
        p = hf_hub_download(REPO, f"GUE/{task}/{split}.csv", repo_type="dataset",
                            cache_dir=CACHE)
        frames[split] = pd.read_csv(p)
    seq_col = next(c for c in frames["train"].columns if c.lower() in ("sequence", "seq"))
    lab_col = next(c for c in frames["train"].columns if c.lower() in ("label", "labels"))
    tr_s = frames["train"][seq_col].astype(str).str.upper().tolist()
    te_s = frames["test"][seq_col].astype(str).str.upper().tolist()
    tr_y = frames["train"][lab_col].to_numpy()
    te_y = frames["test"][lab_col].to_numpy()
    if sizes is not None:
        sizes["n_train_full"], sizes["n_test_full"] = len(tr_s), len(te_s)
    rng = np.random.RandomState(seed)
    sub = False
    if cap and len(tr_s) > cap:
        k = rng.choice(len(tr_s), cap, replace=False)
        tr_s = [tr_s[i] for i in k]; tr_y = tr_y[k]; sub = True
    if cap and len(te_s) > cap:
        k = rng.choice(len(te_s), cap, replace=False)
        te_s = [te_s[i] for i in k]; te_y = te_y[k]; sub = True
    return tr_s, tr_y, te_s, te_y, sub


def run_task(task, cap, do_screen=True):
    t0 = time.time()
    try:
        sizes = {}
        tr_s, tr_y, te_s, te_y, sub = load_task(task, cap=cap, sizes=sizes)
    except Exception as ex:                                        # noqa: BLE001
        print(f"  [{task}] unavailable: {type(ex).__name__}", flush=True)
        return None, []
    seqs = tr_s + te_s
    y = np.concatenate([tr_y, te_y]).astype(int)
    tr = np.arange(len(tr_s)); te = np.arange(len(tr_s), len(seqs))
    L = np.array([len(s) for s in seqs])

    sim_j = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="jaccard")
    sim_c = E.max_sim_to_train(seqs, tr, te, k=SIM_K, mode="containment")
    trset = set(tr_s)
    exact = float(np.mean([s in trset for s in te_s]))
    jac07 = float((sim_j > 0.7).mean())
    con07 = float((sim_c > 0.7).mean())
    phi = float((sim_j >= 0.9).mean())
    verdict = ("LEAKY" if jac07 > LEAK_CUT else
               "borderline" if con07 > LEAK_CUT else "clean")
    registered = task in PREREG
    row = dict(suite="GUE", task=task,
               preregistered=PREREG.get(task, "not_registered"),
               registered=registered,
               n_train=len(tr_s), n_test=len(te_s), subsampled=sub,
               n_train_full=sizes.get("n_train_full"),
               n_test_full=sizes.get("n_test_full"),
               train_frac_used=round(len(tr_s) / sizes["n_train_full"], 4)
               if sizes.get("n_train_full") else None,
               # A clean verdict measured on a truncated training set is a LOWER bound:
               # the discarded rows could have been the near-duplicate partners.
               verdict_is_lower_bound=bool(sub),
               len_med=int(np.median(L)), n_classes=int(len(np.unique(y))),
               exact_dup_test_in_train=round(exact, 4),
               leak_jaccard_0p7=round(jac07, 4), leak_containment_0p7=round(con07, 4),
               phi_ge0p9=round(phi, 4), verdict=verdict,
               prereg_correct=(bool(PREREG[task] ==
                                    ("LEAKY" if verdict == "LEAKY" else "CLEAN"))
                               if registered else None),
               seconds=round(time.time() - t0, 1))
    print(f"  [{task:18s}] n={len(tr_s)}/{len(te_s)} len={row['len_med']} "
          f"jac@0.7={jac07:.4f} con@0.7={con07:.4f} exact={exact:.4f} phi={phi:.4f} "
          f"-> {verdict} (prereg {PREREG.get(task)}, "
          f"{'OK' if row['prereg_correct'] else 'WRONG'}) [{row['seconds']}s]", flush=True)

    screens = []
    if do_screen and verdict != "clean" and len(np.unique(y)) == 2:
        X = E.featurize(seqs, FEAT_K)
        novel, high = sim_j < 0.5, sim_j >= 0.9
        if novel.any() and high.any():
            acc, strat = {}, {}
            for m in MODELS:
                c = E.correctness(E.models()[m], X, y, tr, te)
                acc[m] = float(c.mean())
                n_m = float(c[novel].mean()); h_m = float(c[high].mean())
                strat[m] = dict(n=n_m, g=h_m - n_m)
            for A, B in itertools.permutations(MODELS, 2):
                if acc[A] <= acc[B]:
                    continue
                delta = strat[B]["n"] - strat[A]["n"]
                Dg = strat[A]["g"] - strat[B]["g"]
                phis = delta / Dg if Dg > 0 else np.nan
                screens.append(dict(
                    suite="GUE", task=task, leader_A=A, challenger_B=B,
                    phi=round(phi, 4), delta=round(delta, 4), Dg=round(Dg, 4),
                    phi_star=(round(phis, 4) if np.isfinite(phis) else None),
                    informative=bool(delta > 0),
                    PREDICTED_swap=bool(delta > 0 and phi * Dg > delta)))
            cand = [s for s in screens if s["PREDICTED_swap"]]
            print(f"      screen: {sum(s['informative'] for s in screens)}/{len(screens)} "
                  f"informative (delta>0); {len(cand)} predicted to swap"
                  f"{' <-- CANDIDATE' if cand else ''}", flush=True)
    return row, screens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=DEFAULT)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--no-screen", action="store_true")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    rows, screens = [], []
    for t in a.tasks:
        r, s = run_task(t, a.cap, do_screen=not a.no_screen)
        if r:
            rows.append(r); screens += s
            pd.DataFrame(rows).to_csv(f"{R}/gue_census.csv.partial", index=False)
    if not rows:
        print("nothing censused"); return
    for frame, name in ((pd.DataFrame(rows), "gue_census"),
                        (pd.DataFrame(screens), "gue_screen")):
        if frame.empty:
            continue
        tmp = f"{R}/{name}.csv.tmp"; frame.to_csv(tmp, index=False)
        os.replace(tmp, f"{R}/{name}.csv")
    part = f"{R}/gue_census.csv.partial"
    if os.path.exists(part):
        os.remove(part)

    df = pd.DataFrame(rows)
    print("\n== GUE census vs the pre-registration ==")
    print(df[["task", "preregistered", "len_med", "leak_jaccard_0p7",
              "leak_containment_0p7", "exact_dup_test_in_train", "verdict",
              "prereg_correct"]].to_string(index=False))
    reg = df[df.registered]
    unreg = df[~df.registered]
    print(f"\npre-registered predictions correct: "
          f"{int(reg.prereg_correct.sum())}/{len(reg)}")
    for grp in ("LEAKY", "CLEAN"):
        g = reg[reg.preregistered == grp]
        if len(g):
            print(f"  predicted {grp}: {int(g.prereg_correct.sum())}/{len(g)} "
                  f"(median jac@0.7 {g.leak_jaccard_0p7.median():.4f})")
    if len(unreg):
        print(f"\n  [{len(unreg)} exploratory task(s), NOT pre-registered and NOT scored: "
              f"{', '.join(unreg.task)}] "
              f"verdicts: {', '.join(f'{t}={v}' for t, v in zip(unreg.task, unreg.verdict))}")
        print("  These are reported for completeness only. They are not part of the "
              "registered tally and must not be added to it.")
    if screens:
        s = pd.DataFrame(screens)
        cand = s[s.PREDICTED_swap]
        print(f"\n== inversion screen ==\n  informative pairs (delta>0): "
              f"{int(s.informative.sum())}/{len(s)}")
        print(f"  pairs predicted to swap: {len(cand)}")
        if len(cand):
            print(cand[["task", "leader_A", "challenger_B", "delta", "Dg",
                        "phi", "phi_star"]].to_string(index=False))
            print("  ^ these are candidates for a full ranking run")
    print("\nEXP_GUE_DONE")


if __name__ == "__main__":
    main()
