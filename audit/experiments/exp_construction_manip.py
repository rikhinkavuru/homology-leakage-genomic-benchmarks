#!/usr/bin/env python3
"""
Deepener #3, layer 3 -- construct-and-break manipulations (the causal test).

The coordinate signatures in exp_construction.py are CORRELATIONAL: leaky datasets have
un-merged, self-overlapping coordinates and clean ones do not. This module tests the
CAUSAL direction by intervening on the construction step and leaving everything else
fixed -- mirroring the logic of the paper's existing injected-leakage check.

  MANIPULATION A -- BREAK A CLEAN SET.
      human_ocr_ensembl comes from the MERGED Ensembl Regulatory Build and is clean
      (coordinate self-overlap exactly 0.0000). We rebuild it the way an UN-MERGED
      multi-sample union is built, replicating the mechanism measured on
      human_enhancers_ensembl exactly: 94.3% of positive rows sit on a coordinate of
      multiplicity 2. So we duplicate a matching share of positive rows and re-split at
      random. PREDICTION: it flips clean -> LEAKY, RF inflates, and the ranking moves.

  MANIPULATION B -- FIX A LEAKY SET.
      human_enhancers_ensembl ships 77,421 positives on only 40,934 unique coordinates.
      We apply the merge step its curators omitted -- collapse every duplicated
      coordinate to ONE representative row -- and re-split. Nothing else changes: same
      sequences, same labels, same model code, same split ratio. PREDICTION: the leak
      fraction collapses toward 0 and RF's demotion disappears.

Together these close the causal loop in both directions: adding the construction defect
to a clean set creates the pathology, and removing it from a leaky set cures it.

CONTROL. Each manipulation is paired with an UNMANIPULATED run of the same pipeline at
the same scale and the same split seed, so any difference is attributable to the
intervention and not to re-splitting, subsampling, or model variance.

Run:  python -m audit.experiments.exp_construction_manip
Out:  results/construction_manipulation.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from audit.core import expkit as E
from audit.experiments.exp_construction import recover_coords

R = os.path.join(HERE, "results")
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
SEED = 0
CAP_CLEAN = 20000          # frozen convention for non-leaky datasets


def evaluate(seqs, y, tr, te, tag, extra, comp=None):
    """Leak census + 4-model ranking on a given split, plus the homology-corrected
    ranking on the same data. Returns one row.

    `comp` lets a caller pass precomputed near-duplicate components; the full-scale
    ensembl clustering is already on disk from the frozen pipeline and recomputing it
    costs far more than the rest of this module combined."""
    t0 = time.time()
    X = E.featurize(seqs, 6)
    sim = E.max_sim_to_train(seqs, tr, te, k=8, mode="jaccard")
    acc_o = {m: float(E.correctness(E.models()[m], X, y, tr, te).mean()) for m in MODELS}

    if comp is None:
        comp = E.clusters(seqs, 0.7, 8)
    ctr, cte = E.assign(comp, y, SEED, len(te) / len(seqs))
    acc_c = {m: float(E.correctness(E.models()[m], X, y, ctr, cte).mean()) for m in MODELS}

    tr_set = {seqs[i] for i in tr}
    exact = float(np.mean([seqs[i] in tr_set for i in te]))
    order_o = ">".join(sorted(MODELS, key=lambda m: -acc_o[m]))
    order_c = ">".join(sorted(MODELS, key=lambda m: -acc_c[m]))
    rank_o = 1 + sum(acc_o[m] > acc_o["RF"] for m in MODELS)
    rank_c = 1 + sum(acc_c[m] > acc_c["RF"] for m in MODELS)
    y = np.asarray(y)
    row = dict(
        condition=tag, n=len(seqs), n_train=len(tr), n_test=len(te),
        positive_frac=round(float((y == y.max()).mean()), 4),
        majority_baseline=round(float(max((y == y.max()).mean(),
                                          (y != y.max()).mean())), 4),
        leak_jaccard_0p7=round(float((sim > 0.7).mean()), 4),
        leak_jaccard_0p9=round(float((sim >= 0.9).mean()), 4),
        exact_dup_test_in_train=round(exact, 4),
        **{f"acc_orig_{m}": round(acc_o[m], 4) for m in MODELS},
        **{f"acc_corr_{m}": round(acc_c[m], 4) for m in MODELS},
        rf_drop=round(acc_o["RF"] - acc_c["RF"], 4),
        lr_drop=round(acc_o["LR"] - acc_c["LR"], 4),
        order_orig=order_o, order_corr=order_c,
        rf_rank_orig=rank_o, rf_rank_corr=rank_c,
        ranking_inverts=bool(order_o.split(">")[0] != order_c.split(">")[0]),
        seconds=round(time.time() - t0, 1), **extra)
    print(f"  [{tag}] n={len(seqs)} leak@0.7={row['leak_jaccard_0p7']:.4f} "
          f"RF {acc_o['RF']:.4f}->{acc_c['RF']:.4f} (drop {row['rf_drop']:+.4f}) "
          f"rank {rank_o}->{rank_c} | {order_o}  =>  {order_c} ({row['seconds']}s)",
          flush=True)
    return row


def random_split(n, test_frac, seed=SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cut = int(round((1 - test_frac) * n))
    return perm[:cut], perm[cut:]


def manipulation_A():
    """BREAK A CLEAN SET: give human_ocr_ensembl the un-merged-union construction."""
    d = "human_ocr_ensembl"
    print(f"== Manipulation A: break a clean set ({d}) ==", flush=True)
    seqs, y, otr, ote, tf = E.load(d, full=False, cap=CAP_CLEAN)
    rows = []
    # control: same data, same random-split machinery, no intervention
    tr, te = random_split(len(seqs), tf)
    rows.append(evaluate(seqs, np.asarray(y), tr, te, "A_control_merged_consensus",
                         dict(dataset=d, manipulation="none",
                              dup_share_of_positives=0.0)))
    # Intervention: reproduce the ensembl signature, where 0.9426 of positive ROWS sit
    # on a multiplicity-2 coordinate. Note the rate to DUPLICATE is not 0.9426:
    # duplicating a fraction r of P positives yields 2rP rows on multiplicity-2
    # coordinates out of P(1+r) total, i.e. a share of 2r/(1+r). Solving
    # 2r/(1+r)=TARGET gives r = TARGET/(2-TARGET).
    TARGET = 0.9426
    RATE = TARGET / (2 - TARGET)                       # = 0.8914
    y = np.asarray(y)
    pos = np.where(y == y.max())[0]
    rng = np.random.RandomState(SEED)
    dup = rng.choice(pos, size=int(round(RATE * len(pos))), replace=False)
    seqs2 = list(seqs) + [seqs[i] for i in dup]
    y2 = np.concatenate([y, y[dup]])
    tr2, te2 = random_split(len(seqs2), tf)
    rows.append(evaluate(seqs2, y2, tr2, te2, "A_manipulated_unmerged_union",
                         dict(dataset=d,
                              manipulation="duplicate positives (multiplicity 2)",
                              dup_rate_applied=round(RATE, 4),
                              target_share_on_dup_coord=TARGET)))

    # BALANCE + SCALE CONTROL for A. Duplicating only positives moves the set from
    # 50/50 to ~65/35 AND adds ~47% more rows than the control, either of which can move
    # accuracy on its own. We therefore repeat the intervention and then downsample to
    # the control's size AND balance, so the manipulated and control arms differ only in
    # whether near-duplicates are present.
    n_ctrl = len(seqs)
    pos2 = np.where(y2 == y2.max())[0]
    neg2 = np.where(y2 != y2.max())[0]
    half = min(n_ctrl // 2, len(pos2), len(neg2))
    sel = np.sort(np.concatenate([rng.choice(pos2, half, replace=False),
                                  rng.choice(neg2, half, replace=False)]))
    seqs3 = [seqs2[i] for i in sel]
    y3 = y2[sel]
    tr3, te3 = random_split(len(seqs3), tf)
    rows.append(evaluate(seqs3, y3, tr3, te3, "A_manipulated_matched",
                         dict(dataset=d,
                              manipulation="duplicate positives + match control size/balance",
                              dup_rate_applied=round(RATE, 4),
                              target_share_on_dup_coord=TARGET)))
    return rows


def manipulation_B():
    """FIX A LEAKY SET: apply the merge step human_enhancers_ensembl omitted."""
    d = "human_enhancers_ensembl"
    print(f"== Manipulation B: fix a leaky set ({d}) ==", flush=True)
    seqs, y, otr, ote, tf = E.load(d, full=True)
    y = np.asarray(y)
    iv, ver = recover_coords(d)
    # Row count alone would pass on a same-length but mis-ordered table. recover_coords
    # already verifies per-row correspondence (interval width vs sequence length, and
    # label agreement); require those, not just the count.
    assert len(iv) == len(seqs), f"coordinate alignment failed: {len(iv)} vs {len(seqs)}"
    assert ver.get("len_match") == 1.0, f"interval/sequence width mismatch: {ver.get('len_match')}"
    assert ver.get("label_match") == 1.0, f"label mismatch: {ver.get('label_match')}"
    rows = []
    # control: unmanipulated, random split at the same ratio (isolates the merge step
    # from the as-shipped split, which is itself random)
    tr, te = random_split(len(seqs), tf)
    rows.append(evaluate(seqs, y, tr, te, "B_control_unmerged_union",
                         dict(dataset=d, manipulation="none", kept_fraction=1.0),
                         comp=E.cached_clusters(d, seqs, 0.7, 8)))
    # intervention: collapse each duplicated coordinate to a single representative
    key = (iv["region"] + ":" + iv["start"].astype(str) + "-" + iv["end"].astype(str)).to_numpy()
    _uniq, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    seqs2 = [seqs[i] for i in keep]
    y2 = y[keep]
    tr2, te2 = random_split(len(seqs2), tf)
    rows.append(evaluate(seqs2, y2, tr2, te2, "B_manipulated_merged",
                         dict(dataset=d, manipulation="collapse duplicate coordinates",
                              kept_fraction=round(len(keep) / len(seqs), 4),
                              positive_fraction=round(float((y2 == y2.max()).mean()), 4))))

    # BALANCE CONTROL. Only the POSITIVE class carries duplicated coordinates, so
    # collapsing them takes the set from 50/50 to ~35/65 positive. Class balance alone
    # moves accuracy (the majority-class baseline rises from 0.500 to ~0.654), so the
    # unbalanced comparison cannot separate "we removed the leakage" from "we changed
    # the prior". We therefore repeat the intervention with negatives subsampled to the
    # surviving positive count, restoring 50/50 against a control that is also 50/50.
    pos2 = np.where(y2 == y2.max())[0]
    neg2 = np.where(y2 != y2.max())[0]
    rng = np.random.RandomState(SEED)
    neg_keep = rng.choice(neg2, size=len(pos2), replace=False)
    sel = np.sort(np.concatenate([pos2, neg_keep]))
    seqs3 = [seqs2[i] for i in sel]
    y3 = y2[sel]
    tr3, te3 = random_split(len(seqs3), tf)
    rows.append(evaluate(seqs3, y3, tr3, te3, "B_manipulated_merged_balanced",
                         dict(dataset=d,
                              manipulation="collapse duplicate coordinates + rebalance",
                              kept_fraction=round(len(sel) / len(seqs), 4),
                              positive_fraction=round(float((y3 == y3.max()).mean()), 4))))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B"],
                    help="run one manipulation and merge into the existing CSV")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(R, exist_ok=True)
    out = f"{R}/construction_manipulation.csv"
    rows = []
    if a.only in (None, "B"):
        rows += manipulation_B()
    if a.only in (None, "A"):
        rows += manipulation_A()
    df = pd.DataFrame(rows)
    if a.only and os.path.exists(out):
        prev = pd.read_csv(out)
        prev = prev[~prev.condition.isin(set(df.condition))]
        df = pd.concat([prev, df], ignore_index=True)
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)

    print("\n== construct-and-break summary ==")
    print(df[["condition", "n", "leak_jaccard_0p7", "exact_dup_test_in_train",
              "acc_orig_RF", "acc_corr_RF", "rf_drop", "rf_rank_orig", "rf_rank_corr",
              "ranking_inverts"]].to_string(index=False))
    for pair, label in [(("B_control_unmerged_union", "B_manipulated_merged_balanced"),
                         "FIX-A-LEAKY, balance-matched (expect leak -> ~0, RF drop -> ~0)"),
                        (("B_control_unmerged_union", "B_manipulated_merged"),
                         "FIX-A-LEAKY, unbalanced (confounded by class prior)"),
                        (("A_control_merged_consensus", "A_manipulated_matched"),
                         "BREAK-A-CLEAN, size+balance-matched (expect leak -> high)"),
                        (("A_control_merged_consensus", "A_manipulated_unmerged_union"),
                         "BREAK-A-CLEAN, unmatched (confounded by size and prior)")]:
        if not (df.condition == pair[0]).any() or not (df.condition == pair[1]).any():
            continue
        a = df[df.condition == pair[0]].iloc[0]
        b = df[df.condition == pair[1]].iloc[0]
        print(f"\n{label}")
        print(f"  n / pos% : {a.n}/{a.positive_frac:.3f}  ->  {b.n}/{b.positive_frac:.3f} "
              f"(majority baseline {a.majority_baseline:.3f} -> {b.majority_baseline:.3f})")
        print(f"  leak@0.7 : {a.leak_jaccard_0p7:.4f}  ->  {b.leak_jaccard_0p7:.4f}")
        print(f"  RF drop  : {a.rf_drop:+.4f}  ->  {b.rf_drop:+.4f}")
        print(f"  RF rank  : {a.rf_rank_orig}->{a.rf_rank_corr}   "
              f"becomes   {b.rf_rank_orig}->{b.rf_rank_corr}")
        print(f"  inverts  : {bool(a.ranking_inverts)}  ->  {bool(b.ranking_inverts)}")
    print(f"\nEXP_CONSTRUCTION_MANIP_DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
