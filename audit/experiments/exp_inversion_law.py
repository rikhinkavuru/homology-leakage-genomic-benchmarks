#!/usr/bin/env python3
"""
Deepener #2 -- the theoretical inversion law (CPU-only reanalysis of frozen artifacts).

THEORY
------
Stratify the leaky test set by maximum k-mer similarity to train into a LEAKED
stratum (similarity >= 0.9, fraction phi) and a NOVEL stratum (similarity < 0.5).
For a model M with leaked-stratum accuracy h_M and novel-stratum accuracy n_M, the
two-stratum identity gives its accuracy on the as-shipped (leaky) split

    a_M  =  n_M + phi * g_M ,        g_M = h_M - n_M   (graded memorization gap)

so the split-induced drop is not an empirical coincidence but a product:

    Delta_M = a_M - a_M^corr = phi * g_M      (using a^corr ~= n, tested below).

For an ordered pair (A = leader on the as-shipped split, B = challenger) write
    delta   = n_B - n_A            (B's true novel-sequence advantage)
    Dg      = g_A - g_B            (A's excess memorization propensity)
The as-shipped margin is mu = a_A - a_B = phi*Dg - delta, so A leads the leaky
benchmark while B is genuinely better exactly when

    0 < delta < phi * Dg                      <=>      phi > phi* = delta / Dg

phi* is a RANKING BREAKDOWN POINT: the leak fraction above which the benchmark
reports the wrong winner.

CAUTION -- the "observable form" carries NO predictive content. Substituting
Delta_M = a_M - a_M^corr into 0 < mu < Delta_A - Delta_B reduces it to
a_B^corr > a_A^corr, i.e. it IS the definition of a corrected-split inversion,
restated. We compute it only as an internal-consistency check on the frozen tables
(it must be 12/12 by algebra; anything else means an arithmetic error in the CSVs).
ALL predictive content lives in the phi*g form, whose inputs (phi, n, g) are measured
on the as-shipped split ALONE and never touch the corrected split.

WHAT IS ACTUALLY TESTED (each part states its own circularity status)
--------------------------------------------------------------------
 P1 reconstruction  : a_orig ~= n + phi*g            -- IDENTITY, not a prediction.
                      a_orig IS the bin-weighted mean of the same graded accuracies, so
                      the multi-bin form is exact by construction (residual ~1e-4) and
                      the 2-bin residual measures only the mid-similarity bins' mass.
                      Reported as an arithmetic-consistency check between two CSVs.
 P2 corrected split : a_corr ~= n, hence the corrected winner is argmax(n)
                      -- OUT OF SAMPLE (the theory never sees the corrected split,
                      which requires re-splitting AND retraining). NOTE the novel_only
                      readout is excluded from the score: its ground truth is itself
                      argmax(n), so it is true by construction.
 P3 reg-path sweep  : P1 applied at 22 RF regularization cells, competitors fixed
                      -- IN SAMPLE (same as-shipped test set, restated as an ordering).
                      Zero free parameters, and it shows the identity survives a
                      12-fold change in g, but it is NOT held-out validation.
 P4 injection sweep : h_M fitted at f=0.1 ONLY, then f=0.2/0.4 predicted
                      -- OUT OF SAMPLE along the phi-axis, one parameter per model
 P5 residual        : r_M = a_corr - n_M, the assumption diagnostic
 P6 metric invariance (MMseqs2/containment), P7 CNN instantiation, P8 cluster
    bootstrap CIs on delta and on the inversion margin m_inv = phi*Dg - delta.

Inputs are FROZEN CSVs only (plus cached similarity/cluster arrays for P8). Every
accuracy consumed here is on the canonical `.predict()` decision rule (see
results/reconciled_numbers.md S5): graded_performance.csv (run_graded.py uses
`mdl.predict`), exp_rankings.csv / exp_regpath.csv / exp_alignment.csv /
exp_inject3class.csv / exp_deep_cnn.csv (all via expkit).

Run:  python -m audit.experiments.exp_inversion_law            (P1-P7, seconds)
      python -m audit.experiments.exp_inversion_law --bootstrap  (adds P8, minutes)
"""
from __future__ import annotations
import argparse, itertools, os, sys, time
import numpy as np
import pandas as pd

R = "results"
LEAKY = ["human_enhancers_ensembl", "human_nontata_promoters"]
MODELS = ["LR", "LinearSVC", "RF", "HGB"]
NOVEL_BIN, HIGH_BIN = "[0,0.5)", "[0.9,1.0]"
BOOT = 10000
BSEED = 20240524


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


# ---------------------------------------------------------------------------
# strata (phi, n, g) from the frozen graded table
# ---------------------------------------------------------------------------
def load_strata():
    """phi, n_M, g_M per leaky dataset from results/graded_performance.csv.

    phi is the [0.9,1.0] bin's share of the test set, i.e. the leak fraction under
    the SAME binning that defines n and g -- self-consistency inside the identity
    matters more than matching leakage_fraction.csv's strict-inequality count."""
    gp = pd.read_csv(f"{R}/graded_performance.csv")
    out = {}
    for d in LEAKY:
        sub = gp[(gp.dataset == d) & (gp.split == "original")]
        bins = sub[sub.model == "LR"].set_index("sim_bin")["n"]
        n_test = int(bins.sum())
        phi = float(bins[HIGH_BIN]) / n_test
        mid = 1.0 - phi - float(bins[NOVEL_BIN]) / n_test
        per = {}
        for m in MODELS:
            s = sub[sub.model == m].set_index("sim_bin")["accuracy"]
            n_m, h_m = float(s[NOVEL_BIN]), float(s[HIGH_BIN])
            per[m] = dict(n=n_m, h=h_m, g=h_m - n_m)
        # exact multi-bin identity (law of total probability) for reference
        w = (bins / n_test).to_dict()
        exact = {m: float(sum(w[b] * sub[(sub.model == m) & (sub.sim_bin == b)]["accuracy"].iloc[0]
                              for b in w)) for m in MODELS}
        out[d] = dict(phi=phi, mid_mass=mid, n_test=n_test, per=per, exact=exact)
    return out


# ---------------------------------------------------------------------------
# P1 / P2 -- anchors: reconstruction + corrected-split prediction
# ---------------------------------------------------------------------------
def part_anchors(S):
    rk = pd.read_csv(f"{R}/exp_rankings.csv")
    nv = pd.read_csv(f"{R}/novelonly_ranking.csv").set_index("dataset")
    rows, pair_rows = [], []
    for d in LEAKY:
        st, phi = S[d], S[d]["phi"]
        orig = rk[(rk.dataset == d) & (rk.metric == "acc") & (rk.split == "original")].iloc[0]
        corr = rk[(rk.dataset == d) & (rk.metric == "acc") & (rk.split == "corrected")].iloc[0]
        for m in MODELS:
            n_m, g_m = st["per"][m]["n"], st["per"][m]["g"]
            a_hat, a_obs = n_m + phi * g_m, float(orig[m])
            a_corr = float(corr[m])
            rows.append(dict(
                dataset=d, model=m, phi=_r(phi), mid_bin_mass=_r(st["mid_mass"]),
                novel_acc_n=_r(n_m), leaked_acc_h=_r(st["per"][m]["h"]), graded_gap_g=_r(g_m),
                # P1 two-stratum reconstruction of the as-shipped accuracy
                acc_orig_pred_2bin=_r(a_hat), acc_orig_pred_4bin=_r(st["exact"][m]),
                acc_orig_obs=_r(a_obs), recon_err_2bin=_r(a_hat - a_obs),
                recon_err_4bin=_r(st["exact"][m] - a_obs),
                # P5 assumption residual: does the corrected split land on the novel stratum?
                acc_corr_obs=_r(a_corr), residual_corr_minus_novel=_r(a_corr - n_m),
                # Delta = phi*g vs observed drop
                drop_pred_phi_g=_r(phi * g_m), drop_obs=_r(a_obs - a_corr)))
        # ---- pairwise breakdown points
        for A, B in itertools.permutations(MODELS, 2):
            if float(orig[A]) <= float(orig[B]):
                continue  # A is by definition the as-shipped leader of the pair
            nA, nB = st["per"][A]["n"], st["per"][B]["n"]
            gA, gB = st["per"][A]["g"], st["per"][B]["g"]
            delta, Dg = nB - nA, gA - gB
            mu = float(orig[A]) - float(orig[B])
            phi_star = delta / Dg if Dg > 0 else np.nan
            m_inv = phi * Dg - delta
            pred_inv = bool(delta > 0 and phi * Dg > delta)          # 0 < delta < phi*Dg
            # NOT a prediction: 0 < mu < Delta_A - Delta_B reduces algebraically to
            # a_B^corr > a_A^corr. Computed only as an arithmetic-consistency check on
            # the frozen tables -- it must agree with obs_inversion_corrected_acc 12/12.
            dA = float(orig[A]) - float(corr[A])
            dB = float(orig[B]) - float(corr[B])
            pred_inv_obs = bool(0 < mu < dA - dB)
            obs_corr_inv = bool(float(corr[B]) > float(corr[A]))
            novel_order = nv.loc[d, "c_novelonly_ranking"].split(">")
            obs_novel_inv = bool(novel_order.index(B) < novel_order.index(A))
            pair_rows.append(dict(
                dataset=d, leader_A=A, challenger_B=B, phi=_r(phi),
                n_A=_r(nA), n_B=_r(nB), delta=_r(delta), g_A=_r(gA), g_B=_r(gB), Dg=_r(Dg),
                phi_star=_r(phi_star), m_inv=_r(m_inv), mu_orig=_r(mu),
                pred_inversion_phi_g=pred_inv,
                obs_form_identity_check=pred_inv_obs,
                obs_inversion_novel=obs_novel_inv, obs_inversion_corrected_acc=obs_corr_inv,
                agree_phi_g_vs_novel=bool(pred_inv == obs_novel_inv),
                obs_form_is_tautological=bool(pred_inv_obs == obs_corr_inv)))
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def part_winner(S):
    """P2 -- the out-of-sample claim: the corrected-split winner is argmax(n)."""
    rk = pd.read_csv(f"{R}/exp_rankings.csv")
    nv = pd.read_csv(f"{R}/novelonly_ranking.csv").set_index("dataset")
    rows = []
    for d in LEAKY:
        st = S[d]
        pred_orig = max(MODELS, key=lambda m: st["per"][m]["n"] + st["phi"] * st["per"][m]["g"])
        pred_corr = max(MODELS, key=lambda m: st["per"][m]["n"])
        for metric in ["acc", "auroc", "f1"]:
            o = rk[(rk.dataset == d) & (rk.metric == metric) & (rk.split == "original")].iloc[0]
            c = rk[(rk.dataset == d) & (rk.metric == metric) & (rk.split == "corrected")].iloc[0]
            rows.append(dict(dataset=d, readout=f"corrected_{metric}",
                             pred_orig_winner=pred_orig, obs_orig_winner=str(o["best"]),
                             pred_corr_winner=pred_corr, obs_corr_winner=str(c["best"]),
                             orig_match=bool(pred_orig == str(o["best"])),
                             corr_match=bool(pred_corr == str(c["best"])),
                             tautological=False))
        # NOT a test: novelonly_ranking.best_novel_model is itself argmax over the same
        # novel-stratum accuracies that define pred_corr_winner, so this row is True by
        # construction. Recorded for completeness, excluded from the score.
        rows.append(dict(dataset=d, readout="novel_only", tautological=True,
                         pred_orig_winner=pred_orig, obs_orig_winner=str(
                             nv.loc[d, "a_leaky_ranking"].split(">")[0]),
                         pred_corr_winner=pred_corr,
                         obs_corr_winner=str(nv.loc[d, "best_novel_model"]),
                         orig_match=bool(pred_orig == nv.loc[d, "a_leaky_ranking"].split(">")[0]),
                         corr_match=bool(pred_corr == nv.loc[d, "best_novel_model"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P3 -- out-of-sample sweep #1: the regularization path (g-axis)
# ---------------------------------------------------------------------------
def part_regpath(S):
    """Predict RF's rank on the AS-SHIPPED split at 22 regularization cells from
    (phi, novel_acc(cell), graded_gap(cell)) alone. phi is measured ONCE per dataset;
    the three competitors are fixed at their frozen as-shipped accuracies. No free
    parameters, no per-cell fitting."""
    rp = pd.read_csv(f"{R}/exp_regpath.csv")
    rk = pd.read_csv(f"{R}/exp_rankings.csv")
    rows = []
    for d in LEAKY:
        phi = S[d]["phi"]
        o = rk[(rk.dataset == d) & (rk.metric == "acc") & (rk.split == "original")].iloc[0]
        comp = {m: float(o[m]) for m in MODELS if m != "RF"}
        for _, c in rp[rp.dataset == d].iterrows():
            a_hat = float(c.novel_acc) + phi * float(c.graded_gap)
            pred_rank = 1 + sum(v > a_hat for v in comp.values())
            obs_rank = int(c.rf_rank_orig)
            rows.append(dict(
                dataset=d, axis=c.axis, value=c.value, phi=_r(phi),
                novel_acc=_r(c.novel_acc), graded_gap=_r(c.graded_gap),
                acc_orig_pred=_r(a_hat), acc_orig_obs=_r(c.acc_orig),
                recon_err=_r(a_hat - float(c.acc_orig)),
                rf_rank_pred=pred_rank, rf_rank_obs=obs_rank,
                rank_match=bool(pred_rank == obs_rank)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P4 -- out-of-sample sweep #2: injected leakage (phi-axis)
# ---------------------------------------------------------------------------
def part_inject():
    """exp_inject3class.py appends n_inj mutated copies of TRAIN rows to the TEST set,
    so the contaminated leak fraction is known BY CONSTRUCTION:
        phi(f) = n_inj / (n_test_orig + n_inj).
    n_M is read off the f=0 cell. h_M is fitted at f=0.1 ONLY; f=0.2 and f=0.4 are
    then pure predictions. Finally phi* = delta/Dg gives the critical injection f*."""
    ij = pd.read_csv(f"{R}/exp_inject3class.csv")
    d = ij.dataset.iloc[0]
    # recover the split sizes the experiment actually used (cap=20000, seed=0)
    from audit.core import expkit as E
    _seqs, _y, otr, ote, _tf = E.load(d, full=False, cap=20000)
    n_tr, n_te = len(otr), len(ote)
    rows = []
    fit = {}
    for m in ["RF", "LR"]:
        sub = ij[ij.model == m].sort_values("inject_frac")
        n_M = float(sub[sub.inject_frac == 0].acc_orig.iloc[0])
        c01 = sub[sub.inject_frac == 0.1].iloc[0]
        phi01 = float(c01.n_injected) / (n_te + float(c01.n_injected))
        h_M = (float(c01.acc_orig) - (1 - phi01) * n_M) / phi01     # the ONE fitted number
        fit[m] = dict(n=n_M, h=h_M, g=h_M - n_M)
        for _, c in sub.iterrows():
            phi = float(c.n_injected) / (n_te + float(c.n_injected))
            a_hat = n_M + phi * (h_M - n_M)
            rows.append(dict(
                dataset=d, model=m, inject_frac=float(c.inject_frac),
                n_injected=int(c.n_injected), n_train=n_tr, n_test_orig=n_te,
                phi_by_construction=_r(phi), novel_acc_n=_r(n_M),
                leaked_acc_h_fit_at_f0p1=_r(h_M), graded_gap_g=_r(h_M - n_M),
                acc_orig_pred=_r(a_hat), acc_orig_obs=_r(c.acc_orig),
                pred_err=_r(a_hat - float(c.acc_orig)),
                held_out_cell=bool(float(c.inject_frac) not in (0.0, 0.1)),
                acc_corr_obs=_r(c.acc_corr), residual_corr_minus_novel=_r(float(c.acc_corr) - n_M)))
    df = pd.DataFrame(rows)
    # Critical injection fraction, using the SAME (A,B) convention as the main law:
    # A = leader on the contaminated (as-shipped) split = RF, the memorizer;
    # B = challenger that is genuinely better on novel sequences = LR (it leads at f=0,
    # where phi=0 so the as-shipped split IS the novel stratum).
    # delta = n_B - n_A > 0 is LR's true advantage; Dg = g_A - g_B > 0 is RF's excess
    # memorization propensity; RF overtakes exactly when phi > phi* = delta/Dg.
    A, B = "RF", "LR"
    delta = fit[B]["n"] - fit[A]["n"]
    Dg = fit[A]["g"] - fit[B]["g"]
    phi_star = delta / Dg
    n_inj_star = phi_star * n_te / (1 - phi_star)
    f_star = n_inj_star / n_tr
    order = df.pivot_table(index="inject_frac", columns="model", values="acc_orig_obs")
    crit = dict(dataset=d, pair="RF_overtakes_LR", leader_A="RF", challenger_B="LR",
                n_LR=_r(fit["LR"]["n"]), n_RF=_r(fit["RF"]["n"]),
                delta_n_LR_minus_n_RF=_r(delta), g_LR=_r(fit["LR"]["g"]), g_RF=_r(fit["RF"]["g"]),
                Dg=_r(Dg), phi_star=_r(phi_star), n_injected_star=_r(n_inj_star, 1),
                f_star=_r(f_star), f_star_bracket="(0, 0.1)",
                obs_order_f0="LR>RF" if order.loc[0.0, "LR"] > order.loc[0.0, "RF"] else "RF>LR",
                obs_order_f0p1="LR>RF" if order.loc[0.1, "LR"] > order.loc[0.1, "RF"] else "RF>LR",
                obs_flip_between_f0_and_f0p1=bool(
                    (order.loc[0.0, "LR"] > order.loc[0.0, "RF"]) !=
                    (order.loc[0.1, "LR"] > order.loc[0.1, "RF"])))
    return df, pd.DataFrame([crit])


# ---------------------------------------------------------------------------
# P6 -- metric invariance; P7 -- CNN instantiation
# ---------------------------------------------------------------------------
def part_alignment(S):
    """Does the verdict survive swapping the homology detector? Re-derive phi* under
    the k-mer-Jaccard, containment and MMseqs2 views and compare against each view's
    own phi."""
    al = pd.read_csv(f"{R}/exp_alignment.csv")
    gb = pd.read_csv(f"{R}/exp_gb1_containment.csv").set_index("dataset")
    rows = []
    for d in LEAKY:
        st = S[d]
        sub = al[al.dataset == d].set_index("model")
        A = sub.acc_orig.idxmax()
        for det, phi in [("kmer_jaccard_0.9", st["phi"]),
                         ("containment_0.9", float(gb.loc[d, "leak_contain_0p9"]))]:
            for B in MODELS:
                if B == A:
                    continue
                delta = st["per"][B]["n"] - st["per"][A]["n"]
                Dg = st["per"][A]["g"] - st["per"][B]["g"]
                if Dg <= 0:
                    continue
                rows.append(dict(dataset=d, detector=det, leader_A=A, challenger_B=B,
                                 phi=_r(phi), phi_star=_r(delta / Dg),
                                 delta=_r(delta), Dg=_r(Dg),
                                 phi_exceeds_phi_star=bool(phi > delta / Dg),
                                 pred_inversion=bool(delta > 0 and phi > delta / Dg)))
        # MMseqs2 view: the corrected split rebuilt with alignment clustering
        for B in MODELS:
            if B == A:
                continue
            rows.append(dict(dataset=d, detector="mmseqs2_corrected_split", leader_A=A,
                             challenger_B=B, phi=None, phi_star=None,
                             delta=_r(float(sub.loc[B, "acc_corr_mmseqs"]) -
                                      float(sub.loc[A, "acc_corr_mmseqs"])),
                             Dg=None, phi_exceeds_phi_star=None,
                             pred_inversion=bool(float(sub.loc[B, "acc_corr_mmseqs"]) >
                                                 float(sub.loc[A, "acc_corr_mmseqs"]))))
    return pd.DataFrame(rows)


def part_cnn(S):
    """P7 -- the identity is model-agnostic: instantiate a = n + phi*g on the
    from-scratch 1D CNN dose-response cells (exp_deep_cnn.csv)."""
    cn = pd.read_csv(f"{R}/exp_deep_cnn.csv")
    rows = []
    for _, c in cn.iterrows():
        d = c.dataset
        phi = S[d]["phi"]
        a_hat = float(c.novel_acc) + phi * float(c.graded_gap)
        rows.append(dict(
            dataset=d, dropout=float(c.dropout), weight_decay=float(c.weight_decay),
            variant=c.variant, phi=_r(phi), novel_acc=_r(c.novel_acc),
            graded_gap=_r(c.graded_gap), acc_orig_pred=_r(a_hat), acc_orig_obs=_r(c.acc_orig),
            recon_err=_r(a_hat - float(c.acc_orig)),
            drop_pred_phi_g=_r(phi * float(c.graded_gap)), drop_obs=_r(c["drop"]),
            drop_err=_r(phi * float(c.graded_gap) - float(c["drop"])),
            residual_corr_minus_novel=_r(float(c.acc_corr) - float(c.novel_acc))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P8 -- cluster (block) bootstrap on delta and on the inversion margin
# ---------------------------------------------------------------------------
def part_bootstrap(S, boot=BOOT, seed=BSEED):
    """Pairs cluster bootstrap resampling whole WITHIN-TEST near-duplicate components
    as blocks.

    Blocks are defined by `cluster_bootstrap.test_internal_clusters` -- components of
    the 8-mer Jaccard > 0.7 graph built on the TEST SET ALONE. This is the paper's
    frozen estimator (results/cluster_bootstrap_full.csv) and is the right one here:
    the dependence being corrected for is between TEST examples, so two test sequences
    must not share a block merely because both resemble a common training sequence.
    (Using global components restricted to test gives a coarser, different blocking --
    7014 vs 7618 blocks on nonTATA -- so we deliberately reuse the frozen function
    rather than introduce a second definition; cf. reconciled_numbers.md S5.)

    The estimand is a difference of stratum means, so the pairs cluster bootstrap is
    the natural analogue of the wild cluster bootstrap-t used for regression
    coefficients, and it is already validated in this repo."""
    from audit.core import expkit as E
    from audit.experiments.cluster_bootstrap import test_internal_clusters
    rows = []
    for d in LEAKY:
        t0 = time.time()
        seqs, y, otr, ote, _tf = E.load(d)
        X = E.featurize(seqs, 6)
        sim = E.cached_max_sim(d, seqs, otr, ote, k=8, mode="jaccard")
        comp, _ncomp = test_internal_clusters([seqs[i] for i in ote], 0.7)
        novel, high = sim < 0.5, sim >= 0.9
        corr = {m: E.correctness(E.models()[m], X, y, otr, ote) for m in MODELS}
        print(f"  [{d}] fitted 4 models on {len(otr)} train / {len(ote)} test "
              f"({time.time()-t0:.0f}s)", flush=True)

        # per-cluster aggregates -> vectorised block resampling
        uniq, inv = np.unique(comp, return_inverse=True)
        G = len(uniq)
        cnt_n = np.bincount(inv, weights=novel.astype(float), minlength=G)
        cnt_h = np.bincount(inv, weights=high.astype(float), minlength=G)
        cnt_a = np.bincount(inv, minlength=G).astype(float)
        s_n = {m: np.bincount(inv, weights=corr[m] * novel, minlength=G) for m in MODELS}
        s_h = {m: np.bincount(inv, weights=corr[m] * high, minlength=G) for m in MODELS}

        rng = np.random.RandomState(seed)
        draws = {k: np.empty(boot) for k in ["phi"]}
        for m in MODELS:
            draws[f"n_{m}"] = np.empty(boot)
            draws[f"g_{m}"] = np.empty(boot)
        CH = 250
        for s in range(0, boot, CH):
            e = min(s + CH, boot)
            pick = rng.randint(0, G, size=(e - s, G))
            tn, th, ta = cnt_n[pick].sum(1), cnt_h[pick].sum(1), cnt_a[pick].sum(1)
            draws["phi"][s:e] = th / ta
            for m in MODELS:
                nn = s_n[m][pick].sum(1) / np.maximum(tn, 1)
                hh = s_h[m][pick].sum(1) / np.maximum(th, 1)
                draws[f"n_{m}"][s:e] = nn
                draws[f"g_{m}"][s:e] = hh - nn

        phi_pt = float(high.mean())
        for A, B in itertools.permutations(MODELS, 2):
            aA = float(corr[A].mean()); aB = float(corr[B].mean())
            if aA <= aB:
                continue
            nA = float(corr[A][novel].mean()); nB = float(corr[B][novel].mean())
            gA = float(corr[A][high].mean()) - nA
            gB = float(corr[B][high].mean()) - nB
            delta_d = draws[f"n_{B}"] - draws[f"n_{A}"]
            Dg_d = draws[f"g_{A}"] - draws[f"g_{B}"]
            minv_d = draws["phi"] * Dg_d - delta_d
            dlo, dhi = np.percentile(delta_d, [2.5, 97.5])
            mlo, mhi = np.percentile(minv_d, [2.5, 97.5])
            # two-sided bootstrap p-values, floored at 1/boot (a p of 0 is not a p)
            p_delta = max(1.0 / boot, 2 * min((delta_d <= 0).mean(), (delta_d >= 0).mean()))
            p_minv = max(1.0 / boot, 2 * min((minv_d <= 0).mean(), (minv_d >= 0).mean()))
            clean = bool(mlo > 0 and dlo > 0)
            fragile = bool(dlo <= 0 <= dhi)
            rows.append(dict(
                dataset=d, leader_A=A, challenger_B=B, n_clusters=G, n_boot=boot,
                phi=_r(phi_pt), delta=_r(nB - nA), Dg=_r(gA - gB),
                phi_star=_r((nB - nA) / (gA - gB)) if gA > gB else None,
                m_inv=_r(phi_pt * (gA - gB) - (nB - nA)),
                delta_ci=f"[{dlo:.4f}, {dhi:.4f}]", delta_excl0=bool(dlo > 0 or dhi < 0),
                m_inv_ci=f"[{mlo:.4f}, {mhi:.4f}]", m_inv_excl0=bool(mlo > 0 or mhi < 0),
                p_delta=round(float(p_delta), 5), p_m_inv=round(float(p_minv), 5),
                verdict="CLEAN" if clean else ("FRAGILE" if fragile else "no-inversion")))
        print(f"  [{d}] cluster bootstrap G={G} boot={boot} done "
              f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    # MULTIPLICITY. 12 pairs x 2 estimands = 24 tests, and the single headline CLEAN
    # verdict is the best of them -- that is post-selection inference and must be
    # corrected, not just mentioned. Benjamini-Hochberg at q=0.05, applied over all
    # pairwise delta tests and separately over the m_inv tests.
    for col in ("p_delta", "p_m_inv"):
        keep = bh(df[col].to_numpy(), q=0.05)
        df[col.replace("p_", "bh_sig_")] = keep
    # A pair only keeps its CLEAN badge if BOTH estimands survive BH.
    df["verdict_bh"] = np.where(
        (df.verdict == "CLEAN") & df.bh_sig_delta & df.bh_sig_m_inv, "CLEAN",
        np.where(df.verdict == "CLEAN", "CLEAN_fails_BH", df.verdict))
    return df


def bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


# ---------------------------------------------------------------------------
def make_figure(pairs, regpath, inject, crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    lim = 1.02
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#dfe7f2", zorder=0)
    ax.fill_between([0, lim], [0, 0], [0, lim], color="#f7e3e3", zorder=0)
    ax.plot([0, lim], [0, lim], color="0.35", lw=1.4, zorder=2)
    ax.text(0.62, 0.30, r"$\varphi>\varphi^*$: ranking inverts", fontsize=9.5, color="#8c2f2f")
    ax.text(0.10, 0.83, r"$\varphi<\varphi^*$: ranking holds", fontsize=9.5, color="#2f4d8c")

    def _plot(x, y, **kw):
        ax.scatter(np.clip(x, 0, lim), np.clip(y, 0, lim), zorder=4, **kw)

    p = pairs.dropna(subset=["phi_star"])
    p = p[p.phi_star.between(0, lim)]
    for d, mk, lab in [("human_enhancers_ensembl", "o", "Ensembl enhancers (pairs)"),
                       ("human_nontata_promoters", "s", "nonTATA promoters (pairs)")]:
        s = p[p.dataset == d]
        _plot(s.phi, s.phi_star, marker=mk, s=78, facecolor="none",
              edgecolor="#1f3f7a" if "ensembl" in d else "#7a1f3f", lw=1.6, label=lab)
    ci = crit.iloc[0]
    _plot([float(inject[inject.inject_frac == f].phi_by_construction.iloc[0])
           for f in [0.1, 0.2, 0.4]], [float(ci.phi_star)] * 3, marker="^", s=70,
          color="#c8791a", label=r"injection sweep ($\varphi$-axis, 3-class)")
    _plot([float(ci.phi_star)], [float(ci.phi_star)], marker="*", s=180, color="#c8791a",
          label=r"predicted $f^*$ crossing")
    ax.set_xlabel(r"measured leak fraction  $\varphi$")
    ax.set_ylabel(r"breakdown point  $\varphi^*=\delta/\Delta g$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_title(r"Ranking breakdown point: inversion iff $\varphi>\varphi^*=\delta/\Delta g$",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(f"{R}/figures", exist_ok=True)
    fig.savefig(f"{R}/figures/fig_inversion_phase.png", dpi=200)

    fig2, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for axx, d, ttl in [(axes[0], "human_enhancers_ensembl", "Ensembl enhancers"),
                        (axes[1], "human_nontata_promoters", "nonTATA promoters")]:
        s = regpath[(regpath.dataset == d) & (regpath.axis == "min_samples_leaf")]
        axx.plot(s.value, s.acc_orig_obs, "o-", color="#1f3f7a", label="observed")
        axx.plot(s.value, s.acc_orig_pred, "s--", color="#c8791a",
                 label=r"predicted $n+\varphi g$")
        axx.set_xscale("log"); axx.set_xlabel("RF min_samples_leaf")
        axx.set_ylabel("as-shipped accuracy"); axx.set_title(ttl, fontsize=10)
        axx.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{R}/figures/fig_inversion_regpath.png", dpi=200)
    print("wrote results/figures/fig_inversion_phase.png, fig_inversion_regpath.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true",
                    help="also run P8 (refits 4 models per leaky dataset; minutes)")
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(R, exist_ok=True)
    S = load_strata()
    for d in LEAKY:
        print(f"[strata] {d}: phi={S[d]['phi']:.4f} n_test={S[d]['n_test']} "
              f"mid-bin mass={S[d]['mid_mass']:.4f}")

    anchors, pairs = part_anchors(S)
    winner = part_winner(S)
    regpath = part_regpath(S)
    inject, crit = part_inject()
    align = part_alignment(S)
    cnn = part_cnn(S)

    anchors.to_csv(f"{R}/inversion_law_anchors.csv", index=False)
    pairs.to_csv(f"{R}/inversion_law_pairs.csv", index=False)
    winner.to_csv(f"{R}/inversion_law_winner.csv", index=False)
    regpath.to_csv(f"{R}/inversion_law_regpath.csv", index=False)
    inject.to_csv(f"{R}/inversion_law_inject.csv", index=False)
    crit.to_csv(f"{R}/inversion_law_critical_f.csv", index=False)
    align.to_csv(f"{R}/inversion_law_alignment.csv", index=False)
    cnn.to_csv(f"{R}/inversion_law_cnn.csv", index=False)

    print("\n== P1 two-stratum IDENTITY check (arithmetic consistency, NOT a prediction) ==")
    print("   a_orig is the bin-weighted mean of the same graded accuracies, so the")
    print("   4-bin form is exact by construction; the 2-bin residual measures only how")
    print("   much accuracy mass the mid-similarity bins carry.")
    for d in LEAKY:
        s = anchors[anchors.dataset == d]
        print(f"  {d}: mean|err| 2-bin={s.recon_err_2bin.abs().mean():.4f} "
              f"4-bin={s.recon_err_4bin.abs().mean():.4f} "
              f"(mid-bin mass {S[d]['mid_mass']:.4f})")
    print("\n== P2 corrected-split winner (the genuinely out-of-sample claim) ==")
    print(winner.to_string(index=False))
    scored = winner[~winner.tautological]
    print(f"  SCORED readouts (novel_only excluded as tautological): "
          f"corrected {int(scored.corr_match.sum())}/{len(scored)}")
    for d in LEAKY:
        s = scored[scored.dataset == d]
        print(f"    {d}: {int(s.corr_match.sum())}/{len(s)}")
    print("\n== pairwise breakdown points ==")
    print(f"  phi*g form agrees with novel-stratum order: "
          f"{int(pairs.agree_phi_g_vs_novel.sum())}/{len(pairs)}")
    print(f"  pairs predicted to invert (0 < delta < phi*Dg): "
          f"{list(pairs[pairs.pred_inversion_phi_g].apply(lambda r: f'{r.dataset}:{r.leader_A}->{r.challenger_B}', axis=1))}")
    print(f"  observable-form identity check (must be {len(pairs)}/{len(pairs)} by algebra): "
          f"{int(pairs.obs_form_is_tautological.sum())}/{len(pairs)}")
    print("\n== P3 reg-path (g-axis): IN-SAMPLE restatement of P1 across 22 HP cells ==")
    print("   It predicts the AS-SHIPPED rank from strata of that same as-shipped test")
    print("   set, so it is the identity re-expressed as an ordering, not held-out data.")
    print("   Its value is that the identity survives a 12-fold change in g.")
    print(f"  rank match {int(regpath.rank_match.sum())}/{len(regpath)}; "
          f"mean|recon err| = {regpath.recon_err.abs().mean():.4f}")
    for d in LEAKY:
        s = regpath[regpath.dataset == d]
        print(f"    {d}: {int(s.rank_match.sum())}/{len(s)} "
              f"mean|err|={s.recon_err.abs().mean():.4f}")
    print("\n== P4 injection (phi-axis) ==")
    ho = inject[inject.held_out_cell]
    print(f"  held-out cells (f=0.2,0.4): max|err| = {ho.pred_err.abs().max():.4f}")
    print(crit.to_string(index=False))
    print("\n== P5 residual a_corr - n ==")
    print(anchors[["dataset", "model", "residual_corr_minus_novel"]].to_string(index=False))

    if a.bootstrap:
        print("\n== P8 cluster bootstrap ==", flush=True)
        bs = part_bootstrap(S, boot=a.boot)
        bs.to_csv(f"{R}/inversion_law_bootstrap.csv", index=False)
        print(bs.to_string(index=False))

    make_figure(pairs, regpath, inject, crit)
    print(f"\nEXP_INVERSION_LAW_DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
