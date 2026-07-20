#!/usr/bin/env python3
"""
exp_detector_specificity.py -- MEASURE the 8-mer Jaccard detector's precision,
recall and false-positive rate against local-alignment ground truth, instead of
arguing (as the manuscript currently does) that specificity cannot be settled by
pairwise sampling.

Design
------
The manuscript's ~10^7 argument is about a MAXIMUM-over-training-set statistic;
that caveat is separate and survives.  What is estimable, and what this measures,
is the PER-PAIR operating characteristic of the flag rule Jaccard(8-mer) >= 0.7.

Stage 1 (enumerate + stratify).  Re-run the same blocked sparse product used by
expkit.max_sim_to_train over ALL test x train pairs on the two leaky datasets,
but instead of collapsing to a per-test maximum, tally the exact population count
of pairs in each Jaccard band and reservoir-sample pairs within each band:

    flagged   : [0.7,0.8) [0.8,0.9) [0.9,1.0]
    unflagged : [0.5,0.7) [0.3,0.5) [0.1,0.3)  and a random draw from [0,0.1)

Sampling every band (not just the flagged ones) is what makes RECALL estimable:
recall needs the homologs the detector MISSED, which live in the unflagged bands.
Horvitz-Thompson weights (band population / band sample) put every rate back on
the full pair population.

Stage 2 (adjudicate).  Smith-Waterman local alignment (Biopython PairwiseAligner)
on every sampled pair.  Ground truth = the community-standard CD-HIT-EST-like
criterion the benchmark's curators omitted: local alignment covering >= 50% of the
shorter sequence at >= 80% identity.  Two alternative cutoffs are carried through
as a sensitivity arm.

Outputs
-------
results/detector_specificity_pairs.csv      one row per adjudicated pair
results/detector_specificity_measured.csv   per-band and pooled precision/recall/FPR
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import expkit as E
from audit.core import homology_split as H

RESULTS = E.RESULTS
CACHE = os.path.join(os.path.dirname(RESULTS), "datacache")
SEED = 20240524
K = 8
FLAG = 0.7

# band edges, low -> high.  label, lo, hi(exclusive except the last)
BANDS = [
    ("j_lt_0p1",   0.0, 0.1),
    ("j_0p1_0p3",  0.1, 0.3),
    ("j_0p3_0p5",  0.3, 0.5),
    ("j_0p5_0p7",  0.5, 0.7),
    ("j_0p7_0p8",  0.7, 0.8),
    ("j_0p8_0p9",  0.8, 0.9),
    ("j_ge_0p9",   0.9, 1.01),
]
FLAGGED_BANDS = {"j_0p7_0p8", "j_0p8_0p9", "j_ge_0p9"}
# target sample sizes: 500 flagged split 3 ways, 500 unflagged split 4 ways
TARGET = {"j_0p7_0p8": 167, "j_0p8_0p9": 167, "j_ge_0p9": 166,
          "j_0p5_0p7": 200, "j_0p3_0p5": 200, "j_0p1_0p3": 400,
          # the sub-0.1 band is >99.9% of all pairs, so the recall denominator is
          # dominated by ITS sampling error, not by the flagged bands'.  Sample it
          # hard enough that the Clopper-Pearson upper bound is informative.
          "j_lt_0p1": 3000}


# ---------------------------------------------------------------------------
# stage 1: enumerate all test x train pairs, tally bands, reservoir-sample
# ---------------------------------------------------------------------------
class Reservoir:
    """Uniform sample of size n from a stream of unknown length (Algorithm R)."""
    def __init__(self, n, rng):
        self.n, self.rng, self.seen, self.buf = n, rng, 0, []

    def offer(self, items):
        for it in items:
            self.seen += 1
            if len(self.buf) < self.n:
                self.buf.append(it)
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.n:
                    self.buf[j] = it


def enumerate_bands(dset, block=400):
    seqs, y, otr, ote, _ = E.load(dset)
    M = H.kmer_binary_matrix(seqs, K)
    Mq, Mr = M[ote], M[otr]
    sq = np.asarray(Mq.sum(1)).ravel().astype(np.float32)
    sr = np.asarray(Mr.sum(1)).ravel().astype(np.float32)
    RT = Mr.T.tocsr()
    rng = np.random.default_rng(SEED)

    counts = {b[0]: 0 for b in BANDS}
    res = {b[0]: Reservoir(TARGET[b[0]], rng) for b in BANDS}
    n_te, n_tr = Mq.shape[0], Mr.shape[0]
    t0 = time.time()
    for s in range(0, n_te, block):
        e = min(s + block, n_te)
        inter = (Mq[s:e] @ RT).toarray()
        denom = sq[s:e, None] + sr[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, inter / denom, 0.0).astype(np.float32)
        # one pass: everything at/above 0.1 is rare, so locate it once
        hi_i, hi_j = np.where(sim >= 0.1)
        hv = sim[hi_i, hi_j]
        for lab, lo, hi in BANDS:
            if lab == "j_lt_0p1":
                # billions of pairs: count by subtraction, sample by random draw
                continue
            sel = (hv >= lo) & (hv < hi)
            counts[lab] += int(sel.sum())
            if sel.any():
                res[lab].offer([(int(ote[s + i]), int(otr[j]), float(v))
                                for i, j, v in zip(hi_i[sel], hi_j[sel], hv[sel])])
        # uniform draw for the low band, from this block only
        nlow_block = sim.size - len(hi_i)
        counts["j_lt_0p1"] += nlow_block
        if nlow_block:
            pick = min(400, nlow_block)
            fi = rng.integers(0, e - s, size=pick * 3)
            fj = rng.integers(0, n_tr, size=pick * 3)
            ok = sim[fi, fj] < 0.1
            fi, fj = fi[ok][:pick], fj[ok][:pick]
            # weight the offer by how many low pairs this block really held
            res["j_lt_0p1"].seen += max(0, nlow_block - len(fi))
            res["j_lt_0p1"].offer([(int(ote[s + i]), int(otr[j]), float(sim[i, j]))
                                   for i, j in zip(fi, fj)])
        if s % (block * 20) == 0:
            print(f"  {dset} {e}/{n_te} test rows  {time.time()-t0:.0f}s", flush=True)
    total = n_te * n_tr
    assert sum(counts.values()) == total, (sum(counts.values()), total)
    rows = []
    for lab, _, _ in BANDS:
        for (i, j, sim) in res[lab].buf:
            rows.append(dict(dataset=dset, band=lab, test_idx=i, train_idx=j,
                             jaccard8=sim,
                             flagged=int(sim >= FLAG),
                             test_label=int(y[i]), train_label=int(y[j])))
    pop = pd.DataFrame([dict(dataset=dset, band=lab, band_pairs=counts[lab],
                             n_test=n_te, n_train=n_tr, total_pairs=total)
                        for lab, _, _ in BANDS])
    return pd.DataFrame(rows), pop, seqs


# ---------------------------------------------------------------------------
# stage 2: Smith-Waterman adjudication
# ---------------------------------------------------------------------------
def make_aligner():
    from Bio.Align import PairwiseAligner
    a = PairwiseAligner()
    a.mode = "local"
    # EMBOSS water / blastn-like DNA scoring
    a.match_score = 5
    a.mismatch_score = -4
    a.open_gap_score = -10
    a.extend_gap_score = -0.5
    return a


def adjudicate(aligner, s1, s2):
    """Best local alignment -> (score, identity, coverage_of_shorter, aln_len).

    A local aligner returns an EMPTY alignment set when no positively-scoring local
    alignment exists, and indexing it raises IndexError rather than returning a
    zero-score hit. That is not an edge case here: `human_enhancers_ensembl` ships
    sequences as short as 2 bp, and a 2 bp query against a long training sequence can
    score below the gap-open penalty from every start position. Such a pair is a real
    observation -- no detectable local homology -- so it is scored 0 and kept in the
    denominator rather than dropped, which would bias every rate.
    """
    try:
        aln = aligner.align(s1, s2)[0]
    except (IndexError, OverflowError):
        return 0.0, 0.0, 0.0, 0
    a, b = aln[0], aln[1]
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    aln_len = len(a)
    ident = matches / aln_len if aln_len else 0.0
    cov = matches / min(len(s1), len(s2))
    return float(aln.score), ident, cov, aln_len


def run_dataset(dset):
    print(f"[{dset}] stage 1: enumerating all test x train pairs", flush=True)
    samp, pop, seqs = enumerate_bands(dset)
    print(f"[{dset}] stage 2: adjudicating {len(samp)} pairs by Smith-Waterman", flush=True)
    al = make_aligner()
    out = []
    t0 = time.time()
    for n, r in enumerate(samp.itertuples()):
        sc, ident, cov, alen = adjudicate(al, seqs[r.test_idx], seqs[r.train_idx])
        out.append(dict(sw_score=sc, sw_identity=ident, sw_coverage=cov, sw_aln_len=alen,
                        len_test=len(seqs[r.test_idx]), len_train=len(seqs[r.train_idx])))
        if n % 200 == 0:
            print(f"  {n}/{len(samp)}  {time.time()-t0:.0f}s", flush=True)
    samp = pd.concat([samp.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    # ground truth: three cutoffs, primary first
    samp["homolog"] = ((samp.sw_identity >= 0.80) & (samp.sw_coverage >= 0.50)).astype(int)
    samp["homolog_strict"] = ((samp.sw_identity >= 0.90) & (samp.sw_coverage >= 0.80)).astype(int)
    samp["homolog_loose"] = ((samp.sw_identity >= 0.70) & (samp.sw_coverage >= 0.30)).astype(int)
    return samp, pop


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval; returns (lo, hi).  Handles k=0 and k=n."""
    from scipy import stats
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


#: Minimum length, in bp, for BOTH sequences of a pair to be adjudicable.
#:
#: `human_enhancers_ensembl` ships rows as short as 2 bp. On those, an alignment-based
#: ground truth is not merely noisy, it is undefined in both directions at once: a 6 bp
#: exact match between a 7 bp row and a 300 bp row scores identity 1.00 and coverage
#: 0.86 and is called a homolog, though a 6 bp match is expected by chance between any
#: two genomic sequences; while a 15 bp BYTE-IDENTICAL pair -- a true duplicate, Jaccard
#: 1.0 -- is called a non-homolog by any criterion carrying a minimum alignment length.
#: Both errors are properties of the shipped rows, not of the detector, and each biases
#: a different rate. Pairs with a sequence below this length are therefore excluded from
#: the operating-characteristic estimate and counted separately, and the unrestricted
#: numbers are reported beside them so the size of the exclusion stays visible.
MIN_SEQ_LEN = 50


def summarise(pairs, pop, min_len=MIN_SEQ_LEN):
    """Horvitz-Thompson: reweight each band's sample by band_pairs / n_sampled.

    Every rate carries an interval built from the per-band Clopper-Pearson bounds,
    pushed through the same weighting in the direction that is WORST for the
    detector: the recall lower bound uses each unflagged band's UPPER homolog
    bound (more missed homologs) and each flagged band's LOWER bound.  Recall is
    dominated by the sub-0.1 band, which holds >99.9% of all pairs, so its bound
    is reported separately as `recall_lo_driver`.
    """
    rows = []
    for dset, P_all in pairs.groupby("dataset"):
        popd = pop[pop.dataset == dset].set_index("band")
        P = P_all[(P_all.len_test >= min_len) & (P_all.len_train >= min_len)]
        for gt in ("homolog", "homolog_strict", "homolog_loose"):
            per_band = {}
            for band, G in P.groupby("band"):
                # The band POPULATION has to be restricted the same way the sample is,
                # or the weights put the excluded pairs straight back. The sample is
                # uniform within a band, so the eligible share of the sample estimates
                # the eligible share of the band: a Horvitz-Thompson estimator with an
                # estimated stratum total.
                n_all = int((P_all.band == band).sum())
                N = float(popd.loc[band, "band_pairs"]) * (len(G) / n_all if n_all else 0.0)
                n = len(G)
                k = int(G[gt].sum())
                w = N / n if n else 0.0
                clo, chi = clopper_pearson(k, n)
                per_band[band] = dict(
                    n_sampled=n, band_pairs=N, weight=w, k=k,
                    p_homolog=(k / n) if n else np.nan, p_lo=clo, p_hi=chi,
                    W_homolog=k * w, W_lo=clo * N, W_hi=chi * N, W_total=N,
                    flagged=int(band in FLAGGED_BANDS),
                )
                if gt == "homolog":
                    rows.append(dict(dataset=dset, ground_truth=gt, band=band,
                                     flagged=int(band in FLAGGED_BANDS),
                                     n_sampled=n, n_homolog=k, band_pairs=int(N),
                                     frac_homolog=per_band[band]["p_homolog"],
                                     frac_homolog_lo=clo, frac_homolog_hi=chi,
                                     est_homolog_pairs=per_band[band]["W_homolog"]))
            F = [v for v in per_band.values() if v["flagged"]]
            U = [v for v in per_band.values() if not v["flagged"]]
            TP = sum(v["W_homolog"] for v in F)
            FP = sum(v["W_total"] - v["W_homolog"] for v in F)
            FN = sum(v["W_homolog"] for v in U)
            TN = sum(v["W_total"] - v["W_homolog"] for v in U)
            # worst-case-for-the-detector bounds
            TP_lo, TP_hi = sum(v["W_lo"] for v in F), sum(v["W_hi"] for v in F)
            FN_lo, FN_hi = sum(v["W_lo"] for v in U), sum(v["W_hi"] for v in U)
            FP_hi = sum(v["W_total"] - v["W_lo"] for v in F)
            TN_lo = sum(v["W_total"] - v["W_hi"] for v in U)
            low = per_band.get("j_lt_0p1", {})
            rows.append(dict(
                dataset=dset, ground_truth=gt, band="POOLED", flagged=-1,
                n_sampled=len(P), band_pairs=int(TP + FP + FN + TN),
                precision=TP / (TP + FP) if TP + FP else np.nan,
                precision_lo=TP_lo / (TP_lo + FP_hi) if TP_lo + FP_hi else np.nan,
                recall=TP / (TP + FN) if TP + FN else np.nan,
                recall_lo=TP_lo / (TP_lo + FN_hi) if TP_lo + FN_hi else np.nan,
                recall_hi=TP_hi / (TP_hi + FN_lo) if TP_hi + FN_lo else np.nan,
                fpr=FP / (FP + TN) if FP + TN else np.nan,
                fpr_hi=FP_hi / (FP_hi + TN_lo) if FP_hi + TN_lo else np.nan,
                est_TP=TP, est_FP=FP, est_FN=FN, est_TN=TN,
                est_homolog_pairs=TP + FN,
                # transparency: how much of the recall denominator is the sub-0.1
                # band, and what its own sampling bound alone permits
                recall_lo_driver=low.get("W_hi", np.nan),
                low_band_n=low.get("n_sampled", np.nan),
                low_band_k=low.get("k", np.nan),
                low_band_p_hi=low.get("p_hi", np.nan),
                # what the length-eligibility restriction cost, so it can be audited
                min_seq_len=min_len,
                n_sampled_all=len(P_all),
                n_sampled_eligible=len(P),
                frac_pairs_ineligible=round(1 - len(P) / len(P_all), 4) if len(P_all) else np.nan,
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    todo = sys.argv[1:] or list(E.LEAKY)
    allp, allpop = [], []
    for d in todo:
        p, pp = run_dataset(d)
        allp.append(p); allpop.append(pp)
        pd.concat(allp).to_csv(os.path.join(RESULTS, "detector_specificity_pairs.csv"), index=False)
        pd.concat(allpop).to_csv(os.path.join(RESULTS, "detector_specificity_bands.csv"), index=False)
    pairs = pd.concat(allp, ignore_index=True)
    pop = pd.concat(allpop, ignore_index=True)
    summ = summarise(pairs, pop)
    summ.to_csv(os.path.join(RESULTS, "detector_specificity_measured.csv"), index=False)
    print(summ.to_string())
