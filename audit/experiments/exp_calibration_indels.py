#!/usr/bin/env python3
"""
Detector calibration, part 2: INDELS and REPEAT-DRIVEN SIMILARITY (reviewer item C7).

The calibration in `exp_estimator_sensitivity.py` (Supplementary Table `tab:sens`) covers
substitutions and window shifts only. Both are benign for a k-mer set metric in a specific
way, and the omission is not cosmetic:

  * a SUBSTITUTION at position p destroys at most k k-mers -- the k windows that span p --
    and leaves every other k-mer in the sequence byte-identical and in place;
  * a WINDOW SHIFT of s bp deletes s k-mers from one end and adds s at the other, and
    leaves the interior k-mer SET untouched (a k-mer set is order-free, so sliding the
    frame does not perturb the k-mers that remain inside the window);
  * an INDEL of length d at position p also destroys only ~k k-mers locally -- BUT it
    shifts the reading frame of every downstream base by d. Whether that matters depends
    entirely on whether the metric is order-aware. A k-mer SET is not. This module
    measures the consequence instead of assuming it in either direction.

That last point is the substantive question this module exists to answer, and the answer
is not obvious a priori: an alignment-based detector would be hurt badly by an indel and
barely at all by a substitution, so if the paper's detector behaves the other way round,
that is a property of the estimator that a reader is entitled to see stated.

Two arms:

(A) INDELS. Insertions and deletions of 1, 2, 5 and 10 bp at a uniformly random position,
    at each of 70/200/300/500 bp. Detection fraction at Jaccard > 0.7 AND at containment
    > 0.7, >= 200 replicates per cell, each with a Clopper-Pearson 95% interval.

(B) REPEAT-DRIVEN SIMILARITY. The manuscript reports the leaked stratum of
    `human_nontata_promoters` as ~7x AluY-enriched, so repeat content is present in these
    data and is a candidate driver of false similarity between NON-homologous sequences.
    The measured per-pair false-positive rate the paper reports is 0.000, and that number
    is only as good as the pairs it was measured over: if repeat-rich pairs are rare in a
    uniform draw, a uniform draw will not find the false positives even if they exist.
    This arm draws pairs of REAL sequences from datasets this paper certifies CLEAN and
    splits them by repeat content, so the repeat-rich stratum is measured on its own.

    *** PROXY DISCLOSURE -- READ THIS BEFORE QUOTING ANY NUMBER FROM ARM (B). ***
    RepeatMasker is NOT installed in this environment and no repeat library (Dfam/RepBase)
    is available offline; obtaining interspersed-repeat annotations would require
    downloading a genome and a repeat library, which we do not do. Arm (B) therefore uses
    `dustmasker` (BLAST+ 2.17.0) low-complexity masked fraction per sequence, thresholded,
    as a PROXY for repeat content. This proxy is genuinely weaker than the thing it stands
    in for, in a direction that matters:

        DUST detects COMPOSITIONALLY BIASED tracts -- homopolymers, microsatellites, simple
        repeats. It does NOT detect interspersed repeats. An AluY element is ~300 bp of
        ordinary-composition sequence and is largely INVISIBLE to DUST. The manuscript's
        own supplementary already says this ("silent on interspersed repeats").

    So arm (B) bounds the SIMPLE-repeat / low-complexity contribution to false similarity,
    and does not bound the Alu/SINE contribution. It is a partial control, and is reported
    as one. Where the Alu question is answerable at all with what is on disk, arm (B) adds
    a direct k-mer-composition screen (`repeatish`: max single-k-mer share + distinct-k-mer
    deficit) as a second, RepeatMasker-free stratification, which does have some purchase
    on interspersed elements because members of one repeat family share k-mers with each
    other. Neither proxy is RepeatMasker and neither is presented as RepeatMasker.

Conventions are inherited EXACTLY from exp_estimator_sensitivity.py and are not re-chosen
here: k = 8, threshold 0.7, detection is a STRICT inequality (score > 0.7, not >=),
lengths 70/200/300/500, uniform-random ACGT synthetic ground truth, and a single
`np.random.RandomState(0)` stream drawn in a fixed order.

Run:  AUDIT_N_JOBS=2 AUDIT_NO_PARENT_WATCH=1 python -m audit.experiments.exp_calibration_indels
Out:  results/calibration_indels.csv, results/calibration_repeats.csv
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

R = os.path.join(HERE, "results")

# ---- inherited verbatim from exp_estimator_sensitivity.py ----
K = 8               # the paper's k-mer size
THR = 0.7           # the paper's threshold
SEED = 0

#: The four lengths in Supplementary Table tab:sens. Deliberately the same four, so the
#: indel rows can be read directly against the substitution rows already published.
LENGTHS = [
    (70,   "GUE core-promoter"),
    (200,  "Genomic Benchmarks nontata (251 bp)"),
    (300,  "GUE promoter-300"),
    (500,  "GUE yeast EMP"),
]

#: Indel sizes. 1/2/5 line up one-for-one with the published 1/2/5 substitution rows, so
#: "is an indel worse than the same number of substitutions?" is read off the same grid.
INDEL_SIZES = (1, 2, 5, 10)


# ---------------------------------------------------------------------------
# metrics -- copied unchanged from exp_estimator_sensitivity.py
# ---------------------------------------------------------------------------
def kmers(s, k=K):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def jaccard(a, b, k=K):
    ka, kb = kmers(a, k), kmers(b, k)
    return len(ka & kb) / len(ka | kb) if (ka or kb) else 0.0


def containment(a, b, k=K):
    """The length-robust complement: |A n B| / min(|A|,|B|)."""
    ka, kb = kmers(a, k), kmers(b, k)
    return len(ka & kb) / min(len(ka), len(kb)) if (ka and kb) else 0.0


def rand_seq(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def mutate(rng, s, n_snp):
    """n_snp point substitutions at distinct uniformly chosen positions.

    Reproduced here so the substitution baseline in this file is computed by the SAME
    code path as the indel arm, rather than being read across from a stored CSV.
    """
    pos = rng.choice(len(s), n_snp, replace=False)
    out = list(s)
    for p in pos:
        out[p] = rng.choice([c for c in "ACGT" if c != out[p]])
    return "".join(out)


# ---------------------------------------------------------------------------
# (A) indel perturbations
# ---------------------------------------------------------------------------
def insertion(rng, s, d):
    """Insert `d` uniformly random bases at a uniformly random position.

    The result is len(s) + d long. We do NOT trim back to len(s): trimming would silently
    fold a window shift into the perturbation, and the window shift is already calibrated
    separately. The length change is part of what an indel is, and containment (which
    divides by the SMALLER k-mer set) is the metric that is supposed to absorb it.
    """
    p = int(rng.randint(0, len(s) + 1))
    return s[:p] + rand_seq(rng, d) + s[p:]


def deletion(rng, s, d):
    """Delete `d` contiguous bases starting at a uniformly random position.

    Position is drawn so the deleted block lies wholly inside the sequence.
    """
    p = int(rng.randint(0, len(s) - d + 1))
    return s[:p] + s[p + d:]


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval; returns (lo, hi). Handles k=0 and k=n.

    Same estimator and same alpha as exp_detector_specificity.clopper_pearson.
    """
    from scipy import stats
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def run_indels(n_trials=500):
    """Detection fraction for indels, with matched substitution controls.

    `n_trials` is the replicate count per cell; the task floor is 200 and the default here
    is 500, which puts the Clopper-Pearson half-width under ~0.045 at p = 0.5.
    """
    rng = np.random.RandomState(SEED)
    rows = []

    for L, where in LENGTHS:
        base = [rand_seq(rng, L) for _ in range(n_trials)]

        cases = [("exact copy", "control", 0, lambda s: s)]
        for d in INDEL_SIZES:
            if d >= L:
                continue
            cases.append((f"insertion {d} bp", "insertion", d,
                          (lambda d_: lambda s: insertion(rng, s, d_))(d)))
            cases.append((f"deletion {d} bp", "deletion", d,
                          (lambda d_: lambda s: deletion(rng, s, d_))(d)))
        # matched substitution controls, recomputed here rather than cited across, so the
        # indel-vs-substitution comparison is within-file and within-seed
        for n in INDEL_SIZES:
            if n < L:
                cases.append((f"{n} point mutation" + ("s" if n > 1 else ""),
                              "substitution", n,
                              (lambda n_: lambda s: mutate(rng, s, n_))(n)))

        for label, kind, mag, mk in cases:
            pert = [mk(s) for s in base]
            j = np.array([jaccard(s, p) for s, p in zip(base, pert)])
            c = np.array([containment(s, p) for s, p in zip(base, pert)])
            jk, ck = int((j > THR).sum()), int((c > THR).sum())
            jlo, jhi = clopper_pearson(jk, n_trials)
            clo, chi = clopper_pearson(ck, n_trials)
            rows.append(dict(
                length=L, regime=where, perturbation=label, kind=kind, magnitude=mag,
                n_trials=n_trials,
                jaccard_median=round(float(np.median(j)), 4),
                jaccard_detect_rate=round(jk / n_trials, 4),
                jaccard_detect_lo=round(jlo, 4), jaccard_detect_hi=round(jhi, 4),
                containment_median=round(float(np.median(c)), 4),
                containment_detect_rate=round(ck / n_trials, 4),
                containment_detect_lo=round(clo, 4), containment_detect_hi=round(chi, 4)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (B) repeat-driven similarity on REAL sequence
# ---------------------------------------------------------------------------
# See exp_alignment.py / exp_repeat.py: scratch dir is configurable and created lazily.
TMP = os.path.join(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir()),
                   "homology_audit", "dust_calib")


def dust_masked_fraction(seqs, tag):
    """Per-sequence fraction of bases DUST masks as low-complexity.

    Same invocation as exp_repeat.py: dustmasker writes soft-masked FASTA, lowercase =
    masked. This is the repeat PROXY -- see the module docstring. It sees simple repeats
    and misses interspersed ones.
    """
    os.makedirs(TMP, exist_ok=True)
    fa = os.path.join(TMP, f"{tag}.fasta")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    out = os.path.join(TMP, f"{tag}.masked")
    subprocess.run(["dustmasker", "-in", fa, "-out", out, "-outfmt", "fasta"],
                   check=True, capture_output=True)
    frac, cur = [], []

    def flush():
        if cur:
            s = "".join(cur)
            frac.append(sum(1 for ch in s if ch.islower()) / len(s) if s else 0.0)

    with open(out) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                cur = []
            else:
                cur.append(line.strip())
    flush()
    return np.array(frac)


def repeatish_score(s, k=K):
    """A dustmasker-free, RepeatMasker-free internal-repetition score in [0, 1].

    DUST is blind to interspersed repeats (module docstring). This second score is a crude
    complement: a sequence carrying a tandem or internally duplicated element re-uses its
    own k-mers, so its DISTINCT k-mer count falls below the number of k-mer windows. We
    score 1 - (distinct k-mers / k-mer windows), which is 0 for a sequence whose every
    window is unique and rises with internal duplication.

    This is NOT a repeat annotation. It cannot name a family, and a single dispersed AluY
    copy inside one sequence leaves it near 0. It is reported because it stratifies on a
    different axis from DUST, not because it is a substitute for RepeatMasker.
    """
    n = len(s) - k + 1
    if n <= 0:
        return 0.0
    return 1.0 - len(kmers(s, k)) / n


def run_repeats(cap=6000, n_pairs=20000, dust_hi=0.10):
    """Jaccard distribution over pairs of real sequences, split by repeat proxy.

    Pairs are drawn from datasets this paper certifies CLEAN, exactly as
    exp_estimator_sensitivity.real_dna_floor does, and for the same reason: on a clean
    dataset, similarity between two random members is by that verdict NOT leakage, so any
    pair above 0.7 is a false positive of the detector rather than a missed duplicate.

    `dust_hi` is the DUST masked-fraction cut separating "repeat-rich" from "repeat-poor".
    The manuscript quotes leaked/novel DUST fractions in the 0.03-0.11 range, so 0.10 sits
    at the top of the range the manuscript itself works in and yields a non-trivial
    high-complexity stratum on all three datasets.
    """
    from audit.core import expkit as E
    rng = np.random.RandomState(SEED)
    rows, examples = [], []

    for d in ("human_ocr_ensembl", "drosophila_enhancers_stark", "human_enhancers_cohn"):
        seqs, _y, _tr, _te, _tf = E.load(d, full=False, cap=cap)
        t0 = time.time()
        dust = dust_masked_fraction(seqs, f"{d}_calib")
        rep = np.array([repeatish_score(s) for s in seqs])
        hi_dust = dust > dust_hi
        # `repeatish` is near 0 for most real sequence, so an absolute cut would put every
        # sequence in one bin. Split at the dataset's own upper decile instead, which asks
        # the comparative question ("the most internally repetitive tenth of THIS dataset")
        # rather than an absolute one the score is not calibrated to answer.
        rep_cut = float(np.quantile(rep, 0.90))
        hi_rep = rep > rep_cut

        idx = rng.choice(len(seqs), (n_pairs, 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        v = np.array([jaccard(seqs[i], seqs[j]) for i, j in idx])

        for proxy, flag, cutdesc in (("dust_lowcomplexity", hi_dust, f">{dust_hi}"),
                                     ("kmer_repeatish", hi_rep, f">p90={rep_cut:.4f}")):
            fi, fj = flag[idx[:, 0]], flag[idx[:, 1]]
            strata = {"high-high": fi & fj,
                      "high-low": fi ^ fj,
                      "low-low": ~fi & ~fj}
            # "high-low" in the task's sense = at least one repeat-rich member
            strata["any-high"] = fi | fj
            for name, m in strata.items():
                if not m.any():
                    continue
                vv = v[m]
                nflag = int((vv > THR).sum())
                lo, hi = clopper_pearson(nflag, len(vv))
                rows.append(dict(
                    dataset=d, proxy=proxy, cut=cutdesc, stratum=name,
                    n_seqs=len(seqs), n_seqs_high=int(flag.sum()),
                    frac_seqs_high=round(float(flag.mean()), 4),
                    len_med=int(np.median([len(x) for x in seqs])),
                    n_pairs=int(len(vv)),
                    jaccard_median=round(float(np.median(vv)), 6),
                    jaccard_mean=round(float(vv.mean()), 6),
                    jaccard_p99=round(float(np.percentile(vv, 99)), 6),
                    jaccard_max=round(float(vv.max()), 6),
                    n_pairs_over_0p7=nflag,
                    frac_pairs_over_0p7=round(nflag / len(vv), 8),
                    frac_over_0p7_hi=round(hi, 8),
                    n_pairs_over_0p3=int((vv > 0.3).sum()),
                    n_pairs_over_0p5=int((vv > 0.5).sum())))
        # keep the worst real pair seen, so the tail is auditable rather than just quoted
        w = int(np.argmax(v))
        examples.append(dict(dataset=d, worst_jaccard=round(float(v[w]), 6),
                             i=int(idx[w, 0]), j=int(idx[w, 1]),
                             dust_i=round(float(dust[idx[w, 0]]), 4),
                             dust_j=round(float(dust[idx[w, 1]]), 4),
                             rep_i=round(float(rep[idx[w, 0]]), 4),
                             rep_j=round(float(rep[idx[w, 1]]), 4)))
        print(f"[{d}] {len(seqs)} seqs, {len(idx)} pairs, dust>{dust_hi} on "
              f"{hi_dust.mean():.3f}, max J={v.max():.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    return pd.DataFrame(rows), pd.DataFrame(examples)


def run_repeats_exhaustive(cap=6000):
    """EVERY within-dataset pair, not a uniform sample, then adjudicate the flagged ones.

    Why this arm exists. `exp_estimator_sensitivity.real_dna_floor` estimates the real-DNA
    false-positive rate from 4,000 uniformly drawn pairs per dataset and reports 0.000. A
    rate of 0.000 from 4,000 draws only excludes rates above ~7e-4; the density of
    near-duplicate pairs in these datasets is ~1e-6, so a 4,000-pair draw CANNOT see them
    and would have returned 0.000 whether or not they exist. The sample size, not the
    detector, produced the zero.

    The pair count here is small enough to settle it outright: 6,000 sequences is
    ~18M pairs per dataset, one blocked sparse product, so we enumerate all of them, take
    every pair above the 0.7 flag, and adjudicate each by the same Smith-Waterman criterion
    `exp_detector_specificity` uses. That converts "no false positives were sampled" into
    "no false positives exist among 54M pairs", which is a different and much stronger
    statement -- and it is the statement the repeat question actually needs, since a
    repeat-driven false positive would be a rare-tail event by construction.

    Note this measures WITHIN-dataset pairs, including train-train and test-test, so it is
    not a leak fraction and does not restate one. The paper's own leakage report card
    already records these datasets at leak 0.010 and 0.016 at threshold 0.7 -- i.e. `clean`
    is a verdict about being under the 0.1 cut, not a claim of zero duplicate pairs. That
    matters for how the null is read: pairs drawn from a clean dataset are NOT guaranteed
    non-homologous, so the >0.7 rate over them is an upper bound on the false-positive
    rate, and only adjudication separates the two.
    """
    from audit.core import expkit as E
    from audit.core import homology_split as H
    from audit.experiments.exp_detector_specificity import make_aligner, adjudicate
    al = make_aligner()
    rows = []
    for d in ("human_ocr_ensembl", "drosophila_enhancers_stark", "human_enhancers_cohn"):
        t0 = time.time()
        seqs, y, otr, ote, _ = E.load(d, full=False, cap=cap)
        tr, te = set(otr.tolist()), set(ote.tolist())
        M = H.kmer_binary_matrix(seqs, K)
        sums = np.asarray(M.sum(1)).ravel().astype(np.float32)
        n = M.shape[0]
        RT = M.T.tocsr()
        hits, n05 = [], 0
        for a in range(0, n, 500):
            b = min(a + 500, n)
            inter = (M[a:b] @ RT).toarray()
            den = sums[a:b, None] + sums[None, :] - inter
            sim = np.where(den > 0, inter / den, 0.0)
            for r in range(b - a):
                sim[r, a + r] = 0.0          # drop the diagonal (self-pairs)
            n05 += int((sim > 0.5).sum()) // 2
            ii, jj = np.where(sim > THR)
            for i, j in zip(ii, jj):
                if a + i < j:                # upper triangle only, no double count
                    hits.append((a + int(i), int(j), float(sim[i, j])))
        total = n * (n - 1) // 2
        dust = dust_masked_fraction(seqs, f"{d}_exh")
        n_hom = 0
        n_dusty = 0
        for i, j, v in hits:
            _sc, ident, cov, _al = adjudicate(al, seqs[i], seqs[j])
            if ident >= 0.80 and cov >= 0.50:
                n_hom += 1
            if max(dust[i], dust[j]) > dust_hi_default():
                n_dusty += 1
        cross = sum(1 for i, j, _v in hits
                    if (i in tr and j in te) or (i in te and j in tr))
        fp = len(hits) - n_hom
        lo, hi = clopper_pearson(fp, total)
        rows.append(dict(
            dataset=d, proxy="EXHAUSTIVE_all_pairs", cut="-", stratum="ALL",
            n_seqs=n, n_seqs_high=int((dust > dust_hi_default()).sum()),
            frac_seqs_high=round(float((dust > dust_hi_default()).mean()), 4),
            len_med=int(np.median([len(x) for x in seqs])),
            n_pairs=total,
            jaccard_median=np.nan, jaccard_mean=np.nan, jaccard_p99=np.nan,
            jaccard_max=round(max([v for _i, _j, v in hits], default=0.0), 6),
            n_pairs_over_0p7=len(hits),
            frac_pairs_over_0p7=round(len(hits) / total, 10),
            frac_over_0p7_hi=np.nan,
            n_pairs_over_0p3=np.nan, n_pairs_over_0p5=n05,
            # what the flagged pairs actually ARE, once adjudicated
            n_flagged_homologous_sw=n_hom,
            n_flagged_false_positive=fp,
            false_positive_rate=round(fp / total, 10),
            false_positive_rate_hi=round(hi, 10),
            n_flagged_cross_split=cross,
            n_flagged_repeat_rich=n_dusty))
        print(f"[{d}] EXHAUSTIVE {total:,} pairs: {len(hits)} flagged >0.7, "
              f"{n_hom} homologous by SW, {fp} false positives, {n_dusty} repeat-rich, "
              f"{cross} cross-split ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def dust_hi_default():
    return 0.10


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--skip-repeats", action="store_true")
    ap.add_argument("--skip-exhaustive", action="store_true")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)

    ind = run_indels(a.trials)
    tmp = f"{R}/calibration_indels.csv.tmp"
    ind.to_csv(tmp, index=False)
    os.replace(tmp, f"{R}/calibration_indels.csv")

    print("== (A) INDEL SENSITIVITY: detection rate at Jaccard > 0.7 ==")
    print(ind.pivot_table(index=["kind", "perturbation"], columns="length",
                          values="jaccard_detect_rate", sort=False).to_string())
    print("\n== (A) INDEL SENSITIVITY: detection rate at containment > 0.7 ==")
    print(ind.pivot_table(index=["kind", "perturbation"], columns="length",
                          values="containment_detect_rate", sort=False).to_string())

    print("\n== (A) indel vs the SAME number of substitutions, Jaccard ==")
    for L, _ in LENGTHS:
        s = ind[ind.length == L]
        for m in INDEL_SIZES:
            g = s[s.magnitude == m]
            sub = g[g.kind == "substitution"].jaccard_detect_rate
            ins = g[g.kind == "insertion"].jaccard_detect_rate
            dele = g[g.kind == "deletion"].jaccard_detect_rate
            if not len(sub) or not len(ins):
                continue
            print(f"  {L:4d} bp  n={m:2d}   sub={sub.iloc[0]:.3f}  "
                  f"ins={ins.iloc[0]:.3f}  del={dele.iloc[0]:.3f}")

    if not a.skip_repeats:
        rep, ex = run_repeats(n_pairs=a.pairs)
        print("\n== (B) REPEAT PROXY (dustmasker low-complexity -- NOT RepeatMasker) ==")
        cols = ["dataset", "proxy", "stratum", "n_pairs", "jaccard_median",
                "jaccard_p99", "jaccard_max", "n_pairs_over_0p7", "frac_over_0p7_hi"]
        print(rep[cols].to_string(index=False))
        print("\n== (B) worst real pair per dataset ==")
        print(ex.to_string(index=False))

        if not a.skip_exhaustive:
            exh = run_repeats_exhaustive()
            rep = pd.concat([rep, exh], ignore_index=True)
            print("\n== (B) EXHAUSTIVE all-pairs scan + Smith-Waterman adjudication ==")
            print(exh[["dataset", "n_pairs", "n_pairs_over_0p7",
                       "n_flagged_homologous_sw", "n_flagged_false_positive",
                       "false_positive_rate_hi", "n_flagged_repeat_rich",
                       "jaccard_max"]].to_string(index=False))

        tmp = f"{R}/calibration_repeats.csv.tmp"
        rep.to_csv(tmp, index=False)
        os.replace(tmp, f"{R}/calibration_repeats.csv")

    print("\nEXP_CALIBRATION_INDELS_DONE")


if __name__ == "__main__":
    main()
