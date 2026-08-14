#!/usr/bin/env python3
"""
exp_hashfrag.py -- head-to-head against hashFrag, the published BLAST-based
homology-leakage tool, on all eight Genomic Benchmarks datasets.

WHY
---
hashFrag (Rafi et al.) is this work's closest methodological comparator and the only
one distributed as a runnable command-line tool. Three of the review's open items are
answered by the same run:

  * the head-to-head a novelty claim needs -- do an aligner-free 8-mer Jaccard census
    and a BLAST + Smith-Waterman-scored homology search return the same per-dataset
    verdicts, and where do they disagree per test sequence;
  * strand coverage -- hashFrag builds its BLAST database from BOTH orientations of
    every training sequence by default, so this is the suite-wide both-orientation
    run the forward-only census cannot supply. The reverse-strand hits are reported
    separately, since a hit found ONLY on the reverse strand is leakage the paper's
    forward-only census is blind to by construction;
  * an external corroboration of the detector's operating characteristic, alongside
    the Smith-Waterman adjudication in `exp_detector_specificity.py`.

THRESHOLD -- why this sweeps instead of picking one
---------------------------------------------------
hashFrag requires an alignment-score threshold and supplies no default. Its
documentation recommends scoring dinucleotide-shuffled sequences and choosing a
threshold above that null. We compute that null per dataset with a doublet-preserving
(Altschul-Erikson) shuffle -- but we do NOT report its verdict as *the* answer, because
this manuscript's own supplement already establishes why it cannot be: real genomic DNA
is far heavier-tailed than any shuffle, since genomes carry repeats and low-complexity
tracts that shuffling destroys, so a synthetic null puts the threshold too low. Run at
the shuffled-null threshold on `drosophila_enhancers_stark`, hashFrag calls 75% of the
test split homologous where the 8-mer census calls 1.6% -- a disagreement that is
almost entirely threshold choice, not a disagreement about the data.

So the comparison is made honest by sweeping: BLAST is run once per dataset, all
alignment scores are kept, and the leak fraction, the verdict and the agreement with
the 8-mer census are recomputed across the whole threshold range. What the sweep
answers is the question a fixed threshold cannot -- over what range of hashFrag
thresholds do the two tools return the same per-dataset verdicts, and is our operating
point inside it.

Outputs
-------
results/hashfrag_sweep.csv        per dataset x threshold: hashFrag leak fraction,
                                  verdict, 2x2 confusion against the 8-mer flags,
                                  Cohen's kappa, reverse-strand-only hits
results/hashfrag_comparison.csv   per dataset at the shuffled-null threshold, plus the
                                  threshold range over which the verdicts agree, plus
                                  wall-clock
results/hashfrag_pairs.csv        every scored pair, our Jaccard beside their score.
                                  NOT COMMITTED: 144 MB, above GitHub's file-size limit.
                                  Re-run this module to regenerate it; the summary tables
                                  the supplement cites are committed.
"""
from __future__ import annotations
import os, sys, shutil, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from audit.core import resources as _R          # noqa: F401  (BLAS caps; must precede numpy)
import numpy as np
import pandas as pd
from audit.core import expkit as E

SEED = 20240524
K = 8
OUR_FLAG = 0.7            # the census cut in sequence space
VERDICT_CUT = 0.1         # the report card's leak-fraction cut
NULL_N = 400              # sequences drawn for the null distribution
HF = os.environ.get(
    "HASHFRAG_BIN",
    "/private/tmp/claude-502/-Users-rikhinkavuru-homology-audit/"
    "36140e5b-a4d5-4d75-9940-323b9efa442f/scratchpad/hfvenv/bin/hashFrag",
)
WORK = os.environ.get(
    "HASHFRAG_WORK",
    "/private/tmp/claude-502/-Users-rikhinkavuru-homology-audit/"
    "36140e5b-a4d5-4d75-9940-323b9efa442f/scratchpad/hfwork",
)
THREADS = str(_R.N_JOBS)


# ---------------------------------------------------------------------------
# doublet-preserving shuffle (Altschul & Erikson 1985), for the null threshold
# ---------------------------------------------------------------------------
def dinuc_shuffle(seq, rng):
    """Uniform random sequence with the same dinucleotide counts as `seq`.

    Euler-path construction: the last outgoing edge of every vertex must lie on a
    spanning tree rooted at the final vertex, otherwise the traversal strands part of
    the graph. With a four-letter alphabet a rejection loop converges immediately.
    """
    if len(seq) < 3:
        return seq
    verts = sorted(set(seq))
    last = seq[-1]
    edges = {v: [] for v in verts}
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    for _ in range(100):
        chosen = {}
        for v in verts:
            if v != last:
                chosen[v] = edges[v][rng.integers(len(edges[v]))]
        # does following the chosen last-edges from every vertex reach `last`?
        ok = True
        for v in verts:
            if v == last:
                continue
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in chosen:
                    ok = False
                    break
                seen.add(cur)
                cur = chosen[cur]
            if not ok:
                break
        if ok:
            break
    else:
        return seq                                  # give up; keep the original
    rest = {}
    for v in verts:
        pool = list(edges[v])
        if v != last:
            pool.remove(chosen[v])
        pool = list(rng.permutation(pool))
        if v != last:
            pool.append(chosen[v])
        rest[v] = pool
    out, cur, pos = [seq[0]], seq[0], {v: 0 for v in verts}
    for _ in range(len(seq) - 1):
        nxt = rest[cur][pos[cur]]
        pos[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def write_fasta(path, ids, seqs):
    with open(path, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n{s}\n")


def blast_pairs(fa_a, fa_b, outdir, tag, threads=THREADS):
    """Raw blastn of fa_b against a database built from fa_a; returns scores.

    Uses hashFrag's own BLAST parameters and its corrected-score formula, so the null
    is on exactly the scale the threshold is compared against.
    """
    db = os.path.join(outdir, f"{tag}_db")
    subprocess.run(["makeblastdb", "-in", fa_a, "-dbtype", "nucl", "-out", db],
                   check=True, capture_output=True)
    fmt = ("6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send "
           "evalue bitscore score positive gaps")
    out = os.path.join(outdir, f"{tag}.blastn.out")
    subprocess.run(["blastn", "-query", fa_b, "-db", db, "-outfmt", fmt,
                    "-word_size", "11", "-gapopen", "2", "-gapextend", "1",
                    "-penalty", "-1", "-reward", "1", "-evalue", "10",
                    "-dust", "no", "-max_target_seqs", "500",
                    "-num_threads", threads, "-out", out],
                   check=True, capture_output=True)
    cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart",
            "qend", "sstart", "send", "evalue", "bitscore", "score", "positive", "gaps"]
    if os.path.getsize(out) == 0:
        return pd.DataFrame(columns=cols + ["corrected"])
    df = pd.read_csv(out, sep="\t", names=cols)
    df["corrected"] = (1 * df["positive"] - 1 * df["mismatch"]
                       - 2 * df["gapopen"] - 1 * (df["gaps"] - df["gapopen"]))
    return df


def null_threshold(seqs, outdir, rng):
    """Highest corrected alignment score between dinucleotide-shuffled sequences."""
    pick = rng.choice(len(seqs), size=min(NULL_N, len(seqs)), replace=False)
    sh = [dinuc_shuffle(seqs[i], rng) for i in pick]
    fa = os.path.join(outdir, "null.fa")
    write_fasta(fa, [f"n{i}" for i in range(len(sh))], sh)
    df = blast_pairs(fa, fa, outdir, "null")
    df = df[df.qseqid != df.sseqid]
    if df.empty:
        return 0.0, 0.0, 0
    return float(df["corrected"].max()), float(df["corrected"].quantile(0.999)), len(df)


# ---------------------------------------------------------------------------
def run_dataset(dset, rng):
    outdir = os.path.join(WORK, dset)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    seqs, y, otr, ote, _ = E.load(dset)
    tr_fa = os.path.join(outdir, "train.fa")
    te_fa = os.path.join(outdir, "test.fa")
    write_fasta(tr_fa, [f"tr{i}" for i in otr], [seqs[i] for i in otr])
    write_fasta(te_fa, [f"te{i}" for i in ote], [seqs[i] for i in ote])

    thr_max, thr_p999, n_null = null_threshold(seqs, outdir, rng)
    # hashFrag's CLI takes an integer score; round up so the threshold stays strictly
    # above the null maximum rather than tying with it.
    threshold = int(np.ceil(thr_max))
    print(f"[{dset}] null: max={thr_max} p99.9={thr_p999} over {n_null} shuffled pairs "
          f"-> shuffled-null threshold={threshold}", flush=True)

    # Run the pipeline once at a floor far below any sensible operating point and
    # re-threshold offline, so one BLAST run serves the whole sweep and every arm is
    # scored on identical alignments. The floor is 20 rather than 1 because a score of
    # 20 is already a 20 bp exact match, which is chance-level between any two genomic
    # sequences, and keeping everything above 1 makes the hits file enormous on the
    # 154k-sequence datasets for no information. `-m 100` caps hits per query, which
    # cannot affect a per-query MAXIMUM since BLAST returns best hits first.
    t0 = time.time()
    subprocess.run([HF, "filter_existing_splits",
                    "--train-fasta-path", tr_fa, "--test-fasta-path", te_fa,
                    "-t", "20", "-m", "100", "-o", outdir, "-T", THREADS, "--force"],
                   check=True, capture_output=True)
    wall_hf = time.time() - t0

    hits_path = os.path.join(outdir, "hashFrag.similar_pairs.tsv")
    if os.path.getsize(hits_path) if os.path.exists(hits_path) else 0:
        hits = pd.read_csv(hits_path, sep="\t", names=["id_i", "id_j", "score"])
    else:
        hits = pd.DataFrame(columns=["id_i", "id_j", "score"])

    # hashFrag queries test against a train database carrying both orientations;
    # reverse-strand database entries are suffixed "_Reversed".
    def norm(col):
        rev = col.str.endswith("_Reversed")
        return col.str.replace("_Reversed", "", regex=False), rev

    if len(hits):
        a, arev = norm(hits.id_i)
        b, brev = norm(hits.id_j)
        # orient each row as (test, train); a query id starts with "te"
        q_is_i = a.str.startswith("te")
        hits["test_id"] = np.where(q_is_i, a, b)
        hits["train_id"] = np.where(q_is_i, b, a)
        hits["reversed"] = (arev | brev).astype(int)
        hits = hits[hits.test_id.str.startswith("te") & hits.train_id.str.startswith("tr")]
    else:
        hits = hits.assign(test_id=pd.Series(dtype=str), train_id=pd.Series(dtype=str),
                           reversed=pd.Series(dtype=int))

    test_ids = [f"te{i}" for i in ote]
    idx_of = {tid: n for n, tid in enumerate(test_ids)}
    n = len(test_ids)

    t1 = time.time()
    ours_sim = E.cached_max_sim(dset, seqs, otr, ote, K)
    wall_ours = time.time() - t1                    # cached: reported, not comparable
    ours = ours_sim > OUR_FLAG
    lf_ours = float(ours.mean())
    v_ours = "LEAKY" if lf_ours > VERDICT_CUT else "clean"

    # per-test-sequence maximum hashFrag score, forward-only and both-orientation
    best = np.zeros(n, dtype=float)
    best_fwd = np.zeros(n, dtype=float)
    if len(hits):
        pos = hits.test_id.map(idx_of).to_numpy()
        sc = hits.score.to_numpy(dtype=float)
        np.maximum.at(best, pos, sc)
        f = hits["reversed"].to_numpy() == 0
        if f.any():
            np.maximum.at(best_fwd, pos[f], sc[f])

    grid = sorted({int(t) for t in
                   list(np.arange(20, 105, 5)) + list(np.arange(120, 420, 20))
                   + [max(20, threshold)]})
    sweep = []
    for t in grid:
        hf = best >= t
        hf_f = best_fwd >= t
        tp = int((ours & hf).sum()); fp = int((ours & ~hf).sum())
        fn = int((~ours & hf).sum()); tn = int((~ours & ~hf).sum())
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        lf_hf = float(hf.mean())
        sweep.append(dict(
            dataset=dset, threshold=t, is_shuffled_null_threshold=int(t == threshold),
            n_test=n, n_train=len(otr),
            leak_ours=round(lf_ours, 4), verdict_ours=v_ours,
            leak_hashfrag=round(lf_hf, 4),
            leak_hashfrag_fwd_only=round(float(hf_f.mean()), 4),
            rev_only_hits=int((hf & ~hf_f).sum()),
            rev_only_frac=round(float((hf & ~hf_f).mean()), 4),
            verdict_hashfrag="LEAKY" if lf_hf > VERDICT_CUT else "clean",
            verdicts_agree=int((lf_ours > VERDICT_CUT) == (lf_hf > VERDICT_CUT)),
            both=tp, ours_only=fp, hashfrag_only=fn, neither=tn,
            agreement=round(po, 4),
            kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else np.nan,
            recall_vs_hashfrag=round(tp / (tp + fn), 4) if tp + fn else np.nan,
            precision_vs_hashfrag=round(tp / (tp + fp), 4) if tp + fp else np.nan,
        ))
    sw = pd.DataFrame(sweep)

    agree = sw[sw.verdicts_agree == 1]
    at_null = sw[sw.threshold == threshold].iloc[0]
    best_k = sw.loc[sw.kappa.idxmax()] if sw.kappa.notna().any() else at_null
    row = dict(
        dataset=dset, n_test=n, n_train=len(otr),
        null_threshold=threshold, null_max=thr_max, null_p999=thr_p999, null_pairs=n_null,
        leak_ours=round(lf_ours, 4), verdict_ours=v_ours,
        leak_hf_at_null=at_null.leak_hashfrag, verdict_hf_at_null=at_null.verdict_hashfrag,
        agree_at_null=int(at_null.verdicts_agree),
        agree_threshold_lo=int(agree.threshold.min()) if len(agree) else np.nan,
        agree_threshold_hi=int(agree.threshold.max()) if len(agree) else np.nan,
        n_thresholds_agreeing=len(agree), n_thresholds=len(sw),
        best_kappa=float(best_k.kappa), best_kappa_threshold=int(best_k.threshold),
        rev_only_frac_at_best=float(best_k.rev_only_frac),
        wall_hashfrag_s=round(wall_hf, 1), wall_ours_cached_s=round(wall_ours, 1),
        n_scored_pairs=len(hits),
    )
    pairs = hits.assign(dataset=dset)
    return row, sw, pairs


if __name__ == "__main__":
    todo = sys.argv[1:] or (list(E.LEAKY) + list(E.CLEAN) + [E.THREECLASS])
    os.makedirs(WORK, exist_ok=True)
    print(_R.describe(), flush=True)
    rng = np.random.default_rng(SEED)
    rows, allsweeps, allpairs = [], [], []
    for d in todo:
        try:
            r, sw, p = run_dataset(d, rng)
        except subprocess.CalledProcessError as exc:
            print(f"[{d}] FAILED: {exc.stderr.decode()[-2000:]}", flush=True)
            continue
        rows.append(r); allsweeps.append(sw); allpairs.append(p)
        pd.DataFrame(rows).to_csv(os.path.join(E.RESULTS, "hashfrag_comparison.csv"), index=False)
        pd.concat(allsweeps).to_csv(os.path.join(E.RESULTS, "hashfrag_sweep.csv"), index=False)
        pd.concat(allpairs).to_csv(os.path.join(E.RESULTS, "hashfrag_pairs.csv"), index=False)
        print(f"[{d}] ours={r['leak_ours']} ({r['verdict_ours']})  "
              f"hashFrag@null({r['null_threshold']})={r['leak_hf_at_null']} "
              f"({r['verdict_hf_at_null']})  verdicts agree over "
              f"[{r['agree_threshold_lo']},{r['agree_threshold_hi']}] "
              f"({r['n_thresholds_agreeing']}/{r['n_thresholds']} of the grid)  "
              f"best kappa={r['best_kappa']} @{r['best_kappa_threshold']}  "
              f"{r['wall_hashfrag_s']}s", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no dataset completed -- see the FAILED lines above")
    print("\n" + df.to_string(index=False))
    print(f"\nverdicts agree at the shuffled-null threshold: "
          f"{int(df.agree_at_null.sum())}/{len(df)} datasets")
    print("EXP_HASHFRAG_DONE")
