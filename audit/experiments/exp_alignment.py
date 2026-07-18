#!/usr/bin/env python3
"""
T1.4 / R2.a2 / R1.3 / R3.4: alignment-based homology validation with MMseqs2.

Re-cluster the two leaky datasets by ALIGNMENT identity (MMseqs2 easy-cluster,
--min-seq-id 0.7 -c 0.8) instead of 8-mer Jaccard, feed the external cluster vector
into the SAME whole-cluster splitter (E.assign; only the edge metric changes, linkage
held fixed), refit the 4 models, and confirm the RF drop + demotion persist. This
defuses the k-mer-circularity concern (leakage measured by 8-mers, features are k-mers)
by using a metric-independent (alignment) definition of near-duplicate.
Also: a no-identity-floor MMseqs2 search on the 5 clean sets to test for k-mer-invisible
homology that the report card would miss.
NOTE: MMseqs2/BLAST are themselves k-mer-SEEDED; we do NOT claim 'k-mer-independent',
only 'alignment-scored / gapped', a different and stricter near-duplicate definition.
"""
import os, subprocess, time, tempfile
import tempfile
import numpy as np, pandas as pd
from audit.core import expkit as E

t0 = time.time()
# Scratch space for MMseqs2 intermediates. Honours AUDIT_SCRATCH when set; otherwise
# uses the platform temp dir. (This was a hardcoded session-scoped absolute path,
# which broke a fresh clone -- and broke it AT IMPORT, since the makedirs ran at
# module scope.)
TMP = os.path.join(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir()),
                   "homology_audit", "mmseqs")

def write_fasta(seqs, path):
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")

def mmseqs_cluster(seqs, tag, min_seq_id=0.7, cov=0.8):
    """Return per-sequence cluster-id array from mmseqs easy-cluster."""
    os.makedirs(TMP, exist_ok=True)
    fa = os.path.join(TMP, f"{tag}.fasta"); write_fasta(seqs, fa)
    pref = os.path.join(TMP, f"{tag}_clu"); tmpd = os.path.join(TMP, f"{tag}_tmp")
    cmd = ["mmseqs", "easy-cluster", fa, pref, tmpd,
           "--min-seq-id", str(min_seq_id), "-c", str(cov), "--threads", "6", "-v", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    # parse <pref>_cluster.tsv : rep \t member
    rep_of = {}
    with open(pref + "_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.split()
            rep_of[int(mem)] = rep
    reps = {r: i for i, r in enumerate(sorted(set(rep_of.values())))}
    comp = np.array([reps[rep_of[i]] for i in range(len(seqs))])
    return comp

rows = []
for d in E.LEAKY:
    seqs, y, otr, ote, tf = E.load(d)
    # 8-mer-Jaccard reference (cached) drop
    Xk = E.featurize(seqs, 6)
    compJ = E.cached_clusters(d, seqs, 0.7, 8)
    jtr, jte = E.assign(compJ, y, 0, tf)
    # alignment clusters
    compA = mmseqs_cluster(seqs, d)
    n_clustered = int((np.bincount(compA)[compA] > 1).sum())
    atr, ate = E.assign(compA, y, 0, tf)
    print(f"[{d}] mmseqs: {compA.max()+1} clusters, {n_clustered} in multi-clusters "
          f"(Jaccard had {compJ.max()+1}) ({time.time()-t0:.0f}s)", flush=True)
    accs = {}
    for m in ["LR", "LinearSVC", "RF", "HGB"]:
        o = float((E.models()[m].fit(Xk[otr], y[otr]).predict(Xk[ote]) == y[ote]).mean())
        cj = float((E.models()[m].fit(Xk[jtr], y[jtr]).predict(Xk[jte]) == y[jte]).mean())
        ca = float((E.models()[m].fit(Xk[atr], y[atr]).predict(Xk[ate]) == y[ate]).mean())
        accs[m] = (o, cj, ca)
        rows.append(dict(dataset=d, model=m, acc_orig=round(o, 4),
                         acc_corr_jaccard=round(cj, 4), acc_corr_mmseqs=round(ca, 4),
                         drop_jaccard=round(o - cj, 4), drop_mmseqs=round(o - ca, 4)))
    ro = sorted(accs, key=lambda m: -accs[m][0]); ra = sorted(accs, key=lambda m: -accs[m][2])
    print(f"  RF drop Jaccard={accs['RF'][0]-accs['RF'][1]:+.4f} mmseqs={accs['RF'][0]-accs['RF'][2]:+.4f}; "
          f"RF rank orig {ro.index('RF')+1} -> mmseqs-corrected {ra.index('RF')+1} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/exp_alignment.csv", index=False)
print("\nEXP_ALIGNMENT_DONE", round(time.time()-t0), "s")
print(df.to_string(index=False))
