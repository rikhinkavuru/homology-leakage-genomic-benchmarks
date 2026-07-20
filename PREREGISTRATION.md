# Registration provenance

This file records what was committed before what, so the manuscript's priority claims
can be checked against git rather than taken on trust.

**This is a repository-internal commitment, not third-party registration.** Nothing here
was deposited with OSF, AsPredicted, or any external timestamping service. Commit
timestamps are author-controlled: `git commit --date` can set them to anything, and a
history can be rewritten. What the record below establishes is that the predictions and
the results are in *different* commits in the stated order — which is meaningfully
stronger than a claim asserted only in prose, and meaningfully weaker than an external
timestamp. Both bounds are stated in the manuscript (§2.5 and Supplementary §S1).

## The two registration documents

| document | what it registers | added in | timestamp | priority over its results? |
|---|---|---|---|---|
| `results/tier1_preregistration.md` | the cross-suite predictions, including the GUE task-type call | `2edad5c` | 2026-07-18 17:00:21 -0400 | **yes — 9 h 21 min** |
| `results/deep_preregistration.md` | the CNN grid and its refutation condition | `33b1825` | 2026-07-18 14:05:15 -0400 | **no — same commit as the results** |

### The GUE prediction: priority is real

The prediction that GUE's short fixed-length human regulatory tasks would be leaky was
committed 9 h 21 min before the census that scored it, in a separate commit.

Prediction added:

```
$ git log --diff-filter=A --format='%H %ad %s' --date=iso -- results/tier1_preregistration.md
2edad5c3b7a396925d522b6ca672c5ed7276fa09 2026-07-18 17:00:21 -0400 Tier-1 deepeners: pre-registration and analysis code
```

Results added:

```
$ git log --diff-filter=A --format='%H %ad %s' --date=iso -- results/gue_census.csv
3af3eb467a1f69cfa11df9cbcfb4a1366a903524 2026-07-19 02:21:24 -0400 Execute the pre-registration's binding GUE predictions: 5/17, and all 11 LEAKY fail
```

The prediction was already in the earlier blob, not added retroactively. Verify with:

```
$ git show 2edad5c:results/tier1_preregistration.md | sed -n '149,152p'
| GUE core-promoter (70 bp), TF-binding (100 bp), promoter (300 bp), human | **LEAKY** | short fixed-length human regulatory windows — the regime the construction rule flags |
| GUE yeast EMP, virus CVC (multi-species) | **CLEAN** | different genomes, not window-tiled |
```

The census then falsified it: all eleven tasks registered LEAKY are clean at full scale.
The prediction that failed is the one whose priority is documented, which is the
direction that matters — a registration only constrains an author if it can lose.

### The CNN registration: priority is *not* claimed

`results/deep_preregistration.md` was added in `33b1825`, the same commit that added the
experiment code, and `results/exp_deep_cnn.csv` was added in that same commit. It
therefore carries **no** priority evidence whatsoever, and the manuscript does not claim
any: §2.5 calls it "a binding stated condition, not an externally timestamped
registration." It binds because the refutation condition was written down and is
reported whether or not it held — not because git can prove it came first.

This asymmetry between the two documents is deliberate and is stated rather than smoothed
over. Only the GUE registration has provenance; the CNN one has a commitment.

### The six in-code predictions

P1–P6 were committed inside the module that tests each, before that module was run. Their
priority rests on the same repository-internal basis as the above, and all six are scored
in the Results whether or not they held (P1 fails on its registered raw form; §3.7).

## On external timestamping

An OSF deposit of this file's hash was considered and not made. The reason is that it
would timestamp only the *present* state of the record, in July 2026, well after every
prediction and result in it — so it would certify that this document existed on the
deposit date, which is not in dispute, and would not add evidence about the ordering it
describes. Depositing it would look like external validation while supplying none. Future
registrations in this line of work should be deposited *before* the predicted experiment
runs, which is the only point at which a deposit carries information.
