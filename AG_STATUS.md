# A-G status

Regenerated 2026-07-20. CLOSED = in the built bundle and gated.

| item | status | note |
|---|---|---|
| A1 | CLOSED | Rafi v2 cited at both sites; monotone-vs-non-monotonic contrast measured and written up |
| A2 | CLOSED | hashFrag run on all 8; threshold sweep; 7/8 verdict agreement; wall-clock recorded |
| A3 | CLOSED | recall/FPR against alignment ground truth, per band, hashFrag Fig-2C form |
| A4 | CLOSED | GB notebooks + delegated scraper audited; omitted merge is now a documented absence |
| A5 | CLOSED | Ensembl 97/100, UCSC accessions, archive hosts, access dates, all re-resolved |
| B1 | CLOSED | three-arm decomposition; 8.98 eval-side / 7.41 split-side on ensembl; in abstract + Table 4 |
| B2 | CLOSED | precision 1.000, FPR 0.000 (<=2.05e-7 exhaustive); 'unmeasurable' argument retired |
| B3 | CLOSED | clean-dataset control passes; difficulty-matched control WEAKENS the claim; claim rewritten |
| B4 | CLOSED | retitled to the conjunctive condition; exception scoped in abstract/intro/conclusion |
| C1 | RUNNING | break-a-clean replicating on cohn (done: leak 0.463, RF rank 1->3) and stark; size-matched arm queued |
| C2 | CLOSED | PREREGISTRATION.md with git log; CNN registration's lack of priority stated |
| C3 | CLOSED | exploratory header; corrected gap registered forward |
| C4 | CLOSED | canonical census suite-wide (no verdict moves) + hashFrag reverse-strand bound <=0.0100 |
| C5 | CLOSED | third tier wired CSV -> table -> caption -> abstract -> figure |
| C6 | QUEUED | dose replication, 3 doses x 5 split x 3 model seeds |
| C7 | CLOSED | indels (detected better than substitutions); repeats; FPR re-measured exhaustively |
| C8 | QUEUED | the suite's own published CNN on the corrected split |
| C9 | DROPPED | maintainer disclosure removed at author's request |
| C10 | CLOSED | SHA-256 manifest of 9 cache entries + upstream release pinning |
| D1-D8 | CLOSED | all eight internal inconsistencies |
| E1 | QUEUED | Welch-Satterthwaite df implemented; needs the clusterboot rerun |
| E2 | QUEUED | BCa implemented beside percentile; needs the clusterboot rerun |
| E3 | CLOSED | pooled BH sensitivity; adds one flag, changes no conclusion |
| E4 | QUEUED | CR2 + t(G-1) cluster-robust SE implemented; needs the clusterboot rerun |
| E5 | QUEUED | 5x5 split-seed x train-seed factorial variance decomposition |
| F1 | CLOSED | Table 4 generated from CSV; ~40 numbers out of running prose |
| F2-F5 | CLOSED | were already done at audit time |

Gate: `./venv/bin/python -m audit.tools.check_numbers` (and `--self-test`).
Build: `bash paper/build.sh`, letters `bash paper/build_letters.sh`.
