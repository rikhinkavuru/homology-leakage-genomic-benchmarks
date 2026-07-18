#!/bin/zsh
# Serial experiment chain — ONE ensembl-heavy job at a time (avoids the OOM that
# killed the 4-way concurrent run). exp_stats runs first to warm the cached pairwise;
# subsequent jobs reuse datacache/{sim,comp}_*.npy. `;` not `&&` so one failure never
# halts the chain. Tier-1 results first.
cd /Users/rikhinkavuru/homology_audit
PY=./venv/bin/python
for exp in exp_stats exp_regpath exp_canonical exp_alignment exp_imbalance exp_inject3class exp_geometry exp_clusterboot_full exp_repeat; do
  echo "SERIAL_START $exp $(date +%H:%M:%S)"
  $PY $exp.py > $exp.log 2>&1
  echo "SERIAL_DONE $exp EXIT=$? $(date +%H:%M:%S)"
done
echo "RUN_SERIAL_ALL_DONE $(date +%H:%M:%S)"
