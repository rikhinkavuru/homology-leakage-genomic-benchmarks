#!/usr/bin/env python3
"""
resources.py -- process-level CPU limits for the audit experiments.

WHY THIS EXISTS
---------------
Nothing in this project uses multiprocessing or joblib.Parallel directly, so the
oversubscription that pegged this machine did not come from nested parallelism
inside one script. It came from running many top-level experiment scripts at once,
each of which assumed it owned the box:

  * run_audit.py and exp_regpath.py asked for n_jobs=-1 (every core),
  * exp_roster.py and exp_tuning.py defaulted to n_jobs=4,
  * and the BLAS backend behind numpy/scikit-learn was left unbounded, so every
    process spawned its own pool of Accelerate/OpenBLAS threads on top of that.

Nine concurrent scripts x (4-10 workers each) x unbounded BLAS threads produced a
load average above 40 on a 10-core machine. The fix is to make each process
modest by default and let the caller opt out explicitly.

IMPORT ORDER MATTERS. The BLAS thread caps are read by the native libraries when
numpy first loads them, so this module must be imported BEFORE numpy anywhere it
is used. audit/core/expkit.py imports it on its first line for exactly that
reason; standalone scripts that do not go through expkit should do the same.

ENV OVERRIDES
-------------
  AUDIT_N_JOBS=<int>        worker cap (default: cores - 2, floor 1)
  AUDIT_BLAS_THREADS=<int>  threads per BLAS pool (default: 1)
  AUDIT_NO_PARENT_WATCH=1   disable the orphan watchdog (for deliberate nohup runs)

Determinism is unaffected. A RandomForestClassifier with a fixed random_state
returns bit-identical results regardless of n_jobs -- chromosome_holdout.py:40
already relies on this -- so capping workers changes wall-clock and nothing else.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# --- BLAS thread caps: must be set before numpy imports the native libraries ---
_BLAS = os.environ.get("AUDIT_BLAS_THREADS", "1")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # Accelerate, i.e. the macOS default backend
    "NUMEXPR_NUM_THREADS",
):
    # setdefault, not assignment: an explicitly exported value still wins.
    os.environ.setdefault(_var, _BLAS)


def _cores() -> int:
    """Cores actually available to this process, not merely present on the host."""
    try:
        return len(os.sched_getaffinity(0))       # respects cpuset/taskset
    except AttributeError:                         # macOS has no sched_getaffinity
        return os.cpu_count() or 1


CORES = _cores()

#: Worker cap for every estimator that takes n_jobs. Leaves headroom so the
#: machine stays usable and the fans stay down; never returns 0 or -1.
N_JOBS = max(1, int(os.environ.get("AUDIT_N_JOBS", max(1, CORES - 2))))

BLAS_THREADS = int(_BLAS)


def die_with_parent(poll_seconds: float = 5.0) -> None:
    """Exit if this process is orphaned.

    A long experiment launched as `bash -c "python -m audit.experiments.foo"` keeps
    running when the shell that started it is killed: the child is reparented to
    init and carries on burning cores with nobody watching. That is precisely how
    nine experiments survived their own supervisor here.

    This starts a daemon thread that notices reparenting and exits. Cheap -- one
    getppid() every few seconds -- and a no-op for a process that was already
    orphaned when it started, or that opted out.
    """
    if os.environ.get("AUDIT_NO_PARENT_WATCH"):
        return
    original_ppid = os.getppid()
    if original_ppid == 1:
        return                                     # already detached; nothing to watch

    def _watch() -> None:
        while True:
            time.sleep(poll_seconds)
            if os.getppid() != original_ppid:
                print(
                    f"[resources] parent {original_ppid} exited; terminating orphan "
                    f"pid={os.getpid()}",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(143)                      # 128 + SIGTERM

    threading.Thread(target=_watch, daemon=True, name="die-with-parent").start()


def describe() -> str:
    """One line for experiment logs, so a run records the limits it ran under."""
    return (
        f"[resources] cores={CORES} n_jobs={N_JOBS} blas_threads={BLAS_THREADS} "
        f"pid={os.getpid()} ppid={os.getppid()}"
    )


# Any process importing this module is an audit experiment, and an audit
# experiment should not outlive its supervisor.
die_with_parent()
