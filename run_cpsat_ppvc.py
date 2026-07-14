"""
run_cpsat_ppvc.py
-----------------
Solve the held-out PPVC test set with the lag-aware CP-SAT solver and store
per-instance reference results for the IEEE TII paper main table.

For each test instance (data/PPVC/10x25+ppvc-mixed/instance_000..099) this:

  1. Loads (job_length, op_pt, meta) via ppvc_instance_generator.load_instance
     and converts the matrix to the OR-Tools nested-list `jobs` format with
     ortools_solver.matrix_to_the_format_for_solving (EXACTLY as test_cpsat_lag).
  2. Solves LAG-AWARE: time_lag=meta['time_lag'], return_schedule=True, 300 s
     cap. Records objective, status, solve seconds, and an independent
     schedule_validator verdict (feasible bool + #violations) on the extracted
     schedule.
  3. Solves LAG-FREE: time_lag=None, same 300 s cap (no schedule needed). The
     lag-aware / lag-free pair quantifies the per-instance lag inflation.
  4. Streams one JSON line per instance to
     or_solution/PPVC/10x25+ppvc-mixed.jsonl as soon as it finishes
     (crash-safe; consumable while the run is live).
  5. Resume-safe: skips instance ids already present in the .jsonl on startup.
  6. At the end writes or_solution/PPVC/10x25+ppvc-mixed_summary.json.

CPU-only (no GPU training is disturbed). Run:
    python -u run_cpsat_ppvc.py
Smoke-test a single instance first:
    ... run_cpsat_ppvc.py --smoke 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Parse our own CLI args and strip them BEFORE importing the heavy modules:
# ortools_solver imports `params`, whose `parser.parse_args()` runs at import
# time and would choke on our flags. We stash a clean argv, import, restore.
_argp = argparse.ArgumentParser()
_argp.add_argument("--smoke", type=int, default=None,
                   help="solve only this single instance id (inline smoke test)")
_argp.add_argument("--dataset", type=str, default="10x25+ppvc-mixed",
                   help="dataset dir name under data/PPVC/ (outputs keyed by the same name)")
_argp.add_argument("--time_limit", type=float, default=300.0,
                   help="CP-SAT time limit per solve in seconds")
_ARGS, _ = _argp.parse_known_args()
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]  # hide our flags from params' argparse at import time

from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
from ppvc_instance_generator import load_instance
from schedule_validator import validate_schedule

sys.argv = _saved_argv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = _ARGS.dataset
DATA_DIR = os.path.join(REPO, "data", "PPVC", DATASET)
OUT_DIR = os.path.join(REPO, "or_solution", "PPVC")
JSONL_PATH = os.path.join(OUT_DIR, f"{DATASET}.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, f"{DATASET}_summary.json")
# instance count discovered from the directory (test sets vary: 100 or 50)
N_INSTANCES = len([f for f in os.listdir(DATA_DIR) if f.endswith(".fjs")])
TIME_LIMIT = float(_ARGS.time_limit)  # seconds, per solve


def _jsonable(x):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def solve_one(inst_id):
    """Solve instance `inst_id` lag-aware (+validate) and lag-free.

    Returns a JSON-serializable dict (one .jsonl record).
    """
    stem = os.path.join(DATA_DIR, f"instance_{inst_id:03d}")
    job_length, op_pt, meta = load_instance(stem)
    time_lag = meta["time_lag"]

    # EXACT reuse of the conversion test_cpsat_lag.py uses.
    jobs, num_machines = matrix_to_the_format_for_solving(job_length, op_pt)

    # ----- Lag-aware solve (with schedule extraction + validation) -----
    obj_lag, t_lag, status_lag, asg, op_ct = fjsp_solver(
        jobs, num_machines, time_limits=TIME_LIMIT,
        time_lag=time_lag, return_schedule=True)

    val = validate_schedule(job_length, op_pt, time_lag, asg, op_ct)
    feasible = bool(val["feasible"])
    n_violations = len(val["violations"])
    if not feasible:
        # Log loudly to stdout (captured in the run log) but DO NOT abort.
        print(f"!!! VALIDATOR VIOLATIONS on instance {inst_id:03d}: "
              f"{n_violations} violation(s); makespan={val['makespan']} "
              f"lower_bound={val['lower_bound']}", flush=True)
        for v in val["violations"][:10]:
            print(f"    {v}", flush=True)

    # ----- Lag-free solve (objective only) -----
    obj_free, t_free = fjsp_solver(
        jobs, num_machines, time_limits=TIME_LIMIT, time_lag=None)

    # Per-instance lag inflation (% increase of lag-aware over lag-free).
    inflation_pct = None
    if obj_free and obj_free > 0:
        inflation_pct = float((obj_lag - obj_free) / obj_free * 100.0)

    rec = {
        "instance_id": int(inst_id),
        "instance": f"instance_{inst_id:03d}",
        "n_jobs": int(len(job_length)),
        "n_ops": int(op_pt.shape[0]),
        "n_machines": int(num_machines),
        "time_limit_s": TIME_LIMIT,
        # lag-aware
        "makespan_lag": _jsonable(obj_lag),
        "status_lag": status_lag,
        "solve_s_lag": float(t_lag),
        # lag-free
        "makespan_free": _jsonable(obj_free),
        "solve_s_free": float(t_free),
        # comparison
        "inflation_pct": inflation_pct,
        # validator verdict on the lag-aware schedule
        "validator_feasible": feasible,
        "validator_n_violations": int(n_violations),
        "validator_makespan": _jsonable(val["makespan"]),
        "validator_lower_bound": _jsonable(val["lower_bound"]),
        "validator_violations": val["violations"][:20],  # cap, full set in log
        "wall_time_s": float(t_lag + t_free),
    }
    return rec


def load_done_ids():
    """Resume support: ids already present (one record per line) in the .jsonl."""
    done = set()
    if not os.path.exists(JSONL_PATH):
        return done
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(int(rec["instance_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Tolerate a truncated final line from an interrupted run.
                continue
    return done


def append_record(rec):
    """Atomically append one JSON line and flush+fsync so consumers see it."""
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(rec, default=_jsonable) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_summary():
    """Aggregate all .jsonl records into the summary JSON."""
    recs = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not recs:
        return None
    recs.sort(key=lambda r: r["instance_id"])

    ms_lag = np.array([r["makespan_lag"] for r in recs], dtype=float)
    ms_free = np.array([r["makespan_free"] for r in recs], dtype=float)
    infl = np.array([r["inflation_pct"] for r in recs
                     if r["inflation_pct"] is not None], dtype=float)

    status_counts = {}
    for r in recs:
        status_counts[r["status_lag"]] = status_counts.get(r["status_lag"], 0) + 1

    n_infeasible = sum(1 for r in recs if not r["validator_feasible"])
    total_wall = float(sum(r["wall_time_s"] for r in recs))

    summary = {
        "dataset": DATASET,
        "n_solved": len(recs),
        "time_limit_s": TIME_LIMIT,
        "makespan_lag": {
            "mean": float(ms_lag.mean()),
            "min": float(ms_lag.min()),
            "max": float(ms_lag.max()),
        },
        "makespan_free": {
            "mean": float(ms_free.mean()),
            "min": float(ms_free.min()),
            "max": float(ms_free.max()),
        },
        "mean_inflation_pct": float(infl.mean()) if len(infl) else None,
        "status_counts_lag": status_counts,
        "validator_infeasible_count": n_infeasible,
        "total_wall_time_s": total_wall,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return summary


def main():
    args = _ARGS

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke is not None:
        print(f"=== SMOKE TEST: instance {args.smoke:03d} "
              f"(cap {TIME_LIMIT:.0f}s/solve) ===", flush=True)
        t0 = time.time()
        rec = solve_one(args.smoke)
        print(json.dumps(rec, indent=2, default=_jsonable), flush=True)
        print(f"=== smoke wall = {time.time() - t0:.1f}s ===", flush=True)
        return

    done = load_done_ids()
    print(f"=== CP-SAT PPVC batch: {N_INSTANCES} instances, "
          f"{TIME_LIMIT:.0f}s/solve, lag-aware + lag-free ===", flush=True)
    print(f"=== resume: {len(done)} already done, skipping them ===", flush=True)

    batch_t0 = time.time()
    for inst_id in range(N_INSTANCES):
        if inst_id in done:
            continue
        print(f"--- instance {inst_id:03d} "
              f"({time.strftime('%m-%d %H:%M:%S')}) ---", flush=True)
        rec = solve_one(inst_id)
        append_record(rec)
        print(f"    instance {inst_id:03d}: "
              f"makespan_lag={rec['makespan_lag']} ({rec['status_lag']}, "
              f"{rec['solve_s_lag']:.1f}s)  "
              f"makespan_free={rec['makespan_free']} ({rec['solve_s_free']:.1f}s)  "
              f"inflation={rec['inflation_pct']:.1f}%  "
              f"valid={rec['validator_feasible']} "
              f"(viol={rec['validator_n_violations']})", flush=True)
        # Refresh the summary after every instance so it is always current.
        write_summary()

    summary = write_summary()
    print(f"=== batch DONE in {time.time() - batch_t0:.1f}s ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
