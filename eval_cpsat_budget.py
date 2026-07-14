#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CP-SAT anytime curve: solve the M-class FJSP-TL test set under several short
time budgets (1, 5, 30 s) to characterise how much wall-clock the exact solver
needs to reach the learned policy's operating points (A3 greedy ~210.9 h @ ~2 s;
A3 sampling-100 ~207.4 h @ 40 s; CP-SAT 300 s reference 203.2 h).

Writes its OWN result file and never touches the cached 300 s reference in
or_solution/.  Reuses the exact solver/loader the reference run uses, so the
numbers are directly comparable.
"""
import argparse, json, os, sys, time

# Parse our flags, then hide them before importing ortools_solver (its `params`
# module runs argparse at import time and would choke on our flags).
_p = argparse.ArgumentParser()
_p.add_argument("--dataset", default="10x25+ppvc-mixed")
_p.add_argument("--budgets", default="1,5,30", help="comma-separated seconds")
_p.add_argument("--max_instances", type=int, default=100)
_ARGS = _p.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
from ppvc_instance_generator import load_instance

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = _ARGS.dataset
DATA_DIR = os.path.join(REPO, "data", "PPVC", DATASET)
OUT_DIR = os.path.join(REPO, "test_results", "PPVC", DATASET)
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "cpsat_budget_curve.json")
BUDGETS = [float(b) for b in _ARGS.budgets.split(",")]

# discover instance count
n_avail = len([f for f in os.listdir(DATA_DIR) if f.startswith("instance_") and f.endswith("_jobs.npy")]) \
    if os.path.isdir(DATA_DIR) else 0
if n_avail == 0:  # fallback: count by meta files
    n_avail = len([f for f in os.listdir(DATA_DIR) if f.startswith("instance_") and f.endswith(".json")])
N = min(_ARGS.max_instances, n_avail) if n_avail else _ARGS.max_instances
print(f"[cpsat-budget] dataset={DATASET}  instances={N}  budgets={BUDGETS}s", flush=True)


def solve_lag(inst_id, budget):
    """Return (makespan, status, solve_s) or (None, status, solve_s) if no incumbent."""
    stem = os.path.join(DATA_DIR, f"instance_{inst_id:03d}")
    job_length, op_pt, meta = load_instance(stem)
    time_lag = meta["time_lag"]
    jobs, num_machines = matrix_to_the_format_for_solving(job_length, op_pt)
    t0 = time.time()
    try:
        obj, t_solve, status, asg, op_ct = fjsp_solver(
            jobs, num_machines, time_limits=budget,
            time_lag=time_lag, return_schedule=True)
        if status in ("OPTIMAL", "FEASIBLE") and obj and obj > 0:
            return float(obj), status, float(t_solve)
        return None, status, time.time() - t0
    except Exception as e:  # no incumbent -> ObjectiveValue() raises
        return None, f"NOSOL({type(e).__name__})", time.time() - t0


results = {"dataset": DATASET, "n_instances": N, "budgets": {}}
for budget in BUDGETS:
    per = []
    n_opt = n_feas = n_nosol = 0
    tstart = time.time()
    for i in range(N):
        mk, status, ts = solve_lag(i, budget)
        if mk is None:
            n_nosol += 1
        else:
            n_feas += 1
            if status == "OPTIMAL":
                n_opt += 1
        per.append({"id": i, "makespan": mk, "status": status, "solve_s": ts})
        if (i + 1) % 20 == 0:
            feas_mk = [r["makespan"] for r in per if r["makespan"] is not None]
            cur = float(np.mean(feas_mk)) if feas_mk else float("nan")
            print(f"  budget={budget:g}s  {i+1}/{N}  running mean(feasible)={cur:.1f} "
                  f"opt={n_opt} feas={n_feas} nosol={n_nosol}", flush=True)
    feas_mk = [r["makespan"] for r in per if r["makespan"] is not None]
    summary = {
        "budget_s": budget,
        "mean_makespan_feasible": float(np.mean(feas_mk)) if feas_mk else None,
        "std_makespan_feasible": float(np.std(feas_mk)) if feas_mk else None,
        "n_feasible": n_feas, "n_proven_optimal": n_opt, "n_no_solution": n_nosol,
        "mean_solve_s": float(np.mean([r["solve_s"] for r in per])),
        "wall_s": time.time() - tstart,
        "per_instance": per,
    }
    results["budgets"][str(budget)] = summary
    print(f"[budget {budget:g}s] mean(feasible)={summary['mean_makespan_feasible']} "
          f"opt={n_opt}/{N} feas={n_feas}/{N} nosol={n_nosol} "
          f"wall={summary['wall_s']:.0f}s", flush=True)
    # save incrementally so partial curves are usable
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved {OUT_JSON}", flush=True)

print("[cpsat-budget] DONE", flush=True)
