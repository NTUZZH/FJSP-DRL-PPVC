#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SPT-warm-started CP-SAT anytime curve. Seeds CP-SAT with the SPT dispatching
schedule (so its first incumbent is >= SPT) and measures the anytime makespan
under short budgets. Complements the cold-start curve: a warm-started exact
solver matches the heuristic instantly and only improves with time.

Reuses eval_ppvc.rollout_heuristic for the EXACT SPT schedule, and the same solver
as the cold-start curve, so the two are directly comparable.
"""
import argparse, json, os, sys, time

_p = argparse.ArgumentParser()
_p.add_argument("--dataset", default="40x25+ppvc-mixed")
_p.add_argument("--budgets", default="1,5,30")
_p.add_argument("--max_instances", type=int, default=20)
_p.add_argument("--seed_test", type=int, default=50)
_A = _p.parse_args()
sys.argv = [sys.argv[0]]  # scrub before importing argparse-at-import modules

import numpy as np
from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
from ppvc_instance_generator import load_instance
from eval_ppvc import rollout_heuristic

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "data", "PPVC", _A.dataset)
N = min(_A.max_instances, len([f for f in os.listdir(DATA) if f.endswith(".fjs")]))
BUD = [float(b) for b in _A.budgets.split(",")]
print(f"[warmstart] {_A.dataset}  N={N}  budgets={BUD}s", flush=True)

results = {"dataset": _A.dataset, "n": N, "budgets": {}}
for b in BUD:
    mks, nopt, tsum = [], 0, 0.0
    for i in range(N):
        stem = os.path.join(DATA, f"instance_{i:03d}")
        jl, pt, meta = load_instance(stem)
        # the warm start = SPT dispatching schedule (assigned machine + completion time)
        _, _, asg, op_ct = rollout_heuristic("SPT", jl, pt, meta, _A.seed_test)
        jobs, nm = matrix_to_the_format_for_solving(jl, pt)
        obj, tsolve, status, _, _ = fjsp_solver(
            jobs, nm, time_limits=b, time_lag=meta["time_lag"],
            return_schedule=True, warmstart=(asg, op_ct))
        mks.append(float(obj)); nopt += (status == "OPTIMAL"); tsum += tsolve
    results["budgets"][str(b)] = {"mean_makespan": float(np.mean(mks)),
                                  "n_proven_optimal": nopt, "mean_solve_s": tsum / N,
                                  "per_instance": [float(x) for x in mks]}
    print(f"  budget={b:g}s  warm-start mean={np.mean(mks):.1f}  opt={nopt}/{N}", flush=True)
    out = os.path.join(REPO, "test_results", "PPVC", _A.dataset, "cpsat_warmstart_curve.json")
    json.dump(results, open(out, "w"), indent=2)
print("[warmstart] DONE", flush=True)
