"""
test_cpsat_lag.py
-----------------
Standalone test for post-operation time-lag support in the OR-Tools CP-SAT
FJSP solver (ortools_solver.fjsp_solver). This is the exact-baseline used by the
PPVC scheduling paper, where lags model concrete curing / water-ponding test /
paint drying: after op i of a job completes a mandatory wait time_lag[i] must
elapse before that job's NEXT op may start (machines are FREE during the lag).

Checks (PASS/FAIL printed, nonzero exit on any failure):
  1. 1-module RC-wet instance with 1 station/type -> a forced serial schedule.
     CP-SAT OPTIMAL makespan must equal exactly
         sum(chosen pt) + sum(time_lag[:-1])           (serial oracle, == 215.0)
  2. 5-module mixed instance with lags (60s budget): extract the schedule and
     run validate_schedule -> feasible, 0 violations; makespan <= 206.0 (the
     known FIFO lag-aware makespan on this same instance).
  3. Same 5-module instance solved with time_lag=None and with explicit zero
     lags -> identical makespan (backward-compat).

Run:  python test_cpsat_lag.py
"""
import sys
import numpy as np

from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
from ppvc_instance_generator import (
    ppvc_instance_generator, SMALL_FACTORY, DEFAULT_FACTORY,
)
from schedule_validator import validate_schedule


def _ok(flag, label, detail=""):
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return flag


def main():
    all_pass = True

    # ----------------------------------------------------------------------
    # Check 1: 1-module RC-wet, 1 station/type -> forced serial; OPTIMAL must
    # match the serial oracle exactly.
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("Check 1: 1-module RC-wet, SMALL_FACTORY (1 station/type), serial oracle")
    print("=" * 70)
    job_length, op_pt, meta = ppvc_instance_generator(
        n_modules=1, class_mix=[0], station_counts=SMALL_FACTORY, seed=7)
    time_lag = meta['time_lag']

    # Every op has exactly one compatible machine -> "chosen pt" is forced.
    chosen_pt = np.array([row[row > 0][0] for row in op_pt])
    oracle = float(chosen_pt.sum() + time_lag[:-1].sum())
    print(f"  ops={op_pt.shape[0]}  machines={op_pt.shape[1]}  "
          f"sum(pt)={chosen_pt.sum()}  sum(lag[:-1])={time_lag[:-1].sum()}")
    print(f"  serial-oracle makespan = {oracle}")

    jobs, num_machines = matrix_to_the_format_for_solving(job_length, op_pt)
    obj, t, status, asg, op_ct = fjsp_solver(
        jobs, num_machines, time_limits=30, time_lag=time_lag,
        return_schedule=True)
    print(f"  CP-SAT status={status}  makespan={obj}  solveTime={t:.2f}s")

    all_pass &= _ok(status == "OPTIMAL", "Check 1 status is OPTIMAL", status)
    all_pass &= _ok(oracle == 215.0,
                    "Check 1 oracle reproduces 215.0", f"oracle={oracle}")
    all_pass &= _ok(obj == oracle,
                    "Check 1 CP-SAT makespan == serial oracle",
                    f"cpsat={obj} oracle={oracle}")

    # Sanity: the extracted schedule of check 1 must itself validate cleanly.
    res1 = validate_schedule(job_length, op_pt, time_lag, asg, op_ct)
    all_pass &= _ok(res1["feasible"] and not res1["violations"],
                    "Check 1 extracted schedule is feasible",
                    f"violations={len(res1['violations'])} makespan={res1['makespan']}")

    # ----------------------------------------------------------------------
    # Check 2: 5-module mixed, DEFAULT_FACTORY, with lags. Extract + validate.
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("Check 2: 5-module mixed, DEFAULT_FACTORY, with lags (60s), validate")
    print("=" * 70)
    jl5, pt5, meta5 = ppvc_instance_generator(
        n_modules=5, class_mix='mixed', station_counts=DEFAULT_FACTORY, seed=42)
    lag5 = meta5['time_lag']
    jobs5, nm5 = matrix_to_the_format_for_solving(jl5, pt5)

    obj5, t5, status5, asg5, ct5 = fjsp_solver(
        jobs5, nm5, time_limits=60, time_lag=lag5, return_schedule=True)
    print(f"  CP-SAT status={status5}  makespan={obj5}  solveTime={t5:.2f}s")

    res5 = validate_schedule(jl5, pt5, lag5, asg5, ct5)
    print(f"  validator: feasible={res5['feasible']}  "
          f"violations={len(res5['violations'])}  "
          f"makespan={res5['makespan']}  lower_bound={res5['lower_bound']}")
    for v in res5['violations'][:5]:
        print(f"    {v}")

    all_pass &= _ok(status5 in ("OPTIMAL", "FEASIBLE"),
                    "Check 2 status is OPTIMAL or FEASIBLE", status5)
    all_pass &= _ok(res5["feasible"] and not res5["violations"],
                    "Check 2 schedule feasible with 0 violations",
                    f"violations={len(res5['violations'])}")
    # CP-SAT makespan as reported and as reconstructed by the validator must agree.
    all_pass &= _ok(res5["makespan"] == obj5,
                    "Check 2 validator makespan == CP-SAT objective",
                    f"validator={res5['makespan']} cpsat={obj5}")
    all_pass &= _ok(obj5 <= 206.0,
                    "Check 2 CP-SAT makespan <= 206.0 (FIFO lag-aware bound)",
                    f"cpsat={obj5}")

    # ----------------------------------------------------------------------
    # Check 3: backward-compat -- time_lag=None vs explicit zero lags.
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("Check 3: backward-compat -- time_lag=None vs explicit zeros")
    print("=" * 70)
    obj_none, t_none = fjsp_solver(jobs5, nm5, time_limits=60)
    obj_zero, t_zero = fjsp_solver(
        jobs5, nm5, time_limits=60, time_lag=np.zeros(pt5.shape[0], dtype=int))
    print(f"  makespan(time_lag=None)        = {obj_none}  ({t_none:.2f}s)")
    print(f"  makespan(explicit zero lags)   = {obj_zero}  ({t_zero:.2f}s)")

    all_pass &= _ok(obj_none == obj_zero,
                    "Check 3 None lags == explicit zero lags (same makespan)",
                    f"none={obj_none} zero={obj_zero}")
    # With lags removed the makespan must drop below the lag-aware one.
    all_pass &= _ok(obj_none <= obj5,
                    "Check 3 lag-free makespan <= lag-aware makespan",
                    f"nolag={obj_none} lag={obj5}")

    # ----------------------------------------------------------------------
    print("=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
