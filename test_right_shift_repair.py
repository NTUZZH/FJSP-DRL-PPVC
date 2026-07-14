"""
Tests for right_shift_repair.py — the ablation arm A0
("lag-blind + right-shift repair", current industry practice).

Layers of verification:
  T1  Hand-computable micro case (2 jobs, 3 machines, a lag). Every repaired
      start/ct is verified by hand in the comments and asserted exactly.
  T2  Property test on real PPVC instances (seeds 42, 43, 44):
        - build a lag-BLIND schedule via a FIFO rollout WITHOUT lags,
        - repair it,
        - assert (a) the REPAIRED schedule is feasible under the validator
          WITH lags, (b) repaired starts >= lag-blind starts elementwise
          (right-shift only), (c) the machine ORDER is unchanged, (d) repaired
          makespan >= lag-blind makespan.
  T3  Sanity: with ALL-ZERO lags, repair returns the original schedule exactly.

Run:  python test_right_shift_repair.py
Prints PASS/FAIL lines; exits nonzero on any failure.
"""
import sys
import numpy as np

from right_shift_repair import right_shift_repair
from schedule_validator import validate_schedule, reconstruct_starts
from ppvc_instance_generator import ppvc_instance_generator, DEFAULT_FACTORY
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from common_utils import heuristic_select_action


_FAILED = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not cond:
        _FAILED.append(name)
    return cond


# ----------------------------------------------------------------------------
# Shared helper: lag-BLIND FIFO rollout (no lags passed -> true_op_ct is
# lag-blind). Mirrors rollout_fifo in test_lag_env.py but always lag-free.
# ----------------------------------------------------------------------------
def rollout_fifo_lagblind(job_length, op_pt, seed=0):
    J, M = len(job_length), op_pt.shape[1]
    np.random.seed(seed)
    env = FJSPEnvForSameOpNums(n_j=J, n_m=M)
    env.set_initial_data([job_length], [op_pt])            # NO lags -> lag-blind
    n_ops = op_pt.shape[0]
    assigned_mch = np.full(n_ops, -1, dtype=int)
    while not env.done().all():
        action = heuristic_select_action('FIFO', env)
        chosen_job = action // M
        chosen_mch = action % M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = chosen_mch
        env.step(np.array([action]))
    assert (assigned_mch >= 0).all(), 'some op was never scheduled'
    return assigned_mch, env.true_op_ct[0].copy(), float(env.current_makespan[0])


def _mch_order_by_start(assigned_mch, starts, n_mch):
    """Per-machine op order sorted by start time (tie-broken by op index)."""
    order = []
    for m in range(n_mch):
        ops = [i for i in range(len(assigned_mch)) if assigned_mch[i] == m]
        ops.sort(key=lambda i: (starts[i], i))
        order.append(ops)
    return order


# ============================================================================
# T1  Hand-computable micro case
# ============================================================================
def t1_micro():
    print("T1  hand-computable micro case (2 jobs, 3 machines, a lag) ...")
    # ---------- instance ----------
    # job0: ops 0,1,2 ; job1: ops 3,4   (job-by-job, precedence order)
    # op_pt:        m0  m1  m2     (single compatible machine per op -> forced)
    #   op0:         2   0   0     -> m0
    #   op1:         0   3   0     -> m1
    #   op2:         4   0   0     -> m0
    #   op3:         0   3   0     -> m1
    #   op4:         0   0   2     -> m2
    job_length = [3, 2]
    op_pt = np.array([
        [2, 0, 0],
        [0, 3, 0],
        [4, 0, 0],
        [0, 3, 0],
        [0, 0, 2],
    ], dtype=float)
    assigned = np.array([0, 1, 0, 1, 2])
    # lags: only op0 has a (large) lag of 5h; op2 & op4 are job-final.
    time_lag = np.array([5.0, 0.0, 0.0, 0.0, 0.0])

    # ---------- lag-BLIND schedule (built by hand, ignoring lags) ----------
    # Machine sequences chosen: m0:[op0,op2], m1:[op3,op1], m2:[op4].
    # Note m1 runs op3 BEFORE op1 -> a machine-arc op3->op1 that the repair
    # must honour (this is what makes "keep the sequence" non-trivial).
    #   op0 (m0): start 0,  ct 2
    #   op3 (m1): start 0,  ct 3
    #   op1 (m1): job-ready ct0=2, m1-free 3   -> start 3,  ct 6
    #   op4 (m2): job-ready ct3=3, m2-free 0   -> start 3,  ct 5
    #   op2 (m0): job-ready ct1=6, m0-free 2   -> start 6,  ct 10
    op_ct_lagblind = np.array([2.0, 6.0, 10.0, 3.0, 5.0])
    lagblind_starts = op_ct_lagblind - np.array(
        [op_pt[i, assigned[i]] for i in range(5)])
    # lagblind_starts == [0, 3, 6, 0, 3], makespan 10.

    # ---------- repair by hand (lag op0 = 5 now applies) ----------
    # Arcs:  job 0->1 (w ct0+lag0), 1->2 (w ct1+lag1=ct1), 3->4 (w ct3);
    #        m0 0->2 (w ct0), m1 3->1 (w ct3).
    # Longest path:
    #   op0: start 0, ct 2
    #   op3: start 0, ct 3
    #   op4: pred job op3 -> ready ct3=3            -> start 3,  ct 5
    #   op1: preds job op0 -> ct0+lag0=2+5=7 ;
    #               mch op3 -> ct3=3                -> start 7,  ct 10
    #   op2: preds job op1 -> ct1+lag1=10 ;
    #               mch op0 -> ct0=2                -> start 10, ct 14
    expected_ct = np.array([2.0, 10.0, 14.0, 3.0, 5.0])
    expected_makespan = 14.0

    # ---------- run via op_ct_lagblind (order derived by sorting) ----------
    rct, rms = right_shift_repair(
        job_length, op_pt, time_lag, assigned, op_ct_lagblind=op_ct_lagblind)
    _check("micro exact repaired ct (derived order)",
           np.allclose(rct, expected_ct),
           f"got {rct.tolist()}")
    _check("micro exact makespan (derived order)",
           rms == expected_makespan, f"got {rms}")

    # ---------- run via explicit per-machine sequences (same result) ----------
    seqs = [[0, 2], [3, 1], [4]]
    rct2, rms2 = right_shift_repair(
        job_length, op_pt, time_lag, assigned, op_sequence_per_mch=seqs)
    _check("micro exact repaired ct (explicit seq)",
           np.allclose(rct2, expected_ct), f"got {rct2.tolist()}")
    _check("micro makespan (explicit seq == derived)",
           rms2 == rms, f"{rms2} vs {rms}")

    # ---------- repaired schedule must be feasible under the validator ----------
    res = validate_schedule(job_length, op_pt, time_lag, assigned, rct)
    _check("micro repaired schedule feasible", res["feasible"],
           f"violations={res['violations'][:3]}")

    # ---------- argument-guard checks ----------
    def _expect_valueerror(fn, label):
        try:
            fn()
        except ValueError:
            _check(label, True)
        except Exception as e:               # noqa: BLE001
            _check(label, False, f"raised {type(e).__name__} not ValueError")
        else:
            _check(label, False, "no error raised")

    _expect_valueerror(
        lambda: right_shift_repair(job_length, op_pt, time_lag, assigned),
        "guard: neither optional arg -> ValueError")
    _expect_valueerror(
        lambda: right_shift_repair(
            job_length, op_pt, time_lag, assigned,
            op_sequence_per_mch=seqs, op_ct_lagblind=op_ct_lagblind),
        "guard: both optional args -> ValueError")

    # ---------- cycle detection ----------
    # Force a machine sequence that contradicts job precedence: put op1 BEFORE
    # op0's job-successor in a way that creates a cycle. Construct a 1-job,
    # 2-op instance on a single machine but order the machine seq backwards.
    jl_c = [2]
    pt_c = np.array([[1.0], [1.0]])     # op0,op1 both on m0
    lag_c = np.array([0.0, 0.0])
    assigned_c = np.array([0, 0])
    # job-arc 0->1 ; machine seq says 1 before 0 -> machine-arc 1->0 -> cycle
    _expect_valueerror(
        lambda: right_shift_repair(
            jl_c, pt_c, lag_c, assigned_c, op_sequence_per_mch=[[1, 0]]),
        "guard: cyclic machine seq -> ValueError")


# ============================================================================
# T2  Property test on real PPVC instances
# ============================================================================
def t2_property():
    print("T2  property test on real PPVC instances (seeds 42, 43, 44) ...")
    seed42_makespans = None
    for s in (42, 43, 44):
        jl, op_pt, meta = ppvc_instance_generator(
            n_modules=5, class_mix='mixed',
            station_counts=DEFAULT_FACTORY, seed=s)
        time_lag = meta['time_lag'].astype(float)
        n_mch = op_pt.shape[1]

        # lag-blind FIFO schedule
        assigned, op_ct_lb, ms_lb = rollout_fifo_lagblind(jl, op_pt, seed=s)
        starts_lb = reconstruct_starts(op_pt, assigned, op_ct_lb)

        # repair (derive machine order from the lag-blind start times)
        rct, rms = right_shift_repair(
            jl, op_pt, time_lag, assigned, op_ct_lagblind=op_ct_lb)
        starts_rep = reconstruct_starts(op_pt, assigned, rct)

        # (a) repaired schedule feasible WITH lags
        res = validate_schedule(jl, op_pt, time_lag, assigned, rct)
        _check(f"seed {s}: repaired feasible (0 violations)",
               res["feasible"] and len(res["violations"]) == 0,
               f"{len(res['violations'])} violations: {res['violations'][:2]}")

        # (b) right-shift only: repaired starts >= lag-blind starts
        _check(f"seed {s}: repaired starts >= lag-blind starts",
               bool(np.all(starts_rep >= starts_lb - 1e-6)),
               f"min delta {float((starts_rep - starts_lb).min()):.3f}")

        # (c) machine ORDER unchanged (per machine, op order by start time)
        order_lb = _mch_order_by_start(assigned, starts_lb, n_mch)
        order_rep = _mch_order_by_start(assigned, starts_rep, n_mch)
        _check(f"seed {s}: machine order unchanged",
               order_lb == order_rep)

        # (d) repaired makespan >= lag-blind makespan
        _check(f"seed {s}: repaired makespan >= lag-blind makespan",
               rms >= ms_lb - 1e-6, f"{rms:.1f} >= {ms_lb:.1f}")

        print(f"      seed {s}: lag-blind makespan = {ms_lb:.1f}h  ->  "
              f"repaired = {rms:.1f}h  (+{rms - ms_lb:.0f}h, "
              f"+{100 * (rms / ms_lb - 1):.0f}%)")
        if s == 42:
            seed42_makespans = (ms_lb, rms)

    if seed42_makespans is not None:
        print(f"      [paper data point] seed 42: lag-blind {seed42_makespans[0]:.1f}h"
              f" -> right-shift-repaired {seed42_makespans[1]:.1f}h "
              f"(lag-aware FIFO reference = 206h)")


# ============================================================================
# T3  Sanity: zero lags -> identity repair
# ============================================================================
def t3_zero_lags():
    print("T3  zero-lag sanity: repair must return the original schedule ...")
    for s in (42, 43, 44):
        jl, op_pt, _ = ppvc_instance_generator(
            n_modules=5, class_mix='mixed',
            station_counts=DEFAULT_FACTORY, seed=s)
        assigned, op_ct_lb, ms_lb = rollout_fifo_lagblind(jl, op_pt, seed=s)
        zero_lag = np.zeros(op_pt.shape[0], dtype=float)
        rct, rms = right_shift_repair(
            jl, op_pt, zero_lag, assigned, op_ct_lagblind=op_ct_lb)
        _check(f"seed {s}: zero-lag repair == original ct",
               np.allclose(rct, op_ct_lb),
               f"max|delta|={float(np.abs(rct - op_ct_lb).max()):.4f}")
        _check(f"seed {s}: zero-lag makespan unchanged",
               abs(rms - ms_lb) < 1e-6, f"{rms:.1f} vs {ms_lb:.1f}")


# ============================================================================
if __name__ == '__main__':
    t1_micro()
    t2_property()
    t3_zero_lags()
    print()
    if _FAILED:
        print(f"SOME TESTS FAILED ({len(_FAILED)}): {_FAILED}")
        sys.exit(1)
    print("ALL RIGHT-SHIFT-REPAIR TESTS PASSED")
