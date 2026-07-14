"""
Correctness tests for the lag-aware environment (PPVC Adaptation 2a).

Three independent layers of verification:
  T1  Serial oracle: 1 module, 1 station per type -> the schedule is fully
      serial, so makespan MUST equal sum(chosen pt) + sum(lags of non-final
      ops). Exact analytical check, no scheduler logic shared with the env.
  T2  Feasibility: FIFO rollout on a multi-module instance; every constraint
      (compatibility, precedence WITH lags, machine no-overlap, bounds) is
      re-checked by the independent schedule_validator (separate
      implementation, no env code shared).
  T3  Consistency: env-reported makespan == validator-recomputed makespan,
      and the lag-aware makespan strictly exceeds the lag-free one.

Run:  python test_lag_env.py
"""
import numpy as np

from ppvc_instance_generator import (ppvc_instance_generator, DEFAULT_FACTORY,
                                     SMALL_FACTORY)
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from common_utils import heuristic_select_action
from schedule_validator import validate_schedule


def rollout_fifo(job_length, op_pt, time_lag=None, seed=0):
    """FIFO rollout recording per-op machine assignment; returns env + arrays."""
    J, M = len(job_length), op_pt.shape[1]
    np.random.seed(seed)
    env = FJSPEnvForSameOpNums(n_j=J, n_m=M)
    if time_lag is None:
        env.set_initial_data([job_length], [op_pt])
    else:
        env.set_initial_data([job_length], [op_pt], time_lag_list=[time_lag])
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
    return env, assigned_mch, env.true_op_ct[0].copy()


def t1_serial_oracle():
    print('T1  serial oracle (1 module, 1 station/type) ...')
    jl, op_pt, meta = ppvc_instance_generator(
        n_modules=1, class_mix=[0], station_counts=SMALL_FACTORY, seed=7)  # RC-wet
    env, assigned, op_ct = rollout_fifo(jl, op_pt, meta['time_lag'])
    chosen_pt = np.array([op_pt[i, assigned[i]] for i in range(op_pt.shape[0])])
    lags = meta['time_lag'].astype(float)
    expected = chosen_pt.sum() + lags[:-1].sum()   # last op's lag has no successor
    got = env.current_makespan[0]
    assert abs(got - expected) < 1e-6, f'serial oracle mismatch: {got} != {expected}'
    print(f'    makespan = {got:.1f}h == sum(pt)={chosen_pt.sum()} '
          f'+ sum(lag)={lags[:-1].sum():.0f}  OK')


def t2_t3_fifo_validator():
    print('T2  FIFO + independent validator (5 modules, mixed classes) ...')
    jl, op_pt, meta = ppvc_instance_generator(
        n_modules=5, class_mix='mixed', station_counts=DEFAULT_FACTORY, seed=42)

    # lag-free baseline
    env0, _, _ = rollout_fifo(jl, op_pt, None)
    ms0 = env0.current_makespan[0]

    # lag-aware run
    env1, assigned, op_ct = rollout_fifo(jl, op_pt, meta['time_lag'])
    ms1 = env1.current_makespan[0]

    res = validate_schedule(jl, op_pt, meta['time_lag'], assigned, op_ct)
    assert res['feasible'], 'validator found violations:\n  ' + \
        '\n  '.join(res['violations'][:10])
    print(f'    validator: feasible, 0 violations '
          f'(lower bound {res["lower_bound"]:.0f}h)')

    print('T3  consistency checks ...')
    assert abs(res['makespan'] - ms1) < 1e-6, \
        f'env makespan {ms1} != validator makespan {res["makespan"]}'
    assert ms1 > ms0, f'lag-aware makespan {ms1} must exceed lag-free {ms0}'
    # zero-lag through the lag-aware code path must equal the no-lag path
    env2, _, _ = rollout_fifo(jl, op_pt, np.zeros_like(meta['time_lag']))
    assert abs(env2.current_makespan[0] - ms0) < 1e-9, \
        'explicit zero lags diverge from default no-lag path'
    print(f'    env == validator makespan ({ms1:.1f}h); '
          f'lag-free {ms0:.1f}h -> lag-aware {ms1:.1f}h '
          f'(+{ms1 - ms0:.0f}h, +{100 * (ms1 / ms0 - 1):.0f}%)')


if __name__ == '__main__':
    t1_serial_oracle()
    t2_t3_fifo_validator()
    print('ALL LAG-ENV TESTS PASSED')
