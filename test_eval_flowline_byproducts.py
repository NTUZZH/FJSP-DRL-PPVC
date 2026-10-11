"""
test_eval_flowline_byproducts.py
--------------------------------
Checks for eval_flowline.py and eval_byproducts.py.

  flowline_restriction   on the first --max_instances instances of --dataset:
                         exactly one compatible station per operation, the
                         station follows the (j mod n_t) rule, the flow-line
                         SPT schedule never leaves its line and is feasible
                         for the flexible instance; reports (does not fail)
                         any instance where the flow-line SPT makespan is
                         below the flexible SPT makespan
  flowline_cpsat         CP-SAT on the flow-line restriction of a 5-module
                         generator instance (seed 42, DEFAULT_FACTORY) is
                         validator-clean on the flow-line and the flexible
                         instance
  byproducts_hand        compute_byproducts on the validator's hand-checked
                         2-job schedule returns the hand-computed values
  byproducts_bounds      on the first --max_instances instances, for greedy
                         and every dispatching rule: station utilization in
                         [0, 1] with the mean in (0, 1], mean WIP <= number
                         of modules, module flow time >= its own processing
                         time plus lag sum, zero tardiness at a due factor of
                         100
  cli                    both scripts end to end with --max_instances into a
                         temporary folder (flow-line CP-SAT at --cpsat_limit s)

Run pinned, e.g.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
      python test_eval_flowline_byproducts.py [--max_instances 3]
Exits nonzero on any failed check.
"""
import argparse
import json
import os
import sys
import tempfile
import time

_p = argparse.ArgumentParser()
_p.add_argument('--dataset', default='10x25+ppvc-mixed')
_p.add_argument('--max_instances', type=int, default=3)
_p.add_argument('--cpsat_limit', type=float, default=20.0)
_p.add_argument('--only', nargs='+', default=None,
                help='subset of test names to run')
ARGS = _p.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np

import eval_byproducts as bp
import eval_flowline as fl

FAILS = []
NOTES = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def _stems():
    ep = bp.import_eval_ppvc()
    return ep, ep.list_instances(f'data/PPVC/{ARGS.dataset}')[:ARGS.max_instances]


def test_flowline_restriction():
    ep, stems = _stems()
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        pt_fl, fixed = fl.flowline_restrict(jl, pt, meta)
        name = os.path.basename(stem)
        n_compat = (pt_fl > 0).sum(axis=1)
        check((n_compat == 1).all(),
              f'{name}: one compatible station per op '
              f'(min {n_compat.min()}, max {n_compat.max()}; flexible mean '
              f'{fl.mean_compatible(pt):.2f})')
        # the (j mod n_t) rule, recomputed independently
        mt = np.asarray(meta['mch_type'])
        ok_rule, op = True, 0
        for j, n in enumerate(jl):
            for _ in range(int(n)):
                st = [m for m in range(len(mt)) if mt[m] == meta['op_station'][op]]
                ok_rule &= fixed[op] == st[j % len(st)]
                ok_rule &= pt_fl[op, fixed[op]] == pt[op, fixed[op]]
                op += 1
        check(bool(ok_rule), f'{name}: station = S_t[j mod n_t], original pt kept')
        ms_fl, _, asg, ct = bp.run_validated(ep, 'SPT', jl, pt_fl, meta, i, 50)
        check(np.array_equal(asg, fixed), f'{name}: flow-line SPT stays on its line')
        check(validate_schedule(jl, pt, meta['time_lag'], asg, ct)['feasible'],
              f'{name}: flow-line SPT schedule feasible for the flexible instance')
        ms_fx, _, _, _ = bp.run_validated(ep, 'SPT', jl, pt, meta, i, 50)
        rel = '>=' if ms_fl >= ms_fx else '<'
        print(f'  info {name}: SPT flow-line {ms_fl:.0f} {rel} flexible {ms_fx:.0f}')
        if ms_fl < ms_fx:
            NOTES.append(f'{name}: flow-line SPT {ms_fl:.0f} < flexible SPT {ms_fx:.0f}')


def test_flowline_cpsat():
    from ppvc_instance_generator import ppvc_instance_generator, DEFAULT_FACTORY
    from schedule_validator import validate_schedule
    jl, pt, meta = ppvc_instance_generator(n_modules=5, class_mix='mixed',
                                           station_counts=DEFAULT_FACTORY,
                                           seed=42)
    pt_fl, fixed = fl.flowline_restrict(jl, pt, meta)
    check(((pt_fl > 0).sum(axis=1) == 1).all(), 'gen-5: one station per op')
    t0 = time.time()
    r = fl.solve_cpsat(jl, pt_fl, meta['time_lag'], ARGS.cpsat_limit, 2)
    print(f'  info gen-5 flow-line CP-SAT: status {r["status"]}, makespan '
          f'{r["makespan"]}, bound {r["bound"]}, {time.time() - t0:.2f} s')
    check(r['makespan'] is not None, 'gen-5: CP-SAT found a schedule')
    v = validate_schedule(jl, pt_fl, meta['time_lag'], r['assigned_mch'], r['op_ct'])
    check(v['feasible'] and len(v['violations']) == 0,
          f'gen-5: validator-clean on the flow-line instance '
          f'({len(v["violations"])} violations)')
    check(abs(v['makespan'] - r['makespan']) < 1e-6,
          'gen-5: validator makespan = CP-SAT objective')
    check(np.array_equal(r['assigned_mch'], fixed), 'gen-5: CP-SAT stays on the line')
    v2 = validate_schedule(jl, pt, meta['time_lag'], r['assigned_mch'], r['op_ct'])
    check(v2['feasible'], 'gen-5: flow-line CP-SAT schedule feasible for the flexible instance')


def test_byproducts_hand():
    from schedule_validator import _build_base_instance
    jl, pt, lag = _build_base_instance()
    asg = np.array([0, 2, 0, 0, 1])
    ct = np.array([2.0, 5.0, 10.0, 3.0, 6.0])
    cp = bp.critical_path_lengths(jl, pt, lag)
    # job0: min pt 2 + 2 + 3 = 7, lags 1 + 2 = 3 -> 10; job1: 1 + 2 = 3, lag 1 -> 4
    check(np.allclose(cp, [10.0, 4.0]), f'hand: critical paths {cp.tolist()} == [10, 4]')
    due = bp.due_dates(cp, 0.9)          # [9.0, 3.6] -> job0 late by 1, job1 by 2.4
    r = bp.compute_byproducts(jl, pt, lag, asg, ct, np.array([0, 0, 1]), due)
    exp = {'makespan': 10.0, 'flow_mean': 7.0, 'flow_max': 10.0, 'wip_mean': 1.4,
           'util_mean': 1.0 / 3.0, 'lag_hold': 4.0, 'queue_wait': 0.0,
           'buffer_time': 4.0, 'tard_total': 3.4, 'tard_max': 2.4, 'n_tardy': 2}
    for k, v in exp.items():
        check(abs(r[k] - v) < 1e-9, f'hand: {k} = {r[k]:.4g} (expected {v:.4g})')
    check(abs(r['util_by_type'][0] - 0.4) < 1e-9 and abs(r['util_by_type'][1] - 0.2) < 1e-9,
          f'hand: utilization by type {r["util_by_type"]} == {{0: 0.4, 1: 0.2}}')
    r2 = bp.compute_byproducts(jl, pt, lag, asg, ct, np.array([0, 0, 1]),
                               bp.due_dates(cp, 100.0))
    check(r2['tard_total'] == 0 and r2['n_tardy'] == 0, 'hand: factor 100 -> no tardiness')


def test_byproducts_bounds():
    ep, stems = _stems()
    from ppvc_instance_generator import load_instance
    ppo, use_lag, use_type, _ = bp.load_policy(ep, '10x25+ppvc-mixed+full', 50)
    methods = ['greedy'] + list(ep.PDR_METHODS)
    worst = {'util_max': 0.0, 'util_mean_min': 1.0, 'wip_over_J': 0.0,
             'flow_slack_min': np.inf}
    ok = {'util': True, 'wip': True, 'flow': True, 'tard': True}
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        cp = bp.critical_path_lengths(jl, pt, meta['time_lag'])
        big = bp.due_dates(cp, 100.0)
        for m in methods:
            ms, _, asg, ct = bp.run_validated(ep, m, jl, pt, meta, i, 50,
                                              (ppo, use_lag, use_type))
            r = bp.compute_byproducts(jl, pt, meta['time_lag'], asg, ct,
                                      meta['mch_type'], big)
            util = np.zeros(pt.shape[1])
            np.add.at(util, asg, pt[np.arange(len(asg)), asg])
            util /= r['makespan']
            ok['util'] &= bool((util >= 0).all() and (util <= 1 + 1e-12).all()
                               and 0 < r['util_mean'] <= 1)
            worst['util_max'] = max(worst['util_max'], util.max())
            worst['util_mean_min'] = min(worst['util_mean_min'], r['util_mean'])
            ok['wip'] &= r['wip_mean'] <= len(jl) + 1e-12
            worst['wip_over_J'] = max(worst['wip_over_J'], r['wip_mean'] / len(jl))
            own = [pt[s:e][np.arange(e - s), asg[s:e]].sum()
                   + meta['time_lag'][s:e - 1].sum() for s, e in bp.job_ranges(jl)]
            slack = np.array(r['flow_time']) - np.array(own)
            ok['flow'] &= bool((slack >= -1e-9).all())
            ok['flow'] &= bool((np.array(r['flow_time']) >= cp - 1e-9).all())
            worst['flow_slack_min'] = min(worst['flow_slack_min'], slack.min())
            ok['tard'] &= r['tard_total'] == 0 and r['n_tardy'] == 0
    tag = f'{len(stems)} instances x {len(methods)} methods'
    check(ok['util'], f'{tag}: every station utilization in [0, 1], mean in (0, 1] '
          f'(max station {worst["util_max"]:.3f}, min mean {worst["util_mean_min"]:.3f})')
    check(ok['wip'], f'{tag}: mean WIP <= modules (max WIP/J {worst["wip_over_J"]:.3f})')
    check(ok['flow'], f'{tag}: flow time >= own processing + lags and >= CP_j '
          f'(min slack {worst["flow_slack_min"]:.1f} h)')
    check(ok['tard'], f'{tag}: due factor 100 -> zero tardiness')


def test_cli():
    tmp = tempfile.mkdtemp(prefix='flowline_byproducts_')
    n = ARGS.max_instances
    ds = ARGS.dataset
    rows, written = bp.main(['--data_path', f'data/PPVC/{ds}', '--max_instances',
                             str(n), '--save_dir', tmp, '--quiet'])
    check(all(os.path.exists(p) for p in written) and len(written) == 10,
          f'byproducts CLI wrote {len(written)} files (9 JSON + summary)')
    d = json.load(open(os.path.join(tmp, 'byproducts_greedy.json')))
    check(d['n_instances'] == n and len(d['per_instance']) == n,
          'byproducts JSON has one record per instance')
    ref = f'test_results/PPVC/{ds}/Result_greedy+10x25+ppvc-mixed+full_{ds}.npy'
    if os.path.exists(ref):
        a = np.load(ref)[:n, 0]
        mine = [r['makespan'] for r in d['per_instance']]
        check(np.allclose(a, mine), f'byproducts greedy makespans {mine} equal the '
              f'main-table file')
    js = fl.main(['--data_path', f'data/PPVC/{ds}', '--max_instances', str(n),
                  '--save_dir', tmp, '--cpsat_time_limit', str(ARGS.cpsat_limit),
                  '--quiet'])
    spt = np.load(os.path.join(tmp, f'Result_SPT+flowline_{ds}.npy'))
    cps = np.load(os.path.join(tmp, f'Result_CPSAT+flowline_{ds}.npy'))
    check(spt.shape == (n, 2) and cps.shape == (n, 2), 'flowline npy files [N, 2]')
    check(bool((cps[:, 0] <= spt[:, 0] + 1e-9).all()) or
          any(r.get('cpsat_flowline_status') != 'OPTIMAL' for r in js['per_instance']),
          'flow-line CP-SAT <= flow-line SPT wherever CP-SAT is optimal')
    check(os.path.exists(os.path.join(tmp, f'summary_flowline_{ds}.md')),
          'flowline summary written')
    for c in js['cross_checks']:
        print(f'  info cross-check {c}')
    print(f'  info outputs in {tmp}')


TESTS = [('flowline_restriction', test_flowline_restriction),
         ('flowline_cpsat', test_flowline_cpsat),
         ('byproducts_hand', test_byproducts_hand),
         ('byproducts_bounds', test_byproducts_bounds),
         ('cli', test_cli)]

if __name__ == '__main__':
    for name, fn in TESTS:
        if ARGS.only and name not in ARGS.only:
            continue
        t0 = time.time()
        print(f'[{name}]', flush=True)
        fn()
        print(f'  ({time.time() - t0:.1f} s)', flush=True)
    print()
    for n_ in NOTES:
        print(f'NOTE {n_}')
    if FAILS:
        print(f'{len(FAILS)} CHECK(S) FAILED')
        sys.exit(1)
    print('ALL CHECKS PASSED')
