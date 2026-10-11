"""
eval_flowline.py
================
Flow-line restriction of the PPVC instances: how much makespan the routing
flexibility of the flexible job shop is worth.

FLOW-LINE RULE
  Every operation keeps only ONE compatible station of its station type. The
  stations of type t are taken in increasing index order S_t[0..n_t-1];
  module j uses station S_t[j mod n_t] for every operation of type t. This
  mimics n_t fixed production lines without routing flexibility (module j
  always stays on line j mod n_t at a multi-station type). The operation
  keeps the processing time it has on that station in the flexible instance;
  all other columns of op_pt are set to 0 (incompatible). Lags, routes and
  station counts are unchanged, so every flow-line schedule is also feasible
  for the flexible instance.

COMPARISON (per instance, all schedules validated against the true lags)
  flow-line  SPT      lag-aware SPT rule on the flow-line instance
  flow-line  CP-SAT   the lag-aware CP-SAT model (same formulation as
                      ortools_solver.fjsp_solver) on the flow-line instance,
                      --cpsat_time_limit seconds, --cpsat_workers workers,
                      cold start by default (--cpsat_warm hints the
                      flow-line SPT schedule)
  flexible   SPT      the same SPT rule on the flexible instance (re-run with
                      eval_ppvc's seeds; checked against an existing
                      Result_SPT+*_<dataset>.npy when one exists)
  flexible   CP-SAT   the reference makespan_lag of
                      or_solution/PPVC/<dataset>.jsonl, when present
  flexible   greedy   the learned policy (default 10x25+ppvc-mixed+full, A3)

  flexibility gain (%) = 100 x (flow-line makespan - flexible makespan)
                         / flow-line makespan, per instance, then averaged.
  Pairs: SPT vs SPT, CP-SAT vs CP-SAT (reference coverage only), and the
  greedy policy vs the better of the two flow-line schedules.

OUTPUT (default directory test_results/PPVC/<dataset>/flowline/; a sub-folder
so that analyze_results.py, which globs Result_*.npy in the dataset folder,
does not mistake the flow-line SPT for the flexible SPT)
  Result_SPT+flowline_<dataset>.npy      [N, 2] (makespan, seconds)
  Result_CPSAT+flowline_<dataset>.npy    [N, 2] (makespan, solve seconds)
  flowline_per_instance_<dataset>.json   every column, CP-SAT status and bound
  summary_flowline_<dataset>.md          means and flexibility gains

CONTENTION EVIDENCE
  The flow-line CP-SAT solve stops at --cpsat_time_limit, so its result
  depends on the CPU it received. Every per-instance record holds
  cpsat_flowline_timing, the solve's evidence from
  eval_stochastic_lags.cpsat_timing (solve_wall_s, solve_cpu_s, workers,
  cpu_share = CPU s / (wall s x workers), status, time_limit_s, hit_limit),
  and the record-level fields of contention_fields (cpu_share_min,
  n_timed_solves, n_hit_limit, n_low_share, n_no_evidence, affinity,
  loadavg_start). The JSON key "contention" and a table in the summary give
  the run-level view; a budget-limited solve with cpu_share below 0.9 ran on
  shared cores and is not a valid result.

CLI
  python eval_flowline.py --data_path data/PPVC/10x25+ppvc-mixed \
      [--model_name 10x25+ppvc-mixed+full] [--cpsat_time_limit 300] \
      [--cpsat_workers 2] [--cpsat_warm] [--no_cpsat] [--no_greedy] \
      [--seed_test 50] [--max_instances 0] [--save_dir DIR]

Run pinned on the CPU with one thread per runtime, for example
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 python eval_flowline.py ...
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from eval_stochastic_lags import (SolveClock, contention_fields,
                                  contention_summary, contention_table,
                                  cpsat_timing, loadavg_now)


# ---------------------------------------------------------------------------
# Pure functions (numpy only)
# ---------------------------------------------------------------------------

def flowline_stations(job_length, op_station, mch_type):
    """Fixed station per operation under the dedicated-line rule.

    op_station[i] is the station-type index of op i and mch_type[m] the type of
    station m. Module j uses the (j mod n_t)-th station of type t."""
    op_station = np.asarray(op_station, dtype=int)
    mch_type = np.asarray(mch_type, dtype=int)
    by_type = {int(t): np.where(mch_type == t)[0] for t in np.unique(mch_type)}
    fixed = np.empty(len(op_station), dtype=int)
    idx = 0
    for j, n in enumerate(np.asarray(job_length, dtype=int)):
        for i in range(idx, idx + int(n)):
            st = by_type[int(op_station[i])]
            fixed[i] = st[j % len(st)]
        idx += int(n)
    return fixed


def flowline_restrict(job_length, op_pt, meta):
    """Return (op_pt_flowline, fixed_station) for one instance."""
    op_pt = np.asarray(op_pt)
    fixed = flowline_stations(job_length, meta['op_station'], meta['mch_type'])
    rows = np.arange(op_pt.shape[0])
    assert (op_pt[rows, fixed] > 0).all(), \
        'fixed station incompatible with its operation'
    pt_fl = np.zeros_like(op_pt)
    pt_fl[rows, fixed] = op_pt[rows, fixed]
    return pt_fl, fixed


def mean_compatible(op_pt):
    """Mean number of compatible stations per operation."""
    return float((np.asarray(op_pt) > 0).sum(axis=1).mean())


def gain_pct(flowline, flexible):
    """Per-instance flexibility gain in percent of the flow-line makespan."""
    fl = np.asarray(flowline, dtype=float)
    fx = np.asarray(flexible, dtype=float)
    return 100.0 * (fl - fx) / fl


# ---------------------------------------------------------------------------
# CP-SAT on one instance
# ---------------------------------------------------------------------------

def solve_cpsat(jl, pt, time_lag, time_limit, num_workers, hint=None,
                random_seed=0):
    """Lag-aware CP-SAT (formulation of ortools_solver.fjsp_solver).

    hint = (assigned_mch, op_ct) of a feasible schedule, or None.
    Returns dict(status, makespan, bound, seconds, assigned_mch, op_ct,
    timing), timing being the solve's contention evidence (cpsat_timing)."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from ortools.sat.python import cp_model
        from ortools_solver import matrix_to_the_format_for_solving
        import eval_cpsat_anytime as ca
    finally:
        sys.argv = saved
    jobs, n_m = matrix_to_the_format_for_solving(np.asarray(jl), np.asarray(pt))
    lm = ca.build_lag_model(jobs, n_m, time_lag)
    if hint is not None:
        ca.add_full_hint(lm, jobs, hint[0], hint[1])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.random_seed = int(random_seed)
    t0 = time.perf_counter()
    with SolveClock() as clk:
        status = solver.Solve(lm.model)
    sec = time.perf_counter() - t0
    out = {'status': solver.StatusName(status), 'seconds': sec,
           'bound': float(solver.BestObjectiveBound()),
           'makespan': None, 'assigned_mch': None, 'op_ct': None,
           'timing': cpsat_timing(solver, status, clk)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        asg, ct = ca.extract_schedule(solver, lm, jobs)
        out.update(makespan=float(solver.ObjectiveValue()),
                   assigned_mch=asg, op_ct=ct)
    return out


def existing_result(dataset, method, exclude_tags=('flowline',)):
    """First Result_<method>+*_<dataset>.npy of the dataset folder (sorted),
    preferring a file whose tag has no 'smoke'; None if there is none."""
    files = sorted(glob.glob(os.path.join(
        'test_results', 'PPVC', dataset, f'Result_{method}+*_{dataset}.npy')))
    files = [f for f in files
             if not any(t in os.path.basename(f) for t in exclude_tags)]
    if not files:
        return None, None
    files.sort(key=lambda f: 'smoke' in os.path.basename(f))
    return files[0], np.load(files[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description='Flow-line restriction vs flexible job shop (PPVC)')
    ap.add_argument('--data_path', default='data/PPVC/10x25+ppvc-mixed')
    ap.add_argument('--model_name', default='10x25+ppvc-mixed+full')
    ap.add_argument('--cpsat_time_limit', type=float, default=300.0,
                    help='seconds per flow-line instance (the flexible '
                         'reference used 300 s)')
    ap.add_argument('--cpsat_workers', type=int, default=2)
    ap.add_argument('--cpsat_warm', action='store_true',
                    help='hint the flow-line SPT schedule to CP-SAT')
    ap.add_argument('--no_cpsat', action='store_true')
    ap.add_argument('--no_greedy', action='store_true')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=0,
                    help='first N instances only (0 = all); a partial run '
                         'needs --save_dir so it cannot overwrite full results')
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    if args.max_instances > 0 and args.save_dir is None:
        ap.error('--max_instances needs --save_dir (partial runs must not '
                 'overwrite the full-set outputs)')
    if args.cpsat_workers < 1:
        ap.error('--cpsat_workers must be >= 1 (0 would use every core)')
    return args


def _ms(x):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f'{x:.1f}'


def main(argv=None):
    args = parse_cli(argv)
    from eval_byproducts import import_eval_ppvc, load_policy, run_validated
    ep = import_eval_ppvc()
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule

    policy, ckpt = None, None
    if not args.no_greedy:
        ppo, use_lag, use_type, ckpt = load_policy(ep, args.model_name,
                                                   args.seed_test)
        policy = (ppo, use_lag, use_type)

    stems = ep.list_instances(args.data_path)
    if args.max_instances > 0:
        stems = stems[:args.max_instances]
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    save_dir = args.save_dir or os.path.join('test_results', 'PPVC', dataset,
                                             'flowline')
    os.makedirs(save_dir, exist_ok=True)
    ref, ref_path = ep.load_or_reference(args.data_path)
    ref_status = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref_status[rec.get('instance')] = rec.get('status_lag')
    ref_tl = None
    if ref:
        with open(ref_path) as f:
            ref_tl = json.loads(f.readline()).get('time_limit_s')

    res_spt_fl = np.zeros((n, 2))
    res_cp_fl = np.full((n, 2), np.nan)
    per = []
    t0 = time.time()
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        la_start = loadavg_now()
        lag = meta['time_lag']
        pt_fl, fixed = flowline_restrict(jl, pt, meta)
        rec = {'instance': base,
               'compat_flexible': mean_compatible(pt),
               'compat_flowline': mean_compatible(pt_fl)}

        # flow-line SPT, validated on the flow-line AND the flexible instance
        ms, sec, asg, ct = run_validated(ep, 'SPT', jl, pt_fl, meta, i,
                                         args.seed_test)
        assert np.array_equal(asg, fixed), 'flow-line SPT left its line'
        assert validate_schedule(jl, pt, lag, asg, ct)['feasible']
        res_spt_fl[i] = [ms, sec]
        rec['spt_flowline'] = ms
        spt_fl_sched = (asg, ct)

        # flow-line CP-SAT
        if not args.no_cpsat:
            cp = solve_cpsat(jl, pt_fl, lag, args.cpsat_time_limit,
                             args.cpsat_workers,
                             hint=spt_fl_sched if args.cpsat_warm else None)
            rec.update(cpsat_flowline=cp['makespan'],
                       cpsat_flowline_status=cp['status'],
                       cpsat_flowline_bound=cp['bound'],
                       cpsat_flowline_s=cp['seconds'],
                       cpsat_flowline_timing=cp['timing'])
            if cp['makespan'] is not None:
                v = validate_schedule(jl, pt_fl, lag, cp['assigned_mch'],
                                      cp['op_ct'])
                assert v['feasible'], (f'flow-line CP-SAT infeasible on '
                                       f'{base}: {v["violations"][:3]}')
                assert abs(v['makespan'] - cp['makespan']) < 1e-6
                res_cp_fl[i] = [cp['makespan'], cp['seconds']]

        # flexible SPT and greedy
        ms_fx, sec_fx, _, _ = run_validated(ep, 'SPT', jl, pt, meta, i,
                                            args.seed_test)
        rec['spt_flexible'] = ms_fx
        if policy is not None:
            ms_g, sec_g, _, _ = run_validated(ep, 'greedy', jl, pt, meta, i,
                                              args.seed_test, policy)
            rec['greedy_flexible'] = ms_g
            rec['greedy_flexible_s'] = sec_g
        rec['cpsat_flexible_ref'] = ref.get(base)
        rec['cpsat_flexible_ref_status'] = ref_status.get(base)
        fl_best = [x for x in (rec['spt_flowline'], rec.get('cpsat_flowline'))
                   if x is not None]
        rec['best_flowline'] = min(fl_best)
        timed = [rec['cpsat_flowline_timing']] if not args.no_cpsat else []
        rec.update(contention_fields(timed, la_start))
        per.append(rec)
        if not args.quiet:
            print(f'  [{i + 1:3d}/{n}] {base}  SPT fl={rec["spt_flowline"]:.0f}'
                  f' fx={ms_fx:.0f}  CP-SAT fl={_ms(rec.get("cpsat_flowline"))}'
                  f' ({rec.get("cpsat_flowline_status", "-")})'
                  f' fx-ref={_ms(rec["cpsat_flexible_ref"])}'
                  + (f'  greedy fx={rec["greedy_flexible"]:.0f}'
                     if policy is not None else '')
                  + (f'  cpu_share={rec["cpsat_flowline_timing"]["cpu_share"]:.2f}'
                     if not args.no_cpsat else ''), flush=True)
    wall = time.time() - t0

    # cross-check the flexible SPT / greedy against the main-table files
    checks = []
    for method, key in (('SPT', 'spt_flexible'), ('greedy', 'greedy_flexible')):
        if method == 'greedy':
            path = os.path.join('test_results', 'PPVC', dataset,
                                f'Result_greedy+{args.model_name}_{dataset}.npy')
            arr = np.load(path) if os.path.exists(path) else None
            if arr is None or policy is None:
                path = None
        else:
            path, arr = existing_result(dataset, 'SPT')
        if path is None:
            checks.append(f'{method}: no existing result file to compare')
            continue
        mine = np.array([r[key] for r in per])
        k = min(len(mine), len(arr))
        same = int((np.abs(arr[:k, 0] - mine[:k]) < 1e-6).sum())
        checks.append(f'{method}: {same}/{k} instances equal to `{path}`')

    # outputs
    np.save(os.path.join(save_dir, f'Result_SPT+flowline_{dataset}.npy'),
            res_spt_fl)
    if not args.no_cpsat:
        np.save(os.path.join(save_dir, f'Result_CPSAT+flowline_{dataset}.npy'),
                res_cp_fl)
    js = {'dataset': dataset, 'n_instances': n,
          'rule': 'module j uses station S_t[j mod n_t] of each type t',
          'cpsat_time_limit_s': args.cpsat_time_limit,
          'cpsat_workers': args.cpsat_workers,
          'cpsat_start': 'warm (flow-line SPT hint)' if args.cpsat_warm
          else 'cold',
          'flexible_reference': ref_path if ref else None,
          'flexible_reference_time_limit_s': ref_tl,
          'model': None if policy is None else args.model_name,
          'checkpoint': ckpt, 'seed_test': args.seed_test,
          'cross_checks': checks, 'wall_s': wall, 'per_instance': per,
          'contention': contention_summary(per)}
    with open(os.path.join(save_dir,
                           f'flowline_per_instance_{dataset}.json'), 'w') as f:
        json.dump(js, f, indent=1)

    def col(key):
        return np.array([np.nan if r.get(key) is None else r[key]
                         for r in per], dtype=float)

    rows = []

    def pair(label, fl_key, fx_key):
        fl, fx = col(fl_key), col(fx_key)
        ok = ~np.isnan(fl) & ~np.isnan(fx)
        if not ok.any():
            rows.append(f'| {label} | n/a | n/a | n/a | 0/{n} | n/a |')
            return
        g = gain_pct(fl[ok], fx[ok])
        rows.append(f'| {label} | {fl[ok].mean():.1f} | {fx[ok].mean():.1f} '
                    f'| {g.mean():.2f} +/- {g.std():.2f} | {ok.sum()}/{n} | '
                    f'{int((fx[ok] < fl[ok]).sum())}/{ok.sum()} |')

    pair('SPT (flow line) vs SPT (flexible)', 'spt_flowline', 'spt_flexible')
    if not args.no_cpsat:
        pair('CP-SAT (flow line) vs CP-SAT reference (flexible)',
             'cpsat_flowline', 'cpsat_flexible_ref')
    if policy is not None:
        pair('best flow-line schedule vs greedy policy (flexible)',
             'best_flowline', 'greedy_flexible')

    st_fl = {}
    for r in per:
        s = r.get('cpsat_flowline_status')
        if s:
            st_fl[s] = st_fl.get(s, 0) + 1
    st_ref = {}
    for r in per:
        s = r.get('cpsat_flexible_ref_status')
        if s:
            st_ref[s] = st_ref.get(s, 0) + 1
    lines = [f'# Flow-line restriction vs flexible job shop: `{dataset}`', '',
             f'- instances: {n}; every schedule validated against the true lags',
             '- flow-line rule: every operation is fixed to one station of its '
             'type; module j uses the (j mod n_t)-th station of type t (n_t '
             'stations of that type), with the processing time it has there',
             f'- compatible stations per operation: flexible '
             f'{np.mean([r["compat_flexible"] for r in per]):.2f}, flow line '
             f'{np.mean([r["compat_flowline"] for r in per]):.2f}',
             f'- flow-line CP-SAT: {args.cpsat_time_limit:g} s, '
             f'{args.cpsat_workers} workers, '
             f'{"warm start from flow-line SPT" if args.cpsat_warm else "cold start"}'
             f'; status counts {st_fl}' if not args.no_cpsat
             else '- flow-line CP-SAT: not run',
             f'- flexible CP-SAT reference: `{ref_path}`'
             f' ({ref_tl} s; status counts {st_ref})' if ref
             else f'- flexible CP-SAT reference: not available (`{ref_path}`)',
             f'- greedy policy: `{args.model_name}`' if policy is not None
             else '- greedy policy: not run',
             '- flexibility gain = 100 x (flow-line - flexible) / flow-line '
             'makespan, per instance; mean +/- std',
             '- cross-checks: ' + '; '.join(checks), '',
             '| comparison | flow-line makespan | flexible makespan | '
             'flexibility gain (%) | instances | flexible shorter |',
             '|---|---|---|---|---|---|'] + rows
    if not args.no_cpsat:
        lines += [''] + contention_table(
            {'flow-line CP-SAT': js['contention']})
    md = os.path.join(save_dir, f'summary_flowline_{dataset}.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if not args.quiet:
        print('\n'.join(lines))
        print(f'\n{n} instances in {wall:.1f} s; wrote {save_dir}/')
    return js


if __name__ == '__main__':
    main()
