"""
eval_byproducts.py
==================
Secondary (non-makespan) objectives measured on the makespan schedules that
the learned policy and the dispatching rules produce on a PPVC instance set.

Every schedule comes from the same validated rollouts as eval_ppvc.py
(rollout_greedy / rollout_heuristic, same seeds), is re-validated against the
TRUE post-operation lags with schedule_validator.validate_schedule, and only
then measured. No method is re-optimised for these objectives; the numbers
describe what a makespan-driven schedule delivers on the other criteria.

METRICS (per instance; all times in hours, release time 0 for every module)
  makespan            max completion time over all operations
  flow time           per module: completion of its last operation minus the
                      start of its first operation; reported as the mean and
                      the max over modules
  mean WIP            time-average number of modules in process over
                      [0, makespan]; a module is in process from the start of
                      its first operation to the completion of its last, so
                      mean WIP = sum of module flow times / makespan
  station utilization busy time of a station / makespan; reported as the mean
                      over all stations and the mean over the stations of each
                      station type (a type no module visits scores 0)
  lag hold            sum of the post-operation lags of every non-terminal
                      operation (curing, ponding, drying); fixed by the
                      instance, the same for every method
  queue wait          sum over consecutive operation pairs of
                      (successor start - predecessor completion - lag), i.e.
                      the time a module waits for a station after its lag
                      has elapsed
  buffer time         lag hold + queue wait = total time modules spend off a
                      station between two of their operations
  tardiness           due date of module j: d_j = factor_j x CP_j, where CP_j
                      is the module's own lag-inclusive critical path (sum of
                      its minimum processing time over compatible stations
                      plus the lags of its non-terminal operations) and
                      factor_j = --due_factor (default 1.5). With
                      --due_jitter r > 0, factor_j = due_factor x U(1-r, 1+r),
                      drawn from a generator seeded with
                      --due_seed + instance index. T_j = max(0, C_j - d_j);
                      reported as the total, the max and the number of tardy
                      modules

OUTPUT (default directory test_results/PPVC/<dataset>/)
  byproducts_<method>.json   provenance, definitions, instance means and
                             per-instance values
  byproducts_summary.md      one row per method (means over instances)

CLI
  python eval_byproducts.py --data_path data/PPVC/10x25+ppvc-mixed \
      [--model_name 10x25+ppvc-mixed+full] \
      [--methods greedy FIFO MOR SPT MWKR SPTL LLF MWKRL LRPTL] \
      [--due_factor 1.5] [--due_jitter 0.0] [--due_seed 0] \
      [--seed_test 50] [--max_instances 0] [--save_dir DIR]

Run pinned on the CPU with one thread per runtime, for example
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 python eval_byproducts.py ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_METHODS = ('greedy',)


# ---------------------------------------------------------------------------
# Pure metric functions (numpy only; importable without the project config)
# ---------------------------------------------------------------------------

def job_ranges(job_length):
    """[(first_op, one_past_last_op)] per job in the flat op order."""
    out, idx = [], 0
    for n in np.asarray(job_length, dtype=int):
        out.append((idx, idx + int(n)))
        idx += int(n)
    return out


def critical_path_lengths(job_length, op_pt, time_lag):
    """Per-module lag-inclusive critical path: sum of the minimum compatible
    processing time of every operation plus the lags of the non-terminal
    operations (the lag of the last operation delays no successor)."""
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    min_pt = np.where(op_pt > 0, op_pt, np.inf).min(axis=1)
    assert np.isfinite(min_pt).all(), 'an operation has no compatible station'
    cp = []
    for s, e in job_ranges(job_length):
        cp.append(float(min_pt[s:e].sum() + time_lag[s:e - 1].sum()))
    return np.array(cp)


def due_dates(cp, factor, jitter=0.0, rng=None):
    """d_j = factor_j x CP_j (release 0). factor_j = factor unless jitter > 0."""
    cp = np.asarray(cp, dtype=float)
    if jitter > 0:
        assert rng is not None, 'jittered due dates need a seeded generator'
        f = factor * rng.uniform(1.0 - jitter, 1.0 + jitter, size=cp.shape)
    else:
        f = np.full(cp.shape, float(factor))
    return f * cp


def compute_byproducts(job_length, op_pt, time_lag, assigned_mch, op_ct,
                       mch_type, due):
    """All secondary metrics of one validated schedule (see module docstring).

    Returns a dict of scalars plus 'util_by_type' {type index: utilization}
    and per-module arrays 'flow_time', 'tardiness' (as lists).
    """
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    asg = np.asarray(assigned_mch, dtype=int)
    ct = np.asarray(op_ct, dtype=float)
    mch_type = np.asarray(mch_type, dtype=int)
    n_ops, n_m = op_pt.shape
    pt_used = op_pt[np.arange(n_ops), asg]
    st = ct - pt_used
    makespan = float(ct.max())

    flow, comp, lag_hold, queue = [], [], 0.0, 0.0
    for s, e in job_ranges(job_length):
        flow.append(float(ct[e - 1] - st[s]))
        comp.append(float(ct[e - 1]))
        gaps = st[s + 1:e] - ct[s:e - 1]
        lag_hold += float(time_lag[s:e - 1].sum())
        queue += float((gaps - time_lag[s:e - 1]).sum())
    flow = np.array(flow)
    comp = np.array(comp)

    busy = np.zeros(n_m)
    np.add.at(busy, asg, pt_used)
    util = busy / makespan
    util_by_type = {int(t): float(util[mch_type == t].mean())
                    for t in np.unique(mch_type)}

    tard = np.maximum(0.0, comp - np.asarray(due, dtype=float))
    return {
        'makespan': makespan,
        'flow_mean': float(flow.mean()),
        'flow_max': float(flow.max()),
        'wip_mean': float(flow.sum() / makespan),
        'util_mean': float(util.mean()),
        'util_min_station': float(util.min()),
        'util_max_station': float(util.max()),
        'lag_hold': lag_hold,
        'queue_wait': queue,
        'buffer_time': lag_hold + queue,
        'tard_total': float(tard.sum()),
        'tard_max': float(tard.max()),
        'n_tardy': int((tard > 1e-9).sum()),
        'util_by_type': util_by_type,
        'flow_time': flow.tolist(),
        'tardiness': tard.tolist(),
    }


SCALAR_KEYS = ('makespan', 'flow_mean', 'flow_max', 'wip_mean', 'util_mean',
               'util_min_station', 'util_max_station', 'lag_hold',
               'queue_wait', 'buffer_time', 'tard_total', 'tard_max',
               'n_tardy')


# ---------------------------------------------------------------------------
# Runtime helpers shared with eval_flowline.py
# ---------------------------------------------------------------------------

def import_eval_ppvc():
    """Import eval_ppvc with a clean argv (it and params.py parse sys.argv at
    import time), force the CPU, and restore argv afterwards."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        import eval_ppvc
    finally:
        sys.argv = saved
    from params import configs
    configs.device = 'cpu'
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    return eval_ppvc


def load_policy(ep, model_name, seed_test):
    """Load the checkpoint on the CPU exactly as eval_ppvc.main does.

    Returns (ppo, use_lag_features, use_type_embedding, ckpt_path)."""
    import torch
    from params import configs
    from common_utils import setup_seed
    ep.load_train_config(model_name)
    configs.device = 'cpu'
    use_lag = bool(getattr(configs, 'use_lag_features', False))
    use_type = bool(getattr(configs, 'use_type_embedding', False))
    setup_seed(seed_test)
    from model.PPO import PPO_initialize
    ckpt = f'./trained_network/PPVC/{model_name}.pth'
    if not os.path.exists(ckpt):
        sys.exit(f'[byproducts] checkpoint not found: {ckpt}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt, map_location='cpu'))
    ppo.policy.eval()
    return ppo, use_lag, use_type, ckpt


def run_validated(ep, method, jl, pt, meta, idx, seed_test, policy=None):
    """One rollout with eval_ppvc's seeds, re-validated against the true lags.

    Returns (makespan, seconds, assigned_mch, op_ct)."""
    from schedule_validator import validate_schedule
    if method == 'greedy':
        ppo, use_lag, use_type = policy
        ms, sec, asg, ct = ep.rollout_greedy(ppo, jl, pt, meta, use_lag,
                                             use_type)
    elif method in ep.PDR_METHODS:
        ms, sec, asg, ct = ep.rollout_heuristic(method, jl, pt, meta,
                                                seed=seed_test + idx)
    else:
        raise ValueError(f'unknown method {method}')
    res = validate_schedule(jl, pt, meta['time_lag'], asg, ct)
    assert res['feasible'], (f'{method} schedule infeasible:\n  '
                             + '\n  '.join(res['violations'][:10]))
    assert abs(res['makespan'] - ms) < 1e-6, (
        f'{method}: env makespan {ms} != validator {res["makespan"]}')
    return ms, sec, asg, ct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description='Secondary objectives of validated PPVC schedules')
    ap.add_argument('--data_path', default='data/PPVC/10x25+ppvc-mixed')
    ap.add_argument('--model_name', default='10x25+ppvc-mixed+full',
                    help='checkpoint for the greedy policy (A3 by default)')
    ap.add_argument('--methods', nargs='+', default=None,
                    help='greedy and/or any dispatching rule of '
                         'common_utils.heuristic_select_action (default: '
                         'greedy + every rule)')
    ap.add_argument('--due_factor', type=float, default=1.5)
    ap.add_argument('--due_jitter', type=float, default=0.0,
                    help='relative half-width of a per-module factor '
                         'U(1-r, 1+r); 0 = deterministic due dates')
    ap.add_argument('--due_seed', type=int, default=0)
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
    if args.due_factor <= 0 or not (0.0 <= args.due_jitter < 1.0):
        ap.error('--due_factor must be > 0 and 0 <= --due_jitter < 1')
    return args


def _agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'min': float(v.min()), 'max': float(v.max())}


def main(argv=None):
    args = parse_cli(argv)
    ep = import_eval_ppvc()
    from ppvc_instance_generator import load_instance, STATION_TYPES

    methods = args.methods or ['greedy'] + list(ep.PDR_METHODS)
    for m in methods:
        if m not in MODEL_METHODS and m not in ep.PDR_METHODS:
            sys.exit(f'[byproducts] unknown method {m}; choose from '
                     f'{MODEL_METHODS + tuple(ep.PDR_METHODS)}')
    policy, ckpt = None, None
    if 'greedy' in methods:
        ppo, use_lag, use_type, ckpt = load_policy(ep, args.model_name,
                                                   args.seed_test)
        policy = (ppo, use_lag, use_type)

    stems = ep.list_instances(args.data_path)
    if args.max_instances > 0:
        stems = stems[:args.max_instances]
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    save_dir = args.save_dir or f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    rows = {m: [] for m in methods}
    t0 = time.time()
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        cp = critical_path_lengths(jl, pt, meta['time_lag'])
        rng = (np.random.default_rng(args.due_seed + i)
               if args.due_jitter > 0 else None)
        due = due_dates(cp, args.due_factor, args.due_jitter, rng)
        for m in methods:
            ms, sec, asg, ct = run_validated(ep, m, jl, pt, meta, i,
                                             args.seed_test, policy)
            r = compute_byproducts(jl, pt, meta['time_lag'], asg, ct,
                                   meta['mch_type'], due)
            r['instance'] = os.path.basename(stem)
            r['rollout_s'] = float(sec)
            r['critical_path'] = cp.tolist()
            r['due_date'] = due.tolist()
            rows[m].append(r)
        if not args.quiet and ((i + 1) % 10 == 0 or i == n - 1):
            print(f'  [{i + 1:3d}/{n}] ' + '  '.join(
                f'{m}={rows[m][-1]["makespan"]:.0f}h' for m in methods),
                flush=True)

    definitions = {
        'flow_time': 'last completion minus first start of the module (h)',
        'wip_mean': 'sum of module flow times / makespan',
        'util': 'station busy time / makespan',
        'lag_hold': 'sum of lags of non-terminal operations (h)',
        'queue_wait': 'successor start - predecessor completion - lag, '
                      'summed over consecutive pairs (h)',
        'buffer_time': 'lag_hold + queue_wait (h)',
        'due_date': f'factor x own lag-inclusive critical path (min '
                    f'processing time); factor={args.due_factor}, '
                    f'jitter={args.due_jitter}, seed={args.due_seed}+i',
    }
    type_names = {int(k): STATION_TYPES[int(k)] for k in range(len(STATION_TYPES))}
    # PLAN runs tag their file names; DANIEL names are unchanged
    from common_utils import backbone_file_tag
    btag = backbone_file_tag(args.model_name,
                             getattr(ep.configs, 'framework', 'DANIEL'))
    written = []
    for m in methods:
        rs = rows[m]
        types = sorted(rs[0]['util_by_type'])
        agg = {k: _agg(rs, k) for k in SCALAR_KEYS}
        agg['util_by_type'] = {
            type_names[t]: float(np.mean([r['util_by_type'][t] for r in rs]))
            for t in types}
        out = {
            'dataset': dataset, 'method': m, 'n_instances': n,
            'model': args.model_name if m == 'greedy' else None,
            'checkpoint': ckpt if m == 'greedy' else None,
            'seed_test': args.seed_test,
            'due_factor': args.due_factor, 'due_jitter': args.due_jitter,
            'due_seed': args.due_seed,
            'validated': True, 'definitions': definitions,
            'aggregate': agg,
            'per_instance': [
                {**{k: v for k, v in r.items() if k != 'util_by_type'},
                 'util_by_type': {type_names[t]: u
                                  for t, u in r['util_by_type'].items()}}
                for r in rs],
        }
        path = os.path.join(save_dir, f'byproducts_{m}{btag}.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        written.append(path)

    # markdown table: means over instances
    lines = [f'# Secondary objectives of makespan schedules: `{dataset}`', '',
             f'- instances: {n}; every schedule re-validated against the '
             f'true lags',
             f'- greedy policy: `{args.model_name}`; dispatching rules '
             f'seeded as eval_ppvc (seed_test={args.seed_test} + index)',
             f'- due date d_j = {args.due_factor} x CP_j (own lag-inclusive '
             f'critical path, min processing time; release 0)'
             + (f', factor jitter U(1-{args.due_jitter}, 1+{args.due_jitter}),'
                f' seed {args.due_seed}+index' if args.due_jitter > 0 else ''),
             '- values are means over instances; times in hours', '',
             '| method | makespan | mean flow | max flow | mean WIP | '
             'util. (mean) | lag hold | queue wait | total tard. | '
             'max tard. | tardy modules |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in methods:
        a = {k: _agg(rows[m], k)['mean'] for k in SCALAR_KEYS}
        lines.append(
            f'| {m} | {a["makespan"]:.1f} | {a["flow_mean"]:.1f} | '
            f'{a["flow_max"]:.1f} | {a["wip_mean"]:.2f} | '
            f'{a["util_mean"]:.3f} | {a["lag_hold"]:.1f} | '
            f'{a["queue_wait"]:.1f} | {a["tard_total"]:.1f} | '
            f'{a["tard_max"]:.1f} | {a["n_tardy"]:.2f} |')
    types = sorted(rows[methods[0]][0]['util_by_type'])
    lines += ['', '## Station utilization by type (mean over instances)', '',
              '| method | ' + ' | '.join(type_names[t] for t in types) + ' |',
              '|---|' + '---|' * len(types)]
    for m in methods:
        lines.append(f'| {m} | ' + ' | '.join(
            f'{np.mean([r["util_by_type"][t] for r in rows[m]]):.3f}'
            for t in types) + ' |')
    md = os.path.join(save_dir, f'byproducts_summary{btag}.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    written.append(md)
    if not args.quiet:
        print('\n'.join(lines))
        print(f'\n{n} instances x {len(methods)} methods in '
              f'{time.time() - t0:.1f} s')
        for p in written:
            print(f'  wrote {p}')
    return rows, written


if __name__ == '__main__':
    main()
