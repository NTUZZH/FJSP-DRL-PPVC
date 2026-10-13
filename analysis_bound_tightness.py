#!/usr/bin/env python3
"""analysis_bound_tightness.py -- tightness of the lag-augmented and the
lag-free completion-time bound along the greedy trajectories of one policy.

What it computes (evaluation only, no training, CPU)
  The SLAB policy of seed 301 (10x25+ppvc-mixed+a1lagx05-2k-s301) is rolled
  out greedily, one order at a time, in the default environment
  (FJSPEnvForSameOpNums, bound_lag_free=False). Its action sequence is
  replayed, action by action, in a second environment built with
  bound_lag_free=True; both environments must give the identical schedule.
  At every decision state s_t (t = 0 .. T-1, T = number of operations) and at
  the terminal state s_T the script records, in hours:
    aug_env   max(op_ct_lb) of the default env (the reward bound and state
              feature the policy was trained with), times the env's
              normalisation scale
    free_env  the same quantity in the replay env (bound_lag_free=True)
    aug_dir / free_dir
              both bounds recomputed directly from the true-unit state
              (realized completion of the last scheduled operation of each
              module, plus the minimum processing times and, for the
              lag-augmented bound, the waits of the remaining chain); they
              must equal aug_env / free_env
    aug_t / free_t
              Eq. (4) of the paper as printed, i.e. with the decision time
              t inside the max: c_hat = max(c_hat_prev + lag_prev, t) + p_min
              (the environment's bound has no t term; reported for
              comparison only)
  and the final makespan Cmax of the schedule.

Per order it stores Chat(s_0)/Cmax, the step average (1/T) sum_t
Chat(s_t)/Cmax over t = 0..T-1 for every bound, the step-averaged
difference (aug - free)/Cmax, and the checks: steps with free > aug, steps
with aug > Cmax + 1e-6 (Proposition 1 says none), and whether Chat(s_T) ==
Cmax for both bounds. Every schedule is re-validated with
schedule_validator.validate_schedule against the true waits, and its
makespan is compared with the stored greedy result of the same model (GPU,
Result_greedy+<model>_<set>.npy) and with the CPU run of Section S5
(byproducts_hstar/byproducts_greedy.json).

Outputs (new directory, nothing existing is overwritten)
  test_results/PPVC/bound_tightness/bound_tightness_<set>.json  per order
  test_results/PPVC/bound_tightness/bound_tightness.json         per size
  test_results/PPVC/bound_tightness/bound_tightness.md           table

usage
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 \
      python analysis_bound_tightness.py --set 80x25+ppvc-mixed
  python analysis_bound_tightness.py --summarize
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, 'test_results', 'PPVC', 'bound_tightness')
MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
SETS = ('10x25+ppvc-mixed', '20x25+ppvc-mixed', '30x25+ppvc-mixed',
        '50x25+ppvc-mixed', '80x25+ppvc-mixed')
SEED_TEST = 50
TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def chain_bounds(jl, min_pt, lag, ct, sched, t):
    """Direct true-unit bounds of one state.

    jl     ops per job; min_pt, lag: per op (true units)
    ct     realized completion of scheduled ops (true units)
    sched  bool per op, True if scheduled
    t      decision time (true units)
    Returns (aug, free, aug_t, free_t): max over jobs of the bound on the
    last operation of each job.
    """
    out = np.full(4, -np.inf)
    first = 0
    for n in jl:
        last = first + int(n) - 1
        s = sched[first:last + 1]
        k = int(s.sum())                # ops of the job already scheduled
        assert np.all(s[:k]) and not np.any(s[k:]), 'non-prefix schedule'
        if k == int(n):
            v = float(ct[last])
            vals = (v, v, v, v)
        else:
            j0 = first + k              # first unscheduled op
            pm = min_pt[j0:last + 1]
            if k == 0:
                base_aug = base_free = 0.0
                lag_in = lag[j0:last]   # lags of j0..last-1 enter after j0
                first_lag = 0.0
            else:
                c_prev = float(ct[j0 - 1])
                base_aug = c_prev
                base_free = c_prev
                first_lag = float(lag[j0 - 1])
                lag_in = lag[j0:last]
            rest_pm = float(pm[1:].sum())
            rest_lag = float(lag_in.sum())
            aug = base_aug + first_lag + float(pm[0]) + rest_pm + rest_lag
            free = base_free + float(pm[0]) + rest_pm
            aug_t = max(base_aug + first_lag, t) + float(pm[0]) + rest_pm + rest_lag
            free_t = max(base_free, t) + float(pm[0]) + rest_pm
            vals = (aug, free, aug_t, free_t)
        out = np.maximum(out, vals)
        first = last + 1
    return out


def true_decision_time(env):
    """min over eligible pairs of max(job ready, station free), true units."""
    pf = np.maximum(env.true_candidate_free_time[0][:, None],
                    env.true_mch_free_time[0][None, :])
    pf = np.where(env.candidate_process_relation[0], np.inf, pf)
    return float(pf.min())


def run_set(ds, max_orders=0, out_dir=OUT):
    from eval_byproducts import import_eval_ppvc, load_policy
    ep = import_eval_ppvc()
    import torch
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    from common_utils import greedy_select_action

    ppo, use_lag, use_type, ckpt = load_policy(ep, MODEL, SEED_TEST)
    from params import configs
    assert not bool(getattr(configs, 'bound_lag_free', False)), \
        'SLAB policy must be trained with the lag-augmented bound'
    assert not use_lag and not use_type, 'SLAB policy runs without the aids'
    n_par = sum(p.numel() for p in ppo.policy.parameters())

    stems = ep.list_instances(os.path.join(REPO, 'data', 'PPVC', ds))
    n_all = len(stems)
    gpu_res_path = os.path.join(REPO, 'test_results', 'PPVC', ds,
                                f'Result_greedy+{MODEL}_{ds}.npy')
    gpu = np.load(gpu_res_path)[:, 0]
    cpu_bp_path = os.path.join(REPO, 'test_results', 'PPVC', ds,
                               'byproducts_hstar', 'byproducts_greedy.json')
    cpu_bp = json.load(open(cpu_bp_path))
    assert cpu_bp['model'] == MODEL and cpu_bp['dataset'] == ds
    cpu = np.array([r['makespan'] for r in cpu_bp['per_instance']], float)
    assert n_all == len(gpu) == len(cpu)
    if max_orders > 0:
        stems = stems[:max_orders]

    orders = []
    t0 = time.time()
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        n_ops = int(pt.shape[0])
        envA, stA = ep._new_env(jl, pt, meta, use_lag, use_type,
                                bound_lag_free=False)
        envB, _ = ep._new_env(jl, pt, meta, use_lag, use_type,
                              bound_lag_free=True)
        assert envA.bound_lag_free is False and envB.bound_lag_free is True
        scaleA = envA.pt_upper_bound - envA.pt_lower_bound + 1e-8
        scaleB = envB.pt_upper_bound - envB.pt_lower_bound + 1e-8
        min_pt = np.where(pt > 0, pt, np.inf).min(axis=1).astype(float)
        lag = np.asarray(meta['time_lag'], dtype=float)
        rec = {k: [] for k in ('aug_env', 'free_env', 'aug_dir', 'free_dir',
                               'aug_t', 'free_t', 't')}
        assigned = np.full(n_ops, -1, dtype=int)
        M = pt.shape[1]
        state = stA
        max_env_dir_dev = 0.0
        for step in range(n_ops + 1):
            t_true = true_decision_time(envA) if step < n_ops else np.nan
            if step < n_ops:
                assert abs(t_true - envA.next_schedule_time[0] * scaleA) < 1e-6
            sched = envA.op_scheduled_flag[0].astype(bool)
            assert np.array_equal(sched, envB.op_scheduled_flag[0].astype(bool))
            a_env = float(envA.max_endTime[0] * scaleA) if step > 0 else \
                float(envA.init_quality[0] * scaleA)
            f_env = float(envB.max_endTime[0] * scaleB) if step > 0 else \
                float(envB.init_quality[0] * scaleB)
            d = chain_bounds(jl, min_pt, lag, envA.true_op_ct[0], sched,
                             t_true if step < n_ops else 0.0)
            max_env_dir_dev = max(max_env_dir_dev, abs(a_env - d[0]),
                                  abs(f_env - d[1]))
            for k, v in zip(('aug_env', 'free_env', 'aug_dir', 'free_dir',
                             'aug_t', 'free_t', 't'),
                            (a_env, f_env, d[0], d[1], d[2], d[3], t_true)):
                rec[k].append(float(v))
            if step == n_ops:
                break
            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                                   op_mask=state.op_mask_tensor,
                                   candidate=state.candidate_tensor,
                                   fea_m=state.fea_m_tensor,
                                   mch_mask=state.mch_mask_tensor,
                                   comp_idx=state.comp_idx_tensor,
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                   fea_pairs=state.fea_pairs_tensor,
                                   op_type=state.op_type_tensor,
                                   mch_type=state.mch_type_tensor)
            action = greedy_select_action(pi)
            a = int(action.cpu().numpy()[0])
            cj, cm = a // M, a % M
            opA = int(envA.candidate[0, cj])
            opB = int(envB.candidate[0, cj])
            assert opA == opB, 'replay env diverged (candidate)'
            assert not envB.dynamic_pair_mask[0, cj, cm], \
                'replayed action is masked in the lag-free env'
            assigned[opA] = cm
            state, _, _ = envA.step(actions=np.array([a]))
            envB.step(actions=np.array([a]))
        assert envA.done().all() and envB.done().all()
        same_sched = bool(np.array_equal(envA.true_op_ct[0], envB.true_op_ct[0]))
        assert same_sched, 'the two envs gave different schedules'
        cmax = float(envA.current_makespan[0])
        val = validate_schedule(jl, pt, meta['time_lag'], assigned,
                                envA.true_op_ct[0])
        assert val['feasible'] and abs(val['makespan'] - cmax) < 1e-6

        arr = {k: np.asarray(v) for k, v in rec.items()}
        T = n_ops
        dec = slice(0, T)               # decision states s_0..s_{T-1}
        o = {
            'instance': os.path.basename(stem), 'T': T, 'cmax': cmax,
            'cmax_gpu_stored': float(gpu[i]), 'cmax_cpu_s5': float(cpu[i]),
            'validator_feasible': bool(val['feasible']),
            'same_schedule_both_envs': same_sched,
            'max_abs_env_minus_direct_h': max_env_dir_dev,
        }
        for k in ('aug_env', 'free_env', 'aug_t', 'free_t'):
            o[f'{k}_s0'] = float(arr[k][0])
            o[f'{k}_s0_ratio'] = float(arr[k][0] / cmax)
            o[f'{k}_stepavg_ratio'] = float(np.mean(arr[k][dec] / cmax))
            o[f'{k}_sT'] = float(arr[k][T])
            o[f'{k}_sT_equals_cmax'] = bool(abs(arr[k][T] - cmax) < TOL)
            o[f'{k}_n_above_cmax'] = int(np.sum(arr[k][dec] > cmax + TOL))
        o['stepavg_aug_minus_free_over_cmax'] = float(
            np.mean((arr['aug_env'][dec] - arr['free_env'][dec]) / cmax))
        o['n_steps_free_gt_aug'] = int(np.sum(
            arr['free_env'][:T + 1] > arr['aug_env'][:T + 1] + TOL))
        o['n_steps_free_gt_aug_direct'] = int(np.sum(
            arr['free_dir'][:T + 1] > arr['aug_dir'][:T + 1] + 1e-9))
        o['n_steps_auget_gt_aug'] = int(np.sum(arr['aug_t'][dec] > arr['aug_env'][dec] + TOL))
        o['stepavg_augt_minus_aug_over_cmax'] = float(
            np.mean((arr['aug_t'][dec] - arr['aug_env'][dec]) / cmax))
        orders.append(o)
        if (i + 1) % 10 == 0 or i == len(stems) - 1:
            print(f'[{ds}] {i + 1}/{len(stems)}  cmax={cmax:.0f}  '
                  f'{time.time() - t0:.0f}s', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = {
        'dataset': ds, 'model': MODEL, 'checkpoint': os.path.relpath(ckpt, '.'),
        'checkpoint_sha256': sha256(os.path.join(REPO, ckpt)),
        'n_parameters': int(n_par), 'device': 'cpu', 'decoding': 'greedy',
        'seed_test': SEED_TEST, 'n_orders': len(orders),
        'stored_gpu_result': os.path.relpath(gpu_res_path, REPO),
        'stored_cpu_result': os.path.relpath(cpu_bp_path, REPO),
        'definitions': {
            'aug_env': 'max(op_ct_lb) * scale in the default env (bound_lag_free=False)',
            'free_env': 'max(op_ct_lb) * scale in the replay env (bound_lag_free=True)',
            'aug_t/free_t': 'Eq. (4) as printed, with max(., t); not computed by the env',
            'stepavg': 'mean over decision states s_0..s_{T-1}',
            'tolerance_h': TOL,
        },
        'torch_threads': __import__('torch').get_num_threads(),
        'affinity': sorted(os.sched_getaffinity(0)),
        'wall_s': time.time() - t0,
        'orders': orders,
    }
    path = os.path.join(out_dir, f'bound_tightness_{ds}.json')
    assert not os.path.exists(path), f'{path} exists; not overwritten'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


def summarize():
    rows = []
    for ds in SETS:
        p = os.path.join(OUT, f'bound_tightness_{ds}.json')
        d = json.load(open(p))
        o = d['orders']
        cm = np.array([x['cmax'] for x in o])
        gpu = np.array([x['cmax_gpu_stored'] for x in o])
        cpu = np.array([x['cmax_cpu_s5'] for x in o])
        r = {'dataset': ds, 'n_orders': len(o), 'source': os.path.relpath(p, REPO),
             'mean_cmax_cpu_rerun': float(cm.mean()),
             'mean_cmax_gpu_stored': float(gpu.mean()),
             'mean_cmax_cpu_s5': float(cpu.mean()),
             'n_orders_rerun_ne_gpu_stored': int(np.sum(cm != gpu)),
             'max_abs_rerun_minus_gpu_per_order': float(np.max(np.abs(cm - gpu))),
             'mean_rerun_minus_gpu': float(np.mean(cm - gpu)),
             'n_orders_rerun_ne_cpu_s5': int(np.sum(cm != cpu))}
        for k in ('aug_env', 'free_env', 'aug_t', 'free_t'):
            r[f'mean_{k}_s0_ratio'] = float(np.mean([x[f'{k}_s0_ratio'] for x in o]))
            r[f'mean_{k}_stepavg_ratio'] = float(np.mean([x[f'{k}_stepavg_ratio'] for x in o]))
            r[f'n_orders_{k}_sT_ne_cmax'] = int(sum(not x[f'{k}_sT_equals_cmax'] for x in o))
            r[f'n_steps_{k}_above_cmax'] = int(sum(x[f'{k}_n_above_cmax'] for x in o))
        r['mean_stepavg_aug_minus_free_over_cmax'] = float(np.mean(
            [x['stepavg_aug_minus_free_over_cmax'] for x in o]))
        r['mean_stepavg_augt_minus_aug_over_cmax'] = float(np.mean(
            [x['stepavg_augt_minus_aug_over_cmax'] for x in o]))
        r['n_steps_free_gt_aug'] = int(sum(x['n_steps_free_gt_aug'] for x in o))
        r['n_steps_free_gt_aug_direct'] = int(sum(x['n_steps_free_gt_aug_direct'] for x in o))
        r['n_steps_total'] = int(sum(x['T'] + 1 for x in o))
        r['max_abs_env_minus_direct_h'] = float(max(x['max_abs_env_minus_direct_h'] for x in o))
        r['all_same_schedule_both_envs'] = bool(all(x['same_schedule_both_envs'] for x in o))
        r['all_validator_feasible'] = bool(all(x['validator_feasible'] for x in o))
        rows.append(r)
    out = {'model': MODEL, 'device': 'cpu', 'decoding': 'greedy',
           'script': 'analysis_bound_tightness.py', 'sizes': rows}
    with open(os.path.join(OUT, 'bound_tightness.json'), 'w') as f:
        json.dump(out, f, indent=1)
    L = ['# Tightness of the two bounds along SLAB seed-301 greedy trajectories (CPU)', '',
         f'Model `{MODEL}`, greedy decoding on the CPU, true waits, every order of each test set. '
         'Ratios are Chat/Cmax of the same order, averaged over orders; the step average runs over the '
         'T decision states s_0..s_{T-1}. "aug" = lag-augmented bound of the environment (bound_lag_free=False), '
         '"free" = lag-free bound (replay env with bound_lag_free=True, identical schedule). '
         '"aug+t" / "free+t" = Eq. (4) as printed, with max(., t), which the environment does not compute.', '',
         '| Modules | Orders | aug s0/Cmax | free s0/Cmax | aug step-avg | free step-avg | (aug-free)/Cmax step-avg | aug+t s0/Cmax | aug+t step-avg | free+t step-avg |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        L.append(f"| {r['dataset'].split('x')[0]} | {r['n_orders']} | {r['mean_aug_env_s0_ratio']:.4f} | "
                 f"{r['mean_free_env_s0_ratio']:.4f} | {r['mean_aug_env_stepavg_ratio']:.4f} | "
                 f"{r['mean_free_env_stepavg_ratio']:.4f} | {r['mean_stepavg_aug_minus_free_over_cmax']:.4f} | "
                 f"{r['mean_aug_t_s0_ratio']:.4f} | {r['mean_aug_t_stepavg_ratio']:.4f} | "
                 f"{r['mean_free_t_stepavg_ratio']:.4f} |")
    L += ['', '## Checks', '',
          '| Modules | steps | free > aug | aug > Cmax | free > Cmax | aug+t > Cmax | orders aug(s_T) != Cmax | orders free(s_T) != Cmax | max abs env - direct (h) | same schedule | validator | orders CPU != stored GPU | max abs diff (h) | mean diff (h) | orders CPU != S5 CPU run |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---:|---:|---:|']
    for r in rows:
        L.append(f"| {r['dataset'].split('x')[0]} | {r['n_steps_total']} | {r['n_steps_free_gt_aug']} | "
                 f"{r['n_steps_aug_env_above_cmax']} | {r['n_steps_free_env_above_cmax']} | "
                 f"{r['n_steps_aug_t_above_cmax']} | {r['n_orders_aug_env_sT_ne_cmax']} | "
                 f"{r['n_orders_free_env_sT_ne_cmax']} | {r['max_abs_env_minus_direct_h']:.2e} | "
                 f"{r['all_same_schedule_both_envs']} | {r['all_validator_feasible']} | "
                 f"{r['n_orders_rerun_ne_gpu_stored']} | {r['max_abs_rerun_minus_gpu_per_order']:.0f} | "
                 f"{r['mean_rerun_minus_gpu']:+.2f} | {r['n_orders_rerun_ne_cpu_s5']} |")
    L += ['', 'Mean makespan: CPU rerun / stored GPU result / S5 CPU run', '']
    for r in rows:
        L.append(f"- {r['dataset']}: {r['mean_cmax_cpu_rerun']:.2f} / {r['mean_cmax_gpu_stored']:.2f} / "
                 f"{r['mean_cmax_cpu_s5']:.2f} h")
    with open(os.path.join(OUT, 'bound_tightness.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--set', choices=SETS)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--max_orders', type=int, default=0,
                    help='smoke test: first N orders only (use with --out_dir)')
    ap.add_argument('--out_dir', default=OUT)
    a = ap.parse_args()
    if a.summarize:
        summarize()
    elif a.set:
        if a.max_orders and os.path.abspath(a.out_dir) == OUT:
            ap.error('a partial run must not write to the result directory')
        run_set(a.set, a.max_orders, os.path.abspath(a.out_dir))
    else:
        ap.error('give --set or --summarize')


if __name__ == '__main__':
    os.chdir(REPO)
    main()
