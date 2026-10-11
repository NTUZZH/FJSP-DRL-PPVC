"""
Greedy evaluation of an HGNN checkpoint on a saved PPVC instance set.

For every instance data/PPVC/<name>/instance_xxx.fjs (+ .meta.json):
  * arm 'aware': greedy rollout in the lag-aware environment;
  * arm 'blind': greedy rollout with the lags zeroed inside the environment,
    then right_shift_repair against the true lags (machine order kept);
  * the final schedule is checked by schedule_validator.validate_schedule with
    the true lags (zero violations asserted) and its makespan must equal the
    reported one.

Outputs in test_results/PPVC/<name>/ (same [n_instances, 2] layout of
(makespan, seconds) as eval_ppvc.py writes for its greedy results):
    Result_greedy+<model_name>_<name>.npy        final makespan (repaired for 'blind')
    Result_greedy+<model_name>-plan_<name>.npy   'blind' only: lag-blind plan makespan
    summary_<model_name>.md
with <model_name> = hgnn-<arm>[+<tag>].

The architecture flag (lag features, in_size_ope 8) is read from the
run_config.json saved next to the checkpoint by train_ppvc.py; a checkpoint
without it (e.g. the upstream model/save_10_5.pt) uses the upstream config.

Example:
    python backbones/hgnn/eval_ppvc_hgnn.py --ckpt backbones/hgnn/save/<run>/best.pt \
        --arm aware --data_path data/PPVC/10x25+ppvc-mixed --tag <run>
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppvc_hgnn as P  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate an HGNN checkpoint on a PPVC instance set')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--arm', type=str, required=True, choices=['aware', 'blind'])
    ap.add_argument('--data_path', type=str, default='data/PPVC/10x25+ppvc-mixed',
                    help='instance folder, relative to the repository root or absolute')
    ap.add_argument('--tag', type=str, default=None, help='appended to the model name in output files')
    ap.add_argument('--lag_features', type=int, default=None, choices=[0, 1],
                    help='override the architecture flag read from run_config.json')
    ap.add_argument('--max_instances', type=int, default=None)
    ap.add_argument('--save_dir', type=str, default=None,
                    help='output folder (default test_results/PPVC/<name> under the repository root)')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--threads', type=int, default=1)
    return ap.parse_args()


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(P.REPO_ROOT, path)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    from right_shift_repair import right_shift_repair
    from schedule_validator import validate_schedule

    cfg = P.load_config()
    ckpt = resolve(args.ckpt)
    run_cfg_path = os.path.join(os.path.dirname(ckpt), 'run_config.json')
    lag_features, bound_pt = False, 'mean'
    if os.path.exists(run_cfg_path):
        with open(run_cfg_path) as f:
            run_env = json.load(f)['env_paras']
        lag_features = bool(run_env.get('lag_features', False))
        bound_pt = run_env.get('bound_pt', 'mean')
    if args.lag_features is not None:
        lag_features = bool(args.lag_features)

    data_path = resolve(args.data_path)
    files = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if args.max_instances is not None:
        files = files[:args.max_instances]
    if not files:
        sys.exit('no instance_*.fjs under ' + data_path)
    dataset = os.path.basename(os.path.normpath(data_path))
    model_name = 'hgnn-' + args.arm + ('+' + args.tag if args.tag else '')

    env_paras, model_paras = P.make_paras(cfg, device, args.arm, lag_features, 1, 1, 1, bound_pt=bound_pt)
    model = P.build_model(model_paras, cfg['train_paras'], ckpt=ckpt)
    model.policy.eval()
    model.policy_old.eval()

    n = len(files)
    results = np.zeros((n, 2))
    plan = np.zeros((n, 2))
    print('eval_ppvc_hgnn  model={0}  ckpt={1}  dataset={2}  instances={3}  lag_features={4}  bound_pt={5}'.format(
        model_name, args.ckpt, dataset, n, lag_features, bound_pt))
    for i, fjs in enumerate(files):
        with open(os.path.splitext(fjs)[0] + '.meta.json') as f:
            true_lag = np.asarray(json.load(f)['time_lag'], dtype=float)
        env = P.make_file_env(fjs, env_paras)
        t0 = time.time()
        ms = float(P.greedy_rollout(env, model)[0])
        sec = time.time() - t0
        jl, pt, amch, ct = P.extract_schedule(env)
        if args.arm == 'blind':
            plan[i] = [ms, sec]
            t1 = time.time()
            ct, ms = right_shift_repair(jl, pt, true_lag, amch, op_ct_lagblind=ct)
            sec += time.time() - t1
            assert ms >= plan[i, 0] - 1e-6, 'repair shortened the schedule'
        res = validate_schedule(jl, pt, true_lag, amch, ct)
        assert res['feasible'], '{0}: infeasible schedule\n  {1}'.format(
            fjs, '\n  '.join(res['violations'][:10]))
        assert abs(res['makespan'] - ms) < 1e-6, '{0}: makespan {1} != validator {2}'.format(
            fjs, ms, res['makespan'])
        results[i] = [ms, sec]
        if (i + 1) % 10 == 0 or i == n - 1:
            print('  [{0:3d}/{1}] makespan={2:.0f}h'.format(i + 1, n, ms)
                  + ('  plan={0:.0f}h'.format(plan[i, 0]) if args.arm == 'blind' else ''))

    save_dir = resolve(args.save_dir) if args.save_dir else os.path.join(P.REPO_ROOT, 'test_results', 'PPVC', dataset)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'Result_greedy+{0}_{1}.npy'.format(model_name, dataset))
    np.save(out, results)
    print('saved ' + out)
    lines = ['# HGNN evaluation summary - {0}\n'.format(model_name),
             '- dataset: `{0}` ({1} instances)'.format(dataset, n),
             '- checkpoint: `{0}`'.format(args.ckpt),
             '- arm: {0}, lag_features: {1}, bound_pt: {2}'.format(args.arm, lag_features, bound_pt),
             '- feasible (validator, true lags): {0}/{0}'.format(n), '',
             '| method | mean makespan | std | mean time (s) |', '|---|---|---|---|',
             '| greedy | {0:.1f} | {1:.1f} | {2:.4f} |'.format(
                 results[:, 0].mean(), results[:, 0].std(), results[:, 1].mean())]
    if args.arm == 'blind':
        out_plan = os.path.join(save_dir, 'Result_greedy+{0}-plan_{1}.npy'.format(model_name, dataset))
        np.save(out_plan, plan)
        print('saved ' + out_plan)
        lines.append('| lag-blind plan (before repair) | {0:.1f} | {1:.1f} | {2:.4f} |'.format(
            plan[:, 0].mean(), plan[:, 0].std(), plan[:, 1].mean()))
    with open(os.path.join(save_dir, 'summary_{0}.md'.format(model_name)), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('mean makespan {0:.2f}  mean seconds {1:.3f}'.format(results[:, 0].mean(), results[:, 1].mean()))


if __name__ == '__main__':
    main()
