"""
Train the HGNN scheduler on generated PPVC instances with post-operation
time-lags, using the upstream PPO hyperparameters from config.json.

The loop follows the upstream train.py: a batch of `batch_size` instances is
redrawn every `parallel_iter` iterations, the policy is updated every
`update_timestep` iterations, and it is validated greedily every
`save_timestep` iterations; the best validation checkpoint is kept.

Instances: ppvc_instance_generator(n_modules, class_mix, station_counts).
Training instance seeds are drawn from a numpy stream seeded by --seed.
Validation instance i uses generator seed --vali_seed0 + i (default 100 + i).
With --vali_sizes 10,20 the same seeds are drawn at every listed size and the
best checkpoint minimizes the mean over sizes of makespan / SPT mean.

lag_mode 'aware' trains and validates with the true lags; 'blind' zeros the
lags inside the environment for training and validation (the lags are
reintroduced only at evaluation, by right-shift repair).

Outputs under <save_root>/<run_name>/ (default save_root: backbones/hgnn/save):
    run_config.json       resolved configuration (read by eval_ppvc_hgnn.py)
    best.pt / last.pt     policy state dicts
    train_log.json        per-iteration loss and reward, validation curve

Example:
    python backbones/hgnn/train_ppvc.py --n_modules 10 --lag_mode aware \
        --run_name hgnn-aware-10x25 --device cuda:0
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppvc_hgnn as P  # noqa: E402  (sets sys.path and compatibility modules)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args():
    cfg = P.load_config()
    tp, ep = cfg['train_paras'], cfg['env_paras']
    ap = argparse.ArgumentParser(description='Train HGNN on PPVC instances with time-lags')
    ap.add_argument('--n_modules', type=int, default=10)
    ap.add_argument('--factory', type=str, default='default', choices=sorted(P.FACTORIES))
    ap.add_argument('--mix', type=str, default='mixed')
    ap.add_argument('--lag_mode', type=str, default='aware', choices=['aware', 'blind'])
    ap.add_argument('--lag_features', action='store_true',
                    help='append the static-lag and remaining-lag operation features (in_size_ope 8)')
    ap.add_argument('--bound_pt', type=str, default='mean', choices=['mean', 'min'],
                    help="processing time of unscheduled operations in the completion estimate "
                         "(upstream 'mean'; 'min' makes the estimate a lower bound)")
    ap.add_argument('--batch_size', type=int, default=ep['batch_size'])
    ap.add_argument('--max_iterations', type=int, default=tp['max_iterations'])
    ap.add_argument('--parallel_iter', type=int, default=tp['parallel_iter'])
    ap.add_argument('--update_timestep', type=int, default=tp['update_timestep'])
    ap.add_argument('--save_timestep', type=int, default=tp['save_timestep'])
    ap.add_argument('--vali_size', type=int, default=ep['valid_batch_size'])
    ap.add_argument('--vali_seed0', type=int, default=100)
    ap.add_argument('--lag_scale', type=float, default=1.0,
                    help='multiply every training and validation lag by this factor (rounded), as train.py; '
                         'evaluation always uses the true lags')
    ap.add_argument('--vali_sizes', type=str, default=None,
                    help="comma-separated validation order sizes, e.g. '10,20'; the checkpoint is selected on the "
                         "mean over sizes of (mean makespan / SPT mean of that size), as in the DANIEL-family "
                         "train.py. Default: validate at --n_modules only, on the raw mean makespan (upstream).")
    ap.add_argument('--seed', type=int, default=300)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--run_name', type=str, default=None)
    ap.add_argument('--save_root', type=str, default=os.path.join(P.HGNN_DIR, 'save'))
    args = ap.parse_args()
    if args.run_name is None:
        args.run_name = 'hgnn-{0}{1}{2}-{3}x{4}-s{5}'.format(
            args.lag_mode, '-lagfeat' if args.lag_features else '',
            '-minbound' if args.bound_pt == 'min' else '', args.n_modules,
            sum(P.FACTORIES[args.factory].values()), args.seed)
    return args, cfg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(env, policy):
    '''Greedy validation over the whole validation batch (upstream validate.py logic).'''
    import PPO_model
    memory = PPO_model.Memory()
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, _, dones = env.step(actions)
        done = dones.all()
    if not env.validate_gantt()[0]:
        print('Scheduling Error in validation')
    makespans = copy.deepcopy(env.makespan_batch)
    env.reset()
    return makespans


def main():
    args, cfg = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    setup_seed(args.seed)

    num_mas = sum(P.FACTORIES[args.factory].values())
    train_paras = copy.deepcopy(cfg['train_paras'])
    train_paras.update(max_iterations=args.max_iterations, parallel_iter=args.parallel_iter,
                       update_timestep=args.update_timestep, save_timestep=args.save_timestep)
    env_paras, model_paras = P.make_paras(cfg, device, args.lag_mode, args.lag_features,
                                          args.batch_size, args.n_modules, num_mas, bound_pt=args.bound_pt)
    vali_paras = dict(env_paras, batch_size=args.vali_size)

    save_dir = os.path.join(args.save_root, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    snapshot = {'args': vars(args), 'env_paras': {k: v for k, v in env_paras.items() if k != 'device'},
                'model_paras': {k: v for k, v in model_paras.items() if k != 'device'},
                'train_paras': train_paras, 'upstream_config': cfg}
    with open(os.path.join(save_dir, 'run_config.json'), 'w') as f:
        json.dump(snapshot, f, indent=1)

    import PPO_model
    memories = PPO_model.Memory()
    model = P.build_model(model_paras, train_paras, num_envs=args.batch_size)
    n_params = sum(p.numel() for p in model.policy.parameters())

    vali_case = P.PPVCCaseGenerator(args.n_modules, [args.vali_seed0 + i for i in range(args.vali_size)],
                                    factory=args.factory, class_mix=args.mix, lag_scale=args.lag_scale)
    env_vali = P.FJSPEnvLag(case=vali_case, env_paras=vali_paras)
    # optional multi-size checkpoint selection (same normalizers as train.py's SPT_SCALE)
    SPT_SCALE = {10: 216.0, 20: 265.0, 30: 388.0}
    vali_sets = []
    if args.vali_sizes:
        for sz in [int(x) for x in args.vali_sizes.split(',')]:
            if sz == args.n_modules:
                env_sz = env_vali
            else:
                case_sz = P.PPVCCaseGenerator(sz, [args.vali_seed0 + i for i in range(args.vali_size)],
                                              factory=args.factory, class_mix=args.mix, lag_scale=args.lag_scale)
                env_sz = P.FJSPEnvLag(case=case_sz, env_paras=dict(vali_paras, num_jobs=sz))
            vali_sets.append((sz, env_sz, SPT_SCALE.get(sz, 216.0 * sz / 10.0)))
        print('vali sizes={0} norms={1}'.format([v[0] for v in vali_sets], [v[2] for v in vali_sets]))
    print('run={0} lag_mode={1} lag_features={2} bound_pt={3} modules={4} machines={5} ops/instance={6} '
          'batch={7} vali={8} params={9} device={10}'.format(
              args.run_name, args.lag_mode, args.lag_features, args.bound_pt, args.n_modules, num_mas,
              int(env_vali.nums_opes[0]), args.batch_size, args.vali_size, n_params, device))

    log = {'iteration': [], 'loss': [], 'reward': [], 'seconds': [], 'vali_iteration': [], 'vali_makespan': []}
    best = float('inf')
    start = time.time()
    env = None
    seed_stream = np.random.RandomState(args.seed)
    for i in range(1, args.max_iterations + 1):
        if (i - 1) % args.parallel_iter == 0:
            seeds = seed_stream.randint(0, 2 ** 31 - 1, size=args.batch_size)
            case = P.PPVCCaseGenerator(args.n_modules, seeds, factory=args.factory, class_mix=args.mix, lag_scale=args.lag_scale)
            env = P.FJSPEnvLag(case=case, env_paras=env_paras)

        state = env.state
        done = False
        dones = env.done_batch
        t0 = time.time()
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
        if not env.validate_gantt()[0]:
            print('Scheduling Error in training')
        env.reset()

        if i % args.update_timestep == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            memories.clear_memory()
            log['iteration'].append(i)
            log['loss'].append(float(loss))
            log['reward'].append(float(reward))
            log['seconds'].append(time.time() - t0)
            print('iter {0}  reward {1:.3f}  loss {2:.4f}  {3:.1f}s'.format(i, reward, loss, time.time() - t0))

        if i % args.save_timestep == 0:
            if vali_sets:
                per_size = [float(validate(e, model.policy_old).mean()) for _, e, _ in vali_sets]
                mean_ms = float(np.mean([m / n for m, (_, _, n) in zip(per_size, vali_sets)]))
                log.setdefault('vali_per_size', []).append(per_size)
            else:
                vali = validate(env_vali, model.policy_old)
                mean_ms = float(vali.mean())
            log['vali_iteration'].append(i)
            log['vali_makespan'].append(mean_ms)
            if mean_ms < best:
                best = mean_ms
                torch.save(model.policy.state_dict(), os.path.join(save_dir, 'best.pt'))
            print('validation iter {0}: mean makespan {1:.2f} (best {2:.2f})'.format(i, mean_ms, best))
        torch.save(model.policy.state_dict(), os.path.join(save_dir, 'last.pt'))
        log['total_seconds'] = time.time() - start
        with open(os.path.join(save_dir, 'train_log.json'), 'w') as f:
            json.dump(log, f)
    print('done in {0:.1f}s, checkpoints in {1}'.format(time.time() - start, save_dir))


if __name__ == '__main__':
    main()
