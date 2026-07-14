"""
eval_a0_repair.py
=================
Evaluation entry point for ablation arm **A0** — "lag-blind + right-shift
repair" — the PPVC paper's stand-in for *current industry practice*.

THE A0 STORY
  A scheduler that was TRAINED with the time-lags ZEROED (lag-blind) is, at test
  time, asked to schedule each instance on a LAG-FREE environment. It therefore
  plans as if curing / ponding / paint-drying delays did not exist. That plan
  fixes a machine assignment and a per-machine operation ORDER. Reality then
  imposes the TRUE lags: the only legal recovery a plant has, without
  re-optimising, is to push operations RIGHT (later) until every lag and
  no-overlap constraint holds. The makespan of that right-shift-repaired
  schedule is A0's score; the gap between it and the lag-blind plan makespan is
  the "plan-vs-reality inflation" reported in the paper.

WHAT THIS SCRIPT DOES (per instance, batch-of-1)
  1. Load (job_length, op_pt, meta) via ppvc_instance_generator.load_instance.
  2. Build the env WITHOUT lags (time_lag_list=None) so the rollout is lag-blind.
     The checkpoint's ARCHITECTURE is honoured exactly: use_lag_features sets the
     op-feature width, and if the snapshot says use_type_embedding=True we still
     feed op_type / mch_type (they are static instance attributes the network
     embeds; the lag-free env path is orthogonal to type embeddings).
  3. Greedy rollout. Record assigned_mch[N] and the lag-blind true_op_ct[N]
     (== the env's completion times, which are lag-free because lags are zero),
     plus the lag-blind plan makespan.
  4. right_shift_repair(jl, pt, meta['time_lag'], assigned_mch,
                        op_ct_lagblind=lag_blind_ct) -> (repaired_ct, repaired_ms)
     Keeps the machine ORDER from the lag-blind plan; only shifts start times.
  5. validate_schedule(jl, pt, meta['time_lag'], assigned_mch, repaired_ct)
     against the TRUE lags — must be feasible (asserted + counted).

OUTPUTS  (test_results/PPVC/<dataset>/)
  * Result_A0REPAIR+<model>_<dataset>.npy   [N,2] (repaired makespan, seconds)
  * Result_A0BLIND+<model>_<dataset>.npy    [N,2] (lag-blind plan makespan, sec)
  * summary_a0_<model>.md                    human-readable roll-up

CLI
  python eval_a0_repair.py \
      --model_name 10x25+ppvc-mixed+a0-lagblind \
      --data_path data/PPVC/10x25+ppvc-mixed \
      [--seed_test 50] [--max_instances N]

NOTE ON ARCHITECTURE FLAGS
  A0's real checkpoint is trained lag-blind, so its snapshot will carry
  use_lag_features=False / use_type_embedding=False (10 op-features, no type
  args). This script does NOT assume that — it reads whatever the snapshot says
  and feeds exactly the inputs that architecture expects, so it also runs
  cleanly against the +smoke checkpoint (which has BOTH flags on) for mechanics
  verification.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse our flags FIRST, then scrub argv before `from params import configs`.

    params.py calls parser.parse_args() at IMPORT time against the process argv
    (it is the project-wide config singleton). If our flags are still on argv
    when params imports, it rejects them — so we strip argv to just the program
    name here, exactly as eval_ppvc.py does.
    """
    ap = argparse.ArgumentParser(
        description='Evaluate ablation arm A0 (lag-blind + right-shift repair)')
    ap.add_argument('--model_name', type=str,
                    default='10x25+ppvc-mixed+a0-lagblind',
                    help='checkpoint stem under trained_network/PPVC/ and the '
                         'config snapshot train_log/PPVC/config_<name>.json')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed',
                    help='directory with instance_*.fjs + .meta.json')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap the number of instances (verification convenience)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]  # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli()

# params.configs is the module-level singleton read by PPO_initialize / the env;
# we mutate the architecture flags on it BEFORE building the network.
from params import configs

# Architecture-critical keys copied from the training-config snapshot into
# params.configs. These determine the SHAPE of the network, so they must match
# the checkpoint exactly before PPO_initialize() builds DANIEL.  (Same list as
# eval_ppvc.py, so any architecture this repo trains loads here too.)
ARCH_KEYS = (
    'use_lag_features', 'use_type_embedding',
    'fea_j_input_dim', 'fea_m_input_dim',
    'type_emb_dim', 'n_op_types', 'n_mch_types',
    'n_j', 'n_m', 'n_op',
    'num_heads_OAB', 'num_heads_MAB', 'layer_fea_output_dim',
    'num_mlp_layers_actor', 'hidden_dim_actor',
    'num_mlp_layers_critic', 'hidden_dim_critic',
    'dropout_prob',
)


def load_train_config(model_name):
    """Read train_log/PPVC/config_<model>.json; apply arch flags to configs."""
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[eval_a0] training-config snapshot not found: {path}\n'
                 f'  (needed to set the network architecture before loading '
                 f'the checkpoint)')
    with open(path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(configs, k, snap[k])
    return snap


def list_instances(data_path, cap=None):
    """Return sorted instance stems (path without .fjs) under data_path."""
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[eval_a0] no instance_*.fjs found under {data_path}')
    stems = [p[:-len('.fjs')] for p in fjs]
    if cap is not None:
        stems = stems[:cap]
    return stems


def rollout_greedy_lagblind(ppo, jl, pt, meta, use_lag_features,
                            use_type_embedding):
    """Greedy model rollout on a LAG-FREE env (the policy plans lag-blind).

    Builds the env with time_lag_list=None so true_op_lag is all-zero; the env's
    true_op_ct / current_makespan are then exactly the lag-blind plan values.
    The checkpoint's architecture is honoured: use_lag_features sets the op-
    feature width, and op_type / mch_type are still fed when the snapshot turns
    type embeddings on (they are static instance attributes, independent of the
    lag-free scheduling path).

    Returns (lagblind_makespan, seconds, assigned_mch[N], lagblind_op_ct[N]).
    """
    import torch
    from common_utils import greedy_select_action
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums

    M = pt.shape[1]
    n_ops = pt.shape[0]

    env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=M,
                               use_lag_features=use_lag_features)
    # LAG-FREE: time_lag_list=None -> the policy is blind to the true lags.
    kwargs = dict(time_lag_list=None)
    if use_type_embedding:
        # static type indices the network embeds; orthogonal to lag-free path.
        kwargs['op_type_list'] = [meta['op_type']]
        kwargs['mch_type_list'] = [meta['mch_type']]
    state = env.set_initial_data([jl], [pt], **kwargs)

    assigned_mch = np.full(n_ops, -1, dtype=int)
    t1 = time.time()
    while True:
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
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, done = env.step(actions=action.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    assert (assigned_mch >= 0).all(), 'greedy: some op never scheduled'
    return float(env.current_makespan[0]), t2 - t1, assigned_mch, \
        env.true_op_ct[0].copy()


def main():
    args = _ARGS

    # 1) apply the training-config architecture flags BEFORE building anything
    snap = load_train_config(args.model_name)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    # torch + device setup (torch>=2.3 API, mirroring train.py / eval_ppvc.py).
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from common_utils import setup_seed
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    setup_seed(args.seed_test)

    # 2) load the lag-blind checkpoint into the matching architecture
    from model.PPO import PPO_initialize
    ckpt_path = f'./trained_network/PPVC/{args.model_name}.pth'
    if not os.path.exists(ckpt_path):
        sys.exit(f'[eval_a0] checkpoint not found: {ckpt_path}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    ppo.policy.eval()

    # 3) collect instances + the repair / validation tools
    from ppvc_instance_generator import load_instance
    from right_shift_repair import right_shift_repair
    from schedule_validator import validate_schedule

    stems = list_instances(args.data_path, cap=args.max_instances)
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))

    repair_res = np.zeros((n, 2))   # [repaired_makespan, total_seconds]
    blind_res = np.zeros((n, 2))    # [lagblind_makespan, rollout_seconds]
    feas_count = 0
    inflations = np.zeros(n)        # per-instance plan-vs-reality inflation %

    print('=' * 70)
    print(f'eval_a0  model={args.model_name}  dataset={dataset}  instances={n}')
    print(f'  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  seed_test={args.seed_test}')
    print(f'  rollout env: LAG-FREE (time_lag_list=None) -> lag-blind plan, '
          f'repaired against TRUE lags')
    print('=' * 70)

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)

        # (2) lag-blind greedy rollout
        ms_blind, sec_roll, assigned_mch, oct_blind = rollout_greedy_lagblind(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding)

        # (3) right-shift repair against the TRUE lags (keeps machine order)
        t1 = time.time()
        repaired_ct, ms_repair = right_shift_repair(
            jl, pt, meta['time_lag'], assigned_mch,
            op_ct_lagblind=oct_blind)
        sec_repair = time.time() - t1

        # (4) independent feasibility check with the TRUE lags
        res = validate_schedule(jl, pt, meta['time_lag'], assigned_mch,
                                repaired_ct)
        assert res['feasible'], (
            f'A0 repaired schedule INFEASIBLE on {base}:\n  '
            + '\n  '.join(res['violations'][:10]))
        # validator makespan must equal the repair's reported makespan
        assert abs(res['makespan'] - ms_repair) < 1e-6, (
            f'{base}: repair makespan {ms_repair} != validator '
            f'{res["makespan"]}')
        # right-shift can only push ops LATER -> repaired >= lag-blind plan
        assert ms_repair >= ms_blind - 1e-6, (
            f'{base}: repaired makespan {ms_repair} < lag-blind {ms_blind} '
            f'(right-shift must not shorten the schedule)')
        feas_count += 1

        # (5) bookkeeping
        repair_res[i] = [ms_repair, sec_roll + sec_repair]
        blind_res[i] = [ms_blind, sec_roll]
        inflations[i] = (100.0 * (ms_repair - ms_blind) / ms_blind
                         if ms_blind > 0 else 0.0)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f'  [{i + 1:3d}/{n}] blind={ms_blind:7.0f}h  '
                  f'repaired={ms_repair:7.0f}h  '
                  f'inflation=+{inflations[i]:5.1f}%')

    # 6) save npy outputs (mirror the test_trained_model / eval_ppvc naming)
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_repair = os.path.join(
        save_dir, f'Result_A0REPAIR+{args.model_name}_{dataset}.npy')
    out_blind = os.path.join(
        save_dir, f'Result_A0BLIND+{args.model_name}_{dataset}.npy')
    np.save(out_repair, repair_res)
    np.save(out_blind, blind_res)
    print(f'  saved {out_repair}')
    print(f'  saved {out_blind}')

    # 7) summary markdown
    mean_repair = repair_res[:, 0].mean()
    std_repair = repair_res[:, 0].std()
    mean_blind = blind_res[:, 0].mean()
    std_blind = blind_res[:, 0].std()
    mean_infl = inflations.mean()
    mean_time = repair_res[:, 1].mean()

    lines = []
    lines.append(f'# PPVC ablation A0 (lag-blind + right-shift repair) '
                 f'— {args.model_name}\n')
    lines.append(f'- dataset: `{dataset}`  ({n} instances)')
    lines.append(f'- checkpoint: `{ckpt_path}`')
    lines.append(f'- seed_test: {args.seed_test}')
    lines.append(f'- architecture (from snapshot): '
                 f'use_lag_features={use_lag_features}, '
                 f'use_type_embedding={use_type_embedding}, '
                 f'fea_j_input_dim={getattr(configs, "fea_j_input_dim", "?")}, '
                 f'n_op_types={getattr(configs, "n_op_types", "?")}, '
                 f'n_mch_types={getattr(configs, "n_mch_types", "?")}')
    lines.append(f'- method: lag-blind greedy plan (env time_lag_list=None) '
                 f'-> right-shift repair vs TRUE lags -> validated vs TRUE lags')
    lines.append('')
    lines.append('| metric | value |')
    lines.append('|---|---|')
    lines.append(f'| mean repaired makespan (A0 score) | {mean_repair:.1f} |')
    lines.append(f'| std repaired makespan | {std_repair:.1f} |')
    lines.append(f'| mean lag-blind plan makespan | {mean_blind:.1f} |')
    lines.append(f'| std lag-blind plan makespan | {std_blind:.1f} |')
    lines.append(f'| mean plan-vs-reality inflation % | {mean_infl:.2f}% |')
    lines.append(f'| feasible (repaired vs TRUE lags) | {feas_count}/{n} |')
    lines.append(f'| mean time (rollout + repair, s) | {mean_time:.4f} |')
    lines.append('')
    lines.append('Inflation % = 100 * (repaired - lag-blind) / lag-blind, the '
                 'cost of ignoring lags at plan time then repairing.')

    summary_path = os.path.join(save_dir, f'summary_a0_{args.model_name}.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # console echo
    print('\n' + '=' * 70)
    print(f'A0 SUMMARY  ({dataset}, {n} instances)')
    print('=' * 70)
    print(f'  mean repaired makespan (A0 score) : {mean_repair:.1f}  '
          f'(std {std_repair:.1f})')
    print(f'  mean lag-blind plan makespan      : {mean_blind:.1f}  '
          f'(std {std_blind:.1f})')
    print(f'  mean plan-vs-reality inflation    : +{mean_infl:.2f}%')
    print(f'  feasible (repaired vs TRUE lags)  : {feas_count}/{n}')
    print(f'  mean time (rollout + repair)      : {mean_time:.4f}s')
    print('=' * 70)
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
