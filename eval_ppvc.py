"""
eval_ppvc.py
============
Evaluation entry point for trained PPVC (DANIEL) models + PDR baselines on a
saved PPVC instance set. Produces the paper's main-table numbers, so every
schedule (model AND heuristic) is independently re-validated against the TRUE
lags before its makespan is counted.

WHAT IT DOES
  * Reads the training-config snapshot train_log/PPVC/config_<model>.json and
    pushes the architecture-critical flags into params.configs BEFORE the
    network is built, so the checkpoint loads into a matching DANIEL.
  * Loads each instance via ppvc_instance_generator.load_instance (carries
    op_type / mch_type / true time_lag in the .meta.json side-car).
  * Rolls out greedy / sampling (model) and FIFO / MOR / SPT / MWKR (heuristics)
    one instance at a time (batch of 1 -> memory hygiene).
  * Re-checks every produced schedule with schedule_validator.validate_schedule
    using the TRUE lags; asserts feasibility and counts it.
  * Saves per-method [n_instances, 2] (makespan, seconds) npy files mirroring
    the test_trained_model.py naming convention, plus a human-readable
    summary_<model>.md with per-method mean/std makespan, mean time,
    feasibility count, and (if the CP-SAT reference jsonl exists) mean gap%.

CLI
  python eval_ppvc.py --model_name 10x25+ppvc-mixed+full \
      --data_path data/PPVC/10x25+ppvc-mixed \
      [--methods greedy sampling FIFO MOR SPT MWKR] \
      [--sample_times 100] [--seed_test 50]

  greedy / sampling use the model; the four PDR names run heuristics (no model
  needed). Default methods = greedy + the four PDRs (sampling is opt-in: it is
  ~100x slower).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse eval_ppvc's CLI.

    IMPORTANT: params.py calls parser.parse_args() at IMPORT time against the
    process argv (it is the project-wide config singleton). So we MUST parse our
    own flags here FIRST, then scrub argv to just the program name before
    `from params import configs` runs — otherwise params rejects our flags.
    """
    ap = argparse.ArgumentParser(description='Evaluate PPVC DANIEL models + PDRs')
    ap.add_argument('--model_name', type=str, default='10x25+ppvc-mixed+full',
                    help='checkpoint stem under trained_network/PPVC/ and the '
                         'config snapshot train_log/PPVC/config_<name>.json')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed',
                    help='directory with instance_*.fjs + .meta.json')
    ap.add_argument('--methods', nargs='+', default=None,
                    help='subset of: greedy sampling FIFO MOR SPT MWKR '
                         'SPTL LLF MWKRL LRPTL '
                         '(default: greedy + the four classic PDRs)')
    ap.add_argument('--out_tag', type=str, default=None,
                    help='label used in the output file names '
                         '(Result_<method>+<tag>_<dataset>.npy and '
                         'summary_<tag>.md); default: --model_name')
    ap.add_argument('--sample_times', type=int, default=100)
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--device', type=str, default=None,
                    help="override the network device (e.g. 'cpu'); "
                         "default: params.py's --device")
    ap.add_argument('--max_instances', type=int, default=0,
                    help='evaluate only the first N instances (0 = all)')
    ap.add_argument('--save_dir', type=str, default=None,
                    help='output directory for the npy and summary files; '
                         'default: test_results/PPVC/<dataset>')
    ap.add_argument('--route_variant', type=str, default='auto',
                    choices=['auto', 'fixed', 'variable'],
                    help="module routes of the dataset: 'variable' rolls out in "
                         "FJSPEnvForVariousOpNums; 'auto' reads route_variant from "
                         "<data_path>/dataset_meta.json (absent key = fixed)")
    args = ap.parse_args()
    # scrub argv so params.py's import-time parse_args() sees a clean argv
    sys.argv = [sys.argv[0]]
    return args


_ARGS = parse_cli()

# params.configs is a module-level singleton read by PPO_initialize / the env;
# we mutate the architecture flags on it BEFORE importing/initializing anything
# that reads them.
from params import configs

PDR_METHODS = ('FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL')
MODEL_METHODS = ('greedy', 'sampling')

# Architecture-critical keys copied from the training-config snapshot into
# params.configs. These determine the SHAPE of the network, so they must match
# the checkpoint exactly before PPO_initialize() builds DANIEL.
ARCH_KEYS = (
    'use_lag_features', 'use_type_embedding',
    'fea_j_input_dim', 'fea_m_input_dim',
    'type_emb_dim', 'n_op_types', 'n_mch_types',
    'n_j', 'n_m', 'n_op',
    'num_heads_OAB', 'num_heads_MAB', 'layer_fea_output_dim',
    'num_mlp_layers_actor', 'hidden_dim_actor',
    'num_mlp_layers_critic', 'hidden_dim_critic',
    'dropout_prob',
    'framework',
    # not a shape flag: the lower-bound variant the policy was trained with
    # (it enters the op features), applied to the model rollouts only
    'bound_lag_free',
)


def load_train_config(model_name):
    """Read train_log/PPVC/config_<model>.json and apply arch flags to configs.

    Returns the raw snapshot dict (for provenance in the summary).
    """
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[eval_ppvc] training-config snapshot not found: {path}\n'
                 f'  (needed to set the network architecture before loading '
                 f'the checkpoint)')
    with open(path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(configs, k, snap[k])
    # snapshots written before --framework existed are DANIEL
    configs.framework = snap.get('framework', 'DANIEL')
    from common_utils import check_backbone_tag
    check_backbone_tag(model_name, configs.framework)
    return snap


def list_instances(data_path):
    """Return sorted instance stems (path without .fjs) under data_path."""
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[eval_ppvc] no instance_*.fjs found under {data_path}')
    return [p[:-len('.fjs')] for p in fjs]


# ---------------------------------------------------------------------------
# Rollouts. Each returns (makespan, seconds, assigned_mch, true_op_ct) for a
# single instance, tracking the per-op machine assignment + true completion
# times so the schedule can be re-validated with the TRUE lags.
# ---------------------------------------------------------------------------

ROUTE_VARIANT = 'fixed'   # set in main() from --route_variant / dataset_meta.json


def detect_route_variant(data_path, cli_value):
    """Return 'fixed' or 'variable' for the dataset under data_path."""
    if cli_value != 'auto':
        return cli_value
    meta_path = os.path.join(data_path, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get('route_variant', 'fixed')
    return 'fixed'


def _make_env(n_j, n_m, use_lag_features, bound_lag_free=False):
    """Environment for the dataset's route variant: variable routes (modules
    with different op counts) use FJSPEnvForVariousOpNums, fixed routes the
    original FJSPEnvForSameOpNums. bound_lag_free selects the lag-free
    lower bound (fixed routes only)."""
    if ROUTE_VARIANT == 'variable':
        if bound_lag_free:
            sys.exit('[eval_ppvc] bound_lag_free is not wired for variable routes')
        from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
        return FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m,
                                       use_lag_features=use_lag_features,
                                       isolate_padding=True)
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    return FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m,
                                use_lag_features=use_lag_features,
                                bound_lag_free=bound_lag_free)


def _model_bound_lag_free():
    """Lower-bound variant of the loaded model (set from its snapshot)."""
    return bool(getattr(configs, 'bound_lag_free', False))


def _new_env(jl, pt, meta, use_lag_features, use_type_embedding,
             bound_lag_free=False):
    env = _make_env(jl.shape[0], pt.shape[1], use_lag_features, bound_lag_free)
    kwargs = dict(time_lag_list=[meta['time_lag']])
    if use_type_embedding:
        kwargs['op_type_list'] = [meta['op_type']]
        kwargs['mch_type_list'] = [meta['mch_type']]
    state = env.set_initial_data([jl], [pt], **kwargs)
    return env, state


def rollout_heuristic(method, jl, pt, meta, seed):
    """PDR rollout (batch of 1). Seeded per-instance for tie-break determinism."""
    from common_utils import heuristic_select_action
    np.random.seed(seed)
    M = pt.shape[1]
    n_ops = pt.shape[0]
    # heuristics never read the model -> type embedding irrelevant; pass lags so
    # the env's lag-aware candidate_free_time drives the PDR decisions.
    env, _ = _new_env(jl, pt, meta, use_lag_features=False,
                      use_type_embedding=False)
    assigned_mch = np.full(n_ops, -1, dtype=int)
    t1 = time.time()
    while not env.done().all():
        action = heuristic_select_action(method, env)
        chosen_job = action // M
        chosen_mch = action % M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = chosen_mch
        env.step(np.array([action]))
    t2 = time.time()
    assert (assigned_mch >= 0).all(), f'{method}: some op never scheduled'
    return float(env.current_makespan[0]), t2 - t1, assigned_mch, \
        env.true_op_ct[0].copy()


def rollout_greedy(ppo, jl, pt, meta, use_lag_features, use_type_embedding):
    """Greedy model rollout (batch of 1). Deterministic given the checkpoint."""
    import torch
    from common_utils import greedy_select_action
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state = _new_env(jl, pt, meta, use_lag_features, use_type_embedding,
                          bound_lag_free=_model_bound_lag_free())
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
        chosen_mch = a % M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = chosen_mch
        state, _, done = env.step(actions=action.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    assert (assigned_mch >= 0).all(), 'greedy: some op never scheduled'
    return float(env.current_makespan[0]), t2 - t1, assigned_mch, \
        env.true_op_ct[0].copy()


def rollout_sampling(ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                     sample_times):
    """Sampling model rollout: sample_times parallel envs; keep the best.

    Tracks the assignment of the best-makespan sample so it can be validated.
    """
    import torch
    from common_utils import sample_action
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env = _make_env(jl.shape[0], M, use_lag_features, _model_bound_lag_free())
    jl_b = np.tile(np.expand_dims(jl, 0), (sample_times, 1))
    pt_b = np.tile(np.expand_dims(pt, 0), (sample_times, 1, 1))
    kwargs = dict(time_lag_list=[meta['time_lag']] * sample_times)
    if use_type_embedding:
        kwargs['op_type_list'] = [meta['op_type']] * sample_times
        kwargs['mch_type_list'] = [meta['mch_type']] * sample_times
    state = env.set_initial_data(jl_b, pt_b, **kwargs)
    assigned_mch = np.full((sample_times, n_ops), -1, dtype=int)
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
        action_envs, _ = sample_action(pi)
        acts = action_envs.cpu().numpy()
        # record assignment per env (only envs not yet done keep advancing;
        # done envs no-op in the env, and their assigned_mch is already full)
        not_done = ~env.done().astype(bool)  # done() returns a float (0./1.) array; cast before bitwise NOT
        chosen_job = acts // M
        chosen_mch = acts % M
        for e in np.where(not_done)[0]:
            chosen_op = env.candidate[e, chosen_job[e]]
            assigned_mch[e, chosen_op] = chosen_mch[e]
        state, _, done = env.step(acts)
        if done.all():
            break
    t2 = time.time()
    best = int(np.argmin(env.current_makespan))
    assert (assigned_mch[best] >= 0).all(), 'sampling: best sample incomplete'
    return float(env.current_makespan[best]), t2 - t1, assigned_mch[best], \
        env.true_op_ct[best].copy()


# ---------------------------------------------------------------------------
# CP-SAT reference (optional, partial-tolerant)
# ---------------------------------------------------------------------------

def load_or_reference(data_path):
    """Read or_solution/PPVC/<dataset>.jsonl -> {instance_stem_basename: makespan}.

    Tolerates a missing/partial file (the CP-SAT batch may still be running).
    Keys are matched on the instance basename (e.g. 'instance_000').
    """
    dataset = os.path.basename(os.path.normpath(data_path))
    path = f'./or_solution/PPVC/{dataset}.jsonl'
    ref = {}
    if not os.path.exists(path):
        return ref, path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written final line
            # be liberal about the field names the CP-SAT job may use.
            # The PPVC CP-SAT batch writes 'makespan_lag' (the LAG-AWARE
            # objective — the correct apples-to-apples reference, since the
            # env/PDRs/model all schedule WITH lags); fall back to the plain
            # names for other producers.
            key = rec.get('instance') or rec.get('stem') or rec.get('name')
            ms = rec.get('makespan_lag',
                         rec.get('makespan',
                                 rec.get('objective', rec.get('obj'))))
            if key is None or ms is None:
                continue
            ref[os.path.basename(str(key)).replace('.fjs', '')] = float(ms)
    return ref, path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _ARGS

    methods = args.methods or ['greedy', 'FIFO', 'MOR', 'SPT', 'MWKR']
    need_model = any(m in MODEL_METHODS for m in methods)

    # 1) apply the training-config architecture flags BEFORE building anything
    snap = load_train_config(args.model_name)
    if args.out_tag:
        # a custom output label must carry the same backbone tag as the model
        from common_utils import check_backbone_tag
        check_backbone_tag(args.out_tag, configs.framework)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))
    if not need_model:
        # dispatching rules never touch the network: stay on the CPU
        configs.device = 'cpu'
    if args.device:
        configs.device = args.device

    # torch + device setup (torch>=2.3 API, mirroring train.py). Must happen
    # AFTER CUDA_VISIBLE_DEVICES is set and BEFORE the network is built.
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from common_utils import setup_seed
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')

    # one global seed for reproducibility; PDRs additionally reseed per instance
    setup_seed(args.seed_test)

    ppo = None
    ckpt_path = f'./trained_network/PPVC/{args.model_name}.pth'
    if need_model:
        from model.PPO import PPO_initialize
        if not os.path.exists(ckpt_path):
            sys.exit(f'[eval_ppvc] checkpoint not found: {ckpt_path}')
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        ppo.policy.eval()

    # 2) collect instances
    from ppvc_instance_generator import load_instance
    stems = list_instances(args.data_path)
    if args.max_instances > 0:
        stems = stems[:args.max_instances]
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    global ROUTE_VARIANT
    ROUTE_VARIANT = detect_route_variant(args.data_path, args.route_variant)
    print(f'  route_variant={ROUTE_VARIANT}')

    from schedule_validator import validate_schedule

    # results[method] = np.array [n, 2]; feas[method] = int feasibility count
    results = {m: np.zeros((n, 2)) for m in methods}
    feas = {m: 0 for m in methods}

    print('=' * 70)
    print(f'eval_ppvc  model={args.model_name}  dataset={dataset}  '
          f'instances={n}')
    print(f'  framework={configs.framework}  '
          f'use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  '
          f'bound_lag_free={_model_bound_lag_free()}  '
          f'seed_test={args.seed_test}')
    print(f'  methods={methods}'
          + (f'  sample_times={args.sample_times}' if 'sampling' in methods
             else ''))
    print('=' * 70)

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        for method in methods:
            if method == 'greedy':
                ms, sec, amch, oct_ = rollout_greedy(
                    ppo, jl, pt, meta, use_lag_features, use_type_embedding)
            elif method == 'sampling':
                ms, sec, amch, oct_ = rollout_sampling(
                    ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                    args.sample_times)
            elif method in PDR_METHODS:
                # per-instance reseed so PDR tie-breaks are deterministic
                ms, sec, amch, oct_ = rollout_heuristic(
                    method, jl, pt, meta, seed=args.seed_test + i)
            else:
                sys.exit(f'[eval_ppvc] unknown method: {method}')

            # independent re-validation against the TRUE lags
            res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
            assert res['feasible'], (
                f'{method} produced an INFEASIBLE schedule on {base}:\n  '
                + '\n  '.join(res['violations'][:10]))
            # cross-check: validator makespan must equal the env makespan
            assert abs(res['makespan'] - ms) < 1e-6, (
                f'{method} {base}: env makespan {ms} != validator '
                f'{res["makespan"]}')
            feas[method] += 1
            results[method][i] = [ms, sec]

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f'  [{i + 1:3d}/{n}] ' + '  '.join(
                f'{m}={results[m][i, 0]:.0f}h' for m in methods))

    # 3) save per-method npy (mirror test_trained_model naming)
    save_dir = args.save_dir or f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    for method in methods:
        out = os.path.join(
            save_dir, f'Result_{method}+{args.out_tag or args.model_name}'
                      f'_{dataset}.npy')
        np.save(out, results[method])
        print(f'  saved {out}')

    # 4) CP-SAT reference + summary table
    ref, ref_path = load_or_reference(args.data_path)
    bases = [os.path.basename(s) for s in stems]

    lines = []
    lines.append(f'# PPVC evaluation summary — {args.model_name}\n')
    lines.append(f'- dataset: `{dataset}`  ({n} instances)')
    lines.append(f'- checkpoint: `{ckpt_path}`')
    lines.append(f'- seed_test: {args.seed_test}'
                 + ('  ' if 'sampling' not in methods
                    else f'  |  sample_times: {args.sample_times}'))
    lines.append(f'- architecture (from snapshot): '
                 f'framework={configs.framework}, '
                 f'use_lag_features={use_lag_features}, '
                 f'use_type_embedding={use_type_embedding}, '
                 f'fea_j_input_dim={getattr(configs, "fea_j_input_dim", "?")}, '
                 f'n_op_types={getattr(configs, "n_op_types", "?")}, '
                 f'n_mch_types={getattr(configs, "n_mch_types", "?")}'
                 + (', bound_lag_free=True (model rollouts)'
                    if _model_bound_lag_free() else ''))

    if ref:
        cov = sum(1 for b in bases if b in ref)
        lines.append(f'- CP-SAT reference: `{ref_path}` '
                     f'(coverage {cov}/{n} instances)')
    else:
        lines.append(f'- CP-SAT reference: not available '
                     f'(`{ref_path}` missing/empty) — gap% omitted')
    lines.append('')

    header = ('| method | mean makespan | std | mean time (s) | feasible | '
              'mean gap% vs CP-SAT (cov) |')
    sep = '|---|---|---|---|---|---|'
    lines.append(header)
    lines.append(sep)

    summary_print = []
    for method in methods:
        arr = results[method]
        mean_ms = arr[:, 0].mean()
        std_ms = arr[:, 0].std()
        mean_t = arr[:, 1].mean()
        # gap% computed only over instances with a CP-SAT reference
        if ref:
            gaps, cov = [], 0
            for k, b in enumerate(bases):
                if b in ref and ref[b] > 0:
                    gaps.append(100.0 * (arr[k, 0] - ref[b]) / ref[b])
                    cov += 1
            gap_str = (f'{np.mean(gaps):.2f}% ({cov})' if gaps else 'n/a (0)')
        else:
            gap_str = 'n/a'
        lines.append(f'| {method} | {mean_ms:.1f} | {std_ms:.1f} | '
                     f'{mean_t:.4f} | {feas[method]}/{n} | {gap_str} |')
        summary_print.append((method, mean_ms, std_ms, mean_t,
                              feas[method], gap_str))

    summary_path = os.path.join(
        save_dir, f'summary_{args.out_tag or args.model_name}.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # console echo
    print('\n' + '=' * 70)
    print(f'SUMMARY  ({dataset}, {n} instances)')
    print('=' * 70)
    print(f'{"method":<10}{"mean_ms":>12}{"std":>10}{"mean_t(s)":>12}'
          f'{"feasible":>11}{"gap%(cov)":>18}')
    for method, mean_ms, std_ms, mean_t, fc, gap_str in summary_print:
        print(f'{method:<10}{mean_ms:>12.1f}{std_ms:>10.1f}{mean_t:>12.4f}'
              f'{fc:>8}/{n:<2}{gap_str:>18}')
    print('=' * 70)
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
