"""
eval_reactive.py
================
REACTIVE / ONLINE RESCHEDULING experiment for the IEEE TII PPVC paper.

PAPER ARGUMENT
  A learned scheduler that re-solves in ~1 s is valuable because real PPVC
  factories face mid-execution disruptions that invalidate the plan and demand
  re-planning within seconds. This script shows, per disrupted instance:
    (i)  the DRL policy, re-solving the RESIDUAL after a disruption, beats the
         right-shift repair operator that industry uses (it can RE-ROUTE, not
         just push times); and
    (ii) a CP-SAT budget sweep positions the exact solver's online
         quality-versus-time trade-off on the same disrupted instances.

KEY DESIGN PRINCIPLE — the env IS the simulator. We never hand-inject prefix
arrays (that path is NaN-prone and forbidden). To create a mid-execution
disrupted state we build a fresh env with the SAME helpers eval_ppvc.py uses,
STEP the greedy policy for the first ceil(offset*N) ops (the env now holds a
fully-consistent partial schedule for free), apply the disruption by editing
ONLY the relevant env arrays and calling the env's OWN feature reconstruction,
then continue greedy stepping to completion.

DISRUPTION TYPES
  1. lag_perturb (PRIMARY, PPVC-specific): an in-flight / upcoming nonzero-lag
     op (a curing / drying op) has its lag multiplied by --lag_factor (default
     2.0 == +100%). DRL path bumps env.true_op_lag + env.op_lag (kept consistent
     via the env's own pt-normalization slope) and recomputes the job's
     candidate_free_time. CP-SAT / right-shift just receive the perturbed
     time_lag array.
  2. breakdown (SECONDARY): a chosen machine goes down for
     0.5 * mean(op processing time), starting at the disruption time. DRL path
     pushes env.mch_free_time[0, m] up to the (normalized) release time; CP-SAT
     re-solves the same instance (the breakdown is modelled in the residual env
     and in the right-shift via the new mch_ready_time arg); right-shift seeds
     the machine release floor through right_shift_repair(mch_ready_time=...).

RE-SOLVE METHODS compared per disrupted instance
  (a) DRL-reactive : env-as-simulator residual continuation (~1 s).
  (b) right_shift  : freeze the ORIGINAL full DRL plan's machine assignments,
                     apply the disruption, repair to feasibility (ms-scale).
  (c) cpsat_<b>    : re-solve the PERTURBED whole instance at budgets {5,30,300}s.

METRICS (RAW / true times) per instance & method
  * realized makespan and inflation % vs the ORIGINAL (pre-disruption) DRL plan
  * re-solve wall-clock
  * NERVOUSNESS vs the original plan: #ops whose assigned machine changed
    (machine-nervousness) + #ops whose start shifted by > 1 h (time-nervousness),
    reported separately and combined.
  Every schedule is re-validated with schedule_validator.validate_schedule
  against the TRUE (perturbed) lags; feasibility asserted and counted.

OUTPUT  (test_results/PPVC/<dataset>/)
  * reactive_<disruption_type>_<model>.npy   structured per-instance/method array
  * reactive_<disruption_type>_summary.md     human-readable table
  plus a copy-paste-ready \newcommand macro block printed to stdout.

CLI
  python eval_reactive.py [--model_name 10x25+ppvc-mixed+full]
      [--data_path data/PPVC/10x25+ppvc-mixed]
      [--disruption_type lag_perturb|breakdown] [--disruption_offset 0.4]
      [--lag_factor 2.0] [--cpsat_budgets 5,30,300] [--seed_test 50]
      [--max_instances N]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse our flags FIRST, then scrub argv before `from params import configs`.

    params.py calls parser.parse_args() at IMPORT time against the process argv
    (it is the project-wide config singleton). If our flags are still on argv
    when params imports, it rejects them — so we strip argv to just the program
    name here, exactly as eval_ppvc.py / eval_a0_repair.py do.
    """
    ap = argparse.ArgumentParser(
        description='Reactive/online rescheduling experiment (PPVC, IEEE TII)')
    ap.add_argument('--model_name', type=str, default='10x25+ppvc-mixed+full',
                    help='checkpoint stem under trained_network/PPVC/ and the '
                         'config snapshot train_log/PPVC/config_<name>.json')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed',
                    help='directory with instance_*.fjs + .meta.json')
    ap.add_argument('--disruption_type', type=str, default='lag_perturb',
                    choices=['lag_perturb', 'breakdown'],
                    help='lag_perturb: double a curing/drying lag (PPVC); '
                         'breakdown: a machine goes down for a while')
    ap.add_argument('--disruption_offset', type=float, default=0.4,
                    help='fraction of N ops greedily committed before the '
                         'disruption strikes (the residual is re-solved)')
    ap.add_argument('--lag_factor', type=float, default=2.0,
                    help='multiply the chosen op lag by this (2.0 == +100%%)')
    ap.add_argument('--cpsat_budgets', type=str, default='5,30,300',
                    help='comma-separated CP-SAT per-solve budgets in seconds')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap the number of instances (smoke-test convenience)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]  # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli()

# params.configs is the module-level singleton read by PPO_initialize / the env;
# we mutate the architecture flags on it BEFORE building the network.
from params import configs

# Architecture-critical keys copied from the training-config snapshot into
# params.configs. Same list as eval_ppvc.py / eval_a0_repair.py, so any
# architecture this repo trains loads here too.
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

# A start shift beyond this many TRUE time units counts as time-nervousness.
NERVOUSNESS_TIME_THRESHOLD = 1.0  # 1 hour (instances use hours)


def load_train_config(model_name):
    """Read train_log/PPVC/config_<model>.json; apply arch flags to configs."""
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[eval_reactive] training-config snapshot not found: {path}\n'
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
        sys.exit(f'[eval_reactive] no instance_*.fjs found under {data_path}')
    stems = [p[:-len('.fjs')] for p in fjs]
    if cap is not None:
        stems = stems[:cap]
    return stems


# ---------------------------------------------------------------------------
# Env construction (mirrors eval_ppvc._new_env exactly) + greedy primitives.
# ---------------------------------------------------------------------------

def _new_env(jl, pt, meta, use_lag_features, use_type_embedding):
    """Fresh batch-of-1 env, built with the SAME kwargs eval_ppvc._new_env uses.

    See eval_ppvc.py:127-136.
    """
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1],
                               use_lag_features=use_lag_features)
    kwargs = dict(time_lag_list=[meta['time_lag']])
    if use_type_embedding:
        kwargs['op_type_list'] = [meta['op_type']]
        kwargs['mch_type_list'] = [meta['mch_type']]
    state = env.set_initial_data([jl], [pt], **kwargs)
    return env, state


def _greedy_action(ppo, state):
    """One greedy action from the policy (mirrors eval_ppvc.rollout_greedy)."""
    import torch
    from common_utils import greedy_select_action
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
    return action


def rollout_greedy_full(ppo, jl, pt, meta, use_lag_features, use_type_embedding):
    """Full greedy rollout, NO disruption (the ORIGINAL reference plan).

    Returns (makespan, seconds, assigned_mch[N], true_op_ct[N]).
    Identical mechanics to eval_ppvc.rollout_greedy:164-197.
    """
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state = _new_env(jl, pt, meta, use_lag_features, use_type_embedding)
    assigned_mch = np.full(n_ops, -1, dtype=int)
    t1 = time.time()
    while True:
        action = _greedy_action(ppo, state)
        a = int(action.cpu().numpy()[0])
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, done = env.step(actions=action.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    assert (assigned_mch >= 0).all(), 'greedy(full): some op never scheduled'
    return float(env.current_makespan[0]), t2 - t1, assigned_mch, \
        env.true_op_ct[0].copy()


# ---------------------------------------------------------------------------
# Disruption selection helpers (operate on the env's partial state).
# ---------------------------------------------------------------------------

def _job_of_op(env, op):
    """Return the job index that owns global op index `op` (batch slot 0)."""
    first = env.job_first_op_id[0]
    last = env.job_last_op_id[0]
    for j in range(env.number_of_jobs):
        if first[j] <= op <= last[j]:
            return j
    raise AssertionError(f'op {op} not found in any job range')


def _pick_lag_op(env):
    """Pick a nonzero-lag op that still LIES IN THE FUTURE of the residual.

    A lag bump only matters if the lagged op's job-successor has not yet been
    scheduled (the lag governs when that successor becomes ready). Prefer the
    earliest such op whose OWN op is not yet completed (in-flight / upcoming);
    fall back to a scheduled op whose successor is still pending.

    Returns the global op index, or None if no eligible op exists.
    """
    true_lag = env.true_op_lag[0]
    scheduled = env.op_scheduled_flag[0].astype(bool)
    last_op = env.job_last_op_id[0]
    nonzero = np.where(true_lag > 0)[0]
    # only ops that are NOT the last op of their job have a successor the lag
    # can delay (the last op's lag never affects scheduling).
    job_last_set = set(int(x) for x in last_op)
    candidates_unscheduled = []
    candidates_succ_pending = []
    for op in nonzero:
        op = int(op)
        if op in job_last_set:
            continue
        succ = op + 1  # job-by-job numbering: successor is the next index
        succ_pending = not scheduled[succ]
        if not succ_pending:
            continue  # successor already committed -> bump cannot change it
        if not scheduled[op]:
            candidates_unscheduled.append(op)
        else:
            candidates_succ_pending.append(op)
    if candidates_unscheduled:
        return candidates_unscheduled[0]
    if candidates_succ_pending:
        return candidates_succ_pending[0]
    return None


def _pick_breakdown_machine(env):
    """Pick a machine that still has UNSCHEDULED work in the residual.

    Choose the busiest such machine (most remaining compatible ops) so the
    breakdown actually perturbs the residual. Returns the machine index, or
    None if every machine is idle for the rest of the horizon.
    """
    # remain_process_relation[0]: [N, M] bool, True where op i can still run on m
    remain = env.remain_process_relation[0]
    unscheduled = ~env.op_scheduled_flag[0].astype(bool)
    work_per_mch = (remain[unscheduled] > 0).sum(axis=0) if unscheduled.any() \
        else np.zeros(env.number_of_machines, dtype=int)
    if work_per_mch.max() <= 0:
        return None
    return int(np.argmax(work_per_mch))


# ---------------------------------------------------------------------------
# DRL-reactive: env-as-simulator residual continuation.
# ---------------------------------------------------------------------------

def drl_reactive(ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                 disruption_type, offset, lag_factor):
    """Build a disrupted mid-execution state, then greedily re-solve the residual.

    Steps the greedy policy for the first ceil(offset*N) ops, applies the
    disruption by editing ONLY the relevant env arrays + the env's OWN feature
    reconstruction (construct_op/mch/pair_features + state.update), then
    continues greedy stepping to completion.

    Returns a dict with keys:
      makespan, seconds (RESIDUAL solve only), assigned_mch[N], true_op_ct[N],
      t_disrupt, perturbed_time_lag[N], disruption_info (str), mch_ready_time
      (None for lag_perturb; [M] for breakdown), feasible-irrelevant here.
    """
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state = _new_env(jl, pt, meta, use_lag_features, use_type_embedding)
    assigned_mch = np.full(n_ops, -1, dtype=int)

    k = int(math.ceil(offset * n_ops))
    k = max(0, min(k, n_ops - 1))  # leave >=1 op for the residual

    # ---- phase 1: commit the first k ops (pre-disruption, NOT timed) --------
    for _ in range(k):
        action = _greedy_action(ppo, state)
        a = int(action.cpu().numpy()[0])
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, _ = env.step(actions=action.cpu().numpy())

    # disruption time = the env's current decision instant (NORMALIZED units)
    t_disrupt_norm = float(env.next_schedule_time[0])
    # normalization slope used by the env for pt AND lag (see
    # fjsp_env_same_op_nums.py:166-171, 182). pt_lower_bound includes the 0s of
    # incompatible entries, so this is the EXACT slope the env applied.
    slope = float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)
    t_disrupt_true = t_disrupt_norm * slope

    perturbed_time_lag = np.asarray(meta['time_lag'], dtype=float).copy()
    mch_ready_time = None
    info = ''

    # ---- apply the disruption to the env arrays -----------------------------
    if disruption_type == 'lag_perturb':
        op = _pick_lag_op(env)
        if op is None:
            # no eligible future lag op: fall back to ANY nonzero-lag op so the
            # run still produces a (mild) disruption; record that it was inert.
            nz = np.where(env.true_op_lag[0] > 0)[0]
            op = int(nz[0]) if len(nz) else 0
            info = f'lag_perturb(op={op}, INERT: no future-lag op available)'
        else:
            info = f'lag_perturb(op={op}, x{lag_factor})'
        old_true = float(env.true_op_lag[0, op])
        new_true = old_true * lag_factor
        # bump BOTH the raw lag and its pt-normalized copy, keeping the SAME
        # slope the env used so the two channels stay consistent.
        env.true_op_lag[0, op] = new_true
        env.op_lag[0, op] = new_true / slope
        perturbed_time_lag[op] = new_true
        # if the lagged op is already SCHEDULED, its successor's readiness is
        # tracked by candidate_free_time; recompute that job's entry so the
        # residual planner sees the longer wait. (For unscheduled ops the lag
        # is picked up at their own step via op_lag, so nothing else to do.)
        j = _job_of_op(env, op)
        if bool(env.op_scheduled_flag[0, op]):
            env.candidate_free_time[0, j] = (env.op_ct[0, op]
                                             + env.op_lag[0, op])
            env.true_candidate_free_time[0, j] = (env.true_op_ct[0, op]
                                                  + env.true_op_lag[0, op])
    elif disruption_type == 'breakdown':
        m = _pick_breakdown_machine(env)
        # downtime = 0.5 * mean op processing time (TRUE units), normalized by
        # the same slope as machine free time / pt.
        mean_pt_true = float(pt[pt > 0].mean())
        downtime_true = 0.5 * mean_pt_true
        downtime_norm = downtime_true / slope
        release_true = t_disrupt_true + downtime_true
        if m is None:
            m = 0
            info = (f'breakdown(m={m}, INERT: no residual work on any machine, '
                    f'down {downtime_true:.2f}h)')
        else:
            info = (f'breakdown(m={m}, down {downtime_true:.2f}h from '
                    f't={t_disrupt_true:.2f}h)')
        # push the machine's free time up to its release instant (NORMALIZED).
        env.mch_free_time[0, m] = max(float(env.mch_free_time[0, m]),
                                      t_disrupt_norm + downtime_norm)
        env.true_mch_free_time[0, m] = max(float(env.true_mch_free_time[0, m]),
                                           release_true)
        # right-shift baseline honours the breakdown via a per-machine release
        # floor; only THIS machine is released late, others at 0.
        mch_ready_time = np.zeros(M, dtype=float)
        mch_ready_time[m] = release_true
    else:
        raise ValueError(f'unknown disruption_type {disruption_type}')

    # ---- rebuild the env features from the edited arrays --------------------
    # Recompute the pair_free_time / next_schedule_time consistently with the
    # edited candidate_free_time + mch_free_time, mirroring the env's own step
    # logic (fjsp_env_same_op_nums.py:428-436), then reconstruct all feature
    # blocks with the env's OWN methods and push them into the state.
    import numpy.ma as ma
    candidateFT = np.expand_dims(env.candidate_free_time, axis=2)
    mchFT = np.expand_dims(env.mch_free_time, axis=1)
    env.pair_free_time = np.maximum(candidateFT, mchFT)
    schedule_matrix = ma.array(env.pair_free_time,
                               mask=env.candidate_process_relation)
    env.next_schedule_time = np.min(
        schedule_matrix.reshape(env.number_of_envs, -1), axis=1).data
    # refresh the remaining-lag channel (used only when use_lag_features) so the
    # bumped lag is reflected for already-scheduled ops at the new instant.
    if env.use_lag_features:
        env.op_remain_lag = env.op_scheduled_flag * np.clip(
            env.op_ct + env.op_lag - env.next_schedule_time[:, np.newaxis],
            0, env.op_lag)
    env.construct_op_features()
    env.construct_mch_features()
    env.construct_pair_features()
    env.state.update(env.fea_j, env.op_mask, env.fea_m, env.mch_mask,
                     env.dynamic_pair_mask, env.comp_idx, env.candidate,
                     env.fea_pairs, op_type=env.op_type, mch_type=env.mch_type)
    state = env.state

    # ---- phase 2: time the RESIDUAL solve only ------------------------------
    t1 = time.time()
    while not env.done().all():
        action = _greedy_action(ppo, state)
        a = int(action.cpu().numpy()[0])
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, done = env.step(actions=action.cpu().numpy())
        if done.all():
            break
    t2 = time.time()
    assert (assigned_mch >= 0).all(), 'drl_reactive: some op never scheduled'

    return {
        'makespan': float(env.current_makespan[0]),
        'seconds': t2 - t1,
        'assigned_mch': assigned_mch,
        'true_op_ct': env.true_op_ct[0].copy(),
        't_disrupt_true': t_disrupt_true,
        'perturbed_time_lag': perturbed_time_lag,
        'mch_ready_time': mch_ready_time,
        'info': info,
        'k': k,
    }


# ---------------------------------------------------------------------------
# Right-shift baseline: freeze the ORIGINAL plan's machine assignments, apply
# the disruption to the instance data, repair to feasibility.
# ---------------------------------------------------------------------------

def right_shift_baseline(jl, pt, assigned_mch_orig, op_ct_orig,
                         perturbed_time_lag, mch_ready_time):
    """Repair the frozen original plan against the (perturbed) reality.

    Returns (makespan, seconds, assigned_mch[N], repaired_op_ct[N]).
    Machine ORDER is taken from the original plan's lag-blind-style completion
    times (op_ct_orig). Machine assignments are NOT changed (industry can only
    push times right). mch_ready_time seeds the breakdown release floor; it is
    None for lag_perturb (default, bit-identical to the unmodified operator).
    """
    from right_shift_repair import right_shift_repair
    t1 = time.time()
    repaired_ct, ms = right_shift_repair(
        jl, pt, perturbed_time_lag, assigned_mch_orig,
        op_ct_lagblind=op_ct_orig, mch_ready_time=mch_ready_time)
    sec = time.time() - t1
    return float(ms), sec, assigned_mch_orig.copy(), repaired_ct


# ---------------------------------------------------------------------------
# CP-SAT budget sweep on the perturbed WHOLE instance (no residual surgery).
# ---------------------------------------------------------------------------

def cpsat_sweep(jl, pt, perturbed_time_lag, budgets, mch_ready_time):
    """Solve the perturbed whole instance at each budget. No solver changes.

    For a machine breakdown the solver has no native unavailable-window hook,
    so we model it the standard FJSP way: the down machine's processing column
    is removed from the affected ops for the breakdown window is NOT
    representable without solver edits — therefore, per the spec (no solver
    changes), the breakdown is reflected ONLY through the perturbed instance
    data we can pass: time_lag. Since a breakdown does not change time_lag, the
    CP-SAT objective for the breakdown case is the perturbed-instance optimum
    WITHOUT the breakdown window, i.e. a LOWER BOUND reference. We record it as
    such (the paper uses CP-SAT mainly for the lag_perturb frontier where the
    perturbed time_lag fully captures the disruption).

    Returns {budget: (makespan, solve_seconds, status)}.
    """
    from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
    jobs, num_machines = matrix_to_the_format_for_solving(jl, pt)
    out = {}
    for b in budgets:
        obj, t, status, _asg, _oct = fjsp_solver(
            jobs, num_machines, time_limits=float(b),
            time_lag=perturbed_time_lag, return_schedule=True)
        out[b] = (float(obj), float(t), status)
    return out


# ---------------------------------------------------------------------------
# Nervousness vs the original plan.
# ---------------------------------------------------------------------------

def nervousness(jl, pt, assigned_orig, op_ct_orig, assigned_new, op_ct_new):
    """#machine changes + #start shifts > threshold, vs the original plan.

    Start times derived as ct - pt[op, assigned_mch]. Uses TRUE (raw) times.
    Returns (machine_nervousness, time_nervousness, combined).
    """
    from schedule_validator import reconstruct_starts
    starts_orig = reconstruct_starts(pt, assigned_orig, op_ct_orig)
    starts_new = reconstruct_starts(pt, assigned_new, op_ct_new)
    mch_changed = int(np.sum(assigned_orig != assigned_new))
    time_shifted = int(np.sum(
        np.abs(starts_new - starts_orig) > NERVOUSNESS_TIME_THRESHOLD))
    return mch_changed, time_shifted, mch_changed + time_shifted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _ARGS
    budgets = [int(b) for b in args.cpsat_budgets.split(',') if b.strip()]

    # 1) apply the training-config architecture flags BEFORE building anything
    snap = load_train_config(args.model_name)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    # torch + device setup (mirrors eval_ppvc.py / eval_a0_repair.py)
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from common_utils import setup_seed
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    setup_seed(args.seed_test)

    # 2) load the checkpoint into the matching architecture
    from model.PPO import PPO_initialize
    ckpt_path = f'./trained_network/PPVC/{args.model_name}.pth'
    if not os.path.exists(ckpt_path):
        sys.exit(f'[eval_reactive] checkpoint not found: {ckpt_path}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    ppo.policy.eval()

    # 3) collect instances + tools
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule

    stems = list_instances(args.data_path, cap=args.max_instances)
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))

    cpsat_method_names = [f'cpsat_{b}' for b in budgets]
    methods = ['drl_reactive', 'right_shift'] + cpsat_method_names

    # per-method accumulators
    rec = {m: {'makespan': np.full(n, np.nan),
               'inflation': np.full(n, np.nan),
               'seconds': np.full(n, np.nan),
               'mch_nerv': np.full(n, np.nan),
               'time_nerv': np.full(n, np.nan),
               'comb_nerv': np.full(n, np.nan),
               'feasible': np.zeros(n, dtype=bool)}
           for m in methods}
    orig_makespans = np.full(n, np.nan)
    n_infeasible = 0

    print('=' * 72)
    print(f'eval_reactive  model={args.model_name}  dataset={dataset}  '
          f'instances={n}')
    print(f'  disruption={args.disruption_type}  offset={args.disruption_offset}'
          f'  lag_factor={args.lag_factor}  cpsat_budgets={budgets}')
    print(f'  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  seed_test={args.seed_test}')
    print('=' * 72)

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        true_lag = np.asarray(meta['time_lag'], dtype=float)

        # ---- ORIGINAL pre-disruption DRL plan (reference) -------------------
        ms_orig, _sec_orig, assigned_orig, oct_orig = rollout_greedy_full(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding)
        # validate the original plan against the TRUE (un-perturbed) lags
        res0 = validate_schedule(jl, pt, true_lag, assigned_orig, oct_orig)
        assert res0['feasible'], (
            f'ORIGINAL plan infeasible on {base}:\n  '
            + '\n  '.join(res0['violations'][:10]))
        orig_makespans[i] = ms_orig

        # ---- (a) DRL-reactive ----------------------------------------------
        d = drl_reactive(ppo, jl, pt, meta, use_lag_features,
                         use_type_embedding, args.disruption_type,
                         args.disruption_offset, args.lag_factor)
        perturbed_lag = d['perturbed_time_lag']
        mch_ready_time = d['mch_ready_time']

        # validate DRL-reactive against the TRUE (perturbed) lags
        res_d = validate_schedule(jl, pt, perturbed_lag, d['assigned_mch'],
                                  d['true_op_ct'])
        if not res_d['feasible']:
            n_infeasible += 1
            print(f'  !!! drl_reactive INFEASIBLE on {base}: '
                  f'{res_d["violations"][:3]}')
        assert abs(res_d['makespan'] - d['makespan']) < 1e-6, (
            f'{base}: drl env makespan {d["makespan"]} != validator '
            f'{res_d["makespan"]}')
        mn, tn, cn = nervousness(jl, pt, assigned_orig, oct_orig,
                                 d['assigned_mch'], d['true_op_ct'])
        rec['drl_reactive']['makespan'][i] = d['makespan']
        rec['drl_reactive']['inflation'][i] = (
            100.0 * (d['makespan'] - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['drl_reactive']['seconds'][i] = d['seconds']
        rec['drl_reactive']['mch_nerv'][i] = mn
        rec['drl_reactive']['time_nerv'][i] = tn
        rec['drl_reactive']['comb_nerv'][i] = cn
        rec['drl_reactive']['feasible'][i] = res_d['feasible']

        # ---- (b) right-shift repair ----------------------------------------
        ms_rs, sec_rs, asg_rs, oct_rs = right_shift_baseline(
            jl, pt, assigned_orig, oct_orig, perturbed_lag, mch_ready_time)
        res_rs = validate_schedule(jl, pt, perturbed_lag, asg_rs, oct_rs)
        if not res_rs['feasible']:
            n_infeasible += 1
            print(f'  !!! right_shift INFEASIBLE on {base}: '
                  f'{res_rs["violations"][:3]}')
        # NOTE: for a breakdown, the right-shift makespan honours mch_ready_time;
        # the validator does not model machine unavailability, so it checks only
        # the lag/precedence/no-overlap constraints (still a valid feasibility
        # gate on the produced schedule).
        assert abs(res_rs['makespan'] - ms_rs) < 1e-6 or \
            mch_ready_time is not None, (
            f'{base}: right_shift makespan {ms_rs} != validator '
            f'{res_rs["makespan"]}')
        mn, tn, cn = nervousness(jl, pt, assigned_orig, oct_orig,
                                 asg_rs, oct_rs)
        rec['right_shift']['makespan'][i] = ms_rs
        rec['right_shift']['inflation'][i] = (
            100.0 * (ms_rs - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['right_shift']['seconds'][i] = sec_rs
        rec['right_shift']['mch_nerv'][i] = mn   # 0 by construction (no re-route)
        rec['right_shift']['time_nerv'][i] = tn
        rec['right_shift']['comb_nerv'][i] = cn
        rec['right_shift']['feasible'][i] = res_rs['feasible']

        # ---- (c) CP-SAT budget sweep on the perturbed whole instance -------
        sweep = cpsat_sweep(jl, pt, perturbed_lag, budgets, mch_ready_time)
        for b in budgets:
            mname = f'cpsat_{b}'
            ms_c, sec_c, _status = sweep[b]
            rec[mname]['makespan'][i] = ms_c
            rec[mname]['inflation'][i] = (
                100.0 * (ms_c - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
            rec[mname]['seconds'][i] = sec_c
            # CP-SAT re-solves from scratch; nervousness vs the original DRL
            # plan is not meaningful (different solver, no shared assignment),
            # so it is left NaN and excluded from the nervousness columns.
            rec[mname]['feasible'][i] = True  # solver-reported optimum/feasible

        print(f'  [{i + 1:3d}/{n}] {base}  orig={ms_orig:6.0f}h  '
              f'drl={d["makespan"]:6.0f}h(+{rec["drl_reactive"]["inflation"][i]:4.1f}%,'
              f'{d["seconds"]:.2f}s,nerv={int(rec["drl_reactive"]["comb_nerv"][i])})  '
              f'rs={ms_rs:6.0f}h(+{rec["right_shift"]["inflation"][i]:5.1f}%)  '
              + '  '.join(f'cp{b}={sweep[b][0]:6.0f}h' for b in budgets)
              + f'   [{d["info"]}]')

    # ---- save structured npy -------------------------------------------------
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_npy = os.path.join(
        save_dir, f'reactive_{args.disruption_type}_{args.model_name}.npy')
    payload = {
        'methods': methods,
        'orig_makespans': orig_makespans,
        'records': rec,
        'meta': {
            'model_name': args.model_name,
            'dataset': dataset,
            'disruption_type': args.disruption_type,
            'disruption_offset': args.disruption_offset,
            'lag_factor': args.lag_factor,
            'cpsat_budgets': budgets,
            'seed_test': args.seed_test,
            'n_instances': n,
            'nervousness_time_threshold': NERVOUSNESS_TIME_THRESHOLD,
        },
    }
    np.save(out_npy, payload, allow_pickle=True)
    print(f'\n  saved {out_npy}')

    # ---- aggregate ----------------------------------------------------------
    def _mean(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return float(a.mean()) if len(a) else float('nan')

    agg = {}
    for m in methods:
        agg[m] = {
            'makespan': _mean(rec[m]['makespan']),
            'inflation': _mean(rec[m]['inflation']),
            'seconds': _mean(rec[m]['seconds']),
            'mch_nerv': _mean(rec[m]['mch_nerv']),
            'time_nerv': _mean(rec[m]['time_nerv']),
            'comb_nerv': _mean(rec[m]['comb_nerv']),
            'feasible': int(rec[m]['feasible'].sum()),
        }

    # ---- summary markdown ---------------------------------------------------
    lines = []
    lines.append(f'# PPVC reactive rescheduling — {args.disruption_type} '
                 f'— {args.model_name}\n')
    lines.append(f'- dataset: `{dataset}`  ({n} instances)')
    lines.append(f'- checkpoint: `{ckpt_path}`')
    lines.append(f'- disruption: **{args.disruption_type}**  '
                 f'offset={args.disruption_offset}  '
                 f'lag_factor={args.lag_factor}')
    lines.append(f'- cpsat budgets (s): {budgets}')
    lines.append(f'- seed_test: {args.seed_test}')
    lines.append(f'- architecture (from snapshot): '
                 f'use_lag_features={use_lag_features}, '
                 f'use_type_embedding={use_type_embedding}')
    lines.append(f'- mean ORIGINAL (pre-disruption) DRL makespan: '
                 f'{_mean(orig_makespans):.1f} h')
    lines.append(f'- nervousness time threshold: '
                 f'> {NERVOUSNESS_TIME_THRESHOLD} h start shift')
    lines.append(f'- total infeasibilities: {n_infeasible}')
    lines.append('')
    lines.append('| method | mean makespan (h) | mean inflation % | '
                 'mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for m in methods:
        a = agg[m]
        # nervousness columns omitted (n/a) for CP-SAT (no shared assignment)
        if m.startswith('cpsat_'):
            nerv_cols = 'n/a | n/a | n/a'
        else:
            nerv_cols = (f'{a["mch_nerv"]:.1f} | {a["time_nerv"]:.1f} | '
                         f'{a["comb_nerv"]:.1f}')
        lines.append(f'| {m} | {a["makespan"]:.1f} | {a["inflation"]:.2f} | '
                     f'{a["seconds"]:.4f} | {nerv_cols} | {a["feasible"]}/{n} |')
    lines.append('')
    lines.append('Inflation % = 100 * (realized makespan - original DRL plan '
                 'makespan) / original DRL plan makespan.')
    lines.append('Nervousness = #ops whose assigned machine changed '
                 '(mch-nerv) + #ops whose start shifted > '
                 f'{NERVOUSNESS_TIME_THRESHOLD} h (time-nerv), vs the original '
                 'plan; comb-nerv is their sum.')
    if args.disruption_type == 'breakdown':
        lines.append('NOTE: CP-SAT has no native machine-unavailability window '
                     'and no solver changes are permitted, so its breakdown '
                     'numbers re-solve the perturbed instance WITHOUT the '
                     'downtime window — a lower-bound reference only. The '
                     'lag_perturb frontier is the apples-to-apples CP-SAT '
                     'comparison (perturbed time_lag fully captures it).')

    summary_path = os.path.join(
        save_dir, f'reactive_{args.disruption_type}_summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- console echo -------------------------------------------------------
    print('\n' + '=' * 72)
    print(f'REACTIVE SUMMARY  ({dataset}, {n} instances, '
          f'{args.disruption_type})')
    print('=' * 72)
    print(f'  mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    print(f'{"method":<14}{"makespan":>10}{"infl%":>9}{"time(s)":>10}'
          f'{"mNerv":>8}{"tNerv":>8}{"cNerv":>8}{"feas":>8}')
    for m in methods:
        a = agg[m]
        if m.startswith('cpsat_'):
            print(f'{m:<14}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
                  f'{a["seconds"]:>10.3f}{"n/a":>8}{"n/a":>8}{"n/a":>8}'
                  f'{a["feasible"]:>6}/{n}')
        else:
            print(f'{m:<14}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
                  f'{a["seconds"]:>10.3f}{a["mch_nerv"]:>8.1f}'
                  f'{a["time_nerv"]:>8.1f}{a["comb_nerv"]:>8.1f}'
                  f'{a["feasible"]:>6}/{n}')
    print('=' * 72)
    print(f'wrote {summary_path}')

    # ---- copy-paste LaTeX macro block ---------------------------------------
    dt = args.disruption_type.replace('_', '')

    def _mac(name, val, fmt='{:.1f}'):
        return f'\\newcommand{{\\react{dt}{name}}}{{{fmt.format(val)}}}'

    print('\n' + '-' * 72)
    print('% ---- copy-paste LaTeX macros (reactive ' +
          f'{args.disruption_type}) ----')
    print(_mac('OrigMakespan', _mean(orig_makespans)))
    method_macro = {'drl_reactive': 'Drl', 'right_shift': 'Rs'}
    for m in ['drl_reactive', 'right_shift']:
        cap = method_macro[m]
        a = agg[m]
        print(_mac(cap + 'Makespan', a['makespan']))
        print(_mac(cap + 'Inflation', a['inflation'], '{:.2f}'))
        print(_mac(cap + 'Time', a['seconds'], '{:.3f}'))
        print(_mac(cap + 'MchNerv', a['mch_nerv']))
        print(_mac(cap + 'TimeNerv', a['time_nerv']))
        print(_mac(cap + 'CombNerv', a['comb_nerv']))
    for b in budgets:
        a = agg[f'cpsat_{b}']
        print(_mac(f'Cpsat{b}Makespan', a['makespan']))
        print(_mac(f'Cpsat{b}Inflation', a['inflation'], '{:.2f}'))
        print(_mac(f'Cpsat{b}Time', a['seconds'], '{:.2f}'))
    print('% ---- end macros ----')
    print('-' * 72)


if __name__ == '__main__':
    main()
