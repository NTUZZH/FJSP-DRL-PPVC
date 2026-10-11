"""
eval_reactive.py
================
REACTIVE / ONLINE RESCHEDULING experiment for the IEEE TII PPVC paper.

PAPER ARGUMENT
  A learned scheduler that re-solves in ~1 s is valuable because real PPVC
  factories face mid-execution disruptions that invalidate the plan and demand
  re-planning within seconds. This script shows, per disrupted instance:
    (i)  the DRL policy, re-solving the RESIDUAL after a disruption, against
         the right-shift repair operator that industry uses (it can RE-ROUTE,
         not just push times), a lag-aware SPT rule and a short-budget GA on
         the same residual; and
    (ii) a CP-SAT budget sweep positions the exact solver's online
         quality-versus-time trade-off on the same disrupted instances.

KEY DESIGN PRINCIPLE -- the env IS the simulator. We never hand-inject prefix
arrays (that path is NaN-prone and forbidden). To create a mid-execution
disrupted state we build a fresh env with the SAME helpers eval_ppvc.py uses,
STEP the greedy policy for the first k = ceil(offset*N) ops (the env now holds
a fully-consistent partial schedule for free), apply the disruption by editing
ONLY the relevant env arrays and calling the env's OWN feature reconstruction,
then continue stepping to completion. The disruption instant t_d is the env's
decision instant after the k-th step. The committed prefix is the set of ops
stepped before t_d; they keep their realized machine and times in every
residual method (DRL, SPT, GA, the residual right-shift and the residual
CP-SAT of eval_residual_cpsat.py).

DISRUPTION TYPES (all applied at the same offset)
  1. lag_perturb (PRIMARY, PPVC-specific, unchanged): an in-flight / upcoming
     nonzero-lag op (a curing / drying op) has its lag multiplied by
     --lag_factor (default 2.0 == +100%). Env: env.true_op_lag + env.op_lag
     (same pt-normalization slope) and, if the op is already scheduled, the
     job's candidate_free_time. CP-SAT / right-shift receive the perturbed
     time_lag array.
  2. breakdown (SECONDARY, unchanged): the busiest machine with residual work
     goes down for 0.5 * mean(op processing time) from t_d. Env:
     env.mch_free_time[0, m] (and the true copy) is raised to the release
     instant. Right-shift receives mch_ready_time (a floor on the FIRST op of
     that machine's frozen sequence). The whole-instance CP-SAT cannot encode
     the outage (lower-bound reference only).
     These two legacy types keep their original feature refresh (pair free
     times, next decision instant, feature blocks) so their numbers are
     bit-identical to the earlier runs.
  3. urgent_arrival: a new module is released at t_d. Its routing class,
     durations and lags are drawn by ppvc_instance_generator with the
     dataset's class mix and the seed (instance seed + URGENT_SEED_OFFSET),
     where the instance seed is dataset_meta.json seed0 + the instance index.
     The env cannot grow, so a NEW env is built for the realized instance
     (original modules + the new module appended as the last job, so every
     old op keeps its index); the k committed actions are REPLAYED in the
     same order (asserted to reproduce the committed completion times
     exactly), then the new job's candidate_free_time is raised to t_d and
     its completion-time lower bounds are shifted to start at t_d.
     Right-shift: the original plan stays frozen and the new module's ops
     are appended, in route order, to the end of the queue of the compatible
     machine that finishes each op earliest. Whole CP-SAT: solves the
     realized instance from t=0 (the new module may start before t_d and the
     prefix is not fixed: reference only).
  4. material_shortage: the next unscheduled op of the critical job (most
     remaining mean-pt work among unfinished jobs) cannot start before
     t_d + delay, delay = --shortage_factor (0.5) x the job's mean remaining
     processing time (--shortage_basis op_mean: the mean, over the job's
     remaining ops, of each op's mean compatible pt; remaining_work: their
     sum). Implemented as a job release time: env.candidate_free_time and
     env.true_candidate_free_time of the job are raised to the release (and
     the job's completion-time lower bounds shifted). Right-shift and GA
     receive it as a per-op release floor; the residual CP-SAT as
     start >= ceil(release). The whole-instance CP-SAT has no release hook
     (fjsp_solver cannot be changed), so it solves the undisrupted instance
     (lower-bound reference only).
  5. pt_deviation: the in-flight op (start <= t_d < completion) on the
     busiest machine that has one takes --pt_factor (1.5) x its processing
     time, rounded UP to whole hours (the CP-SAT models are integer). Env:
     the op's op_ct / true_op_ct, the machine's free time, the job's
     candidate free time, the current makespan and the job's completion-time
     lower bounds are pushed by the extra duration. The realized instance
     carries the longer pt on that (op, machine) pair; right-shift, GA and
     the residual CP-SAT receive the longer realized completion as part of
     the committed prefix. Whole CP-SAT solves the instance with the longer
     pt on that pair (reference: the prefix is not fixed).
  6. cancellation: the critical unfinished job's remaining (uncommitted) ops
     are removed; its committed ops stay as realized history (a job with no
     committed op disappears). The env cannot shrink, so a NEW env is built
     for the realized instance and the committed actions are replayed (job
     indices remapped; asserted to reproduce the committed times). The op
     count drops by the removed ops. Right-shift keeps the frozen machine
     order of the remaining ops and re-times them (not before t_d); whole
     CP-SAT solves the reduced instance from t=0 (reference only).
  New types 3-6 refresh every decision feature the env's step() refreshes
  (pair free times, next decision instant, deleted nodes, waiting times,
  dynamic pair mask, competition index, machine mask and features).

RE-SOLVE METHODS compared per disrupted instance
  (a) drl_reactive : env-as-simulator residual continuation with the greedy
                     policy (~1 s).
  (b) spt_reactive : the lag-aware SPT rule (SPTL: processing time plus the
                     op's post-op lag, common_utils.heuristic_select_action)
                     continuing from the SAME disrupted env state
                     (np.random seeded with --seed_test for tie-breaks).
  (c) ga_reactive  : eval_ga's GA operators (tournament, POX on the operation
                     sequence, uniform crossover on the machine vector, swap /
                     reassign mutation, elitism, random immigrants on
                     stagnation, machine-reassignment local search) on the
                     RESIDUAL problem: the chromosome covers only the
                     uncommitted ops, and the decoder places them after the
                     committed prefix (start >= t_d, job ready, machine free,
                     release floors). Seeded with the residual right-shift
                     plan, so by elitism it is never worse than that plan;
                     --ga_budget seconds (default 5).
  (d) right_shift  : freeze the ORIGINAL full DRL plan's machine assignments
                     and machine order, apply the disruption, repair to
                     feasibility (ms-scale). lag_perturb uses
                     right_shift_repair unchanged; every other type,
                     breakdown included, uses the prefix-respecting residual
                     version (right_shift_residual), which keeps every
                     executed operation fixed. (The legacy operator put the
                     breakdown release floor on the FIRST op of the broken
                     machine's whole sequence, which moved operations that
                     had already run; see right_shift_arm.)
  (e) cpsat_<b>    : re-solve the PERTURBED whole instance from scratch at
                     budgets {5,30,300}s (reference; the prefix-respecting
                     residual CP-SAT lives in eval_residual_cpsat.py).

METRICS (RAW / true times) per instance & method
  * realized makespan and inflation % vs the ORIGINAL (pre-disruption) DRL plan
  * re-planning wall-clock (residual solve only)
  * NERVOUSNESS vs the original plan, over the ops present in both plans:
    #ops whose assigned machine changed (mch-nerv), #ops whose start shifted
    by > 1 h (time-nerv), their sum (comb-nerv), and the mean absolute start
    shift in hours (abs-shift).
  Every schedule is re-validated with schedule_validator.validate_schedule on
  the REALIZED instance (perturbed lags / longer pt / added or removed ops).
  Residual methods are also checked against the residual contract: committed
  ops unchanged, uncommitted ops start at or after t_d and after every release
  floor (job release, machine outage). Feasibility is asserted and counted.

OUTPUT  (test_results/PPVC/<dataset>/, or --out_dir)
  * reactive_<disruption_type>_<model>.npy   structured per-instance/method array
  * reactive_<disruption_type>_summary.md     human-readable table
  plus a copy-paste-ready \\newcommand macro block printed to stdout.
  Every record also carries contention evidence (cpu_share_min,
  n_timed_solves, n_hit_limit, n_low_share, n_no_evidence, affinity,
  loadavg_start); the time-limited arms (ga_reactive, cpsat_<b>) add their
  solve's solve_wall_s, solve_cpu_s, workers, cpu_share, status, time_limit_s
  and hit_limit. A budget-limited solve with cpu_share < 0.9 is invalid.

CLI
  python eval_reactive.py [--model_name 10x25+ppvc-mixed+full]
      [--data_path data/PPVC/10x25+ppvc-mixed]
      [--disruption_type lag_perturb|breakdown|urgent_arrival|
                         material_shortage|pt_deviation|cancellation]
      [--disruption_offset 0.4] [--lag_factor 2.0] [--pt_factor 1.5]
      [--shortage_factor 0.5] [--shortage_basis op_mean|remaining_work]
      [--methods drl_reactive,spt_reactive,ga_reactive,right_shift]
      [--ga_budget 5] [--ga_pop 100] [--ga_seed 0]
      [--cpsat_budgets 5,30,300] [--cpsat_workers 0] [--device cpu]
      [--seed_test 50] [--max_instances N] [--out_dir DIR]

The module is importable (eval_residual_cpsat.py and the tests reuse its
disruption construction); the CLI is parsed only when run as a script.
"""
import argparse
import contextlib
import copy
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

# contention evidence of the time-limited solves (no import-time side effects)
from eval_stochastic_lags import (SolveClock, contention_arrays,
                                  contention_summary, contention_table,
                                  cpsat_timing, loadavg_now, solve_timing,
                                  store_contention, HIT_LIMIT_FRAC,
                                  LOW_CPU_SHARE)

LEGACY_DISRUPTIONS = ('lag_perturb', 'breakdown')
NEW_DISRUPTIONS = ('urgent_arrival', 'material_shortage', 'pt_deviation',
                   'cancellation')
ALL_DISRUPTIONS = LEGACY_DISRUPTIONS + NEW_DISRUPTIONS
RESIDUAL_METHODS = ('drl_reactive', 'spt_reactive', 'ga_reactive',
                    'right_shift')

# Seed offset for the urgent module: well above every dataset seed range, so
# the arrival draw never coincides with an instance of any test set.
URGENT_SEED_OFFSET = 10_000_000


def parse_cli():
    """Parse our flags FIRST, then scrub argv before `from params import configs`.

    params.py calls parser.parse_args() at IMPORT time against the process argv
    (it is the project-wide config singleton). If our flags are still on argv
    when params imports, it rejects them -- so we strip argv to just the program
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
                    choices=list(ALL_DISRUPTIONS),
                    help='lag_perturb: double a curing/drying lag (PPVC); '
                         'breakdown: a machine goes down for a while; '
                         'urgent_arrival: a new module is released at t_d; '
                         'material_shortage: a job waits for material; '
                         'pt_deviation: an in-flight op runs longer; '
                         'cancellation: a job\'s remaining ops are removed')
    ap.add_argument('--disruption_offset', type=float, default=0.4,
                    help='fraction of N ops greedily committed before the '
                         'disruption strikes (the residual is re-solved)')
    ap.add_argument('--lag_factor', type=float, default=2.0,
                    help='multiply the chosen op lag by this (2.0 == +100%%)')
    ap.add_argument('--pt_factor', type=float, default=1.5,
                    help='pt_deviation: in-flight op takes this x its pt '
                         '(rounded up to whole hours)')
    ap.add_argument('--shortage_factor', type=float, default=0.5,
                    help='material_shortage: delay = factor x the job\'s mean '
                         'remaining processing time')
    ap.add_argument('--shortage_basis', type=str, default='op_mean',
                    choices=['op_mean', 'remaining_work'],
                    help='op_mean: mean over remaining ops of the mean '
                         'compatible pt; remaining_work: their sum')
    ap.add_argument('--methods', type=str,
                    default=','.join(RESIDUAL_METHODS),
                    help='comma-separated residual methods to run (CP-SAT '
                         'arms come from --cpsat_budgets)')
    ap.add_argument('--ga_budget', type=float, default=5.0,
                    help='ga_reactive wall-clock budget per instance (s)')
    ap.add_argument('--ga_pop', type=int, default=100,
                    help='ga_reactive population size')
    ap.add_argument('--ga_seed', type=int, default=0,
                    help='ga_reactive RNG seed (per instance: seed*100003+i)')
    ap.add_argument('--cpsat_budgets', type=str, default='5,30,300',
                    help='comma-separated CP-SAT per-solve budgets in seconds')
    ap.add_argument('--cpsat_workers', type=int, default=0,
                    help='CP-SAT num_workers for the whole re-solve '
                         '(0 = solver default)')
    ap.add_argument('--device', type=str, default=None,
                    help='torch device override (e.g. cpu); default: params')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap the number of instances (smoke-test convenience)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='output directory (default test_results/PPVC/'
                         '<dataset>)')
    args = ap.parse_args()
    if getattr(args, 'cpsat_workers', 1) == 0:
        # 0 would start one CP-SAT worker per machine core whatever the affinity mask;
        # use one worker per pinned core so the solve matches its CPU allocation
        args.cpsat_workers = len(os.sched_getaffinity(0))
    sys.argv = [sys.argv[0]]  # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli() if __name__ == '__main__' else None

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
    'framework',
)

# A start shift beyond this many TRUE time units counts as time-nervousness.
NERVOUSNESS_TIME_THRESHOLD = 1.0  # 1 hour (instances use hours)

_TOL = 1e-6


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
    # snapshots written before --framework existed are DANIEL
    configs.framework = snap.get('framework', 'DANIEL')
    from common_utils import check_backbone_tag
    check_backbone_tag(model_name, configs.framework)
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


def instance_context(stem, meta):
    """Instance seed, dataset class mix and route variant for one instance.

    The instance seed is dataset_meta.json seed0 + the index in the file name,
    exactly as make_scale_testsets.py / Trainer.make_ppvc_dataset generate
    the sets. Used only by urgent_arrival to draw the new module.
    """
    ds_meta_path = os.path.join(os.path.dirname(stem), 'dataset_meta.json')
    ds_meta = {}
    if os.path.exists(ds_meta_path):
        with open(ds_meta_path) as f:
            ds_meta = json.load(f)
    m = re.search(r'instance_(\d+)$', os.path.basename(stem))
    idx = int(m.group(1)) if m else 0
    seed0 = int(ds_meta.get('seed0', 0))
    return {
        'instance_seed': seed0 + idx,
        'ppvc_mix': ds_meta.get('ppvc_mix', 'mixed'),
        'route_variant': meta.get('route_variant', 'fixed'),
    }


# ---------------------------------------------------------------------------
# Env construction (mirrors eval_ppvc._new_env exactly) + greedy primitives.
# ---------------------------------------------------------------------------

def _new_env(jl, pt, meta, use_lag_features, use_type_embedding):
    """Fresh batch-of-1 env, built with the SAME kwargs eval_ppvc._new_env uses."""
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


def drl_policy(ppo):
    """Policy callable (env, state) -> int action for continue_residual."""
    def _pol(env, state):
        return int(_greedy_action(ppo, state).cpu().numpy()[0])
    return _pol


def rule_policy(rule):
    """Priority-rule callable (env, state) -> int action (lag-aware rules)."""
    from common_utils import heuristic_select_action

    def _pol(env, state):
        return int(heuristic_select_action(rule, env))
    return _pol


def rollout_greedy_full(ppo, jl, pt, meta, use_lag_features, use_type_embedding):
    """Full greedy rollout, NO disruption (the ORIGINAL reference plan).

    Returns (makespan, seconds, assigned_mch[N], true_op_ct[N]).
    Identical mechanics to eval_ppvc.rollout_greedy.
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


def continue_residual(env, state, assigned_mch, policy):
    """Step `policy` on the ALREADY-disrupted env until every op is scheduled.

    Times only the residual solve. Returns (makespan, seconds, assigned_mch,
    true_op_ct); assigned_mch is filled in place (prefix entries preset).
    """
    M = env.number_of_machines
    t1 = time.time()
    while not env.done().all():
        a = policy(env, state)
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, done = env.step(actions=np.array([a]))
        if done.all():
            break
    t2 = time.time()
    assert (assigned_mch >= 0).all(), 'continue_residual: op never scheduled'
    ct = env.true_op_ct[0].copy()
    return float(np.max(ct)), t2 - t1, assigned_mch, ct


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


def _residual_work_per_mch(env):
    """#unscheduled ops each machine can process (residual workload proxy)."""
    remain = env.remain_process_relation[0]
    unscheduled = ~env.op_scheduled_flag[0].astype(bool)
    if not unscheduled.any():
        return np.zeros(env.number_of_machines, dtype=int)
    return (remain[unscheduled] > 0).sum(axis=0)


def _pick_breakdown_machine(env):
    """Pick a machine that still has UNSCHEDULED work in the residual.

    Choose the busiest such machine (most remaining compatible ops) so the
    breakdown actually perturbs the residual. Returns the machine index, or
    None if every machine is idle for the rest of the horizon.
    """
    work_per_mch = _residual_work_per_mch(env)
    if work_per_mch.max() <= 0:
        return None
    return int(np.argmax(work_per_mch))


def _job_remaining_work(env, pt):
    """Per job: (#unscheduled ops, sum and mean of their mean compatible pt)."""
    pt = np.asarray(pt, dtype=float)
    comp = pt > 0
    op_mean_pt = pt.sum(axis=1) / np.maximum(comp.sum(axis=1), 1)
    unscheduled = ~env.op_scheduled_flag[0].astype(bool)
    J = env.number_of_jobs
    n_rem = np.zeros(J, dtype=int)
    work = np.zeros(J, dtype=float)
    for j in range(J):
        s = int(env.job_first_op_id[0, j])
        e = int(env.job_last_op_id[0, j]) + 1
        rem = unscheduled[s:e]
        n_rem[j] = int(rem.sum())
        work[j] = float(op_mean_pt[s:e][rem].sum())
    mean = np.where(n_rem > 0, work / np.maximum(n_rem, 1), 0.0)
    return n_rem, work, mean


def _pick_critical_job(env, pt):
    """Unfinished job with the most remaining mean-pt work (ties: lowest id)."""
    n_rem, work, _ = _job_remaining_work(env, pt)
    if not (n_rem > 0).any():
        return None
    w = np.where(n_rem > 0, work, -1.0)
    return int(np.argmax(w))


def _pick_inflight_op(env, assigned_mch, t_disrupt_true):
    """In-flight committed op (start <= t_d < completion) on the busiest machine.

    Returns (op, machine, in_flight). Among machines running an op at t_d,
    choose the one with the most residual compatible work (the breakdown rule);
    ties: longer processing time, then lower op index. Falls back to the
    committed op that finishes last (in_flight False) when no op is running.
    """
    committed = env.op_scheduled_flag[0].astype(bool)
    ct = env.true_op_ct[0]
    tpt = env.true_op_pt[0]
    work = _residual_work_per_mch(env)
    best = None
    for op in np.where(committed)[0]:
        op = int(op)
        m = int(assigned_mch[op])
        start = float(ct[op]) - float(tpt[op, m])
        if start <= t_disrupt_true + _TOL and float(ct[op]) > t_disrupt_true + _TOL:
            key = (int(work[m]), float(tpt[op, m]), -op)
            if best is None or key > best[0]:
                best = (key, op, m)
    if best is not None:
        return best[1], best[2], True
    ops = np.where(committed)[0]
    op = int(ops[np.argmax(ct[ops])])
    return op, int(assigned_mch[op]), False


# ---------------------------------------------------------------------------
# Env refresh after an edit.
# ---------------------------------------------------------------------------

def _legacy_refresh(env):
    """Feature refresh used by lag_perturb / breakdown (unchanged).

    Recompute pair_free_time / next_schedule_time from the edited
    candidate_free_time + mch_free_time (fjsp_env_same_op_nums step logic),
    refresh the remaining-lag channel, rebuild the three feature blocks and
    push them into the state. The dynamic pair mask is NOT recomputed here;
    kept as is so the legacy numbers stay bit-identical.
    """
    import numpy.ma as ma
    candidateFT = np.expand_dims(env.candidate_free_time, axis=2)
    mchFT = np.expand_dims(env.mch_free_time, axis=1)
    env.pair_free_time = np.maximum(candidateFT, mchFT)
    schedule_matrix = ma.array(env.pair_free_time,
                               mask=env.candidate_process_relation)
    env.next_schedule_time = np.min(
        schedule_matrix.reshape(env.number_of_envs, -1), axis=1).data
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
    return env.state


def _full_refresh(env):
    """Refresh every decision quantity env.step() refreshes (new types).

    Mirrors fjsp_env_same_op_nums.step() after the chosen-op updates: pair free
    times, next decision instant, deleted nodes + op mask, waiting / remaining
    work, remaining lag, op features, dynamic pair mask, competition index,
    machine mask, machine waiting / working flags, machine and pair features.
    """
    import numpy.ma as ma
    candidateFT = np.expand_dims(env.candidate_free_time, axis=2)
    mchFT = np.expand_dims(env.mch_free_time, axis=1)
    env.pair_free_time = np.maximum(candidateFT, mchFT)
    schedule_matrix = ma.array(env.pair_free_time,
                               mask=env.candidate_process_relation)
    env.next_schedule_time = np.min(
        schedule_matrix.reshape(env.number_of_envs, -1), axis=1).data
    nst = env.next_schedule_time

    env.deleted_op_nodes = np.logical_and(env.op_ct <= nst[:, np.newaxis],
                                          env.op_scheduled_flag)
    env.delete_mask_fea_j = np.tile(env.deleted_op_nodes[:, :, np.newaxis],
                                    (1, 1, env.op_fea_dim))
    env.update_op_mask()

    env.op_waiting_time = np.zeros((env.number_of_envs, env.number_of_ops))
    env.op_waiting_time[env.env_job_idx, env.candidate] = \
        (1 - env.mask) * np.maximum(np.expand_dims(nst, axis=1)
                                    - env.candidate_free_time, 0)
    env.op_remain_work = np.maximum(env.op_ct - np.expand_dims(nst, axis=1), 0)
    if env.use_lag_features:
        env.op_remain_lag = env.op_scheduled_flag * np.clip(
            env.op_ct + env.op_lag - nst[:, np.newaxis], 0, env.op_lag)
    env.construct_op_features()

    env.unavailable_pairs = env.pair_free_time > nst[:, np.newaxis, np.newaxis]
    env.dynamic_pair_mask = np.logical_or(
        np.copy(env.candidate_process_relation), env.unavailable_pairs)
    env.comp_idx = env.logic_operator(x=~env.dynamic_pair_mask)
    env.update_mch_mask()
    env.mch_current_available_jc_nums = np.sum(~env.dynamic_pair_mask, axis=1)
    mch_free_duration = np.expand_dims(nst, axis=1) - env.mch_free_time
    mch_free_flag = mch_free_duration < 0
    env.mch_working_flag = mch_free_flag + 0
    env.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration
    env.mch_remain_work = np.maximum(-mch_free_duration, 0)
    env.construct_mch_features()
    env.construct_pair_features()
    env.state.update(env.fea_j, env.op_mask, env.fea_m, env.mch_mask,
                     env.dynamic_pair_mask, env.comp_idx, env.candidate,
                     env.fea_pairs, op_type=env.op_type, mch_type=env.mch_type)
    return env.state


def _shift_job_lb(env, j, first_op, release_norm):
    """Shift job j's completion-time lower bounds so first_op starts >= release."""
    last = int(env.job_last_op_id[0, j])
    lb_start = float(env.op_ct_lb[0, first_op] - env.op_min_pt[0, first_op])
    diff = release_norm - lb_start
    if diff > 0:
        env.op_ct_lb[0, first_op:last + 1] += diff


# ---------------------------------------------------------------------------
# Disrupted-state construction (shared by every method and by
# eval_residual_cpsat.py).
# ---------------------------------------------------------------------------

def _commit_prefix(ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                   offset):
    """Greedy-commit the first k = ceil(offset*N) ops (pre-disruption)."""
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state = _new_env(jl, pt, meta, use_lag_features, use_type_embedding)
    assigned_mch = np.full(n_ops, -1, dtype=int)
    k = int(math.ceil(offset * n_ops))
    k = max(0, min(k, n_ops - 1))  # leave >=1 op for the residual
    actions = []
    for _ in range(k):
        action = _greedy_action(ppo, state)
        a = int(action.cpu().numpy()[0])
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        actions.append(a)
        state, _, _ = env.step(actions=action.cpu().numpy())
    return env, state, assigned_mch, actions, k


def _replay_env(jl2, pt2, meta2, use_lag_features, use_type_embedding,
                actions, job_map, M, op_map, ref_ct):
    """Build an env on the realized instance and replay the committed actions.

    Job indices are remapped through job_map. The replayed completion times
    must equal the realized ones exactly (asserted), which holds because the
    replay dispatches the same ops on the same machines in the same order.
    """
    env, state = _new_env(jl2, pt2, meta2, use_lag_features,
                          use_type_embedding)
    for a in actions:
        j_old, m = a // M, a % M
        j_new = int(job_map[j_old])
        assert j_new >= 0, 'replay: a committed action belongs to a removed job'
        state, _, _ = env.step(actions=np.array([j_new * M + m]))
    for op_old in np.where(np.isfinite(ref_ct))[0]:
        op_new = int(op_map[op_old])
        assert op_new >= 0
        assert abs(float(env.true_op_ct[0, op_new]) - float(ref_ct[op_old])) \
            < _TOL, (f'replay mismatch at op {op_old}: '
                     f'{env.true_op_ct[0, op_new]} != {ref_ct[op_old]}')
    return env, state


def draw_urgent_module(meta, ictx):
    """Draw the urgent module with the generator (class from the dataset mix).

    Returns (job_length [1], op_pt [22, M], meta of the one-module instance).
    """
    from ppvc_instance_generator import ppvc_instance_generator, PRESET_MIXES
    mix = PRESET_MIXES.get(ictx['ppvc_mix'], PRESET_MIXES['mixed'])
    jl_n, pt_n, meta_n = ppvc_instance_generator(
        n_modules=1, class_mix=tuple(float(x) for x in mix),
        station_counts=meta['station_counts'],
        seed=int(ictx['instance_seed']) + URGENT_SEED_OFFSET,
        route_variant=ictx['route_variant'])
    assert np.array_equal(np.asarray(meta_n['mch_type']),
                          np.asarray(meta['mch_type'])), \
        'urgent module drawn for a different factory layout'
    return jl_n, pt_n, meta_n


def build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                          use_type_embedding, disruption_type, offset,
                          dparams, ictx=None):
    """Greedy-commit the first k ops, then apply the disruption.

    dparams: dict with lag_factor, pt_factor, shortage_factor, shortage_basis.
    ictx:    instance_context(stem, meta) (needed for urgent_arrival).

    Returns a dict ds. Instance-level fields are in the REALIZED instance's op
    indexing (identical to the original except for urgent_arrival, which
    appends ops, and cancellation, which removes ops):
      env, state        disrupted env ready for residual continuation
      jl, pt, time_lag  realized instance (perturbed lag / longer pt / ...)
      meta              realized meta (time_lag, op_type, mch_type)
      op_map            int[N_orig]: realized index of each original op, -1 if
                        removed
      new_ops           realized indices of added ops (urgent_arrival)
      removed_ops       original indices of removed ops (cancellation)
      committed, committed_mch, committed_start, committed_ct  prefix (TRUE)
      assigned_mch      int[N_new]: prefix machines, -1 elsewhere
      t_disrupt_true, t_disrupt_norm
      mch_ready_time    legacy right-shift argument (None / [M])
      mch_release       float[M]: floor for uncommitted ops on each machine
      op_release        float[N_new]: floor for each uncommitted op
      broken_mch, downtime_true   (breakdown)
      perturbed_time_lag          alias of time_lag (legacy key)
      k, info, legacy
    """
    dparams = dict(dparams or {})
    lag_factor = float(dparams.get('lag_factor', 2.0))
    pt_factor = float(dparams.get('pt_factor', 1.5))
    shortage_factor = float(dparams.get('shortage_factor', 0.5))
    shortage_basis = dparams.get('shortage_basis', 'op_mean')

    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state, assigned_mch, actions, k = _commit_prefix(
        ppo, jl, pt, meta, use_lag_features, use_type_embedding, offset)

    # disruption time = the env's current decision instant (NORMALIZED units)
    t_disrupt_norm = float(env.next_schedule_time[0])
    # normalization slope used by the env for pt AND lag. pt_lower_bound
    # includes the 0s of incompatible entries, so this is the EXACT slope the
    # env applied.
    slope = float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)
    t_disrupt_true = t_disrupt_norm * slope

    # ---- committed prefix at the disruption instant (original indexing) ----
    committed0 = env.op_scheduled_flag[0].astype(bool).copy()
    committed_ct0 = np.full(n_ops, np.nan, dtype=float)
    committed_start0 = np.full(n_ops, np.nan, dtype=float)
    true_op_pt = env.true_op_pt[0]
    for op in np.where(committed0)[0]:
        m = int(assigned_mch[op])
        assert m >= 0, f'committed op {op} has no recorded machine'
        committed_ct0[op] = float(env.true_op_ct[0, op])
        committed_start0[op] = committed_ct0[op] - float(true_op_pt[op, m])

    base_lag = np.asarray(meta['time_lag'], dtype=float)
    jl_r = np.asarray(jl).copy()
    pt_r = np.asarray(pt).copy()
    lag_r = base_lag.copy()
    meta_r = {'time_lag': lag_r, 'op_type': meta.get('op_type'),
              'mch_type': meta.get('mch_type')}
    op_map = np.arange(n_ops)
    new_ops, removed_ops = [], []
    mch_ready_time = None
    mch_release = np.zeros(M, dtype=float)
    broken_mch = None
    downtime_true = None
    release_ops = {}   # realized op -> release floor (TRUE)
    info = ''
    legacy = disruption_type in LEGACY_DISRUPTIONS
    pt_dev = None      # (op, m, new_ct) for pt_deviation

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
        lag_r[op] = new_true
        # if the lagged op is already SCHEDULED, its successor's readiness is
        # tracked by candidate_free_time; recompute that job's entry. (For
        # unscheduled ops the lag is picked up at their own step via op_lag.)
        j = _job_of_op(env, op)
        if bool(env.op_scheduled_flag[0, op]):
            env.candidate_free_time[0, j] = (env.op_ct[0, op]
                                             + env.op_lag[0, op])
            env.true_candidate_free_time[0, j] = (env.true_op_ct[0, op]
                                                  + env.true_op_lag[0, op])
        state = _legacy_refresh(env)

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
        broken_mch = m
        # push the machine's free time up to its release instant (NORMALIZED).
        env.mch_free_time[0, m] = max(float(env.mch_free_time[0, m]),
                                      t_disrupt_norm + downtime_norm)
        env.true_mch_free_time[0, m] = max(float(env.true_mch_free_time[0, m]),
                                           release_true)
        # right-shift baseline honours the breakdown via a per-machine release
        # floor; only THIS machine is released late, others at 0.
        mch_ready_time = np.zeros(M, dtype=float)
        mch_ready_time[m] = release_true
        mch_release[m] = release_true
        state = _legacy_refresh(env)

    elif disruption_type == 'material_shortage':
        j = _pick_critical_job(env, pt)
        assert j is not None, 'material_shortage: no unfinished job'
        n_rem, work, mean = _job_remaining_work(env, pt)
        basis = float(mean[j] if shortage_basis == 'op_mean' else work[j])
        delay_true = shortage_factor * basis
        release_true = t_disrupt_true + delay_true
        op = int(env.candidate[0, j])
        assert not env.op_scheduled_flag[0, op]
        env.true_candidate_free_time[0, j] = max(
            float(env.true_candidate_free_time[0, j]), release_true)
        env.candidate_free_time[0, j] = max(
            float(env.candidate_free_time[0, j]), release_true / slope)
        _shift_job_lb(env, j, op, release_true / slope)
        release_ops[op] = release_true
        info = (f'material_shortage(job={j}, op={op}, delay {delay_true:.2f}h '
                f'[{shortage_basis}], release t={release_true:.2f}h)')
        state = _full_refresh(env)

    elif disruption_type == 'pt_deviation':
        op, m, in_flight = _pick_inflight_op(env, assigned_mch, t_disrupt_true)
        old_pt = int(pt[op, m])
        new_pt = int(math.ceil(pt_factor * old_pt - 1e-9))
        delta = float(new_pt - old_pt)
        j = _job_of_op(env, op)
        new_ct = float(env.true_op_ct[0, op]) + delta
        env.true_op_ct[0, op] = new_ct
        env.op_ct[0, op] = env.op_ct[0, op] + delta / slope
        env.true_mch_free_time[0, m] = max(float(env.true_mch_free_time[0, m]),
                                           new_ct)
        env.mch_free_time[0, m] = max(float(env.mch_free_time[0, m]),
                                      float(env.op_ct[0, op]))
        succ_pending = (op < int(env.job_last_op_id[0, j])
                        and not env.op_scheduled_flag[0, op + 1])
        if succ_pending:
            env.true_candidate_free_time[0, j] = new_ct + float(
                env.true_op_lag[0, op])
            env.candidate_free_time[0, j] = float(env.op_ct[0, op]) + float(
                env.op_lag[0, op])
        env.current_makespan = np.maximum(env.current_makespan, new_ct)
        env.op_ct_lb[0, op:int(env.job_last_op_id[0, j]) + 1] += delta / slope
        pt_r[op, m] = new_pt
        committed_ct0[op] = new_ct
        pt_dev = (op, m, new_ct)
        tag = '' if in_flight else ', no op in flight: latest-finishing ' \
                                   'committed op extended'
        info = (f'pt_deviation(op={op}, m={m}, pt {old_pt}->{new_pt}h, '
                f'ct {new_ct - delta:.1f}->{new_ct:.1f}h{tag})')
        state = _full_refresh(env)

    elif disruption_type == 'urgent_arrival':
        assert ictx is not None, 'urgent_arrival needs instance_context()'
        jl_n, pt_n, meta_n = draw_urgent_module(meta, ictx)
        n_new = int(pt_n.shape[0])
        jl_r = np.concatenate([np.asarray(jl), np.asarray(jl_n)]).astype(int)
        pt_r = np.vstack([np.asarray(pt), np.asarray(pt_n)]).astype(int)
        lag_r = np.concatenate([base_lag,
                                np.asarray(meta_n['time_lag'], dtype=float)])
        op_type_r = None
        if meta.get('op_type') is not None:
            op_type_r = np.concatenate([np.asarray(meta['op_type']),
                                        np.asarray(meta_n['op_type'])])
        meta_r = {'time_lag': lag_r, 'op_type': op_type_r,
                  'mch_type': meta.get('mch_type')}
        J = len(jl)
        job_map = np.arange(J)
        env, state = _replay_env(jl_r, pt_r, meta_r, use_lag_features,
                                 use_type_embedding, actions, job_map, M,
                                 op_map, committed_ct0)
        slope2 = float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)
        jn = J
        first_new = int(env.job_first_op_id[0, jn])
        env.true_candidate_free_time[0, jn] = max(
            float(env.true_candidate_free_time[0, jn]), t_disrupt_true)
        env.candidate_free_time[0, jn] = max(
            float(env.candidate_free_time[0, jn]), t_disrupt_true / slope2)
        _shift_job_lb(env, jn, first_new, t_disrupt_true / slope2)
        new_ops = list(range(n_ops, n_ops + n_new))
        release_ops[first_new] = t_disrupt_true
        from ppvc_instance_generator import ROUTING_CLASSES
        cls = ROUTING_CLASSES[int(meta_n['routing_class'][0])]
        info = (f'urgent_arrival(job={jn}, class={cls}, {n_new} ops, '
                f'released t={t_disrupt_true:.2f}h)')
        state = _full_refresh(env)

    elif disruption_type == 'cancellation':
        j = _pick_critical_job(env, pt)
        assert j is not None, 'cancellation: no unfinished job'
        s = int(env.job_first_op_id[0, j])
        e = int(env.job_last_op_id[0, j]) + 1
        keep = np.ones(n_ops, dtype=bool)
        for op in range(s, e):
            if not committed0[op]:
                keep[op] = False
                removed_ops.append(op)
        n_kept_j = int(keep[s:e].sum())
        op_map = np.where(keep, np.cumsum(keep) - 1, -1).astype(int)
        J = len(jl)
        if n_kept_j > 0:
            jl_r = np.asarray(jl).copy()
            jl_r[j] = n_kept_j
            job_map = np.arange(J)
        else:
            jl_r = np.delete(np.asarray(jl), j)
            job_map = np.array([x if x < j else (x - 1 if x > j else -1)
                                for x in range(J)])
        pt_r = np.asarray(pt)[keep].copy()
        lag_r = base_lag[keep].copy()
        op_type_r = None
        if meta.get('op_type') is not None:
            op_type_r = np.asarray(meta['op_type'])[keep]
        meta_r = {'time_lag': lag_r, 'op_type': op_type_r,
                  'mch_type': meta.get('mch_type')}
        assert (jl_r > 0).all()
        env, state = _replay_env(jl_r, pt_r, meta_r, use_lag_features,
                                 use_type_embedding, actions, job_map, M,
                                 op_map, committed_ct0)
        info = (f'cancellation(job={j}, {len(removed_ops)} remaining ops '
                f'removed, {n_kept_j} committed kept)')
        state = _full_refresh(env)
    else:
        raise ValueError(f'unknown disruption_type {disruption_type}')

    # ---- realized-index prefix arrays ---------------------------------------
    n_new = int(pt_r.shape[0])
    committed = np.zeros(n_new, dtype=bool)
    committed_mch = np.full(n_new, -1, dtype=int)
    committed_ct = np.full(n_new, np.nan, dtype=float)
    committed_start = np.full(n_new, np.nan, dtype=float)
    for op in np.where(committed0)[0]:
        r = int(op_map[op])
        committed[r] = True
        committed_mch[r] = int(assigned_mch[op])
        committed_ct[r] = committed_ct0[op]
        committed_start[r] = committed_start0[op]
    if pt_dev is not None:  # the realized start is unchanged, ct is longer
        committed_start[pt_dev[0]] = committed_start0[pt_dev[0]]
    op_release = np.zeros(n_new, dtype=float)
    for r, v in release_ops.items():
        op_release[r] = v
    assigned_new = committed_mch.copy()

    return {
        'env': env, 'state': state,
        'jl': np.asarray(jl_r, dtype=int), 'pt': pt_r, 'time_lag': lag_r,
        'meta': meta_r, 'op_map': op_map, 'new_ops': new_ops,
        'removed_ops': removed_ops,
        'committed': committed, 'committed_mch': committed_mch,
        'committed_start': committed_start, 'committed_ct': committed_ct,
        'assigned_mch': assigned_new,
        't_disrupt_true': t_disrupt_true, 't_disrupt_norm': t_disrupt_norm,
        'mch_ready_time': mch_ready_time, 'mch_release': mch_release,
        'op_release': op_release, 'broken_mch': broken_mch,
        'downtime_true': downtime_true, 'perturbed_time_lag': lag_r,
        'k': k, 'info': info, 'legacy': legacy,
        'disruption_type': disruption_type,
    }


def drl_reactive(ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                 disruption_type, offset, lag_factor, dparams=None, ictx=None):
    """Build a disrupted mid-execution state, then greedily re-solve the residual.

    Kept for callers of the earlier API. Returns a dict with keys:
      makespan, seconds (RESIDUAL solve only), assigned_mch, true_op_ct,
      t_disrupt_true, perturbed_time_lag, mch_ready_time, info, k.
    """
    dp = dict(dparams or {})
    dp['lag_factor'] = lag_factor
    ds = build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                               use_type_embedding, disruption_type, offset,
                               dp, ictx)
    ms, sec, asg, ct = continue_residual(ds['env'], ds['state'],
                                         ds['assigned_mch'].copy(),
                                         drl_policy(ppo))
    return {
        'makespan': ms, 'seconds': sec, 'assigned_mch': asg,
        'true_op_ct': ct, 't_disrupt_true': ds['t_disrupt_true'],
        'perturbed_time_lag': ds['time_lag'],
        'mch_ready_time': ds['mch_ready_time'], 'info': ds['info'],
        'k': ds['k'],
    }


# ---------------------------------------------------------------------------
# Residual problem: committed prefix fixed, uncommitted ops placed after it.
# Shared by the residual right-shift and the residual GA decoder.
# ---------------------------------------------------------------------------

class ResidualProblem:
    """Semi-active decoder for the residual after a disruption.

    Uncommitted ops are placed in a given order (each job's ops in route
    order) on given machines: start = max(t_d, job ready (ct + lag of the
    job's previous op), machine free (ct of the previous op on it), op
    release, machine release). Committed ops keep their realized times.
    """

    def __init__(self, ds):
        jl = np.asarray(ds['jl'], dtype=int)
        pt = np.asarray(ds['pt'], dtype=float)
        lag = np.asarray(ds['time_lag'], dtype=float)
        self.n_ops, self.n_mch = pt.shape
        self.jl = jl
        self.n_jobs = len(jl)
        self.pt = pt
        self.pt_l = pt.tolist()
        self.lag_l = lag.tolist()
        self.job_first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        self.op_job = np.repeat(np.arange(self.n_jobs), jl).astype(int)
        self.op_job_l = self.op_job.tolist()
        self.committed = np.asarray(ds['committed'], dtype=bool)
        self.committed_mch = np.asarray(ds['committed_mch'], dtype=int)
        self.ct0 = np.where(self.committed, ds['committed_ct'], 0.0)
        self.t_d = float(ds['t_disrupt_true'])
        self.op_release_l = np.asarray(ds['op_release'], dtype=float).tolist()
        self.mch_release_l = np.asarray(ds['mch_release'],
                                        dtype=float).tolist()
        tail = np.zeros(self.n_mch)
        for op in np.where(self.committed)[0]:
            m = self.committed_mch[op]
            tail[m] = max(tail[m], self.ct0[op])
        self.mch_tail0 = tail.tolist()
        ready = np.zeros(self.n_jobs)
        self.job_next0 = []
        for j in range(self.n_jobs):
            s = self.job_first[j]
            e = s + jl[j]
            nxt = None
            for op in range(s, e):
                if self.committed[op]:
                    ready[j] = self.ct0[op] + lag[op]
                elif nxt is None:
                    nxt = op
            self.job_next0.append(nxt if nxt is not None else e)
        self.job_ready0 = ready.tolist()
        self.uncommitted = [op for op in range(self.n_ops)
                            if not self.committed[op]]
        self.base_os = np.array([self.op_job[op] for op in self.uncommitted],
                                dtype=int)
        self.compat = []
        for op in range(self.n_ops):
            if self.committed[op]:
                self.compat.append(np.array([self.committed_mch[op]]))
            else:
                self.compat.append(np.where(pt[op] > 0)[0])

    def order_from_os(self, os_vec):
        """Job-id sequence -> uncommitted op sequence (route order per job)."""
        nxt = list(self.job_next0)
        order = []
        for j in os_vec:
            order.append(nxt[j])
            nxt[j] += 1
        return order

    def place(self, order, ma, ct=None, tail=None, ready=None):
        """Place `order` on machines `ma`; returns (ct list, tail, ready)."""
        ct = self.ct0.tolist() if ct is None else ct
        tail = list(self.mch_tail0) if tail is None else tail
        ready = list(self.job_ready0) if ready is None else ready
        pt_l, lag_l, oj = self.pt_l, self.lag_l, self.op_job_l
        orl, mrl, t_d = self.op_release_l, self.mch_release_l, self.t_d
        for op in order:
            m = ma[op]
            j = oj[op]
            s = t_d
            if ready[j] > s:
                s = ready[j]
            if tail[m] > s:
                s = tail[m]
            if orl[op] > s:
                s = orl[op]
            if mrl[m] > s:
                s = mrl[m]
            c = s + pt_l[op][m]
            ct[op] = c
            tail[m] = c
            ready[j] = c + lag_l[op]
        return ct, tail, ready

    def makespan_os(self, os_vec, ma):
        ct, _, _ = self.place(self.order_from_os(os_vec), ma)
        return max(ct)

    def os_from_schedule(self, ma, ct):
        """OS that decodes to the given residual schedule (sort by start)."""
        starts = [(ct[op] - self.pt_l[op][ma[op]], op)
                  for op in self.uncommitted]
        starts.sort()
        return np.array([self.op_job[op] for _, op in starts], dtype=int)


def right_shift_residual(ds, assigned_orig, oct_orig, pt_orig):
    """Prefix-respecting right-shift repair of the frozen original plan.

    Committed ops keep their realized times. The remaining original ops keep
    their machine and machine order (sorted by original start) and are re-timed
    to the earliest feasible start (not before t_d or any release floor). New
    ops (urgent_arrival) are appended, in route order, to the queue of the
    compatible machine that finishes each one earliest.

    Returns (makespan, seconds, assigned_mch[N_new], op_ct[N_new]).
    """
    t1 = time.time()
    prob = ResidualProblem(ds)
    op_map = ds['op_map']
    pt_orig = np.asarray(pt_orig, dtype=float)
    n0 = len(assigned_orig)
    ma = [int(x) for x in ds['committed_mch']]
    keyed = []
    for i in range(n0):
        r = int(op_map[i])
        if r < 0 or prob.committed[r]:
            continue
        m = int(assigned_orig[i])
        ma[r] = m
        keyed.append((float(oct_orig[i]) - float(pt_orig[i, m]), i, r))
    keyed.sort()
    ct, tail, ready = prob.place([r for _, _, r in keyed], ma)
    for r in ds['new_ops']:
        j = prob.op_job_l[r]
        best = None
        for m in prob.compat[r]:
            m = int(m)
            s = max(prob.t_d, ready[j], tail[m], prob.op_release_l[r],
                    prob.mch_release_l[m])
            c = s + prob.pt_l[r][m]
            if best is None or c < best[0] - 1e-12:
                best = (c, m)
        ma[r] = best[1]
        ct, tail, ready = prob.place([r], ma, ct, tail, ready)
    sec = time.time() - t1
    asg = np.array(ma, dtype=int)
    assert (asg >= 0).all(), 'right_shift_residual: unassigned op'
    ct = np.array(ct, dtype=float)
    return float(ct.max()), sec, asg, ct


def _import_eval_ga():
    """eval_ga parses the process argv at import; give it a clean argv."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import eval_ga
    finally:
        sys.argv = saved
    return eval_ga


def ga_reactive(ds, seed_asg, seed_ct, budget_s, pop_size=100, rng=None,
                elite=2, tour_k=3, timing=None):
    """eval_ga's GA on the residual problem, seeded with the right-shift plan.

    The chromosome is (operation sequence over the uncommitted ops, machine
    vector with committed entries fixed). Operators are eval_ga's own
    (tournament, pox_crossover, uniform_crossover_ma, mutate); the decoder is
    ResidualProblem.place, which keeps the committed prefix and every release
    floor. The seed (the residual right-shift plan) is asserted to decode to
    its own makespan, so with elitism the result is never worse than it.

    Returns (makespan, seconds, assigned_mch[N_new], op_ct[N_new], gens).
    When timing is a dict it receives the run's contention evidence
    (solve_timing: one worker thread, status 'budget').
    """
    ga = _import_eval_ga()
    cpu0, wall0 = time.process_time(), time.perf_counter()

    def _evidence():
        if timing is not None:
            timing.update(solve_timing(time.perf_counter() - wall0,
                                       time.process_time() - cpu0, 1,
                                       budget_s, 'budget'))

    t0 = time.time()
    rng = np.random.default_rng(0) if rng is None else rng
    prob = ResidualProblem(ds)
    compat = prob.compat
    n_jobs = prob.n_jobs
    seed_ma = np.asarray(seed_asg, dtype=int).copy()
    seed_os = prob.os_from_schedule(seed_ma.tolist(), list(seed_ct))
    seed_fit = prob.makespan_os(seed_os, seed_ma.tolist())
    assert abs(seed_fit - float(np.max(seed_ct))) < 1e-6, (
        f'GA seed does not decode to the right-shift plan: {seed_fit} != '
        f'{float(np.max(seed_ct))}')

    def _fit(ind):
        return prob.makespan_os(ind[0], ind[1].tolist())

    if len(prob.uncommitted) == 0:
        out = (seed_fit, time.time() - t0, seed_ma, np.asarray(seed_ct), 0)
        _evidence()
        return out

    pop = [(seed_os.copy(), seed_ma.copy())]
    # a quarter of the population: the seed with a few random moves
    for _ in range(max(0, pop_size // 4)):
        o, m = seed_os.copy(), seed_ma.copy()
        for _ in range(3):
            o, m = ga.mutate(rng, o, m, compat, p_os=1.0, p_ma=1.0)
        pop.append((o, m))
    while len(pop) < pop_size:
        pop.append(ga.random_individual(rng, prob.base_os, compat))
    pop_fit = np.array([_fit(ind) for ind in pop])
    best_idx = int(np.argmin(pop_fit))
    best = (pop[best_idx][0].copy(), pop[best_idx][1].copy())
    best_fit = float(pop_fit[best_idx])

    stagnate_limit, immigrant_frac, stagnation = 25, 0.4, 0
    gen = 0
    while time.time() - t0 < budget_s:
        gen += 1
        order = np.argsort(pop_fit)
        new_pop = [(pop[i][0].copy(), pop[i][1].copy()) for i in order[:elite]]
        while len(new_pop) < pop_size:
            i1 = ga.tournament(rng, pop_fit, tour_k)
            i2 = ga.tournament(rng, pop_fit, tour_k)
            (o1, m1), (o2, m2) = pop[i1], pop[i2]
            c_os = ga.pox_crossover(rng, o1, o2, n_jobs)
            c_ma = ga.uniform_crossover_ma(rng, m1, m2)
            c_os, c_ma = ga.mutate(rng, c_os, c_ma, compat)
            new_pop.append((c_os, c_ma))
        pop = new_pop
        pop_fit = np.array([_fit(ind) for ind in pop])
        cur = int(np.argmin(pop_fit))
        if pop_fit[cur] < best_fit - 1e-9:
            best_fit = float(pop_fit[cur])
            best = (pop[cur][0].copy(), pop[cur][1].copy())
            stagnation = 0
        else:
            stagnation += 1
        if stagnation == stagnate_limit:
            deadline = t0 + budget_s
            improved = _residual_ma_local_search(prob, best, best_fit, deadline)
            if improved is not None:
                best, best_fit = improved
                w = int(np.argmax(pop_fit))
                pop[w] = (best[0].copy(), best[1].copy())
                pop_fit[w] = best_fit
            n_imm = int(pop_size * immigrant_frac)
            for w in np.argsort(pop_fit)[-n_imm:]:
                pop[w] = ga.random_individual(rng, prob.base_os, compat)
                pop_fit[w] = _fit(pop[w])
            stagnation = 0
    best_os, best_ma = best
    ct, _, _ = prob.place(prob.order_from_os(best_os), best_ma.tolist())
    ct = np.array(ct, dtype=float)
    out = (float(ct.max()), time.time() - t0, best_ma.copy(), ct, gen)
    _evidence()
    return out


def _residual_ma_local_search(prob, ind, cur_fit, deadline):
    """First-improvement machine reassignment of uncommitted ops (time-capped)."""
    os_vec, ma_vec = ind[0].copy(), ind[1].copy()
    ma_l = ma_vec.tolist()
    f = cur_fit
    improved = False
    for op in prob.uncommitted:
        if time.time() > deadline:
            break
        choices = prob.compat[op]
        if len(choices) < 2:
            continue
        orig = ma_l[op]
        for m in choices:
            m = int(m)
            if m == orig:
                continue
            ma_l[op] = m
            ms = prob.makespan_os(os_vec, ma_l)
            if ms < f - 1e-9:
                f, orig, improved = ms, m, True
            else:
                ma_l[op] = orig
    if improved:
        return (os_vec, np.array(ma_l, dtype=int)), float(f)
    return None


# ---------------------------------------------------------------------------
# Right-shift baseline (legacy types): freeze the ORIGINAL plan's machine
# assignments, apply the disruption to the instance data, repair.
# ---------------------------------------------------------------------------

def right_shift_baseline(jl, pt, assigned_mch_orig, op_ct_orig,
                         perturbed_time_lag, mch_ready_time):
    """Repair the frozen original plan against the (perturbed) reality.

    Returns (makespan, seconds, assigned_mch[N], repaired_op_ct[N]).
    Machine ORDER is taken from the original plan's completion times
    (op_ct_orig). Machine assignments are NOT changed (industry can only
    push times right). mch_ready_time seeds the breakdown release floor on the
    first op of the machine's sequence; it is None for lag_perturb.
    """
    from right_shift_repair import right_shift_repair
    t1 = time.time()
    repaired_ct, ms = right_shift_repair(
        jl, pt, perturbed_time_lag, assigned_mch_orig,
        op_ct_lagblind=op_ct_orig, mch_ready_time=mch_ready_time)
    sec = time.time() - t1
    return float(ms), sec, assigned_mch_orig.copy(), repaired_ct


# Types whose right-shift arm still uses the legacy whole-plan operator. For
# lag_perturb it equals the residual version (the perturbed op's successor is
# pending, so no committed op moves). breakdown is NOT here: the legacy
# operator seeds the release floor on the first op of the broken machine's
# whole sequence, usually an op that already ran, and so moves executed ops.
LEGACY_RIGHT_SHIFT = ('lag_perturb',)


def right_shift_arm(ds, jl, pt, assigned_orig, oct_orig, rs_res):
    """The right-shift arm of the disruption study.

    Returns (makespan, seconds, assigned_mch, op_ct, contract) where contract
    tells whether the residual contract (committed ops fixed, floors) is
    checked for this arm. rs_res is right_shift_residual(ds, ...).
    """
    if ds['disruption_type'] in LEGACY_RIGHT_SHIFT:
        ms, sec, asg, ct = right_shift_baseline(
            jl, pt, assigned_orig, oct_orig, ds['time_lag'],
            ds['mch_ready_time'])
        return ms, sec, asg, ct, False
    ms, sec, asg, ct = rs_res
    return ms, sec, asg, ct, True


# ---------------------------------------------------------------------------
# CP-SAT budget sweep on the perturbed WHOLE instance (no residual surgery).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def cpsat_worker_cap(n_workers):
    """Cap CP-SAT num_workers for solvers built inside the block (0 = default).

    ortools_solver.fjsp_solver builds its own CpSolver and takes no worker
    argument; this swaps cp_model.CpSolver for a subclass that sets
    parameters.num_workers, and restores it afterwards.
    """
    if not n_workers:
        yield
        return
    from ortools.sat.python import cp_model
    orig = cp_model.CpSolver

    class _CappedCpSolver(orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.parameters.num_workers = int(n_workers)

    cp_model.CpSolver = _CappedCpSolver
    try:
        yield
    finally:
        cp_model.CpSolver = orig


@contextlib.contextmanager
def cpsat_solve_probe(sink):
    """Append the contention evidence (cpsat_timing plus loadavg_start) of
    every CP-SAT solve made inside the block to the list sink.

    ortools_solver.fjsp_solver builds its own CpSolver, so this swaps
    cp_model.CpSolver for a subclass of the current class (which may be the
    worker-capped one of cpsat_worker_cap) whose solve only adds the clock
    around the parent call; the model, parameters and search are unchanged.
    The original class is restored afterwards.
    """
    from ortools.sat.python import cp_model
    orig = cp_model.CpSolver

    class _TimedCpSolver(orig):
        _in_timed_solve = False

        def _timed(self, fn, model, callback):
            if self._in_timed_solve:   # Solve() delegates to solve()
                return fn(model, callback)
            self._in_timed_solve = True
            try:
                la = loadavg_now()
                with SolveClock() as clk:
                    status = fn(model, callback)
                rec = cpsat_timing(self, status, clk)
                rec['loadavg_start'] = la
                sink.append(rec)
                return status
            finally:
                self._in_timed_solve = False

        def Solve(self, model, solution_callback=None):
            return self._timed(super().Solve, model, solution_callback)

        def solve(self, model, solution_callback=None):
            return self._timed(super().solve, model, solution_callback)

    cp_model.CpSolver = _TimedCpSolver
    try:
        yield sink
    finally:
        cp_model.CpSolver = orig


def cpsat_sweep(jl, pt, perturbed_time_lag, budgets, mch_ready_time,
                n_workers=0, timing=None):
    """Solve the perturbed whole instance at each budget. No solver changes.

    The whole re-solve starts from t=0 and does not fix the committed prefix.
    It encodes a disruption only through the instance data it is given
    (time_lag, pt, the op set): lag_perturb, pt_deviation, urgent_arrival and
    cancellation change that data; breakdown (outage window) and
    material_shortage (release time) cannot be represented without solver
    edits, so those rows re-solve the instance WITHOUT the disruption and are
    lower-bound references only.

    Returns {budget: (makespan, solve_seconds, status, assigned_mch, op_ct)}.
    When timing is a dict, timing[budget] receives that solve's contention
    evidence (cpsat_solve_probe), or None if no solve was observed.
    """
    from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
    jobs, num_machines = matrix_to_the_format_for_solving(jl, pt)
    out = {}
    with cpsat_worker_cap(n_workers):
        for b in budgets:
            sink = []
            with (cpsat_solve_probe(sink) if timing is not None
                  else contextlib.nullcontext()):
                obj, t, status, asg, oct_ = fjsp_solver(
                    jobs, num_machines, time_limits=float(b),
                    time_lag=perturbed_time_lag, return_schedule=True)
            if timing is not None:
                timing[b] = sink[-1] if sink else None
            out[b] = (float(obj), float(t), status,
                      np.asarray(asg, dtype=int), np.asarray(oct_, dtype=float))
    return out


# ---------------------------------------------------------------------------
# Nervousness vs the original plan + residual-contract checks.
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


def nervousness_mapped(pt_orig, assigned_orig, op_ct_orig, pt_new,
                       assigned_new, op_ct_new, op_map):
    """Nervousness over the ops present in both plans (op_map[i] >= 0).

    Returns (mch_nerv, time_nerv, comb_nerv, mean_abs_start_shift_h). With an
    identity op_map and pt_new == pt_orig the first three equal nervousness().
    """
    from schedule_validator import reconstruct_starts
    s_orig = reconstruct_starts(pt_orig, assigned_orig, op_ct_orig)
    s_new = reconstruct_starts(pt_new, assigned_new, op_ct_new)
    op_map = np.asarray(op_map, dtype=int)
    keep = op_map >= 0
    if not keep.any():
        return 0, 0, 0, 0.0
    so, sn = s_orig[keep], s_new[op_map[keep]]
    mo = np.asarray(assigned_orig)[keep]
    mn = np.asarray(assigned_new)[op_map[keep]]
    mch_changed = int(np.sum(mo != mn))
    shift = np.abs(sn - so)
    time_shifted = int(np.sum(shift > NERVOUSNESS_TIME_THRESHOLD))
    return mch_changed, time_shifted, mch_changed + time_shifted, \
        float(shift.mean())


def check_residual_contract(ds, assigned, op_ct, tol=_TOL):
    """Violations of the residual contract (empty list == respected).

    Committed ops keep machine and completion time; every uncommitted op
    starts at or after t_d, its op release floor and its machine's release.
    """
    from schedule_validator import reconstruct_starts
    viol = []
    pt = np.asarray(ds['pt'], dtype=float)
    starts = reconstruct_starts(pt, assigned, op_ct)
    t_d = float(ds['t_disrupt_true'])
    for op in range(len(assigned)):
        if ds['committed'][op]:
            if int(assigned[op]) != int(ds['committed_mch'][op]):
                viol.append(f'prefix op {op}: machine {assigned[op]} != '
                            f'{ds["committed_mch"][op]}')
            if abs(float(op_ct[op]) - float(ds['committed_ct'][op])) > tol:
                viol.append(f'prefix op {op}: ct {op_ct[op]:.3f} != '
                            f'{ds["committed_ct"][op]:.3f}')
            continue
        floor = max(t_d, float(ds['op_release'][op]),
                    float(ds['mch_release'][int(assigned[op])]))
        if starts[op] < floor - tol:
            viol.append(f'op {op} starts {starts[op]:.3f} before its floor '
                        f'{floor:.3f}')
    return viol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _new_record(n):
    return {'makespan': np.full(n, np.nan),
            'inflation': np.full(n, np.nan),
            'seconds': np.full(n, np.nan),
            'mch_nerv': np.full(n, np.nan),
            'time_nerv': np.full(n, np.nan),
            'comb_nerv': np.full(n, np.nan),
            'abs_shift': np.full(n, np.nan),
            'feasible': np.zeros(n, dtype=bool)}


def main():
    args = _ARGS
    budgets = [int(b) for b in args.cpsat_budgets.split(',') if b.strip()]
    run_methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for m in run_methods:
        if m not in RESIDUAL_METHODS:
            sys.exit(f'[eval_reactive] unknown method {m}')
    dparams = {'lag_factor': args.lag_factor, 'pt_factor': args.pt_factor,
               'shortage_factor': args.shortage_factor,
               'shortage_basis': args.shortage_basis}
    dtype = args.disruption_type

    # 1) apply the training-config architecture flags BEFORE building anything
    snap = load_train_config(args.model_name)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    # torch + device setup (mirrors eval_ppvc.py / eval_a0_repair.py)
    if args.device is not None:
        configs.device = args.device
    if str(configs.device).startswith('cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from common_utils import setup_seed
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    # a batch-1 forward on a small graph is fastest single-threaded; greedy
    # argmax decisions do not depend on this knob
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
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
    methods = [m for m in RESIDUAL_METHODS if m in run_methods] \
        + cpsat_method_names

    rec = {m: _new_record(n) for m in methods}
    # contention evidence per (instance, method); the time-limited arms (GA
    # and CP-SAT) also keep their single solve's wall/CPU seconds, workers,
    # cpu_share, status and hit_limit
    timed_methods = [m for m in methods
                     if m == 'ga_reactive' or m in cpsat_method_names]
    for m in methods:
        rec[m].update(contention_arrays(n, per_solve=m in timed_methods))
    orig_makespans = np.full(n, np.nan)
    n_ops_realized = np.zeros(n, dtype=int)
    infos = []
    n_infeasible = 0
    n_contract = 0

    print('=' * 72)
    print(f'eval_reactive  model={args.model_name}  dataset={dataset}  '
          f'instances={n}')
    print(f'  disruption={dtype}  offset={args.disruption_offset}'
          f'  lag_factor={args.lag_factor}  pt_factor={args.pt_factor}'
          f'  shortage={args.shortage_factor}x{args.shortage_basis}')
    print(f'  methods={methods}  ga_budget={args.ga_budget}s  '
          f'cpsat_budgets={budgets}  cpsat_workers={args.cpsat_workers}')
    print(f'  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  seed_test={args.seed_test}'
          f'  device={configs.device}')
    print('=' * 72)

    def _record(m, i, ms, sec, nerv, feasible, ms_orig):
        rec[m]['makespan'][i] = ms
        rec[m]['inflation'][i] = (
            100.0 * (ms - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec[m]['seconds'][i] = sec
        if nerv is not None:
            mn, tn, cn, sh = nerv
            rec[m]['mch_nerv'][i] = mn
            rec[m]['time_nerv'][i] = tn
            rec[m]['comb_nerv'][i] = cn
            rec[m]['abs_shift'][i] = sh
        rec[m]['feasible'][i] = feasible

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        true_lag = np.asarray(meta['time_lag'], dtype=float)
        ictx = instance_context(stem, meta)

        # ---- ORIGINAL pre-disruption DRL plan (reference) -------------------
        ms_orig, _sec_orig, assigned_orig, oct_orig = rollout_greedy_full(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding)
        res0 = validate_schedule(jl, pt, true_lag, assigned_orig, oct_orig)
        assert res0['feasible'], (
            f'ORIGINAL plan infeasible on {base}:\n  '
            + '\n  '.join(res0['violations'][:10]))
        orig_makespans[i] = ms_orig

        # ---- the disrupted state (shared by every residual method) ----------
        ds = build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                                   use_type_embedding, dtype,
                                   args.disruption_offset, dparams, ictx)
        jl_r, pt_r, lag_r = ds['jl'], ds['pt'], ds['time_lag']
        n_ops_realized[i] = pt_r.shape[0]
        infos.append(ds['info'])
        env_spt = copy.deepcopy(ds['env']) if 'spt_reactive' in methods \
            else None

        def _check(name, asg, ct, ms, contract=True):
            nonlocal n_infeasible, n_contract
            res = validate_schedule(jl_r, pt_r, lag_r, asg, ct)
            ok = res['feasible']
            if not ok:
                n_infeasible += 1
                print(f'  !!! {name} INFEASIBLE on {base}: '
                      f'{res["violations"][:3]}')
            if contract:
                viol = check_residual_contract(ds, asg, ct)
                if viol:
                    n_contract += 1
                    ok = False
                    print(f'  !!! {name} RESIDUAL-CONTRACT VIOLATION on '
                          f'{base}: {viol[:3]}')
            return ok, res

        line = []

        # ---- (a) DRL-reactive ----------------------------------------------
        if 'drl_reactive' in methods:
            store_contention(rec['drl_reactive'], i, [], loadavg_now())
            ms_d, sec_d, asg_d, oct_d = continue_residual(
                ds['env'], ds['state'], ds['assigned_mch'].copy(),
                drl_policy(ppo))
            ok, res_d = _check('drl_reactive', asg_d, oct_d, ms_d)
            assert abs(res_d['makespan'] - ms_d) < 1e-6, (
                f'{base}: drl env makespan {ms_d} != validator '
                f'{res_d["makespan"]}')
            nerv = nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                      asg_d, oct_d, ds['op_map'])
            _record('drl_reactive', i, ms_d, sec_d, nerv, ok, ms_orig)
            line.append(f'drl={ms_d:6.1f}h({sec_d:.2f}s,nerv={nerv[2]})')

        # ---- (b) SPT-reactive (lag-aware SPT from the same state) -----------
        if 'spt_reactive' in methods:
            store_contention(rec['spt_reactive'], i, [], loadavg_now())
            np.random.seed(args.seed_test)
            ms_s, sec_s, asg_s, oct_s = continue_residual(
                env_spt, env_spt.state, ds['assigned_mch'].copy(),
                rule_policy('SPTL'))
            ok, res_s = _check('spt_reactive', asg_s, oct_s, ms_s)
            nerv = nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                      asg_s, oct_s, ds['op_map'])
            _record('spt_reactive', i, ms_s, sec_s, nerv, ok, ms_orig)
            line.append(f'spt={ms_s:6.1f}h')

        # ---- (d) right-shift repair ----------------------------------------
        rs_res = None
        if 'right_shift' in methods or 'ga_reactive' in methods:
            rs_res = right_shift_residual(ds, assigned_orig, oct_orig, pt)
        if 'right_shift' in methods:
            store_contention(rec['right_shift'], i, [], loadavg_now())
            ms_rs, sec_rs, asg_rs, oct_rs, rs_contract = right_shift_arm(
                ds, jl, pt, assigned_orig, oct_orig, rs_res)
            if not rs_contract:
                # the legacy operator re-times the whole plan (lag_perturb
                # only), so the residual contract is not applied to it
                ok, res_rs = _check('right_shift', asg_rs, oct_rs, ms_rs,
                                    contract=False)
                assert abs(res_rs['makespan'] - ms_rs) < 1e-6 or \
                    ds['mch_ready_time'] is not None, (
                    f'{base}: right_shift makespan {ms_rs} != validator '
                    f'{res_rs["makespan"]}')
            else:
                ok, _ = _check('right_shift', asg_rs, oct_rs, ms_rs)
            nerv = nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                      asg_rs, oct_rs, ds['op_map'])
            _record('right_shift', i, ms_rs, sec_rs, nerv, ok, ms_orig)
            line.append(f'rs={ms_rs:6.1f}h')

        # ---- (c) GA-reactive (seeded with the residual right-shift plan) ----
        if 'ga_reactive' in methods:
            rng = np.random.default_rng(args.ga_seed * 100003 + i)
            la_g, ga_t = loadavg_now(), {}
            ms_g, sec_g, asg_g, oct_g, gens = ga_reactive(
                ds, rs_res[2], rs_res[3], args.ga_budget,
                pop_size=args.ga_pop, rng=rng, timing=ga_t)
            store_contention(rec['ga_reactive'], i, [ga_t], la_g)
            sec_g += rs_res[1]
            ok, _ = _check('ga_reactive', asg_g, oct_g, ms_g)
            nerv = nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                      asg_g, oct_g, ds['op_map'])
            _record('ga_reactive', i, ms_g, sec_g, nerv, ok, ms_orig)
            line.append(f'ga={ms_g:6.1f}h({gens}g,cpu{ga_t["cpu_share"]:.2f})')

        # ---- (e) CP-SAT budget sweep on the perturbed whole instance -------
        if budgets:
            sweep_t = {}
            sweep = cpsat_sweep(jl_r, pt_r, lag_r, budgets,
                                ds['mch_ready_time'], args.cpsat_workers,
                                timing=sweep_t)
            for b in budgets:
                mname = f'cpsat_{b}'
                t_b = sweep_t.get(b)
                store_contention(rec[mname], i, [t_b],
                                 t_b['loadavg_start'] if t_b else None)
                ms_c, sec_c, status, asg_c, oct_c = sweep[b]
                feas = False
                if (asg_c >= 0).all():
                    vres = validate_schedule(jl_r, pt_r, lag_r, asg_c, oct_c)
                    feas = bool(vres['feasible'])
                    if not feas:
                        n_infeasible += 1
                        print(f'  !!! {mname} INFEASIBLE on {base}: '
                              f'{vres["violations"][:3]}')
                # CP-SAT re-solves from scratch; nervousness vs the original
                # DRL plan is not meaningful, so it is left NaN.
                _record(mname, i, ms_c, sec_c, None, feas, ms_orig)
            line.append('  '.join(f'cp{b}={sweep[b][0]:6.0f}h'
                                  for b in budgets))

        print(f'  [{i + 1:3d}/{n}] {base}  orig={ms_orig:6.0f}h  '
              + '  '.join(line) + f'   [{ds["info"]}]')

    contention = {m: contention_summary(rec[m]) for m in methods}

    # ---- save structured npy -------------------------------------------------
    save_dir = args.out_dir or f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_npy = os.path.join(save_dir, f'reactive_{dtype}_{args.model_name}.npy')
    payload = {
        'methods': methods,
        'orig_makespans': orig_makespans,
        'records': rec,
        'n_ops_realized': n_ops_realized,
        'info': infos,
        'meta': {
            'model_name': args.model_name,
            'dataset': dataset,
            'disruption_type': dtype,
            'disruption_offset': args.disruption_offset,
            'lag_factor': args.lag_factor,
            'pt_factor': args.pt_factor,
            'shortage_factor': args.shortage_factor,
            'shortage_basis': args.shortage_basis,
            'ga_budget': args.ga_budget,
            'ga_pop': args.ga_pop,
            'ga_seed': args.ga_seed,
            'cpsat_budgets': budgets,
            'cpsat_workers': args.cpsat_workers,
            'seed_test': args.seed_test,
            'n_instances': n,
            'nervousness_time_threshold': NERVOUSNESS_TIME_THRESHOLD,
            'n_infeasible': n_infeasible,
            'n_contract_violations': n_contract,
            'low_cpu_share': LOW_CPU_SHARE,
            'hit_limit_frac': HIT_LIMIT_FRAC,
        },
        'contention': contention,
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
        agg[m] = {k: _mean(rec[m][k]) for k in
                  ('makespan', 'inflation', 'seconds', 'mch_nerv',
                   'time_nerv', 'comb_nerv', 'abs_shift')}
        agg[m]['feasible'] = int(rec[m]['feasible'].sum())

    # ---- summary markdown ---------------------------------------------------
    lines = []
    lines.append(f'# PPVC reactive rescheduling — {dtype} '
                 f'— {args.model_name}\n')
    lines.append(f'- dataset: `{dataset}`  ({n} instances)')
    lines.append(f'- checkpoint: `{ckpt_path}`')
    lines.append(f'- disruption: **{dtype}**  '
                 f'offset={args.disruption_offset}  '
                 f'lag_factor={args.lag_factor}  pt_factor={args.pt_factor}  '
                 f'shortage={args.shortage_factor} x {args.shortage_basis}')
    lines.append(f'- ga_reactive: budget {args.ga_budget} s, population '
                 f'{args.ga_pop}, seed {args.ga_seed}, seeded with the '
                 f'residual right-shift plan')
    lines.append(f'- cpsat budgets (s): {budgets}  (num_workers='
                 f'{args.cpsat_workers or "default"})')
    lines.append(f'- seed_test: {args.seed_test}')
    lines.append(f'- architecture (from snapshot): '
                 f'use_lag_features={use_lag_features}, '
                 f'use_type_embedding={use_type_embedding}')
    lines.append(f'- mean ORIGINAL (pre-disruption) DRL makespan: '
                 f'{_mean(orig_makespans):.1f} h')
    lines.append(f'- mean realized op count: {n_ops_realized.mean():.1f}')
    lines.append(f'- nervousness time threshold: '
                 f'> {NERVOUSNESS_TIME_THRESHOLD} h start shift')
    lines.append(f'- total infeasibilities: {n_infeasible}; residual-contract '
                 f'violations: {n_contract}')
    lines.append('')
    lines.append('| method | mean makespan (h) | mean inflation % | '
                 'mean time (s) | mch-nerv | time-nerv | comb-nerv | '
                 'abs-shift (h) | feasible |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for m in methods:
        a = agg[m]
        if m.startswith('cpsat_'):
            nerv_cols = 'n/a | n/a | n/a | n/a'
        else:
            nerv_cols = (f'{a["mch_nerv"]:.1f} | {a["time_nerv"]:.1f} | '
                         f'{a["comb_nerv"]:.1f} | {a["abs_shift"]:.2f}')
        lines.append(f'| {m} | {a["makespan"]:.1f} | {a["inflation"]:.2f} | '
                     f'{a["seconds"]:.4f} | {nerv_cols} | {a["feasible"]}/{n} |')
    lines.append('')
    lines.append('Inflation % = 100 * (realized makespan - original DRL plan '
                 'makespan) / original DRL plan makespan.')
    lines.append('Nervousness = #ops whose assigned machine changed '
                 '(mch-nerv) + #ops whose start shifted > '
                 f'{NERVOUSNESS_TIME_THRESHOLD} h (time-nerv), vs the original '
                 'plan; comb-nerv is their sum; abs-shift is the mean absolute '
                 'start shift. All over the ops present in both plans.')
    if dtype == 'breakdown':
        lines.append('NOTE: CP-SAT has no native machine-unavailability window '
                     'and no solver changes are permitted, so its breakdown '
                     'numbers re-solve the perturbed instance WITHOUT the '
                     'downtime window — a lower-bound reference only. The '
                     'lag_perturb frontier is the apples-to-apples CP-SAT '
                     'comparison (perturbed time_lag fully captures it).')
    elif dtype == 'material_shortage':
        lines.append('NOTE: the whole-instance CP-SAT cannot encode the '
                     'release time, so it re-solves the undisrupted instance '
                     '— a lower-bound reference only. The prefix-respecting '
                     'residual CP-SAT (eval_residual_cpsat.py) encodes it.')
    elif dtype in NEW_DISRUPTIONS:
        lines.append('NOTE: the whole-instance CP-SAT re-solves the realized '
                     'instance from t=0 and does not fix the committed prefix '
                     '— a reference only. The prefix-respecting residual '
                     'CP-SAT is in eval_residual_cpsat.py.')
    if timed_methods:
        lines.append('')
        lines += contention_table({m: contention[m] for m in timed_methods})

    from common_utils import backbone_file_tag
    btag = backbone_file_tag(args.model_name, configs.framework)
    summary_path = os.path.join(save_dir, f'reactive_{dtype}{btag}_summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- console echo -------------------------------------------------------
    print('\n' + '=' * 72)
    print(f'REACTIVE SUMMARY  ({dataset}, {n} instances, {dtype})')
    print('=' * 72)
    print(f'  mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    print(f'{"method":<14}{"makespan":>10}{"infl%":>9}{"time(s)":>10}'
          f'{"mNerv":>8}{"tNerv":>8}{"cNerv":>8}{"shift":>8}{"feas":>8}')
    for m in methods:
        a = agg[m]
        if m.startswith('cpsat_'):
            print(f'{m:<14}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
                  f'{a["seconds"]:>10.3f}{"n/a":>8}{"n/a":>8}{"n/a":>8}'
                  f'{"n/a":>8}{a["feasible"]:>6}/{n}')
        else:
            print(f'{m:<14}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
                  f'{a["seconds"]:>10.3f}{a["mch_nerv"]:>8.1f}'
                  f'{a["time_nerv"]:>8.1f}{a["comb_nerv"]:>8.1f}'
                  f'{a["abs_shift"]:>8.2f}{a["feasible"]:>6}/{n}')
    for m in timed_methods:
        c = contention[m]
        print(f'  contention {m}: min cpu_share {c["cpu_share_min"]}  '
              f'budget-limited {c["n_hit_limit"]}/{c["n_timed_solves"]}  '
              f'cpu_share<{LOW_CPU_SHARE} {c["n_low_share"]}')
    print('=' * 72)
    print(f'wrote {summary_path}')

    # ---- copy-paste LaTeX macro block ---------------------------------------
    dt = dtype.replace('_', '')

    def _mac(name, val, fmt='{:.1f}'):
        return f'\\newcommand{{\\react{dt}{name}}}{{{fmt.format(val)}}}'

    print('\n' + '-' * 72)
    print('% ---- copy-paste LaTeX macros (reactive ' + f'{dtype}) ----')
    print(_mac('OrigMakespan', _mean(orig_makespans)))
    method_macro = {'drl_reactive': 'Drl', 'spt_reactive': 'Spt',
                    'ga_reactive': 'Ga', 'right_shift': 'Rs'}
    for m in [x for x in RESIDUAL_METHODS if x in methods]:
        cap = method_macro[m]
        a = agg[m]
        print(_mac(cap + 'Makespan', a['makespan']))
        print(_mac(cap + 'Inflation', a['inflation'], '{:.2f}'))
        print(_mac(cap + 'Time', a['seconds'], '{:.3f}'))
        print(_mac(cap + 'MchNerv', a['mch_nerv']))
        print(_mac(cap + 'TimeNerv', a['time_nerv']))
        print(_mac(cap + 'CombNerv', a['comb_nerv']))
        print(_mac(cap + 'AbsShift', a['abs_shift'], '{:.2f}'))
    for b in budgets:
        a = agg[f'cpsat_{b}']
        print(_mac(f'Cpsat{b}Makespan', a['makespan']))
        print(_mac(f'Cpsat{b}Inflation', a['inflation'], '{:.2f}'))
        print(_mac(f'Cpsat{b}Time', a['seconds'], '{:.2f}'))
    print('% ---- end macros ----')
    print('-' * 72)


if __name__ == '__main__':
    main()
