"""
eval_residual_cpsat.py
======================
PREFIX-RESPECTING RESIDUAL CP-SAT baseline for reactive rescheduling (IEEE TII
PPVC paper).

WHY THIS SCRIPT EXISTS
  eval_reactive.py compares DRL-reactive and right-shift against a CP-SAT
  budget sweep, but that sweep re-solves the *whole* perturbed instance from
  scratch (and, for breakdown, cannot even encode the machine outage -- it is a
  lower-bound reference only). That is NOT the strongest CP-SAT competitor
  for online rescheduling. The honest, strong baseline is
  a RESIDUAL re-solve that:
    (a) RESPECTS the committed prefix  -- the ops already executed before the
        disruption keep their realized machine AND start/completion time, and
    (b) PROPERLY ENCODES the disruption -- a machine breakdown becomes a fixed
        unavailable interval the solver must route around.
  This script builds EXACTLY the same disrupted mid-execution state that
  eval_reactive.py builds (greedy-commit the first k=ceil(0.4*N) ops with the
  trained model, then apply lag_perturb or breakdown), then re-solves the
  residual with CP-SAT at 1 / 5 / 30 s budgets, and compares against
  DRL-reactive and right-shift on the SAME states.

CORRECTNESS IS PARAMOUNT
  Every reported makespan comes from a solver run and is independently
  re-validated with schedule_validator.validate_schedule against the perturbed
  lags. We ALSO assert the committed prefix is untouched (machine + start + ct)
  and, for breakdown, that no op runs on the broken machine during the outage.
  Any infeasibility / prefix violation is reported LOUDLY, never averaged over.

CLI (matches eval_reactive defaults)
  python eval_residual_cpsat.py
      [--model_name 10x25+ppvc-mixed+full]
      [--data_path data/PPVC/10x25+ppvc-mixed]
      [--disruption_type lag_perturb|breakdown|both]   (default both)
      [--disruption_offset 0.4] [--lag_factor 2.0]
      [--cpsat_budgets 1,5,30] [--seed_test 50] [--max_instances N]

This script REPLICATES (does not import) the needed logic from eval_reactive.py
because eval_reactive.parse_cli() runs at import time and would eat our argv.
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

    params.py parses the process argv at IMPORT time, so we strip argv to just
    the program name here (same trick eval_reactive.py / eval_ppvc.py use).
    """
    ap = argparse.ArgumentParser(
        description='Prefix-respecting residual CP-SAT baseline (PPVC, IEEE TII)')
    ap.add_argument('--model_name', type=str, default='10x25+ppvc-mixed+full')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed')
    ap.add_argument('--disruption_type', type=str, default='both',
                    choices=['lag_perturb', 'breakdown', 'both'])
    ap.add_argument('--disruption_offset', type=float, default=0.4)
    ap.add_argument('--lag_factor', type=float, default=2.0)
    ap.add_argument('--cpsat_budgets', type=str, default='1,5,30')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=None)
    # --- Stability-aware CP-SAT frontier --------------------------------------
    ap.add_argument('--stability_sweep', action='store_true',
                    help='sweep reassignment+start-shift penalty weights to '
                         'trace the residual CP-SAT stability-makespan frontier')
    ap.add_argument('--stability_weights', type=str,
                    default='0,0.5,1,2,4,8',
                    help='penalty weights w (hours per nervousness unit); '
                         'w=0 is the makespan-only residual')
    ap.add_argument('--stability_budget', type=int, default=30,
                    help='per-solve CP-SAT budget (s) for the sweep')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]  # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli()

from params import configs

# Architecture-critical keys copied from the training-config snapshot into
# params.configs (same list as eval_reactive.py / eval_ppvc.py).
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

NERVOUSNESS_TIME_THRESHOLD = 1.0  # 1 hour (instances are in hours)


def load_train_config(model_name):
    """Read train_log/PPVC/config_<model>.json; apply arch flags to configs."""
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[residual_cpsat] training-config snapshot not found: {path}')
    with open(path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(configs, k, snap[k])
    return snap


def list_instances(data_path, cap=None):
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[residual_cpsat] no instance_*.fjs found under {data_path}')
    stems = [p[:-len('.fjs')] for p in fjs]
    if cap is not None:
        stems = stems[:cap]
    return stems


# ---------------------------------------------------------------------------
# Env construction + greedy primitives (mirror eval_reactive.py exactly).
# ---------------------------------------------------------------------------

def _new_env(jl, pt, meta, use_lag_features, use_type_embedding):
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
    return greedy_select_action(pi)


def rollout_greedy_full(ppo, jl, pt, meta, use_lag_features, use_type_embedding):
    """Full greedy rollout, NO disruption -- the ORIGINAL reference plan."""
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
# Disruption selection helpers (mirror eval_reactive.py exactly).
# ---------------------------------------------------------------------------

def _job_of_op(env, op):
    first = env.job_first_op_id[0]
    last = env.job_last_op_id[0]
    for j in range(env.number_of_jobs):
        if first[j] <= op <= last[j]:
            return j
    raise AssertionError(f'op {op} not found in any job range')


def _pick_lag_op(env):
    """Pick a nonzero-lag op whose successor is still pending (mirror reactive)."""
    true_lag = env.true_op_lag[0]
    scheduled = env.op_scheduled_flag[0].astype(bool)
    last_op = env.job_last_op_id[0]
    nonzero = np.where(true_lag > 0)[0]
    job_last_set = set(int(x) for x in last_op)
    candidates_unscheduled = []
    candidates_succ_pending = []
    for op in nonzero:
        op = int(op)
        if op in job_last_set:
            continue
        succ = op + 1
        if scheduled[succ]:
            continue
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
    """Pick the busiest machine with unscheduled residual work (mirror reactive)."""
    remain = env.remain_process_relation[0]
    unscheduled = ~env.op_scheduled_flag[0].astype(bool)
    work_per_mch = (remain[unscheduled] > 0).sum(axis=0) if unscheduled.any() \
        else np.zeros(env.number_of_machines, dtype=int)
    if work_per_mch.max() <= 0:
        return None
    return int(np.argmax(work_per_mch))


# ---------------------------------------------------------------------------
# Build the disrupted state + capture the committed prefix.
# ---------------------------------------------------------------------------

def build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                          use_type_embedding, disruption_type, offset,
                          lag_factor):
    """Greedy-commit the first k=ceil(offset*N) ops, then apply the disruption.

    Returns a dict capturing the FULL state needed by BOTH the residual CP-SAT
    model and DRL-reactive / right-shift comparators on the SAME disruption:
      committed        : bool[N]  (op_scheduled_flag at the disruption instant)
      committed_mch    : int[N]   (assigned machine for committed ops, else -1)
      committed_start  : float[N] (TRUE start = true_op_ct - true_op_pt[op,mch])
      committed_ct     : float[N] (TRUE completion time)
      t_disrupt_true   : float    (TRUE-unit disruption instant)
      perturbed_time_lag : float[N]
      mch_ready_time   : None | float[M]  (breakdown release floor)
      broken_mch       : None | int
      downtime_true    : None | float
      k                : int
      info             : str
    Also returns (env, state) AFTER the disruption is applied, ready for the
    DRL-reactive residual continuation (so we reuse the exact same env).
    """
    M = pt.shape[1]
    n_ops = pt.shape[0]
    env, state = _new_env(jl, pt, meta, use_lag_features, use_type_embedding)
    assigned_mch = np.full(n_ops, -1, dtype=int)

    k = int(math.ceil(offset * n_ops))
    k = max(0, min(k, n_ops - 1))  # leave >=1 op for the residual

    # ---- phase 1: commit the first k ops (pre-disruption) -------------------
    for _ in range(k):
        action = _greedy_action(ppo, state)
        a = int(action.cpu().numpy()[0])
        chosen_job = a // M
        chosen_op = env.candidate[0, chosen_job]
        assigned_mch[chosen_op] = a % M
        state, _, _ = env.step(actions=action.cpu().numpy())

    # disruption instant (env current decision time), normalized -> TRUE
    t_disrupt_norm = float(env.next_schedule_time[0])
    slope = float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)
    t_disrupt_true = t_disrupt_norm * slope

    # ---- CAPTURE THE COMMITTED PREFIX at the disruption instant -------------
    committed = env.op_scheduled_flag[0].astype(bool).copy()
    committed_mch = np.full(n_ops, -1, dtype=int)
    committed_start = np.full(n_ops, np.nan, dtype=float)
    committed_ct = np.full(n_ops, np.nan, dtype=float)
    true_op_pt = env.true_op_pt[0]  # [N, M], raw int durations
    for op in range(n_ops):
        if committed[op]:
            m = int(assigned_mch[op])
            assert m >= 0, f'committed op {op} has no recorded machine'
            committed_mch[op] = m
            ct = float(env.true_op_ct[0, op])
            committed_ct[op] = ct
            committed_start[op] = ct - float(true_op_pt[op, m])

    perturbed_time_lag = np.asarray(meta['time_lag'], dtype=float).copy()
    mch_ready_time = None
    broken_mch = None
    downtime_true = None
    info = ''

    # ---- apply the disruption (mirror eval_reactive.drl_reactive) -----------
    if disruption_type == 'lag_perturb':
        op = _pick_lag_op(env)
        if op is None:
            nz = np.where(env.true_op_lag[0] > 0)[0]
            op = int(nz[0]) if len(nz) else 0
            info = f'lag_perturb(op={op}, INERT: no future-lag op available)'
        else:
            info = f'lag_perturb(op={op}, x{lag_factor})'
        old_true = float(env.true_op_lag[0, op])
        new_true = old_true * lag_factor
        env.true_op_lag[0, op] = new_true
        env.op_lag[0, op] = new_true / slope
        perturbed_time_lag[op] = new_true
        j = _job_of_op(env, op)
        if bool(env.op_scheduled_flag[0, op]):
            env.candidate_free_time[0, j] = (env.op_ct[0, op]
                                             + env.op_lag[0, op])
            env.true_candidate_free_time[0, j] = (env.true_op_ct[0, op]
                                                  + env.true_op_lag[0, op])
    elif disruption_type == 'breakdown':
        m = _pick_breakdown_machine(env)
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
        env.mch_free_time[0, m] = max(float(env.mch_free_time[0, m]),
                                      t_disrupt_norm + downtime_norm)
        env.true_mch_free_time[0, m] = max(float(env.true_mch_free_time[0, m]),
                                           release_true)
        mch_ready_time = np.zeros(M, dtype=float)
        mch_ready_time[m] = release_true
    else:
        raise ValueError(f'unknown disruption_type {disruption_type}')

    # ---- rebuild env features from the edited arrays (mirror reactive) ------
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

    return {
        'committed': committed,
        'committed_mch': committed_mch,
        'committed_start': committed_start,
        'committed_ct': committed_ct,
        't_disrupt_true': t_disrupt_true,
        'perturbed_time_lag': perturbed_time_lag,
        'mch_ready_time': mch_ready_time,
        'broken_mch': broken_mch,
        'downtime_true': downtime_true,
        'assigned_mch_prefix': assigned_mch.copy(),
        'k': k,
        'info': info,
    }, env, env.state


def drl_reactive_continue(ppo, env, state, assigned_mch):
    """Continue the greedy rollout on the ALREADY-disrupted env (residual).

    Times only the residual solve, as eval_reactive.drl_reactive does.
    Returns (makespan, seconds, assigned_mch[N], true_op_ct[N]).
    """
    M = env.number_of_machines
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
    return float(env.current_makespan[0]), t2 - t1, assigned_mch.copy(), \
        env.true_op_ct[0].copy()


# ---------------------------------------------------------------------------
# Right-shift baseline (mirror eval_reactive.py).
# ---------------------------------------------------------------------------

def right_shift_baseline(jl, pt, assigned_mch_orig, op_ct_orig,
                         perturbed_time_lag, mch_ready_time):
    from right_shift_repair import right_shift_repair
    t1 = time.time()
    repaired_ct, ms = right_shift_repair(
        jl, pt, perturbed_time_lag, assigned_mch_orig,
        op_ct_lagblind=op_ct_orig, mch_ready_time=mch_ready_time)
    sec = time.time() - t1
    return float(ms), sec, assigned_mch_orig.copy(), repaired_ct


# ---------------------------------------------------------------------------
# THE CORE DELIVERABLE: prefix-respecting residual CP-SAT model.
# ---------------------------------------------------------------------------

def residual_cpsat_solve(jl, pt, perturbed_time_lag, disrupt, budget,
                         mk_coef=1, pen_coef=0, assigned_orig=None,
                         starts_orig=None):
    """Re-solve the RESIDUAL with CP-SAT honoring the committed prefix.

    STABILITY-AWARE OPTION: when
    ``pen_coef > 0`` the objective becomes ``mk_coef*makespan +
    pen_coef*nervousness``, where nervousness counts, per UNCOMMITTED op,
    a machine reassignment and a start-shift (>1 h) relative to the
    ORIGINAL pre-disruption plan (``assigned_orig``/``starts_orig``) -- the
    SAME quantity the nervousness metric measures. The effective weight is
    ``w = pen_coef / mk_coef`` hours per nervousness unit. With the default
    ``pen_coef=0`` the objective and every variable are IDENTICAL to the
    makespan-only model, so existing callers are bit-unchanged.

    Mirrors ortools_solver.fjsp_solver structure (alternative intervals,
    AddNoOverlap per machine, finish-start lag precedence), but:
      * committed ops -> FIXED interval [start, start+pt] on their realized
        machine (immovable);
      * uncommitted ops -> free routing over compatible machines, each start
        >= round(t_disrupt);
      * lag precedence start(next) >= end(op) + round(lag[op]) for ALL
        consecutive ops in a job (couples first uncommitted op to its committed
        predecessor);
      * breakdown -> a FIXED unavailable interval [t_disrupt, t_disrupt+downtime]
        on the broken machine, added to that machine's no-overlap set.

    Integer time model (rounded TRUE hours), consistent with ortools_solver.

    Returns dict:
      objective (float), seconds (float), status (str),
      assigned_mch (int[N]), op_ct (float[N]) -- TRUE units,
      t_disrupt_int (int), downtime_int (None|int), broken_mch (None|int).
    """
    from ortools.sat.python import cp_model

    n_ops, M = pt.shape
    jl = np.asarray(jl, dtype=int)
    true_pt = np.asarray(pt, dtype=int)  # generator durations are int hours

    committed = disrupt['committed']
    committed_mch = disrupt['committed_mch']
    committed_start = disrupt['committed_start']
    committed_ct = disrupt['committed_ct']
    t_disrupt_int = int(round(disrupt['t_disrupt_true']))
    lag = np.rint(np.asarray(perturbed_time_lag, dtype=float)).astype(int)

    # op -> job mapping (job-by-job precedence order)
    op_to_job = np.empty(n_ops, dtype=int)
    job_ranges = []
    idx = 0
    for j, L in enumerate(jl):
        job_ranges.append((idx, idx + int(L)))
        op_to_job[idx:idx + int(L)] = j
        idx += int(L)

    # --- generous horizon: prefix finish + all remaining max durations + lags
    horizon = t_disrupt_int
    horizon += int(np.nanmax(committed_ct)) if committed.any() else 0
    for op in range(n_ops):
        if committed[op]:
            continue
        pos = true_pt[op][true_pt[op] > 0]
        horizon += int(pos.max()) if len(pos) else 0
    horizon += int(lag.sum())
    downtime_int = None
    broken_mch = disrupt['broken_mch']
    if disrupt['downtime_true'] is not None:
        downtime_int = int(round(disrupt['downtime_true']))
        horizon += downtime_int
    horizon = max(horizon, 1)

    model = cp_model.CpModel()
    intervals_per_machine = {m: [] for m in range(M)}
    starts = {}   # op -> start IntVar
    ends = {}     # op -> end IntVar
    presences = {}  # (op, m) -> BoolVar (for uncommitted ops over machines)

    for op in range(n_ops):
        if committed[op]:
            # FIXED interval on the realized machine; immovable.
            m = int(committed_mch[op])
            s_val = int(round(committed_start[op]))
            e_val = int(round(committed_ct[op]))
            if s_val < 0:
                s_val = 0
            if e_val <= s_val:
                e_val = s_val + max(1, int(true_pt[op, m]))
            start = model.NewIntVar(s_val, s_val, f'start_op{op}')
            end = model.NewIntVar(e_val, e_val, f'end_op{op}')
            dur = e_val - s_val
            interval = model.NewIntervalVar(start, dur, end, f'iv_op{op}')
            starts[op] = start
            ends[op] = end
            intervals_per_machine[m].append(interval)
        else:
            able = np.where(true_pt[op] > 0)[0]
            assert len(able) > 0, f'op {op} has no compatible machine'
            min_d = int(true_pt[op, able].min())
            max_d = int(true_pt[op, able].max())
            start = model.NewIntVar(t_disrupt_int, horizon, f'start_op{op}')
            end = model.NewIntVar(t_disrupt_int, horizon, f'end_op{op}')
            dur = model.NewIntVar(min_d, max_d, f'dur_op{op}')
            model.NewIntervalVar(start, dur, end, f'iv_op{op}')
            starts[op] = start
            ends[op] = end
            if len(able) > 1:
                l_pres = []
                for m in able:
                    m = int(m)
                    pres = model.NewBoolVar(f'pres_op{op}_m{m}')
                    l_start = model.NewIntVar(t_disrupt_int, horizon,
                                              f'lstart_op{op}_m{m}')
                    l_dur = int(true_pt[op, m])
                    l_end = model.NewIntVar(t_disrupt_int, horizon,
                                            f'lend_op{op}_m{m}')
                    l_iv = model.NewOptionalIntervalVar(
                        l_start, l_dur, l_end, pres, f'liv_op{op}_m{m}')
                    model.Add(start == l_start).OnlyEnforceIf(pres)
                    model.Add(dur == l_dur).OnlyEnforceIf(pres)
                    model.Add(end == l_end).OnlyEnforceIf(pres)
                    intervals_per_machine[m].append(l_iv)
                    presences[(op, m)] = pres
                    l_pres.append(pres)
                model.AddExactlyOne(l_pres)
            else:
                m = int(able[0])
                model.Add(dur == int(true_pt[op, m]))
                # single-machine op: its master interval lives on machine m
                iv = model.NewIntervalVar(start, int(true_pt[op, m]), end,
                                          f'iv_solo_op{op}_m{m}')
                intervals_per_machine[m].append(iv)
                presences[(op, m)] = model.NewConstant(1)

    # --- lag finish-start precedence for ALL consecutive ops in each job -----
    for (s, e) in job_ranges:
        for k in range(s + 1, e):
            pred = k - 1
            model.Add(starts[k] >= ends[pred] + int(lag[pred]))

    # --- breakdown: fixed unavailable interval on the broken machine ---------
    # The outage is [t_disrupt, t_disrupt + downtime]. BUT a committed op already
    # in flight on the broken machine at t_disrupt is REALIZED history and cannot
    # move (it physically ran), so the machine cannot also be "down" during that
    # same realized run. We therefore make the machine UNAVAILABLE to NEW
    # (uncommitted) work until max(t_disrupt + downtime, last realized committed
    # op end on this machine) -- which is exactly the release time DRL-reactive
    # uses (env.true_mch_free_time = max(committed-end, t_disrupt + downtime)) and
    # the floor right_shift uses. We encode this as a single fixed interval that
    # starts after the last committed op on the machine, so it never overlaps a
    # realized committed interval but still reserves the post-disruption downtime.
    outage_added = False
    outage_start_int = None
    outage_end_int = None
    if downtime_int is not None and broken_mch is not None and downtime_int > 0:
        last_committed_end = 0
        for op in np.where(committed)[0]:
            if int(committed_mch[op]) == broken_mch:
                last_committed_end = max(last_committed_end,
                                         int(round(committed_ct[op])))
        # release = when the machine is free for new work (matches DRL/right-shift)
        release_int = max(t_disrupt_int + downtime_int, last_committed_end)
        outage_start_int = max(t_disrupt_int, last_committed_end)
        outage_end_int = release_int
        if outage_end_int > outage_start_int:
            outage = model.NewIntervalVar(
                outage_start_int, outage_end_int - outage_start_int,
                outage_end_int, f'outage_m{broken_mch}')
            intervals_per_machine[broken_mch].append(outage)
            outage_added = True

    # --- machine no-overlap over ALL intervals on each machine ---------------
    for m in range(M):
        ivs = intervals_per_machine[m]
        if len(ivs) > 1:
            model.AddNoOverlap(ivs)

    # --- objective: minimize makespan (+ optional stability penalty) --------
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [ends[op] for op in range(n_ops)])
    if pen_coef and pen_coef > 0:
        # Stability-aware objective: penalize, per UNCOMMITTED op, machine
        # reassignment and start-shift (>1 h) vs the ORIGINAL plan, i.e. the
        # exact nervousness the metric measures.  w = pen_coef / mk_coef.
        assert assigned_orig is not None and starts_orig is not None, \
            'penalized solve needs assigned_orig and starts_orig'
        pen_terms = []
        for op in range(n_ops):
            if committed[op]:
                continue
            able = np.where(true_pt[op] > 0)[0]
            orig_m = int(assigned_orig[op])
            orig_s = int(round(float(starts_orig[op])))
            # machine-reassignment indicator (1 == moved off original machine)
            if len(able) > 1 and (op, orig_m) in presences:
                pen_terms.append(1 - presences[(op, orig_m)])
            elif len(able) > 1:
                pen_terms.append(model.NewConstant(1))  # orig mch ineligible
            # start-shift indicator: |start - orig_start| > 1 h  (int: >= 2)
            dev = model.NewIntVar(0, 2 * horizon, f'dev_op{op}')
            model.AddAbsEquality(dev, starts[op] - orig_s)
            shifted = model.NewBoolVar(f'shift_op{op}')
            model.Add(dev >= 2).OnlyEnforceIf(shifted)
            model.Add(dev <= 1).OnlyEnforceIf(shifted.Not())
            pen_terms.append(shifted)
        model.Minimize(int(mk_coef) * makespan + int(pen_coef) * sum(pen_terms))
    else:
        model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget)
    t1 = time.time()
    status = solver.Solve(model)
    t2 = time.time()

    assigned_mch = np.full(n_ops, -1, dtype=int)
    op_ct = np.zeros(n_ops, dtype=float)
    obj = float('nan')
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        obj = float(solver.ObjectiveValue())
        for op in range(n_ops):
            if committed[op]:
                assigned_mch[op] = int(committed_mch[op])
            else:
                able = np.where(true_pt[op] > 0)[0]
                if len(able) == 1:
                    assigned_mch[op] = int(able[0])
                else:
                    for m in able:
                        if solver.Value(presences[(op, int(m))]):
                            assigned_mch[op] = int(m)
                            break
            op_ct[op] = float(solver.Value(ends[op]))

    return {
        'objective': obj,
        'seconds': t2 - t1,
        'status': status_name,
        'assigned_mch': assigned_mch,
        'op_ct': op_ct,
        't_disrupt_int': t_disrupt_int,
        'downtime_int': downtime_int,
        'broken_mch': broken_mch,
        'outage_start_int': outage_start_int,
        'outage_end_int': outage_end_int,
        'committed': committed,
    }


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------

def check_prefix_respected(disrupt, assigned_mch, op_ct, tol=0.51):
    """Return list of prefix-violation strings (empty == respected).

    Committed ops must keep machine AND start/ct (within int-rounding tol).
    """
    viol = []
    committed = disrupt['committed']
    for op in np.where(committed)[0]:
        op = int(op)
        if assigned_mch[op] != disrupt['committed_mch'][op]:
            viol.append(
                f'prefix op {op}: machine {assigned_mch[op]} != committed '
                f'{int(disrupt["committed_mch"][op])}')
        if abs(float(op_ct[op]) - float(disrupt['committed_ct'][op])) > tol:
            viol.append(
                f'prefix op {op}: ct {op_ct[op]:.3f} != committed '
                f'{disrupt["committed_ct"][op]:.3f}')
    return viol


def check_breakdown_window(res, pt, tol=1e-6):
    """For breakdown: no NEW (uncommitted) op may occupy the broken machine
    during the reserved outage window [outage_start, outage_end).

    Committed ops are realized history -- an op already in flight on the machine
    when it breaks down physically ran, so it legitimately occupies the machine
    up to its realized completion; the machine then stays down for the residual
    downtime. We therefore only require that re-routed (uncommitted) work avoids
    the outage window the solver was given.

    Returns list of violation strings (empty == respected).
    """
    viol = []
    m = res['broken_mch']
    dt = res['downtime_int']
    if m is None or dt is None or dt <= 0:
        return viol
    t0 = res.get('outage_start_int')
    t1 = res.get('outage_end_int')
    if t0 is None or t1 is None or t1 <= t0:
        return viol  # outage fully absorbed by a committed op; nothing reserved
    assigned = res['assigned_mch']
    op_ct = res['op_ct']
    committed = res['committed']
    pt = np.asarray(pt, dtype=float)
    for op in range(len(assigned)):
        if assigned[op] != m:
            continue
        if committed[op]:
            continue  # realized history, exempt
        s = float(op_ct[op]) - float(pt[op, m])
        e = float(op_ct[op])
        # overlap with [t0, t1) iff s < t1 and e > t0
        if s < t1 - tol and e > t0 + tol:
            viol.append(
                f'uncommitted op {op} on broken m{m} runs [{s:.1f},{e:.1f}] '
                f'overlapping reserved outage [{t0},{t1}]')
    return viol


def nervousness(pt, assigned_orig, op_ct_orig, assigned_new, op_ct_new):
    from schedule_validator import reconstruct_starts
    starts_orig = reconstruct_starts(pt, assigned_orig, op_ct_orig)
    starts_new = reconstruct_starts(pt, assigned_new, op_ct_new)
    mch_changed = int(np.sum(assigned_orig != assigned_new))
    time_shifted = int(np.sum(
        np.abs(starts_new - starts_orig) > NERVOUSNESS_TIME_THRESHOLD))
    return mch_changed, time_shifted, mch_changed + time_shifted


# ---------------------------------------------------------------------------
# Per-disruption driver.
# ---------------------------------------------------------------------------

def run_disruption(disruption_type, args, ppo, stems, dataset, ckpt_path,
                   use_lag_features, use_type_embedding, budgets):
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule

    n = len(stems)
    cpsat_names = [f'residual_cpsat_{b}s' for b in budgets]
    methods = cpsat_names + ['drl_reactive', 'right_shift']

    rec = {m: {'makespan': np.full(n, np.nan),
               'inflation': np.full(n, np.nan),
               'seconds': np.full(n, np.nan),
               'mch_nerv': np.full(n, np.nan),
               'time_nerv': np.full(n, np.nan),
               'comb_nerv': np.full(n, np.nan),
               'feasible': np.zeros(n, dtype=bool)}
           for m in methods}
    orig_makespans = np.full(n, np.nan)
    # CP-SAT proof status counters per budget
    optimal_count = {b: 0 for b in budgets}
    feasible_only_count = {b: 0 for b in budgets}
    no_sol_count = {b: 0 for b in budgets}
    n_infeasible = 0
    n_prefix_violations = 0
    n_breakdown_window_violations = 0

    print('=' * 78)
    print(f'RESIDUAL CP-SAT  model={args.model_name}  dataset={dataset}  '
          f'instances={n}')
    print(f'  disruption={disruption_type}  offset={args.disruption_offset}  '
          f'lag_factor={args.lag_factor}  budgets(s)={budgets}')
    print(f'  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  seed_test={args.seed_test}')
    print('=' * 78)

    sample_checks_printed = 0

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        true_lag = np.asarray(meta['time_lag'], dtype=float)

        # ---- ORIGINAL pre-disruption DRL plan (inflation baseline) ----------
        ms_orig, _s, assigned_orig, oct_orig = rollout_greedy_full(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding)
        res0 = validate_schedule(jl, pt, true_lag, assigned_orig, oct_orig)
        assert res0['feasible'], (
            f'ORIGINAL plan infeasible on {base}:\n  '
            + '\n  '.join(res0['violations'][:10]))
        orig_makespans[i] = ms_orig

        # ---- build the disrupted state + capture the committed prefix -------
        disrupt, env, state = build_disrupted_state(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding,
            disruption_type, args.disruption_offset, args.lag_factor)
        perturbed_lag = disrupt['perturbed_time_lag']
        mch_ready_time = disrupt['mch_ready_time']

        # ---- (a) DRL-reactive: continue the SAME disrupted env --------------
        ms_d, sec_d, asg_d, oct_d = drl_reactive_continue(
            ppo, env, state, disrupt['assigned_mch_prefix'].copy())
        res_d = validate_schedule(jl, pt, perturbed_lag, asg_d, oct_d)
        if not res_d['feasible']:
            n_infeasible += 1
            print(f'  !!! drl_reactive INFEASIBLE on {base}: '
                  f'{res_d["violations"][:3]}')
        mn, tn, cn = nervousness(pt, assigned_orig, oct_orig, asg_d, oct_d)
        rec['drl_reactive']['makespan'][i] = ms_d
        rec['drl_reactive']['inflation'][i] = (
            100.0 * (ms_d - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['drl_reactive']['seconds'][i] = sec_d
        rec['drl_reactive']['mch_nerv'][i] = mn
        rec['drl_reactive']['time_nerv'][i] = tn
        rec['drl_reactive']['comb_nerv'][i] = cn
        rec['drl_reactive']['feasible'][i] = res_d['feasible']

        # ---- (b) right-shift repair -----------------------------------------
        ms_rs, sec_rs, asg_rs, oct_rs = right_shift_baseline(
            jl, pt, assigned_orig, oct_orig, perturbed_lag, mch_ready_time)
        res_rs = validate_schedule(jl, pt, perturbed_lag, asg_rs, oct_rs)
        if not res_rs['feasible']:
            n_infeasible += 1
            print(f'  !!! right_shift INFEASIBLE on {base}: '
                  f'{res_rs["violations"][:3]}')
        mn, tn, cn = nervousness(pt, assigned_orig, oct_orig, asg_rs, oct_rs)
        rec['right_shift']['makespan'][i] = ms_rs
        rec['right_shift']['inflation'][i] = (
            100.0 * (ms_rs - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['right_shift']['seconds'][i] = sec_rs
        rec['right_shift']['mch_nerv'][i] = mn
        rec['right_shift']['time_nerv'][i] = tn
        rec['right_shift']['comb_nerv'][i] = cn
        rec['right_shift']['feasible'][i] = res_rs['feasible']

        # ---- (c) PREFIX-RESPECTING RESIDUAL CP-SAT at each budget -----------
        cp_summ = {}
        for b in budgets:
            mname = f'residual_cpsat_{b}s'
            res = residual_cpsat_solve(jl, pt, perturbed_lag, disrupt, b)
            cp_summ[b] = res
            if res['status'] == 'OPTIMAL':
                optimal_count[b] += 1
            elif res['status'] == 'FEASIBLE':
                feasible_only_count[b] += 1
            else:
                no_sol_count[b] += 1

            if res['status'] in ('OPTIMAL', 'FEASIBLE'):
                # independent feasibility validation vs perturbed lags
                vres = validate_schedule(jl, pt, perturbed_lag,
                                         res['assigned_mch'], res['op_ct'])
                pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                               res['op_ct'])
                bviol = check_breakdown_window(res, pt)
                feasible = vres['feasible'] and not pviol and not bviol
                if not vres['feasible']:
                    n_infeasible += 1
                    print(f'  !!! {mname} VALIDATOR-INFEASIBLE on {base}: '
                          f'{vres["violations"][:3]}')
                if pviol:
                    n_prefix_violations += 1
                    print(f'  !!! {mname} PREFIX VIOLATION on {base}: '
                          f'{pviol[:3]}')
                if bviol:
                    n_breakdown_window_violations += 1
                    print(f'  !!! {mname} BREAKDOWN-WINDOW VIOLATION on {base}: '
                          f'{bviol[:3]}')
                # cross-check solver objective vs validator makespan
                if feasible and abs(vres['makespan'] - res['objective']) > 0.51:
                    print(f'  !!! {mname} OBJ/VALIDATOR MISMATCH on {base}: '
                          f'obj={res["objective"]} valid={vres["makespan"]}')
                rec[mname]['makespan'][i] = res['objective']
                rec[mname]['inflation'][i] = (
                    100.0 * (res['objective'] - ms_orig) / ms_orig
                    if ms_orig > 0 else 0.0)
                rec[mname]['feasible'][i] = feasible
                mn, tn, cn = nervousness(pt, assigned_orig, oct_orig,
                                         res['assigned_mch'], res['op_ct'])
                rec[mname]['mch_nerv'][i] = mn
                rec[mname]['time_nerv'][i] = tn
                rec[mname]['comb_nerv'][i] = cn
            else:
                print(f'  !!! {mname} NO SOLUTION ({res["status"]}) on {base}')
                rec[mname]['feasible'][i] = False
            rec[mname]['seconds'][i] = res['seconds']

        # ---- explicit sample-instance validation printout -------------------
        if sample_checks_printed < 2 and budgets:
            b = budgets[-1]  # the largest budget
            res = cp_summ[b]
            if res['status'] in ('OPTIMAL', 'FEASIBLE'):
                vres = validate_schedule(jl, pt, perturbed_lag,
                                         res['assigned_mch'], res['op_ct'])
                pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                               res['op_ct'])
                bviol = check_breakdown_window(res, pt)
                n_committed = int(disrupt['committed'].sum())
                print(f'  --- SAMPLE CHECK [{base}, {disruption_type}, '
                      f'residual_cpsat_{b}s] ---')
                print(f'      k committed = {disrupt["k"]} ('
                      f'{n_committed} flagged scheduled), '
                      f't_disrupt = {disrupt["t_disrupt_true"]:.2f}h '
                      f'(int {res["t_disrupt_int"]})')
                print(f'      (a) validator feasible = {vres["feasible"]}  '
                      f'makespan = {vres["makespan"]:.1f}h  '
                      f'(solver obj {res["objective"]:.1f}h, '
                      f'status {res["status"]})')
                print(f'      (b) committed prefix unchanged = {not pviol}  '
                      f'({len(pviol)} violations)')
                if disruption_type == 'breakdown':
                    print(f'      (c) no NEW op on broken m{res["broken_mch"]} '
                          f'during reserved outage '
                          f'[{res["outage_start_int"]},{res["outage_end_int"]}] '
                          f'(downtime {res["downtime_int"]}h from t_disrupt '
                          f'{res["t_disrupt_int"]}) = {not bviol}  '
                          f'({len(bviol)} violations)')
                # HARD asserts on the sample
                assert vres['feasible'], f'sample {base} infeasible!'
                assert not pviol, f'sample {base} prefix moved: {pviol[:3]}'
                assert not bviol, f'sample {base} ran during outage: {bviol[:3]}'
                sample_checks_printed += 1

        # ---- progress line --------------------------------------------------
        cp_str = '  '.join(
            f'rcp{b}={cp_summ[b]["objective"]:6.0f}h({cp_summ[b]["status"][:4]})'
            for b in budgets)
        print(f'  [{i + 1:3d}/{n}] {base}  orig={ms_orig:6.0f}h  '
              f'drl={ms_d:6.0f}h(+{rec["drl_reactive"]["inflation"][i]:4.1f}%,'
              f'{sec_d:.2f}s)  rs={ms_rs:6.0f}h(+'
              f'{rec["right_shift"]["inflation"][i]:5.1f}%)  {cp_str}'
              f'   [{disrupt["info"]}]')

    # ---- HARD GATE: stop loudly if any prefix/window violation occurred -----
    if n_prefix_violations or n_breakdown_window_violations:
        print('\n' + '!' * 78)
        print(f'STOP: residual CP-SAT broke the contract on {disruption_type} -- '
              f'{n_prefix_violations} prefix violation(s), '
              f'{n_breakdown_window_violations} breakdown-window violation(s). '
              f'Numbers are NOT trustworthy; investigate before using.')
        print('!' * 78)

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

    # ---- save structured npy ------------------------------------------------
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_npy = os.path.join(save_dir, f'residual_cpsat_{disruption_type}.npy')
    payload = {
        'methods': methods,
        'orig_makespans': orig_makespans,
        'records': rec,
        'optimal_count': optimal_count,
        'feasible_only_count': feasible_only_count,
        'no_sol_count': no_sol_count,
        'meta': {
            'model_name': args.model_name,
            'dataset': dataset,
            'disruption_type': disruption_type,
            'disruption_offset': args.disruption_offset,
            'lag_factor': args.lag_factor,
            'cpsat_budgets': budgets,
            'seed_test': args.seed_test,
            'n_instances': n,
            'n_infeasible': n_infeasible,
            'n_prefix_violations': n_prefix_violations,
            'n_breakdown_window_violations': n_breakdown_window_violations,
        },
    }
    np.save(out_npy, payload, allow_pickle=True)

    # ---- console summary ----------------------------------------------------
    print('\n' + '=' * 78)
    print(f'RESIDUAL CP-SAT SUMMARY  ({dataset}, {n} instances, '
          f'{disruption_type})')
    print('=' * 78)
    print(f'  mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    print(f'{"method":<20}{"makespan":>10}{"infl%":>9}{"time(s)":>10}'
          f'{"mNerv":>8}{"tNerv":>8}{"cNerv":>8}{"feas":>9}')
    for m in methods:
        a = agg[m]
        print(f'{m:<20}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
              f'{a["seconds"]:>10.3f}{a["mch_nerv"]:>8.1f}'
              f'{a["time_nerv"]:>8.1f}{a["comb_nerv"]:>8.1f}'
              f'{a["feasible"]:>6}/{n}')
    print('-' * 78)
    print('  CP-SAT proof status per budget:')
    for b in budgets:
        print(f'    {b:>3}s : OPTIMAL={optimal_count[b]:3d}  '
              f'FEASIBLE-only={feasible_only_count[b]:3d}  '
              f'no-solution={no_sol_count[b]:3d}  (of {n})')
    print(f'  total validator-infeasibilities: {n_infeasible}')
    print(f'  total prefix violations: {n_prefix_violations}')
    print(f'  total breakdown-window violations: '
          f'{n_breakdown_window_violations}')
    print(f'  saved {out_npy}')
    print('=' * 78)

    # ---- LaTeX macro block --------------------------------------------------
    dt = disruption_type.replace('_', '').replace('lagperturb', 'LP') \
        .replace('breakdown', 'BD')

    def _mac(name, val, fmt='{:.1f}'):
        return f'\\newcommand{{\\ResidCpsat{dt}{name}}}{{{fmt.format(val)}}}'

    macro_lines = []
    macro_lines.append('% ---- copy-paste LaTeX macros (residual CP-SAT '
                       + disruption_type + ') ----')
    macro_lines.append(_mac('OrigMakespan', _mean(orig_makespans)))
    for b in budgets:
        a = agg[f'residual_cpsat_{b}s']
        macro_lines.append(_mac(f'{b}sMakespan', a['makespan']))
        macro_lines.append(_mac(f'{b}sInflation', a['inflation'], '{:.2f}'))
        macro_lines.append(_mac(f'{b}sTime', a['seconds'], '{:.2f}'))
        macro_lines.append(_mac(f'{b}sOptimal', optimal_count[b], '{:d}'))
        macro_lines.append(_mac(f'{b}sFeasOnly', feasible_only_count[b], '{:d}'))
        macro_lines.append(_mac(f'{b}sCombNerv', a['comb_nerv']))
    for m, cap in (('drl_reactive', 'Drl'), ('right_shift', 'Rs')):
        a = agg[m]
        macro_lines.append(_mac(cap + 'Makespan', a['makespan']))
        macro_lines.append(_mac(cap + 'Inflation', a['inflation'], '{:.2f}'))
        macro_lines.append(_mac(cap + 'Time', a['seconds'], '{:.3f}'))
        macro_lines.append(_mac(cap + 'CombNerv', a['comb_nerv']))
    macro_lines.append('% ---- end macros ----')
    print('\n' + '\n'.join(macro_lines))

    # ---- markdown summary ---------------------------------------------------
    lines = [f'# PPVC residual CP-SAT -- {disruption_type} -- {args.model_name}\n']
    lines.append(f'- dataset: `{dataset}` ({n} instances)')
    lines.append(f'- checkpoint: `{ckpt_path}`')
    lines.append(f'- disruption: **{disruption_type}**  '
                 f'offset={args.disruption_offset}  '
                 f'lag_factor={args.lag_factor}')
    lines.append(f'- budgets (s): {budgets}')
    lines.append(f'- mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    lines.append(f'- total validator-infeasibilities: {n_infeasible}')
    lines.append(f'- total prefix violations: {n_prefix_violations}')
    lines.append(f'- total breakdown-window violations: '
                 f'{n_breakdown_window_violations}\n')
    lines.append('| method | mean makespan (h) | mean inflation % | '
                 'mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for m in methods:
        a = agg[m]
        lines.append(f'| {m} | {a["makespan"]:.1f} | {a["inflation"]:.2f} | '
                     f'{a["seconds"]:.4f} | {a["mch_nerv"]:.1f} | '
                     f'{a["time_nerv"]:.1f} | {a["comb_nerv"]:.1f} | '
                     f'{a["feasible"]}/{n} |')
    lines.append('')
    lines.append('| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |')
    lines.append('|---|---|---|---|')
    for b in budgets:
        lines.append(f'| {b} | {optimal_count[b]} | '
                     f'{feasible_only_count[b]} | {no_sol_count[b]} |')
    md_path = os.path.join(save_dir,
                           f'residual_cpsat_{disruption_type}_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {md_path}')

    return {
        'agg': agg,
        'methods': methods,
        'orig_mean': _mean(orig_makespans),
        'optimal_count': optimal_count,
        'feasible_only_count': feasible_only_count,
        'no_sol_count': no_sol_count,
        'n': n,
        'out_npy': out_npy,
        'md_path': md_path,
        'macro_block': '\n'.join(macro_lines),
        'n_infeasible': n_infeasible,
        'n_prefix_violations': n_prefix_violations,
        'n_breakdown_window_violations': n_breakdown_window_violations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _weight_to_coefs(w):
    """Map weight w (hours per nervousness unit) to integer (mk_coef, pen_coef)
    with w = pen_coef / mk_coef. w == 0 -> (1, 0) = makespan-only."""
    from fractions import Fraction
    if float(w) == 0.0:
        return 1, 0
    fr = Fraction(str(w)).limit_denominator(1000)
    return int(fr.denominator), int(fr.numerator)


def run_stability_sweep(disruption_type, args, ppo, stems, dataset,
                        use_lag_features, use_type_embedding, weights, budget):
    """Trace the residual CP-SAT stability-makespan frontier and place the
    DRL-reactive operating point on it, on the SAME disrupted states as the
    main residual experiment."""
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule, reconstruct_starts

    coefs = [_weight_to_coefs(w) for w in weights]
    n = len(stems)
    W = {w: {'infl': [], 'nerv': [], 'mch_nerv': [], 'time_nerv': [],
             'sec': [], 'optimal': 0, 'feas': 0} for w in weights}
    drl = {'infl': [], 'nerv': [], 'mch_nerv': [], 'time_nerv': [], 'sec': [],
           'feas': 0}
    orig_ms = []

    print('=' * 78)
    print(f'STABILITY SWEEP  disruption={disruption_type}  n={n}  '
          f'budget={budget}s')
    print(f'  weights(w=pen/mk)={list(zip(weights, coefs))}')
    print('=' * 78)

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        true_lag = np.asarray(meta['time_lag'], dtype=float)

        ms_orig, _s, assigned_orig, oct_orig = rollout_greedy_full(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding)
        r0 = validate_schedule(jl, pt, true_lag, assigned_orig, oct_orig)
        assert r0['feasible'], f'ORIGINAL plan infeasible on {base}'
        starts_orig = reconstruct_starts(pt, assigned_orig, oct_orig)
        orig_ms.append(ms_orig)

        disrupt, env, state = build_disrupted_state(
            ppo, jl, pt, meta, use_lag_features, use_type_embedding,
            disruption_type, args.disruption_offset, args.lag_factor)
        perturbed_lag = disrupt['perturbed_time_lag']

        # ---- DRL-reactive operating point -----------------------------------
        ms_d, sec_d, asg_d, oct_d = drl_reactive_continue(
            ppo, env, state, disrupt['assigned_mch_prefix'].copy())
        rd = validate_schedule(jl, pt, perturbed_lag, asg_d, oct_d)
        if rd['feasible']:
            drl['feas'] += 1
        else:
            print(f'  !!! drl_reactive INFEASIBLE on {base}')
        mn, tn, cn = nervousness(pt, assigned_orig, oct_orig, asg_d, oct_d)
        drl['infl'].append(100.0 * (ms_d - ms_orig) / ms_orig)
        drl['nerv'].append(cn); drl['mch_nerv'].append(mn)
        drl['time_nerv'].append(tn); drl['sec'].append(sec_d)

        # ---- penalized residual CP-SAT at each weight -----------------------
        for w, (mk, pc) in zip(weights, coefs):
            if pc == 0:
                res = residual_cpsat_solve(jl, pt, perturbed_lag, disrupt,
                                           budget)
            else:
                res = residual_cpsat_solve(jl, pt, perturbed_lag, disrupt,
                                           budget, mk_coef=mk, pen_coef=pc,
                                           assigned_orig=assigned_orig,
                                           starts_orig=starts_orig)
            if res['status'] not in ('OPTIMAL', 'FEASIBLE'):
                print(f'  !!! w={w} NO SOLUTION on {base} ({res["status"]})')
                continue
            ms_new = float(np.max(res['op_ct']))
            vres = validate_schedule(jl, pt, perturbed_lag,
                                     res['assigned_mch'], res['op_ct'])
            pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                           res['op_ct'])
            if not vres['feasible'] or pviol:
                print(f'  !!! w={w} INVALID on {base}: feas={vres["feasible"]} '
                      f'prefix_viol={len(pviol)}')
                continue
            mn, tn, cn = nervousness(pt, assigned_orig, oct_orig,
                                     res['assigned_mch'], res['op_ct'])
            W[w]['infl'].append(100.0 * (ms_new - ms_orig) / ms_orig)
            W[w]['nerv'].append(cn); W[w]['mch_nerv'].append(mn)
            W[w]['time_nerv'].append(tn); W[w]['sec'].append(res['seconds'])
            W[w]['feas'] += 1
            if res['status'] == 'OPTIMAL':
                W[w]['optimal'] += 1

        if (i + 1) % 10 == 0:
            print(f'  ...{i + 1}/{n} done')

    def _m(a):
        return float(np.mean(a)) if len(a) else float('nan')

    print('\n' + '#' * 78)
    print(f'# STABILITY FRONTIER  [{disruption_type}]  '
          f'(n={n}, orig mean {_m(orig_ms):.1f}h)')
    print('#' * 78)
    print(f'{"method":<22}{"infl%":>9}{"nerv":>9}{"mch":>7}{"tnv":>7}'
          f'{"time_s":>9}{"opt":>8}')
    print(f'{"DRL-reactive":<22}{_m(drl["infl"]):>9.2f}{_m(drl["nerv"]):>9.1f}'
          f'{_m(drl["mch_nerv"]):>7.1f}{_m(drl["time_nerv"]):>7.1f}'
          f'{_m(drl["sec"]):>9.3f}{"-":>8}')
    for w in weights:
        d = W[w]
        print(f'{("resid CP-SAT w=" + str(w)):<22}{_m(d["infl"]):>9.2f}'
              f'{_m(d["nerv"]):>9.1f}{_m(d["mch_nerv"]):>7.1f}'
              f'{_m(d["time_nerv"]):>7.1f}{_m(d["sec"]):>9.3f}'
              f'{(str(d["optimal"]) + "/" + str(d["feas"])):>8}')

    payload = {
        'disruption_type': disruption_type, 'n': n, 'weights': weights,
        'coefs': coefs, 'budget': budget, 'orig_mean': _m(orig_ms),
        'drl': {k: (drl[k] if k == 'feas' else [float(x) for x in drl[k]])
                for k in drl},
        'frontier': {w: {k: (W[w][k] if k in ('optimal', 'feas')
                             else [float(x) for x in W[w][k]])
                         for k in W[w]} for w in weights},
        'meta': {'offset': args.disruption_offset,
                 'lag_factor': args.lag_factor, 'seed_test': args.seed_test},
    }
    outdir = f'test_results/PPVC/{dataset}'
    os.makedirs(outdir, exist_ok=True)
    out = f'{outdir}/stability_sweep_{disruption_type}.npy'
    np.save(out, payload, allow_pickle=True)
    print(f'\n[saved] {out}')
    return payload


def main():
    args = _ARGS
    budgets = [int(b) for b in args.cpsat_budgets.split(',') if b.strip()]

    snap = load_train_config(args.model_name)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    # force CPU to avoid GPU contention with a concurrent training run
    configs.device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from common_utils import setup_seed
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    # Per-step inference is over tiny graphs; torch's default of one thread per
    # CPU core thrashes (24 threads fighting over sub-millisecond ops), making
    # the greedy rollouts ~10x slower. Cap intra-op threads so the sequential
    # rollout runs fast. This is a pure SPEED knob -- greedy argmax decisions
    # are deterministic and bit-identical regardless of thread count.
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    setup_seed(args.seed_test)

    from model.PPO import PPO_initialize
    ckpt_path = f'./trained_network/PPVC/{args.model_name}.pth'
    if not os.path.exists(ckpt_path):
        sys.exit(f'[residual_cpsat] checkpoint not found: {ckpt_path}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    ppo.policy.eval()

    stems = list_instances(args.data_path, cap=args.max_instances)
    dataset = os.path.basename(os.path.normpath(args.data_path))

    if args.disruption_type == 'both':
        disruptions = ['lag_perturb', 'breakdown']
    else:
        disruptions = [args.disruption_type]

    if args.stability_sweep:
        weights = [float(x) for x in args.stability_weights.split(',')
                   if x.strip()]
        for dt in disruptions:
            run_stability_sweep(
                dt, args, ppo, stems, dataset,
                use_lag_features, use_type_embedding,
                weights, args.stability_budget)
        return

    results = {}
    for dt in disruptions:
        results[dt] = run_disruption(
            dt, args, ppo, stems, dataset, ckpt_path,
            use_lag_features, use_type_embedding, budgets)

    # final cross-disruption recap
    print('\n' + '#' * 78)
    print('# FINAL RECAP')
    print('#' * 78)
    for dt in disruptions:
        r = results[dt]
        print(f'\n[{dt}]  (n={r["n"]}, orig mean {r["orig_mean"]:.1f}h)')
        for m in r['methods']:
            a = r['agg'][m]
            print(f'  {m:<20} ms={a["makespan"]:7.1f}h  '
                  f'infl={a["inflation"]:+6.2f}%  '
                  f't={a["seconds"]:7.3f}s  feas={a["feasible"]}/{r["n"]}')
        for b in budgets:
            print(f'  budget {b}s: OPTIMAL={r["optimal_count"][b]}  '
                  f'FEASIBLE-only={r["feasible_only_count"][b]}  '
                  f'no-sol={r["no_sol_count"][b]}')
        if (r['n_prefix_violations'] or r['n_breakdown_window_violations']
                or r['n_infeasible']):
            print(f'  !!! issues: infeasible={r["n_infeasible"]} '
                  f'prefix_viol={r["n_prefix_violations"]} '
                  f'window_viol={r["n_breakdown_window_violations"]}')


if __name__ == '__main__':
    main()
