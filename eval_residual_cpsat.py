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

CONTENTION EVIDENCE
  Every time-limited solve records its wall and process-CPU seconds, workers,
  cpu_share = CPU s / (wall s x workers), status and hit_limit (>= 95 % of
  the budget used); each record adds cpu_share_min, n_timed_solves,
  n_hit_limit, n_low_share, n_no_evidence, affinity and loadavg_start. A
  budget-limited solve with cpu_share < 0.9 ran on shared cores and is
  invalid (the summary table and the npy 'contention' entry show it).

NEW DISRUPTION TYPES (urgent_arrival, material_shortage, pt_deviation,
cancellation)
  Their disrupted state is built by eval_reactive.build_disrupted_state (now
  importable), so every arm here sees exactly the state eval_reactive builds:
    * urgent_arrival: the realized instance has the new module appended; its
      ops are uncommitted and start >= t_d like every residual op.
    * material_shortage: the job's next op gets start >= ceil(release).
    * pt_deviation: the in-flight op is part of the committed prefix with its
      longer realized completion; the realized pt matrix carries the longer pt.
    * cancellation: the removed ops are absent from the realized instance.
  Validation, prefix checks and the right-shift arm use the realized instance;
  the right-shift arm for these types is eval_reactive.right_shift_residual,
  and nervousness is measured over the ops present in both plans
  (eval_reactive.nervousness_mapped). The stability sweep accepts them too
  (added ops carry no penalty: they have no original plan).

CLI (matches eval_reactive defaults)
  python eval_residual_cpsat.py
      [--model_name 10x25+ppvc-mixed+full]
      [--data_path data/PPVC/10x25+ppvc-mixed]
      [--disruption_type lag_perturb|breakdown|both|urgent_arrival|
                         material_shortage|pt_deviation|cancellation|all]
                                                        (default both)
      [--disruption_offset 0.4] [--lag_factor 2.0] [--pt_factor 1.5]
      [--shortage_factor 0.5] [--shortage_basis op_mean|remaining_work]
      [--cpsat_budgets 1,5,30] [--cpsat_workers 0] [--seed_test 50]
      [--max_instances N] [--out_dir DIR] [--hint none|right_shift]
      [--breakdown_hours H] [--state_fp]
      [--stability_sweep [--stability_weights ...] [--stability_budget s]
                         [--stability_hint none|right_shift]]
  --disruption_type also takes a comma list of single types.
  --breakdown_hours H sets the breakdown outage to H hours (default None =
  0.5 x the mean processing time, as before); eval_reactive.py takes the same
  option and builds the same outage. With H given, every breakdown state built
  by the replicated legacy branch below is asserted to equal the state of
  eval_reactive.build_disrupted_state (state_fingerprint), in this process.
  --state_fp stores the per-instance state fingerprints in the npy, for the
  cross-run check against eval_reactive.py --state_fp (same TORCH_THREADS).
  --match_reactive FILE asserts, order by order, that this run rolls SLAB
  exactly as the eval_reactive.py run that wrote FILE (its
  reactive_<type>_<model>.npy for the same data, type and options): the
  original plan's makespan, the disruption info string, SLAB's re-planned
  makespan and change counts and, when FILE carries state_fp, the state
  fingerprint must all be equal, or the run stops. eval_reactive.py rolls the
  policy with torch.set_num_threads(1) (TORCH_THREADS default 1); this
  script's default is 4 threads, which changes a few near-tie greedy
  decisions, so a matching run needs TORCH_THREADS=1.
  --stability_hint right_shift starts every penalized solve of the stability
  sweep from the right-shift plan, built exactly as the --hint right_shift
  main run builds it (the hint also sets the penalty variables of that plan).
  Default none: the sweep starts cold, as before.
  --hint right_shift starts every residual CP-SAT solve from the right-shift
  plan of the same instance (retimed to the integer model; see
  right_shift_hint); the records then carry hint_given, hint_exact_rs,
  rs_makespan, hint_makespan, cpsat_makespan, above_rs and above_hint. The
  stability sweep ignores --hint (it takes --stability_hint instead).

For lag_perturb and breakdown this script keeps its own REPLICATED copy of the
disruption logic of eval_reactive.py (written when eval_reactive parsed its CLI
at import time); the copy is kept line-for-line consistent with
eval_reactive.build_disrupted_state, and both reproduce the earlier numbers.
The four new types import the shared construction instead of replicating it.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

# contention evidence of the time-limited solves (no import-time side effects)
from eval_stochastic_lags import (SolveClock, contention_arrays,
                                  contention_fields, contention_summary,
                                  contention_table, cpsat_timing, loadavg_now,
                                  store_contention, HIT_LIMIT_FRAC,
                                  LOW_CPU_SHARE)


SINGLE_DISRUPTIONS = ('lag_perturb', 'breakdown', 'urgent_arrival',
                      'material_shortage', 'pt_deviation', 'cancellation')
DISRUPTION_CHOICES = SINGLE_DISRUPTIONS + ('both', 'all')


def _disruption_arg(s):
    """--disruption_type: a known choice, or a comma list of single types."""
    if s in DISRUPTION_CHOICES:
        return s
    parts = [p.strip() for p in s.split(',') if p.strip()]
    bad = [p for p in parts if p not in SINGLE_DISRUPTIONS]
    if not parts or bad:
        raise argparse.ArgumentTypeError(
            f'invalid disruption type(s) {bad or s!r}; choose from '
            f'{", ".join(DISRUPTION_CHOICES)} or a comma list of single types')
    return ','.join(parts)


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
    ap.add_argument('--disruption_type', type=_disruption_arg, default='both',
                    help='one of ' + ', '.join(DISRUPTION_CHOICES)
                         + '; both = lag_perturb + breakdown; all = all six; '
                         'or a comma list of single types')
    ap.add_argument('--hint', type=str, default='none',
                    choices=['none', 'right_shift'],
                    help='right_shift: start each residual CP-SAT solve from '
                         'the right-shifted plan (machine and start of every '
                         'uncommitted op, retimed to the integer residual '
                         'model) as a solution hint; none = cold start')
    ap.add_argument('--disruption_offset', type=float, default=0.4)
    ap.add_argument('--lag_factor', type=float, default=2.0)
    ap.add_argument('--pt_factor', type=float, default=1.5)
    ap.add_argument('--shortage_factor', type=float, default=0.5)
    ap.add_argument('--shortage_basis', type=str, default='op_mean',
                    choices=['op_mean', 'remaining_work'])
    ap.add_argument('--cpsat_budgets', type=str, default='1,5,30')
    ap.add_argument('--cpsat_workers', type=int, default=0,
                    help='CP-SAT num_workers (0 = solver default)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='output directory (default test_results/PPVC/'
                         '<dataset>)')
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
    ap.add_argument('--stability_hint', type=str, default='none',
                    choices=['none', 'right_shift'],
                    help='right_shift: start every solve of the stability '
                         'sweep from the right-shift plan (as --hint '
                         'right_shift does in the main run); none = cold')
    ap.add_argument('--breakdown_hours', type=float, default=None,
                    help='breakdown: outage length in hours (default None = '
                         '0.5 x the mean processing time of the order)')
    ap.add_argument('--state_fp', action='store_true',
                    help='store the per-instance fingerprint of the disrupted '
                         'state (eval_reactive.state_fingerprint) in the npy')
    ap.add_argument('--match_reactive', type=str, default=None,
                    help='eval_reactive npy of the same data/type/options; '
                         'assert per order that the original plan, the '
                         'disruption and SLAB re-planning are identical')
    args = ap.parse_args()
    if getattr(args, 'cpsat_workers', 1) == 0:
        # 0 would start one CP-SAT worker per machine core whatever the affinity mask;
        # use one worker per pinned core so the solve matches its CPU allocation
        args.cpsat_workers = len(os.sched_getaffinity(0))
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
    'framework',
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
    # snapshots written before --framework existed are DANIEL
    configs.framework = snap.get('framework', 'DANIEL')
    from common_utils import check_backbone_tag
    check_backbone_tag(model_name, configs.framework)
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

def _import_eval_reactive():
    """eval_reactive parses its CLI only as a script, so a plain import is safe
    once params has been imported with a clean argv (done at module top)."""
    import eval_reactive
    return eval_reactive


def _dparams(args):
    dp = {'lag_factor': args.lag_factor, 'pt_factor': args.pt_factor,
          'shortage_factor': args.shortage_factor,
          'shortage_basis': args.shortage_basis}
    if getattr(args, 'breakdown_hours', None) is not None:
        dp['breakdown_hours'] = float(args.breakdown_hours)
    return dp


def build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                          use_type_embedding, disruption_type, offset,
                          lag_factor, dparams=None, stem=None):
    """Greedy-commit the first k=ceil(offset*N) ops, then apply the disruption.

    New types (urgent_arrival, material_shortage, pt_deviation, cancellation)
    are built by eval_reactive.build_disrupted_state; the returned dict then
    also carries the realized instance ('jl', 'pt', 'time_lag'), 'op_map',
    'op_release', 'mch_release', 'new_ops' and 'legacy' (False). The legacy
    branch below adds the same keys with identity values.

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
    if disruption_type not in ('lag_perturb', 'breakdown'):
        ER = _import_eval_reactive()
        dp = dict(dparams or {})
        dp['lag_factor'] = lag_factor
        ictx = ER.instance_context(stem, meta) if stem is not None else None
        ds = ER.build_disrupted_state(ppo, jl, pt, meta, use_lag_features,
                                      use_type_embedding, disruption_type,
                                      offset, dp, ictx)
        env, state = ds.pop('env'), ds.pop('state')
        ds['assigned_mch_prefix'] = ds['assigned_mch'].copy()
        return ds, env, state

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
        breakdown_hours = (dparams or {}).get('breakdown_hours')
        if breakdown_hours is None:
            downtime_true = 0.5 * mean_pt_true
        else:
            downtime_true = float(breakdown_hours)
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
        # realized-instance keys shared with the new disruption types
        'jl': np.asarray(jl, dtype=int), 'pt': np.asarray(pt),
        'time_lag': perturbed_time_lag, 'op_map': np.arange(n_ops),
        'op_release': np.zeros(n_ops, dtype=float),
        'mch_release': (np.zeros(M, dtype=float) if mch_ready_time is None
                        else mch_ready_time.copy()),
        'new_ops': [], 'legacy': True,
    }, env, env.state


def assert_same_state_as_reactive(ppo, jl, pt, meta, use_lag_features,
                                  use_type_embedding, disruption_type, args,
                                  stem, disrupt, env):
    """Assert that this script's disrupted state equals the state
    eval_reactive.build_disrupted_state builds from the same order and options
    (state_fingerprint, env included; call before any arm steps env).
    Returns the fingerprint."""
    ER = _import_eval_reactive()
    dp = dict(_dparams(args))
    dp['lag_factor'] = args.lag_factor
    ds_er = ER.build_disrupted_state(
        ppo, jl, pt, meta, use_lag_features, use_type_embedding,
        disruption_type, args.disruption_offset, dp,
        ER.instance_context(stem, meta))
    fp_er = ER.state_fingerprint(ds_er, ds_er['env'])
    fp = ER.state_fingerprint(disrupt, env)
    assert fp == fp_er, (f'{os.path.basename(stem)}: disrupted state differs '
                         f'from eval_reactive.build_disrupted_state '
                         f'({disrupt["info"]} vs {ds_er["info"]})')
    return fp


class ReactiveMatch:
    """Per-order identity check against an eval_reactive.py result file."""

    def __init__(self, path, disruption_type, dataset, model_name):
        d = np.load(path, allow_pickle=True).item()
        m = d['meta']
        assert m['disruption_type'] == disruption_type, \
            f'{path}: disruption {m["disruption_type"]} != {disruption_type}'
        assert m['dataset'] == dataset and m['model_name'] == model_name, \
            f'{path}: dataset/model {m["dataset"]}/{m["model_name"]}'
        self.path = path
        self.orig = np.asarray(d['orig_makespans'], dtype=float)
        self.info = list(d['info'])
        self.drl = d['records']['drl_reactive']
        self.fp = d.get('state_fp')
        self.n_checked = 0

    def check(self, i, base, ms_orig, info, fp, ms_d, nerv3):
        """nerv3 = (mch_nerv, time_nerv, comb_nerv) of SLAB's re-plan."""
        where = f'{base} vs {os.path.basename(self.path)}'
        assert i < len(self.orig), f'{where}: order {i} not in the file'
        assert ms_orig == self.orig[i], \
            f'{where}: original SLAB plan {ms_orig} != {self.orig[i]}'
        assert info == self.info[i], f'{where}: {info} != {self.info[i]}'
        if self.fp is not None:
            assert fp == self.fp[i], f'{where}: state fingerprint differs'
        assert ms_d == float(self.drl['makespan'][i]), \
            f'{where}: SLAB re-plan {ms_d} != {self.drl["makespan"][i]}'
        for k, v in zip(('mch_nerv', 'time_nerv', 'comb_nerv'), nerv3):
            assert float(v) == float(self.drl[k][i]), \
                f'{where}: SLAB {k} {v} != {self.drl[k][i]}'
        self.n_checked += 1


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


def right_shift_hint(jl, pt, perturbed_time_lag, disrupt, asg_rs, oct_rs):
    """Turn the right-shift plan into a solution of the INTEGER residual model.

    The right-shift plan fixes, for every uncommitted op, its machine
    (asg_rs) and its position in that machine's queue (order of its
    right-shift start). This function keeps both and gives every uncommitted
    op the earliest start the residual CP-SAT model of residual_cpsat_solve
    allows in that order: start >= round(t_d), the rounded lag after the job's
    previous op (committed or not), the integer end of the previous op on the
    machine (committed ops first, at their fixed integer times), the op's
    ceil(release) (material_shortage, urgent_arrival) and, on a broken
    machine, the end of the reserved outage. When the right-shift times are
    already integer and satisfy those constraints, the result equals
    oct_rs - pt exactly ('exact' is True); otherwise it is the same queue
    order retimed (rounding of a fractional downtime or release, or a legacy
    right-shift that moved a committed op).

    Returns dict(assigned_mch int[N], start int[N] (committed: fixed start),
    end int[N], makespan int, exact bool, ok bool, reason str).
    """
    n_ops, M = pt.shape
    true_pt = np.asarray(pt, dtype=int)
    jl = np.asarray(jl, dtype=int)
    committed = np.asarray(disrupt['committed'], dtype=bool)
    committed_mch = disrupt['committed_mch']
    t_d = int(round(disrupt['t_disrupt_true']))
    lag = np.rint(np.asarray(perturbed_time_lag, dtype=float)).astype(int)
    asg = np.asarray(asg_rs, dtype=int).copy()
    oct_rs = np.asarray(oct_rs, dtype=float)
    first_of_job = np.zeros(n_ops, dtype=bool)
    idx = 0
    for L in jl:
        first_of_job[idx] = True
        idx += int(L)

    start = np.full(n_ops, -1, dtype=np.int64)
    end = np.full(n_ops, -1, dtype=np.int64)
    tail = np.zeros(M, dtype=np.int64)
    for op in np.where(committed)[0]:
        m = int(committed_mch[op])
        s_val = max(int(round(disrupt['committed_start'][op])), 0)
        e_val = int(round(disrupt['committed_ct'][op]))
        if e_val <= s_val:
            e_val = s_val + max(1, int(true_pt[op, m]))
        start[op], end[op] = s_val, e_val
        asg[op] = m
        tail[m] = max(tail[m], e_val)
    # release floors exactly as the model encodes them
    rel = {}
    op_release = disrupt.get('op_release')
    if op_release is not None:
        for op in range(n_ops):
            if not committed[op] and float(op_release[op]) > t_d:
                rel[op] = int(math.ceil(float(op_release[op]) - 1e-9))
    # breakdown outage: new work on the broken machine waits for its end
    mfloor = np.zeros(M, dtype=np.int64)
    bm = disrupt.get('broken_mch')
    if disrupt.get('downtime_true') is not None and bm is not None:
        dt = int(round(disrupt['downtime_true']))
        if dt > 0:
            mfloor[bm] = max(t_d + dt, int(tail[bm]))

    unc = [op for op in range(n_ops) if not committed[op]]
    for op in unc:
        m = int(asg[op])
        if m < 0 or true_pt[op, m] <= 0:
            return {'ok': False, 'reason': f'op {op}: machine {m} not able',
                    'exact': False}
    rs_start = {op: float(oct_rs[op]) - float(pt[op, asg[op]]) for op in unc}
    order = sorted(unc, key=lambda op: (rs_start[op], op))
    for op in order:
        m = int(asg[op])
        s = t_d
        if not first_of_job[op]:
            p = op - 1
            if end[p] < 0:
                return {'ok': False, 'exact': False,
                        'reason': f'op {op} queued before its predecessor'}
            s = max(s, int(end[p]) + int(lag[p]))
        s = max(s, int(tail[m]), int(mfloor[m]), rel.get(op, 0))
        start[op] = s
        end[op] = s + int(true_pt[op, m])
        tail[m] = end[op]
    exact = all(abs(float(start[op]) - rs_start[op]) < 1e-6 for op in unc)
    return {'ok': True, 'reason': '', 'exact': bool(exact),
            'assigned_mch': asg, 'start': start, 'end': end,
            'makespan': int(end.max())}


# ---------------------------------------------------------------------------
# THE CORE DELIVERABLE: prefix-respecting residual CP-SAT model.
# ---------------------------------------------------------------------------

def residual_cpsat_solve(jl, pt, perturbed_time_lag, disrupt, budget,
                         mk_coef=1, pen_coef=0, assigned_orig=None,
                         starts_orig=None, num_workers=0, hint=None,
                         hint_check=True):
    """Re-solve the RESIDUAL with CP-SAT honoring the committed prefix.

    jl / pt / perturbed_time_lag describe the REALIZED instance (for the new
    disruption types: disrupt['jl'], disrupt['pt'], disrupt['time_lag']).
    disrupt['op_release'] (optional) adds start >= ceil(release) per
    uncommitted op (material_shortage). assigned_orig / starts_orig are in the
    realized indexing; an entry assigned_orig[op] < 0 (an op added by
    urgent_arrival) carries no stability penalty. num_workers > 0 caps the
    CP-SAT search workers (0 = solver default).

    STABILITY-AWARE OPTION: when
    ``pen_coef > 0`` the objective becomes ``mk_coef*makespan +
    pen_coef*nervousness``, where nervousness counts, per UNCOMMITTED op,
    a machine reassignment and a start-shift (>1 h) relative to the
    ORIGINAL pre-disruption plan (``assigned_orig``/``starts_orig``) -- the
    SAME quantity the nervousness metric measures. The effective weight is
    ``w = pen_coef / mk_coef`` hours per nervousness unit. With the default
    ``pen_coef=0`` the objective and every variable are IDENTICAL to the
    makespan-only model, so existing callers are bit-unchanged.

    WARM START (``hint``): a dict from right_shift_hint() with the integer
    machine 'assigned_mch' and 'start' of every op. Every variable of every
    UNCOMMITTED op (start, end, duration, machine presence literals and the
    optional per-machine intervals) and the makespan get model.AddHint; the
    committed ops are fixed already. With ``hint_check`` the hint is first
    verified by a separate, untimed solve with
    fix_variables_to_their_hinted_value; a hint that this check does not find
    feasible is cleared and the timed solve starts cold. The timed solve's
    parameters are unchanged. hint=None (default) adds nothing.

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
      t_disrupt_int (int), downtime_int (None|int), broken_mch (None|int),
      timing (dict) -- the solve's contention evidence (wall and CPU seconds,
      workers, cpu_share, status, hit_limit; see eval_stochastic_lags).
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
    op_release = disrupt.get('op_release')
    release_int = {}
    if op_release is not None:
        for op in range(n_ops):
            if not committed[op] and float(op_release[op]) > t_disrupt_int:
                release_int[op] = int(math.ceil(float(op_release[op]) - 1e-9))
        if release_int:
            horizon += max(release_int.values()) - t_disrupt_int
    horizon = max(horizon, 1)

    model = cp_model.CpModel()
    intervals_per_machine = {m: [] for m in range(M)}
    starts = {}   # op -> start IntVar
    ends = {}     # op -> end IntVar
    presences = {}  # (op, m) -> BoolVar (for uncommitted ops over machines)
    durs = {}       # op -> duration IntVar (uncommitted ops)
    alts = {}       # (op, m) -> (l_start, l_end, l_dur) of the optional interval
    fixed_end = {}  # committed op -> its fixed integer end

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
            fixed_end[op] = e_val
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
            durs[op] = dur
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
                    alts[(op, m)] = (l_start, l_end, l_dur)
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

    # --- per-op release floors (material_shortage) ---------------------------
    for op, r in release_int.items():
        model.Add(starts[op] >= r)

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
    pen_vars = []   # (op, dev, shifted, orig_s) of the penalized model
    if pen_coef and pen_coef > 0:
        # Stability-aware objective: penalize, per UNCOMMITTED op, machine
        # reassignment and start-shift (>1 h) vs the ORIGINAL plan, i.e. the
        # exact nervousness the metric measures.  w = pen_coef / mk_coef.
        assert assigned_orig is not None and starts_orig is not None, \
            'penalized solve needs assigned_orig and starts_orig'
        pen_terms = []
        for op in range(n_ops):
            if committed[op] or int(assigned_orig[op]) < 0:
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
            pen_vars.append((op, dev, shifted, orig_s))
        model.Minimize(int(mk_coef) * makespan + int(pen_coef) * sum(pen_terms))
    else:
        model.Minimize(makespan)

    # --- optional warm start from a feasible plan (e.g. the right-shift) ----
    hint_given = False
    hint_check_status = None
    hint_check_seconds = float('nan')
    hint_makespan = float('nan')
    if hint is not None:
        h_asg = np.asarray(hint['assigned_mch'], dtype=int)
        h_start = np.asarray(hint['start'], dtype=np.int64)
        h_end_all = []
        for op in range(n_ops):
            if committed[op]:
                h_end_all.append(fixed_end[op])
                continue
            m_h = int(h_asg[op])
            s_h = int(h_start[op])
            d_h = int(true_pt[op, m_h])
            model.AddHint(starts[op], s_h)
            model.AddHint(ends[op], s_h + d_h)
            model.AddHint(durs[op], d_h)
            h_end_all.append(s_h + d_h)
            able = np.where(true_pt[op] > 0)[0]
            if len(able) > 1:
                for m in able:
                    m = int(m)
                    l_start, l_end, l_dur = alts[(op, m)]
                    on = (m == m_h)
                    model.AddHint(presences[(op, m)], 1 if on else 0)
                    # an absent optional interval is unconstrained; give it
                    # the same start, clamped to its domain
                    ls = min(max(s_h, t_disrupt_int), horizon)
                    model.AddHint(l_start, s_h if on else ls)
                    model.AddHint(l_end, s_h + d_h if on
                                  else min(ls + l_dur, horizon))
        hint_makespan = float(max(h_end_all))
        model.AddHint(makespan, int(hint_makespan))
        # penalized model: the hint also sets each op's start deviation and
        # shift indicator, so it is a complete solution
        for op, dev, shifted, orig_s in pen_vars:
            d_h = abs(int(h_start[op]) - int(orig_s))
            model.AddHint(dev, d_h)
            model.AddHint(shifted, 1 if d_h >= 2 else 0)
        hint_given = True
        if hint_check:
            chk = cp_model.CpSolver()
            chk.parameters.fix_variables_to_their_hinted_value = True
            chk.parameters.max_time_in_seconds = 30.0
            if num_workers:
                chk.parameters.num_workers = int(num_workers)
            t_chk = time.time()
            st = chk.Solve(model)
            hint_check_seconds = time.time() - t_chk
            hint_check_status = chk.StatusName(st)
            ok_st = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            if pen_vars:
                # penalized objective: check the makespan of the hint plan
                ok_val = ok_st and abs(chk.Value(makespan)
                                       - hint_makespan) <= 1e-6
            else:
                ok_val = ok_st and abs(chk.ObjectiveValue()
                                       - hint_makespan) <= 1e-6
            if not ok_val:
                model.ClearHints()
                hint_given = False

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget)
    if num_workers:
        solver.parameters.num_workers = int(num_workers)
    with SolveClock() as clk:
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
        'timing': cpsat_timing(solver, status, clk),
        'hint_given': hint_given,
        'hint_check_status': hint_check_status,
        'hint_check_seconds': hint_check_seconds,
        'hint_makespan': hint_makespan,
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
               'abs_shift': np.full(n, np.nan),
               'feasible': np.zeros(n, dtype=bool)}
           for m in methods}
    # contention evidence per (instance, method); the CP-SAT arms also keep
    # their single solve's wall/CPU seconds, workers, cpu_share and status
    for m in methods:
        rec[m].update(contention_arrays(n, per_solve=m in cpsat_names))
    use_hint = getattr(args, 'hint', 'none') == 'right_shift'
    if use_hint:
        # warm-start evidence per (instance, CP-SAT arm)
        for m in cpsat_names:
            rec[m].update({
                'hint_given': np.zeros(n, dtype=bool),
                'hint_exact_rs': np.zeros(n, dtype=bool),
                'hint_check_status': np.array([''] * n, dtype=object),
                'hint_check_s': np.full(n, np.nan),
                'rs_makespan': np.full(n, np.nan),
                'rs_residual_makespan': np.full(n, np.nan),
                'hint_makespan': np.full(n, np.nan),
                'cpsat_makespan': np.full(n, np.nan),
                'above_rs': np.zeros(n, dtype=bool),
                'above_hint': np.zeros(n, dtype=bool)})
    n_hint_failed = 0   # instances solved cold: hint could not be built/checked
    state_fps = []
    match = (ReactiveMatch(args.match_reactive, disruption_type, dataset,
                           args.model_name)
             if getattr(args, 'match_reactive', None) else None)
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
          f'lag_factor={args.lag_factor}  budgets(s)={budgets}'
          + (f'  hint={args.hint}' if getattr(args, 'hint', 'none') != 'none'
             else '')
          + (f'  breakdown_hours={args.breakdown_hours}'
             if getattr(args, 'breakdown_hours', None) is not None else ''))
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
            disruption_type, args.disruption_offset, args.lag_factor,
            dparams=_dparams(args), stem=stem)
        perturbed_lag = disrupt['perturbed_time_lag']
        mch_ready_time = disrupt['mch_ready_time']
        # realized instance (identical to the loaded one for lag_perturb and
        # breakdown except for the perturbed lag)
        jl_r, pt_r = disrupt['jl'], disrupt['pt']
        legacy = disrupt['legacy']
        ER = _import_eval_reactive()
        fp_i = None
        if getattr(args, 'breakdown_hours', None) is not None and legacy:
            fp_i = assert_same_state_as_reactive(
                ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                disruption_type, args, stem, disrupt, env)
        elif getattr(args, 'state_fp', False) or match is not None:
            fp_i = ER.state_fingerprint(disrupt, env)
        if getattr(args, 'state_fp', False):
            state_fps.append(fp_i)

        def _nerv(asg_new, oct_new):
            return ER.nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                         asg_new, oct_new, disrupt['op_map'])

        def _contract(name, asg_new, oct_new):
            nonlocal n_prefix_violations
            if legacy:
                return True
            viol = ER.check_residual_contract(disrupt, asg_new, oct_new)
            if viol:
                n_prefix_violations += 1
                print(f'  !!! {name} RESIDUAL-CONTRACT VIOLATION on {base}: '
                      f'{viol[:3]}')
            return not viol

        # ---- (a) DRL-reactive: continue the SAME disrupted env --------------
        la_d = loadavg_now()
        ms_d, sec_d, asg_d, oct_d = drl_reactive_continue(
            ppo, env, state, disrupt['assigned_mch_prefix'].copy())
        store_contention(rec['drl_reactive'], i, [], la_d)
        res_d = validate_schedule(jl_r, pt_r, perturbed_lag, asg_d, oct_d)
        if not res_d['feasible']:
            n_infeasible += 1
            print(f'  !!! drl_reactive INFEASIBLE on {base}: '
                  f'{res_d["violations"][:3]}')
        ok_d = _contract('drl_reactive', asg_d, oct_d)
        mn, tn, cn, sh = _nerv(asg_d, oct_d)
        rec['drl_reactive']['makespan'][i] = ms_d
        rec['drl_reactive']['inflation'][i] = (
            100.0 * (ms_d - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['drl_reactive']['seconds'][i] = sec_d
        rec['drl_reactive']['mch_nerv'][i] = mn
        rec['drl_reactive']['time_nerv'][i] = tn
        rec['drl_reactive']['comb_nerv'][i] = cn
        rec['drl_reactive']['abs_shift'][i] = sh
        rec['drl_reactive']['feasible'][i] = res_d['feasible'] and ok_d
        if match is not None:
            match.check(i, base, ms_orig, disrupt['info'], fp_i, ms_d,
                        (mn, tn, cn))

        # ---- (b) right-shift repair -----------------------------------------
        store_contention(rec['right_shift'], i, [], loadavg_now())
        if legacy:
            ms_rs, sec_rs, asg_rs, oct_rs = right_shift_baseline(
                jl, pt, assigned_orig, oct_orig, perturbed_lag, mch_ready_time)
        else:
            ms_rs, sec_rs, asg_rs, oct_rs = ER.right_shift_residual(
                disrupt, assigned_orig, oct_orig, pt)
        res_rs = validate_schedule(jl_r, pt_r, perturbed_lag, asg_rs, oct_rs)
        if not res_rs['feasible']:
            n_infeasible += 1
            print(f'  !!! right_shift INFEASIBLE on {base}: '
                  f'{res_rs["violations"][:3]}')
        ok_rs = _contract('right_shift', asg_rs, oct_rs)
        mn, tn, cn, sh = _nerv(asg_rs, oct_rs)
        rec['right_shift']['makespan'][i] = ms_rs
        rec['right_shift']['inflation'][i] = (
            100.0 * (ms_rs - ms_orig) / ms_orig if ms_orig > 0 else 0.0)
        rec['right_shift']['seconds'][i] = sec_rs
        rec['right_shift']['mch_nerv'][i] = mn
        rec['right_shift']['time_nerv'][i] = tn
        rec['right_shift']['comb_nerv'][i] = cn
        rec['right_shift']['abs_shift'][i] = sh
        rec['right_shift']['feasible'][i] = res_rs['feasible'] and ok_rs

        # ---- (c) PREFIX-RESPECTING RESIDUAL CP-SAT at each budget -----------
        hint = None
        ms_rs_res = np.nan
        if use_hint:
            # prefix-respecting right-shift (eval_reactive.right_shift_residual)
            # for reference: for breakdown the legacy right-shift above also
            # delays committed ops queued on the broken machine
            ms_rs_res, _, a_rr, c_rr = ER.right_shift_residual(
                disrupt, assigned_orig, oct_orig, pt)
            v_rr = validate_schedule(jl_r, pt_r, perturbed_lag, a_rr, c_rr)
            if not v_rr['feasible'] or ER.check_residual_contract(
                    disrupt, a_rr, c_rr):
                print(f'  !!! residual right-shift invalid on {base}')
                ms_rs_res = np.nan
            hb = right_shift_hint(jl_r, pt_r, perturbed_lag, disrupt,
                                  asg_rs, oct_rs)
            if hb['ok']:
                hint = hb
            else:
                print(f'  !!! right-shift hint not built on {base}: '
                      f'{hb["reason"]} -> cold solve')
        cp_summ = {}
        for b in budgets:
            mname = f'residual_cpsat_{b}s'
            la_c = loadavg_now()
            res = residual_cpsat_solve(jl_r, pt_r, perturbed_lag, disrupt, b,
                                       num_workers=args.cpsat_workers,
                                       hint=hint)
            store_contention(rec[mname], i, [res['timing']], la_c)
            if use_hint:
                r_h = rec[mname]
                r_h['hint_given'][i] = res['hint_given']
                r_h['hint_exact_rs'][i] = bool(hint is not None
                                               and hint['exact'])
                r_h['hint_check_status'][i] = str(res['hint_check_status'])
                r_h['hint_check_s'][i] = res['hint_check_seconds']
                r_h['rs_makespan'][i] = ms_rs
                r_h['rs_residual_makespan'][i] = ms_rs_res
                r_h['hint_makespan'][i] = (hint['makespan'] if hint is not None
                                           else np.nan)
                if hint is not None and not res['hint_given']:
                    print(f'  !!! {mname} hint REJECTED by the check on {base} '
                          f'({res["hint_check_status"]}) -> cold solve')
                if res['status'] in ('OPTIMAL', 'FEASIBLE'):
                    r_h['cpsat_makespan'][i] = res['objective']
                    r_h['above_rs'][i] = res['objective'] > ms_rs + 1e-6
                    r_h['above_hint'][i] = (
                        res['hint_given']
                        and res['objective'] > res['hint_makespan'] + 1e-6)
                    if r_h['above_rs'][i]:
                        print(f'  !!! {mname} ends ABOVE right-shift on {base}: '
                              f'{res["objective"]:.1f} > {ms_rs:.2f} (hint '
                              f'{r_h["hint_makespan"][i]:.0f}, given '
                              f'{res["hint_given"]})')
            cp_summ[b] = res
            if res['status'] == 'OPTIMAL':
                optimal_count[b] += 1
            elif res['status'] == 'FEASIBLE':
                feasible_only_count[b] += 1
            else:
                no_sol_count[b] += 1

            if res['status'] in ('OPTIMAL', 'FEASIBLE'):
                # independent feasibility validation vs perturbed lags
                vres = validate_schedule(jl_r, pt_r, perturbed_lag,
                                         res['assigned_mch'], res['op_ct'])
                pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                               res['op_ct'])
                if not legacy:
                    pviol = pviol + ER.check_residual_contract(
                        disrupt, res['assigned_mch'], res['op_ct'], tol=0.51)
                bviol = check_breakdown_window(res, pt_r)
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
                mn, tn, cn, sh = _nerv(res['assigned_mch'], res['op_ct'])
                rec[mname]['mch_nerv'][i] = mn
                rec[mname]['time_nerv'][i] = tn
                rec[mname]['comb_nerv'][i] = cn
                rec[mname]['abs_shift'][i] = sh
            else:
                print(f'  !!! {mname} NO SOLUTION ({res["status"]}) on {base}')
                rec[mname]['feasible'][i] = False
            rec[mname]['seconds'][i] = res['seconds']

        # ---- explicit sample-instance validation printout -------------------
        if sample_checks_printed < 2 and budgets:
            b = budgets[-1]  # the largest budget
            res = cp_summ[b]
            if res['status'] in ('OPTIMAL', 'FEASIBLE'):
                vres = validate_schedule(jl_r, pt_r, perturbed_lag,
                                         res['assigned_mch'], res['op_ct'])
                pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                               res['op_ct'])
                bviol = check_breakdown_window(res, pt_r)
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

        if use_hint and (hint is None or not all(
                rec[f'residual_cpsat_{b}s']['hint_given'][i] for b in budgets)):
            n_hint_failed += 1

        # ---- progress line --------------------------------------------------
        cp_str = '  '.join(
            f'rcp{b}={cp_summ[b]["objective"]:6.0f}h({cp_summ[b]["status"][:4]},'
            f'cpu{cp_summ[b]["timing"]["cpu_share"]:.2f})'
            for b in budgets)
        print(f'  [{i + 1:3d}/{n}] {base}  orig={ms_orig:6.0f}h  '
              f'drl={ms_d:6.0f}h(+{rec["drl_reactive"]["inflation"][i]:4.1f}%,'
              f'{sec_d:.2f}s)  rs={ms_rs:6.0f}h(+'
              f'{rec["right_shift"]["inflation"][i]:5.1f}%)  {cp_str}'
              + (f'  hint={hint["makespan"] if hint is not None else "-"}'
                 f'{"(exact)" if hint is not None and hint["exact"] else ""}'
                 if use_hint else '')
              + f'   [{disrupt["info"]}]')

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
            'abs_shift': _mean(rec[m]['abs_shift']),
            'feasible': int(rec[m]['feasible'].sum()),
        }
    contention = {m: contention_summary(rec[m]) for m in methods}

    # ---- save structured npy ------------------------------------------------
    save_dir = args.out_dir or f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    from common_utils import backbone_file_tag
    btag = backbone_file_tag(args.model_name, configs.framework)
    out_npy = os.path.join(save_dir, f'residual_cpsat_{disruption_type}{btag}.npy')
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
            'pt_factor': args.pt_factor,
            'shortage_factor': args.shortage_factor,
            'shortage_basis': args.shortage_basis,
            'cpsat_budgets': budgets,
            'cpsat_workers': args.cpsat_workers,
            'seed_test': args.seed_test,
            'n_instances': n,
            'n_infeasible': n_infeasible,
            'n_prefix_violations': n_prefix_violations,
            'n_breakdown_window_violations': n_breakdown_window_violations,
            'low_cpu_share': LOW_CPU_SHARE,
            'hit_limit_frac': HIT_LIMIT_FRAC,
        },
        'contention': contention,
    }
    # new fields only when their option is used, so a default run writes
    # exactly the keys it wrote before
    if getattr(args, 'breakdown_hours', None) is not None:
        payload['meta']['breakdown_hours'] = float(args.breakdown_hours)
        payload['meta']['state_checked_against_eval_reactive'] = bool(
            disruption_type in ('lag_perturb', 'breakdown'))
    if getattr(args, 'state_fp', False):
        payload['state_fp'] = state_fps
    if match is not None:
        payload['meta']['match_reactive'] = args.match_reactive
        payload['meta']['n_matched_reactive'] = match.n_checked
        payload['meta']['torch_threads'] = int(torch_threads())
    hint_stats = None
    if use_hint:
        hint_stats = {}
        for m in cpsat_names:
            r = rec[m]
            hint_stats[m] = {
                'n_hint_given': int(r['hint_given'].sum()),
                'n_hint_exact_rs': int(r['hint_exact_rs'].sum()),
                'n_above_rs': int(r['above_rs'].sum()),
                'n_above_hint': int(r['above_hint'].sum()),
                'mean_rs_makespan': _mean(r['rs_makespan']),
                'mean_rs_residual_makespan': _mean(r['rs_residual_makespan']),
                'mean_hint_makespan': _mean(r['hint_makespan']),
                'mean_cpsat_makespan': _mean(r['cpsat_makespan'])}
        payload['meta']['hint'] = 'right_shift'
        payload['meta']['n_hint_failed'] = n_hint_failed
        payload['hint_stats'] = hint_stats
    np.save(out_npy, payload, allow_pickle=True)

    # ---- console summary ----------------------------------------------------
    print('\n' + '=' * 78)
    print(f'RESIDUAL CP-SAT SUMMARY  ({dataset}, {n} instances, '
          f'{disruption_type})')
    print('=' * 78)
    print(f'  mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    print(f'{"method":<20}{"makespan":>10}{"infl%":>9}{"time(s)":>10}'
          f'{"mNerv":>8}{"tNerv":>8}{"cNerv":>8}{"shift":>8}{"feas":>9}')
    for m in methods:
        a = agg[m]
        print(f'{m:<20}{a["makespan"]:>10.1f}{a["inflation"]:>9.2f}'
              f'{a["seconds"]:>10.3f}{a["mch_nerv"]:>8.1f}'
              f'{a["time_nerv"]:>8.1f}{a["comb_nerv"]:>8.1f}'
              f'{a["abs_shift"]:>8.2f}{a["feasible"]:>6}/{n}')
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
    for m in cpsat_names:
        c = contention[m]
        print(f'  contention {m}: min cpu_share '
              f'{c["cpu_share_min"]}  budget-limited {c["n_hit_limit"]}/'
              f'{c["n_timed_solves"]}  cpu_share<{LOW_CPU_SHARE} '
              f'{c["n_low_share"]}')
    if use_hint:
        print(f'  right-shift hint: instances solved cold = {n_hint_failed}')
        for m in cpsat_names:
            h = hint_stats[m]
            print(f'  hint {m}: given {h["n_hint_given"]}/{n}  exact right-'
                  f'shift times {h["n_hint_exact_rs"]}/{n}  above right-shift '
                  f'{h["n_above_rs"]}  above hint {h["n_above_hint"]}  mean '
                  f'rs {h["mean_rs_makespan"]:.2f} hint '
                  f'{h["mean_hint_makespan"]:.2f} cpsat '
                  f'{h["mean_cpsat_makespan"]:.2f}')
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
                 f'lag_factor={args.lag_factor}  pt_factor={args.pt_factor}  '
                 f'shortage={args.shortage_factor} x {args.shortage_basis}'
                 + (f'  breakdown_hours={args.breakdown_hours}'
                    if getattr(args, 'breakdown_hours', None) is not None
                    else ''))
    lines.append(f'- budgets (s): {budgets}  (num_workers='
                 f'{args.cpsat_workers or "default"})')
    if use_hint:
        lines.append('- warm start: right-shift plan as CP-SAT solution hint '
                     f'(instances solved cold: {n_hint_failed})')
    lines.append(f'- mean ORIGINAL DRL makespan: {_mean(orig_makespans):.1f} h')
    lines.append(f'- total validator-infeasibilities: {n_infeasible}')
    lines.append(f'- total prefix violations: {n_prefix_violations}')
    lines.append(f'- total breakdown-window violations: '
                 f'{n_breakdown_window_violations}\n')
    lines.append('| method | mean makespan (h) | mean inflation % | '
                 'mean time (s) | mch-nerv | time-nerv | comb-nerv | '
                 'abs-shift (h) | feasible |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for m in methods:
        a = agg[m]
        lines.append(f'| {m} | {a["makespan"]:.1f} | {a["inflation"]:.2f} | '
                     f'{a["seconds"]:.4f} | {a["mch_nerv"]:.1f} | '
                     f'{a["time_nerv"]:.1f} | {a["comb_nerv"]:.1f} | '
                     f'{a["abs_shift"]:.2f} | {a["feasible"]}/{n} |')
    lines.append('')
    lines.append('| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |')
    lines.append('|---|---|---|---|')
    for b in budgets:
        lines.append(f'| {b} | {optimal_count[b]} | '
                     f'{feasible_only_count[b]} | {no_sol_count[b]} |')
    lines.append('')
    lines += contention_table({m: contention[m] for m in cpsat_names})
    if use_hint:
        lines.append('')
        lines.append('Warm start: the hint is the right-shift plan retimed to '
                     'the integer residual model; "above" counts solves whose '
                     'makespan exceeds the right-shift makespan or the hint '
                     'makespan.')
        lines.append('')
        lines.append('| method | hint given | hint = right-shift times | '
                     'mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | '
                     'above right-shift | above hint |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for m in cpsat_names:
            h = hint_stats[m]
            lines.append(f'| {m} | {h["n_hint_given"]}/{n} | '
                         f'{h["n_hint_exact_rs"]}/{n} | '
                         f'{h["mean_rs_makespan"]:.2f} | '
                         f'{h["mean_hint_makespan"]:.2f} | '
                         f'{h["mean_cpsat_makespan"]:.2f} | '
                         f'{h["n_above_rs"]} | {h["n_above_hint"]} |')
    md_path = os.path.join(save_dir,
                           f'residual_cpsat_{disruption_type}{btag}_summary.md')
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
        'contention': contention,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def torch_threads():
    import torch
    return torch.get_num_threads()


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
    # contention evidence of every penalized solve, one record per
    # (instance, weight), including solves whose result is then discarded
    T = {w: [] for w in weights}
    orig_ms = []
    # per-order detail (NaN / '' where a solve gave no valid schedule):
    # makespan, objective, status and warm-start flag of every solve, and the
    # SLAB makespan; kept outside 'frontier' / 'drl', whose form is fixed
    X = {w: {'ms': [], 'objective': [], 'status': [], 'hint_given': [],
             'valid': []} for w in weights}
    drl_ms, hint_ms, state_fps = [], [], []
    use_hint = getattr(args, 'stability_hint', 'none') == 'right_shift'
    n_hint_failed = 0
    match = (ReactiveMatch(args.match_reactive, disruption_type, dataset,
                           args.model_name)
             if getattr(args, 'match_reactive', None) else None)

    print('=' * 78)
    print(f'STABILITY SWEEP  disruption={disruption_type}  n={n}  '
          f'budget={budget}s'
          + ('  hint=right_shift' if use_hint else '')
          + (f'  breakdown_hours={args.breakdown_hours}'
             if getattr(args, 'breakdown_hours', None) is not None else ''))
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
            disruption_type, args.disruption_offset, args.lag_factor,
            dparams=_dparams(args), stem=stem)
        perturbed_lag = disrupt['perturbed_time_lag']
        jl_r, pt_r = disrupt['jl'], disrupt['pt']
        ER = _import_eval_reactive()
        # original plan in the realized indexing (-1 / NaN for added ops)
        op_map = np.asarray(disrupt['op_map'], dtype=int)
        n_r = pt_r.shape[0]
        asg_orig_r = np.full(n_r, -1, dtype=int)
        starts_orig_r = np.full(n_r, np.nan)
        keep = op_map >= 0
        asg_orig_r[op_map[keep]] = assigned_orig[keep]
        starts_orig_r[op_map[keep]] = starts_orig[keep]

        def _nerv(asg_new, oct_new):
            return ER.nervousness_mapped(pt, assigned_orig, oct_orig, pt_r,
                                         asg_new, oct_new, op_map)[:3]

        if getattr(args, 'breakdown_hours', None) is not None and \
                disrupt['legacy']:
            fp_i = assert_same_state_as_reactive(
                ppo, jl, pt, meta, use_lag_features, use_type_embedding,
                disruption_type, args, stem, disrupt, env)
        else:
            fp_i = ER.state_fingerprint(disrupt, env)
        state_fps.append(fp_i)

        # ---- warm start: the right-shift plan, built as the main run builds
        # it (legacy operator for lag_perturb / breakdown, the prefix-
        # respecting residual one otherwise), retimed to the integer model
        hint = None
        if use_hint:
            if disrupt['legacy']:
                _ms, _s2, asg_rs, oct_rs = right_shift_baseline(
                    jl, pt, assigned_orig, oct_orig, perturbed_lag,
                    disrupt['mch_ready_time'])
            else:
                _ms, _s2, asg_rs, oct_rs = ER.right_shift_residual(
                    disrupt, assigned_orig, oct_orig, pt)
            hb = right_shift_hint(jl_r, pt_r, perturbed_lag, disrupt,
                                  asg_rs, oct_rs)
            if hb['ok']:
                hint = hb
            else:
                print(f'  !!! right-shift hint not built on {base}: '
                      f'{hb["reason"]} -> cold solves')
        hint_ms.append(float(hint['makespan']) if hint is not None
                       else float('nan'))
        hint_failed_i = use_hint and hint is None

        # ---- DRL-reactive operating point -----------------------------------
        ms_d, sec_d, asg_d, oct_d = drl_reactive_continue(
            ppo, env, state, disrupt['assigned_mch_prefix'].copy())
        drl_ms.append(float(ms_d))
        rd = validate_schedule(jl_r, pt_r, perturbed_lag, asg_d, oct_d)
        if rd['feasible']:
            drl['feas'] += 1
        else:
            print(f'  !!! drl_reactive INFEASIBLE on {base}')
        mn, tn, cn = _nerv(asg_d, oct_d)
        if match is not None:
            match.check(i, base, ms_orig, disrupt['info'], fp_i, ms_d,
                        (mn, tn, cn))
        drl['infl'].append(100.0 * (ms_d - ms_orig) / ms_orig)
        drl['nerv'].append(cn); drl['mch_nerv'].append(mn)
        drl['time_nerv'].append(tn); drl['sec'].append(sec_d)

        # ---- penalized residual CP-SAT at each weight -----------------------
        for w, (mk, pc) in zip(weights, coefs):
            la_w = loadavg_now()
            if pc == 0:
                res = residual_cpsat_solve(jl_r, pt_r, perturbed_lag, disrupt,
                                           budget,
                                           num_workers=args.cpsat_workers,
                                           hint=hint)
            else:
                res = residual_cpsat_solve(jl_r, pt_r, perturbed_lag, disrupt,
                                           budget, mk_coef=mk, pen_coef=pc,
                                           assigned_orig=asg_orig_r,
                                           starts_orig=starts_orig_r,
                                           num_workers=args.cpsat_workers,
                                           hint=hint)
            T[w].append({'instance': base, **res['timing'],
                         **contention_fields([res['timing']], la_w)})
            X[w]['status'].append(res['status'])
            X[w]['hint_given'].append(bool(res['hint_given']))
            X[w]['objective'].append(float(res['objective']))
            if hint is not None and not res['hint_given']:
                hint_failed_i = True
                print(f'  !!! w={w} hint REJECTED by the check on {base} '
                      f'({res["hint_check_status"]}) -> cold solve')
            if res['status'] not in ('OPTIMAL', 'FEASIBLE'):
                print(f'  !!! w={w} NO SOLUTION on {base} ({res["status"]})')
                X[w]['ms'].append(float('nan'))
                X[w]['valid'].append(False)
                continue
            ms_new = float(np.max(res['op_ct']))
            vres = validate_schedule(jl_r, pt_r, perturbed_lag,
                                     res['assigned_mch'], res['op_ct'])
            pviol = check_prefix_respected(disrupt, res['assigned_mch'],
                                           res['op_ct'])
            if not disrupt['legacy']:
                pviol = pviol + ER.check_residual_contract(
                    disrupt, res['assigned_mch'], res['op_ct'], tol=0.51)
            bviol = check_breakdown_window(res, pt_r)
            if bviol:
                print(f'  !!! w={w} BREAKDOWN-WINDOW VIOLATION on {base}: '
                      f'{bviol[:3]}')
            if not vres['feasible'] or pviol or (bviol and getattr(
                    args, 'stability_hint', 'none') != 'none'):
                print(f'  !!! w={w} INVALID on {base}: feas={vres["feasible"]} '
                      f'prefix_viol={len(pviol)}')
                X[w]['ms'].append(float('nan'))
                X[w]['valid'].append(False)
                continue
            X[w]['ms'].append(ms_new)
            X[w]['valid'].append(True)
            mn, tn, cn = _nerv(res['assigned_mch'], res['op_ct'])
            W[w]['infl'].append(100.0 * (ms_new - ms_orig) / ms_orig)
            W[w]['nerv'].append(cn); W[w]['mch_nerv'].append(mn)
            W[w]['time_nerv'].append(tn); W[w]['sec'].append(res['seconds'])
            W[w]['feas'] += 1
            if res['status'] == 'OPTIMAL':
                W[w]['optimal'] += 1

        if hint_failed_i:
            n_hint_failed += 1
        if use_hint or getattr(args, 'breakdown_hours', None) is not None:
            print(f'  [{i + 1:3d}/{n}] {base}  orig={ms_orig:6.0f}h  '
                  f'drl={ms_d:6.0f}h  hint={hint_ms[-1]:6.0f}h  '
                  + '  '.join(f'w{w}={X[w]["ms"][-1]:6.0f}h('
                              f'{X[w]["status"][-1][:4]},'
                              f'{T[w][-1]["solve_wall_s"]:.1f}s,'
                              f'cpu{T[w][-1]["cpu_share"]:.2f})'
                              for w in weights)
                  + f'   [{disrupt["info"]}]', flush=True)
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
    contention = {w: contention_summary(T[w]) for w in weights}
    for w in weights:
        c = contention[w]
        print(f'  contention w={w}: min cpu_share {c["cpu_share_min"]}  '
              f'budget-limited {c["n_hit_limit"]}/{c["n_timed_solves"]}  '
              f'cpu_share<{LOW_CPU_SHARE} {c["n_low_share"]}')

    payload = {
        'disruption_type': disruption_type, 'n': n, 'weights': weights,
        'coefs': coefs, 'budget': budget, 'orig_mean': _m(orig_ms),
        'drl': {k: (drl[k] if k == 'feas' else [float(x) for x in drl[k]])
                for k in drl},
        'frontier': {w: {k: (W[w][k] if k in ('optimal', 'feas')
                             else [float(x) for x in W[w][k]])
                         for k in W[w]} for w in weights},
        'meta': {'offset': args.disruption_offset,
                 'lag_factor': args.lag_factor, 'seed_test': args.seed_test,
                 'cpsat_workers': args.cpsat_workers,
                 'low_cpu_share': LOW_CPU_SHARE,
                 'hit_limit_frac': HIT_LIMIT_FRAC},
        'timing': T,
        'contention': contention,
    }
    # new fields only when their option is used, so a default run writes
    # exactly the keys it wrote before
    if use_hint or getattr(args, 'breakdown_hours', None) is not None or \
            getattr(args, 'state_fp', False):
        payload['per_order'] = {
            'instance': [os.path.basename(s) for s in stems],
            'orig_ms': [float(x) for x in orig_ms],
            'drl_ms': drl_ms, 'hint_ms': hint_ms,
            'frontier': {w: X[w] for w in weights}}
        payload['state_fp'] = state_fps
        payload['meta']['dataset'] = dataset
        payload['meta']['model_name'] = args.model_name
        payload['meta']['stability_hint'] = getattr(args, 'stability_hint',
                                                    'none')
        payload['meta']['n_hint_failed'] = n_hint_failed
    if getattr(args, 'breakdown_hours', None) is not None:
        payload['meta']['breakdown_hours'] = float(args.breakdown_hours)
    if match is not None:
        payload['meta']['match_reactive'] = args.match_reactive
        payload['meta']['n_matched_reactive'] = match.n_checked
        payload['meta']['torch_threads'] = int(torch_threads())
    outdir = args.out_dir or f'test_results/PPVC/{dataset}'
    os.makedirs(outdir, exist_ok=True)
    from common_utils import backbone_file_tag
    btag = backbone_file_tag(args.model_name, configs.framework)
    out = f'{outdir}/stability_sweep_{disruption_type}{btag}.npy'
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
    elif args.disruption_type == 'all':
        disruptions = ['lag_perturb', 'breakdown', 'urgent_arrival',
                       'material_shortage', 'pt_deviation', 'cancellation']
    else:
        disruptions = args.disruption_type.split(',')

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
