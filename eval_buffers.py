"""
eval_buffers.py
===============
Finite lag-buffer study for the PPVC time-lag scheduler.

The lag-aware model lets a module wait out its post-operation time-lag
(curing / ponding / paint drying) while the station is free. During that wait
the module sits in a lag buffer (curing yard, ponding bay, drying rack). The
main model places no limit on how many modules share a buffer at one instant.
This script measures how many buffer slots the schedules actually use and what
a finite capacity B costs in makespan.

BUFFER WINDOW
  Every NON-TERMINAL operation O_i with lag_i > 0 occupies one buffer slot over
  the half-open interval [c_i, c_i + lag_i), where c_i is its completion time.
  The lag type of O_i follows its operation name:
      concrete_pour               -> curing
      waterproofing               -> ponding
      paint_coat_1, paint_coat_2  -> drying
  The occupancy of a buffer at time t is the number of its windows that contain
  t. The peak occupancy is the smallest capacity under which the schedule is
  feasible. Two buffer layouts are supported:
      shared    one buffer holds every window (capacity B)
      per-type  one buffer per lag type (capacity B each)

SUB-COMMANDS
  occupancy       peak occupancy (overall and per lag type) of the schedules of
                  the trained policy (greedy), SPT, and CP-SAT when schedules
                  are available. Writes buffers_occupancy_<method>.json and
                  buffers_occupancy_summary.md.
  cpsat-capacity  lag-aware CP-SAT with a buffer-capacity constraint
                  (AddCumulative over the lag windows, demand 1, capacity B).
                  Solves the unconstrained model first, then B = peak, ..., 1
                  (or --B). Every schedule is checked by schedule_validator and
                  by an independent occupancy sweep. Resume-safe JSONL.
                  Each solve stops at --time_limit, so every record also holds
                  the contention evidence of its solve (fields of
                  eval_stochastic_lags.cpsat_timing: solve_wall_s,
                  solve_cpu_s, workers, cpu_share = CPU s / (wall s x
                  workers), hit_limit; and of contention_fields:
                  cpu_share_min, n_timed_solves, n_hit_limit, n_low_share,
                  n_no_evidence, affinity, loadavg_start). The summary adds a
                  contention table; a budget-limited solve with cpu_share
                  below 0.9 ran on shared cores and is not a valid result.
  masked-policy   zero-shot buffer-aware dispatching: at every decision the
                  (operation, machine) pairs whose lag window would push the
                  buffer above B are masked. When every candidate pair is
                  masked, the dispatcher waits: see BufferTracker and
                  buffer_action_mask below. Resume-safe JSONL.

CLI (run from the repository root)
  python eval_buffers.py occupancy      --data_path data/PPVC/10x25+ppvc-mixed
  python eval_buffers.py cpsat-capacity --data_path data/PPVC/10x25+ppvc-mixed \
         [--buffer shared|per-type] [--B 5 3 1] [--time_limit 60] [--num_workers 2]
  python eval_buffers.py masked-policy  --data_path data/PPVC/10x25+ppvc-mixed \
         [--methods greedy SPT] [--buffer shared|per-type] [--B 5 3 1]
  Common flags: --max_instances N, --instances i j k, --out_dir DIR,
                --model_name NAME, --seed_test 50.
"""
import argparse
import glob
import json
import os
import sys
import time

# CPU-only study: hide every GPU before torch is imported anywhere.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

from eval_stochastic_lags import (SolveClock, contention_fields,
                                  contention_summary, contention_table,
                                  cpsat_timing, loadavg_now)

DEFAULT_MODEL = '10x25+ppvc-mixed+full'

LAG_TYPE_BY_OPNAME = {
    'concrete_pour': 'curing',
    'waterproofing': 'ponding',
    'paint_coat_1': 'drying',
    'paint_coat_2': 'drying',
}
LAG_TYPES = ('curing', 'ponding', 'drying')

# identical list to eval_ppvc.ARCH_KEYS: the keys that fix the network shape
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

TOL = 1e-6


# ---------------------------------------------------------------------------
# CLI. params.py parses sys.argv at import time, so our flags are parsed first
# and argv is scrubbed before params is imported.
# ---------------------------------------------------------------------------

def parse_cli(argv=None):
    ap = argparse.ArgumentParser(description='Finite lag-buffer study')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--data_path', type=str,
                       default='data/PPVC/10x25+ppvc-mixed')
        p.add_argument('--max_instances', type=int, default=0,
                       help='first N instances only (0 = all)')
        p.add_argument('--instances', type=int, nargs='+', default=None,
                       help='explicit instance indices (overrides '
                            '--max_instances)')
        p.add_argument('--out_dir', type=str, default=None,
                       help='default: test_results/PPVC/<dataset>')
        p.add_argument('--model_name', type=str, default=DEFAULT_MODEL)
        p.add_argument('--seed_test', type=int, default=50)

    p = sub.add_parser('occupancy', help='peak buffer occupancy of schedules')
    common(p)
    p.add_argument('--methods', nargs='+', default=['greedy', 'SPT', 'cpsat'],
                   help='subset of: greedy SPT cpsat')
    p.add_argument('--cpsat_jsonl', type=str, default=None,
                   help='JSONL holding CP-SAT schedules (default: the '
                        'unconstrained records of buffers_cpsat_shared_t60.jsonl '
                        'in --out_dir)')

    p = sub.add_parser('cpsat-capacity',
                       help='CP-SAT with a lag-buffer capacity constraint')
    common(p)
    p.add_argument('--buffer', choices=['shared', 'per-type'],
                   default='shared')
    p.add_argument('--B', type=int, nargs='+', default=None,
                   help='capacities to solve (default: unconstrained peak '
                        'down to 1, per instance)')
    p.add_argument('--time_limit', type=float, default=60.0)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--cpsat_seed', type=int, default=0)
    p.add_argument('--no_hint', action='store_true',
                   help='do not warm-start from the best known feasible '
                        'schedule')

    p = sub.add_parser('masked-policy',
                       help='buffer-masked dispatching (policy and SPT)')
    common(p)
    p.add_argument('--methods', nargs='+', default=['greedy', 'SPT'],
                   help='subset of: greedy SPT')
    p.add_argument('--buffer', choices=['shared', 'per-type'],
                   default='shared')
    p.add_argument('--B', type=int, nargs='+', default=None,
                   help='capacities to run (default: unmasked peak down to '
                        '1, per instance)')
    p.add_argument('--save_schedules', action='store_true',
                   help='store assigned machines and completion times in '
                        'every JSONL record')

    args = ap.parse_args(argv)
    if getattr(args, 'B', None):
        if min(args.B) < 1:
            ap.error('--B values must be >= 1')
    if getattr(args, 'num_workers', 1) < 1:
        ap.error('--num_workers must be >= 1 (0 would start one CP-SAT '
                 'worker per machine core, whatever the CPU affinity)')
    sys.argv = [sys.argv[0]]
    return args


if __name__ == '__main__':
    _ARGS = parse_cli()
else:
    _ARGS = None
    sys.argv = [sys.argv[0]] if sys.argv else ['eval_buffers.py']

from params import configs  # noqa: E402

configs.device = 'cpu'


# ---------------------------------------------------------------------------
# Instances, windows, occupancy
# ---------------------------------------------------------------------------

def list_instances(data_path):
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[buffers] no instance_*.fjs found under {data_path}')
    return [p[:-len('.fjs')] for p in fjs]


def check_fixed_routes(data_path):
    meta_path = os.path.join(data_path, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            rv = json.load(f).get('route_variant', 'fixed')
        if rv != 'fixed':
            sys.exit(f'[buffers] route_variant={rv} is not supported '
                     f'(fixed-route datasets only)')


def select_instances(args):
    stems = list_instances(args.data_path)
    if args.instances:
        bad = [i for i in args.instances if i < 0 or i >= len(stems)]
        if bad:
            sys.exit(f'[buffers] instance indices out of range: {bad}')
        return [(i, stems[i]) for i in args.instances]
    if args.max_instances > 0:
        stems = stems[:args.max_instances]
    return list(enumerate(stems))


def dataset_name(data_path):
    return os.path.basename(os.path.normpath(data_path))


def out_dir_for(args):
    d = args.out_dir or f'./test_results/PPVC/{dataset_name(args.data_path)}'
    os.makedirs(d, exist_ok=True)
    return d


def op_categories(meta):
    cats = []
    for nm in meta['op_name']:
        base = nm.split(':', 1)[1] if ':' in nm else nm
        cats.append(LAG_TYPE_BY_OPNAME.get(base))
    return cats


def window_table(jl, meta):
    """Per-op window length (0 = no window) and lag category.

    A window opens after every non-terminal op with lag > 0. Raises if a
    window op has no lag category, so the per-type split is exhaustive.
    """
    jl = np.asarray(jl, dtype=int)
    time_lag = np.asarray(meta['time_lag'], dtype=float)
    n_ops = len(time_lag)
    job_last = np.cumsum(jl) - 1
    nonterminal = np.ones(n_ops, dtype=bool)
    nonterminal[job_last] = False
    win_lag = np.where(nonterminal & (time_lag > 0), time_lag, 0.0)
    cats = op_categories(meta)
    unmapped = [i for i in range(n_ops) if win_lag[i] > 0 and cats[i] is None]
    if unmapped:
        raise ValueError(f'lag>0 ops without a lag category: {unmapped[:10]}')
    return win_lag, cats


def buffer_key(mode, cat):
    return 'shared' if mode == 'shared' else cat


def buffer_keys(mode):
    return ('shared',) if mode == 'shared' else LAG_TYPES


def schedule_windows(win_lag, cats, op_ct):
    """List of (start, end, op, category) lag windows of a schedule."""
    op_ct = np.asarray(op_ct, dtype=float)
    return [(float(op_ct[i]), float(op_ct[i] + win_lag[i]), i, cats[i])
            for i in np.nonzero(win_lag > 0)[0]]


def peak_overlap(windows):
    """Maximum number of half-open windows [s, e) that share an instant.

    Sweep over the endpoints; at equal coordinates an end (-1) is processed
    before a start (+1), because [a, t) and [t, b) do not overlap.
    """
    events = []
    for (s, e, _op, _cat) in windows:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    cur = peak = 0
    for _t, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def occupancy_stats(win_lag, cats, op_ct):
    windows = schedule_windows(win_lag, cats, op_ct)
    by_type = {t: peak_overlap([w for w in windows if w[3] == t])
               for t in LAG_TYPES}
    return {'peak': peak_overlap(windows), 'peak_by_type': by_type,
            'n_windows': len(windows)}


def peak_for_mode(occ, mode):
    return occ['peak'] if mode == 'shared' else max(occ['peak_by_type'].values())


def within_capacity(occ, mode, B):
    return B is None or peak_for_mode(occ, mode) <= B


# ---------------------------------------------------------------------------
# Buffer tracker used by the masked rollouts
# ---------------------------------------------------------------------------

class BufferTracker:
    """Fixed lag windows of the ops scheduled so far, one list per buffer.

    Windows of already scheduled ops never move, so the occupancy a new window
    would meet is known exactly at decision time.
    """

    def __init__(self, win_lag, cats, B, mode):
        self.win_lag = win_lag
        self.cats = cats
        self.B = B
        self.mode = mode
        self.s = {k: [] for k in buffer_keys(mode)}
        self.e = {k: [] for k in buffer_keys(mode)}

    def key(self, op):
        return buffer_key(self.mode, self.cats[op])

    def add(self, op, ct):
        L = self.win_lag[op]
        if L > 0:
            k = self.key(op)
            self.s[k].append(float(ct))
            self.e[k].append(float(ct) + float(L))

    def max_occ(self, key, c, L):
        """Largest number of existing windows of buffer `key` that overlap
        one instant of [c, c + L)."""
        if not self.s[key]:
            return 0
        s = np.asarray(self.s[key])
        e = np.asarray(self.e[key])
        # occupancy only rises at window starts, so the maximum over [c, c+L)
        # is attained at c or at a start inside (c, c+L)
        pts = np.concatenate(([c], s[(s > c) & (s < c + L)]))
        occ = ((s[None, :] <= pts[:, None]) & (e[None, :] > pts[:, None])).sum(1)
        return int(occ.max())

    def fits(self, op, ct):
        L = self.win_lag[op]
        if L <= 0:
            return True
        return self.max_occ(self.key(op), ct, L) < self.B

    def earliest_fit(self, op, ct0):
        """Earliest completion time >= ct0 at which the window of `op` fits."""
        L = self.win_lag[op]
        if L <= 0:
            return ct0
        k = self.key(op)
        # the occupancy only falls at window ends, so the earliest feasible
        # completion time is ct0 or one of the ends after ct0
        cands = [ct0] + sorted({x for x in self.e[k] if x > ct0})
        for c in cands:
            if self.max_occ(k, c, L) < self.B:
                return c
        raise RuntimeError('no feasible completion time found')  # unreachable


def buffer_action_mask(env, tracker, true_pt):
    """Buffer-aware action mask for a single env (batch index 0).

    Step 1: among the pairs the env itself allows (dynamic_pair_mask), keep
    those whose lag window, placed at the pair's would-be completion time on
    that machine, keeps the buffer at or below B. If any remain, return them.
    Step 2 (wait): otherwise consider every compatible (current op, machine)
    pair, compute the earliest start at or after its natural start at which
    its window fits, and allow the pairs with the smallest such start. The
    chosen pair is then started at that time, which advances the clock to the
    next window release. Returns (mask, forced_start) where mask is True for
    forbidden pairs and forced_start is None in step 1 or a [J, M] array of
    start times in step 2.
    """
    base = env.dynamic_pair_mask[0]
    cand = env.candidate[0]
    tcft = env.true_candidate_free_time[0]
    tmf = env.true_mch_free_time[0]
    allowed = ~base.copy()
    for j, m in zip(*np.nonzero(allowed)):
        op = cand[j]
        ct = max(tcft[j], tmf[m]) + true_pt[op, m]
        if not tracker.fits(op, ct):
            allowed[j, m] = False
    if allowed.any():
        return ~allowed, None

    compat = ~env.candidate_process_relation[0]
    start = np.full(base.shape, np.inf)
    for j, m in zip(*np.nonzero(compat)):
        op = cand[j]
        est = max(tcft[j], tmf[m])
        p = true_pt[op, m]
        start[j, m] = tracker.earliest_fit(op, est + p) - p
    t_next = start.min()
    if not np.isfinite(t_next):
        raise RuntimeError('no compatible pair left while the env is not done')
    allowed = start <= t_next + TOL
    return ~allowed, start


# ---------------------------------------------------------------------------
# Rollouts (copied from eval_ppvc.rollout_greedy / rollout_heuristic, with the
# buffer mask added). B=None reproduces the unmasked rollouts exactly.
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


def _spt_action(env, mask):
    """SPT choice of common_utils.heuristic_select_action with a given mask."""
    temp_pt = np.array(env.candidate_pt[0], dtype=float, copy=True)
    temp_pt[mask] = float('inf')
    pt_list = temp_pt.reshape(-1)
    action_list = np.where(pt_list == np.min(pt_list))[0]
    return int(np.random.choice(action_list))


def rollout_masked(method, jl, pt, meta, B=None, mode='shared', ppo=None,
                   use_lag_features=False, use_type_embedding=False,
                   seed=None):
    """One rollout of 'greedy' (trained policy) or 'SPT' under capacity B.

    Returns (makespan, seconds, assigned_mch, true_op_ct, info).
    """
    import torch
    from common_utils import greedy_select_action
    M = pt.shape[1]
    n_ops = pt.shape[0]
    if method == 'greedy':
        env, state = _new_env(jl, pt, meta, use_lag_features,
                              use_type_embedding)
    elif method == 'SPT':
        np.random.seed(seed)
        env, state = _new_env(jl, pt, meta, False, False)
    else:
        raise ValueError(f'unknown method {method}')
    win_lag, cats = window_table(jl, meta)
    tracker = None if B is None else BufferTracker(win_lag, cats, B, mode)
    true_pt = env.true_op_pt[0]
    scale = env.pt_upper_bound - env.pt_lower_bound + 1e-8
    assigned_mch = np.full(n_ops, -1, dtype=int)
    info = {'n_masked_decisions': 0, 'n_wait_decisions': 0,
            'n_delayed_ops': 0, 'total_delay': 0.0}

    t1 = time.time()
    while not env.done().all():
        forced = None
        if tracker is None:
            mask = None
        else:
            mask, forced = buffer_action_mask(env, tracker, true_pt)
            if np.array_equal(mask, env.dynamic_pair_mask[0]):
                mask = None
            else:
                info['n_masked_decisions'] += 1
                if forced is not None:
                    info['n_wait_decisions'] += 1

        if method == 'greedy':
            dpm = state.dynamic_pair_mask_tensor if mask is None else \
                torch.from_numpy(mask[None].copy())
            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                                   op_mask=state.op_mask_tensor,
                                   candidate=state.candidate_tensor,
                                   fea_m=state.fea_m_tensor,
                                   mch_mask=state.mch_mask_tensor,
                                   comp_idx=state.comp_idx_tensor,
                                   dynamic_pair_mask=dpm,
                                   fea_pairs=state.fea_pairs_tensor,
                                   op_type=state.op_type_tensor,
                                   mch_type=state.mch_type_tensor)
            a = int(greedy_select_action(pi).cpu().numpy()[0])
        else:
            a = _spt_action(env, env.dynamic_pair_mask[0] if mask is None
                            else mask)

        j, m = a // M, a % M
        op = int(env.candidate[0, j])
        if mask is not None and mask[j, m]:
            raise RuntimeError(f'{method} chose a masked pair ({j}, {m})')
        start_expected = max(env.true_candidate_free_time[0, j],
                             env.true_mch_free_time[0, m])
        if forced is not None and forced[j, m] > start_expected + TOL:
            # hold the job until its window fits: raise its ready time in the
            # true clock and, by the same amount, in the env's normalized clock
            delay = forced[j, m] - start_expected
            norm_start = max(env.candidate_free_time[0, j],
                             env.mch_free_time[0, m])
            env.true_candidate_free_time[0, j] = forced[j, m]
            env.candidate_free_time[0, j] = norm_start + delay / scale
            start_expected = forced[j, m]
            info['n_delayed_ops'] += 1
            info['total_delay'] += float(delay)
        assigned_mch[op] = m
        state, _, _ = env.step(np.array([a]))
        ct = env.true_op_ct[0, op]
        if abs(ct - (start_expected + true_pt[op, m])) > TOL:
            raise RuntimeError(f'op {op}: completion {ct} != expected '
                               f'{start_expected + true_pt[op, m]}')
        if tracker is not None:
            if not tracker.fits(op, ct):
                raise RuntimeError(f'op {op}: window exceeds capacity {B}')
            tracker.add(op, ct)
    t2 = time.time()
    assert (assigned_mch >= 0).all(), f'{method}: some op never scheduled'
    return (float(env.current_makespan[0]), t2 - t1, assigned_mch,
            env.true_op_ct[0].copy(), info)


# ---------------------------------------------------------------------------
# Trained policy
# ---------------------------------------------------------------------------

def load_train_config(model_name):
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[buffers] training-config snapshot not found: {path}')
    with open(path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(configs, k, snap[k])
    configs.framework = snap.get('framework', 'DANIEL')
    from common_utils import check_backbone_tag
    check_backbone_tag(model_name, configs.framework)
    configs.device = 'cpu'
    return snap


def setup_torch(seed):
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    from common_utils import setup_seed
    setup_seed(seed)
    return torch


def load_policy(model_name, seed):
    load_train_config(model_name)
    torch = setup_torch(seed)
    from model.PPO import PPO_initialize
    ckpt = f'./trained_network/PPVC/{model_name}.pth'
    if not os.path.exists(ckpt):
        sys.exit(f'[buffers] checkpoint not found: {ckpt}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt, map_location='cpu'))
    ppo.policy.eval()
    return (ppo, bool(getattr(configs, 'use_lag_features', False)),
            bool(getattr(configs, 'use_type_embedding', False)))


def method_label(method, model_name):
    return f'greedy+{model_name}' if method == 'greedy' else method


# ---------------------------------------------------------------------------
# CP-SAT with lag windows and a capacity constraint (model building copied from
# ortools_solver.fjsp_solver, flattened to per-op indices)
# ---------------------------------------------------------------------------

def cpsat_solve(jl, pt, time_lag, win_lag, cats, B, mode, time_limit,
                num_workers=2, seed=0, hint=None):
    """Lag-aware FJSP makespan model; with B set, the lag windows of each
    buffer form a cumulative resource of capacity B.

    Each window is the fixed-size interval [end_i, end_i + lag_i) of op i. It
    is always present, because every op is scheduled exactly once, and its
    start is the op's end variable, so it moves with the machine choice.
    hint = (assigned_mch, op_ct) of a feasible schedule, or None.
    The returned 'timing' is the solve's contention evidence (cpsat_timing).
    """
    from ortools.sat.python import cp_model
    jl = np.asarray(jl, dtype=int)
    pt = np.asarray(pt)
    n_ops = pt.shape[0]
    lag = [int(round(x)) for x in np.asarray(time_lag, dtype=float)]
    alts = []
    for i in range(n_ops):
        ms = np.nonzero(pt[i] != 0)[0]
        alts.append([(int(pt[i, k]), int(k)) for k in ms])

    horizon = sum(max(d for d, _ in a) for a in alts) + sum(lag)
    model = cp_model.CpModel()
    starts, ends, presence = [], [], []
    per_machine = {}
    job_ends = []
    op = 0
    for j, L in enumerate(jl):
        prev_end, prev_lag = None, 0
        for t in range(L):
            a = alts[op]
            dmin = min(d for d, _ in a)
            dmax = max(d for d, _ in a)
            sfx = f'_o{op}'
            st = model.NewIntVar(0, horizon, 'start' + sfx)
            du = model.NewIntVar(dmin, dmax, 'dur' + sfx)
            en = model.NewIntVar(0, horizon, 'end' + sfx)
            iv = model.NewIntervalVar(st, du, en, 'iv' + sfx)
            if prev_end is not None:
                model.Add(st >= prev_end + prev_lag)
            prev_end, prev_lag = en, lag[op]
            pres = []
            if len(a) > 1:
                for k, (d, mch) in enumerate(a):
                    p = model.NewBoolVar(f'p{sfx}_a{k}')
                    ls = model.NewIntVar(0, horizon, f's{sfx}_a{k}')
                    le = model.NewIntVar(0, horizon, f'e{sfx}_a{k}')
                    liv = model.NewOptionalIntervalVar(ls, d, le, p,
                                                       f'iv{sfx}_a{k}')
                    model.Add(st == ls).OnlyEnforceIf(p)
                    model.Add(du == d).OnlyEnforceIf(p)
                    model.Add(en == le).OnlyEnforceIf(p)
                    per_machine.setdefault(mch, []).append(liv)
                    pres.append(p)
                model.AddExactlyOne(pres)
            else:
                per_machine.setdefault(a[0][1], []).append(iv)
                pres.append(model.NewConstant(1))
            starts.append(st)
            ends.append(en)
            presence.append(pres)
            op += 1
        job_ends.append(prev_end)

    for ivs in per_machine.values():
        if len(ivs) > 1:
            model.AddNoOverlap(ivs)

    n_cumulative = 0
    if B is not None:
        groups = {}
        for i in np.nonzero(win_lag > 0)[0]:
            w = model.NewFixedSizeIntervalVar(ends[i], int(round(win_lag[i])),
                                              f'win_o{i}')
            groups.setdefault(buffer_key(mode, cats[i]), []).append(w)
        for ivs in groups.values():
            model.AddCumulative(ivs, [1] * len(ivs), int(B))
            n_cumulative += 1

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    if hint is not None:
        h_mch, h_ct = hint
        for i in range(n_ops):
            m_i = int(h_mch[i])
            d_i = int(pt[i, m_i])
            model.AddHint(starts[i], int(round(h_ct[i])) - d_i)
            model.AddHint(ends[i], int(round(h_ct[i])))
            if len(alts[i]) > 1:
                for k, (_d, mch) in enumerate(alts[i]):
                    model.AddHint(presence[i][k], 1 if mch == m_i else 0)
        model.AddHint(makespan, int(round(max(h_ct))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    t1 = time.time()
    with SolveClock() as clk:
        status = solver.Solve(model)
    t2 = time.time()

    out = {'status': solver.StatusName(status), 'solve_s': t2 - t1,
           'objective': None, 'bound': float(solver.BestObjectiveBound()),
           'n_cumulative': n_cumulative,
           'assigned_mch': None, 'op_ct': None,
           'timing': cpsat_timing(solver, status, clk)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        amch = np.full(n_ops, -1, dtype=int)
        oct_ = np.zeros(n_ops, dtype=int)
        for i in range(n_ops):
            for k, (d, mch) in enumerate(alts[i]):
                if solver.Value(presence[i][k]):
                    amch[i] = mch
                    oct_[i] = solver.Value(starts[i]) + d
        out['objective'] = float(solver.ObjectiveValue())
        out['assigned_mch'] = amch
        out['op_ct'] = oct_
    return out


# ---------------------------------------------------------------------------
# JSONL helpers (resume-safe: one record per line, half-written lines skipped)
# ---------------------------------------------------------------------------

def read_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f'not JSON serializable: {type(x)}')


def append_jsonl(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec, default=_jsonable) + '\n')
        f.flush()
        os.fsync(f.fileno())


def pooled(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'max': float(a.max()), 'min': float(a.min()), 'n': int(a.size)}


def b_key(B):
    return 'free' if B is None else int(B)


def default_b_list(peak):
    return list(range(int(peak), 0, -1))


# ---------------------------------------------------------------------------
# Sub-command: occupancy
# ---------------------------------------------------------------------------

def load_cpsat_schedules(args, out_dir):
    """{instance basename: (assigned_mch, op_ct, makespan, source)} or a
    reason string when no schedules exist."""
    ds = dataset_name(args.data_path)
    ref = f'./or_solution/PPVC/{ds}.jsonl'
    scheds = {}
    for rec in read_jsonl(ref):
        if rec.get('assigned_mch') is not None and rec.get('op_ct') is not None:
            scheds[rec['instance']] = (rec['assigned_mch'], rec['op_ct'],
                                       rec.get('makespan_lag'), ref)
    if scheds:
        return scheds
    reason = (f'{ref} stores makespans only (makespan_lag, status, validator '
              f'fields), no schedules')
    path = args.cpsat_jsonl or os.path.join(out_dir,
                                            'buffers_cpsat_shared_t60.jsonl')
    for rec in read_jsonl(path):
        if rec.get('B') is None and rec.get('op_ct') is not None:
            scheds[rec['instance']] = (rec['assigned_mch'], rec['op_ct'],
                                       rec.get('objective'), path)
    if scheds:
        return scheds
    return reason + f'; no unconstrained CP-SAT schedules in {path} either'


def cmd_occupancy(args):
    t_cmd = time.time()
    check_fixed_routes(args.data_path)
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    out_dir = out_dir_for(args)
    ds = dataset_name(args.data_path)
    sel = select_instances(args)

    ppo = None
    ulf = ute = False
    if 'greedy' in args.methods:
        ppo, ulf, ute = load_policy(args.model_name, args.seed_test)
    else:
        setup_torch(args.seed_test)

    cpsat = None
    skipped = {}
    if 'cpsat' in args.methods:
        cpsat = load_cpsat_schedules(args, out_dir)
        if isinstance(cpsat, str):
            skipped['cpsat'] = cpsat
            print(f'[occupancy] cpsat skipped: {cpsat}')
            cpsat = None

    results = {}
    for method in args.methods:
        if method in skipped:
            continue
        if method not in ('greedy', 'SPT', 'cpsat'):
            sys.exit(f'[occupancy] unknown method {method}')
        t_m = time.time()
        rows = []
        for i, stem in sel:
            base = os.path.basename(stem)
            jl, pt, meta = load_instance(stem)
            win_lag, cats = window_table(jl, meta)
            if method == 'cpsat':
                if base not in cpsat:
                    print(f'[occupancy] cpsat: no schedule for {base}')
                    continue
                amch, oct_, ms, src = cpsat[base]
                amch = np.asarray(amch, dtype=int)
                oct_ = np.asarray(oct_, dtype=float)
                sec = None
            else:
                ms, sec, amch, oct_, _info = rollout_masked(
                    method, jl, pt, meta, B=None, ppo=ppo,
                    use_lag_features=ulf, use_type_embedding=ute,
                    seed=args.seed_test + i)
            res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
            if not res['feasible']:
                raise RuntimeError(f'{method} {base} infeasible: '
                                   f'{res["violations"][:5]}')
            if ms is not None and abs(res['makespan'] - float(ms)) > TOL:
                raise RuntimeError(f'{method} {base}: makespan {ms} != '
                                   f'validator {res["makespan"]}')
            occ = occupancy_stats(win_lag, cats, oct_)
            rows.append({'instance': base, 'instance_id': i,
                         'makespan': float(res['makespan']),
                         'rollout_s': sec, 'feasible': True, **occ})
        if not rows:
            skipped[method] = 'no schedules'
            continue
        summary = {
            'peak': pooled([r['peak'] for r in rows]),
            'peak_by_type': {t: pooled([r['peak_by_type'][t] for r in rows])
                             for t in LAG_TYPES},
            'makespan': pooled([r['makespan'] for r in rows]),
        }
        label = method_label(method, args.model_name)
        blob = {'dataset': ds, 'method': method, 'label': label,
                'model_name': args.model_name if method == 'greedy' else None,
                'seed_test': args.seed_test,
                'window': '[c_i, c_i + lag_i) for non-terminal ops with lag>0',
                'n_instances': len(rows), 'pooled': summary,
                'per_instance': rows, 'seconds': time.time() - t_m}
        path = os.path.join(out_dir, f'buffers_occupancy_{label}.json')
        with open(path, 'w') as f:
            json.dump(blob, f, indent=1, default=_jsonable)
        results[label] = blob
        print(f'[occupancy] {label}: {len(rows)} instances, peak mean '
              f'{summary["peak"]["mean"]:.2f} median '
              f'{summary["peak"]["median"]:.1f} max {summary["peak"]["max"]:.0f}'
              f'  ({time.time() - t_m:.1f}s) -> {path}')

    # the summary covers every method JSON of this dataset in out_dir, so
    # separate calls (e.g. cpsat after cpsat-capacity) accumulate
    for fp in sorted(glob.glob(os.path.join(out_dir,
                                            'buffers_occupancy_*.json'))):
        with open(fp) as f:
            b = json.load(f)
        if b.get('dataset') == ds and b['label'] not in results:
            results[b['label']] = b
    lines = [f'# Lag-buffer peak occupancy: {ds}', '',
             '- window: [c_i, c_i + lag_i) for every non-terminal op with '
             'lag > 0; peak = most windows sharing one instant',
             '- every schedule passed schedule_validator', '']
    for m, why in skipped.items():
        lines.append(f'- {m} skipped: {why}')
    lines += ['', '| method | n | peak mean | median | max | curing max | '
                  'ponding max | drying max | mean makespan |',
              '|---|---|---|---|---|---|---|---|---|']
    for label, b in results.items():
        p = b['pooled']
        lines.append(
            f'| {label} | {b["n_instances"]} | {p["peak"]["mean"]:.2f} | '
            f'{p["peak"]["median"]:.1f} | {p["peak"]["max"]:.0f} | '
            f'{p["peak_by_type"]["curing"]["max"]:.0f} | '
            f'{p["peak_by_type"]["ponding"]["max"]:.0f} | '
            f'{p["peak_by_type"]["drying"]["max"]:.0f} | '
            f'{p["makespan"]["mean"]:.1f} |')
    lines += ['', '| method | lag type | peak mean | median | max |',
              '|---|---|---|---|---|']
    for label, b in results.items():
        for t in LAG_TYPES:
            q = b['pooled']['peak_by_type'][t]
            lines.append(f'| {label} | {t} | {q["mean"]:.2f} | '
                         f'{q["median"]:.1f} | {q["max"]:.0f} |')
    md = os.path.join(out_dir, 'buffers_occupancy_summary.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[occupancy] wrote {md}  (total {time.time() - t_cmd:.1f}s)')
    return results


# ---------------------------------------------------------------------------
# Sub-command: cpsat-capacity
# ---------------------------------------------------------------------------

def _check_schedule(jl, pt, meta, win_lag, cats, amch, oct_, mode, B):
    from schedule_validator import validate_schedule
    res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
    occ = occupancy_stats(win_lag, cats, oct_)
    return res, occ


def cmd_cpsat_capacity(args):
    t_cmd = time.time()
    check_fixed_routes(args.data_path)
    from ppvc_instance_generator import load_instance
    setup_torch(args.seed_test)
    out_dir = out_dir_for(args)
    ds = dataset_name(args.data_path)
    tl = f'{args.time_limit:g}'
    path = os.path.join(out_dir, f'buffers_cpsat_{args.buffer}_t{tl}.jsonl')
    # records of another dataset or buffer layout sharing --out_dir are ignored
    existing = [r for r in read_jsonl(path)
                if r.get('dataset') == ds and r.get('buffer') == args.buffer]
    done = {(r['instance'], b_key(r['B'])) for r in existing}
    known = {}  # instance -> list of (assigned_mch, op_ct, makespan, occ, tag)
    for r in existing:
        if r.get('op_ct') is not None and r.get('feasible'):
            known.setdefault(r['instance'], []).append(
                (np.asarray(r['assigned_mch']), np.asarray(r['op_ct']),
                 r['objective'], r['occupancy'], f'cpsat B={b_key(r["B"])}'))
    print(f'[cpsat-capacity] {ds} buffer={args.buffer} time_limit={tl}s '
          f'workers={args.num_workers} -> {path} '
          f'({len(existing)} records already present)')

    for i, stem in select_instances(args):
        base = os.path.basename(stem)
        jl, pt, meta = load_instance(stem)
        win_lag, cats = window_table(jl, meta)
        free = [r for r in existing
                if r['instance'] == base and r['B'] is None]
        b_values = [None]
        if args.B is not None:
            b_values += sorted(set(args.B), reverse=True)
        elif free and free[0].get('occupancy'):
            b_values += default_b_list(peak_for_mode(free[0]['occupancy'],
                                                     args.buffer))
        k = 0
        while k < len(b_values):
            B = b_values[k]
            k += 1
            if (base, b_key(B)) in done:
                continue
            la_start = loadavg_now()
            hint, hint_src, hint_ms = None, None, None
            if not args.no_hint:
                cands = []
                for (am, oc, ms, occ, tag) in known.get(base, []):
                    if within_capacity(occ, args.buffer, B):
                        cands.append((ms, am, oc, tag))
                ms_s, _s, am_s, oc_s, _inf = rollout_masked(
                    'SPT', jl, pt, meta, B=B, mode=args.buffer,
                    seed=args.seed_test + i)
                cands.append((ms_s, am_s, oc_s, 'masked SPT'))
                cands.sort(key=lambda c: c[0])
                hint_ms, hint = cands[0][0], (cands[0][1], cands[0][2])
                hint_src = cands[0][3]
            sol = cpsat_solve(jl, pt, meta['time_lag'], win_lag, cats, B,
                              args.buffer, args.time_limit,
                              num_workers=args.num_workers,
                              seed=args.cpsat_seed, hint=hint)
            rec = {'dataset': ds, 'instance': base, 'instance_id': i,
                   'buffer': args.buffer, 'B': B,
                   'time_limit_s': args.time_limit,
                   'num_workers': args.num_workers,
                   'cpsat_seed': args.cpsat_seed,
                   'status': sol['status'], 'objective': sol['objective'],
                   'bound': sol['bound'], 'solve_s': sol['solve_s'],
                   'n_cumulative': sol['n_cumulative'],
                   'hint_source': hint_src, 'hint_makespan': hint_ms,
                   'feasible': None, 'n_violations': None,
                   'occupancy': None, 'within_capacity': None,
                   'assigned_mch': sol['assigned_mch'], 'op_ct': sol['op_ct']}
            # contention evidence: new keys only ('status' and 'time_limit_s'
            # already hold the same values)
            for key, val in sol['timing'].items():
                rec.setdefault(key, val)
            rec.update(contention_fields([sol['timing']], la_start))
            if sol['objective'] is not None:
                res, occ = _check_schedule(jl, pt, meta, win_lag, cats,
                                           sol['assigned_mch'], sol['op_ct'],
                                           args.buffer, B)
                ok_cap = within_capacity(occ, args.buffer, B)
                ok_ms = abs(res['makespan'] - sol['objective']) <= TOL
                rec.update(feasible=bool(res['feasible'] and ok_cap and ok_ms),
                           n_violations=len(res['violations']),
                           violations=res['violations'][:5],
                           occupancy=occ, within_capacity=ok_cap,
                           validator_makespan=res['makespan'])
                if not rec['feasible']:
                    print(f'  !!! {base} B={b_key(B)}: validator='
                          f'{res["feasible"]} capacity_ok={ok_cap} '
                          f'makespan_match={ok_ms} {res["violations"][:3]}')
                else:
                    known.setdefault(base, []).append(
                        (sol['assigned_mch'], sol['op_ct'], sol['objective'],
                         occ, f'cpsat B={b_key(B)}'))
            append_jsonl(path, rec)
            done.add((base, b_key(B)))
            occ_s = ('-' if rec['occupancy'] is None else
                     str(peak_for_mode(rec['occupancy'], args.buffer)))
            print(f'  {base} B={str(b_key(B)):>4}  {sol["status"]:<9} '
                  f'obj={sol["objective"]}  bound={sol["bound"]:.0f}  '
                  f'peak={occ_s}  valid={rec["feasible"]}  '
                  f'hint={hint_src}({hint_ms})  {sol["solve_s"]:.1f}s  '
                  f'cpu_share={rec["cpu_share"]:.2f}',
                  flush=True)
            if B is None and args.B is None and rec['occupancy'] is not None:
                b_values += default_b_list(peak_for_mode(rec['occupancy'],
                                                         args.buffer))
    write_capacity_summary([r for r in read_jsonl(path)
                            if r.get('dataset') == ds
                            and r.get('buffer') == args.buffer],
                           path, args.buffer, 'CP-SAT objective',
                           timed_method='CP-SAT')
    print(f'[cpsat-capacity] total {time.time() - t_cmd:.1f}s')


# ---------------------------------------------------------------------------
# Sub-command: masked-policy
# ---------------------------------------------------------------------------

def cmd_masked_policy(args):
    t_cmd = time.time()
    check_fixed_routes(args.data_path)
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    out_dir = out_dir_for(args)
    ds = dataset_name(args.data_path)
    ppo = None
    ulf = ute = False
    if 'greedy' in args.methods:
        ppo, ulf, ute = load_policy(args.model_name, args.seed_test)
    else:
        setup_torch(args.seed_test)

    for method in args.methods:
        if method not in ('greedy', 'SPT'):
            sys.exit(f'[masked-policy] unknown method {method}')
        t_m = time.time()
        label = method_label(method, args.model_name)
        path = os.path.join(out_dir,
                            f'buffers_masked_{label}_{args.buffer}.jsonl')
        existing = [r for r in read_jsonl(path) if r.get('dataset') == ds]
        done = {(r['instance'], b_key(r['B'])) for r in existing}
        free_peak = {r['instance']: peak_for_mode(r['occupancy'], args.buffer)
                     for r in existing if r['B'] is None}
        print(f'[masked-policy] {label} buffer={args.buffer} -> {path} '
              f'({len(existing)} records already present)')
        for i, stem in select_instances(args):
            base = os.path.basename(stem)
            jl, pt, meta = load_instance(stem)
            win_lag, cats = window_table(jl, meta)
            b_values = [None]
            if args.B is not None:
                b_values += sorted(set(args.B), reverse=True)
            elif base in free_peak:
                b_values += default_b_list(free_peak[base])
            k = 0
            while k < len(b_values):
                B = b_values[k]
                k += 1
                if (base, b_key(B)) in done:
                    continue
                ms, sec, amch, oct_, info = rollout_masked(
                    method, jl, pt, meta, B=B, mode=args.buffer, ppo=ppo,
                    use_lag_features=ulf, use_type_embedding=ute,
                    seed=args.seed_test + i)
                res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
                occ = occupancy_stats(win_lag, cats, oct_)
                ok_cap = within_capacity(occ, args.buffer, B)
                ok_ms = abs(res['makespan'] - ms) <= TOL
                feasible = bool(res['feasible'] and ok_cap and ok_ms)
                if not feasible:
                    raise RuntimeError(
                        f'{label} {base} B={b_key(B)}: validator='
                        f'{res["feasible"]} capacity_ok={ok_cap} '
                        f'makespan_match={ok_ms} {res["violations"][:5]}')
                rec = {'dataset': ds, 'instance': base, 'instance_id': i,
                       'method': method, 'label': label,
                       'buffer': args.buffer, 'B': B,
                       'seed': args.seed_test + i if method == 'SPT' else None,
                       'objective': ms, 'rollout_s': sec,
                       'feasible': feasible, 'within_capacity': ok_cap,
                       'occupancy': occ, **info}
                if args.save_schedules:
                    rec['assigned_mch'] = amch
                    rec['op_ct'] = oct_
                append_jsonl(path, rec)
                done.add((base, b_key(B)))
                print(f'  {base} B={str(b_key(B)):>4}  makespan={ms:.0f}  '
                      f'peak={peak_for_mode(occ, args.buffer)}  '
                      f'masked={info["n_masked_decisions"]}  '
                      f'waits={info["n_wait_decisions"]}  '
                      f'delayed={info["n_delayed_ops"]}'
                      f'(+{info["total_delay"]:.0f}h)  {sec:.2f}s',
                      flush=True)
                if B is None and args.B is None:
                    b_values += default_b_list(peak_for_mode(occ, args.buffer))
        write_capacity_summary([r for r in read_jsonl(path)
                                if r.get('dataset') == ds],
                               path, args.buffer, f'{label} makespan')
        print(f'[masked-policy] {label}: {time.time() - t_m:.1f}s')
    print(f'[masked-policy] total {time.time() - t_cmd:.1f}s')


# ---------------------------------------------------------------------------
# Makespan-vs-capacity summary (shared by cpsat-capacity and masked-policy)
# ---------------------------------------------------------------------------

def write_capacity_summary(recs, jsonl_path, mode, what, timed_method=None):
    """Markdown next to the JSONL: per-instance rows and a pooled table.

    In the pooled table an instance whose unconstrained peak is at most B
    enters with its unconstrained result (the constraint is slack there); the
    column 'filled' counts those instances. timed_method names the
    time-limited solver of the records; it adds the contention table
    (records written before the evidence existed are counted as such).
    """
    by_inst = {}
    for r in recs:
        by_inst.setdefault(r['instance'], {})[b_key(r['B'])] = r
    lines = [f'# Makespan vs lag-buffer capacity ({mode} buffer)', '',
             f'- source: `{os.path.basename(jsonl_path)}`; value: {what}',
             '- every schedule with a value passed schedule_validator and the '
             'occupancy check (peak <= B)',
             '- inflation = value / unconstrained value - 1', '',
             '| instance | B | status | value | bound | peak | inflation % |',
             '|---|---|---|---|---|---|---|']
    all_b = set()
    for inst in sorted(by_inst):
        rows = by_inst[inst]
        free = rows.get('free')
        f_val = free['objective'] if free else None
        keys = ['free'] + sorted((b for b in rows if b != 'free'),
                                 reverse=True)
        for b in keys:
            if b not in rows:
                continue
            r = rows[b]
            if b != 'free':
                all_b.add(b)
            val = r['objective']
            peak = ('-' if r.get('occupancy') is None else
                    peak_for_mode(r['occupancy'], mode))
            infl = ('-' if val is None or not f_val else
                    f'{100.0 * (val / f_val - 1):.2f}')
            bound = r.get('bound')
            lines.append(f'| {inst} | {b} | {r.get("status", "-")} | '
                         f'{"-" if val is None else f"{val:.0f}"} | '
                         f'{"-" if bound is None else f"{bound:.0f}"} | '
                         f'{peak} | {infl} |')
    lines += ['', '| B | n | filled (slack) | missing | mean value | '
                  'mean inflation % |', '|---|---|---|---|---|---|']
    for b in sorted(all_b, reverse=True):
        vals, infls, filled, missing = [], [], 0, 0
        for inst, rows in by_inst.items():
            free = rows.get('free')
            if free is None or free['objective'] is None:
                missing += 1
                continue
            r = rows.get(b)
            if r is None or r['objective'] is None:
                if (free.get('occupancy') is not None and
                        peak_for_mode(free['occupancy'], mode) <= b):
                    r, filled = free, filled + 1
                else:
                    missing += 1
                    continue
            vals.append(r['objective'])
            infls.append(100.0 * (r['objective'] / free['objective'] - 1))
        if vals:
            lines.append(f'| {b} | {len(vals)} | {filled} | {missing} | '
                         f'{np.mean(vals):.1f} | {np.mean(infls):.2f} |')
    if timed_method is not None:
        lines += [''] + contention_table(
            {timed_method: contention_summary(recs)})
    md = jsonl_path[:-len('.jsonl')] + '_summary.md'
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  wrote {md}')


def main(args):
    if args.cmd == 'occupancy':
        return cmd_occupancy(args)
    if args.cmd == 'cpsat-capacity':
        return cmd_cpsat_capacity(args)
    if args.cmd == 'masked-policy':
        return cmd_masked_policy(args)
    raise ValueError(args.cmd)


if __name__ == '__main__':
    main(_ARGS)
