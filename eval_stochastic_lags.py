"""
eval_stochastic_lags.py
=======================
Closed-loop execution of PPVC schedules under UNCERTAIN post-operation lags.

LAG MODEL
  For every op with nominal lag L > 0 the realized lag is round(L * X), where X
  is lognormal with mean 1 and coefficient of variation cv, truncated to
  [0.5, 2.0] by rejection (redraw until inside). This follows the
  scenario-based stochastic FJSP of Smit et al. (AAAI 2025), which samples
  processing times from a lognormal around the nominal value; here the lags
  are sampled. One draw per (instance, cv, replication), from a generator
  seeded by (lag_seed, instance index, cv, replication), so every method runs
  against the same realizations. Ops with L = 0 keep a zero lag.

INFORMATION MODEL
  A realized lag is revealed only when its op completes (event-driven). Until
  then every method plans with the nominal lag.

METHODS (all executed against the same realizations)
  slab_reactive     the trained greedy policy plans with nominal lags; the env
                    is the simulator. Whenever a lagged op completes, its true
                    lag is written into the env (true_op_lag / op_lag, the
                    job's candidate_free_time, the job's C_LB features), the
                    decision state is rebuilt with the env's own feature code,
                    and the policy continues from the current instant.
  spt_reactive      the same closed-loop protocol with the SPT rule.
  slab_repair       the policy's nominal-lag plan (machine assignment and
                    machine order) is kept and right_shift_repair retimes it
                    with the realized lags. The repair is causal (a start
                    depends only on its predecessors), so applying it once with
                    all realized lags equals applying it at every reveal.
  cpsat_repair      a CP-SAT plan on nominal lags (fresh solve, --cpsat_budget,
                    cached per instance), repaired the same way. With
                    --cpsat_hint spt the nominal plan is solved in the mode of
                    eval_cpsat_anytime.py --mode warm-spt: the lag-aware SPT
                    schedule (seed seed_test + instance index, the anytime
                    script's 'per-instance' rule) is hinted in full to that
                    script's model (build_lag_model / add_full_hint), solved
                    with num_search_workers = --cpsat_workers, random_seed 0.
                    The default --cpsat_hint none keeps the cold solve.
  cpsat_rolling     (opt-in, --rolling_cpsat) starts from the cpsat_repair
                    nominal plan. At every reveal that changes a lag, the ops
                    that have started are frozen, and the residual is
                    re-solved with CP-SAT (--resolve_budget, warm-started
                    from the current plan retimed with the known lags),
                    following the prefix-respecting model of
                    eval_residual_cpsat.py. The better of the re-solve and the
                    retimed current plan is kept.
  slab_perfect_info reference: the policy run once with the realized lags
                    known in advance.

RECORDED per (instance, cv, rep, method)
  realized makespan, nominal-plan makespan and inflation over it, number of
  lag reveals, number of re-plans (reveals that changed a lag with a job
  successor, or CP-SAT re-solves), total re-planning seconds, plan seconds,
  wall seconds. Every final schedule is checked with
  schedule_validator.validate_schedule against the realized lags.
  Contention evidence of every time-limited CP-SAT solve (nominal plan and
  re-solves: wall and CPU seconds, workers, cpu_share, status, hit_limit) and
  per record cpu_share_min, n_timed_solves, n_hit_limit, n_low_share,
  n_no_evidence, affinity, loadavg_start (see contention_fields).

OUTPUT (test_results/PPVC/<dataset>/ unless --save_dir)
  stochastic_lags_<method>.jsonl        one record per line, resume-safe
  stochastic_lags_cpsat_nominal.jsonl   cached nominal CP-SAT plans
  stochastic_lags_summary.md            mean realized makespan per method, cv

CLI
  python eval_stochastic_lags.py [--model_name 10x25+ppvc-mixed+full]
      [--data_path data/PPVC/10x25+ppvc-mixed] [--cv 0.1 0.2 0.3] [--n_rep 3]
      [--methods slab_reactive spt_reactive slab_repair cpsat_repair
                 slab_perfect_info] [--rolling_cpsat]
      [--cpsat_budget 60] [--resolve_budget 5] [--cpsat_workers 2]
      [--cpsat_hint {none,spt}]
      [--seed_test 50] [--lag_seed 2027] [--max_instances N] [--save_dir DIR]

The module has no import-time side effects (no argv parse, no params import),
so train.py can import realize_lags from it.
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import numpy as np

LAG_TRUNC = (0.5, 2.0)
DEFAULT_METHODS = ('slab_reactive', 'spt_reactive', 'slab_repair',
                   'cpsat_repair', 'slab_perfect_info')
ALL_METHODS = DEFAULT_METHODS + ('cpsat_rolling',)
POLICY_METHODS = ('slab_reactive', 'slab_repair', 'slab_perfect_info')
CPSAT_METHODS = ('cpsat_repair', 'cpsat_rolling')
EPS = 1e-9


# ---------------------------------------------------------------------------
# Lag model (pure numpy; shared with train.py's --ppvc_lag_cv)
# ---------------------------------------------------------------------------

def lognormal_params(cv):
    """(mu, sigma) of the normal whose exponential has mean 1 and the given cv."""
    sigma2 = np.log1p(cv * cv)
    return -0.5 * sigma2, float(np.sqrt(sigma2))


def sample_lag_factors(n, cv, rng, lo=LAG_TRUNC[0], hi=LAG_TRUNC[1]):
    """n lognormal factors (mean 1, coefficient of variation cv), truncated to
    [lo, hi] by rejection. rng is a numpy Generator or the np.random module
    (both expose lognormal(mean, sigma, size)). cv <= 0 returns ones and
    draws nothing."""
    if cv <= 0 or n == 0:
        return np.ones(n)
    mu, sigma = lognormal_params(cv)
    x = np.asarray(rng.lognormal(mu, sigma, size=n), dtype=float)
    bad = (x < lo) | (x > hi)
    while bad.any():
        x[bad] = rng.lognormal(mu, sigma, size=int(bad.sum()))
        bad = (x < lo) | (x > hi)
    return x


def realize_lags(nominal_lag, cv, rng):
    """Realized lags: round(L * X) where L > 0, zero elsewhere (float array).
    Rounding is half-up, so a realized lag is a whole number of hours."""
    nominal = np.asarray(nominal_lag, dtype=float)
    out = nominal.copy()
    pos = nominal > 0
    if cv > 0 and pos.any():
        x = sample_lag_factors(int(pos.sum()), cv, rng)
        out[pos] = np.floor(nominal[pos] * x + 0.5)
    return out


def scenario_rng(lag_seed, instance_idx, cv, rep):
    """Generator for one (instance, cv, replication) lag scenario."""
    return np.random.default_rng(
        [int(lag_seed), int(instance_idx), int(round(cv * 1e6)), int(rep)])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _import_with_retry(name, tries=6, wait=5.0):
    """Import a repo module; other work may be rewriting it at this moment."""
    for k in range(tries):
        try:
            return importlib.import_module(name)
        except (ImportError, SyntaxError, NameError, AttributeError) as e:
            if k == tries - 1:
                raise
            print(f'  [retry] import {name} failed ({type(e).__name__}: {e}); '
                  f'retrying in {wait:.0f}s', flush=True)
            time.sleep(wait)


def job_structure(jl):
    """op -> job index, and a bool mask of the last op of every job."""
    jl = np.asarray(jl, dtype=int)
    n = int(jl.sum())
    op_job = np.repeat(np.arange(len(jl)), jl)
    is_last = np.zeros(n, dtype=bool)
    is_last[np.cumsum(jl) - 1] = True
    return op_job, is_last


def retime(jl, pt, lag, mch, order_start, floor):
    """Earliest-start timing of a fixed plan (machine assignment and machine
    order from order_start), with a per-op release floor.

    start[op] = max(floor[op], ct[job pred] + lag[job pred], ct[mch pred]).
    Returns (start[N], ct[N]). Raises on a cyclic order.
    """
    jl = np.asarray(jl, dtype=int)
    pt = np.asarray(pt, dtype=float)
    lag = np.asarray(lag, dtype=float)
    mch = np.asarray(mch, dtype=int)
    n = pt.shape[0]
    dur = pt[np.arange(n), mch]
    assert (dur > 0).all(), 'retime: op assigned to an incompatible machine'
    op_job, _ = job_structure(jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    job_pred = np.full(n, -1, dtype=int)
    for j, f in enumerate(first):
        job_pred[f + 1:f + jl[j]] = np.arange(f, f + jl[j] - 1)
    mch_pred = np.full(n, -1, dtype=int)
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for op in range(n):
        if job_pred[op] >= 0:
            succ[job_pred[op]].append(op)
            indeg[op] += 1
    for m in np.unique(mch):
        ops = np.where(mch == m)[0]
        ops = sorted(ops.tolist(), key=lambda i: (order_start[i], i))
        for a, b in zip(ops[:-1], ops[1:]):
            mch_pred[b] = a
            succ[a].append(b)
            indeg[b] += 1
    start = np.zeros(n)
    ct = np.zeros(n)
    queue = [i for i in range(n) if indeg[i] == 0]
    done = 0
    while queue:
        u = queue.pop()
        done += 1
        s = float(floor[u])
        if job_pred[u] >= 0:
            s = max(s, ct[job_pred[u]] + lag[job_pred[u]])
        if mch_pred[u] >= 0:
            s = max(s, ct[mch_pred[u]])
        start[u] = s
        ct[u] = s + dur[u]
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if done != n:
        raise ValueError('retime: machine order is cyclic with job precedence')
    return start, ct


# ---------------------------------------------------------------------------
# Contention evidence for wall-clock-budgeted solves (shared with
# eval_reactive.py and eval_residual_cpsat.py; this module has no import-time
# side effects). A solve that stops at its time limit returns a result that
# depends on the CPU it received, so each such solve records its wall and
# process-CPU seconds. cpu_share = CPU s / (wall s x worker threads); a
# budget-limited solve with cpu_share below LOW_CPU_SHARE ran on shared cores
# and is not a valid measurement. time.process_time() covers every thread of
# the process, and CP-SAT runs its workers as threads inside the process.
# ---------------------------------------------------------------------------

LOW_CPU_SHARE = 0.9    # budget-limited solves below this share are invalid
HIT_LIMIT_FRAC = 0.95  # a solve using >= 95 % of its limit is budget-limited
CONTENTION_KEYS = ('cpu_share_min', 'n_timed_solves', 'n_hit_limit',
                   'n_low_share', 'n_no_evidence', 'affinity',
                   'loadavg_start')
SOLVE_KEYS = ('solve_wall_s', 'solve_cpu_s', 'workers', 'cpu_share',
              'status', 'time_limit_s', 'hit_limit')


def affinity_list():
    """Sorted CPU affinity of this process (None where unsupported)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def loadavg_now():
    """1/5/15-minute load averages, rounded (None where unsupported)."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def resolve_threads(n):
    """Threads a solve can run at once: n, or the affinity size when n <= 0
    (CP-SAT reads 0 workers as 'use every core')."""
    n = int(n or 0)
    if n > 0:
        return n
    aff = affinity_list()
    return len(aff) if aff else (os.cpu_count() or 1)


class SolveClock:
    """Wall (perf_counter) and process-CPU seconds of the enclosed block."""

    def __enter__(self):
        self.cpu0 = time.process_time()
        self.wall0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self.wall0
        self.cpu_s = time.process_time() - self.cpu0
        return False


def solve_timing(wall_s, cpu_s, workers, limit_s, status):
    """Contention evidence of one timed solve (keys SOLVE_KEYS)."""
    workers = max(int(workers), 1)
    wall_s, cpu_s, limit_s = float(wall_s), float(cpu_s), float(limit_s)
    return {'solve_wall_s': round(wall_s, 4),
            'solve_cpu_s': round(cpu_s, 4),
            'workers': workers,
            'cpu_share': round(cpu_s / max(wall_s * workers, 1e-9), 3),
            'status': str(status),
            'time_limit_s': limit_s,
            'hit_limit': bool(0 < limit_s < float('inf')
                              and wall_s >= HIT_LIMIT_FRAC * limit_s)}


def cpsat_timing(solver, status, clock):
    """solve_timing of a finished CP-SAT solve timed by a SolveClock. The
    worker count is the configured num_workers (num_search_workers when that
    is 0), and the affinity size when both are 0."""
    p = solver.parameters
    workers = resolve_threads(int(p.num_workers) or int(p.num_search_workers))
    return solve_timing(clock.wall_s, clock.cpu_s, workers,
                        p.max_time_in_seconds, solver.StatusName(status))


def contention_fields(timings, loadavg_start):
    """Per-record contention summary (keys CONTENTION_KEYS) over the timed
    solves the record's result depends on. A None entry is a solve without
    evidence (for example a plan cached by an older run)."""
    have = [t for t in timings if t is not None]
    shares = [t['cpu_share'] for t in have if t['hit_limit']]
    return {'cpu_share_min': min(shares) if shares else None,
            'n_timed_solves': len(timings),
            'n_hit_limit': len(shares),
            'n_low_share': sum(1 for s in shares if s < LOW_CPU_SHARE),
            'n_no_evidence': len(timings) - len(have),
            'affinity': affinity_list(),
            'loadavg_start': loadavg_start}


def contention_arrays(n, per_solve=True):
    """Per-instance arrays of the contention fields, for result records kept
    as one dict of arrays per method. per_solve adds the SOLVE_KEYS arrays
    for methods with exactly one timed solve per instance."""
    a = {'cpu_share_min': np.full(n, np.nan),
         'n_timed_solves': np.zeros(n, dtype=int),
         'n_hit_limit': np.zeros(n, dtype=int),
         'n_low_share': np.zeros(n, dtype=int),
         'n_no_evidence': np.zeros(n, dtype=int),
         'affinity': np.full(n, None, dtype=object),
         'loadavg_start': np.full((n, 3), np.nan)}
    if per_solve:
        a.update({'solve_wall_s': np.full(n, np.nan),
                  'solve_cpu_s': np.full(n, np.nan),
                  'workers': np.zeros(n, dtype=int),
                  'cpu_share': np.full(n, np.nan),
                  'status': np.full(n, '', dtype=object),
                  'time_limit_s': np.full(n, np.nan),
                  'hit_limit': np.zeros(n, dtype=bool)})
    return a


def store_contention(r, i, timings, loadavg_start):
    """Write record i's contention evidence into the dict of arrays r."""
    f = contention_fields(timings, loadavg_start)
    r['cpu_share_min'][i] = (np.nan if f['cpu_share_min'] is None
                             else f['cpu_share_min'])
    for k in ('n_timed_solves', 'n_hit_limit', 'n_low_share',
              'n_no_evidence'):
        r[k][i] = f[k]
    r['affinity'][i] = f['affinity']
    if loadavg_start is not None:
        r['loadavg_start'][i] = loadavg_start
    if 'solve_wall_s' in r and len(timings) == 1 and timings[0] is not None:
        for k in SOLVE_KEYS:
            r[k][i] = timings[0][k]


def contention_summary(rows):
    """Run-level view over per-record contention fields: the minimum
    cpu_share_min and the summed counts. rows are dicts (missing keys are
    records written before the evidence existed) or one dict of arrays."""
    if isinstance(rows, dict):
        n = len(rows['n_timed_solves'])
        rows = [{k: rows[k][i] for k in CONTENTION_KEYS} for i in range(n)]
    mins, out = [], {'n_records': 0, 'n_records_without_fields': 0,
                     'n_timed_solves': 0, 'n_hit_limit': 0,
                     'n_low_share': 0, 'n_no_evidence': 0}
    for r in rows:
        out['n_records'] += 1
        if 'n_timed_solves' not in r:
            out['n_records_without_fields'] += 1
            continue
        x = r.get('cpu_share_min')
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            mins.append(float(x))
        for k in ('n_timed_solves', 'n_hit_limit', 'n_low_share',
                  'n_no_evidence'):
            out[k] += int(r.get(k) or 0)
    out['cpu_share_min'] = min(mins) if mins else None
    return out


def contention_table(summaries):
    """Markdown lines: one row per method with its run-level evidence."""
    lines = [f'Contention evidence: cpu_share = CPU s / (wall s x workers) of '
             f'each timed solve; a solve is budget-limited when it used >= '
             f'{HIT_LIMIT_FRAC:.0%} of its limit, and a budget-limited solve '
             f'with cpu_share < {LOW_CPU_SHARE} is invalid.', '',
             '| method | records | timed solves | budget-limited | '
             f'min cpu_share | cpu_share < {LOW_CPU_SHARE} | no evidence |',
             '|---|---|---|---|---|---|---|']
    for m, s in summaries.items():
        cs = 'n/a' if s['cpu_share_min'] is None else f'{s["cpu_share_min"]:.3f}'
        miss = str(s['n_no_evidence'])
        if s['n_records_without_fields']:
            miss += f' (+{s["n_records_without_fields"]} records without fields)'
        lines.append(f'| {m} | {s["n_records"]} | {s["n_timed_solves"]} | '
                     f'{s["n_hit_limit"]} | {cs} | {s["n_low_share"]} | '
                     f'{miss} |')
    return lines


# ---------------------------------------------------------------------------
# CP-SAT (prefix-respecting residual model of eval_residual_cpsat.py, with an
# explicit worker cap and an optional solution hint)
# ---------------------------------------------------------------------------

def cpsat_residual(jl, pt, lag, t0, committed, fixed_mch, fixed_start, budget,
                   num_workers=2, hint_mch=None, hint_start=None, seed=0):
    """Minimize makespan; committed ops are fixed intervals, every other op
    starts at or after t0. Lags are finish-to-start job precedences.
    With nothing committed and t0 = 0 this is the full nominal FJSP-TL model.
    Returns dict(status, objective, seconds, assigned_mch, start, op_ct,
    timing), where timing is the solve's contention evidence (solve_timing)."""
    from ortools.sat.python import cp_model
    jl = np.asarray(jl, dtype=int)
    ipt = np.rint(np.asarray(pt, dtype=float)).astype(int)
    ilag = np.rint(np.asarray(lag, dtype=float)).astype(int)
    n, M = ipt.shape
    t0 = int(round(t0))
    committed = np.asarray(committed, dtype=bool)

    horizon = t0
    if committed.any():
        cct = [int(round(fixed_start[o])) + int(ipt[o, fixed_mch[o]])
               for o in np.where(committed)[0]]
        horizon = max(horizon, max(cct))
    horizon += int(sum(ipt[o].max() for o in range(n) if not committed[o]))
    horizon += int(ilag.sum()) + 1
    if hint_start is not None:
        hct = [int(round(hint_start[o])) + int(ipt[o, hint_mch[o]])
               for o in range(n)]
        horizon = max(horizon, max(hct) + 1)

    model = cp_model.CpModel()
    per_mch = {m: [] for m in range(M)}
    starts, ends, pres = {}, {}, {}
    for op in range(n):
        if committed[op]:
            m = int(fixed_mch[op])
            s = int(round(fixed_start[op]))
            d = int(ipt[op, m])
            st = model.NewConstant(s)
            en = model.NewConstant(s + d)
            per_mch[m].append(model.NewFixedSizeIntervalVar(st, d, f'iv{op}'))
            starts[op], ends[op] = st, en
            continue
        able = np.where(ipt[op] > 0)[0]
        assert len(able) > 0, f'op {op} has no compatible machine'
        st = model.NewIntVar(t0, horizon, f's{op}')
        en = model.NewIntVar(t0, horizon, f'e{op}')
        starts[op], ends[op] = st, en
        if len(able) == 1:
            m = int(able[0])
            d = int(ipt[op, m])
            per_mch[m].append(model.NewFixedSizeIntervalVar(st, d, f'iv{op}'))
            model.Add(en == st + d)
            pres[(op, m)] = None
        else:
            lits = []
            for m in able:
                m = int(m)
                p = model.NewBoolVar(f'p{op}_{m}')
                per_mch[m].append(model.NewOptionalFixedSizeIntervalVar(
                    st, int(ipt[op, m]), p, f'iv{op}_{m}'))
                pres[(op, m)] = p
                lits.append(p)
            model.AddExactlyOne(lits)
            model.Add(en == st + sum(int(ipt[op, int(m)]) * pres[(op, int(m))]
                                     for m in able))
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    for j, f in enumerate(first):
        for k in range(f + 1, f + jl[j]):
            model.Add(starts[k] >= ends[k - 1] + int(ilag[k - 1]))
    for m in range(M):
        if len(per_mch[m]) > 1:
            model.AddNoOverlap(per_mch[m])
    mk = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(mk, [ends[o] for o in range(n)])
    model.Minimize(mk)

    if hint_start is not None:
        for op in range(n):
            if committed[op]:
                continue
            model.AddHint(starts[op], int(round(hint_start[op])))
            for (o, m), p in pres.items():
                if o == op and p is not None:
                    model.AddHint(p, 1 if m == int(hint_mch[op]) else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    with SolveClock() as clk:
        t1 = time.time()
        status = solver.Solve(model)
        sec = time.time() - t1
    out = {'status': solver.StatusName(status), 'seconds': sec,
           'objective': float('nan'), 'assigned_mch': None, 'start': None,
           'op_ct': None, 'timing': cpsat_timing(solver, status, clk)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        amch = np.full(n, -1, dtype=int)
        s_arr = np.zeros(n)
        for op in range(n):
            if committed[op]:
                amch[op] = int(fixed_mch[op])
            else:
                able = np.where(ipt[op] > 0)[0]
                if len(able) == 1:
                    amch[op] = int(able[0])
                else:
                    for m in able:
                        if solver.Value(pres[(op, int(m))]):
                            amch[op] = int(m)
                            break
            s_arr[op] = float(solver.Value(starts[op]))
        out.update(objective=float(solver.ObjectiveValue()), assigned_mch=amch,
                   start=s_arr, op_ct=s_arr + ipt[np.arange(n), amch])
    return out


def cpsat_nominal_warm_spt(jl, pt, meta, spt_seed, budget, num_workers,
                           data_dir=None, seed=0):
    """Nominal CP-SAT plan started from the lag-aware SPT schedule, in the mode
    of eval_cpsat_anytime.py --mode warm-spt: the SPT rollout
    (eval_cpsat_anytime.spt_schedule, numpy seed spt_seed), its model
    (build_lag_model) with the complete SPT schedule as hint (add_full_hint),
    and its solver parameters (num_search_workers, random_seed). Returns the
    keys of cpsat_residual plus the hint evidence: spt_makespan,
    hint_makespan, hint_first_solution [wall s, objective, bound],
    hint_taken, bound, n_improving_solutions."""
    import eval_cpsat_anytime as eca
    from ortools.sat.python import cp_model
    from ortools_solver import matrix_to_the_format_for_solving
    from schedule_validator import validate_schedule
    jl = np.asarray(jl, dtype=int)
    lag = meta['time_lag']
    spt_mk, spt_s, spt_asg, spt_ct = eca.spt_schedule(jl, pt, meta, spt_seed,
                                                      data_dir)
    v = validate_schedule(jl, pt, lag, spt_asg, spt_ct)
    assert v['feasible'], f'SPT hint infeasible: {v["violations"][:3]}'
    assert abs(v['makespan'] - spt_mk) < 1e-6, 'SPT makespan != validator'
    jobs, n_mch = matrix_to_the_format_for_solving(jl, pt)
    lm = eca.build_lag_model(jobs, n_mch, lag)
    hinted = float(eca.add_full_hint(lm, jobs, spt_asg, spt_ct))
    assert abs(hinted - spt_mk) < 1e-6, 'hinted makespan != SPT makespan'

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(budget)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    recorder = eca.TrajectoryRecorder()
    with SolveClock() as clk:
        t1 = time.time()
        status = solver.Solve(lm.model, recorder)
        sec = time.time() - t1
    first = recorder.points[0] if recorder.points else None
    out = {'status': solver.StatusName(status), 'seconds': sec,
           'objective': float('nan'), 'assigned_mch': None, 'start': None,
           'op_ct': None, 'timing': cpsat_timing(solver, status, clk),
           'bound': float(solver.BestObjectiveBound()),
           'n_improving_solutions': len(recorder.points),
           'spt_makespan': float(spt_mk), 'spt_rollout_s': float(spt_s),
           'spt_seed': int(spt_seed), 'hint_makespan': hinted,
           'hint_first_solution': first,
           'hint_taken': bool(first is not None and first[1] <= hinted)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        asg, ct = eca.extract_schedule(solver, lm, jobs)
        n = pt.shape[0]
        dur = np.rint(np.asarray(pt, dtype=float))[np.arange(n), asg]
        out.update(objective=float(solver.ObjectiveValue()),
                   assigned_mch=np.asarray(asg, dtype=int),
                   start=ct.astype(float) - dur, op_ct=ct.astype(float))
    return out


def nominal_cache_key(base, args):
    """Key of a cached nominal CP-SAT plan; the cold solve keeps the old key."""
    key = f'{base}|{args.cpsat_budget:g}|{args.cpsat_workers}'
    hint = getattr(args, 'cpsat_hint', 'none')
    return key if hint == 'none' else f'{key}|hint-{hint}'


# ---------------------------------------------------------------------------
# Env-as-simulator: lag reveal and state rebuild (FJSPEnvForSameOpNums)
# ---------------------------------------------------------------------------

def refresh_env_state(env):
    """Rebuild the decision instant, masks and features after the job-ready
    times changed. Mirrors the tail of FJSPEnvForSameOpNums.step()."""
    import numpy.ma as ma
    cand_ft = np.expand_dims(env.candidate_free_time, axis=2)
    mch_ft = np.expand_dims(env.mch_free_time, axis=1)
    env.pair_free_time = np.maximum(cand_ft, mch_ft)
    sched = ma.array(env.pair_free_time, mask=env.candidate_process_relation)
    env.next_schedule_time = np.min(
        sched.reshape(env.number_of_envs, -1), axis=1).data
    t = env.next_schedule_time
    # deleted-node set at the (possibly earlier) new instant; the op mask is
    # the initial mask OR the deleted set, and deleted sets grow with time
    env.deleted_op_nodes = np.logical_and(env.op_ct <= t[:, np.newaxis],
                                          env.op_scheduled_flag)
    env.delete_mask_fea_j = np.tile(env.deleted_op_nodes[:, :, np.newaxis],
                                    (1, 1, env.op_fea_dim))
    env.init_op_mask()
    env.update_op_mask()
    env.op_waiting_time = np.zeros((env.number_of_envs, env.number_of_ops))
    env.op_waiting_time[env.env_job_idx, env.candidate] = \
        (1 - env.mask) * np.maximum(np.expand_dims(t, axis=1)
                                    - env.candidate_free_time, 0)
    env.op_remain_work = np.maximum(env.op_ct - np.expand_dims(t, axis=1), 0)
    if env.use_lag_features:
        env.op_remain_lag = env.op_scheduled_flag * np.clip(
            env.op_ct + env.op_lag - t[:, np.newaxis], 0, env.op_lag)
    env.construct_op_features()
    env.dynamic_pair_mask = np.logical_or(
        env.candidate_process_relation,
        env.pair_free_time > t[:, np.newaxis, np.newaxis])
    env.comp_idx = env.logic_operator(x=~env.dynamic_pair_mask)
    env.update_mch_mask()
    env.mch_current_available_jc_nums = np.sum(~env.dynamic_pair_mask, axis=1)
    free_dur = np.expand_dims(t, axis=1) - env.mch_free_time
    busy = free_dur < 0
    env.mch_working_flag = busy + 0
    env.mch_waiting_time = (1 - busy) * free_dur
    env.mch_remain_work = np.maximum(-free_dur, 0)
    env.construct_mch_features()
    env.construct_pair_features()
    env.state.update(env.fea_j, env.op_mask, env.fea_m, env.mch_mask,
                     env.dynamic_pair_mask, env.comp_idx, env.candidate,
                     env.fea_pairs, op_type=env.op_type, mch_type=env.mch_type)
    return env.state


def reveal_lag(env, op, new_true, op_job, is_last):
    """Write a realized lag into the env. Returns True if job-ready times
    changed (the op has a job successor and the lag differs)."""
    slope = float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)
    old_true = float(env.true_op_lag[0, op])
    if new_true == old_true:
        return False
    delta = (new_true - old_true) / slope
    env.true_op_lag[0, op] = new_true
    env.op_lag[0, op] = new_true / slope
    if is_last[op]:
        return False
    j = int(op_job[op])
    assert int(env.candidate[0, j]) == op + 1 and \
        not env.op_scheduled_flag[0, op + 1], \
        f'reveal of op {op}: its successor was committed before the reveal'
    env.candidate_free_time[0, j] = env.op_ct[0, op] + env.op_lag[0, op]
    env.true_candidate_free_time[0, j] = (env.true_op_ct[0, op]
                                          + env.true_op_lag[0, op])
    if not getattr(env, 'bound_lag_free', False):
        # the lag-augmented bound of the job's remaining ops moves with the lag
        last = int(env.job_last_op_id[0, j])
        env.op_ct_lb[0, op + 1:last + 1] += delta
    return True


def closed_loop(env, select, realized_lag, jl, reveal=True):
    """Roll the env to completion with `select(env) -> action`. With reveal,
    every lagged op's realized lag is written into the env at its completion
    instant, earliest completion first, before the next decision.

    Returns dict(makespan, assigned_mch, true_op_ct, n_reveals, n_replans,
    replan_seconds, wall_seconds)."""
    n_ops = env.number_of_ops
    M = env.number_of_machines
    op_job, is_last = job_structure(jl)
    realized_lag = np.asarray(realized_lag, dtype=float)
    pending = set(np.where(env.true_op_lag[0] > 0)[0].tolist()) if reveal \
        else set()
    assigned = np.full(n_ops, -1, dtype=int)
    n_rev = n_rep = 0
    rep_sec = 0.0
    replan_next = False
    t_start = time.time()
    while True:
        if pending:
            t_r = time.time()
            changed_any = False
            while True:
                t = float(env.next_schedule_time[0])
                ready = [o for o in pending
                         if env.op_scheduled_flag[0, o]
                         and env.op_ct[0, o] <= t + EPS]
                if not ready:
                    break
                cmin = min(env.op_ct[0, o] for o in ready)
                batch = [o for o in ready if env.op_ct[0, o] <= cmin + EPS]
                changed = False
                for o in batch:
                    pending.discard(o)
                    if not is_last[o]:
                        n_rev += 1
                    if reveal_lag(env, o, float(realized_lag[o]), op_job,
                                  is_last):
                        changed = True
                        n_rep += 1
                if changed:
                    changed_any = True
                    refresh_env_state(env)
            if changed_any:
                rep_sec += time.time() - t_r
                replan_next = True
        t_d = time.time()
        a = int(select(env))
        if replan_next:
            rep_sec += time.time() - t_d
            replan_next = False
        job = a // M
        assigned[env.candidate[0, job]] = a % M
        _, _, done = env.step(np.array([a]))
        if done.all():
            break
    wall = time.time() - t_start
    assert (assigned >= 0).all(), 'closed_loop: some op never scheduled'
    return {'makespan': float(env.current_makespan[0]),
            'assigned_mch': assigned,
            'true_op_ct': env.true_op_ct[0].copy(),
            'n_reveals': n_rev, 'n_replans': n_rep,
            'replan_seconds': rep_sec, 'wall_seconds': wall}


def rolling_cpsat(jl, pt, nominal_lag, realized_lag, plan_mch, plan_start,
                  resolve_budget, num_workers):
    """Rolling residual CP-SAT from a semi-active nominal plan (see header)."""
    jl = np.asarray(jl, dtype=int)
    n = pt.shape[0]
    op_job, is_last = job_structure(jl)
    nominal = np.asarray(nominal_lag, dtype=float)
    realized = np.asarray(realized_lag, dtype=float)
    known = nominal.copy()
    mch = np.asarray(plan_mch, dtype=int).copy()
    start, ct = retime(jl, pt, known, mch, plan_start, np.zeros(n))
    pending = set(np.where(nominal > 0)[0].tolist())
    n_rev = n_rep = n_improved = n_nosol = 0
    rep_sec = 0.0
    timings = []
    t_wall = time.time()
    while pending:
        t = min(ct[o] for o in pending)
        batch = [o for o in pending if ct[o] <= t + EPS]
        changed = False
        for o in batch:
            pending.discard(o)
            known[o] = realized[o]
            if not is_last[o]:
                n_rev += 1
                if realized[o] != nominal[o]:
                    changed = True
        if not changed:
            continue
        t1 = time.time()
        committed = start < t - EPS
        floor = np.where(committed, start, t)
        st_a, ct_a = retime(jl, pt, known, mch, start, floor)
        assert np.allclose(st_a[committed], start[committed]), \
            'rolling: retime moved a started op'
        sol = cpsat_residual(jl, pt, known, t, committed, mch, start,
                             resolve_budget, num_workers,
                             hint_mch=mch, hint_start=st_a, seed=n_rep)
        timings.append(sol['timing'])
        new_mch, new_st, new_ct = mch, st_a, ct_a
        if sol['assigned_mch'] is not None:
            st_b, ct_b = retime(jl, pt, known, sol['assigned_mch'],
                                sol['start'], floor)
            assert np.allclose(st_b[committed], start[committed]), \
                'rolling: re-solve moved a started op'
            if ct_b.max() < ct_a.max() - EPS:
                new_mch, new_st, new_ct = sol['assigned_mch'], st_b, ct_b
                n_improved += 1
        else:
            n_nosol += 1
        mch, start, ct = np.asarray(new_mch, dtype=int).copy(), new_st, new_ct
        n_rep += 1
        rep_sec += time.time() - t1
    return {'makespan': float(ct.max()), 'assigned_mch': mch, 'true_op_ct': ct,
            'n_reveals': n_rev, 'n_replans': n_rep, 'replan_seconds': rep_sec,
            'wall_seconds': time.time() - t_wall, 'n_improved': n_improved,
            'n_no_solution': n_nosol, 'timed_solves': timings}


# ---------------------------------------------------------------------------
# JSONL store
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
                continue  # half-written final line of an interrupted run
    return recs


def append_jsonl(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rec_key(instance, cv, rep, sig):
    return (instance, round(float(cv), 6), int(rep), sig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description='Closed-loop execution under random lag realizations')
    ap.add_argument('--model_name', type=str, default='10x25+ppvc-mixed+full')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed')
    ap.add_argument('--cv', type=float, nargs='+', default=[0.1, 0.2, 0.3],
                    help='coefficients of variation of the lag factor')
    ap.add_argument('--n_rep', type=int, default=3,
                    help='lag realizations per (instance, cv)')
    ap.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS),
                    choices=list(ALL_METHODS))
    ap.add_argument('--rolling_cpsat', action='store_true',
                    help='also run the rolling residual CP-SAT arm (expensive)')
    ap.add_argument('--cpsat_budget', type=float, default=60.0,
                    help='seconds for the nominal CP-SAT plan')
    ap.add_argument('--resolve_budget', type=float, default=5.0,
                    help='seconds per rolling residual re-solve')
    ap.add_argument('--cpsat_workers', type=int, default=2)
    ap.add_argument('--cpsat_hint', choices=('none', 'spt'), default='none',
                    help='nominal CP-SAT plan: cold solve (none) or started '
                         'from the lag-aware SPT schedule as a full hint (spt, '
                         'the mode of eval_cpsat_anytime.py --mode warm-spt)')
    ap.add_argument('--seed_test', type=int, default=50,
                    help='torch seed; SPT tie-break seed is seed_test + index')
    ap.add_argument('--lag_seed', type=int, default=2027,
                    help='base seed of the lag realizations')
    ap.add_argument('--max_instances', type=int, default=0)
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--device', type=str, default='cpu')
    return ap.parse_args(argv)


def main():
    args = parse_cli()
    sys.argv = [sys.argv[0]]  # params.py parses argv at import time
    methods = list(dict.fromkeys(
        args.methods + (['cpsat_rolling'] if args.rolling_cpsat else [])))
    need_policy = any(m in POLICY_METHODS for m in methods)
    need_cpsat = any(m in CPSAT_METHODS for m in methods)

    ep = _import_with_retry('eval_ppvc')
    from params import configs
    route = ep.detect_route_variant(args.data_path, 'auto')
    if route != 'fixed':
        sys.exit('[stochastic_lags] only fixed-route datasets are supported '
                 '(the reveal logic mirrors FJSPEnvForSameOpNums)')
    ep.ROUTE_VARIANT = 'fixed'

    use_lag_features = use_type_embedding = False
    if need_policy:
        ep.load_train_config(args.model_name)
        use_lag_features = bool(getattr(configs, 'use_lag_features', False))
        use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))
    configs.device = args.device
    if str(configs.device).startswith('cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', configs.device_id)
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cu = _import_with_retry('common_utils')
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    cu.setup_seed(args.seed_test)

    ppo = None
    if need_policy:
        from model.PPO import PPO_initialize
        ckpt = f'./trained_network/PPVC/{args.model_name}.pth'
        if not os.path.exists(ckpt):
            sys.exit(f'[stochastic_lags] checkpoint not found: {ckpt}')
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(torch.load(ckpt, map_location=device))
        ppo.policy.eval()

    def policy_select(env):
        st = env.state
        with torch.no_grad():
            pi, _ = ppo.policy(fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor,
                               candidate=st.candidate_tensor,
                               fea_m=st.fea_m_tensor,
                               mch_mask=st.mch_mask_tensor,
                               comp_idx=st.comp_idx_tensor,
                               dynamic_pair_mask=st.dynamic_pair_mask_tensor,
                               fea_pairs=st.fea_pairs_tensor,
                               op_type=st.op_type_tensor,
                               mch_type=st.mch_type_tensor)
        return int(cu.greedy_select_action(pi).cpu().numpy()[0])

    blf = bool(getattr(ep, '_model_bound_lag_free', lambda: False)())

    def policy_env(meta_):
        # the policy's env carries its lower-bound variant; older eval_ppvc
        # versions have no bound_lag_free argument
        if blf:
            return ep._new_env(jl_cur[0], pt_cur[0], meta_, use_lag_features,
                               use_type_embedding, bound_lag_free=True)[0]
        return ep._new_env(jl_cur[0], pt_cur[0], meta_, use_lag_features,
                           use_type_embedding)[0]

    jl_cur, pt_cur = [None], [None]

    def spt_select(env):
        return int(cu.heuristic_select_action('SPT', env))

    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    from right_shift_repair import right_shift_repair

    stems = sorted(p[:-len('.fjs')] for p in
                   glob.glob(os.path.join(args.data_path, 'instance_*.fjs')))
    if args.max_instances > 0:
        stems = stems[:args.max_instances]
    if not stems:
        sys.exit(f'[stochastic_lags] no instances under {args.data_path}')
    dataset = os.path.basename(os.path.normpath(args.data_path))
    save_dir = args.save_dir or f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    sig = {m: args.model_name for m in POLICY_METHODS}
    sig['spt_reactive'] = 'SPT'
    cp_sig = f'cpsat{args.cpsat_budget:g}s-w{args.cpsat_workers}'
    if args.cpsat_hint != 'none':
        cp_sig += f'-hint{args.cpsat_hint}'
    sig['cpsat_repair'] = cp_sig
    sig['cpsat_rolling'] = f'{cp_sig}-re{args.resolve_budget:g}s'
    for m in sig:
        sig[m] = f'{sig[m]}|lagseed{args.lag_seed}'
    paths = {m: os.path.join(save_dir, f'stochastic_lags_{m}.jsonl')
             for m in methods}
    done = {m: {rec_key(r['instance'], r['cv'], r['rep'], r['sig'])
                for r in read_jsonl(paths[m])} for m in methods}
    cp_cache_path = os.path.join(save_dir, 'stochastic_lags_cpsat_nominal.jsonl')

    print('=' * 72)
    print(f'stochastic lags  dataset={dataset}  instances={len(stems)}  '
          f'cv={args.cv}  n_rep={args.n_rep}')
    print(f'  methods={methods}  model={args.model_name if need_policy else "-"}'
          f'  cpsat_budget={args.cpsat_budget}  resolve_budget='
          f'{args.resolve_budget}  workers={args.cpsat_workers}'
          f'  cpsat_hint={args.cpsat_hint}')
    print('=' * 72, flush=True)

    n_infeasible = 0
    for idx, stem in enumerate(stems):
        base = os.path.basename(stem)
        todo = [(cv, rep, m) for cv in args.cv for rep in range(args.n_rep)
                for m in methods
                if rec_key(base, cv, rep, sig[m]) not in done[m]]
        if not todo:
            continue
        jl, pt, meta = load_instance(stem)
        jl_cur[0], pt_cur[0] = jl, pt
        nominal = np.asarray(meta['time_lag'], dtype=float)
        todo_m = {m for _, _, m in todo}

        # ---- nominal plans (computed only when a pending record needs them)
        pol_nom = spt_nom = cp_nom = None
        if todo_m & set(POLICY_METHODS):
            env = policy_env(meta)
            pol_nom = closed_loop(env, policy_select, nominal, jl, reveal=False)
        if 'spt_reactive' in todo_m:
            np.random.seed(args.seed_test + idx)
            env, _ = ep._new_env(jl, pt, meta, False, False)
            spt_nom = closed_loop(env, spt_select, nominal, jl, reveal=False)
        if todo_m & set(CPSAT_METHODS):
            key = nominal_cache_key(base, args)
            hit = [r for r in read_jsonl(cp_cache_path) if r['key'] == key]
            if hit:
                c = hit[-1]
            else:
                n = pt.shape[0]
                la_nom = loadavg_now()
                if args.cpsat_hint == 'spt':
                    sol = cpsat_nominal_warm_spt(
                        jl, pt, meta, args.seed_test + idx, args.cpsat_budget,
                        args.cpsat_workers, data_dir=args.data_path, seed=0)
                else:
                    sol = cpsat_residual(jl, pt, nominal, 0, np.zeros(n, bool),
                                         None, None, args.cpsat_budget,
                                         args.cpsat_workers, seed=0)
                assert sol['assigned_mch'] is not None, \
                    f'{base}: nominal CP-SAT found no solution ({sol["status"]})'
                c = {'key': key, 'instance': base,
                     'status': sol['status'], 'objective': sol['objective'],
                     'seconds': sol['seconds'],
                     'assigned_mch': sol['assigned_mch'].tolist(),
                     'start': sol['start'].tolist(),
                     'timing': sol['timing'], 'affinity': affinity_list(),
                     'loadavg_start': la_nom}
                if args.cpsat_hint != 'none':
                    c.update({k: sol[k] for k in (
                        'spt_makespan', 'spt_rollout_s', 'spt_seed',
                        'hint_makespan', 'hint_first_solution', 'hint_taken',
                        'bound', 'n_improving_solutions')})
                    c.update(cpsat_hint=args.cpsat_hint,
                             spt_seed_rule='per-instance',
                             formulation='eval_cpsat_anytime.build_lag_model',
                             loadavg_end=loadavg_now())
                    if not sol['objective'] <= sol['spt_makespan'] + EPS:
                        print(f'  !!! {base}: nominal CP-SAT objective '
                              f'{sol["objective"]} > SPT makespan '
                              f'{sol["spt_makespan"]}', flush=True)
                    print(f'  [nominal] {base} hint={sol["hint_makespan"]:g} '
                          f'first={sol["hint_first_solution"]} '
                          f'taken={sol["hint_taken"]} obj={sol["objective"]:g} '
                          f'bound={sol["bound"]:g} status={sol["status"]} '
                          f'solve_s={sol["seconds"]:.1f} cpu_share='
                          f'{sol["timing"]["cpu_share"]}', flush=True)
                append_jsonl(cp_cache_path, c)
            cmch = np.asarray(c['assigned_mch'], dtype=int)
            cst, cct = retime(jl, pt, nominal, cmch, np.asarray(c['start']),
                              np.zeros(pt.shape[0]))
            chk = validate_schedule(jl, pt, nominal, cmch, cct)
            assert chk['feasible'], f'{base}: nominal CP-SAT plan infeasible'
            cp_nom = {'makespan': float(cct.max()), 'assigned_mch': cmch,
                      'start': cst, 'true_op_ct': cct, 'status': c['status'],
                      'objective': c['objective'], 'seconds': c['seconds'],
                      # None: cached by a run that kept no evidence
                      'timing': c.get('timing'),
                      'hint_makespan': c.get('hint_makespan')}

        for cv, rep, m in todo:
            la_rec = loadavg_now()
            realized = realize_lags(nominal, cv,
                                    scenario_rng(args.lag_seed, idx, cv, rep))
            n_changed = int(np.sum(realized != nominal))
            extra = {}
            timed = {}
            if m == 'slab_reactive':
                env = policy_env(meta)
                r = closed_loop(env, policy_select, realized, jl, reveal=True)
                nom_ms, plan_sec = pol_nom['makespan'], pol_nom['wall_seconds']
            elif m == 'spt_reactive':
                np.random.seed(args.seed_test + idx)
                env, _ = ep._new_env(jl, pt, meta, False, False)
                r = closed_loop(env, spt_select, realized, jl, reveal=True)
                nom_ms, plan_sec = spt_nom['makespan'], spt_nom['wall_seconds']
            elif m == 'slab_perfect_info':
                meta_r = dict(meta)
                meta_r['time_lag'] = realized
                env = policy_env(meta_r)
                r = closed_loop(env, policy_select, realized, jl, reveal=False)
                r['n_reveals'] = r['n_replans'] = 0
                nom_ms, plan_sec = pol_nom['makespan'], r['wall_seconds']
            elif m in ('slab_repair', 'cpsat_repair'):
                nom = pol_nom if m == 'slab_repair' else cp_nom
                t1 = time.time()
                rct, rms = right_shift_repair(
                    jl, pt, realized, nom['assigned_mch'],
                    op_ct_lagblind=nom['true_op_ct'])
                sec = time.time() - t1
                _, is_last = job_structure(jl)
                lagged = (nominal > 0) & ~is_last
                r = {'makespan': float(rms), 'assigned_mch': nom['assigned_mch'],
                     'true_op_ct': rct, 'n_reveals': int(lagged.sum()),
                     'n_replans': int(np.sum(lagged & (realized != nominal))),
                     'replan_seconds': sec, 'wall_seconds': sec}
                nom_ms = nom['makespan']
                plan_sec = (pol_nom['wall_seconds'] if m == 'slab_repair'
                            else cp_nom['seconds'])
                if m == 'cpsat_repair':
                    extra = {'cpsat_status': cp_nom['status'],
                             'cpsat_objective': cp_nom['objective']}
                    if args.cpsat_hint != 'none':
                        extra.update(cpsat_hint=args.cpsat_hint,
                                     cpsat_hint_makespan=cp_nom['hint_makespan'])
                    timed = {'nominal_solve_timing': cp_nom['timing']}
            elif m == 'cpsat_rolling':
                r = rolling_cpsat(jl, pt, nominal, realized,
                                  cp_nom['assigned_mch'], cp_nom['start'],
                                  args.resolve_budget, args.cpsat_workers)
                nom_ms, plan_sec = cp_nom['makespan'], cp_nom['seconds']
                extra = {'cpsat_status': cp_nom['status'],
                         'cpsat_objective': cp_nom['objective'],
                         'n_resolve_improved': r['n_improved'],
                         'n_resolve_no_solution': r['n_no_solution']}
                timed = {'nominal_solve_timing': cp_nom['timing'],
                         'resolve_timings': r['timed_solves']}
            else:
                raise ValueError(m)

            res = validate_schedule(jl, pt, realized, r['assigned_mch'],
                                    r['true_op_ct'])
            if not res['feasible']:
                n_infeasible += 1
                print(f'  !!! {m} INFEASIBLE on {base} cv={cv} rep={rep}: '
                      f'{res["violations"][:3]}', flush=True)
            assert abs(res['makespan'] - r['makespan']) < 1e-6, (
                f'{m} {base}: makespan {r["makespan"]} != validator '
                f'{res["makespan"]}')
            rec = {'instance': base, 'instance_idx': idx, 'dataset': dataset,
                   'cv': float(cv), 'rep': int(rep), 'method': m,
                   'sig': sig[m],
                   'model_name': args.model_name if m in POLICY_METHODS
                   else None,
                   'realized_makespan': r['makespan'],
                   'nominal_plan_makespan': nom_ms,
                   'inflation_pct': 100.0 * (r['makespan'] - nom_ms) / nom_ms,
                   'n_lags_changed': n_changed,
                   'n_reveals': int(r['n_reveals']),
                   'n_replans': int(r['n_replans']),
                   'replan_seconds': float(r['replan_seconds']),
                   'plan_seconds': float(plan_sec),
                   'wall_seconds': float(r['wall_seconds']),
                   'feasible': bool(res['feasible']),
                   'n_violations': len(res['violations'])}
            rec.update(extra)
            # contention evidence over the timed solves this record's result
            # depends on: the (cached) nominal plan and every re-solve
            solves = ([timed['nominal_solve_timing']] if timed else []) \
                + timed.get('resolve_timings', [])
            rec.update(timed)
            rec.update(contention_fields(solves, la_rec))
            append_jsonl(paths[m], rec)
            done[m].add(rec_key(base, cv, rep, sig[m]))
            print(f'  [{idx + 1}/{len(stems)}] {base} cv={cv:g} rep={rep} '
                  f'{m:<18} realized={r["makespan"]:6.0f}  nominal={nom_ms:6.0f}'
                  f'  replans={r["n_replans"]:3d}  replan_s='
                  f'{r["replan_seconds"]:7.3f}  wall_s={r["wall_seconds"]:7.3f}'
                  f'  feasible={res["feasible"]}'
                  + (f'  cpu_share_min={rec["cpu_share_min"]}'
                     f'  n_low_share={rec["n_low_share"]}' if timed else ''),
                  flush=True)

    write_summary(args, methods, sig, paths, stems, save_dir, dataset)
    print(f'total infeasible schedules: {n_infeasible}')
    return n_infeasible


def write_summary(args, methods, sig, paths, stems, save_dir, dataset):
    bases = {os.path.basename(s) for s in stems}
    cvs = [round(float(c), 6) for c in args.cv]
    recs = {}
    for m in methods:
        for r in read_jsonl(paths[m]):
            if (r['sig'] == sig[m] and r['instance'] in bases
                    and round(r['cv'], 6) in cvs and r['rep'] < args.n_rep):
                recs[rec_key(r['instance'], r['cv'], r['rep'], m)] = r
    lines = [f'# Closed-loop execution under uncertain lags — {dataset}', '',
             f'- instances: {len(bases)}; replications: {args.n_rep}; '
             f'lag seed: {args.lag_seed}',
             f'- lag model: realized = round(L * X), X lognormal (mean 1, cv), '
             f'truncated to [{LAG_TRUNC[0]}, {LAG_TRUNC[1]}]; revealed at the '
             f'op\'s completion',
             f'- policy: `{args.model_name}`; nominal CP-SAT budget '
             f'{args.cpsat_budget:g} s; re-solve budget {args.resolve_budget:g} '
             f's; CP-SAT workers {args.cpsat_workers}'
             + ('' if getattr(args, 'cpsat_hint', 'none') == 'none' else
                f'; nominal CP-SAT plan started from the {args.cpsat_hint} '
                f'schedule (full hint)'), '',
             '| cv | method | n | mean realized makespan | mean nominal plan '
             '| mean inflation % | gap to perfect info % | mean re-plans '
             '| mean re-plan s | mean wall s | feasible |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for cv in cvs:
        for m in methods:
            rows = [r for k, r in recs.items() if k[1] == cv and k[3] == m]
            if not rows:
                continue
            gaps = []
            for r in rows:
                pi = recs.get(rec_key(r['instance'], cv, r['rep'],
                                      'slab_perfect_info'))
                if pi is not None:
                    gaps.append(100.0 * (r['realized_makespan']
                                         - pi['realized_makespan'])
                                / pi['realized_makespan'])
            g = f'{np.mean(gaps):.2f}' if gaps else 'n/a'
            lines.append(
                f'| {cv:g} | {m} | {len(rows)} | '
                f'{np.mean([r["realized_makespan"] for r in rows]):.1f} | '
                f'{np.mean([r["nominal_plan_makespan"] for r in rows]):.1f} | '
                f'{np.mean([r["inflation_pct"] for r in rows]):.2f} | {g} | '
                f'{np.mean([r["n_replans"] for r in rows]):.1f} | '
                f'{np.mean([r["replan_seconds"] for r in rows]):.3f} | '
                f'{np.mean([r["wall_seconds"] for r in rows]):.3f} | '
                f'{sum(r["feasible"] for r in rows)}/{len(rows)} |')
    lines += ['', 'Inflation % = 100 * (realized - nominal plan) / nominal '
              'plan, where the nominal plan is the method\'s own schedule '
              'under nominal lags (the policy plan for slab_perfect_info). '
              'Gap to perfect info is paired per (instance, cv, rep).']
    timed_methods = [m for m in methods if m in CPSAT_METHODS]
    if timed_methods:
        summ = {m: contention_summary(
                    [r for k, r in recs.items() if k[3] == m])
                for m in timed_methods}
        # the nominal plan is solved once per instance and counted in every
        # record that uses it, so it also gets a row of its own
        key_set = {nominal_cache_key(b, args) for b in bases}
        nom = {}
        for c in read_jsonl(os.path.join(save_dir,
                                         'stochastic_lags_cpsat_nominal.jsonl')):
            if c.get('key') in key_set:
                nom[c['key']] = c
        summ['nominal CP-SAT plan (one per instance)'] = contention_summary(
            [contention_fields([c.get('timing')], c.get('loadavg_start'))
             for c in nom.values()])
        lines += [''] + contention_table(summ)
        lines.append('Each cpsat_repair and cpsat_rolling record counts the '
                     'cached nominal plan solve among its timed solves.')
    out = os.path.join(save_dir, 'stochastic_lags_summary.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out}')


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
