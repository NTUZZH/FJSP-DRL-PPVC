"""Tests for the contention evidence of the time-limited solves in
eval_reactive.py, eval_residual_cpsat.py and eval_stochastic_lags.py.

Run (CPU only, two pinned cores):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      taskset -c 18-19 python -m pytest -q test_cpu_share.py

Checks:
  C1  solve_timing / contention_fields / contention_summary arithmetic;
  C2  a time-limited CP-SAT solve (eval_stochastic_lags.cpsat_residual and
      eval_residual_cpsat.residual_cpsat_solve) returns its evidence with
      plausible values (0 < cpu_share <= 1.2, hit_limit at a 1 s budget);
  C3  ga_reactive fills its evidence, and the CP-SAT probe of cpsat_sweep
      leaves the solver class and an optimal tiny solve unchanged;
  C4  one-instance smoke runs of eval_reactive.py and eval_residual_cpsat.py
      (main run and stability sweep) write the new fields and the summary
      table.
About one and a half minutes in total.
"""
import functools
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import eval_stochastic_lags as esl  # noqa: E402  (no import-time side effects)

MODEL = '10x25+ppvc-mixed+full'
DATA = os.path.join('data', 'PPVC', '10x25+ppvc-mixed')
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', TORCH_THREADS='1',
           PYTORCH_NVML_BASED_CUDA_CHECK='1', PYTHONPATH=REPO)
DPARAMS = {'lag_factor': 2.0, 'pt_factor': 1.5, 'shortage_factor': 0.5,
           'shortage_basis': 'op_mean'}


def _import_clean(name):
    """Import a module that parses the process argv at import time."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        mod = __import__(name)
    finally:
        sys.argv = saved
    return mod


def _plausible(t, workers=None):
    assert set(esl.SOLVE_KEYS) <= set(t), sorted(t)
    assert t['solve_wall_s'] > 0 and t['solve_cpu_s'] >= 0
    assert 0 < t['cpu_share'] <= 1.2, t
    if workers is not None:
        assert t['workers'] == workers
    assert t['hit_limit'] == (
        t['solve_wall_s'] >= esl.HIT_LIMIT_FRAC * t['time_limit_s'])
    if t['status'] == 'FEASIBLE':   # not proven optimal: stopped at the limit
        assert t['hit_limit']


# --------------------------------------------------------------------------- #
# C1: arithmetic
# --------------------------------------------------------------------------- #
def test_timing_arithmetic():
    ok = esl.solve_timing(2.0, 3.8, 2, 2.0, 'FEASIBLE')
    assert ok['cpu_share'] == pytest.approx(0.95) and ok['hit_limit']
    low = esl.solve_timing(5.0, 4.0, 2, 5.0, 'FEASIBLE')
    assert low['cpu_share'] == pytest.approx(0.4) and low['hit_limit']
    early = esl.solve_timing(0.3, 0.1, 2, 5.0, 'OPTIMAL')
    assert not early['hit_limit']
    assert not esl.solve_timing(3.0, 3.0, 1, float('inf'), 'X')['hit_limit']
    assert esl.resolve_threads(0) == len(os.sched_getaffinity(0))
    assert esl.resolve_threads(3) == 3

    f = esl.contention_fields([ok, low, early, None], [1.0, 2.0, 3.0])
    assert f['cpu_share_min'] == pytest.approx(0.4)
    assert (f['n_timed_solves'], f['n_hit_limit'], f['n_low_share'],
            f['n_no_evidence']) == (4, 2, 1, 1)
    assert f['affinity'] == sorted(os.sched_getaffinity(0))
    assert f['loadavg_start'] == [1.0, 2.0, 3.0]
    none = esl.contention_fields([early], None)
    assert none['cpu_share_min'] is None and none['n_low_share'] == 0

    # dict-of-arrays storage and the run-level summary
    r = esl.contention_arrays(3, per_solve=True)
    esl.store_contention(r, 0, [ok], [1.0, 1.0, 1.0])
    esl.store_contention(r, 1, [low], [1.0, 1.0, 1.0])
    esl.store_contention(r, 2, [early], None)
    assert r['affinity'][0] == sorted(os.sched_getaffinity(0))
    assert r['status'][2] == 'OPTIMAL' and not r['hit_limit'][2]
    assert np.isnan(r['cpu_share_min'][2])
    s = esl.contention_summary(r)
    assert s['cpu_share_min'] == pytest.approx(0.4)
    assert (s['n_records'], s['n_timed_solves'], s['n_hit_limit'],
            s['n_low_share']) == (3, 3, 2, 1)
    # records written before the fields existed are counted, not skipped
    s2 = esl.contention_summary([{'realized_makespan': 1.0},
                                 dict(f)])
    assert s2['n_records_without_fields'] == 1 and s2['n_low_share'] == 1
    lines = esl.contention_table({'m': s, 'old': s2})
    assert any(line.startswith('| m | 3 | 3 | 2 | 0.400 | 1 |')
               for line in lines)


# --------------------------------------------------------------------------- #
# C2: CP-SAT residual solves carry their evidence
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _instance(i=0):
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        from ppvc_instance_generator import load_instance
        return load_instance(os.path.join(DATA, f'instance_{i:03d}'))
    finally:
        sys.argv = saved


def test_stochastic_cpsat_residual_timing():
    jl, pt, meta = _instance()
    n = pt.shape[0]
    sol = esl.cpsat_residual(jl, pt, meta['time_lag'], 0, np.zeros(n, bool),
                             None, None, 1.0, 2, seed=0)
    assert sol['status'] in ('OPTIMAL', 'FEASIBLE')
    t = sol['timing']
    _plausible(t, workers=2)
    assert t['status'] == sol['status'] and t['time_limit_s'] == 1.0


@functools.lru_cache(maxsize=None)
def _reactive_setup():
    er = _import_clean('eval_reactive')
    from params import configs
    er.load_train_config(MODEL)
    configs.device = 'cpu'
    import torch
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    from common_utils import setup_seed
    setup_seed(50)
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/PPVC/{MODEL}.pth', map_location='cpu'))
    ppo.policy.eval()
    use_lag = bool(getattr(configs, 'use_lag_features', False))
    use_type = bool(getattr(configs, 'use_type_embedding', False))
    jl, pt, meta = _instance()
    stem = os.path.join(DATA, 'instance_000')
    _, _, asg_o, ct_o = er.rollout_greedy_full(ppo, jl, pt, meta, use_lag,
                                               use_type)
    ds = er.build_disrupted_state(ppo, jl, pt, meta, use_lag, use_type,
                                  'pt_deviation', 0.4, DPARAMS,
                                  er.instance_context(stem, meta))
    return er, ds, pt, asg_o, ct_o


def test_residual_cpsat_solve_timing():
    er, ds, _, _, _ = _reactive_setup()
    rc = _import_clean('eval_residual_cpsat')
    disrupt = {k: v for k, v in ds.items() if k not in ('env', 'state')}
    res = rc.residual_cpsat_solve(ds['jl'], ds['pt'], ds['time_lag'],
                                  disrupt, 1, num_workers=2)
    assert res['status'] in ('OPTIMAL', 'FEASIBLE')
    _plausible(res['timing'], workers=2)
    assert res['timing']['status'] == res['status']
    assert res['timing']['solve_wall_s'] <= res['seconds'] + 0.05


# --------------------------------------------------------------------------- #
# C3: GA evidence; the cpsat_sweep probe is transparent
# --------------------------------------------------------------------------- #
def test_ga_reactive_timing():
    er, ds, pt, asg_o, ct_o = _reactive_setup()
    rs = er.right_shift_residual(ds, asg_o, ct_o, pt)
    t = {}
    ms, sec, _, _, _ = er.ga_reactive(ds, rs[2], rs[3], 0.5, pop_size=50,
                                      rng=np.random.default_rng(0), timing=t)
    _plausible(t, workers=1)
    assert t['status'] == 'budget' and t['hit_limit']
    assert t['time_limit_s'] == 0.5 and ms <= rs[0] + 1e-6


def test_cpsat_sweep_probe():
    er, ds, _, _, _ = _reactive_setup()
    from ortools.sat.python import cp_model
    cls = cp_model.CpSolver
    # a 1 s whole-instance solve: evidence recorded, worker cap honoured
    t = {}
    sw = er.cpsat_sweep(ds['jl'], ds['pt'], ds['time_lag'], [1],
                        ds['mch_ready_time'], 2, timing=t)
    assert cp_model.CpSolver is cls
    assert sw[1][2] in ('OPTIMAL', 'FEASIBLE')
    _plausible(t[1], workers=2)
    assert t[1]['status'] == sw[1][2] and len(t[1]['loadavg_start']) == 3
    # a tiny instance solved to optimality: same answer with and without the
    # probe, and the solve does not count as budget-limited
    jl = np.array([2, 2])
    pt = np.array([[3, 0], [2, 4], [0, 2], [5, 1]], dtype=int)
    lag = np.array([1.0, 0.0, 2.0, 0.0])
    plain = er.cpsat_sweep(jl, pt, lag, [5], None, 1)
    t = {}
    probed = er.cpsat_sweep(jl, pt, lag, [5], None, 1, timing=t)
    assert cp_model.CpSolver is cls
    assert plain[5][2] == probed[5][2] == 'OPTIMAL'
    assert plain[5][0] == probed[5][0]
    assert not t[5]['hit_limit'] and t[5]['workers'] == 1
    assert esl.contention_fields([t[5]], None)['cpu_share_min'] is None


# --------------------------------------------------------------------------- #
# C4: one-instance smoke runs of the two scripts
# --------------------------------------------------------------------------- #
def _run(args, tmp):
    p = subprocess.run([PY] + args + ['--out_dir', str(tmp)], cwd=REPO,
                       env=ENV, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p.stdout


def _check_arrays(r, i=0, timed=True):
    assert set(esl.CONTENTION_KEYS) <= set(r)
    assert r['affinity'][i] == sorted(os.sched_getaffinity(0))
    assert not np.isnan(r['loadavg_start'][i]).any()
    if timed:
        assert r['n_timed_solves'][i] == 1 and r['n_no_evidence'][i] == 0
        assert 0 < r['cpu_share'][i] <= 1.2
        # a solve that ends early (proved optimal) is not budget-limited
        assert bool(r['hit_limit'][i]) == bool(
            r['solve_wall_s'][i] >= esl.HIT_LIMIT_FRAC * r['time_limit_s'][i])
        assert r['n_hit_limit'][i] == int(r['hit_limit'][i])
        if r['hit_limit'][i]:
            assert r['cpu_share_min'][i] == r['cpu_share'][i]
        else:
            assert np.isnan(r['cpu_share_min'][i])
    else:
        assert r['n_timed_solves'][i] == 0 and np.isnan(r['cpu_share_min'][i])


def test_eval_reactive_smoke(tmp_path):
    out = _run(['eval_reactive.py', '--disruption_type', 'lag_perturb',
                '--max_instances', '1', '--ga_budget', '0.5',
                '--cpsat_budgets', '1', '--cpsat_workers', '2',
                '--device', 'cpu'], tmp_path)
    assert 'contention cpsat_1: min cpu_share' in out
    d = np.load(tmp_path / f'reactive_lag_perturb_{MODEL}.npy',
                allow_pickle=True).item()
    for m in ('ga_reactive', 'cpsat_1'):
        _check_arrays(d['records'][m])
    assert d['records']['cpsat_1']['workers'][0] == 2
    assert d['records']['ga_reactive']['status'][0] == 'budget'
    assert d['records']['ga_reactive']['hit_limit'][0]
    assert d['records']['cpsat_1']['status'][0] in ('OPTIMAL', 'FEASIBLE')
    for m in ('drl_reactive', 'spt_reactive', 'right_shift'):
        _check_arrays(d['records'][m], timed=False)
    assert d['contention']['cpsat_1']['n_timed_solves'] == 1
    md = (tmp_path / 'reactive_lag_perturb_summary.md').read_text()
    assert '| cpsat_1 | 1 | 1 |' in md and '| ga_reactive | 1 | 1 | 1 |' in md


def test_eval_residual_cpsat_smoke(tmp_path):
    out = _run(['eval_residual_cpsat.py', '--disruption_type', 'lag_perturb',
                '--max_instances', '1', '--cpsat_budgets', '1',
                '--cpsat_workers', '2'], tmp_path)
    assert 'contention residual_cpsat_1s: min cpu_share' in out
    d = np.load(tmp_path / 'residual_cpsat_lag_perturb.npy',
                allow_pickle=True).item()
    r = d['records']['residual_cpsat_1s']
    assert r['workers'][0] == 2 and r['status'][0] in ('OPTIMAL', 'FEASIBLE')
    _check_arrays(r)
    _check_arrays(d['records']['drl_reactive'], timed=False)
    md = (tmp_path / 'residual_cpsat_lag_perturb_summary.md').read_text()
    assert '| residual_cpsat_1s | 1 | 1 |' in md


def test_stability_sweep_smoke(tmp_path):
    out = _run(['eval_residual_cpsat.py', '--disruption_type', 'lag_perturb',
                '--max_instances', '1', '--cpsat_workers', '2',
                '--stability_sweep', '--stability_weights', '0,1',
                '--stability_budget', '1'], tmp_path)
    assert 'contention w=1.0: min cpu_share' in out
    d = np.load(tmp_path / 'stability_sweep_lag_perturb.npy',
                allow_pickle=True).item()
    for w in (0.0, 1.0):
        (t,) = d['timing'][w]
        assert t['instance'] == 'instance_000' and t['workers'] == 2
        assert 0 < t['cpu_share'] <= 1.2 and t['n_timed_solves'] == 1
        assert d['contention'][w]['n_records'] == 1
    # the frontier entries the figure script reads are unchanged in form
    assert set(d['frontier'][0.0]) == {'infl', 'nerv', 'mch_nerv',
                                       'time_nerv', 'sec', 'optimal', 'feas'}
