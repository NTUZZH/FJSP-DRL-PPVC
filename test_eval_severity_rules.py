"""Tests for the breakdown-severity option, the SPT / LLF rule re-planners and
the warm-started stability sweep (eval_reactive.py, eval_residual_cpsat.py).

Run (CPU only, two pinned cores):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_THREADS=1 \\
      taskset -c 14-15 python -m pytest -q test_eval_severity_rules.py

Uses the SLAB checkpoint of the disruption study
(10x25+ppvc-mixed+a1lagx05-2k-s301) and the first orders of
data/PPVC/10x25+ppvc-mixed. Checks:
  V1  --breakdown_hours absent or None gives the old outage (0.5 x mean
      processing time) and the same state fingerprint as before; H = 8 and
      H = 24 give an outage of exactly H hours on the busiest station
      (release floor and env free time t_d + H); the replicated legacy branch
      of eval_residual_cpsat builds the identical state (fingerprint) as
      eval_reactive for H in (None, 8, 24) and for lag_perturb;
  V2  the eval_reactive CLI with --breakdown_hours 8 writes a contract-clean
      run with the outage in the info string and the option in the meta, and
      the eval_residual_cpsat CLI with the same option and TORCH_THREADS=1
      reports the same per-order state fingerprints and SLAB makespans, a
      hint on every solve and no outage-window violation;
      --breakdown_hours with another disruption type is rejected;
  V3  adding rule_spt_reactive and rule_llf_reactive changes no other arm
      (SLAB, SPTL and right-shift equal the stored reactive_hstar run field by
      field), their schedules are feasible and contract-clean, and the SPT
      arm equals an in-process continuation with the SPT rule;
  V4  the penalized residual model accepts the right-shift hint (the hint
      check compares the makespan, not the penalized objective) and the
      warm-started sweep (--stability_hint right_shift) uses it on every
      solve, while the default sweep writes exactly its old keys;
  V5  --match_reactive passes with TORCH_THREADS=1 and stops the run on an
      order whose SLAB plan differs (4 torch threads, order 000).
About three minutes in total.
"""
import copy
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

MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
DATASET = '10x25+ppvc-mixed'
DATA = os.path.join('data', 'PPVC', DATASET)
STORED = os.path.join('test_results', 'PPVC', DATASET, 'reactive_hstar')
N_INST = 3
DPARAMS = {'lag_factor': 2.0, 'pt_factor': 1.5, 'shortage_factor': 0.5,
           'shortage_basis': 'op_mean'}
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', TORCH_THREADS='1', PYTHONPATH=REPO)
TIMING = {'seconds', 'loadavg_start', 'affinity', 'solve_wall_s',
          'solve_cpu_s', 'cpu_share', 'cpu_share_min', 'hit_limit',
          'n_hit_limit', 'n_low_share'}
OLD_SWEEP_KEYS = {'disruption_type', 'n', 'weights', 'coefs', 'budget',
                  'orig_mean', 'drl', 'frontier', 'meta', 'timing',
                  'contention'}


def _import_clean(name):
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        mod = __import__(name)
    finally:
        sys.argv = saved
    return mod


@functools.lru_cache(maxsize=None)
def _setup():
    er = _import_clean('eval_reactive')
    rc = _import_clean('eval_residual_cpsat')
    from params import configs
    er.load_train_config(MODEL)
    configs.device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from common_utils import setup_seed
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    setup_seed(50)
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/PPVC/{MODEL}.pth', map_location='cpu'))
    ppo.policy.eval()
    return er, rc, ppo


def _order(i):
    from ppvc_instance_generator import load_instance
    stem = os.path.join(DATA, f'instance_{i:03d}')
    jl, pt, meta = load_instance(stem)
    return stem, jl, pt, meta


def _er_state(i, dtype, dparams):
    er, _, ppo = _setup()
    stem, jl, pt, meta = _order(i)
    ds = er.build_disrupted_state(ppo, jl, pt, meta, False, False, dtype,
                                  0.4, dparams, er.instance_context(stem, meta))
    return ds, jl, pt


# --------------------------------------------------------------------------- #
# V1  outage length and identical states in both scripts
# --------------------------------------------------------------------------- #
def test_breakdown_hours_default_and_values():
    er, _, _ = _setup()
    for i in range(N_INST):
        ds0, _, pt = _er_state(i, 'breakdown', DPARAMS)
        dsn, _, _ = _er_state(i, 'breakdown',
                              dict(DPARAMS, breakdown_hours=None))
        mean_pt = float(pt[pt > 0].mean())
        assert ds0['downtime_true'] == 0.5 * mean_pt
        assert er.state_fingerprint(ds0, ds0['env']) == \
            er.state_fingerprint(dsn, dsn['env'])
        fps = {er.state_fingerprint(ds0, ds0['env'])}
        for h in (8.0, 24.0):
            ds, _, _ = _er_state(i, 'breakdown',
                                 dict(DPARAMS, breakdown_hours=h))
            m, t_d = ds['broken_mch'], ds['t_disrupt_true']
            assert m == ds0['broken_mch'] and t_d == ds0['t_disrupt_true']
            assert ds['downtime_true'] == h
            assert f'down {h:.2f}h' in ds['info']
            assert ds['mch_release'][m] == t_d + h
            assert np.count_nonzero(ds['mch_release']) == 1
            assert float(ds['env'].true_mch_free_time[0, m]) >= t_d + h - 1e-9
            # the prefix is the same as at the default severity
            assert np.array_equal(ds['committed'], ds0['committed'])
            assert np.array_equal(ds['committed_mch'], ds0['committed_mch'])
            fps.add(er.state_fingerprint(ds, ds['env']))
        assert len(fps) == 3          # the fingerprint sees the outage


@pytest.mark.parametrize('dtype,hours', [('breakdown', None),
                                         ('breakdown', 8.0),
                                         ('breakdown', 24.0),
                                         ('lag_perturb', None)])
def test_residual_script_builds_identical_state(dtype, hours):
    er, rc, ppo = _setup()
    dp = dict(DPARAMS)
    if hours is not None:
        dp['breakdown_hours'] = hours
    for i in range(N_INST):
        stem, jl, pt, meta = _order(i)
        ds, _, _ = _er_state(i, dtype, dp)
        disrupt, env, _ = rc.build_disrupted_state(
            ppo, jl, pt, meta, False, False, dtype, 0.4, 2.0, dparams=dp,
            stem=stem)
        assert er.state_fingerprint(disrupt, env) == \
            er.state_fingerprint(ds, ds['env']), (dtype, hours, i)
        if hours is not None:
            assert disrupt['downtime_true'] == hours
            assert disrupt['mch_ready_time'][disrupt['broken_mch']] == \
                disrupt['t_disrupt_true'] + hours


# --------------------------------------------------------------------------- #
# V2  CLIs with --breakdown_hours
# --------------------------------------------------------------------------- #
def _run(script, args, tmp, check=True, env=ENV):
    p = subprocess.run([PY, script, '--model_name', MODEL, '--data_path',
                        DATA] + args + ['--out_dir', str(tmp)], cwd=REPO,
                       env=env, capture_output=True, text=True, timeout=1200)
    if check:
        assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p


@pytest.fixture(scope='module')
def brk8(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('brk8')
    _run('eval_reactive.py',
         ['--disruption_type', 'breakdown', '--breakdown_hours', '8',
          '--max_instances', '2', '--device', 'cpu', '--cpsat_budgets', '',
          '--methods', 'drl_reactive,spt_reactive,ga_reactive,right_shift,'
                       'rule_spt_reactive,rule_llf_reactive',
          '--ga_budget', '0.5', '--state_fp'], tmp)
    _run('eval_residual_cpsat.py',
         ['--disruption_type', 'breakdown', '--breakdown_hours', '8',
          '--max_instances', '2', '--cpsat_budgets', '1', '--cpsat_workers',
          '2', '--hint', 'right_shift', '--state_fp'], tmp)
    a = np.load(tmp / f'reactive_breakdown_{MODEL}.npy',
                allow_pickle=True).item()
    b = np.load(tmp / 'residual_cpsat_breakdown.npy', allow_pickle=True).item()
    return a, b


def test_cli_breakdown_hours(brk8):
    a, b = brk8
    assert a['meta']['breakdown_hours'] == 8.0
    assert a['meta']['n_infeasible'] == 0
    assert a['meta']['n_contract_violations'] == 0
    assert all('down 8.00h' in s for s in a['info'])
    for m in a['methods']:
        assert a['records'][m]['feasible'].all(), m
    rs, ga = a['records']['right_shift'], a['records']['ga_reactive']
    assert (ga['makespan'] <= rs['makespan'] + 1e-6).all()
    # the residual run saw the identical states (same TORCH_THREADS)
    assert b['meta']['breakdown_hours'] == 8.0
    assert b['meta']['state_checked_against_eval_reactive']
    assert b['state_fp'] == a['state_fp'] and len(a['state_fp']) == 2
    assert np.array_equal(a['orig_makespans'], b['orig_makespans'])
    assert np.array_equal(a['records']['drl_reactive']['makespan'],
                          b['records']['drl_reactive']['makespan'])
    assert b['meta']['n_breakdown_window_violations'] == 0
    assert b['meta']['n_prefix_violations'] == 0
    assert b['meta']['n_infeasible'] == 0
    assert b['meta']['n_hint_failed'] == 0
    assert b['records']['residual_cpsat_1s']['hint_given'].all()


def test_breakdown_hours_rejected_for_other_types(tmp_path):
    p = _run('eval_reactive.py',
             ['--disruption_type', 'lag_perturb', '--breakdown_hours', '8',
              '--max_instances', '1', '--device', 'cpu', '--cpsat_budgets',
              ''], tmp_path, check=False)
    assert p.returncode != 0
    assert 'breakdown only' in (p.stdout + p.stderr)


# --------------------------------------------------------------------------- #
# V3  rule re-planners
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', ['lag_perturb', 'cancellation'])
def test_rule_arms_leave_other_arms_unchanged(tmp_path, dtype):
    env1 = dict(ENV)
    env1.pop('TORCH_THREADS')   # eval_reactive default (1), as stored
    _run('eval_reactive.py',
         ['--disruption_type', dtype, '--max_instances', '1', '--device',
          'cpu', '--cpsat_budgets', '', '--methods',
          'drl_reactive,spt_reactive,right_shift,rule_spt_reactive,'
          'rule_llf_reactive'], tmp_path, env=env1)
    new = np.load(tmp_path / f'reactive_{dtype}_{MODEL}.npy',
                  allow_pickle=True).item()
    old = np.load(os.path.join(STORED, f'reactive_{dtype}_{MODEL}.npy'),
                  allow_pickle=True).item()
    assert new['methods'] == ['drl_reactive', 'spt_reactive', 'right_shift',
                              'rule_spt_reactive', 'rule_llf_reactive']
    assert 'state_fp' not in new and 'breakdown_hours' not in new['meta']
    assert new['orig_makespans'][0] == old['orig_makespans'][0]
    for m in ('drl_reactive', 'spt_reactive', 'right_shift'):
        x, y = new['records'][m], old['records'][m]
        for k in x:
            if k in TIMING:
                continue
            u, v = x[k][0], y[k][0]
            if isinstance(u, float) and np.isnan(u):
                assert np.isnan(v), (m, k)
            else:
                assert u == v, (dtype, m, k, u, v)
    for m in ('rule_spt_reactive', 'rule_llf_reactive'):
        assert new['records'][m]['feasible'][0]
    assert new['meta']['n_infeasible'] == 0
    assert new['meta']['n_contract_violations'] == 0
    # the SPT arm equals an in-process continuation with the SPT rule
    er, _, ppo = _setup()
    ds, _, _ = _er_state(0, dtype, dict(DPARAMS))
    env_c = copy.deepcopy(ds['env'])
    np.random.seed(50)
    ms, _, _, _ = er.continue_residual(env_c, env_c.state,
                                       ds['assigned_mch'].copy(),
                                       er.rule_policy('SPT'))
    assert ms == new['records']['rule_spt_reactive']['makespan'][0]


# --------------------------------------------------------------------------- #
# V4  warm-started stability sweep
# --------------------------------------------------------------------------- #
def test_penalized_model_accepts_hint():
    er, rc, ppo = _setup()
    from schedule_validator import reconstruct_starts
    stem, jl, pt, meta = _order(0)
    _, _, asg_o, ct_o = er.rollout_greedy_full(ppo, jl, pt, meta, False,
                                               False)
    disrupt, env, _ = rc.build_disrupted_state(
        ppo, jl, pt, meta, False, False, 'breakdown', 0.4, 2.0,
        dparams=dict(DPARAMS, breakdown_hours=8.0), stem=stem)
    _, _, a_rs, c_rs = rc.right_shift_baseline(
        jl, pt, asg_o, ct_o, disrupt['perturbed_time_lag'],
        disrupt['mch_ready_time'])
    hint = rc.right_shift_hint(jl, pt, disrupt['perturbed_time_lag'],
                               disrupt, a_rs, c_rs)
    assert hint['ok']
    starts_o = reconstruct_starts(pt, asg_o, ct_o)
    for mk, pc in ((1, 0), (1, 1), (2, 1)):
        res = rc.residual_cpsat_solve(
            jl, pt, disrupt['perturbed_time_lag'], disrupt, 2, mk_coef=mk,
            pen_coef=pc, assigned_orig=asg_o, starts_orig=starts_o,
            num_workers=2, hint=hint)
        assert res['hint_given'], (mk, pc, res['hint_check_status'])
        assert res['status'] in ('OPTIMAL', 'FEASIBLE')
        if pc == 0:
            assert float(np.max(res['op_ct'])) <= hint['makespan'] + 1e-6


def test_sweep_with_hint_and_default_keys(tmp_path):
    _run('eval_residual_cpsat.py',
         ['--disruption_type', 'breakdown', '--max_instances', '1',
          '--cpsat_workers', '2', '--stability_sweep', '--stability_weights',
          '0,1', '--stability_budget', '2', '--stability_hint',
          'right_shift'], tmp_path / 'hint')
    d = np.load(tmp_path / 'hint' / 'stability_sweep_breakdown.npy',
                allow_pickle=True).item()
    assert d['meta']['stability_hint'] == 'right_shift'
    assert d['meta']['n_hint_failed'] == 0
    po = d['per_order']
    for w in (0.0, 1.0):
        assert po['frontier'][w]['hint_given'] == [True]
        assert po['frontier'][w]['valid'] == [True]
        assert set(d['frontier'][w]) == {'infl', 'nerv', 'mch_nerv',
                                         'time_nerv', 'sec', 'optimal',
                                         'feas'}
    assert po['frontier'][0.0]['ms'][0] <= po['hint_ms'][0] + 1e-6
    assert len(d['state_fp']) == 1 and len(po['drl_ms']) == 1
    _run('eval_residual_cpsat.py',
         ['--disruption_type', 'breakdown', '--max_instances', '1',
          '--cpsat_workers', '2', '--stability_sweep', '--stability_weights',
          '0,1', '--stability_budget', '2'], tmp_path / 'cold')
    c = np.load(tmp_path / 'cold' / 'stability_sweep_breakdown.npy',
                allow_pickle=True).item()
    assert set(c) == OLD_SWEEP_KEYS
    assert set(c['meta']) == {'offset', 'lag_factor', 'seed_test',
                              'cpsat_workers', 'low_cpu_share',
                              'hit_limit_frac'}


# --------------------------------------------------------------------------- #
# V5  --match_reactive: per-order identity with the eval_reactive run
# --------------------------------------------------------------------------- #
def test_match_reactive_passes_with_one_thread(tmp_path):
    ref = os.path.join(STORED, f'reactive_lag_perturb_{MODEL}.npy')
    _run('eval_residual_cpsat.py',
         ['--disruption_type', 'lag_perturb', '--max_instances', '2',
          '--cpsat_budgets', '1', '--cpsat_workers', '2', '--hint',
          'right_shift', '--match_reactive', ref], tmp_path)
    d = np.load(tmp_path / 'residual_cpsat_lag_perturb.npy',
                allow_pickle=True).item()
    assert d['meta']['n_matched_reactive'] == 2
    assert d['meta']['torch_threads'] == 1
    _run('eval_residual_cpsat.py',
         ['--disruption_type', 'lag_perturb', '--max_instances', '1',
          '--cpsat_workers', '2', '--stability_sweep', '--stability_weights',
          '0', '--stability_budget', '1', '--match_reactive', ref],
         tmp_path / 'sweep')
    s = np.load(tmp_path / 'sweep' / 'stability_sweep_lag_perturb.npy',
                allow_pickle=True).item()
    assert s['meta']['n_matched_reactive'] == 1


def test_match_reactive_stops_on_four_threads(tmp_path):
    # order 000 of 10x25 is one of the orders whose greedy SLAB plan changes
    # with 4 torch threads (the default of eval_residual_cpsat.py)
    ref = os.path.join(STORED, f'reactive_lag_perturb_{MODEL}.npy')
    env4 = dict(ENV, TORCH_THREADS='4')
    p = _run('eval_residual_cpsat.py',
             ['--disruption_type', 'lag_perturb', '--max_instances', '1',
              '--cpsat_budgets', '1', '--cpsat_workers', '2',
              '--match_reactive', ref], tmp_path, check=False, env=env4)
    assert p.returncode != 0
    assert 'AssertionError' in p.stderr and 'instance_000' in p.stderr
