"""Tests for the extended reactive-rescheduling experiment (eval_reactive.py
and the new disruption types in eval_residual_cpsat.py).

Run (CPU, two cores):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 22-23 \\
      python -m pytest -q test_eval_reactive_ext.py

Uses the first three instances of data/PPVC/10x25+ppvc-mixed and the
10x25+ppvc-mixed+full checkpoint. Checks:
  T1  lag_perturb and breakdown reproduce the earlier per-instance numbers
      (DRL-reactive and right-shift makespan and nervousness);
  T2  every method on every new disruption type gives a validator-clean
      schedule on the realized instance, and every residual method keeps the
      committed prefix and the release floors;
  T3  cancellation removes exactly the cancelled job's remaining ops;
  T4  urgent_arrival adds exactly 22 ops (fixed routes);
  T5  ga_reactive is no worse than the reported right_shift arm on every
      instance of every disruption type;
  T6  the residual right-shift equals the legacy right-shift on lag_perturb.
The whole file takes about three minutes.
"""
import functools
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MODEL = '10x25+ppvc-mixed+full'
DATASET = '10x25+ppvc-mixed'
DATA = os.path.join('data', 'PPVC', DATASET)
N_INST = 3
GA_BUDGET = 1.0
CPSAT_BUDGET = 1
CPSAT_WORKERS = 2
DPARAMS = {'lag_factor': 2.0, 'pt_factor': 1.5, 'shortage_factor': 0.5,
           'shortage_basis': 'op_mean'}

# Per-instance numbers of the earlier eval_reactive.py (run on CPU with one
# torch thread, instances 000-002, offset 0.4, seed_test 50).
EXPECTED = {
    'lag_perturb': {
        'drl_makespan': [214.0, 219.0, 214.0], 'drl_comb_nerv': [31, 31, 34],
        'rs_makespan': [224.0, 219.0, 221.0], 'rs_comb_nerv': [31, 27, 44]},
    'breakdown': {
        'drl_makespan': [196.0, 219.0, 214.0], 'drl_comb_nerv': [0, 2, 0],
        'rs_makespan': [201.17668622, 220.17157585, 214.0],
        'rs_comb_nerv': [61, 91, 0]},
}


def _import_clean(name):
    """Import a module that parses the process argv at import time."""
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
    use_lag = bool(getattr(configs, 'use_lag_features', False))
    use_type = bool(getattr(configs, 'use_type_embedding', False))
    return er, ppo, use_lag, use_type


def _stem(i):
    return os.path.join(DATA, f'instance_{i:03d}')


@functools.lru_cache(maxsize=None)
def _original(i):
    er, ppo, use_lag, use_type = _setup()
    from ppvc_instance_generator import load_instance
    jl, pt, meta = load_instance(_stem(i))
    ms, _, asg, ct = er.rollout_greedy_full(ppo, jl, pt, meta, use_lag,
                                            use_type)
    return jl, pt, meta, ms, asg, ct


@functools.lru_cache(maxsize=None)
def _run(dtype, i, with_cpsat=True):
    """Every method on one disrupted instance; returns a result dict."""
    import copy
    er, ppo, use_lag, use_type = _setup()
    jl, pt, meta, ms_orig, asg_o, ct_o = _original(i)
    ictx = er.instance_context(_stem(i), meta)
    ds = er.build_disrupted_state(ppo, jl, pt, meta, use_lag, use_type,
                                  dtype, 0.4, DPARAMS, ictx)
    env_spt = copy.deepcopy(ds['env'])
    out = {'ds': ds, 'n_orig': pt.shape[0], 'jl': jl, 'meta': meta}
    sched = {}
    ms, _, a, c = er.continue_residual(ds['env'], ds['state'],
                                       ds['assigned_mch'].copy(),
                                       er.drl_policy(ppo))
    sched['drl_reactive'] = (ms, a, c)
    np.random.seed(50)
    ms, _, a, c = er.continue_residual(env_spt, env_spt.state,
                                       ds['assigned_mch'].copy(),
                                       er.rule_policy('SPTL'))
    sched['spt_reactive'] = (ms, a, c)
    rs_res = er.right_shift_residual(ds, asg_o, ct_o, pt)
    sched['right_shift_residual'] = (rs_res[0], rs_res[2], rs_res[3])
    if ds['legacy']:
        ms, _, a, c = er.right_shift_baseline(jl, pt, asg_o, ct_o,
                                              ds['time_lag'],
                                              ds['mch_ready_time'])
        sched['right_shift'] = (ms, a, c)
    else:
        sched['right_shift'] = sched['right_shift_residual']
    rng = np.random.default_rng(i)
    ms, _, a, c, _ = er.ga_reactive(ds, rs_res[2], rs_res[3], GA_BUDGET,
                                    pop_size=100, rng=rng)
    sched['ga_reactive'] = (ms, a, c)
    if with_cpsat:
        sw = er.cpsat_sweep(ds['jl'], ds['pt'], ds['time_lag'],
                            [CPSAT_BUDGET], ds['mch_ready_time'],
                            CPSAT_WORKERS)
        obj, _, _, a, c = sw[CPSAT_BUDGET]
        sched['cpsat_whole'] = (obj, a, c)
        rc = _import_clean('eval_residual_cpsat')
        disrupt = dict(ds)
        disrupt.pop('env')
        disrupt.pop('state')
        res = rc.residual_cpsat_solve(ds['jl'], ds['pt'], ds['time_lag'],
                                      disrupt, CPSAT_BUDGET,
                                      num_workers=CPSAT_WORKERS)
        assert res['status'] in ('OPTIMAL', 'FEASIBLE'), res['status']
        sched['residual_cpsat'] = (res['objective'], res['assigned_mch'],
                                   res['op_ct'])
    out['sched'] = sched
    out['orig'] = (pt, asg_o, ct_o)
    return out


# --------------------------------------------------------------------------- #
# T1: legacy types unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', ['lag_perturb', 'breakdown'])
def test_legacy_numbers_unchanged(dtype):
    er, _, _, _ = _setup()
    exp = EXPECTED[dtype]
    for i in range(N_INST):
        r = _run(dtype, i, with_cpsat=False)
        pt, asg_o, ct_o = r['orig']
        ds = r['ds']
        ms, a, c = r['sched']['drl_reactive']
        assert ms == pytest.approx(exp['drl_makespan'][i], abs=1e-6)
        cn = er.nervousness(r['jl'], pt, asg_o, ct_o, a, c)[2]
        assert cn == exp['drl_comb_nerv'][i]
        ms, a, c = r['sched']['right_shift']
        assert ms == pytest.approx(exp['rs_makespan'][i], abs=1e-6)
        cn = er.nervousness(r['jl'], pt, asg_o, ct_o, a, c)[2]
        assert cn == exp['rs_comb_nerv'][i]
        # the mapped nervousness equals the legacy definition here
        assert er.nervousness_mapped(pt, asg_o, ct_o, ds['pt'], a, c,
                                     ds['op_map'])[:3] == \
            er.nervousness(r['jl'], pt, asg_o, ct_o, a, c)


# --------------------------------------------------------------------------- #
# T2: validator-clean schedules on every new type, every method
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', list(('urgent_arrival', 'material_shortage',
                                        'pt_deviation', 'cancellation')))
def test_new_types_validator_clean(dtype):
    er, _, _, _ = _setup()
    from schedule_validator import validate_schedule
    for i in range(N_INST):
        r = _run(dtype, i)
        ds = r['ds']
        for name, (ms, a, c) in r['sched'].items():
            a = np.asarray(a, dtype=int)
            c = np.asarray(c, dtype=float)
            assert (a >= 0).all(), f'{dtype} {i} {name}: unassigned op'
            res = validate_schedule(ds['jl'], ds['pt'], ds['time_lag'], a, c)
            assert res['feasible'], (f'{dtype} {i} {name}: '
                                     f'{res["violations"][:3]}')
            assert res['makespan'] == pytest.approx(ms, abs=0.51)
            if name == 'cpsat_whole':
                continue   # whole re-solve from t=0: reference only
            tol = 0.51 if name == 'residual_cpsat' else 1e-6
            viol = er.check_residual_contract(ds, a, c, tol=tol)
            assert not viol, f'{dtype} {i} {name}: {viol[:3]}'


# --------------------------------------------------------------------------- #
# T3 / T4: op counts
# --------------------------------------------------------------------------- #
def test_cancellation_removes_remaining_ops():
    for i in range(N_INST):
        r = _run('cancellation', i)
        ds = r['ds']
        removed = ds['removed_ops']
        assert len(removed) > 0
        # every removed op belonged to one job and was uncommitted
        jl = np.asarray(r['jl'])
        job_of = np.repeat(np.arange(len(jl)), jl)
        assert len(set(job_of[removed].tolist())) == 1
        j = int(job_of[removed[0]])
        s, e = int(np.sum(jl[:j])), int(np.sum(jl[:j + 1]))
        env_ops_left = [op for op in range(s, e) if op not in removed]
        assert all(ds['op_map'][op] >= 0 for op in env_ops_left)
        assert ds['pt'].shape[0] == r['n_orig'] - len(removed)
        assert int(np.sum(ds['jl'])) == r['n_orig'] - len(removed)
        assert (ds['op_map'][removed] == -1).all()


def test_urgent_arrival_adds_22_ops():
    for i in range(N_INST):
        r = _run('urgent_arrival', i)
        ds = r['ds']
        assert ds['pt'].shape[0] == r['n_orig'] + 22
        assert int(np.sum(ds['jl'])) == r['n_orig'] + 22
        assert len(ds['jl']) == len(r['jl']) + 1
        assert ds['new_ops'] == list(range(r['n_orig'], r['n_orig'] + 22))
        # the new module is not scheduled before its release
        for name in ('drl_reactive', 'spt_reactive', 'ga_reactive',
                     'right_shift'):
            ms, a, c = r['sched'][name]
            first = ds['new_ops'][0]
            start = c[first] - ds['pt'][first, a[first]]
            assert start >= ds['t_disrupt_true'] - 1e-6


# --------------------------------------------------------------------------- #
# T5: GA never worse than right-shift; T6: residual == legacy right-shift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', ['lag_perturb', 'breakdown',
                                   'urgent_arrival', 'material_shortage',
                                   'pt_deviation', 'cancellation'])
def test_ga_not_worse_than_right_shift(dtype):
    for i in range(N_INST):
        r = _run(dtype, i, with_cpsat=dtype not in ('lag_perturb',
                                                    'breakdown'))
        ga = r['sched']['ga_reactive'][0]
        rs = r['sched']['right_shift'][0]
        assert ga <= rs + 1e-6, f'{dtype} {i}: ga {ga} > right_shift {rs}'


def test_residual_right_shift_matches_legacy_on_lag_perturb():
    for i in range(N_INST):
        r = _run('lag_perturb', i, with_cpsat=False)
        ms_l, a_l, c_l = r['sched']['right_shift']
        ms_r, a_r, c_r = r['sched']['right_shift_residual']
        assert ms_r == pytest.approx(ms_l, abs=1e-6)
        assert np.array_equal(np.asarray(a_l), np.asarray(a_r))
        assert np.allclose(np.asarray(c_l), np.asarray(c_r), atol=1e-6)
