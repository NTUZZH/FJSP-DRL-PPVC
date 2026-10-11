"""Tests for the breakdown right-shift arm of eval_reactive.py, which now keeps
every executed (committed) operation fixed.

Run (CPU only, two pinned cores):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      taskset -c 22-23 python -m pytest -q test_eval_reactive_breakdown_rs.py

Uses the SLAB checkpoint of the disruption study
(10x25+ppvc-mixed+a1lagx05-2k-s301) and the first three orders of
data/PPVC/10x25+ppvc-mixed. Checks:
  B1  on breakdown, right_shift_arm never moves a committed operation (machine
      and completion time unchanged), respects every release floor, and gives
      a validator-clean schedule; the legacy operator it replaces does move
      committed operations on at least one of the three orders;
  B2  the new right-shift plan is the hint plan of the hinted residual CP-SAT
      records (eval_residual_cpsat.right_shift_hint): same station for every
      remaining operation, same order on every station, starts within the
      integer rounding of the downtime (< 1 h); where the hinted residual
      breakdown file exists (reactive_hstar_hint), its rs_residual_makespan
      equals the new right-shift makespan and its hint_makespan equals the
      hint built here (on the orders whose original plan matches: that run
      used 4 torch threads, which changes a few greedy decisions);
  B3  the eval_reactive.py CLI on breakdown writes a contract-clean
      right-shift arm and a GA arm never worse than it;
  B4  every other disruption type is unchanged: on order 000 the CLI's
      SLAB, SPTL and right-shift records equal the stored reactive_hstar run
      field by field.
About two minutes in total.
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

MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
DATASET = '10x25+ppvc-mixed'
DATA = os.path.join('data', 'PPVC', DATASET)
STORED = os.path.join('test_results', 'PPVC', DATASET, 'reactive_hstar')
HINTED = os.path.join('test_results', 'PPVC', DATASET, 'reactive_hstar_hint',
                      'residual_cpsat_breakdown.npy')
N_INST = 3
DPARAMS = {'lag_factor': 2.0, 'pt_factor': 1.5, 'shortage_factor': 0.5,
           'shortage_basis': 'op_mean'}
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONPATH=REPO)
ENV.pop('TORCH_THREADS', None)   # eval_reactive default (1), as stored


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
    return er, ppo


_ORIG = {}


def _orig_ms(i):
    _breakdown(i)
    return _ORIG[i]


@functools.lru_cache(maxsize=None)
def _breakdown(i):
    er, ppo = _setup()
    from ppvc_instance_generator import load_instance
    stem = os.path.join(DATA, f'instance_{i:03d}')
    jl, pt, meta = load_instance(stem)
    ms_o, _, asg_o, ct_o = er.rollout_greedy_full(ppo, jl, pt, meta, False,
                                                  False)
    _ORIG[i] = ms_o
    ds = er.build_disrupted_state(ppo, jl, pt, meta, False, False,
                                  'breakdown', 0.4, DPARAMS,
                                  er.instance_context(stem, meta))
    rs_res = er.right_shift_residual(ds, asg_o, ct_o, pt)
    new = er.right_shift_arm(ds, jl, pt, asg_o, ct_o, rs_res)
    old = er.right_shift_baseline(jl, pt, asg_o, ct_o, ds['time_lag'],
                                  ds['mch_ready_time'])
    return er, ds, jl, pt, asg_o, ct_o, new, old


def test_breakdown_right_shift_keeps_executed_ops():
    n_old_moved = 0
    from schedule_validator import validate_schedule
    for i in range(N_INST):
        er, ds, jl, pt, _, _, new, old = _breakdown(i)
        ms, _, asg, ct, contract = new
        assert contract
        assert er.check_residual_contract(ds, asg, ct) == []
        com = np.where(ds['committed'])[0]
        assert np.array_equal(asg[com], ds['committed_mch'][com])
        assert np.allclose(ct[com], ds['committed_ct'][com], atol=1e-9)
        res = validate_schedule(ds['jl'], ds['pt'], ds['time_lag'], asg, ct)
        assert res['feasible'] and abs(res['makespan'] - ms) < 1e-6
        if er.check_residual_contract(ds, old[2], old[3]):
            n_old_moved += 1
    assert n_old_moved >= 1


def test_breakdown_right_shift_equals_hint_plan():
    rc = _import_clean('eval_residual_cpsat')
    stored = (np.load(HINTED, allow_pickle=True).item()
              if os.path.exists(HINTED) else None)
    n_cmp = 0
    for i in range(N_INST):
        er, ds, jl, pt, asg_o, ct_o, new, old = _breakdown(i)
        ms, _, asg, ct, _ = new
        disrupt = {k: v for k, v in ds.items() if k not in ('env', 'state')}
        # the hint as the hinted campaign built it: from the legacy plan
        h = rc.right_shift_hint(ds['jl'], ds['pt'], ds['time_lag'], disrupt,
                                old[2], old[3])
        assert h['ok']
        unc = np.where(~ds['committed'])[0]
        assert np.array_equal(asg[unc], h['assigned_mch'][unc])
        p_new = np.array([pt[op, asg[op]] for op in unc], dtype=float)
        s_new = ct[unc] - p_new
        assert np.max(np.abs(s_new - h['start'][unc])) < 1.0
        for m in np.unique(asg[unc]):
            on = unc[asg[unc] == m]
            assert list(on[np.argsort(s_new[asg[unc] == m], kind='stable')]) \
                == list(on[np.argsort(h['start'][on], kind='stable')])
        assert abs(ms - h['makespan']) < 1.0
        # the stored campaign ran with 4 torch threads, which changes a few
        # greedy decisions; compare only orders whose original plan and legacy
        # right-shift match the ones built here (1 thread)
        if stored is not None and \
                stored['orig_makespans'][i] == _orig_ms(i) and abs(
                stored['records']['right_shift']['makespan'][i] - old[0]) < 1e-9:
            r = stored['records']['residual_cpsat_5s']
            assert r['rs_residual_makespan'][i] == pytest.approx(ms, abs=1e-9)
            assert r['hint_makespan'][i] == h['makespan']
            n_cmp += 1
    if stored is not None:
        assert n_cmp >= 1


def _cli(args, tmp):
    p = subprocess.run([PY, 'eval_reactive.py', '--model_name', MODEL,
                        '--data_path', DATA, '--device', 'cpu',
                        '--cpsat_budgets', '', '--out_dir', str(tmp)] + args,
                       cwd=REPO, env=ENV, capture_output=True, text=True,
                       timeout=900)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p.stdout


def test_breakdown_cli_contract_clean(tmp_path):
    _cli(['--disruption_type', 'breakdown', '--max_instances', str(N_INST),
          '--methods', 'ga_reactive,right_shift', '--ga_budget', '0.5'],
         tmp_path)
    d = np.load(tmp_path / f'reactive_breakdown_{MODEL}.npy',
                allow_pickle=True).item()
    assert d['meta']['n_contract_violations'] == 0
    assert d['meta']['n_infeasible'] == 0
    rs, ga = d['records']['right_shift'], d['records']['ga_reactive']
    assert rs['feasible'].all() and ga['feasible'].all()
    assert (ga['makespan'] <= rs['makespan'] + 1e-6).all()
    for i in range(N_INST):
        assert rs['makespan'][i] == pytest.approx(_breakdown(i)[6][0],
                                                  abs=1e-9)


@pytest.mark.parametrize('dtype', ['lag_perturb', 'urgent_arrival',
                                   'material_shortage', 'pt_deviation',
                                   'cancellation'])
def test_other_types_unchanged(tmp_path, dtype):
    extra = (['--shortage_basis', 'remaining_work', '--shortage_factor',
              '0.5'] if dtype == 'material_shortage' else [])
    _cli(['--disruption_type', dtype, '--max_instances', '1', '--methods',
          'drl_reactive,spt_reactive,right_shift'] + extra, tmp_path)
    new = np.load(tmp_path / f'reactive_{dtype}_{MODEL}.npy',
                  allow_pickle=True).item()
    old = np.load(os.path.join(STORED, f'reactive_{dtype}_{MODEL}.npy'),
                  allow_pickle=True).item()
    skip = {'seconds', 'loadavg_start', 'affinity'}
    for m in ('drl_reactive', 'spt_reactive', 'right_shift'):
        a, b = new['records'][m], old['records'][m]
        for k in a:
            if k in skip:
                continue
            x, y = a[k][0], b[k][0]
            if isinstance(x, float) and np.isnan(x):
                assert np.isnan(y), (dtype, m, k)
            else:
                assert x == y, (dtype, m, k, x, y)
