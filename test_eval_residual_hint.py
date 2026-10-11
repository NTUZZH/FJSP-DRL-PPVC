"""Tests for the right-shift warm start of the residual CP-SAT arm
(eval_residual_cpsat.py --hint right_shift).

Run (CPU only, two pinned cores):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
      taskset -c 22-23 python -m pytest -q test_eval_residual_hint.py

Uses the SLAB checkpoint of the disruption study
(10x25+ppvc-mixed+a1lagx05-2k-s301) and data/PPVC/10x25+ppvc-mixed. Checks:
  H1  with --hint right_shift, 2-s budget, 2 workers, on the first three
      instances of every disruption type: the hint passes the feasibility
      check on every instance (no cold solve), the returned makespan is not
      above the right-shift makespan nor above the hint makespan, every
      schedule is validator-clean and keeps the committed prefix, and every
      record carries cpu_share;
  H2  where the right-shift times are already integer and prefix-respecting
      (doubled wait, urgent module, longer operation, cancellation) the hint
      equals the right-shift plan exactly;
  H3  --hint none reproduces the stored cold run (reactive_hstar,
      lag_perturb, instance_000, budgets 5 and 30 s, 2 workers): the
      deterministic arms (SLAB re-planning, right-shift) are identical field
      by field, the CP-SAT arms reach the same proven-optimal makespan, the
      record and meta keys are identical, and no hint field is written;
  H4  --disruption_type rejects an unknown name in a comma list.
About two minutes in total.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
DATA = os.path.join('data', 'PPVC', '10x25+ppvc-mixed')
STORED = os.path.join('test_results', 'PPVC', '10x25+ppvc-mixed',
                      'reactive_hstar', 'residual_cpsat_lag_perturb.npy')
PY = sys.executable
# same thread settings as the production campaign (TORCH_THREADS unset -> 4,
# as in the stored run; greedy decisions do not depend on it)
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', PYTHONPATH=REPO)
ENV.pop('TORCH_THREADS', None)
TYPES = ['lag_perturb', 'breakdown', 'urgent_arrival', 'material_shortage',
         'pt_deviation', 'cancellation']
EXACT_TYPES = ['lag_perturb', 'urgent_arrival', 'pt_deviation',
               'cancellation']
N_INST = 3


def _run(args, tmp, check=True):
    p = subprocess.run([PY, 'eval_residual_cpsat.py', '--model_name', MODEL,
                        '--data_path', DATA, '--cpsat_workers', '2']
                       + args + ['--out_dir', str(tmp)], cwd=REPO, env=ENV,
                       capture_output=True, text=True, timeout=1200)
    if check:
        assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p


@pytest.fixture(scope='module')
def hinted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('hint')
    p = _run(['--disruption_type', ','.join(TYPES), '--cpsat_budgets', '2',
              '--max_instances', str(N_INST), '--hint', 'right_shift'], tmp)
    out = {}
    for t in TYPES:
        out[t] = np.load(tmp / f'residual_cpsat_{t}.npy',
                         allow_pickle=True).item()
        out[t]['_md'] = (tmp / f'residual_cpsat_{t}_summary.md').read_text()
    return out, p.stdout


@pytest.mark.parametrize('dtype', TYPES)
def test_hint_accepted_and_not_worse(hinted, dtype):
    out, stdout = hinted
    d = out[dtype]
    assert d['meta']['hint'] == 'right_shift'
    assert d['meta']['n_hint_failed'] == 0
    assert d['meta']['n_prefix_violations'] == 0
    assert d['meta']['n_infeasible'] == 0
    assert d['meta']['n_breakdown_window_violations'] == 0
    r = d['records']['residual_cpsat_2s']
    rs = d['records']['right_shift']
    for i in range(N_INST):
        assert r['hint_given'][i], (dtype, i, r['hint_check_status'][i])
        assert r['hint_check_status'][i] == 'OPTIMAL'
        assert r['status'][i] in ('OPTIMAL', 'FEASIBLE')
        assert r['feasible'][i]
        ms = r['makespan'][i]
        assert ms == r['cpsat_makespan'][i]
        assert r['rs_makespan'][i] == rs['makespan'][i]
        # (i) never above the right-shift makespan, never above the hint
        assert ms <= rs['makespan'][i] + 1e-6, (dtype, i, ms, rs['makespan'][i])
        assert ms <= r['hint_makespan'][i] + 1e-6
        assert not r['above_rs'][i] and not r['above_hint'][i]
        assert 0 < r['cpu_share'][i] <= 1.2 and r['workers'][i] == 2
    h = d['hint_stats']['residual_cpsat_2s']
    assert h['n_hint_given'] == N_INST and h['n_above_rs'] == 0
    assert h['n_above_hint'] == 0


@pytest.mark.parametrize('dtype', EXACT_TYPES)
def test_hint_equals_right_shift_when_integer(hinted, dtype):
    out, _ = hinted
    r = out[dtype]['records']['residual_cpsat_2s']
    for i in range(N_INST):
        assert r['hint_exact_rs'][i], (dtype, i)
        assert r['hint_makespan'][i] == pytest.approx(r['rs_makespan'][i],
                                                      abs=1e-6)
        # the prefix-respecting right-shift agrees with the stored arm here
        assert r['rs_residual_makespan'][i] == pytest.approx(
            r['rs_makespan'][i], abs=1e-6)


def test_hint_summary_table(hinted):
    out, stdout = hinted
    assert stdout.count('right-shift hint: instances solved cold = 0') == 6
    assert stdout.count('hint residual_cpsat_2s: given 3/3') == 6
    for t in TYPES:
        assert '| residual_cpsat_2s | 3/3 |' in out[t]['_md']
        assert 'warm start: right-shift plan' in out[t]['_md']


def test_hint_none_reproduces_stored_run(tmp_path):
    stored = np.load(STORED, allow_pickle=True).item()
    _run(['--disruption_type', 'lag_perturb', '--cpsat_budgets', '5,30',
          '--max_instances', '1', '--hint', 'none'], tmp_path)
    new = np.load(tmp_path / 'residual_cpsat_lag_perturb.npy',
                  allow_pickle=True).item()
    assert set(new) == set(stored)               # no hint_stats entry
    assert new['methods'] == stored['methods']
    assert set(new['meta']) == set(stored['meta'])
    for k, v in stored['meta'].items():
        if k not in ('n_instances',):
            assert new['meta'][k] == v, k
    assert new['meta']['n_instances'] == 1
    assert new['orig_makespans'][0] == stored['orig_makespans'][0]
    timing_keys = {'seconds', 'loadavg_start', 'affinity', 'solve_wall_s',
                   'solve_cpu_s', 'cpu_share', 'cpu_share_min', 'hit_limit',
                   'n_hit_limit', 'n_low_share'}
    for m in stored['methods']:
        a, b = new['records'][m], stored['records'][m]
        assert set(a) == set(b), m
        if m in ('drl_reactive', 'right_shift'):
            for k in a:
                if k in timing_keys:
                    continue
                x, y = a[k][0], b[k][0]
                if isinstance(x, float) and np.isnan(x):
                    assert np.isnan(y), (m, k)
                else:
                    assert x == y, (m, k, x, y)
        else:
            # stored run proved this instance optimal at both budgets, so the
            # makespan is deterministic; the chosen optimum (nervousness) is
            # not, with two search workers
            assert b['status'][0] == 'OPTIMAL'
            assert a['status'][0] == 'OPTIMAL'
            assert a['makespan'][0] == b['makespan'][0]
            assert a['inflation'][0] == b['inflation'][0]
            assert a['feasible'][0] == b['feasible'][0]
            assert a['workers'][0] == b['workers'][0] == 2
            assert a['time_limit_s'][0] == b['time_limit_s'][0]
            assert not np.isnan(a['cpu_share'][0])
            assert 'hint_given' not in a


def test_bad_disruption_list_rejected(tmp_path):
    p = _run(['--disruption_type', 'lag_perturb,nonsense',
              '--max_instances', '1'], tmp_path, check=False)
    assert p.returncode != 0
    assert 'invalid disruption type' in p.stderr
