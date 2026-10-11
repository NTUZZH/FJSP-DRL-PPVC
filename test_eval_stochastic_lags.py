"""Tests for eval_stochastic_lags.py and train.py's --ppvc_lag_cv.

Run:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 22-23 \
          python -m pytest -q -s test_eval_stochastic_lags.py
The closed-loop fixture runs every method on 3 instances of 10x25 at cv 0 and
0.2 with one replication (nominal CP-SAT 20 s, re-solves 1 s), about 3 minutes.
"""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import eval_stochastic_lags as esl

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = '10x25+ppvc-mixed'
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONPATH=REPO)
METHODS = ['slab_reactive', 'spt_reactive', 'slab_repair', 'cpsat_repair',
           'slab_perfect_info', 'cpsat_rolling']
GREEDY_DET = [196.0, 219.0, 214.0]
SPT_DET = [210.0, 218.0, 216.0]


# --------------------------------------------------------------------------- #
# Lag model
# --------------------------------------------------------------------------- #

def test_lag_model_properties():
    nominal = np.array([0, 12, 0, 20, 46, 13, 0, 30] * 500, dtype=float)
    same = esl.realize_lags(nominal, 0.0, np.random.default_rng(1))
    assert np.array_equal(same, nominal)
    r = esl.realize_lags(nominal, 0.3, esl.scenario_rng(2027, 0, 0.3, 0))
    assert np.array_equal(r[nominal == 0], nominal[nominal == 0])
    pos = nominal > 0
    assert np.array_equal(r, np.floor(r))
    ratio = r[pos] / nominal[pos]
    slack = 0.5 / nominal[pos]
    assert (ratio >= 0.5 - slack - 1e-12).all()
    assert (ratio <= 2.0 + slack + 1e-12).all()
    # same scenario -> same draw; another replication -> another draw
    r2 = esl.realize_lags(nominal, 0.3, esl.scenario_rng(2027, 0, 0.3, 0))
    r3 = esl.realize_lags(nominal, 0.3, esl.scenario_rng(2027, 0, 0.3, 1))
    assert np.array_equal(r, r2) and not np.array_equal(r, r3)
    # the factor has mean 1 and the requested cv (truncation at [0.5, 2]
    # removes under 0.1 % of the mass at cv 0.2)
    x = esl.sample_lag_factors(200000, 0.2, np.random.default_rng(5))
    assert x.min() >= 0.5 and x.max() <= 2.0
    assert abs(x.mean() - 1.0) < 0.005
    assert abs(x.std() / x.mean() - 0.2) < 0.005


def test_retime_matches_right_shift_repair():
    # load_instance imports params, which parses the process argv at import
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        from ppvc_instance_generator import load_instance
        from right_shift_repair import right_shift_repair
        jl, pt, meta = load_instance(os.path.join(REPO, 'data', 'PPVC',
                                                  DATASET, 'instance_000'))
    finally:
        sys.argv = saved
    rng = np.random.default_rng(0)
    n, M = pt.shape
    # a feasible plan: serial schedule of a random machine choice
    mch = np.array([rng.choice(np.where(pt[o] > 0)[0]) for o in range(n)])
    lag = np.asarray(meta['time_lag'], dtype=float)
    st, ct = esl.retime(jl, pt, lag, mch, np.arange(n, dtype=float),
                        np.zeros(n))
    real = esl.realize_lags(lag, 0.3, np.random.default_rng(3))
    _, ct_a = esl.retime(jl, pt, real, mch, st, np.zeros(n))
    ct_b, _ = right_shift_repair(jl, pt, real, mch, op_ct_lagblind=ct)
    assert np.allclose(ct_a, ct_b)


# --------------------------------------------------------------------------- #
# Closed-loop run: 3 instances, cv 0 and 0.2, one replication, every method
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def records(tmp_path_factory):
    out = tmp_path_factory.mktemp('stochastic_lags')
    cmd = [PY, os.path.join(REPO, 'eval_stochastic_lags.py'),
           '--data_path', os.path.join('data', 'PPVC', DATASET),
           '--cv', '0', '0.2', '--n_rep', '1', '--max_instances', '3',
           '--rolling_cpsat', '--cpsat_budget', '20', '--resolve_budget', '1',
           '--save_dir', str(out)]
    p = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True,
                       timeout=900)
    print(p.stdout[-6000:])
    assert p.returncode == 0, p.stderr[-4000:]
    recs = {}
    for m in METHODS:
        with open(os.path.join(out, f'stochastic_lags_{m}.jsonl')) as f:
            for line in f:
                r = json.loads(line)
                recs[(r['method'], r['cv'], r['instance_idx'])] = r
    return recs


def test_every_schedule_validator_clean(records):
    assert len(records) == len(METHODS) * 2 * 3
    for k, r in records.items():
        assert r['feasible'] and r['n_violations'] == 0, k


def test_cv0_reproduces_deterministic_makespans(records):
    for i in range(3):
        for m in ('slab_reactive', 'slab_repair', 'slab_perfect_info'):
            assert records[(m, 0.0, i)]['realized_makespan'] == GREEDY_DET[i]
        assert records[('spt_reactive', 0.0, i)]['realized_makespan'] == \
            SPT_DET[i]
        cp = records[('cpsat_repair', 0.0, i)]
        roll = records[('cpsat_rolling', 0.0, i)]
        assert cp['realized_makespan'] == cp['nominal_plan_makespan']
        assert roll['realized_makespan'] == cp['realized_makespan']
        assert cp['nominal_plan_makespan'] <= cp['cpsat_objective']
        for m in METHODS:
            assert records[(m, 0.0, i)]['n_replans'] == 0
            assert records[(m, 0.0, i)]['inflation_pct'] == 0.0


def test_cv02_lags_change_and_replans_happen(records):
    for i in range(3):
        assert records[('slab_reactive', 0.2, i)]['n_lags_changed'] > 0
        assert records[('slab_reactive', 0.2, i)]['n_replans'] > 0
        assert records[('cpsat_rolling', 0.2, i)]['n_replans'] > 0


def _mean(records, m, cv=0.2):
    return float(np.mean([records[(m, cv, i)]['realized_makespan']
                          for i in range(3)]))


def test_report_reactive_vs_repair(records):
    react, repair = _mean(records, 'slab_reactive'), _mean(records, 'slab_repair')
    print(f'\n[report] cv 0.2 mean realized: slab_reactive {react:.2f}, '
          f'slab_repair {repair:.2f}')
    if react > repair:
        warnings.warn(f'slab_reactive ({react:.2f}) > slab_repair '
                      f'({repair:.2f}) on average')


def test_report_perfect_info_vs_reactive(records):
    react = _mean(records, 'slab_reactive')
    pinfo = _mean(records, 'slab_perfect_info')
    print(f'\n[report] cv 0.2 mean realized: slab_perfect_info {pinfo:.2f}, '
          f'slab_reactive {react:.2f}')
    if pinfo > react:
        warnings.warn(f'slab_perfect_info ({pinfo:.2f}) > slab_reactive '
                      f'({react:.2f}) on average')


# --------------------------------------------------------------------------- #
# train.py --ppvc_lag_cv
# --------------------------------------------------------------------------- #

_SAMPLER = r'''
import json, random, sys, types
import numpy as np
cv = float(sys.argv[1])
sys.argv = [sys.argv[0], '--data_source', 'PPVC', '--ppvc_lag_cv', str(cv)]
from params import configs
import train
from ppvc_instance_generator import DEFAULT_FACTORY, ppvc_instance_generator
stub = types.SimpleNamespace(
    op_per_job_min=1, op_per_job_max=1, n_j=10, num_envs=4,
    data_source='PPVC', config=configs, ppvc_factory=DEFAULT_FACTORY,
    ppvc_variable=False)
stub.scale_lags = types.MethodType(train.Trainer.scale_lags, stub)
random.seed(300); np.random.seed(300)
jl, pt, lag, ot, mt = train.Trainer.sample_training_instances(stub)
after = np.random.get_state()[1][:8].tolist()
# independent reconstruction of the cv = 0 batch from the same seed stream
random.seed(300); np.random.seed(300)
[random.randint(1, 1) for _ in range(10)]
ref = []
for i in range(4):
    s = int(np.random.randint(0, 2 ** 31 - 1))
    ref.append(ppvc_instance_generator(
        n_modules=10, class_mix=configs.ppvc_mix,
        station_counts=DEFAULT_FACTORY, seed=s,
        route_variant=configs.ppvc_route_variant))
print(json.dumps({
    'cv_in_config': configs.ppvc_lag_cv,
    'snapshot_has_flag': 'ppvc_lag_cv' in json.loads(json.dumps(
        vars(configs), default=str)),
    'jl': [a.tolist() for a in jl], 'pt': [a.tolist() for a in pt],
    'lag': [np.asarray(a, float).tolist() for a in lag],
    'ref_jl': [r[0].tolist() for r in ref], 'ref_pt': [r[1].tolist() for r in ref],
    'ref_lag': [np.asarray(r[2]['time_lag'], float).tolist() for r in ref],
    'state_after': after,
    'state_ref': np.random.get_state()[1][:8].tolist()}))
'''


def _sample(cv):
    p = subprocess.run([PY, '-c', _SAMPLER, str(cv)], cwd=REPO, env=ENV,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_train_lag_cv_flag():
    zero = _sample(0.0)
    assert zero['cv_in_config'] == 0.0 and zero['snapshot_has_flag']
    # cv = 0: the batch equals the generator's instances for the same seed
    # stream, and the stream advanced exactly as without the flag
    assert zero['jl'] == zero['ref_jl'] and zero['pt'] == zero['ref_pt']
    assert zero['lag'] == zero['ref_lag']
    assert zero['state_after'] == zero['state_ref']

    noisy = _sample(0.2)
    assert noisy['cv_in_config'] == 0.2 and noisy['snapshot_has_flag']
    assert noisy['jl'] == zero['jl'] and noisy['pt'] == zero['pt']
    assert noisy['state_after'] == zero['state_after']
    n_changed = 0
    for a, b in zip(noisy['lag'], zero['lag']):
        a, b = np.array(a), np.array(b)
        assert np.array_equal(a[b == 0], b[b == 0])
        pos = b > 0
        r = a[pos] / b[pos]
        assert (r >= 0.5 - 0.5 / b[pos] - 1e-12).all()
        assert (r <= 2.0 + 0.5 / b[pos] + 1e-12).all()
        n_changed += int(np.sum(a != b))
    assert n_changed > 0
    assert _sample(0.2)['lag'] == noisy['lag']


def test_contention_evidence(records):
    """Every record carries the contention fields; a CP-SAT record counts the
    time-limited solves its result depends on (the nominal plan, plus every
    re-solve for cpsat_rolling)."""
    for (m, cv, i), r in records.items():
        assert set(esl.CONTENTION_KEYS) <= set(r), (m, cv, i)
        assert r['affinity'] == sorted(os.sched_getaffinity(0))
        assert len(r['loadavg_start']) == 3
        if m not in esl.CPSAT_METHODS:
            assert r['n_timed_solves'] == 0 and r['cpu_share_min'] is None
            continue
        nom = r['nominal_solve_timing']
        assert nom is not None and nom['time_limit_s'] == 20.0
        resolves = r.get('resolve_timings', [])
        if m == 'cpsat_rolling':
            assert len(resolves) == r['n_replans']
        solves = [nom] + resolves
        assert r['n_timed_solves'] == len(solves) and r['n_no_evidence'] == 0
        for t in solves:
            assert t['workers'] == 2 and 0 < t['cpu_share'] <= 1.2, t
        limited = [t['cpu_share'] for t in solves if t['hit_limit']]
        assert r['n_hit_limit'] == len(limited)
        assert r['cpu_share_min'] == (min(limited) if limited else None)
        assert r['n_low_share'] == sum(s < esl.LOW_CPU_SHARE for s in limited)
