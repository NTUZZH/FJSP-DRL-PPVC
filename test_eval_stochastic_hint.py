"""Tests for eval_stochastic_lags.py --cpsat_hint (nominal CP-SAT plan started
from the lag-aware SPT schedule, the mode of eval_cpsat_anytime.py warm-spt).

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c <2 free cores> \
          python -m pytest -q -s test_eval_stochastic_hint.py
Two CLI runs of cpsat_repair on the first 2 instances of 10x25 (3 cv x 3
replications, nominal CP-SAT 3 s, 2 workers), about 1 minute.
"""
import json
import os
import subprocess
import sys

import pytest

import eval_stochastic_lags as esl

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = '10x25+ppvc-mixed'
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           CUDA_VISIBLE_DEVICES='', PYTHONPATH=REPO)
# lag-aware SPT makespans of instances 000 and 001 with numpy seed 50 + id
# (test_eval_stochastic_lags.SPT_DET and the spt_makespan of
# or_solution/PPVC/anytime/10x25+ppvc-mixed_warm-spt_300s_w4.jsonl)
SPT_DET = [210.0, 218.0]
STORED = os.path.join(REPO, 'test_results', 'PPVC', DATASET, 'stoch_evidence')


def _run(out, hint):
    cmd = [PY, 'eval_stochastic_lags.py', '--data_path', f'data/PPVC/{DATASET}',
           '--max_instances', '2', '--methods', 'cpsat_repair',
           '--cpsat_budget', '3', '--cpsat_workers', '2',
           '--cv', '0.1', '0.2', '0.3', '--n_rep', '3', '--device', 'cpu',
           '--save_dir', str(out)]
    if hint is not None:
        cmd += ['--cpsat_hint', hint]
    p = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True,
                       timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    recs = esl.read_jsonl(os.path.join(out, 'stochastic_lags_cpsat_repair.jsonl'))
    nom = esl.read_jsonl(os.path.join(out, 'stochastic_lags_cpsat_nominal.jsonl'))
    return recs, nom, p.stdout


@pytest.fixture(scope='module')
def hinted(tmp_path_factory):
    return _run(tmp_path_factory.mktemp('hint_spt'), 'spt')


@pytest.fixture(scope='module')
def cold(tmp_path_factory):
    return _run(tmp_path_factory.mktemp('hint_none'), None)


def test_hint_is_accepted(hinted):
    recs, nom, _ = hinted
    assert len(nom) == 2 and len(recs) == 18
    for i, c in enumerate(sorted(nom, key=lambda c: c['instance'])):
        assert c['key'] == f'instance_{i:03d}|3|2|hint-spt'
        assert c['spt_seed'] == 50 + i and c['spt_seed_rule'] == 'per-instance'
        assert c['spt_makespan'] == SPT_DET[i] == c['hint_makespan']
        first = c['hint_first_solution']
        print(f"  {c['instance']}: hint={c['hint_makespan']:g} first={first} "
              f"taken={c['hint_taken']} obj={c['objective']:g} "
              f"status={c['status']} cpu_share={c['timing']['cpu_share']}")
        assert c['hint_taken'] or c['objective'] <= c['spt_makespan']
        assert c['objective'] <= c['spt_makespan']
        assert c['timing']['workers'] == 2
    for r in recs:
        assert r['sig'] == 'cpsat3s-w2-hintspt|lagseed2027'
        assert r['cpsat_hint'] == 'spt'
        assert r['feasible'] and r['n_violations'] == 0
        assert r['nominal_plan_makespan'] <= r['cpsat_hint_makespan']


def test_none_reproduces_existing_path(cold):
    recs, nom, _ = cold
    assert len(nom) == 2 and len(recs) == 18
    stored = {(r['instance'], round(r['cv'], 6), r['rep']): r
              for r in esl.read_jsonl(os.path.join(
                  STORED, 'stochastic_lags_cpsat_repair.jsonl'))}
    stored_nom = {c['instance']: c for c in esl.read_jsonl(
        os.path.join(STORED, 'stochastic_lags_cpsat_nominal.jsonl'))}
    new_nom = {c['instance']: c for c in nom}
    for c in nom:  # the cold cache keeps its old key and its old fields
        assert c['key'] == f"{c['instance']}|3|2"
        assert set(c) == set(stored_nom[c['instance']])
    n_same_plan = 0
    for r in recs:
        s = stored[(r['instance'], round(r['cv'], 6), r['rep'])]
        assert set(r) == set(s), set(r) ^ set(s)
        assert r['sig'] == 'cpsat3s-w2|lagseed2027'
        # the realizations do not depend on the plan
        assert r['n_lags_changed'] == s['n_lags_changed']
        assert r['n_reveals'] == s['n_reveals']
        assert r['n_replans'] == s['n_replans']
        if s['cpsat_status'] == 'OPTIMAL' == r['cpsat_status']:
            assert r['cpsat_objective'] == s['cpsat_objective']
        a, b = new_nom[r['instance']], stored_nom[r['instance']]
        if a['assigned_mch'] == b['assigned_mch'] and a['start'] == b['start']:
            # same nominal plan -> the deterministic repair gives the same
            # realized makespan as the stored 60-s run
            n_same_plan += 1
            assert r['realized_makespan'] == s['realized_makespan']
            assert r['nominal_plan_makespan'] == s['nominal_plan_makespan']
    print(f'  records with the stored nominal plan: {n_same_plan} of 18')
