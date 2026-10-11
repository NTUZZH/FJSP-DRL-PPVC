"""Tests for the contention evidence of the wall-clock-budgeted runs in
eval_memetic.py, eval_ga.py, eval_buffers.py (cpsat-capacity) and
eval_flowline.py (flow-line CP-SAT).

Run (CPU only, two pinned cores):
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; \\
      taskset -c 22-23 python -m pytest -q test_cpu_share_more.py

Checks:
  D1  memetic: a 1 s search records its evidence (workers 1, status
      'budget', hit_limit, plausible cpu_share) and the JSON summary; a
      generation-capped run records status 'max_gens' and no budget-limited
      search, and its result equals the pre-edit script's when that copy is
      available (CPU_SHARE_ORIG_DIR);
  D2  GA: a 1 s run (in a temporary working directory, so test_results/ is
      never touched) writes the .npy and a same-stem JSON with the evidence;
  D3  eval_buffers cpsat-capacity: every record carries the evidence, the
      summary has the contention table, records written before the evidence
      existed are counted as such, and --num_workers 0 is rejected;
  D4  eval_flowline: solve_cpsat returns its evidence and main writes it per
      instance, in the JSON summary and in the markdown table.
Every subprocess inherits the CPU affinity and the thread caps of pytest.
"""
import importlib.util
import json
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

PY = sys.executable
DATA_5 = os.path.join(REPO, 'data', 'PPVC', '5x9+ppvc-mixed')
DATA_10 = os.path.join(REPO, 'data', 'PPVC', '10x25+ppvc-mixed')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONPATH=REPO)
ORIG_DIR = os.environ.get('CPU_SHARE_ORIG_DIR', '')


def _plausible(t, workers, limit):
    """Per-run evidence: every SOLVE_KEYS field, 0 < cpu_share <= 1.2."""
    assert set(esl.SOLVE_KEYS) <= set(t), sorted(t)
    assert t['workers'] == workers
    assert t['solve_wall_s'] > 0 and t['solve_cpu_s'] >= 0
    assert 0 < t['cpu_share'] <= 1.2, t
    assert t['cpu_share'] == pytest.approx(
        t['solve_cpu_s'] / (t['solve_wall_s'] * workers), abs=2e-3)
    if limit is not None:
        assert t['time_limit_s'] == pytest.approx(limit)
        assert t['hit_limit'] == (
            t['solve_wall_s'] >= esl.HIT_LIMIT_FRAC * limit)


def _record_fields(r, n_timed=1):
    """Per-record evidence over the record's own timed runs."""
    assert set(esl.CONTENTION_KEYS) <= set(r), sorted(r)
    assert r['n_timed_solves'] == n_timed and r['n_no_evidence'] == 0
    assert r['affinity'] == sorted(os.sched_getaffinity(0))
    assert r['loadavg_start'] is None or len(r['loadavg_start']) == 3
    if n_timed == 1:
        if r['hit_limit']:
            assert r['cpu_share_min'] == r['cpu_share']
            assert r['n_hit_limit'] == 1
            assert r['n_low_share'] == int(r['cpu_share'] < esl.LOW_CPU_SHARE)
        else:
            assert r['cpu_share_min'] is None and r['n_hit_limit'] == 0


# --------------------------------------------------------------------------- #
# D1: memetic
# --------------------------------------------------------------------------- #
def test_memetic_budget_and_gencap(tmp_path):
    import eval_memetic as em
    results, records = em.main(['--data_path', DATA_10, '--budget_s', '1',
                                '--max_instances', '1', '--seed', '0',
                                '--save_dir', str(tmp_path / 'b')])
    r = records[0]
    _plausible(r, workers=1, limit=1.0)
    assert r['status'] == 'budget' and r['hit_limit']
    _record_fields(r)
    assert results[0, 0] == r['best_makespan']
    with open(tmp_path / 'b' / 'Result_MA1+memetic_10x25+ppvc-mixed.json') as f:
        js = json.load(f)
    assert js['instances'][0]['cpu_share'] == r['cpu_share']
    assert js['contention']['n_records'] == 1
    assert js['contention']['n_hit_limit'] == 1

    # generation cap, no budget: not budget-limited, deterministic
    argv = ['--data_path', DATA_5, '--budget_s', '0', '--max_gens', '200',
            '--max_instances', '1', '--seed', '3']
    res_g, rec_g = em.main(argv + ['--save_dir', str(tmp_path / 'g')])
    g = rec_g[0]
    assert g['status'] == 'max_gens' and g['time_limit_s'] is None
    assert not g['hit_limit'] and g['generations'] == 200
    _plausible(g, workers=1, limit=None)
    _record_fields(g)
    orig = os.path.join(ORIG_DIR, 'eval_memetic.py')
    if ORIG_DIR and os.path.exists(orig):
        spec = importlib.util.spec_from_file_location('eval_memetic_orig',
                                                      orig)
        om = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(om)
        res_o, rec_o = om.main(argv + ['--save_dir', str(tmp_path / 'o')])
        assert res_o[0, 0] == res_g[0, 0]
        for key in ('best_makespan', 'generations', 'ls_calls',
                    'ls_improvements', 'iter_to_best', 'clones_rejected'):
            assert rec_o[0][key] == g[key], key


# --------------------------------------------------------------------------- #
# D2: GA (eval_ga writes under ./test_results, so it runs in tmp_path)
# --------------------------------------------------------------------------- #
def test_ga_sidecar_json(tmp_path):
    p = subprocess.run([PY, os.path.join(REPO, 'eval_ga.py'), '--data_path',
                        DATA_10, '--budget_s', '1', '--max_instances', '1',
                        '--seed', '0'], cwd=tmp_path, env=ENV,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert 'contention' in p.stdout
    d = tmp_path / 'test_results' / 'PPVC' / '10x25+ppvc-mixed'
    stem = 'Result_GA1+10x25+ppvc-mixed_10x25+ppvc-mixed'
    arr = np.load(d / f'{stem}.npy')
    with open(d / f'{stem}.json') as f:
        js = json.load(f)
    r = js['instances'][0]
    _plausible(r, workers=1, limit=1.0)
    assert r['status'] == 'budget' and r['hit_limit']
    _record_fields(r)
    assert r['makespan'] == arr[0, 0] and r['seconds'] == arr[0, 1]
    assert js['contention']['n_records'] == 1
    assert js['seed_rules'] == ['SPT', 'MWKR', 'MOR', 'FIFO']


# --------------------------------------------------------------------------- #
# D3: eval_buffers cpsat-capacity
# --------------------------------------------------------------------------- #
def _buffers(*args):
    return subprocess.run([PY, 'eval_buffers.py', 'cpsat-capacity', *args],
                          cwd=REPO, env=ENV, capture_output=True, text=True,
                          timeout=300)


def test_buffers_cpsat_capacity(tmp_path):
    out = str(tmp_path)
    args = ['--data_path', DATA_10, '--max_instances', '1', '--B', '3',
            '--time_limit', '1', '--num_workers', '2', '--out_dir', out]
    p = _buffers(*args)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    path = os.path.join(out, 'buffers_cpsat_shared_t1.jsonl')
    with open(path) as f:
        recs = [json.loads(x) for x in f if x.strip()]
    assert [r['B'] for r in recs] == [None, 3]
    for r in recs:
        _plausible(r, workers=2, limit=1.0)
        assert r['num_workers'] == 2 and r['time_limit_s'] == 1.0
        _record_fields(r)
    md = open(os.path.join(out, 'buffers_cpsat_shared_t1_summary.md')).read()
    assert 'Contention evidence' in md and '| CP-SAT | 2 | 2 |' in md

    # a record written before the evidence existed is counted, not skipped
    old = {k: v for k, v in recs[1].items()
           if k not in esl.CONTENTION_KEYS and k not in esl.SOLVE_KEYS}
    old['status'], old['time_limit_s'] = recs[1]['status'], 1.0
    with open(path, 'w') as f:
        f.write(json.dumps(recs[0]) + '\n' + json.dumps(old) + '\n')
    p = _buffers(*args)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    md = open(os.path.join(out, 'buffers_cpsat_shared_t1_summary.md')).read()
    assert '(+1 records without fields)' in md

    bad = _buffers('--data_path', DATA_5, '--num_workers', '0',
                   '--out_dir', out)
    assert bad.returncode != 0 and 'num_workers must be >= 1' in bad.stderr


# --------------------------------------------------------------------------- #
# D4: eval_flowline flow-line CP-SAT
# --------------------------------------------------------------------------- #
def test_flowline_cpsat_evidence(tmp_path):
    import eval_flowline as fl
    from ppvc_instance_generator import load_instance
    jl, pt, meta = load_instance(os.path.join(DATA_5, 'instance_000'))
    pt_fl, _ = fl.flowline_restrict(jl, pt, meta)
    sol = fl.solve_cpsat(jl, pt_fl, meta['time_lag'], 1.0, 2)
    t = sol['timing']
    _plausible(t, workers=2, limit=1.0)
    assert t['status'] == sol['status']

    js = fl.main(['--data_path', DATA_10, '--max_instances', '1',
                  '--save_dir', str(tmp_path), '--cpsat_time_limit', '1',
                  '--cpsat_workers', '2', '--no_greedy', '--quiet'])
    r = js['per_instance'][0]
    t = r['cpsat_flowline_timing']
    _plausible(t, workers=2, limit=1.0)
    assert t['status'] == r['cpsat_flowline_status']
    assert set(esl.CONTENTION_KEYS) <= set(r)
    assert r['n_timed_solves'] == 1
    assert r['cpu_share_min'] == (t['cpu_share'] if t['hit_limit'] else None)
    with open(tmp_path / 'flowline_per_instance_10x25+ppvc-mixed.json') as f:
        on_disk = json.load(f)
    assert on_disk['contention']['n_records'] == 1
    md = open(tmp_path / 'summary_flowline_10x25+ppvc-mixed.md').read()
    assert '| flow-line CP-SAT | 1 | 1 |' in md


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
