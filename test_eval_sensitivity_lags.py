"""Tests for eval_sensitivity.py --perturb {pt,lags,both}.

Run:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
          taskset -c 16-17 python -m pytest -q -s test_eval_sensitivity_lags.py
The sweep fixture runs the full script (run()) in a scratch working directory
for every mode on 3 instances of 10x25 at delta 0 and 0.2 on the CPU, about
2 to 3 minutes. The published test_results files are never written.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = '10x25+ppvc-mixed'
N_INST = 3
DELTAS = [0.0, 0.2]
GREEDY_DET = [196.0, 219.0, 214.0]
SPT_DET = [210.0, 218.0, 216.0]
PUBLISHED = os.path.join(REPO, 'test_results', 'PPVC', DATASET,
                         'sensitivity_summary.md')

# eval_sensitivity parses its CLI at import time and params parses the
# (scrubbed) argv after it, so the import runs under a controlled argv
_saved_argv = sys.argv
sys.argv = ['eval_sensitivity.py', '--perturb', 'pt', '--device', 'cpu',
            '--max_instances', str(N_INST), '--deltas', '0,0.2']
import eval_sensitivity as es  # noqa: E402
sys.argv = _saved_argv


def _sha(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load(i):
    from ppvc_instance_generator import load_instance
    stem = os.path.join(REPO, 'data', 'PPVC', DATASET, f'instance_{i:03d}')
    return load_instance(stem)


# --------------------------------------------------------------------------- #
# Perturbation primitives
# --------------------------------------------------------------------------- #

def test_perturb_lags_properties():
    nominal = np.array([0, 12, 0, 20, 46, 1, 0, 30] * 500, dtype=float)
    same = es.perturb_lags(nominal, 0.0, None)
    assert np.array_equal(same, nominal)
    r = es.perturb_lags(nominal, 0.3, es.lag_rng(0, 0.3, 7))
    zero = nominal == 0
    assert np.array_equal(r[zero], nominal[zero])        # zero lags stay zero
    assert (r[~zero] >= 1).all()                         # nonzero stay >= 1 h
    assert np.array_equal(r, np.rint(r))                 # integer hours
    ratio = r[~zero] / nominal[~zero]
    slack = 0.5 / nominal[~zero]
    # the 1 h lag may be lifted by the >= 1 floor, hence the max with 1
    lo = np.minimum(0.7 - slack, 1.0)
    assert (ratio >= lo - 1e-12).all()
    assert (ratio <= 1.3 + slack + 1e-12).all()
    assert not np.array_equal(r, nominal)
    # same (seed, delta, instance) -> same draw; another instance -> another
    r2 = es.perturb_lags(nominal, 0.3, es.lag_rng(0, 0.3, 7))
    r3 = es.perturb_lags(nominal, 0.3, es.lag_rng(0, 0.3, 8))
    assert np.array_equal(r, r2) and not np.array_equal(r, r3)


def test_lag_stream_is_distinct_from_pt_stream():
    for (seed, delta, i) in [(0, 0.2, 0), (0, 0.1, 5), (3, 0.3, 99)]:
        # the pt stream is the original seeding, unchanged
        a = es.pt_rng(seed, delta, i).uniform(size=50)
        b = np.random.default_rng(
            es.instance_seed(seed, delta, i)).uniform(size=50)
        assert np.array_equal(a, b)
        c = es.lag_rng(seed, delta, i).uniform(size=50)
        assert not np.any(np.isin(c, a))


def test_perturbed_instance_modes():
    jl, pt_base, meta = _load(0)
    lag0 = np.array(meta['time_lag'], copy=True)
    d, seed, i = 0.2, 0, 0
    pt_p, m_p = es.perturbed_instance(pt_base, meta, d, 'pt', seed, i)
    pt_l, m_l = es.perturbed_instance(pt_base, meta, d, 'lags', seed, i)
    pt_b, m_b = es.perturbed_instance(pt_base, meta, d, 'both', seed, i)
    # pt mode: same pt draw as the original script, meta untouched
    assert m_p is meta
    assert np.array_equal(
        pt_p, es.perturb_pt(pt_base, d, np.random.default_rng(
            es.instance_seed(seed, d, i))))
    assert not np.array_equal(pt_p, pt_base)
    # lags mode: pt unchanged, nonzero lags changed, zero lags stay zero
    assert np.array_equal(pt_l, pt_base)
    assert not np.array_equal(m_l['time_lag'], lag0)
    assert np.array_equal(m_l['time_lag'] == 0, lag0 == 0)
    # both mode: each part equals its single-mode draw
    assert np.array_equal(pt_b, pt_p)
    assert np.array_equal(m_b['time_lag'], m_l['time_lag'])
    # the caller's meta is never mutated
    assert np.array_equal(meta['time_lag'], lag0)
    # delta 0 is the nominal instance in every mode
    for mode in es.PERTURB_MODES:
        p0, m0 = es.perturbed_instance(pt_base, meta, 0.0, mode, seed, i)
        assert np.array_equal(p0, pt_base)
        assert np.array_equal(m0['time_lag'], lag0)
    with pytest.raises(ValueError):
        es.perturbed_instance(pt_base, meta, d, 'bogus', seed, i)


# --------------------------------------------------------------------------- #
# Full sweep through run(), every mode, in a scratch working directory
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    work = tmp_path_factory.mktemp('sens')
    for name in ('data', 'trained_network', 'train_log'):
        os.symlink(os.path.join(REPO, name), work / name)
    published_before = _sha(PUBLISHED)

    records = {}          # (mode, delta) -> (a3, spt)
    lags_seen = {}        # (mode, delta) -> list of lag vectors used
    orig_eval = es.evaluate_delta
    orig_rollout = es.eval_ppvc.rollout_greedy

    def recorder(ppo, loaded, bases, delta, args, ulf, ute):
        a3, spt = orig_eval(ppo, loaded, bases, delta, args, ulf, ute)
        records[(args.perturb, delta)] = (a3.copy(), spt.copy())
        return a3, spt

    cwd = os.getcwd()
    os.chdir(work)
    try:
        es.evaluate_delta = recorder
        for mode in es.PERTURB_MODES:
            es._ARGS.perturb = mode

            def spy(ppo, jl, pt, meta, *a, _mode=mode, **k):
                lags_seen.setdefault(_mode, []).append(
                    np.array(meta['time_lag'], dtype=float))
                return orig_rollout(ppo, jl, pt, meta, *a, **k)
            es.eval_ppvc.rollout_greedy = spy
            print(f'\n---------------- --perturb {mode} ----------------')
            es.run()
    finally:
        es.evaluate_delta = orig_eval
        es.eval_ppvc.rollout_greedy = orig_rollout
        es._ARGS.perturb = 'pt'
        os.chdir(cwd)
    out_dir = work / 'test_results' / 'PPVC' / DATASET
    return dict(records=records, lags_seen=lags_seen, out_dir=out_dir,
                published_before=published_before)


def test_delta0_reproduces_stored_makespans(sweep):
    for mode in es.PERTURB_MODES:
        a3, spt = sweep['records'][(mode, 0.0)]
        print(f'  {mode:<5} delta=0    A3={a3.tolist()}  SPT={spt.tolist()}')
        assert a3.tolist() == GREEDY_DET
        assert spt.tolist() == SPT_DET
    stored = np.load(os.path.join(
        REPO, 'test_results', 'PPVC', DATASET,
        f'Result_greedy+{DATASET}+full_{DATASET}.npy'))[:N_INST, 0]
    assert stored.tolist() == GREEDY_DET


def test_delta02_draws_differ_by_mode(sweep):
    rec = sweep['records']
    for mode in es.PERTURB_MODES:
        a3, spt = rec[(mode, 0.2)]
        print(f'  {mode:<5} delta=0.2  A3={a3.tolist()}  SPT={spt.tolist()}')
    # the rollouts saw the perturbed lags in lags/both mode, nominal in pt
    nominal = [np.asarray(_load(i)[2]['time_lag'], dtype=float)
               for i in range(N_INST)]
    seen = sweep['lags_seen']
    for mode in es.PERTURB_MODES:
        per_delta = [seen[mode][:N_INST], seen[mode][N_INST:]]
        for i in range(N_INST):
            assert np.array_equal(per_delta[0][i], nominal[i])
            if mode == 'pt':
                assert np.array_equal(per_delta[1][i], nominal[i])
            else:
                assert not np.array_equal(per_delta[1][i], nominal[i])
                assert np.array_equal(per_delta[1][i] == 0, nominal[i] == 0)
    for i in range(N_INST):
        assert np.array_equal(seen['lags'][N_INST + i],
                              seen['both'][N_INST + i])


def test_outputs_carry_mode_and_published_file_untouched(sweep):
    out = sweep['out_dir']
    names = sorted(os.listdir(out))
    print(f'  outputs: {names}')
    assert names == ['sensitivity_summary.md', 'sensitivity_summary_both.md',
                     'sensitivity_summary_lags.md']
    txt = {m: (out / f'sensitivity_summary{es._MODE_SUFFIX[m]}.md').read_text()
           for m in es.PERTURB_MODES}
    assert '\\SensDeltaZeroA3' in txt['pt']
    assert 'time-lags UNCHANGED' in txt['pt']
    assert '\\SensLagDeltaZeroA3' in txt['lags']
    assert '\\SensDelta' not in txt['lags']
    assert '\\SensBothDeltaTwentyP' in txt['both']
    assert 'processing times UNCHANGED' in txt['lags']
    assert _sha(PUBLISHED) == sweep['published_before']
