"""Tests for eval_memetic.py (memetic-algorithm baseline for FJSP-TL).

Run:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
          python -m pytest -q test_eval_memetic.py
T1 runs two 60-second searches, so the whole file takes about three minutes.
"""
import os
import sys

import numpy as np
import pytest

import eval_memetic as em
from schedule_validator import validate_schedule

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = '10x25+ppvc-mixed'
DATA = os.path.join(REPO, 'data', 'PPVC', DATASET)
SPT_FILE = os.path.join(REPO, 'test_results', 'PPVC', DATASET,
                        f'Result_SPT+{DATASET}+smoke_{DATASET}.npy')


def _problem(idx=0, zero_lag=False):
    jl, pt, meta = em.load_instance(os.path.join(DATA, f'instance_{idx:03d}'))
    lag = np.asarray(meta['time_lag'], dtype=float)
    if zero_lag:
        lag = np.zeros_like(lag)
    return em.Problem(jl, pt, lag)


def _import_eval_ga():
    """eval_ga parses the process argv at import; give it a clean argv."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import eval_ga
    finally:
        sys.argv = saved
    return eval_ga


def _classical_fjsp_makespan(prob, os_vec, ma_vec):
    """Lag-free semi-active schedule of a job-repetition chromosome.

    Independent of both decoders: one left-to-right pass, start = max(end of
    the job's previous operation, end of the machine's previous operation).
    """
    job_ready = np.zeros(prob.n_jobs)
    mch_ready = np.zeros(prob.n_mch)
    nxt = prob.job_start.copy()
    for j in os_vec:
        op = nxt[j]
        nxt[j] += 1
        m = ma_vec[op]
        end = max(job_ready[j], mch_ready[m]) + prob.op_pt[op, m]
        job_ready[j] = mch_ready[m] = end
    return float(job_ready.max())


# --------------------------------------------------------------------------- #
# T1: two instances, 60 s budget: validator-clean, never worse than the seeds
# --------------------------------------------------------------------------- #
def test_t1_two_instances_budget_60(tmp_path):
    results, records = em.main(['--data_path', DATA, '--budget_s', '60',
                                '--max_instances', '2', '--seed', '0',
                                '--save_dir', str(tmp_path)])
    assert results.shape == (2, 2)
    spt_pub = np.load(SPT_FILE)[:2, 0] if os.path.exists(SPT_FILE) else None
    for i, r in enumerate(records):
        assert r['validator_violations'] == 0
        assert r['n_seeds'] == em.N1
        assert r['best_makespan'] <= r['best_seed_makespan'] + 1e-9
        assert r['best_makespan'] <= r['spt_makespan'] + 1e-9
        if spt_pub is not None:
            # round-0 SPT seed reproduces the published SPT baseline
            assert r['spt_makespan'] == pytest.approx(spt_pub[i])
        assert results[i, 0] == r['best_makespan']
    out = tmp_path / f'Result_MA60+memetic_{DATASET}.npy'
    assert out.exists() and (tmp_path / out.name.replace('.npy', '.json')).exists()


# --------------------------------------------------------------------------- #
# T2: zero lags -> decoder makespan equals the classical FJSP makespan
# --------------------------------------------------------------------------- #
def test_t2_zero_lag_equals_classical_fjsp():
    eval_ga = _import_eval_ga()
    for idx in (0, 1):
        prob = _problem(idx, zero_lag=True)
        prob_lag = _problem(idx, zero_lag=False)
        eval_ga._N_MCH_CACHE[0] = prob.n_mch
        rng = np.random.default_rng(idx)
        for _ in range(25):
            ind = em.random_individual(prob, rng)
            ref = _classical_fjsp_makespan(prob, ind.os, ind.ma)
            ga_ms = eval_ga.fitness(ind.os, ind.ma, prob.job_length,
                                    prob.job_start, prob.op_pt,
                                    prob.time_lag)
            assert ind.ms == pytest.approx(ref)
            assert ind.ms == pytest.approx(ga_ms)
            # with the true lags both chromosome decodings still agree
            ms_lag, _, _, _ = prob_lag.evaluate(ind.os, ind.ma)
            ga_lag = eval_ga.fitness(ind.os, ind.ma, prob.job_length,
                                     prob.job_start, prob.op_pt,
                                     prob_lag.time_lag)
            assert ms_lag == pytest.approx(ga_lag)
            assert ms_lag >= ind.ms - 1e-9


# --------------------------------------------------------------------------- #
# T3: the local search never returns a worse individual
# --------------------------------------------------------------------------- #
def test_t3_local_search_never_worse():
    prob = _problem(0)
    rng = np.random.default_rng(3)
    n_improved = 0
    starts = [em.random_individual(prob, rng) for _ in range(12)]
    # children of good parents, closer to what the MA feeds the search
    seeds, _ = em.build_seeds(prob, pdr_tie_seed=50, alt_tie_seed=7919)
    for _ in range(8):
        a, b = rng.choice(len(seeds), size=2, replace=False)
        os_c = em.gox(rng, seeds[a].os, seeds[b].os, prob.n_jobs)
        starts.append(em.Individual(prob, os_c,
                                    em.uniform_ma(rng, seeds[a].ma,
                                                  seeds[b].ma)))
    starts.extend(seeds[:4])
    for ind in starts:
        before_os, before_ma = ind.os.copy(), ind.ma.copy()
        out, n_impr = em.local_search(prob, ind)
        assert out.ms <= ind.ms + 1e-9
        # the input individual is left untouched
        assert np.array_equal(ind.os, before_os)
        assert np.array_equal(ind.ma, before_ma)
        # the returned chromosome re-decodes to the reported makespan
        again = em.Individual(prob, out.os, out.ma)
        assert again.ms == pytest.approx(out.ms)
        res = validate_schedule(prob.job_length, prob.op_pt, prob.time_lag,
                                out.ma, again.ct)
        assert res['feasible'], res['violations'][:3]
        assert np.array_equal(np.sort(out.os), np.sort(prob.base_os))
        if n_impr > 0:
            assert out.ms < ind.ms
            n_improved += 1
    assert n_improved > 0


def test_gox_keeps_job_repetition_multiset():
    prob = _problem(0)
    rng = np.random.default_rng(5)
    for _ in range(50):
        a = em.random_individual(prob, rng)
        b = em.random_individual(prob, rng)
        c = em.gox(rng, a.os, b.os, prob.n_jobs)
        assert np.array_equal(np.sort(c), np.sort(prob.base_os))


# --------------------------------------------------------------------------- #
# T4: determinism for a fixed seed and a fixed generation count
# --------------------------------------------------------------------------- #
def test_t4_determinism_fixed_generations(tmp_path):
    runs = []
    for k in range(2):
        d = tmp_path / f'run{k}'
        results, records = em.main(['--data_path', DATA, '--budget_s', '0',
                                    '--max_gens', '300', '--max_instances',
                                    '1', '--seed', '4', '--save_dir', str(d)])
        runs.append((results, records))
    (r0, rec0), (r1, rec1) = runs
    assert r0[0, 0] == r1[0, 0]
    for key in ('best_makespan', 'generations', 'ls_calls',
                'ls_improvements', 'clones_rejected', 'iter_to_best',
                'restarts'):
        assert rec0[0][key] == rec1[0][key], key
    assert rec0[0]['generations'] == 300
    assert [t[0] for t in rec0[0]['improvement_trace']] == \
        [t[0] for t in rec1[0]['improvement_trace']]

    # the chromosome itself is reproduced, not only the makespan
    prob = _problem(0)
    seeds, _ = em.build_seeds(prob, pdr_tie_seed=50, alt_tie_seed=7919)
    b0, s0 = em.run_memetic(prob, np.random.default_rng(11), seeds,
                            max_gens=150)
    b1, s1 = em.run_memetic(prob, np.random.default_rng(11), seeds,
                            max_gens=150)
    assert np.array_equal(b0.os, b1.os) and np.array_equal(b0.ma, b1.ma)
    assert b0.ms == b1.ms and s0['ls_calls'] == s1['ls_calls']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
