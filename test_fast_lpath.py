"""Equivalence tests of fast_lpath.py and of eval_memetic.py --fast_eval.

Run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMBA_NUM_THREADS=1 taskset -c 6,7 python -m pytest -q test_fast_lpath.py

1. The fast longest path is bit-identical (np.array_equal on float64) to
   right_shift_repair: on random chromosomes of real orders of every size
   (random OS permutations of the job multiset, random compatible stations),
   with the real lags and with random non-integer lags, with machine release
   floors, and on the 14 rule seeds of several real orders of every size.
   The compiled kernels and their uncompiled Python source (the fallback)
   agree. Malformed and cyclic inputs raise ValueError like
   right_shift_repair.
2. The memetic operators of the fast path (critical path, OS rebuild after a
   swap, GOX, rank selection) return the same values as the original ones.
3. The search trajectory is identical: run_memetic with and without the fast
   path, a fixed seed and an iteration cap, gives the same best chromosome,
   iteration counts, local-search counts, clone rejections, restarts,
   improvement trace and final random-number state; so does main().
"""
import bisect
import os

import numpy as np
import pytest

import eval_memetic as em
import fast_lpath as fl
from right_shift_repair import right_shift_repair

REPO = os.path.dirname(os.path.abspath(__file__))
SIZES = ('10x25', '20x25', '30x25', '50x25', '80x25')


def _data(ds):
    return os.path.join(REPO, 'data', 'PPVC', f'{ds}+ppvc-mixed')


def _problems(ds, idx, lag=None):
    jl, pt, meta = em.load_instance(os.path.join(_data(ds),
                                                 f'instance_{idx:03d}'))
    tl = np.asarray(meta['time_lag'], dtype=float) if lag is None else lag
    return em.Problem(jl, pt, tl), em.Problem(jl, pt, tl, fast=True)


def _random_chromosome(prob, rng):
    os_vec = prob.base_os.copy()
    rng.shuffle(os_vec)
    ma = np.array([c[rng.integers(len(c))] for c in prob.compat], dtype=int)
    return os_vec, ma


def _check_one(po, pf, os_vec, ma, mrt=None):
    """All evaluators against right_shift_repair on one chromosome."""
    seqs, order = po.decode(os_vec, ma)
    ct_ref, ms_ref = right_shift_repair(po.job_length, po.op_pt, po.time_lag,
                                        ma, op_sequence_per_mch=seqs,
                                        mch_ready_time=mrt)
    ct1, ms1 = fl.longest_path(po.job_length, po.op_pt, po.time_lag, ma,
                               seqs, mch_ready_time=mrt)
    assert ct1.dtype == np.float64 and np.array_equal(ct1, ct_ref)
    assert ms1 == ms_ref
    if mrt is None:
        ct2, ms2, order2, mprev, mark, code = fl.decode_eval(
            os_vec.astype(np.int64), ma.astype(np.int64), pf.job_start64,
            pf.job_end64, pf.op_pt_f, pf.lag64, em.MARK_K)
        assert code == 0
        assert np.array_equal(ct2, ct_ref) and ms2 == ms_ref
        assert np.array_equal(order2, order)
        assert int(mark) == po.mark(ct_ref, ma)
        # machine predecessors equal the decoded machine sequences
        ref_prev = np.full(po.n_ops, -1)
        for s in seqs:
            for a, b in zip(s[:-1], s[1:]):
                ref_prev[b] = a
        assert np.array_equal(mprev, ref_prev)
    return ct_ref


# --------------------------------------------------------------------------- #
# 1. Longest path: bit-identical to right_shift_repair
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ds', SIZES)
def test_random_chromosomes_bit_identical(ds):
    n_samples = 0
    for idx in (0, 1, 2):
        po, pf = _problems(ds, idx)
        rng = np.random.default_rng(1000 + idx)
        for _ in range(150):
            os_vec, ma = _random_chromosome(po, rng)
            _check_one(po, pf, os_vec, ma)
            n_samples += 1
    assert n_samples == 450


@pytest.mark.parametrize('ds', SIZES)
def test_random_noninteger_lags_bit_identical(ds):
    """Lags and processing times that are not small integers stress the
    float64 arithmetic: still bit-identical."""
    po0, _ = _problems(ds, 3)
    rng = np.random.default_rng(77)
    lag = rng.uniform(0.0, 7.3, po0.n_ops) * (rng.random(po0.n_ops) < 0.6)
    lag = np.round(lag, 7) + 0.1          # many non-representable decimals
    po, pf = _problems(ds, 3, lag=lag)
    for _ in range(100):
        os_vec, ma = _random_chromosome(po, rng)
        _check_one(po, pf, os_vec, ma)


@pytest.mark.parametrize('ds', ('10x25', '50x25'))
def test_machine_release_floor_bit_identical(ds):
    po, pf = _problems(ds, 0)
    rng = np.random.default_rng(5)
    for _ in range(100):
        os_vec, ma = _random_chromosome(po, rng)
        mrt = rng.uniform(-5.0, 40.0, po.n_mch)
        mrt[rng.random(po.n_mch) < 0.3] = 0.0
        _check_one(po, pf, os_vec, ma, mrt=mrt)


def test_fallback_python_source_identical():
    """The uncompiled source (pure-Python fallback) gives the same values."""
    for ds in ('10x25', '30x25'):
        po, pf = _problems(ds, 0)
        rng = np.random.default_rng(9)
        for _ in range(15):
            os_vec, ma = _random_chromosome(po, rng)
            o64, m64 = os_vec.astype(np.int64), ma.astype(np.int64)
            a = fl.decode_eval(o64, m64, pf.job_start64, pf.job_end64,
                               pf.op_pt_f, pf.lag64, em.MARK_K)
            b = fl.decode_eval.py_func(o64, m64, pf.job_start64, pf.job_end64,
                                       pf.op_pt_f, pf.lag64, em.MARK_K)
            assert np.array_equal(a[0], b[0]) and a[1] == b[1]
            assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
            assert int(a[4]) == int(b[4]) and a[5] == b[5] == 0
            seqs, _ = po.decode(os_vec, ma)
            flat = np.concatenate([np.asarray(s, dtype=np.int64)
                                   for s in seqs])
            fm = np.repeat(np.arange(po.n_mch, dtype=np.int64),
                           [len(s) for s in seqs])
            pt = pf.op_pt_f[np.arange(po.n_ops), ma]
            k1 = fl._lpath_kernel(pf.job_pred64, pt, pf.lag64, m64, flat, fm,
                                  np.zeros(po.n_mch))
            k2 = fl._lpath_kernel.py_func(pf.job_pred64, pt, pf.lag64, m64,
                                          flat, fm, np.zeros(po.n_mch))
            assert np.array_equal(k1[0], k2[0]) and k1[1] == k2[1] == 0
            assert np.array_equal(k1[0], a[0])
            c1 = fl.critical_path(a[0], m64, pf.op_pt_f, a[3], pf.job_pred64,
                                  pf.lag64, em.TOL)
            c2 = fl.critical_path.py_func(a[0], m64, pf.op_pt_f, a[3],
                                          pf.job_pred64, pf.lag64, em.TOL)
            assert all(np.array_equal(x, y) for x, y in zip(c1, c2))
            g1 = fl.gox(o64, o64[::-1].copy(), pf.job_start64, 7, 40)
            g2 = fl.gox.py_func(o64, o64[::-1].copy(), pf.job_start64, 7, 40)
            assert np.array_equal(g1, g2)


_SEED_CACHE = {}


def _seeds(ds, idx):
    if (ds, idx) not in _SEED_CACHE:
        po, pf = _problems(ds, idx)
        seeds, _ = em.build_seeds(po, pdr_tie_seed=50 + idx,
                                  alt_tie_seed=7919 + idx)
        _SEED_CACHE[(ds, idx)] = (po, pf, seeds)
    return _SEED_CACHE[(ds, idx)]


@pytest.mark.parametrize('ds', SIZES)
def test_rule_seeds_bit_identical(ds):
    """The 14 rule seeds of three real orders of every size."""
    for idx in (0, 1, 2):
        po, pf, seeds = _seeds(ds, idx)
        assert len(seeds) == em.N1
        for s in seeds:
            ct = _check_one(po, pf, s.os, s.ma)
            assert np.array_equal(ct, s.ct)
            f = em.Individual(pf, s.os, s.ma)
            assert f.ms == s.ms and f.mark == s.mark
            assert np.array_equal(f.ct, s.ct)


def test_cycle_and_malformed_inputs_raise():
    po, pf = _problems('10x25', 0)
    rng = np.random.default_rng(3)
    os_vec, ma = _random_chromosome(po, rng)
    # put op 0 and op 1 (consecutive ops of job 0) on one machine, reversed
    m = int(np.intersect1d(po.compat[0], po.compat[1])[0]) \
        if len(np.intersect1d(po.compat[0], po.compat[1])) else None
    seqs, _ = po.decode(os_vec, ma)
    if m is not None:
        ma2 = ma.copy()
        ma2[0] = ma2[1] = m
        seqs2, _ = po.decode(os_vec, ma2)
        s = [op for op in seqs2[m] if op not in (0, 1)]
        seqs2[m] = [1, 0] + s
        with pytest.raises(ValueError):
            right_shift_repair(po.job_length, po.op_pt, po.time_lag, ma2,
                               op_sequence_per_mch=seqs2)
        with pytest.raises(ValueError, match='CYCLE'):
            fl.longest_path(po.job_length, po.op_pt, po.time_lag, ma2, seqs2)
    # a two-machine cycle: a later op of job 0 before an earlier op of job 1
    # on one machine, and the reverse on another machine
    jl = np.array([2, 2])
    op_pt = np.array([[2, 0], [0, 3], [0, 1], [4, 0]])
    lag = np.array([1.0, 0.0, 0.5, 0.0])
    am = np.array([0, 1, 1, 0])
    ok = [[0, 3], [1, 2]]
    ct, ms = fl.longest_path(jl, op_pt, lag, am, ok)
    ct_r, ms_r = right_shift_repair(jl, op_pt, lag, am, op_sequence_per_mch=ok)
    assert np.array_equal(ct, ct_r) and ms == ms_r
    cyc = [[3, 0], [1, 2]]          # 3 before 0 on M0 and 1 before 2 on M1:
    #   0 -> 1 (job), 1 -> 2 (M1), 2 -> 3 (job), 3 -> 0 (M0): a cycle
    with pytest.raises(ValueError):
        right_shift_repair(jl, op_pt, lag, am, op_sequence_per_mch=cyc)
    with pytest.raises(ValueError, match='CYCLE'):
        fl.longest_path(jl, op_pt, lag, am, cyc)
    # the uncompiled fallback reports the same cycle (code 5)
    _, code, _ = fl._lpath_kernel.py_func(
        fl.job_pred_of(jl), np.array([2., 3., 1., 4.]), lag,
        am.astype(np.int64), np.array([3, 0, 1, 2], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64), np.zeros(2))
    assert code == 5
    # malformed sequences: duplicate, missing, wrong machine, out of range
    for bad in ([[0, 3, 0], [1, 2]], [[0], [1, 2]], [[0, 3, 1], [2]],
                [[0, 3, 9], [1, 2]]):
        with pytest.raises(ValueError):
            right_shift_repair(jl, op_pt, lag, am, op_sequence_per_mch=bad)
        with pytest.raises(ValueError):
            fl.longest_path(jl, op_pt, lag, am, bad)
    with pytest.raises(ValueError):
        fl.longest_path(jl, op_pt, lag, am, [[0, 3]])      # one machine short
    # an OS that is not a job-repetition sequence
    with pytest.raises(ValueError):
        pf.evaluate_fast(np.zeros(po.n_ops, dtype=np.int64), ma)


# --------------------------------------------------------------------------- #
# 2. Operators of the fast path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ds', ('10x25', '50x25', '80x25'))
def test_critical_path_and_swap_identical(ds):
    po, pf = _problems(ds, 0)
    rng = np.random.default_rng(21)
    starts = [em.random_individual(po, rng) for _ in range(20)]
    if ds == '10x25':
        starts += list(_seeds(ds, 0)[2])
    n_swaps = 0
    for a in starts:
        b = em.Individual(pf, a.os, a.ma)
        path, kinds = po.critical_path(a.ct, a.seqs, a.ma)
        fpath, fkinds = pf.critical_path_fast(b)
        assert fpath.tolist() == list(path)
        assert fkinds.tolist() == [1 if k == 'mch' else 0 for k in kinds]
        pos_of = np.empty(po.n_ops, dtype=float)
        pos_of[a.order] = np.arange(po.n_ops, dtype=float)
        for k, v in enumerate(path):
            if k < len(kinds) and kinds[k] == 'mch':
                u = path[k + 1]
                m = a.ma[v]
                seqs = [list(s) for s in a.seqs]
                iu = seqs[m].index(u)
                seqs[m][iu], seqs[m][iu + 1] = v, u
                key = pos_of.copy()
                key[v] = pos_of[u] - 0.5
                ref = po.os_from_seqs(seqs, key)
                new = pf.os_after_swap(b, int(u), int(v), pos_of)
                assert np.array_equal(ref, new)
                c_old = em.Individual(po, ref, a.ma)
                c_new = em.Individual(pf, new, b.ma)
                assert np.array_equal(c_old.ct, c_new.ct)
                # the rebuilt OS decodes to exactly the swapped sequences
                assert c_old.seqs == seqs
                n_swaps += 1
    assert n_swaps > 0


def test_gox_and_selection_identical():
    for ds in ('10x25', '50x25'):
        po, pf = _problems(ds, 1)
        rng = np.random.default_rng(4)
        for _ in range(200):
            a = em.random_individual(po, rng)
            b = em.random_individual(po, rng)
            sd = int(rng.integers(1 << 30))
            r1, r2 = np.random.default_rng(sd), np.random.default_rng(sd)
            c1 = em.gox(r1, a.os, b.os, po.n_jobs)
            c2 = em.gox_fast(r2, a.os, b.os, pf)
            assert np.array_equal(c1, c2)
            assert r1.bit_generator.state == r2.bit_generator.state
    nc = em.NC
    ranks = np.arange(nc, 0, -1, dtype=float)
    sel_p = 2.0 * ranks / (nc * (nc + 1))
    cdf = sel_p.cumsum()
    cdf /= cdf[-1]
    cdf = cdf.tolist()
    for seed in range(10):
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        x = [int(r1.choice(nc, p=sel_p)) for _ in range(20000)]
        y = [bisect.bisect_right(cdf, r2.random()) for _ in range(20000)]
        assert x == y
        assert r1.bit_generator.state == r2.bit_generator.state


# --------------------------------------------------------------------------- #
# 3. Identical search trajectory
# --------------------------------------------------------------------------- #
def _run_pair(ds, idx, gens, np_restart=em.NP_RESTART, seed=0):
    po, pf, seeds = _seeds(ds, idx)
    seeds_f = [em.Individual(pf, s.os, s.ma) for s in seeds]
    r_old = np.random.default_rng(seed * 100003 + idx)
    r_new = np.random.default_rng(seed * 100003 + idx)
    b0, s0 = em.run_memetic(po, r_old, seeds, max_gens=gens,
                            np_restart=np_restart)
    b1, s1 = em.run_memetic(pf, r_new, seeds_f, max_gens=gens,
                            np_restart=np_restart)
    assert b0.ms == b1.ms
    assert np.array_equal(b0.os, b1.os) and np.array_equal(b0.ma, b1.ma)
    assert np.array_equal(b0.ct, b1.ct)
    for key in ('generations', 'ls_calls', 'ls_improvements', 'restarts',
                'clones_rejected', 'iter_to_best', 'initial_best'):
        assert s0[key] == s1[key], key
    assert [(g, m) for g, _, m in s0['trace']] == \
        [(g, m) for g, _, m in s1['trace']]
    assert r_old.bit_generator.state == r_new.bit_generator.state
    return s0


@pytest.mark.parametrize('idx', (0, 1, 2, 3))
def test_trajectory_identical_10_modules(idx):
    s = _run_pair('10x25', idx, 3000)
    assert s['generations'] == 3000 and s['ls_calls'] > 0


@pytest.mark.parametrize('idx', (0, 1))
def test_trajectory_identical_10_modules_with_restarts(idx):
    s = _run_pair('10x25', idx, 3000, np_restart=250, seed=3)
    assert s['restarts'] > 0


@pytest.mark.parametrize('idx', (0, 1, 2))
def test_trajectory_identical_50_modules(idx):
    s = _run_pair('50x25', idx, 2000)
    assert s['generations'] == 2000 and s['ls_calls'] > 0


def test_trajectory_identical_50_modules_with_restarts():
    s = _run_pair('50x25', 3, 1500, np_restart=150, seed=2)
    assert s['restarts'] > 0


def test_main_identical_records(tmp_path):
    """main() with and without --fast_eval (iteration cap, instance range)."""
    common = ['--data_path', _data('10x25'), '--budget_s', '0',
              '--max_gens', '1500', '--seed', '0']
    r0, rec0 = em.main(common + ['--max_instances', '3',
                                 '--save_dir', str(tmp_path / 'old')])
    r1, rec1 = em.main(common + ['--max_instances', '3', '--fast_eval',
                                 '--instances', '1:3',
                                 '--save_dir', str(tmp_path / 'new')])
    assert np.array_equal(r0[1:, 0], r1[:, 0])
    timing = {'time_to_best_s', 'seed_s', 'search_s', 'total_s',
              's_per_generation', 'improvement_trace', 'solve_wall_s',
              'solve_cpu_s', 'cpu_share', 'cpu_share_min', 'loadavg_start',
              'affinity'}
    for a, b in zip(rec0[1:], rec1):
        for k in a:
            if k not in timing:
                assert a[k] == b[k], k
        assert [(g, m) for g, _, m in a['improvement_trace']] == \
            [(g, m) for g, _, m in b['improvement_trace']]
    out = tmp_path / 'new' / 'Result_MAg1500+memetic_10x25+ppvc-mixed.i001-003.json'
    assert out.exists()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
