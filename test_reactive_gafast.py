"""Identity tests of the reactive GA fast path (eval_reactive.py --fast_eval).

Run (CPU only, pinned cores):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1 \\
      NUMBA_NUM_THREADS=1 taskset -c 6,7 python -m pytest -q test_reactive_gafast.py

Uses the SLAB checkpoint of the disruption study
(10x25+ppvc-mixed+a1lagx05-2k-s301). Checks:
  G1  fast_lpath.residual_makespan equals ResidualProblem.makespan_os,
      _pox_fast equals eval_ga.pox_crossover and the vectorized random
      individual equals eval_ga.random_individual (same values, same
      generator state), on disrupted states of every type;
  G2  ga_reactive with a fixed generation count (--ga_max_gens semantics,
      no clock) gives, with and without fast=True, the same best plan
      (machines and completion times), makespan, generation count, change
      counts (nervousness against the original plan), number of local-search
      calls and final generator state: all six disruption types on orders
      0-2 of 10 modules, and breakdown, urgent_arrival and material_shortage
      on orders 0-1 of 50 modules;
  G3  the eval_reactive CLI with --ga_max_gens writes identical records with
      and without --fast_eval (timing fields excepted).
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
DPARAMS = {'lag_factor': 2.0, 'pt_factor': 1.5, 'shortage_factor': 0.5,
           'shortage_basis': 'op_mean'}
DPARAMS_SHORT = dict(DPARAMS, shortage_basis='remaining_work')
TYPES = ('lag_perturb', 'breakdown', 'urgent_arrival', 'material_shortage',
         'pt_deviation', 'cancellation')
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', TORCH_THREADS='1', NUMBA_NUM_THREADS='1',
           PYTHONPATH=REPO)


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
    use_lag = bool(getattr(configs, 'use_lag_features', False))
    use_type = bool(getattr(configs, 'use_type_embedding', False))
    import fast_lpath
    fast_lpath.warmup_residual()
    return er, ppo, use_lag, use_type


@functools.lru_cache(maxsize=None)
def _state(size, i, dtype):
    er, ppo, use_lag, use_type = _setup()
    from ppvc_instance_generator import load_instance
    stem = os.path.join('data', 'PPVC', f'{size}+ppvc-mixed',
                        f'instance_{i:03d}')
    jl, pt, meta = load_instance(stem)
    _, _, asg_o, ct_o = er.rollout_greedy_full(ppo, jl, pt, meta, use_lag,
                                               use_type)
    dp = DPARAMS_SHORT if dtype == 'material_shortage' else DPARAMS
    ds = er.build_disrupted_state(ppo, jl, pt, meta, use_lag, use_type,
                                  dtype, 0.4, dp, er.instance_context(stem,
                                                                      meta))
    rs = er.right_shift_residual(ds, asg_o, ct_o, pt)
    return ds, rs, pt, asg_o, ct_o


# --------------------------------------------------------------------------- #
# G1: decoder and crossover
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('dtype', TYPES)
def test_residual_decoder_and_pox_identical(dtype):
    er = _setup()[0]
    ga = er._import_eval_ga()
    n_checked = 0
    for size, i in (('10x25', 0), ('50x25', 0)):
        ds, rs, *_ = _state(size, i, dtype)
        prob = er.ResidualProblem(ds)
        fr = er._FastResidual(prob)
        rng = np.random.default_rng(17)
        seed_ma = np.asarray(rs[2], dtype=int)
        seed_os = prob.os_from_schedule(seed_ma.tolist(), list(rs[3]))
        inds = [(seed_os, seed_ma)]
        inds += [ga.random_individual(rng, prob.base_os, prob.compat)
                 for _ in range(60)]
        for o, m in inds:
            assert fr.makespan(o, m) == prob.makespan_os(o, m.tolist())
            n_checked += 1
        # vectorized random individual = eval_ga.random_individual
        for k in range(20):
            r1, r2 = np.random.default_rng(k), np.random.default_rng(k)
            a1 = ga.random_individual(r1, prob.base_os, prob.compat)
            a2 = fr.random_individual(r2, prob.base_os, prob.compat)
            assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])
            assert a1[1].dtype == a2[1].dtype
            assert r1.bit_generator.state == r2.bit_generator.state
        for _ in range(100):
            a, b = inds[int(rng.integers(len(inds)))], inds[
                int(rng.integers(len(inds)))]
            sd = int(rng.integers(1 << 30))
            r1, r2 = np.random.default_rng(sd), np.random.default_rng(sd)
            c1 = ga.pox_crossover(r1, a[0], b[0], prob.n_jobs)
            c2 = er._pox_fast(r2, a[0], b[0], prob.n_jobs)
            assert c1.dtype == c2.dtype and np.array_equal(c1, c2)
            assert r1.bit_generator.state == r2.bit_generator.state
    assert n_checked == 122


# --------------------------------------------------------------------------- #
# G2: identical GA trajectory for a fixed generation count
# --------------------------------------------------------------------------- #
def _ga_pair(size, i, dtype, gens):
    er = _setup()[0]
    ds, rs, pt, asg_o, ct_o = _state(size, i, dtype)
    calls = {'old': 0, 'fast': 0}
    orig_ls, orig_fast_ls = (er._residual_ma_local_search,
                             er._FastResidual.ma_local_search)

    def ls_old(*a, **k):
        calls['old'] += 1
        return orig_ls(*a, **k)

    def ls_fast(self, *a, **k):
        calls['fast'] += 1
        return orig_fast_ls(self, *a, **k)

    out = {}
    try:
        er._residual_ma_local_search = ls_old
        er._FastResidual.ma_local_search = ls_fast
        for fast in (False, True):
            rng = np.random.default_rng(100003 * 0 + i)
            ms, _, asg, ct, g = er.ga_reactive(ds, rs[2], rs[3], 1e9,
                                               pop_size=100, rng=rng,
                                               fast=fast, max_gens=gens)
            nerv = er.nervousness_mapped(pt, asg_o, ct_o, ds['pt'], asg, ct,
                                         ds['op_map'])
            out[fast] = (ms, asg, ct, g, nerv, rng.bit_generator.state)
    finally:
        er._residual_ma_local_search = orig_ls
        er._FastResidual.ma_local_search = orig_fast_ls
    (m0, a0, c0, g0, n0, s0), (m1, a1, c1, g1, n1, s1) = out[False], out[True]
    assert m0 == m1
    assert np.array_equal(a0, a1) and np.array_equal(c0, c1)
    assert g0 == g1 == gens
    assert tuple(n0) == tuple(n1)
    assert s0 == s1
    assert calls['old'] == calls['fast']
    return dict(ms=m0, rs=float(rs[0]), nerv=n0, ls_calls=calls['old'])


@pytest.mark.parametrize('dtype', TYPES)
@pytest.mark.parametrize('i', (0, 1, 2))
def test_ga_identical_10_modules(dtype, i):
    r = _ga_pair('10x25', i, dtype, 80)
    assert r['ms'] <= r['rs'] + 1e-9


@pytest.mark.parametrize('dtype', ('breakdown', 'urgent_arrival',
                                   'material_shortage'))
@pytest.mark.parametrize('i', (0, 1))
def test_ga_identical_50_modules(dtype, i):
    r = _ga_pair('50x25', i, dtype, 60)
    assert r['ms'] <= r['rs'] + 1e-9


# --------------------------------------------------------------------------- #
# G3: CLI records with and without --fast_eval
# --------------------------------------------------------------------------- #
TIMING = {'seconds', 'loadavg_start', 'affinity', 'solve_wall_s',
          'solve_cpu_s', 'cpu_share', 'cpu_share_min', 'hit_limit',
          'n_hit_limit', 'n_low_share', 'time_limit_s', 'status'}


@pytest.mark.parametrize('dtype', ('breakdown', 'urgent_arrival'))
def test_cli_identical_records(tmp_path, dtype):
    outs = []
    for fast in (False, True):
        d = tmp_path / ('fast' if fast else 'old')
        cmd = [PY, 'eval_reactive.py', '--model_name', MODEL,
               '--data_path', 'data/PPVC/10x25+ppvc-mixed',
               '--disruption_type', dtype, '--methods',
               'drl_reactive,ga_reactive', '--ga_budget', '1000',
               '--ga_max_gens', '40', '--ga_pop', '100', '--ga_seed', '0',
               '--cpsat_budgets', '', '--device', 'cpu', '--state_fp',
               '--max_instances', '2', '--out_dir', str(d)]
        if fast:
            cmd.append('--fast_eval')
        subprocess.run(cmd, cwd=REPO, env=ENV, check=True,
                       stdout=subprocess.DEVNULL)
        outs.append(np.load(d / f'reactive_{dtype}_{MODEL}.npy',
                            allow_pickle=True).item())
    a, b = outs
    assert a['methods'] == b['methods']
    for m in a['methods']:
        for k, v in a['records'][m].items():
            if k in TIMING:
                continue
            va, vb = np.asarray(v), np.asarray(b['records'][m][k])
            nan_ok = va.dtype.kind in 'fc'
            assert np.array_equal(va, vb, equal_nan=nan_ok), (m, k)
    assert list(a['records']['ga_reactive']['generations']) == [40, 40]
    assert a['state_fp'] == b['state_fp']
    assert a['meta']['ga_max_gens'] == 40 and b['meta']['fast_eval'] is True
    assert 'fast_eval' not in a['meta']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
