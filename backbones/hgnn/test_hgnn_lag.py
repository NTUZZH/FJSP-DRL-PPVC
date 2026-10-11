"""
Tests for the lag-aware HGNN environment (env/fjsp_env_lag.py) and the PPVC
entry points. CPU only. Run either way:

    python backbones/hgnn/test_hgnn_lag.py            # all tests, verbose output
    python -m pytest backbones/hgnn/test_hgnn_lag.py

T0 loader: HGNN's .fjs loader and the side-car lag reader reproduce
   data_utils.text_to_matrix and meta.json on instance_000.
T1 zero-lag control: with all lags zero, the lag environment gives
   bit-identical makespans and schedules to the upstream environment, under the
   upstream checkpoint (greedy) and under a fixed random action sequence.
T2 serial oracle: one module on SMALL_FACTORY; makespan equals the sum of
   processing times plus the sum of lags.
T3 validator: lag-aware greedy rollouts are validator-clean and the env
   makespan equals the validator makespan.
T4 bound: along the lag-aware greedy rollouts of T3, the makespan estimate
   that drives the reward (max over jobs of the estimated job completion time)
   never exceeds the final makespan.
   T4a asserts it for bound_pt='min' (the lag-augmented estimate built from the
   minimum processing time over compatible machines).
   T4b reports, without asserting, the upstream estimate bound_pt='mean', which
   uses the mean processing time and is not a lower bound; T4c shows that the
   untouched upstream environment exceeds its final makespan at zero lags too.
T6 lag features: under a fixed random action sequence, the two optional
   channels do not change the schedule; channel 6 equals the lag and channel 7
   equals clip(completion + lag - time, 0, lag) for scheduled operations.
T5 smoke training: 2 iterations of train_ppvc.py, batch 4, 10 modules; finite
   loss, checkpoint saved and loadable by eval_ppvc_hgnn.py.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppvc_hgnn as P  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_default_device('cpu')

from env.fjsp_env import FJSPEnv  # noqa: E402
from env.fjsp_env_lag import FJSPEnvLag  # noqa: E402
from env.load_data import load_fjs, load_time_lags, nums_detec  # noqa: E402

DEVICE = torch.device('cpu')
CFG = P.load_config()
CKPT = os.path.join(P.HGNN_DIR, 'model', 'save_10_5.pt')
DATA = os.path.join(P.REPO_ROOT, 'data', 'PPVC', '10x25+ppvc-mixed')
PY = sys.executable


def instance(i):
    return os.path.join(DATA, 'instance_{0:03d}.fjs'.format(i))


def model_for(lag_features=False):
    _, mp = P.make_paras(CFG, DEVICE, 'aware', lag_features, 1, 1, 1)
    return P.build_model(mp, CFG['train_paras'], ckpt=None if lag_features else CKPT)


def paras(lag_mode='aware', lag_features=False, bound_pt='mean'):
    return dict(P.make_paras(CFG, DEVICE, lag_mode, lag_features, 1, 1, 1)[0], bound_pt=bound_pt)


def eligible_pairs(env):
    '''Eligible (job, machine) pairs of instance 0, same rule as the upstream actor mask.'''
    step = torch.where(env.ope_step_batch > env.end_ope_biases_batch,
                       env.end_ope_biases_batch, env.ope_step_batch)[0]
    adj = env.ope_ma_adj_batch[0, step, :] == 1
    job_ok = ~(env.mask_job_procing_batch[0] | env.mask_job_finish_batch[0])
    ma_ok = ~env.mask_ma_procing_batch[0]
    ok = adj & job_ok[:, None] & ma_ok[None, :]
    return [(int(j), int(m), int(step[j])) for j, m in torch.nonzero(ok).tolist()], ok


def random_rollout(env, rng, actions=None):
    '''Roll out uniformly random eligible actions; replay `actions` if given (asserting eligibility).'''
    record = []
    k = 0
    while not bool(env.done_batch.all()):
        pairs, ok = eligible_pairs(env)
        if actions is None:
            j, m, o = pairs[rng.randint(len(pairs))]
        else:
            o, m, j = actions[k]
            assert bool(ok[j, m]), 'replayed action not eligible'
            k += 1
        record.append((o, m, j))
        env.step(torch.tensor([[o], [m], [j]]))
    return record


def test_t0_loader():
    from importlib import import_module
    argv = sys.argv
    sys.argv = [argv[0]]          # data_utils imports params, which parses argv at import
    try:
        text_to_matrix = import_module('data_utils').text_to_matrix
    finally:
        sys.argv = argv
    path = instance(0)
    with open(path) as f:
        lines = f.readlines()
    num_jobs, num_mas, num_opes = nums_detec(lines)
    loaded = load_fjs(lines, num_mas, num_opes)
    jl, pt = text_to_matrix(lines)
    lags = load_time_lags(path)
    with open(path[:-4] + '.meta.json') as f:
        meta_lag = json.load(f)['time_lag']
    print('T0 jobs {0} (ref {1}), ops {2} (ref {3}), machines {4} (ref {5}), pt equal {6}, '
          'job lengths equal {7}, lags {8} read, equal {9}'.format(
              num_jobs, len(jl), num_opes, int(jl.sum()), num_mas, pt.shape[1],
              np.array_equal(loaded[0].numpy().astype(int), pt), loaded[6].tolist() == jl.tolist(),
              len(lags), lags == meta_lag))
    assert num_jobs == len(jl) and num_opes == int(jl.sum()) == pt.shape[0] and num_mas == pt.shape[1]
    assert np.array_equal(loaded[0].numpy().astype(int), pt)
    assert loaded[6].tolist() == jl.tolist()
    assert lags == meta_lag and len(lags) == num_opes


def test_t1_zero_lag_control():
    model = model_for()
    zero = [np.zeros(220)]
    variants = {'blind': (paras('blind'), None), 'aware+zero-lags': (paras('aware'), zero)}
    for i in range(10):
        ref = P.make_file_env(instance(i), paras(), env_cls=FJSPEnv)
        ms_ref = P.greedy_rollout(ref, model)
        rng = np.random.RandomState(1000 + i)
        ref_r = P.make_file_env(instance(i), paras(), env_cls=FJSPEnv)
        acts = random_rollout(ref_r, rng)
        line = ['T1 instance_{0:03d} greedy orig {1:.1f}'.format(i, float(ms_ref[0])),
                'random orig {0:.1f}'.format(float(ref_r.makespan_batch[0]))]
        for name, (ep, lags) in variants.items():
            env = P.make_file_env(instance(i), ep, time_lags=lags)
            assert float(env.lag_batch.abs().sum()) == 0.0
            ms = P.greedy_rollout(env, model)
            assert torch.equal(ms, ms_ref), (name, ms, ms_ref)
            assert torch.equal(env.schedules_batch, ref.schedules_batch), name
            env_r = P.make_file_env(instance(i), ep, time_lags=lags)
            random_rollout(env_r, None, actions=acts)
            assert torch.equal(env_r.makespan_batch, ref_r.makespan_batch), name
            assert torch.equal(env_r.schedules_batch, ref_r.schedules_batch), name
            line.append('{0}: greedy {1:.1f} random {2:.1f} identical'.format(
                name, float(ms[0]), float(env_r.makespan_batch[0])))
        print(', '.join(line))


def test_t2_serial_oracle():
    import ppvc_instance_generator as pig
    model = model_for()
    for seed, cls in [(1, [0]), (2, [1]), (3, [2]), (4, [3]), (5, 'mixed')]:
        case = P.PPVCCaseGenerator(1, [seed], factory='small', class_mix=cls)
        ep = P.make_paras(CFG, DEVICE, 'aware', False, 1, 1, 9)[0]
        env = FJSPEnvLag(case=case, env_paras=ep)
        jl, pt, meta = case.instances[0]
        assert ((pt > 0).sum(axis=1) == 1).all()
        oracle = float(pt.sum() + np.asarray(meta['time_lag']).sum())
        ms = float(P.greedy_rollout(env, model)[0])
        print('T2 seed {0} class {1}: makespan {2:.1f}, sum pt {3} + sum lag {4} = {5:.1f}'.format(
            seed, pig.ROUTING_CLASSES[meta['routing_class'][0]], ms, int(pt.sum()),
            int(np.sum(meta['time_lag'])), oracle))
        assert ms == oracle


def _aware_rollouts(n=5, bound_pt='mean'):
    from schedule_validator import validate_schedule
    model = model_for()
    out = []
    for i in range(n):
        env = P.make_file_env(instance(i), paras('aware', bound_pt=bound_pt))
        bounds = [float(env.makespan_batch[0])]
        ms = float(P.greedy_rollout(env, model, observer=lambda e: bounds.append(float(e.makespan_batch[0])))[0])
        jl, pt, amch, ct = P.extract_schedule(env)
        lag = np.asarray(load_time_lags(instance(i)), dtype=float)
        res = validate_schedule(jl, pt, lag, amch, ct)
        out.append((i, ms, res, bounds))
    return out


_ROLLOUTS = {}


def rollouts(bound_pt='mean'):
    if bound_pt not in _ROLLOUTS:
        _ROLLOUTS[bound_pt] = _aware_rollouts(bound_pt=bound_pt)
    return _ROLLOUTS[bound_pt]


def test_t3_validator():
    for i, ms, res, _ in rollouts('mean') + rollouts('min'):
        print('T3 instance_{0:03d}: env makespan {1:.1f}, validator makespan {2:.1f}, violations {3}, '
              'lag lower bound {4:.1f}'.format(i, ms, res['makespan'], len(res['violations']), res['lower_bound']))
        assert res['feasible'] and not res['violations']
        assert res['makespan'] == ms


def _bound_line(tag, i, ms, bounds):
    b = np.asarray(bounds)
    worst = float((b - ms).max())
    print('{0} instance_{1:03d}: final {2:.1f}, initial estimate {3:.1f}, {4} estimates, '
          'max(estimate - final) {5:.2f}, estimates above final {6}'.format(
              tag, i, ms, b[0], len(b), worst, int((b > ms + 1e-6).sum())))
    return worst


def test_t4a_bound_min():
    worst = [_bound_line('T4a min', i, ms, bounds) for i, ms, _, bounds in rollouts('min')]
    assert max(worst) <= 1e-6


def test_t4b_estimate_mean_report():
    for i, ms, _, bounds in rollouts('mean'):
        _bound_line('T4b mean', i, ms, bounds)


def test_t4c_upstream_zero_lag_report():
    model = model_for()
    for i in range(10):
        env = P.make_file_env(instance(i), paras(), env_cls=FJSPEnv)
        bounds = [float(env.makespan_batch[0])]
        ms = float(P.greedy_rollout(env, model, observer=lambda e: bounds.append(float(e.makespan_batch[0])))[0])
        _bound_line('T4c upstream env, zero lags', i, ms, bounds)


def test_t6_lag_features():
    rng = np.random.RandomState(7)
    ref = P.make_file_env(instance(0), paras('aware'))
    acts = random_rollout(ref, rng)
    env = P.make_file_env(instance(0), paras('aware', lag_features=True))
    assert env.feat_opes_batch.size(1) == 8
    assert torch.equal(env.feat_opes_batch[:, 6, :], env.lag_batch)
    nonzero_steps = 0
    for o, m, j in acts:
        env.step(torch.tensor([[o], [m], [j]]))
        f = env.feat_opes_batch[0]
        ct = f[5] + f[2]
        expect = torch.minimum(torch.clamp(ct + env.lag_batch[0] - env.time[0], min=0.0), env.lag_batch[0]) * f[0]
        assert torch.equal(f[7], expect)
        assert bool((f[7] >= 0).all()) and bool((f[7] <= env.lag_batch[0]).all())
        assert torch.equal(f[6], env.lag_batch[0])
        nonzero_steps += int(bool((f[7] > 0).any()))
    assert torch.equal(env.schedules_batch, ref.schedules_batch)
    print('T6 identical schedule with lag features ({0:.1f}); remaining-lag channel nonzero at {1} of {2} '
          'decision points'.format(float(env.makespan_batch[0]), nonzero_steps, len(acts)))
    assert nonzero_steps > 0


def test_t5_smoke_training():
    here = P.HGNN_DIR
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [PY, os.path.join(here, 'train_ppvc.py'), '--n_modules', '10', '--batch_size', '4',
               '--max_iterations', '2', '--save_timestep', '2', '--vali_size', '2',
               '--run_name', 'smoke', '--save_root', tmp, '--device', 'cpu']
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stderr[-3000:]
        run = os.path.join(tmp, 'smoke')
        with open(os.path.join(run, 'train_log.json')) as f:
            log = json.load(f)
        assert len(log['loss']) == 2 and all(np.isfinite(log['loss'])), log['loss']
        assert os.path.exists(os.path.join(run, 'best.pt')) and os.path.exists(os.path.join(run, 'last.pt'))
        cmd = [PY, os.path.join(here, 'eval_ppvc_hgnn.py'), '--ckpt', os.path.join(run, 'best.pt'),
               '--arm', 'aware', '--data_path', DATA, '--max_instances', '2', '--save_dir', tmp, '--tag', 'smoke']
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stderr[-3000:]
        res = np.load(os.path.join(tmp, 'Result_greedy+hgnn-aware+smoke_10x25+ppvc-mixed.npy'))
        assert res.shape == (2, 2) and np.isfinite(res).all()
        print('T5 losses {0}, eval result {1}'.format(log['loss'], res[:, 0].tolist()))


if __name__ == '__main__':
    failed = []
    for name in sorted(k for k in dict(globals()) if k.startswith('test_t')):
        print('=' * 20, name)
        try:
            globals()[name]()
            print('PASS', name)
        except AssertionError as e:
            failed.append(name)
            print('FAIL', name, repr(e)[:500])
    print('failed:', failed if failed else 'none')
    sys.exit(1 if failed else 0)
