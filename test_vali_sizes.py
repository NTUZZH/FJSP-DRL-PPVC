"""Tests for --ppvc_vali_sizes under single-size PPVC training (train.py).

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 16-17 \
          python -m pytest -q test_vali_sizes.py

Every run is on the CPU in a temporary working directory, so the repository's
data/, trained_network/, train_log/ and train_time.txt are never written. The
preflight resolutions (test c) read the repository but write nothing.

  (a) a single-size run (--n_j 10) with --ppvc_vali_sizes 10,20 writes the
      20-module validation set to data/data_train_vali/PPVC/20x25+ppvc-mixed,
      seeds seed0 + i, holding the instances the size-mix branch generates in
      memory; a second run leaves the folder untouched, and a folder whose
      dataset_meta.json does not match aborts the run
  (b) the printed validation value equals the normalised mean computed by the
      size-mix code (validate_size_mixed of a size-mixed trainer) on the
      checkpoint the run saved
  (c) data and model names are those of the single-size run; the resolved
      validation sizes are recorded in the config snapshot only when more
      than n_j is validated; the parameter count is unchanged
  (d) variable routes: the same multi-size selection on the padded env
"""
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SUFFIX = 'valisizestest'
NAME = f'10x25+ppvc-mixed+{SUFFIX}'
VALI_SIZE = 10
COMMON = ['--data_source', 'PPVC', '--device', 'cpu', '--device_id', '',
          '--n_j', '10', '--max_updates', '1', '--num_envs', '4',
          '--vali_size', str(VALI_SIZE), '--minibatch_size', '256',
          '--validate_timestep', '1', '--use_lag_features', 'True',
          '--use_type_embedding', 'True', '--seed_train', '301',
          '--model_suffix', SUFFIX]


def _env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def _run(args, cwd, timeout=900):
    return subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=_env())


def _train(cwd, extra):
    return _run([os.path.join(REPO, 'train.py')] + COMMON + extra, cwd)


def _printed_vali(stdout):
    vals = re.findall(r'The validation quality is: ([0-9.eE+-]+)', stdout)
    assert len(vals) == 1, stdout[-2000:]
    return float(vals[0])


def _tree_digest(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, '*'))):
        with open(p, 'rb') as f:
            out[os.path.basename(p)] = (hashlib.sha256(f.read()).hexdigest(),
                                        os.stat(p).st_mtime_ns)
    return out


@pytest.fixture(scope='module')
def single_run():
    d = tempfile.mkdtemp(prefix='valisizes_')
    r = _train(d, ['--ppvc_vali_sizes', '10,20'])
    yield d, r
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (a) the 20-module validation set on disk
# --------------------------------------------------------------------------- #

# instances of the size-mix branch, Trainer.make_ppvc_dataset(save_dir=None,
# n_modules=20) on a stub carrying the run's config, against the files on disk
_SIZEMIX_SETS = r'''
import json, os, sys, types
import numpy as np
sys.path.insert(0, sys.argv[2])
vdir = sys.argv[3]
sys.argv = ['train.py'] + json.loads(sys.argv[1])
from params import configs
import train
from ppvc_instance_generator import DEFAULT_FACTORY, load_instance
stub = types.SimpleNamespace(n_j=10, config=configs, ppvc_factory=DEFAULT_FACTORY)
stub.scale_lags = types.MethodType(train.Trainer.scale_lags, stub)
jl, pt, lag, ot, mt = train.Trainer.make_ppvc_dataset(
    stub, configs.vali_size, seed0=configs.seed_train_vali_datagen,
    save_dir=None, n_modules=20)
same = []
for i in range(configs.vali_size):
    jl_d, pt_d, m_d = load_instance(os.path.join(vdir, f'instance_{i:03d}'))
    same.append(bool(np.array_equal(jl_d, jl[i]) and np.array_equal(pt_d, pt[i])
                     and np.array_equal(np.asarray(m_d['time_lag']), np.asarray(lag[i]))
                     and np.array_equal(np.asarray(m_d['op_type']), np.asarray(ot[i]))))
print(json.dumps({'n': len(jl), 'same': same}))
'''


def test_a_writes_20_module_set_with_size_mix_instances(single_run):
    d, r = single_run
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    v20 = os.path.join(d, 'data', 'data_train_vali', 'PPVC', '20x25+ppvc-mixed')
    v10 = os.path.join(d, 'data', 'data_train_vali', 'PPVC', '10x25+ppvc-mixed')
    assert len(glob.glob(os.path.join(v10, 'instance_*.fjs'))) == VALI_SIZE
    assert len(glob.glob(os.path.join(v20, 'instance_*.fjs'))) == VALI_SIZE
    with open(os.path.join(v20, 'dataset_meta.json')) as f:
        meta = json.load(f)
    assert meta == {'seed0': 100, 'n_instances': VALI_SIZE, 'n_modules': 20,
                    'ppvc_mix': 'mixed', 'ppvc_factory': 'default'}
    # the files hold exactly the size-mix branch's in-memory instances
    w = tempfile.mkdtemp(prefix='valisizes_ref_')
    try:
        p = _run(['-c', _SIZEMIX_SETS, json.dumps(COMMON + ['--ppvc_size_mix', '10,20']),
                  REPO, v20], w)
    finally:
        shutil.rmtree(w, ignore_errors=True)
    assert p.returncode == 0, p.stderr[-3000:]
    ref = json.loads(p.stdout.strip().splitlines()[-1])
    assert ref == {'n': VALI_SIZE, 'same': [True] * VALI_SIZE}


def test_a_existing_folder_is_not_overwritten(single_run):
    d, r = single_run
    assert r.returncode == 0
    vroot = os.path.join(d, 'data', 'data_train_vali', 'PPVC')
    before = {s: _tree_digest(os.path.join(vroot, s))
              for s in ('10x25+ppvc-mixed', '20x25+ppvc-mixed')}
    r2 = _train(d, ['--ppvc_vali_sizes', '10,20', '--model_suffix', SUFFIX + '2'])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    after = {s: _tree_digest(os.path.join(vroot, s)) for s in before}
    assert after == before
    # same instances -> same first validation as the first run
    assert _printed_vali(r2.stdout) == _printed_vali(r.stdout)


def test_a_mismatching_record_aborts():
    d = tempfile.mkdtemp(prefix='valisizes_bad_')
    try:
        v20 = os.path.join(d, 'data', 'data_train_vali', 'PPVC', '20x25+ppvc-mixed')
        os.makedirs(v20)
        with open(os.path.join(v20, 'dataset_meta.json'), 'w') as f:
            json.dump({'seed0': 100, 'n_instances': VALI_SIZE, 'n_modules': 30,
                       'ppvc_mix': 'mixed', 'ppvc_factory': 'default'}, f)
        r = _train(d, ['--ppvc_vali_sizes', '10,20'])
        assert r.returncode != 0
        assert 'n_modules=30' in r.stderr
        assert not glob.glob(os.path.join(d, 'trained_network', 'PPVC', '*.pth'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# (b) the printed value is the size-mix normalised mean
# --------------------------------------------------------------------------- #

# a size-mixed trainer (its validation sets built in memory by the size-mix
# branch) loads the single-size run's checkpoint and validates it
_SIZEMIX_VALI = r'''
import json, sys
argv = json.loads(sys.argv[1])
ckpt = sys.argv[2]
sys.path.insert(0, sys.argv[3])
sys.argv = ['train.py'] + argv
from params import configs
import train, torch
t = train.Trainer(configs)
assert t.size_mix and t.vali_sizes == [10, 20]
t.ppo.policy.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
v = t.validate_size_mixed()
print(json.dumps({'value': v, 'makespans': t.last_vali_makespans, 'norm': t.vali_norm}))
'''


def test_b_value_equals_size_mix_normalised_mean(single_run):
    d, r = single_run
    assert r.returncode == 0
    printed = _printed_vali(r.stdout)
    ckpt = os.path.join(d, 'trained_network', 'PPVC', f'{NAME}.pth')
    w = tempfile.mkdtemp(prefix='valisizes_mix_')
    try:
        # size-mix validation sets are generated in memory (seed0 + i)
        p = _run(['-c', _SIZEMIX_VALI,
                  json.dumps(COMMON + ['--ppvc_size_mix', '10,20',
                                       '--ppvc_vali_sizes', '10,20']), ckpt, REPO], w)
    finally:
        shutil.rmtree(w, ignore_errors=True)
    assert p.returncode == 0, p.stderr[-3000:]
    ref = json.loads(p.stdout.strip().splitlines()[-1])
    assert printed == pytest.approx(ref['value'], rel=1e-12, abs=1e-12)
    assert ref['norm'] == [216.0, 265.0]
    # the log files carry the same value; the curve entry names the sizes
    with open(os.path.join(d, 'train_log', 'PPVC', f'valiquality_{NAME}.txt')) as f:
        assert ast.literal_eval(f.read()) == [pytest.approx(printed)]
    with open(os.path.join(d, 'train_log', 'PPVC', f'valicurve_{NAME}.json')) as f:
        curve = json.load(f)
    assert len(curve) == 1 and curve[0][0] == 1
    note = curve[0][3]
    assert note['vali_sizes'] == [10, 20]
    assert note['norm'] == [216.0, 265.0]
    assert note['mean_makespan'] == pytest.approx(ref['makespans'])
    assert curve[0][1] == pytest.approx(
        np.mean([m / n for m, n in zip(note['mean_makespan'], note['norm'])]))


# --------------------------------------------------------------------------- #
# (c) names and parameter count unchanged; resolved sizes in the snapshot
# --------------------------------------------------------------------------- #

def _preflight(extra, tmp):
    out = os.path.join(tmp, 'pf.json')
    args = [os.path.join(REPO, 'preflight_train.py'), '--expect_params', '32530',
            '--out', out] + COMMON + extra
    p = _run(args, REPO)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with open(out) as f:
        return json.load(f)


def test_c_names_unchanged(single_run):
    d, r = single_run
    assert r.returncode == 0
    tmp = tempfile.mkdtemp(prefix='valisizes_pf_')
    try:
        multi = _preflight(['--ppvc_vali_sizes', '10,20'], tmp)
        only10 = _preflight(['--ppvc_vali_sizes', '10'], tmp)
        absent = _preflight([], tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    keys = ('model_name', 'data_name', 'checkpoint', 'snapshot_path', 'reward_log',
            'vali_data_path', 'n_j', 'n_m', 'size_mix')
    for k in keys:
        assert multi['derived'][k] == only10['derived'][k] == absent['derived'][k], k
    assert multi['derived']['model_name'] == NAME
    assert multi['derived']['vali_data_path'] == './data/data_train_vali/PPVC/10x25+ppvc-mixed'
    for res in (multi, only10, absent):
        assert res['param_count'] == 32530
    assert multi['derived']['vali_sizes'] == [10, 20]
    assert only10['derived']['vali_sizes'] == absent['derived']['vali_sizes'] == [10]
    assert [(v['n_modules'], v['seed0'], v['save_dir']) for v in multi['derived']['vali_sets']] \
        == [(10, 100, './data/data_train_vali/PPVC/10x25+ppvc-mixed'),
            (20, 100, './data/data_train_vali/PPVC/20x25+ppvc-mixed')]
    assert multi['snapshot']['vali_sizes_resolved'] == [10, 20]
    assert 'vali_sizes_resolved' not in only10['snapshot']
    assert 'vali_sizes_resolved' not in absent['snapshot']
    # the training run wrote its files under the single-size names
    for f in (f'trained_network/PPVC/{NAME}.pth', f'train_log/PPVC/config_{NAME}.json',
              f'train_log/PPVC/reward_{NAME}.txt', f'train_log/PPVC/valicurve_{NAME}.json'):
        assert os.path.exists(os.path.join(d, f)), f
    with open(os.path.join(d, 'train_log', 'PPVC', f'config_{NAME}.json')) as f:
        snap = json.load(f)
    assert snap['vali_sizes_resolved'] == [10, 20]
    assert snap['ppvc_vali_sizes'] == '10,20'


# --------------------------------------------------------------------------- #
# (d) variable routes
# --------------------------------------------------------------------------- #

def test_d_variable_routes_multi_size():
    d = tempfile.mkdtemp(prefix='valisizes_var_')
    try:
        r = _train(d, ['--ppvc_vali_sizes', '10,20', '--ppvc_route_variant', 'variable'])
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        name = f'10x25+ppvc-mixed-var+{SUFFIX}'
        v20 = os.path.join(d, 'data', 'data_train_vali', 'PPVC', '20x25+ppvc-mixed-var')
        with open(os.path.join(v20, 'dataset_meta.json')) as f:
            assert json.load(f)['route_variant'] == 'variable'
        with open(os.path.join(d, 'train_log', 'PPVC', f'valicurve_{name}.json')) as f:
            entry = json.load(f)[0]
        note = entry[3]
        assert note['vali_sizes'] == [10, 20]
        assert _printed_vali(r.stdout) == pytest.approx(
            np.mean([m / n for m, n in zip(note['mean_makespan'], note['norm'])]))
    finally:
        shutil.rmtree(d, ignore_errors=True)
