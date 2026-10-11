"""Tests for fine-tuning (--init_from, --lag_scale in train.py) and eval_cost.py.

Run:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 22-23 \
          python -m pytest -q test_finetune_cost.py
Training and evaluation run in a temporary working directory that holds a copy
of the source checkpoint and its config snapshot, so the repository's
trained_network/, train_log/, data/ and train_time.txt are never written.
The whole file takes a few minutes on two CPU cores.
"""
import ast
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SRC = '10x25+ppvc-mixed+full'
FT_NAME = '10x15+ppvc-mixed+fttest'
SCALE = 1.5
BASE_ARGS = ['--data_source', 'PPVC', '--device', 'cpu', '--ppvc_factory', 'tight',
             '--n_j', '10', '--use_type_embedding', 'true', '--init_from', SRC,
             '--model_suffix', 'fttest', '--num_envs', '4', '--vali_size', '5',
             '--validate_timestep', '1']


def _workdir():
    d = tempfile.mkdtemp(prefix='ftcost_')
    for sub in ('trained_network/PPVC', 'train_log/PPVC'):
        os.makedirs(os.path.join(d, sub))
    shutil.copy(os.path.join(REPO, 'trained_network', 'PPVC', f'{SRC}.pth'),
                os.path.join(d, 'trained_network', 'PPVC'))
    shutil.copy(os.path.join(REPO, 'train_log', 'PPVC', f'config_{SRC}.json'),
                os.path.join(d, 'train_log', 'PPVC'))
    return d


def _run(args, cwd, timeout=900):
    return subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


@pytest.fixture(scope='module')
def finetune_run():
    d = _workdir()
    r = _run([os.path.join(REPO, 'train.py')] + BASE_ARGS +
             ['--use_lag_features', 'true', '--max_updates', '2', '--lag_scale', str(SCALE)], d)
    yield d, r
    shutil.rmtree(d, ignore_errors=True)


def test_init_from_finetune_two_updates(finetune_run):
    d, r = finetune_run
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert f'[init_from] loaded' in r.stdout
    # snapshot records the source, its hash and the lag scale
    with open(os.path.join(d, 'train_log', 'PPVC', f'config_{FT_NAME}.json')) as f:
        snap = json.load(f)
    assert snap['init_from'] == SRC
    assert snap['lag_scale'] == SCALE
    assert len(snap['init_from_sha256']) == 64
    # validation log: zero-shot entry at update 0, then one per update
    with open(os.path.join(d, 'train_log', 'PPVC', f'valiquality_{FT_NAME}.txt')) as f:
        vali = ast.literal_eval(f.read())
    assert len(vali) == 3
    with open(os.path.join(d, 'train_log', 'PPVC', f'valicurve_{FT_NAME}.json')) as f:
        curve = json.load(f)
    assert [c[0] for c in curve] == [0, 1, 2]
    assert [c[1] for c in curve] == pytest.approx([float(v) for v in vali])
    assert os.path.exists(os.path.join(d, 'trained_network', 'PPVC', f'{FT_NAME}.pth'))


# Independent zero-shot quality of the source checkpoint on the saved validation
# instances, decoded as ONE batch like the trainer's validation. The environment
# normalises processing times by the batch-wide min and max, so a batched
# validation and a one-instance-at-a-time evaluation (eval_ppvc.py) can differ.
ZERO_SHOT_CODE = r'''
import glob, json, os, sys
import numpy as np
MODEL, VDIR = os.environ['ZS_MODEL'], os.environ['ZS_VDIR']
from params import configs
with open(f'train_log/PPVC/config_{MODEL}.json') as f:
    snap = json.load(f)
for k in ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim', 'fea_m_input_dim',
          'type_emb_dim', 'n_op_types', 'n_mch_types', 'num_heads_OAB', 'num_heads_MAB',
          'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
          'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob'):
    setattr(configs, k, snap[k])
configs.framework = snap.get('framework', 'DANIEL')
configs.device = 'cpu'
import torch
torch.set_default_device('cpu')
from model.PPO import build_policy
from common_utils import greedy_select_action
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from ppvc_instance_generator import load_instance
policy = build_policy(configs)
policy.load_state_dict(torch.load(f'trained_network/PPVC/{MODEL}.pth', map_location='cpu'))
policy.eval()
inst = [load_instance(p[:-4]) for p in sorted(glob.glob(os.path.join(VDIR, 'instance_*.fjs')))]
env = FJSPEnvForSameOpNums(inst[0][0].shape[0], inst[0][1].shape[1],
                           use_lag_features=configs.use_lag_features)
state = env.set_initial_data([i[0] for i in inst], [i[1] for i in inst],
                             time_lag_list=[i[2]['time_lag'] for i in inst],
                             op_type_list=[i[2]['op_type'] for i in inst],
                             mch_type_list=[i[2]['mch_type'] for i in inst])
while True:
    with torch.no_grad():
        pi, _ = policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                       candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                       mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                       fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                       mch_type=state.mch_type_tensor)
    state, _, done = env.step(greedy_select_action(pi).cpu().numpy())
    if done.all():
        break
print('ZERO_SHOT', float(env.current_makespan.mean()))
'''


def test_update0_equals_zero_shot(finetune_run):
    """Update-0 validation equals an independent batched zero-shot greedy decoding of
    the source checkpoint on the saved validation instances. eval_ppvc.py (one
    instance at a time, schedules re-validated against the on-disk lags) is run
    as well and must produce feasible schedules; its mean is printed only."""
    d, r = finetune_run
    assert r.returncode == 0
    with open(os.path.join(d, 'train_log', 'PPVC', f'valiquality_{FT_NAME}.txt')) as f:
        vali0 = float(ast.literal_eval(f.read())[0])
    vdir = os.path.join(d, 'data', 'data_train_vali', 'PPVC', f'10x15+ppvc-mixed-lagx{SCALE:g}')
    assert len(glob.glob(os.path.join(vdir, 'instance_*.fjs'))) == 5
    z = subprocess.run([PY, '-c', ZERO_SHOT_CODE], cwd=d, capture_output=True,
                       text=True, timeout=600, env=dict(os.environ, PYTHONPATH=REPO,
                                               ZS_MODEL=SRC, ZS_VDIR=vdir))
    assert z.returncode == 0, z.stderr[-3000:]
    zero_shot = float(z.stdout.split('ZERO_SHOT')[-1])
    out = os.path.join(d, 'zs')
    e = _run([os.path.join(REPO, 'eval_ppvc.py'), '--model_name', SRC, '--data_path', vdir,
              '--methods', 'greedy', '--device', 'cpu', '--save_dir', out], d, timeout=600)
    assert e.returncode == 0, e.stdout[-3000:] + e.stderr[-3000:]
    res = np.load(glob.glob(os.path.join(out, 'Result_greedy+*.npy'))[0])
    print(f'\nupdate-0 validation {vali0}  batched zero-shot {zero_shot}  '
          f'per-instance eval_ppvc {res[:, 0].mean()}')
    assert zero_shot == pytest.approx(vali0, abs=1e-6)


def test_lag_scale_applied_to_validation_instances(finetune_run):
    d, r = finetune_run
    assert r.returncode == 0
    vdir = os.path.join(d, 'data', 'data_train_vali', 'PPVC', f'10x15+ppvc-mixed-lagx{SCALE:g}')
    sys.path.insert(0, REPO)
    saved = sys.argv
    sys.argv = [saved[0]]   # params.py parses the process argv at import
    try:
        from ppvc_instance_generator import (ppvc_instance_generator, load_instance,
                                             TIGHT_FACTORY)
        load_instance(os.path.join(vdir, 'instance_000'))
    finally:
        sys.argv = saved
    with open(os.path.join(vdir, 'dataset_meta.json')) as f:
        assert json.load(f)['lag_scale'] == SCALE
    for i in range(5):
        _, _, meta = load_instance(os.path.join(vdir, f'instance_{i:03d}'))
        _, _, ref = ppvc_instance_generator(n_modules=10, class_mix='mixed',
                                            station_counts=TIGHT_FACTORY, seed=100 + i)
        assert np.array_equal(meta['time_lag'], np.rint(ref['time_lag'] * SCALE).astype(int))
        assert ref['time_lag'].sum() > 0


def test_eval_ppvc_keeps_dataset_lags():
    """eval_ppvc copies only architecture flags from a snapshot; lag_scale is not
    one of them, so evaluation always uses the lags stored with the dataset."""
    src = open(os.path.join(REPO, 'eval_ppvc.py')).read()
    tree = ast.parse(src)
    keys = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'ARCH_KEYS':
            keys = ast.literal_eval(node.value)
    assert keys is not None and 'lag_scale' not in keys
    assert 'lag_scale' not in src


@pytest.mark.parametrize('bad', [['--use_lag_features', 'false'],
                                 ['--use_lag_features', 'true', '--framework', 'PLAN']])
def test_init_from_architecture_mismatch_aborts(bad):
    d = _workdir()
    try:
        r = _run([os.path.join(REPO, 'train.py')] + BASE_ARGS + bad + ['--max_updates', '2'],
                 d, timeout=300)
        assert r.returncode != 0
        msg = r.stdout + r.stderr
        assert 'architecture of' in msg and 'does not match' in msg, msg[-2000:]
        assert not glob.glob(os.path.join(d, 'trained_network', 'PPVC', '*fttest*'))
        assert not os.path.exists(os.path.join(d, 'data'))  # aborted before data generation
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_eval_cost_10x25_two_instances():
    d = tempfile.mkdtemp(prefix='cost_')
    try:
        out = os.path.join(d, 'cost.json')
        r = _run([os.path.join(REPO, 'eval_cost.py'), '--datasets', '10x25', '--n_inst', '2',
                  '--gpu', 'off', '--out', out], REPO, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        with open(out) as f:
            res = json.load(f)
        assert res['models']
        for m in res['models']:
            s = res['static'][m]
            assert s['params_total'] > 0 and s['checkpoint_bytes'] > 0
            assert res['cpu'][m]['_params_module'] == s['params_total']
            e = res['cpu'][m]['10x25']
            assert e['n_instances'] == 2
            for k in ('latency_s_per_instance_mean', 'latency_ms_per_decision_mean',
                      'forward_ms_per_decision_mean', 'peak_rss_MB'):
                assert math.isfinite(e[k]) and e[k] > 0, (m, k, e[k])
        assert res['runs'][0]['cpu_worker']['torch_threads'] == 1
        assert len(res['runs'][0]['load_avg_start']) == 3
        assert os.path.exists(os.path.join(d, 'cost.md'))
    finally:
        shutil.rmtree(d, ignore_errors=True)
