"""
Tests for the PLAN policy backbone (model/plan.py, selected by --framework PLAN).

  T1  DANIEL regression: framework DANIEL, all PPVC flags off, the published
      checkpoint trained_network/SD2/10x5+mix.pth decoded greedily on
      data/SD2/10x5+mix (100 instances) must reproduce the stored result file
      test_results/SD2/10x5+mix/Result_DANIELG+10x5+mix_10x5+mix.npy
      bit-exactly (mean 408.40, 100/100 instances identical).
  T2  PLAN reproduction: each of the five PLAN SD2 10x5 checkpoints of the
      original PLAN code loads into our PLAN class with strict=True (13,506
      parameters) and is decoded greedily on data/SD2/10x5+mix. Reports the
      five means and their average (reference: 407.86 in the PLAN paper).
  T3  PLAN PPVC smoke: parameter-count table for arms A1 (flags off), A2 (lag
      features) and A3 (lag features + type embeddings) on both backbones,
      one forward pass per arm on a PPVC instance (finite pi summing to 1, and
      finite v), then a 2-update PLAN A3 training run of train.py with finite
      losses.
  T4  eval_ppvc.py loads the PLAN smoke checkpoint of T3 and decodes greedily
      the first 10 instances of data/PPVC/10x25+ppvc-mixed; every schedule
      passes the schedule validator (10/10 feasible, 0 violations).

Run (CPU only):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 python test_plan_backbone.py [T1 T2 T3 T4]
"""
import copy
import glob
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

SELECTED = [a.upper() for a in sys.argv[1:]] or ['T1', 'T2', 'T3', 'T4']
# params.py parses argv at import time: give it a clean CPU configuration
sys.argv = [sys.argv[0], '--device', 'cpu']

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from params import configs
from common_utils import greedy_select_action, setup_seed
from data_utils import load_data_from_files
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from model.PPO import build_policy

PLAN_CKPT_DIR = 'external/PLAN/FJSP_Deterministic/trained_network/PLAN/SD2'
PLAN_N_PARAMS = 13506
SD2_DATA = './data/SD2/10x5+mix'
SD2_REF = './test_results/SD2/10x5+mix/Result_DANIELG+10x5+mix_10x5+mix.npy'
SMOKE_MODEL = '10x25+ppvc-mixed+plan-smoke'
PPVC_DATA = 'data/PPVC/10x25+ppvc-mixed'


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def sub_env():
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    return env


def greedy_sd2(policy, data):
    """greedy decoding, one instance at a time (as test_trained_model.py)"""
    setup_seed(configs.seed_test)
    policy.eval()
    n_j = data[0][0].shape[0]
    n_op, n_m = data[1][0].shape
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    makespans = []
    for i in range(len(data[0])):
        state = env.set_initial_data([data[0][i]], [data[1][i]])
        while True:
            with torch.no_grad():
                pi, _ = policy(fea_j=state.fea_j_tensor,
                               op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor,
                               fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor,
                               comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor)
            action = greedy_select_action(pi)
            state, _, done = env.step(actions=action.cpu().numpy())
            if done:
                break
        makespans.append(float(env.current_makespan[0]))
    return np.array(makespans)


def sd2_config(framework):
    cfg = copy.deepcopy(configs)
    cfg.framework = framework
    cfg.use_lag_features = False
    cfg.use_type_embedding = False
    cfg.fea_j_input_dim = 10
    cfg.fea_m_input_dim = 8
    return cfg


def t1_daniel_regression():
    print('T1  DANIEL regression on SD2 10x5+mix (published checkpoint) ...')
    cfg = sd2_config('DANIEL')
    policy = build_policy(cfg)
    policy.load_state_dict(torch.load('./trained_network/SD2/10x5+mix.pth',
                                      map_location='cpu'), strict=True)
    print(f'    DANIEL parameters: {n_params(policy)}')
    data = load_data_from_files(SD2_DATA)
    ms = greedy_sd2(policy, data)
    ref = np.load(SD2_REF)[:, 0]
    n_same = int(np.sum(ms == ref))
    print(f'    mean makespan {ms.mean():.2f} (reference {ref.mean():.2f}), '
          f'identical instances {n_same}/{len(ref)}')
    assert len(ms) == len(ref) == 100
    assert n_same == 100, 'DANIEL regression broken: not bit-exact'
    assert round(float(ms.mean()), 2) == 408.40
    print('    PASS')


def t2_plan_reproduction():
    print('T2  PLAN checkpoints of the original code on SD2 10x5+mix ...')
    cfg = sd2_config('PLAN')
    data = load_data_from_files(SD2_DATA)
    means = []
    for k in range(5):
        path = os.path.join(PLAN_CKPT_DIR, f'10x5+mix_{k}.pth')
        policy = build_policy(cfg)
        assert type(policy).__name__ == 'PLAN'
        assert n_params(policy) == PLAN_N_PARAMS, n_params(policy)
        sd = torch.load(path, map_location='cpu')
        assert len(sd) == 42, len(sd)
        policy.load_state_dict(sd, strict=True)
        ms = greedy_sd2(policy, data)
        means.append(ms.mean())
        print(f'    {os.path.basename(path)}: parameters {n_params(policy)}, '
              f'strict load OK, greedy mean makespan {ms.mean():.2f}')
    print(f'    average of the five means: {np.mean(means):.2f} '
          f'(PLAN paper, Table 1: 407.86)')
    print('    PASS (strict loads and evaluation completed)')


def arm_config(framework, arm):
    cfg = copy.deepcopy(configs)
    cfg.framework = framework
    cfg.fea_m_input_dim = 8
    cfg.use_lag_features = arm in ('A2', 'A3')
    cfg.use_type_embedding = arm == 'A3'
    cfg.fea_j_input_dim = 12 if cfg.use_lag_features else 10
    return cfg


def t3_ppvc_smoke():
    print('T3  parameter table and PPVC forward pass per arm ...')
    from ppvc_instance_generator import ppvc_instance_generator, DEFAULT_FACTORY
    jl, pt, meta = ppvc_instance_generator(n_modules=5, class_mix='mixed',
                                           station_counts=DEFAULT_FACTORY, seed=42)
    table = {}
    for framework in ('DANIEL', 'PLAN'):
        for arm in ('A1', 'A2', 'A3'):
            cfg = arm_config(framework, arm)
            torch.manual_seed(0)
            policy = build_policy(cfg)
            policy.eval()
            table[(framework, arm)] = n_params(policy)
            env = FJSPEnvForSameOpNums(len(jl), pt.shape[1],
                                       use_lag_features=cfg.use_lag_features)
            state = env.set_initial_data([jl], [pt], time_lag_list=[meta['time_lag']],
                                         op_type_list=[meta['op_type']],
                                         mch_type_list=[meta['mch_type']])
            assert state.fea_j_tensor.shape[-1] == cfg.fea_j_input_dim
            with torch.no_grad():
                pi, v = policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor,
                               op_type=state.op_type_tensor if cfg.use_type_embedding else None,
                               mch_type=state.mch_type_tensor if cfg.use_type_embedding else None)
            assert torch.isfinite(v).all()
            assert torch.isfinite(pi).all() and abs(float(pi.sum()) - 1.0) < 1e-5
    print('    | arm | DANIEL | PLAN |')
    print('    |---|---|---|')
    for arm, label in (('A1', 'A1 flags off'), ('A2', 'A2 lag features'),
                       ('A3', 'A3 lag features + type embeddings')):
        print(f'    | {label} | {table[("DANIEL", arm)]:,} | {table[("PLAN", arm)]:,} |')

    print('T3  PLAN A3 training smoke (train.py, 2 updates) ...')
    cmd = [sys.executable, 'train.py', '--data_source', 'PPVC', '--n_j', '10',
           '--framework', 'PLAN', '--use_lag_features', 'True',
           '--use_type_embedding', 'True', '--max_updates', '2',
           '--validate_timestep', '1', '--vali_size', '10', '--device', 'cpu',
           '--model_suffix', 'smoke']
    print('    $ ' + ' '.join(cmd[1:]))
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=sub_env())
    print('    ' + '\n    '.join(line for line in out.stdout.splitlines()
                                 if 'progress' not in line))
    if out.returncode != 0:
        print(out.stderr[-4000:])
    assert out.returncode == 0, 'train.py failed'
    losses = [float(x) for x in re.findall(r'Mean_loss: (\S+),', out.stdout)]
    valis = [float(x) for x in re.findall(r'validation quality is: (\S+) ', out.stdout)]
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert len(valis) == 2 and all(np.isfinite(valis)), valis
    assert f'model name :{SMOKE_MODEL}' in out.stdout
    assert os.path.exists(f'./trained_network/PPVC/{SMOKE_MODEL}.pth')
    print(f'    losses {losses}, validation makespans {valis}, '
          f'{time.time() - t0:.0f} s')
    print('    PASS')


def t4_eval_ppvc_plan():
    print('T4  eval_ppvc.py on the PLAN smoke checkpoint, first 10 instances ...')
    assert os.path.exists(f'./trained_network/PPVC/{SMOKE_MODEL}.pth'), 'run T3 first'
    save_dir = tempfile.mkdtemp(prefix='plan_t4_')
    cmd = [sys.executable, 'eval_ppvc.py', '--model_name', SMOKE_MODEL,
           '--data_path', PPVC_DATA, '--methods', 'greedy', '--device', 'cpu',
           '--max_instances', '10', '--save_dir', save_dir]
    print('    $ ' + ' '.join(cmd[1:-2]) + ' --save_dir <tmp>')
    out = subprocess.run(cmd, capture_output=True, text=True, env=sub_env())
    print('    ' + '\n    '.join(out.stdout.splitlines()))
    if out.returncode != 0:
        print(out.stderr[-4000:])
    assert out.returncode == 0, 'eval_ppvc.py failed (an infeasible schedule asserts)'
    assert 'framework=PLAN' in out.stdout
    m = re.search(r'^greedy\s+(\S+)\s+\S+\s+\S+\s+(\d+)/(\d+)', out.stdout, re.M)
    assert m, 'summary line not found'
    mean_ms, feas, n = float(m.group(1)), int(m.group(2)), int(m.group(3))
    assert n == 10 and feas == 10, (feas, n)
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(save_dir, '*')))
    assert f'Result_greedy+{SMOKE_MODEL}_10x25+ppvc-mixed.npy' in files, files

    # independent re-check of the validator on the saved checkpoint is done
    # inside eval_ppvc (assert per schedule); report the count here
    print(f'    greedy mean makespan {mean_ms:.1f} over {n} instances, '
          f'validator feasible {feas}/{n} (0 violations); output files {files}')
    print('    PASS')


TESTS = {'T1': t1_daniel_regression, 'T2': t2_plan_reproduction,
         'T3': t3_ppvc_smoke, 'T4': t4_eval_ppvc_plan}

if __name__ == '__main__':
    for name in SELECTED:
        t0 = time.time()
        TESTS[name]()
        print(f'    [{name} {time.time() - t0:.1f} s]\n')
    print('ALL SELECTED TESTS PASSED: ' + ' '.join(SELECTED))
