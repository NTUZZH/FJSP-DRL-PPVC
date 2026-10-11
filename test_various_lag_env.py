"""
test_various_lag_env.py
-----------------------
Tests for the lag-aware FJSPEnvForVariousOpNums (modules with different
operation counts) and the variable-route PPVC generator.

  T1  equivalence: on 5 fixed-route PPVC instances the various-op env and the
      same-op env give identical states (all 12 op channels, lag features on,
      type indices), rewards, bounds and makespans under the same SPT action
      sequence, per instance and as one batch
  T2  serial oracle: one variable-length module on SMALL_FACTORY (one machine
      per op) has makespan == sum(pt) + sum(lags), alone and padded in a batch
  T3  validator: SPT rollouts on 5 variable instances are feasible for
      schedule_validator and the env makespan equals the validator's
  T4  bound admissibility along the T3 rollouts: at every state the bound
      never exceeds the final makespan (per op and for the max)
  T5  DANIEL regression with the PPVC options off: an SD1 checkpoint gives the
      reference makespans through the various-op env on the padded SD1 10x5
      validation set (reference computed with the unmodified env) and on
      Brandimarte (reference = test_results/BenchData)
  T6  smoke: train.py arm A3 (lag features + type embedding) with
      --ppvc_route_variant variable, 2 updates, CPU; losses finite
  T7  padding isolation: a variable instance gets identical features and an
      identical policy output alone and inside a padded batch
  T8  CP-SAT: a 5-module variable instance is solved to optimality and the
      schedule is validated

Run all:       python test_various_lag_env.py
Run a subset:  python test_various_lag_env.py T1 T3
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

SELECTED = [a.upper() for a in sys.argv[1:]]
# params.py parses argv at import time: give it a clean CPU configuration
sys.argv = [sys.argv[0], '--device', 'cpu']

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_device('cpu')

from params import configs
from common_utils import heuristic_select_action, greedy_select_action
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from ppvc_instance_generator import (ppvc_instance_generator, load_instance,
                                     SMALL_FACTORY, DEFAULT_FACTORY)
from schedule_validator import validate_schedule

FIXED_DIR = 'data/PPVC/10x25+ppvc-mixed'
VAR_DIR = 'data/PPVC/10x25+ppvc-mixed-var'

# Greedy makespans of trained_network/SD1/10x5.pth on the 100 instances of
# data/data_train_vali/SD1/10x5 (44 to 56 ops, one padded batch, as in
# Trainer.validate_envs_with_various_op_nums), computed with the unmodified
# DANIEL FJSPEnvForVariousOpNums before the PPVC extensions were added.
SD1_VALI_REF = [
    85, 87, 103, 97, 85, 101, 95, 85, 120, 106, 93, 101, 99, 101, 97, 107, 106, 88, 101, 104,
    108, 113, 107, 93, 91, 96, 91, 95, 100, 125, 113, 101, 99, 104, 124, 81, 103, 105, 115, 104,
    111, 99, 119, 107, 110, 121, 108, 88, 90, 107, 105, 109, 116, 105, 116, 96, 102, 96, 112, 99,
    121, 110, 93, 100, 112, 105, 98, 88, 102, 110, 122, 104, 109, 116, 101, 110, 101, 111, 120, 122,
    106, 103, 123, 119, 95, 92, 103, 108, 112, 84, 116, 91, 120, 100, 108, 89, 116, 103, 114, 93]

RESULTS = []


def check(flag, label, detail=''):
    RESULTS.append((label, bool(flag)))
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f'  ({detail})' if detail else ''))
    return bool(flag)


def stems_of(directory, n):
    return [p[:-len('.fjs')] for p in sorted(glob.glob(os.path.join(directory, 'instance_*.fjs')))[:n]]


def init_env(env, instances, with_types=True):
    """set_initial_data for a list of (jl, pt, meta) instances."""
    kw = dict(time_lag_list=[m['time_lag'] for _, _, m in instances])
    if with_types:
        kw['op_type_list'] = [m['op_type'] for _, _, m in instances]
        kw['mch_type_list'] = [m['mch_type'] for _, _, m in instances]
    return env.set_initial_data([jl for jl, _, _ in instances],
                                [pt for _, pt, _ in instances], **kw)


def spt_actions(env, envs):
    """Deterministic SPT for the listed envs: the compatible, currently
    available (op, machine) pair with the smallest processing time, ties to
    the lowest action index."""
    acts = []
    for k in envs:
        pt = np.where(env.dynamic_pair_mask[k], np.inf, env.candidate_pt[k]).reshape(-1)
        acts.append(int(np.argmin(pt)))
    return np.array(acts)


def incomplete(env):
    if isinstance(env, FJSPEnvForVariousOpNums):
        return list(np.where(~env.done_flag)[0])
    return list(np.where(env.done() == 0)[0])


STATE_FIELDS = ('fea_j_tensor', 'op_mask_tensor', 'fea_m_tensor', 'mch_mask_tensor',
                'dynamic_pair_mask_tensor', 'comp_idx_tensor', 'candidate_tensor',
                'fea_pairs_tensor', 'op_type_tensor', 'mch_type_tensor')


def state_diff(s1, s2, terminal, rows1=None, rows2=None):
    """Names of state fields that differ. At the terminal state the same-op env
    skips the feature normalization (original DANIEL code), so fea_j/fea_m are
    compared only at non-terminal states; the policy never reads them there."""
    bad = []
    for f in STATE_FIELDS:
        if terminal and f in ('fea_j_tensor', 'fea_m_tensor'):
            continue
        a, b = getattr(s1, f), getattr(s2, f)
        if a is None and b is None:
            continue
        a = a.numpy() if rows1 is None else a.numpy()[rows1]
        b = b.numpy() if rows2 is None else b.numpy()[rows2]
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(f)
    return bad


# ---------------------------------------------------------------------------
def t1_equivalence():
    print('=' * 72)
    print('T1 equivalence: various-op env == same-op env on fixed-route instances')
    print('=' * 72)
    insts = [load_instance(s) for s in stems_of(FIXED_DIR, 5)]
    J, M = insts[0][0].shape[0], insts[0][1].shape[1]
    all_actions = []
    ok = True
    for i, inst in enumerate(insts):
        e1 = FJSPEnvForSameOpNums(J, M, use_lag_features=True)
        e2 = FJSPEnvForVariousOpNums(J, M, use_lag_features=True, isolate_padding=True)
        s1, s2 = init_env(e1, [inst]), init_env(e2, [inst])
        bad = state_diff(s1, s2, terminal=False)
        same_lb = np.array_equal(e1.op_ct_lb, e2.op_ct_lb)
        np.random.seed(i)
        actions, n_steps, rew_ok = [], 0, True
        while not e1.done().all():
            a = heuristic_select_action('SPT', e1)
            actions.append(a)
            s1, r1, d1 = e1.step(np.array([a]))
            s2, r2, d2 = e2.step(np.array([a]))
            n_steps += 1
            terminal = bool(d1.all())
            bad += state_diff(s1, s2, terminal)
            rew_ok &= np.array_equal(r1, r2)
            same_lb &= np.array_equal(e1.op_ct_lb, e2.op_ct_lb)
            if bool(d2.all()) != terminal:
                bad.append('done')
        all_actions.append(actions)
        ms1, ms2 = e1.current_makespan[0], e2.current_makespan[0]
        good = (not bad) and rew_ok and same_lb and ms1 == ms2
        ok &= good
        print(f'  instance {i}: steps={n_steps} makespan same-op={ms1:.0f} various-op={ms2:.0f} '
              f'rewards equal={rew_ok} bounds equal={same_lb} state mismatches={sorted(set(bad))}')
    check(ok, 'T1a per-instance: identical states (12 op channels), rewards, bounds, makespans')

    # the same 5 instances as one batch, each env replaying its own actions
    e1 = FJSPEnvForSameOpNums(J, M, use_lag_features=True)
    e2 = FJSPEnvForVariousOpNums(J, M, use_lag_features=True, isolate_padding=True)
    s1, s2 = init_env(e1, insts), init_env(e2, insts)
    bad = state_diff(s1, s2, terminal=False)
    rew_ok = True
    for t in range(len(all_actions[0])):
        acts = np.array([all_actions[k][t] for k in range(len(insts))])
        s1, r1, d1 = e1.step(acts)
        s2, r2, d2 = e2.step(acts)
        bad += state_diff(s1, s2, bool(d1.all()))
        rew_ok &= np.array_equal(r1, r2)
    ms_ok = np.array_equal(e1.current_makespan, e2.current_makespan)
    print(f'  batch of 5: makespans {e1.current_makespan.tolist()} vs {e2.current_makespan.tolist()}')
    check(not bad and rew_ok and ms_ok,
          'T1b batch of 5: identical states, rewards and makespans',
          f'state mismatches={sorted(set(bad))}')


# ---------------------------------------------------------------------------
def t2_serial_oracle():
    print('=' * 72)
    print('T2 serial oracle: one variable-length module, one machine per op')
    print('=' * 72)
    seed = 0
    while True:
        jl, pt, meta = ppvc_instance_generator(1, 'mixed', SMALL_FACTORY, seed=seed,
                                               route_variant='variable')
        if jl[0] < 22:
            break
        seed += 1
    lag = meta['time_lag']
    assert ((pt > 0).sum(axis=1) == 1).all()
    oracle = float(pt.max(axis=1).sum() + lag.sum())
    print(f'  seed={seed} ops={jl[0]} sum(pt)={pt.max(axis=1).sum()} sum(lag)={lag.sum()} '
          f'last-op lag={lag[-1]} oracle={oracle}')
    env = FJSPEnvForVariousOpNums(1, pt.shape[1], use_lag_features=True, isolate_padding=True)
    init_env(env, [(jl, pt, meta)])
    np.random.seed(0)
    while not env.done().all():
        env.step(np.array([heuristic_select_action('SPT', env)]))
    check(env.current_makespan[0] == oracle, 'T2a makespan == sum(pt) + sum(lags)',
          f'env={env.current_makespan[0]} oracle={oracle}')

    # padded: the short module next to a 22-op module of the same factory
    jl2, pt2, meta2 = ppvc_instance_generator(1, 'mixed', SMALL_FACTORY, seed=seed)
    env = FJSPEnvForVariousOpNums(1, pt.shape[1], use_lag_features=True, isolate_padding=True)
    init_env(env, [(jl, pt, meta), (jl2, pt2, meta2)])
    while not env.done_flag.all():
        env.step(spt_actions(env, incomplete(env)))
    oracle2 = float(pt2.max(axis=1).sum() + meta2['time_lag'].sum())
    check(env.current_makespan[0] == oracle and env.current_makespan[1] == oracle2,
          'T2b padded batch [short, 22-op]: both makespans equal their serial oracles',
          f'env={env.current_makespan.tolist()} oracles={[oracle, oracle2]}')


# ---------------------------------------------------------------------------
def spt_rollout_record(inst, seed):
    """SPT rollout (common_utils rule, batch of 1) in the various-op env,
    recording machine assignments and the bound at every state."""
    jl, pt, meta = inst
    env = FJSPEnvForVariousOpNums(jl.shape[0], pt.shape[1], use_lag_features=True,
                                  isolate_padding=True)
    init_env(env, [inst])
    np.random.seed(seed)
    n = pt.shape[0]
    amch = np.full(n, -1, dtype=int)
    bounds = [env.op_ct_lb[0].copy()]
    rewards = []
    while not env.done().all():
        a = heuristic_select_action('SPT', env)
        j, m = a // env.number_of_machines, a % env.number_of_machines
        amch[env.candidate[0, j]] = m
        _, r, _ = env.step(np.array([a]))
        rewards.append(float(r[0]))
        bounds.append(env.op_ct_lb[0].copy())
    return env, amch, bounds, rewards


def t3_t4_validator_and_bound(run_t3=True, run_t4=True):
    print('=' * 72)
    print('T3/T4 validator and bound admissibility on variable instances')
    print('=' * 72)
    insts = [load_instance(s) for s in stems_of(VAR_DIR, 5)]
    v_ok, ms_ok, lb_op_ok, lb_max_ok, tight_ok, tele_ok, true_ok = (True,) * 7
    for i, inst in enumerate(insts):
        jl, pt, meta = inst
        env, amch, bounds, rewards = spt_rollout_record(inst, seed=i)
        res = validate_schedule(jl, pt, meta['time_lag'], amch, env.true_op_ct[0])
        v_ok &= res['feasible'] and not res['violations']
        ms_ok &= abs(res['makespan'] - env.current_makespan[0]) < 1e-6
        # admissibility in the env's normalized time (the reward's units)
        final_ct = env.op_ct[0]
        final_ms = final_ct.max()
        worst_op = max(float((b - final_ct).max()) for b in bounds)
        worst_max = max(float(b.max() - final_ms) for b in bounds)
        lb_op_ok &= worst_op <= 1e-9
        lb_max_ok &= worst_max <= 1e-9
        tight_ok &= abs(bounds[-1].max() - final_ms) < 1e-9
        tele_ok &= abs(sum(rewards) - (bounds[0].max() - bounds[-1].max())) < 1e-6
        # the same bound recomputed in true hours from the true schedule's
        # prefix at a few states: scheduled ops at their true completion,
        # unscheduled ops chained by lag + min true pt
        true_worst = -np.inf
        min_pt = np.where(pt > 0, pt, np.inf).min(axis=1)
        lag = meta['time_lag'].astype(float)
        first = np.concatenate([[0], np.cumsum(jl)[:-1]])
        env2 = FJSPEnvForVariousOpNums(jl.shape[0], pt.shape[1], use_lag_features=True,
                                       isolate_padding=True)
        init_env(env2, [inst])
        np.random.seed(i)
        while not env2.done().all():
            lb = np.zeros(pt.shape[0])
            for j in range(jl.shape[0]):
                for o in range(first[j], first[j] + jl[j]):
                    if env2.op_scheduled_flag[0, o]:
                        lb[o] = env2.true_op_ct[0, o]
                    else:
                        prev = lb[o - 1] + lag[o - 1] if o > first[j] else 0.0
                        lb[o] = prev + min_pt[o]
            true_worst = max(true_worst, lb.max() - env.current_makespan[0])
            env2.step(np.array([heuristic_select_action('SPT', env2)]))
        true_ok &= true_worst <= 1e-9
        print(f'  instance {i}: ops={pt.shape[0]} lengths={jl.tolist()} makespan env='
              f'{env.current_makespan[0]:.0f} validator={res["makespan"]:.0f} '
              f'violations={len(res["violations"])} max(bound-final) per-op={worst_op:.3g} '
              f'max={worst_max:.3g} true-hours={true_worst:.3g}')
    if run_t3:
        check(v_ok, 'T3a SPT schedules are validator-clean (true lags)')
        check(ms_ok, 'T3b env makespan == validator makespan')
    if run_t4:
        check(lb_op_ok, 'T4a per-op bound never exceeds the final completion time')
        check(lb_max_ok, 'T4b max bound never exceeds the final makespan')
        check(tight_ok, 'T4c final bound equals the final makespan')
        check(tele_ok, 'T4d rewards telescope to initial bound - final makespan')
        check(true_ok, 'T4e bound in true hours never exceeds the true final makespan')


# ---------------------------------------------------------------------------
def build_policy(lag_features, type_embedding, seed=0):
    from model.PPO import PPO_initialize
    configs.use_lag_features = lag_features
    configs.use_type_embedding = type_embedding
    configs.fea_j_input_dim = 12 if lag_features else 10
    torch.manual_seed(seed)
    ppo = PPO_initialize()
    ppo.policy.eval()
    return ppo


def policy_pi(ppo, state, rows):
    with torch.no_grad():
        idx = torch.as_tensor(rows, dtype=torch.long)
        pi, _ = ppo.policy(fea_j=state.fea_j_tensor[idx], op_mask=state.op_mask_tensor[idx],
                           candidate=state.candidate_tensor[idx], fea_m=state.fea_m_tensor[idx],
                           mch_mask=state.mch_mask_tensor[idx], comp_idx=state.comp_idx_tensor[idx],
                           dynamic_pair_mask=state.dynamic_pair_mask_tensor[idx],
                           fea_pairs=state.fea_pairs_tensor[idx],
                           op_type=None if state.op_type_tensor is None else state.op_type_tensor[idx],
                           mch_type=None if state.mch_type_tensor is None else state.mch_type_tensor[idx])
    return pi


def t5_daniel_regression():
    print('=' * 72)
    print('T5 DANIEL regression through the various-op env (PPVC options off)')
    print('=' * 72)
    from data_utils import load_data_from_files
    ppo = build_policy(False, False)
    ppo.policy.load_state_dict(torch.load('trained_network/SD1/10x5.pth', map_location='cpu'))
    ppo.policy.eval()

    # (a) padded SD1 validation batch, as in Trainer.validate_envs_with_various_op_nums
    jl, pt = load_data_from_files('./data/data_train_vali/SD1/10x5')
    env = FJSPEnvForVariousOpNums(10, 5)
    state = env.set_initial_data(jl, pt)
    while True:
        rows = incomplete(env)
        a = greedy_select_action(policy_pi(ppo, state, rows))
        state, _, done = env.step(a.numpy())
        if done.all():
            break
    got = env.current_makespan.astype(int).tolist()
    print(f'  SD1 10x5 validation (ops {min(p.shape[0] for p in pt)}-{max(p.shape[0] for p in pt)}): '
          f'mean makespan={np.mean(got):.2f} reference={np.mean(SD1_VALI_REF):.2f}')
    check(got == SD1_VALI_REF, 'T5a SD1 validation set: 100/100 makespans equal the unmodified env',
          f'mismatches={sum(g != r for g, r in zip(got, SD1_VALI_REF))}')

    # (b) Brandimarte, one instance at a time (as test_trained_model.py)
    ref = np.load('test_results/BenchData/Brandimarte/Result_DANIELG+10x5_Brandimarte.npy')[:, 0]
    jl, pt = load_data_from_files('./data/BenchData/Brandimarte')
    ms = []
    for k in range(len(jl)):
        env = FJSPEnvForVariousOpNums(jl[k].shape[0], pt[k].shape[1])
        state = env.set_initial_data([jl[k]], [pt[k]])
        while True:
            a = greedy_select_action(policy_pi(ppo, state, [0]))
            state, _, done = env.step(a.numpy())
            if done.all():
                break
        ms.append(float(env.current_makespan[0]))
    print(f'  Brandimarte makespans: {ms}')
    print(f'  reference (test_results):  {ref.tolist()}')
    check(np.array_equal(np.array(ms), ref), 'T5b Brandimarte: makespans equal test_results/BenchData')


# ---------------------------------------------------------------------------
def t6_train_smoke():
    print('=' * 72)
    print('T6 train.py smoke: A3, variable routes, 2 updates, CPU')
    print('=' * 72)
    suffix = 'var_smoke_test'
    name = f'10x25+ppvc-mixed-var+{suffix}'
    vali_dir = 'data/data_train_vali/PPVC/10x25+ppvc-mixed-var'
    artifacts = [f'trained_network/PPVC/{name}.pth',
                 f'train_log/PPVC/config_{name}.json',
                 f'train_log/PPVC/reward_{name}.txt',
                 f'train_log/PPVC/valiquality_{name}.txt']
    vali_existed = os.path.exists(vali_dir)
    tt = open('train_time.txt').read() if os.path.exists('train_time.txt') else None
    cmd = [sys.executable, 'train.py', '--data_source', 'PPVC', '--n_j', '10',
           '--ppvc_mix', 'mixed', '--ppvc_factory', 'default',
           '--ppvc_route_variant', 'variable',
           '--use_lag_features', 'True', '--use_type_embedding', 'True',
           '--max_updates', '2', '--validate_timestep', '1', '--vali_size', '10',
           '--num_envs', '8', '--device', 'cpu', '--model_suffix', suffix]
    print('  ' + ' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    lines = [ln for ln in out.splitlines()
             if 'Episode' in ln or 'validation quality' in ln or 'Error' in ln or 'Traceback' in ln]
    for ln in lines:
        print('   | ' + ln.strip())
    losses = [float(x) for x in re.findall(r'Mean_loss: ([-\d.eE+naif]+)', out)]
    valis = [float(x) for x in re.findall(r'validation quality is: ([-\d.eE+naif]+)', out)]
    print(f'  exit={proc.returncode} time={time.time() - t0:.0f}s losses={losses} vali={valis}')
    ok = proc.returncode == 0 and len(losses) == 2 and all(np.isfinite(losses)) \
        and len(valis) == 2 and all(np.isfinite(valis))
    if proc.returncode != 0:
        print(out[-3000:])
    # remove the smoke artifacts so no analysis script picks them up
    for p in artifacts:
        if os.path.exists(p):
            os.remove(p)
    if not vali_existed and os.path.exists(vali_dir):
        shutil.rmtree(vali_dir)
    if tt is not None:
        with open('train_time.txt', 'w') as f:
            f.write(tt)
    check(ok, 'T6 train.py A3 variable: 2 updates, finite losses and validation makespans')


# ---------------------------------------------------------------------------
def t7_padding_isolation():
    print('=' * 72)
    print('T7 padding isolation: alone vs inside a padded batch')
    print('=' * 72)
    insts = [load_instance(s) for s in stems_of(VAR_DIR, 5)]
    J, M = insts[0][0].shape[0], insts[0][1].shape[1]
    n_ops = [p.shape[0] for _, p, _ in insts]
    print(f'  op counts in the batch: {n_ops}')
    ppo = build_policy(True, True, seed=1)
    batch = FJSPEnvForVariousOpNums(J, M, use_lag_features=True, isolate_padding=True)
    sb = init_env(batch, insts)
    alone = [FJSPEnvForVariousOpNums(J, M, use_lag_features=True, isolate_padding=True)
             for _ in insts]
    sa = [init_env(alone[k], [insts[k]]) for k in range(len(insts))]
    nmax = max(n_ops)
    feat_bad, dummy_bad, pi_err, ms_ok = [], 0, 0.0, True
    while not batch.done_flag.all():
        rows = incomplete(batch)
        for k in rows:
            n = n_ops[k]
            real = np.arange(n)
            # every state field of the real rows equals the solo run
            for f in STATE_FIELDS:
                a, b = getattr(sb, f), getattr(sa[k], f)
                if a is None:
                    continue
                a, b = a.numpy()[k], b.numpy()[0]
                if f in ('fea_j_tensor', 'op_mask_tensor', 'op_type_tensor'):
                    a = a[real]
                if not np.array_equal(a, b):
                    feat_bad.append(f)
            # dummy rows: zero features, both edges masked
            if n < nmax:
                fj = sb.fea_j_tensor.numpy()[k, n:]
                om = sb.op_mask_tensor.numpy()[k, n:]
                dummy_bad += int(np.abs(fj).sum() != 0 or not (om[:, 0] == 1).all()
                                 or not (om[:, 2] == 1).all())
            pb = policy_pi(ppo, sb, [k])[0]
            ps = policy_pi(ppo, sa[k], [0])[0]
            pi_err = max(pi_err, float((pb - ps).abs().max()))
        acts = spt_actions(batch, rows)
        sb, _, _ = batch.step(acts)
        for k, a in zip(rows, acts):
            sa[k], _, _ = alone[k].step(np.array([a]))
    for k in range(len(insts)):
        ms_ok &= batch.current_makespan[k] == alone[k].current_makespan[0]
    print(f'  field mismatches={sorted(set(feat_bad))} dummy-row violations={dummy_bad} '
          f'max |pi_batch - pi_alone|={pi_err:.2e}')
    check(not feat_bad, 'T7a real-row state identical alone and in the padded batch')
    check(dummy_bad == 0, 'T7b dummy rows are zero with both edges masked')
    check(pi_err < 1e-6, 'T7c policy output (A3 network) alone and in the batch agree within 1e-6',
          f'max abs diff={pi_err:.2e}')
    check(ms_ok, 'T7d makespans identical alone and in the batch')

    # for the record: the original normalization (isolate_padding=False)
    # lets the padding change the features of the shorter instances
    orig = FJSPEnvForVariousOpNums(J, M, use_lag_features=True, isolate_padding=False)
    so = init_env(orig, insts)
    k = int(np.argmin(n_ops))
    diff = float(np.abs(so.fea_j_tensor.numpy()[k, :n_ops[k]]
                        - init_env(FJSPEnvForVariousOpNums(J, M, use_lag_features=True),
                                   [insts[k]]).fea_j_tensor.numpy()[0]).max())
    print(f'  info: with isolate_padding=False the shortest instance\'s initial features '
          f'differ from its solo run by up to {diff:.3f}')


# ---------------------------------------------------------------------------
def t8_cpsat():
    print('=' * 72)
    print('T8 CP-SAT on a 5-module variable instance')
    print('=' * 72)
    from ortools_solver import matrix_to_the_format_for_solving, fjsp_solver
    # default factory (the paper's) and the one-station-per-type factory,
    # where machine contention pushes the optimum above the route bound
    for fac_name, fac in (('default', DEFAULT_FACTORY), ('small', SMALL_FACTORY)):
        jl, pt, meta = ppvc_instance_generator(5, 'mixed', fac, seed=42,
                                               route_variant='variable')
        lag = meta['time_lag']
        print(f'  [{fac_name}] module lengths={jl.tolist()} ops={pt.shape[0]} '
              f'machines={pt.shape[1]}')
        jobs, n_m = matrix_to_the_format_for_solving(jl, pt)
        obj, t, status, amch, oct_ = fjsp_solver(jobs, n_m, time_limits=300, time_lag=lag,
                                                 return_schedule=True)
        res = validate_schedule(jl, pt, lag, amch, oct_)
        env, amch_spt, _, _ = spt_rollout_record((jl, pt, meta), seed=0)
        print(f'  [{fac_name}] CP-SAT status={status} makespan={obj} time={t:.1f}s  validator: '
              f'feasible={res["feasible"]} violations={len(res["violations"])} '
              f'makespan={res["makespan"]} route lower bound={res["lower_bound"]}  '
              f'SPT env makespan={env.current_makespan[0]}')
        check([len(j) for j in jobs] == jl.tolist(),
              f'T8a [{fac_name}] solver input keeps the per-module op counts')
        check(status == 'OPTIMAL', f'T8b [{fac_name}] CP-SAT status OPTIMAL', status)
        check(res['feasible'] and not res['violations'] and res['makespan'] == obj,
              f'T8c [{fac_name}] CP-SAT schedule validator-clean, validator makespan == objective')
        check(obj <= env.current_makespan[0], f'T8d [{fac_name}] optimum <= SPT makespan')


TESTS = {'T1': t1_equivalence, 'T2': t2_serial_oracle,
         'T3': lambda: t3_t4_validator_and_bound(True, False),
         'T4': lambda: t3_t4_validator_and_bound(False, True),
         'T5': t5_daniel_regression, 'T6': t6_train_smoke,
         'T7': t7_padding_isolation, 'T8': t8_cpsat}


def main():
    names = SELECTED or list(TESTS)
    if 'T3' in names and 'T4' in names:
        names = [n for n in names if n != 'T4']
        TESTS['T3'] = t3_t4_validator_and_bound
    for n in names:
        TESTS[n]()
    print('=' * 72)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f'{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
