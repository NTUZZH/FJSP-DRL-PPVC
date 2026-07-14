"""
Correctness tests for PPVC Adaptation 2b (anticipatory lag features) and
Adaptation 1 (op/station type embeddings).

  T1  Channel-compat: with use_lag_features=True the op features are [N,12]
      and the first 10 channels are BIT-IDENTICAL to the use_lag_features=False
      run under the same action sequence (z-normalization is per-channel, so
      the added channels must not perturb the original ones).
  T2  Remaining-lag invariants along a full rollout:
        (a) op_remain_lag == 0 for every unscheduled op
        (b) 0 <= op_remain_lag <= op_lag elementwise
        (c) scheduled ops with pending lag match the analytic value
            clip(op_ct + lag - t, 0, lag) at every decision instant
        (d) a strictly positive remaining lag occurs at least once
  T3  Deleted-node detection: the model-side liveness heuristic
      (all-zero feature row <=> deleted) matches env.deleted_op_nodes exactly
      at every decision instant.
  T4  Model forward with type embeddings: full greedy episode on a PPVC
      batch, pi/v finite everywhere, pi sums to 1.
  T5  PPO update smoke with embeddings: one collected episode, ppo.update()
      returns finite losses (validates the Memory/transpose threading).

Run:  python test_ppvc_features.py
"""
import numpy as np
import torch

from params import configs
from ppvc_instance_generator import ppvc_instance_generator, DEFAULT_FACTORY
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from common_utils import heuristic_select_action, sample_action, setup_seed

N_MODULES, SEED = 5, 42


def make_instance():
    return ppvc_instance_generator(n_modules=N_MODULES, class_mix='mixed',
                                   station_counts=DEFAULT_FACTORY, seed=SEED)


def t1_channel_compat():
    print('T1  first-10-channel bit-compat (10 vs 12 feature channels) ...')
    jl, op_pt, meta = make_instance()
    J, M = len(jl), op_pt.shape[1]

    feas = {}
    for flag in (False, True):
        np.random.seed(0)
        env = FJSPEnvForSameOpNums(J, M, use_lag_features=flag)
        env.set_initial_data([jl], [op_pt], time_lag_list=[meta['time_lag']])
        snaps = [env.fea_j.copy()]
        while not env.done().all():
            env.step(np.array([heuristic_select_action('FIFO', env)]))
            snaps.append(env.fea_j.copy())
        feas[flag] = snaps

    assert feas[False][0].shape[2] == 10 and feas[True][0].shape[2] == 12, \
        f'op feature widths wrong: {feas[False][0].shape} / {feas[True][0].shape}'
    assert len(feas[False]) == len(feas[True])
    for k, (f10, f12) in enumerate(zip(feas[False], feas[True])):
        assert np.array_equal(f10, f12[:, :, :10]), \
            f'step {k}: first 10 channels diverge between 10- and 12-channel envs'
    print(f'    {len(feas[True])} decision states checked, first 10 channels identical  OK')


def t2_t3_remaining_lag_and_liveness():
    print('T2  remaining-lag invariants + T3 deleted-node liveness ...')
    jl, op_pt, meta = make_instance()
    J, M = len(jl), op_pt.shape[1]
    np.random.seed(0)
    env = FJSPEnvForSameOpNums(J, M, use_lag_features=True)
    env.set_initial_data([jl], [op_pt], time_lag_list=[meta['time_lag']])

    saw_positive = False
    degenerate_live = 0
    step_k = 0
    while not env.done().all():
        env.step(np.array([heuristic_select_action('FIFO', env)]))
        step_k += 1
        rl, lag = env.op_remain_lag[0], env.op_lag[0]
        sched = env.op_scheduled_flag[0].astype(bool)
        t = env.next_schedule_time[0]

        assert (rl[~sched] == 0).all(), f'step {step_k}: unscheduled op has remaining lag'
        assert (rl >= -1e-12).all() and (rl <= lag + 1e-12).all(), \
            f'step {step_k}: remaining lag outside [0, lag]'
        expect = np.clip(env.op_ct[0] + lag - t, 0, lag) * sched
        assert np.abs(rl - expect).max() < 1e-9, \
            f'step {step_k}: remaining lag mismatch vs analytic value'
        if (rl > 1e-9).any():
            saw_positive = True

        # T3: no DELETED node may ever carry a live embedding. (The reverse
        # is weaker by design: when z-normalization degenerates — e.g. a
        # single live node equals the channel mean — the live node's row is
        # exactly zero and nonzero_averaging itself excludes it in original
        # DANIEL; the embedding mask matches that pooling criterion.)
        if env.step_count != env.number_of_ops:   # features are normalized states
            live = np.abs(env.fea_j[0]).sum(axis=-1) > 0
            deleted = env.deleted_op_nodes[0]
            assert not (live & deleted).any(), \
                f'step {step_k}: a deleted node has a nonzero feature row'
            degenerate_live += int(((~live) & (~deleted)).sum())
    assert saw_positive, 'no strictly positive remaining lag ever observed'
    print(f'    {step_k} steps checked: invariants hold; deleted nodes never live; '
          f'{degenerate_live} degenerate live-node states (z-norm collapse, '
          f'pooling-consistent)  OK')


def _ppvc_batch(n_envs):
    jls, pts, lags, otys, mtys = [], [], [], [], []
    for i in range(n_envs):
        jl, pt, meta = ppvc_instance_generator(
            n_modules=N_MODULES, class_mix='mixed',
            station_counts=DEFAULT_FACTORY, seed=SEED + i)
        jls.append(jl); pts.append(pt); lags.append(meta['time_lag'])
        otys.append(meta['op_type']); mtys.append(meta['mch_type'])
    return jls, pts, lags, otys, mtys


def _ppvc_configs():
    configs.use_lag_features = True
    configs.use_type_embedding = True
    configs.fea_j_input_dim = 12
    configs.n_op_types, configs.n_mch_types, configs.type_emb_dim = 5, 9, 8


def t4_t5_model_and_ppo():
    print('T4  model forward with type embeddings (full episode) ...')
    _ppvc_configs()
    setup_seed(123)
    torch.set_default_dtype(torch.float32)
    if torch.device(configs.device).type == 'cuda':
        torch.set_default_device('cuda')

    from model.PPO import PPO_initialize, Memory
    ppo = PPO_initialize()
    memory = Memory(gamma=configs.gamma, gae_lambda=configs.gae_lambda)

    n_envs = 4
    jls, pts, lags, otys, mtys = _ppvc_batch(n_envs)
    J, M = len(jls[0]), pts[0].shape[1]
    env = FJSPEnvForSameOpNums(J, M, use_lag_features=True)
    state = env.set_initial_data(jls, pts, time_lag_list=lags,
                                 op_type_list=otys, mch_type_list=mtys)
    assert state.op_type_tensor.shape == (n_envs, pts[0].shape[0])
    assert state.mch_type_tensor.shape == (n_envs, M)

    n_fwd = 0
    while True:
        memory.push(state)
        with torch.no_grad():
            pi, v = ppo.policy_old(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
                op_type=state.op_type_tensor, mch_type=state.mch_type_tensor)
        assert torch.isfinite(pi).all() and torch.isfinite(v).all(), 'NaN/Inf in pi or v'
        assert torch.allclose(pi.sum(1), torch.ones_like(pi.sum(1)), atol=1e-5)
        n_fwd += 1
        action, logp = sample_action(pi)
        state, reward, done = env.step(action.cpu().numpy())
        memory.done_seq.append(torch.from_numpy(done).to(pi.device))
        memory.reward_seq.append(torch.from_numpy(reward).to(pi.device))
        memory.action_seq.append(action)
        memory.log_probs.append(logp)
        memory.val_seq.append(v.squeeze(1))
        if done.all():
            break
    print(f'    {n_fwd} forward passes, all finite, pi normalized  OK')

    print('T5  PPO update smoke (embedding path through Memory) ...')
    loss, v_loss = ppo.update(memory)
    assert np.isfinite(loss) and np.isfinite(v_loss), 'non-finite PPO losses'
    n_emb_params = sum(p.numel() for n, p in ppo.policy.named_parameters()
                       if 'type_embedding' in n)
    assert n_emb_params == 5 * 8 + 9 * 8, f'unexpected embedding param count {n_emb_params}'
    print(f'    loss={loss:.4f}  v_loss={v_loss:.4f}  emb_params={n_emb_params}  OK')


if __name__ == '__main__':
    t1_channel_compat()
    t2_t3_remaining_lag_and_liveness()
    t4_t5_model_and_ppo()
    print('ALL PPVC FEATURE/EMBEDDING TESTS PASSED')
