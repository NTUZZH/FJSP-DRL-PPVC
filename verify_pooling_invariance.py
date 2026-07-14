"""
verify_pooling_invariance.py  (Proposition 2 numerical check)

Loads the trained A3 (full, type-embedding) model and, on real PPVC test
instances, replays FIFO rollouts. At EVERY decision state it reproduces the
gated type-embedding concatenation exactly as DANIEL.forward does, runs the
dual-attention stack (model.feature_exact), and measures the maximum absolute
post-attention activation over DELETED operation rows. It also checks that the
node set selected by nonzero_averaging excludes every deleted node.

If deleted rows stay exactly zero through the attention stack, then DANIEL's
post-attention nonzero_averaging is equivalent to mask-based pooling, and the
liveness-gated embedding (which leaves op_mask untouched) preserves the pooled
set exactly -- Proposition 2.
"""
import os, sys, glob, json
import numpy as np
import torch

sys.argv = [sys.argv[0]]  # keep params.py argparse from choking on our argv

MODEL = '10x25+ppvc-mixed+full'
DATA = 'data/PPVC/10x25+ppvc-mixed'
N_INST = 20  # instances to sweep (every decision step of each)

# --- push arch flags into configs BEFORE PPO is built (mirrors eval_ppvc) ---
from params import configs
snap = json.load(open(f'train_log/PPVC/config_{MODEL}.json'))
for k in ('fea_j_input_dim', 'fea_m_input_dim', 'num_heads_OAB', 'num_heads_MAB',
          'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
          'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob',
          'type_emb_dim', 'n_op_types', 'n_mch_types',
          'use_type_embedding', 'use_lag_features'):
    if k in snap:
        setattr(configs, k, snap[k])
configs.device = 'cpu'

from model.PPO import PPO_initialize
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from common_utils import heuristic_select_action, nonzero_averaging
from ppvc_instance_generator import load_instance

ppo = PPO_initialize()
ppo.policy.load_state_dict(torch.load(f'trained_network/PPVC/{MODEL}.pth', map_location='cpu'))
ppo.policy.eval()
m = ppo.policy

global_max_deleted = 0.0
total_steps = 0
gate_mismatches = 0
pool_violations = 0
# machine-side (stations are zeroed when they have no remaining processable ops)
global_max_deleted_m = 0.0
n_deleted_m_states = 0
pool_violations_m = 0

stems = sorted(glob.glob(os.path.join(DATA, 'instance_*.fjs')))[:N_INST]
for stem in stems:
    jl, pt, meta = load_instance(stem[:-4])
    env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1], use_lag_features=True)
    state = env.set_initial_data([jl], [pt], time_lag_list=[meta['time_lag']],
                                 op_type_list=[meta['op_type']],
                                 mch_type_list=[meta['mch_type']])
    done = False
    while not done:
        with torch.no_grad():
            fea_j = state.fea_j_tensor          # [1, N, 12]
            fea_m = state.fea_m_tensor
            op_type = state.op_type_tensor
            mch_type = state.mch_type_tensor
            # --- exact replica of DANIEL.forward gating ---
            live_j = (fea_j.abs().sum(-1, keepdim=True) > 0).float()
            live_m = (fea_m.abs().sum(-1, keepdim=True) > 0).float()
            fj_aug = torch.cat((fea_j, m.op_type_embedding(op_type) * live_j), dim=-1)
            fm_aug = torch.cat((fea_m, m.mch_type_embedding(mch_type) * live_m), dim=-1)
            fj_out, fm_out, fjg, fmg = m.feature_exact(
                fj_aug, state.op_mask_tensor, state.candidate_tensor,
                fm_aug, state.mch_mask_tensor, state.comp_idx_tensor)

            deleted = torch.from_numpy(env.deleted_op_nodes.astype(bool))[0]  # [N]
            live_in = (live_j[0, :, 0] > 0)                                   # [N]
            # (1) gate criterion exactly matches env deletion?
            gate_mismatches += int((live_in == deleted).sum().item())  # live_in should == ~deleted
            # (2) max |post-attention row| over deleted nodes
            if deleted.any():
                md = fj_out[0][deleted].abs().max().item()
                global_max_deleted = max(global_max_deleted, md)
                # (3) nonzero_averaging must NOT count any deleted node
                postrow_nonzero = (fj_out[0].abs().sum(-1) > 0)
                pool_violations += int((postrow_nonzero & deleted).sum().item())
            # --- machine pool: stations with an all-zero input row (gated/deleted) ---
            deleted_m = (live_m[0, :, 0] == 0)
            if deleted_m.any():
                n_deleted_m_states += 1
                global_max_deleted_m = max(global_max_deleted_m,
                                           fm_out[0][deleted_m].abs().max().item())
                postrow_nonzero_m = (fm_out[0].abs().sum(-1) > 0)
                pool_violations_m += int((postrow_nonzero_m & deleted_m).sum().item())
            total_steps += 1

        action = heuristic_select_action('FIFO', env)
        state, _, d = env.step(np.array([action]))
        done = bool(np.all(d)) if hasattr(d, '__len__') else bool(d)

print(f'instances swept           : {len(stems)}')
print(f'decision states checked   : {total_steps}')
print(f'gate==deletion mismatches : {gate_mismatches}   (0 = live_j exactly matches deleted set)')
print(f'pooling violations        : {pool_violations}   (deleted nodes counted by nonzero_averaging)')
print(f'MAX |deleted post-attn row|: {global_max_deleted:.3e}   (target: 0)')
print(f'--- machine pool ---')
print(f'states with deleted station: {n_deleted_m_states}/{total_steps}')
print(f'machine pooling violations : {pool_violations_m}   (target: 0)')
print(f'MAX |deleted station post-attn|: {global_max_deleted_m:.3e}   (target: 0)')
