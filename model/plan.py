"""
PLAN policy backbone: a parallel attention + liquid (Euler-step) encoder in
place of DANIEL's dual attention network, with DANIEL's actor and critic heads.

Ported from the PLAN fork of DANIEL (class ``AttL`` in that fork's
model/PLAN.py). The layers, dimensions, attention heads, context injection and
pooling are unchanged, and the state-dict key names are preserved, so the
checkpoints of the fork load here with ``strict=True`` when the PPVC flags
are off. Two additions mirror DANIEL in model/main_model.py:

  * the forward signature takes the optional ``op_type`` / ``mch_type``
    tensors of the PPVC environment;
  * with ``use_type_embedding`` on, learned type embeddings are concatenated
    onto the raw op and machine features before the encoder, with the same
    liveness mask as DANIEL (deleted zero-row nodes keep zero embeddings).

Input widths come from ``config.fea_j_input_dim`` / ``config.fea_m_input_dim``
(plus ``config.type_emb_dim`` when the embeddings are on), so the lag feature
channels (``--use_lag_features``) are picked up through the config as for
DANIEL.

Per-backbone detail: the context injection ``fea + fea.mean(dim=1)`` averages
over all rows, deleted zero rows included, and turns those rows nonzero before
the encoder. DANIEL's pooling (``nonzero_averaging``) excludes deleted rows.
This behaviour is kept as in the original PLAN code.
"""
from common_utils import nonzero_averaging
from model.sub_layers import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedParallelLiquidNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # liquid components
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.dt = 1.0

        # attention across nodes, computed in parallel
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )

        # state update network
        self.state_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        """
        :param x: node features with shape [B, T, D]
        :return: node embeddings with shape [B, T, H]
        """
        Ux = self.U(x)                                   # [B, T, H]
        attn_out, _ = self.attention(Ux, Ux, Ux)          # [B, T, H]
        combined = torch.cat([Ux, attn_out], dim=-1)      # [B, T, 2H]
        estimated_h = self.state_net(combined)            # [B, T, H]

        # one explicit Euler step of the liquid dynamics
        Wh = self.W(estimated_h)
        dhdt = torch.tanh(Wh + Ux)
        h = estimated_h + self.dt * (dhdt - estimated_h) / (self.tau.view(1, 1, -1) + 1e-6)
        return h


class AttnLiquidFE(nn.Module):
    def __init__(self, config, fea_j_input_dim=None, fea_m_input_dim=None):
        """
        :param config: a package of parameters
        :param fea_j_input_dim: effective op feature width, overrides
                config.fea_j_input_dim (used when type embeddings are
                concatenated onto the raw features)
        :param fea_m_input_dim: effective machine feature width
        """
        super(AttnLiquidFE, self).__init__()

        self.fea_j_input_dim = fea_j_input_dim if fea_j_input_dim is not None \
            else config.fea_j_input_dim
        self.fea_m_input_dim = fea_m_input_dim if fea_m_input_dim is not None \
            else config.fea_m_input_dim
        self.output_dim = config.layer_fea_output_dim[-1]

        self.job_liquid = LearnedParallelLiquidNN(
            self.fea_j_input_dim, self.output_dim, num_heads=4
        )
        self.mch_liquid = LearnedParallelLiquidNN(
            self.fea_m_input_dim, self.output_dim, num_heads=4
        )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        # context injection (mean over all rows, deleted zero rows included)
        job_context = fea_j.mean(dim=1, keepdim=True)
        mch_context = fea_m.mean(dim=1, keepdim=True)

        fea_j = fea_j + job_context
        fea_m = fea_m + mch_context

        fea_j_out = self.job_liquid(fea_j)
        fea_m_out = self.mch_liquid(fea_m)

        # global pooling
        fea_j_global = nonzero_averaging(fea_j_out)
        fea_m_global = nonzero_averaging(fea_m_out)

        return fea_j_out, fea_m_out, fea_j_global, fea_m_global


class PLAN(nn.Module):
    def __init__(self, config):
        """
            PLAN encoder with DANIEL's actor and critic heads
        :param config: a package of parameters
        """
        super(PLAN, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8
        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        # (PPVC, Adaptation 1) learned type embeddings, as in DANIEL
        self.use_type_embedding = getattr(config, 'use_type_embedding', False)
        fea_j_input_dim = config.fea_j_input_dim
        fea_m_input_dim = config.fea_m_input_dim
        if self.use_type_embedding:
            self.type_emb_dim = config.type_emb_dim
            self.op_type_embedding = nn.Embedding(config.n_op_types,
                                                  config.type_emb_dim).to(device)
            self.mch_type_embedding = nn.Embedding(config.n_mch_types,
                                                   config.type_emb_dim).to(device)
            fea_j_input_dim += config.type_emb_dim
            fea_m_input_dim += config.type_emb_dim

        self.feature_exact = AttnLiquidFE(
            config, fea_j_input_dim, fea_m_input_dim).to(device)

        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1
        ).to(device)

        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1
        ).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                op_type=None, mch_type=None):
        """
            same inputs and outputs as DANIEL.forward
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        if self.use_type_embedding:
            assert op_type is not None and mch_type is not None, \
                'use_type_embedding=True requires op_type/mch_type in the state'
            # same liveness mask as DANIEL: deleted zero-row nodes get zero
            # embeddings, using the zero-row criterion of nonzero_averaging
            live_j = (fea_j.abs().sum(dim=-1, keepdim=True) > 0).float()
            live_m = (fea_m.abs().sum(dim=-1, keepdim=True) > 0).float()
            fea_j = torch.cat((fea_j, self.op_type_embedding(op_type) * live_j), dim=-1)
            fea_m = torch.cat((fea_m, self.mch_type_embedding(mch_type) * live_m), dim=-1)

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).long()
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)

        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((
            Fea_j_JC_serialized,
            Fea_m_serialized,
            Fea_Gj_input,
            Fea_Gm_input,
            fea_pairs
        ), dim=-1)

        candidate_scores = self.actor(candidate_feature).squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores = torch.where(
            dynamic_pair_mask.reshape(sz_b, -1),
            torch.tensor(float('-inf'), device=candidate_scores.device),
            candidate_scores
        )

        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)

        return pi, v
