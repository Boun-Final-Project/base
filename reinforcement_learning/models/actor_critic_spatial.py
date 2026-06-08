"""
Dual-stream spatial CNN actor-critic with recurrent (GRU) temporal reasoning.

Input:
    spatial : (B, 4, 221, 221) — channels: [occupancy, gas, recency, wind_gradient]

CNN output geometry (221×221 input, 0.2 m/cell):
    stride-2 conv ×4 → 111 → 56 → 28 → 14 per spatial dim
    Static stream  (1 ch)  → 28×28×16
    Dynamic stream (3 ch)  → 28×28×32
    1×1 fusion             → 28×28×48
    stride-2 down_conv     → 14×14×48 → flatten → 9408 → proj → 512 → shared MLP → GRU
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .. import config as cfg
from .actor_critic import layer_init, ResidualBlock


class ActorCriticSpatial(nn.Module):

    ACTION_DIM = 2  # (cos θ, sin θ)

    def __init__(
        self,
        cnn_out_channels=cfg.SPATIAL_CNN_OUT_CH,
        proj_dim=cfg.SPATIAL_PROJ_DIM,
        shared_hidden=cfg.SPATIAL_SHARED_HIDDEN,
        actor_head_dim=cfg.SPATIAL_ACTOR_DIM,
        critic_head_dim=cfg.SPATIAL_CRITIC_DIM,
        num_res_blocks=cfg.SPATIAL_RES_BLOCKS,
        gru_hidden=cfg.SPATIAL_GRU_HIDDEN,
    ):
        super().__init__()

        def _conv_out(s, k, stride, pad):
            return (s + 2 * pad - k) // stride + 1
        _h = cfg.SPATIAL_GRID_SIZE
        _h = _conv_out(_h, 5, 2, 2)
        _h = _conv_out(_h, 3, 2, 1)
        _h = _conv_out(_h, 3, 2, 1)
        _h = _conv_out(_h, 3, 2, 1)            # 4th stride-2 conv after fusion
        cnn_flat = _h * _h * cnn_out_channels  # 14×14×48 = 9408 with default config

        # Static stream: occupancy (channel 0)
        self.static_cnn = nn.Sequential(
            layer_init(nn.Conv2d(1,  8,  5, stride=2, padding=2)), nn.ReLU(),  # 111×111
            layer_init(nn.Conv2d(8,  16, 3, stride=2, padding=1)), nn.ReLU(),  #  56×56
            layer_init(nn.Conv2d(16, 16, 3, stride=2, padding=1)), nn.ReLU(),  #  28×28
        )

        # Dynamic stream: gas + recency + wind_gradient (channels 1-3)
        self.dynamic_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3,  8,  5, stride=2, padding=2)), nn.ReLU(),  # 111×111
            layer_init(nn.Conv2d(8,  16, 3, stride=2, padding=1)), nn.ReLU(),  #  56×56
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),  #  28×28
        )

        # 1×1 channel-wise fusion: 48 → cnn_out_channels
        self.fusion_conv = nn.Sequential(
            layer_init(nn.Conv2d(48, cnn_out_channels, 1)), nn.ReLU(),
        )

        # 4th stride-2 conv: 28×28 → 14×14, applied to fused features
        self.down_conv = nn.Sequential(
            layer_init(nn.Conv2d(cnn_out_channels, cnn_out_channels, 3,
                                 stride=2, padding=1)), nn.ReLU(),
        )

        # CNN flat projection
        self.cnn_proj = nn.Sequential(
            layer_init(nn.Linear(cnn_flat, proj_dim)),
            nn.Tanh(),
        )

        # Shared MLP
        layers = []
        in_dim = proj_dim
        for out_dim in shared_hidden:
            layers += [layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh()]
            in_dim = out_dim
        self.shared_mlp = nn.Sequential(*layers)
        shared_out = shared_hidden[-1]

        # Recurrent layer over the shared MLP output (temporal context across steps)
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(
            input_size=shared_out,
            hidden_size=gru_hidden,
            batch_first=True,
        )

        # Actor head: projection → ResBlocks → Beta params
        self.actor_proj = nn.Sequential(
            layer_init(nn.Linear(gru_hidden, actor_head_dim)),
            nn.LayerNorm(actor_head_dim),
            nn.ReLU(),
        )
        self.actor_res = nn.Sequential(
            *[ResidualBlock(actor_head_dim) for _ in range(num_res_blocks)]
        )
        # 4 outputs: mean_cos, mean_sin, log_std_cos, log_std_sin
        self.actor_out = layer_init(nn.Linear(actor_head_dim, 4), std=0.01)

        # Critic head: projection → ResBlocks → V(s)
        self.critic_proj = nn.Sequential(
            layer_init(nn.Linear(gru_hidden, critic_head_dim)),
            nn.LayerNorm(critic_head_dim),
            nn.ReLU(),
        )
        self.critic_res = nn.Sequential(
            *[ResidualBlock(critic_head_dim) for _ in range(num_res_blocks)]
        )
        self.critic_out = layer_init(nn.Linear(critic_head_dim, 1), std=1.0)

    def initial_hidden(self, batch_size, device):
        """Return zero GRU hidden state of shape (1, batch_size, gru_hidden)."""
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _encode_pre_gru(self, spatial):
        """CNN + shared MLP, returns (B, shared_out)."""
        static  = self.static_cnn(spatial[:, 0:1])                         # (B, 16, 28, 28)
        dynamic = self.dynamic_cnn(spatial[:, 1:4])                         # (B, 32, 28, 28)
        fused   = self.fusion_conv(torch.cat([static, dynamic], dim=1))     # (B, 48, 28, 28)
        down    = self.down_conv(fused)                                     # (B, 48, 14, 14)
        flat    = self.cnn_proj(down.flatten(1))                            # (B, proj_dim)
        return self.shared_mlp(flat)                                        # (B, shared_out)

    def _step_gru(self, pre, h):
        """Single-step GRU forward. pre: (B, D). h: (1, B, gru_hidden).

        Returns
        -------
        out : (B, gru_hidden) — GRU output for this step
        h   : (1, B, gru_hidden) — updated hidden state
        """
        out, h = self.gru(pre.unsqueeze(1), h)   # out: (B, 1, gru_hidden)
        return out.squeeze(1), h

    def _encode(self, spatial, h=None):
        """One-step encode. Returns (features, new_h)."""
        if h is None:
            h = self.initial_hidden(spatial.shape[0], spatial.device)
        pre = self._encode_pre_gru(spatial)
        return self._step_gru(pre, h)

    def encode_sequence(self, spatial_seq, h0, dones_seq=None):
        """Run the encoder over a time sequence, resetting h at dones.

        Parameters
        ----------
        spatial_seq : (T, B, 4, H, W) tensor
        h0          : (1, B, gru_hidden) initial hidden state
        dones_seq   : (T, B) — if provided, h is zeroed *after* step t when
                      dones_seq[t] is True (so the next step starts fresh)

        Returns
        -------
        features    : (T, B, gru_hidden)
        h_T         : (1, B, gru_hidden)
        """
        T, B = spatial_seq.shape[0], spatial_seq.shape[1]
        flat_obs = spatial_seq.reshape(T * B, *spatial_seq.shape[2:])
        pre      = self._encode_pre_gru(flat_obs).reshape(T, B, -1)

        if dones_seq is None:
            # Single GRU call, no mid-sequence resets
            out, h_T = self.gru(pre.transpose(0, 1), h0)         # out: (B, T, H)
            return out.transpose(0, 1).contiguous(), h_T

        # Step-by-step so we can mask h at done boundaries
        h     = h0
        feats = []
        for t in range(T):
            out, h = self.gru(pre[t].unsqueeze(1), h)             # out: (B, 1, H)
            feats.append(out.squeeze(1))
            # Reset hidden state for envs whose episode ended at step t
            mask = (1.0 - dones_seq[t]).view(1, B, 1)
            h    = h * mask
        return torch.stack(feats, dim=0), h

    def _critic_value(self, shared):
        return self.critic_out(self.critic_res(self.critic_proj(shared)))

    def get_value(self, spatial, h=None):
        shared, h = self._encode(spatial, h)
        return self._critic_value(shared), h

    def _actor_dist(self, shared):
        out     = self.actor_out(self.actor_res(self.actor_proj(shared)))  # (B, 4)
        mean    = out[:, :2]
        log_std = out[:, 2:].clamp(-5, 0.5)
        return Normal(mean, log_std.exp())

    def get_action_and_value(self, spatial, h=None, action=None):
        shared, h_new = self._encode(spatial, h)
        dist          = self._actor_dist(shared)

        if action is None:
            action = dist.rsample()  # (B, 2): (cos θ, sin θ)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self._critic_value(shared)

        return action, log_prob, entropy, value, h_new

    def evaluate_sequence(self, spatial_seq, h0, actions, dones_seq=None):
        """Re-evaluate log_prob/entropy/value over a time sequence (PPO update path).

        Parameters
        ----------
        spatial_seq : (T, B, 4, H, W)
        h0          : (1, B, gru_hidden) — detached
        actions     : (T, B, 2)
        dones_seq   : (T, B) — see encode_sequence

        Returns
        -------
        log_prob : (T, B)
        entropy  : (T, B)
        value    : (T, B)
        """
        feats, _ = self.encode_sequence(spatial_seq, h0, dones_seq)
        T, B, _  = feats.shape
        flat     = feats.reshape(T * B, -1)
        dist     = self._actor_dist(flat)
        flat_a   = actions.reshape(T * B, -1)
        log_prob = dist.log_prob(flat_a).sum(dim=-1).reshape(T, B)
        entropy  = dist.entropy().sum(dim=-1).reshape(T, B)
        value    = self._critic_value(flat).squeeze(-1).reshape(T, B)
        return log_prob, entropy, value
