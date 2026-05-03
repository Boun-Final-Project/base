"""
Dual-stream spatial CNN actor-critic with FiLM wind conditioning.

Input:
    spatial : (B, 4, 98, 98) — channels: [occupancy, gas, recency, det_count]
    wind    : (B, 2)          — [speed/max_speed, direction/2π]

CNN output geometry (98×98 input, 0.5 m/cell):
    stride-2 conv ×3 → 49 → 25 → 13 per spatial dim
    Static stream  (1 ch)  → 13×13×32
    Dynamic stream (3 ch)  → 13×13×64
    1×1 fusion             → 13×13×128  → flatten → 21632
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
        film_hidden=cfg.SPATIAL_FILM_HIDDEN,
        cnn_out_channels=cfg.SPATIAL_CNN_OUT_CH,
        shared_hidden=cfg.SPATIAL_SHARED_HIDDEN,
        actor_head_dim=cfg.SPATIAL_ACTOR_DIM,
        critic_head_dim=cfg.SPATIAL_CRITIC_DIM,
        num_res_blocks=cfg.SPATIAL_RES_BLOCKS,
    ):
        super().__init__()

        def _conv_out(s, k, stride, pad):
            return (s + 2 * pad - k) // stride + 1
        _h = _conv_out(_conv_out(_conv_out(cfg.SPATIAL_GRID_SIZE, 5, 2, 2), 3, 2, 1), 3, 2, 1)
        cnn_flat = _h * _h * cnn_out_channels  # 13×13×128 = 21632 with default config

        # Static stream: occupancy (channel 0)
        self.static_cnn = nn.Sequential(
            layer_init(nn.Conv2d(1,  16, 5, stride=2, padding=2)), nn.ReLU(),  # 49×49
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),  # 25×25
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)), nn.ReLU(),  # 13×13
        )

        # Dynamic stream: gas + recency + det_count (channels 1-3)
        self.dynamic_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3,  16, 5, stride=2, padding=2)), nn.ReLU(),  # 49×49
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),  # 25×25
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),  # 13×13
        )

        # 1×1 channel-wise fusion: 96 → cnn_out_channels
        self.fusion_conv = nn.Sequential(
            layer_init(nn.Conv2d(96, cnn_out_channels, 1)), nn.ReLU(),
        )

        # FiLM conditioning: wind → (γ, β) for cnn_flat-dim features
        self.film_net = nn.Sequential(
            layer_init(nn.Linear(2, film_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(film_hidden, 2 * cnn_flat)),
        )

        # Shared MLP
        layers = []
        in_dim = cnn_flat
        for out_dim in shared_hidden:
            layers += [layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh()]
            in_dim = out_dim
        self.shared_mlp = nn.Sequential(*layers)
        shared_out = shared_hidden[-1]

        # Actor head: projection → ResBlocks → Beta params
        self.actor_proj = nn.Sequential(
            layer_init(nn.Linear(shared_out, actor_head_dim)),
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
            layer_init(nn.Linear(shared_out, critic_head_dim)),
            nn.LayerNorm(critic_head_dim),
            nn.ReLU(),
        )
        self.critic_res = nn.Sequential(
            *[ResidualBlock(critic_head_dim) for _ in range(num_res_blocks)]
        )
        self.critic_out = layer_init(nn.Linear(critic_head_dim, 1), std=1.0)

    def _encode(self, spatial, wind):
        static  = self.static_cnn(spatial[:, 0:1])                        # (B, 32, 13, 13)
        dynamic = self.dynamic_cnn(spatial[:, 1:4])                        # (B, 64, 13, 13)
        flat    = self.fusion_conv(torch.cat([static, dynamic], dim=1))    # (B, C, 13, 13)
        flat    = flat.flatten(1)                                           # (B, cnn_flat)

        gamma_beta      = self.film_net(wind)
        gamma, beta     = gamma_beta.chunk(2, dim=-1)
        flat            = gamma * flat + beta

        return self.shared_mlp(flat)                                        # (B, shared_out[-1])

    def _critic_value(self, shared):
        return self.critic_out(self.critic_res(self.critic_proj(shared)))

    def get_value(self, spatial, wind):
        return self._critic_value(self._encode(spatial, wind))

    def _actor_dist(self, shared):
        out     = self.actor_out(self.actor_res(self.actor_proj(shared)))  # (B, 4)
        mean    = out[:, :2]
        log_std = out[:, 2:].clamp(-5, 0.5)
        return Normal(mean, log_std.exp())

    def get_action_and_value(self, spatial, wind, action=None):
        shared = self._encode(spatial, wind)
        dist   = self._actor_dist(shared)

        if action is None:
            action = dist.rsample()  # (B, 2): (cos θ, sin θ)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self._critic_value(shared)

        return action, log_prob, entropy, value
