"""
Dual-stream spatial CNN actor-critic with FiLM wind conditioning.

Matches the architecture in rl_new: 5-channel spatial input, 4-dim context
(wind + time), FiLM initialised to identity, state-independent actor_log_std.

Input:
    spatial : (B, 5, 98, 98) — [is_known, is_wall, gas, recency, det_count]
    ctx     : (B, 4)          — [speed/max_speed, cos(dir), sin(dir), step/MAX_STEPS]
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .. import config as cfg
from ..models.actor_critic import layer_init, ResidualBlock


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

        feat_hw = 13
        cnn_flat = feat_hw * feat_hw * cnn_out_channels  # 13×13×128 = 21632

        # Static stream: is_known + is_wall (channels 0-1)
        self.static_cnn = nn.Sequential(
            layer_init(nn.Conv2d(2,  16, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)), nn.ReLU(),
        )

        # Dynamic stream: gas + recency + det_count (channels 2-4)
        self.dynamic_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3,  16, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
        )

        self.fusion_conv = nn.Sequential(
            layer_init(nn.Conv2d(96, cnn_out_channels, 1)), nn.ReLU(),
        )

        # FiLM conditioning with identity init (γ=1, β=0 at step 0).
        self.film_net = nn.Sequential(
            layer_init(nn.Linear(4, film_hidden)),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * cnn_flat),
        )
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias[:cnn_flat].fill_(1.0)
            self.film_net[-1].bias[cnn_flat:].zero_()

        layers = []
        in_dim = cnn_flat
        for out_dim in shared_hidden:
            layers += [layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh()]
            in_dim = out_dim
        self.shared_mlp = nn.Sequential(*layers)
        shared_out = shared_hidden[-1]

        self.actor_proj = nn.Sequential(
            layer_init(nn.Linear(shared_out, actor_head_dim)),
            nn.LayerNorm(actor_head_dim),
            nn.ReLU(),
        )
        self.actor_res = nn.Sequential(
            *[ResidualBlock(actor_head_dim) for _ in range(num_res_blocks)]
        )
        # State-conditional mean only; log_std is a separate state-independent
        # parameter (standard PPO recipe).
        self.actor_out = layer_init(nn.Linear(actor_head_dim, 2), std=0.01)
        self.actor_log_std = nn.Parameter(torch.full((self.ACTION_DIM,), -1.0))

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
        static  = self.static_cnn(spatial[:, 0:2])
        dynamic = self.dynamic_cnn(spatial[:, 2:5])
        flat    = self.fusion_conv(torch.cat([static, dynamic], dim=1))
        flat    = flat.flatten(1)

        gamma_beta  = self.film_net(wind)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        flat        = gamma * flat + beta

        return self.shared_mlp(flat)

    def _critic_value(self, shared):
        return self.critic_out(self.critic_res(self.critic_proj(shared)))

    def get_value(self, spatial, wind):
        return self._critic_value(self._encode(spatial, wind))

    def _actor_dist(self, shared):
        mean    = self.actor_out(self.actor_res(self.actor_proj(shared)))
        log_std = self.actor_log_std.clamp(-3, 2).expand_as(mean)
        return Normal(mean, log_std.exp())

    def get_action_and_value(self, spatial, wind, action=None):
        shared = self._encode(spatial, wind)
        dist   = self._actor_dist(shared)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self._critic_value(shared)

        return action, log_prob, entropy, value
