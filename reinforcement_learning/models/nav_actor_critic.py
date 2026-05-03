"""
NavActorCritic: actor-critic for wall-following navigation.

Encodes 72-ray LiDAR with circular 1D convolutions, lifts the 5-dim
context (pos + goal direction + goal distance) through a small MLP,
fuses both into a shared backbone, and produces:
  - Actor:  2D Gaussian over (cos theta, sin theta) -- 4 outputs
  - Critic: scalar V(s)

The 2D Gaussian avoids the wrapping discontinuity of a Beta distribution
over a raw angle. At inference, heading = atan2(sin theta, cos theta).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .. import config as cfg
from .actor_critic import CircularPad1d, layer_init


class NavActorCritic(nn.Module):
    """
    Observation layout (obs_dim = cfg.LIDAR_NUM_RAYS + 5 = 77):
        obs[:72]    LiDAR (72 rays, normalized [0, 1])
        obs[72:74]  pos   (x/w, y/h)
        obs[74:76]  goal_dir (cos theta_goal, sin theta_goal)
        obs[76]     goal_dist (dist / diagonal)

    Action (2,): (cos theta, sin theta) -- 2D Normal distribution.
    """

    LIDAR_DIM = cfg.LIDAR_NUM_RAYS
    CONTEXT_DIM = 5   # pos(2) + goal_dir(2) + goal_dist(1)

    def __init__(self,
                 lidar_dim: int = cfg.LIDAR_NUM_RAYS,
                 context_dim: int = 5,
                 conv_channels: int = cfg.LIDAR_CONV_CHANNELS,
                 conv_kernel: int = cfg.LIDAR_CONV_KERNEL,
                 hidden_dim: int = 256,
                 actor_head_dim: int = 128,
                 critic_head_dim: int = 128):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.context_dim = context_dim

        # LiDAR encoder -- circular padding preserves wrap-around topology
        pad = conv_kernel // 2
        self.lidar_enc = nn.Sequential(
            CircularPad1d(pad),
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
            CircularPad1d(pad),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
        )
        lidar_out_dim = conv_channels * lidar_dim  # circular padding preserves length

        # Context encoder: lift 5-dim context to 32-dim
        self.ctx_enc = nn.Sequential(
            layer_init(nn.Linear(context_dim, 32)),
            nn.ReLU(),
        )
        ctx_out_dim = 32

        # Shared fusion backbone
        fusion_in = lidar_out_dim + ctx_out_dim
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_in, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Actor head: mean_cos, mean_sin, log_std_cos, log_std_sin
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, actor_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_head_dim, 4), std=0.01),
        )

        # Critic head: scalar V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, critic_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_head_dim, 1), std=1.0),
        )

    # ------------------------------------------------------------------

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        lidar = obs[:, :self.lidar_dim]      # (B, 72)
        context = obs[:, self.lidar_dim:]    # (B, 5)

        lidar_feat = self.lidar_enc(lidar.unsqueeze(1)).flatten(1)  # (B, 144)
        ctx_feat = self.ctx_enc(context)                             # (B, 32)

        return self.fusion(torch.cat([lidar_feat, ctx_feat], dim=1))  # (B, 256)

    def _actor_dist(self, features: torch.Tensor) -> Normal:
        out = self.actor(features)           # (B, 4)
        mean = out[:, :2]                   # (B, 2)
        log_std = out[:, 2:].clamp(-5, 0.5)
        return Normal(mean, log_std.exp())

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._encode(obs))

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Parameters
        ----------
        obs    : (B, 77)
        action : (B, 2) optional -- if provided, evaluate log_prob of that action

        Returns
        -------
        action    : (B, 2)
        log_prob  : (B,)
        entropy   : (B,)
        value     : (B, 1)
        """
        features = self._encode(obs)
        dist = self._actor_dist(features)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features)

        return action, log_prob, entropy, value
