"""
PPO Actor-Critic networks with Beta-distribution actor.

Two architectures:
  - ActorCritic:        flat MLP backbone (original)
  - ActorCriticModular: separate GRU (gas), 1D-conv (LiDAR), MLP (context) encoders
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal

from .. import config as cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (CleanRL convention)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """MLP actor-critic with shared backbone and Beta-distribution actor.

    Architecture:
        Observation (39)
            |
        Shared backbone [256, 256] (tanh)
            |
            +-- Actor head [128] --> (alpha, beta) --> Beta distribution --> angle
            |
            +-- Critic head [128] --> V(s) scalar
    """

    def __init__(self, obs_dim=cfg.STATE_DIM, hidden_dim=cfg.HIDDEN_DIM,
                 backbone_layers=cfg.BACKBONE_LAYERS,
                 actor_head_dim=cfg.ACTOR_HEAD_DIM,
                 critic_head_dim=cfg.CRITIC_HEAD_DIM):
        super().__init__()

        # Shared backbone
        layers = []
        in_dim = obs_dim
        for _ in range(backbone_layers):
            layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Actor head: outputs alpha and beta params for Beta distribution
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, actor_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_head_dim, 2), std=0.01),
            nn.Softplus(),  # alpha, beta must be > 0
        )

        # Critic head: outputs scalar V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, critic_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_head_dim, 1), std=1.0),
        )

    def get_value(self, obs):
        """Compute V(s) for a batch of observations.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch, obs_dim).

        Returns
        -------
        value : torch.Tensor
            Shape (batch, 1).
        """
        return self.critic(self.backbone(obs))

    def get_action_and_value(self, obs, action=None):
        """Sample action and compute log_prob, entropy, V(s).

        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch, obs_dim).
        action : torch.Tensor, optional
            Shape (batch, 1). If provided, compute log_prob of this action
            instead of sampling a new one.

        Returns
        -------
        action : torch.Tensor
            Shape (batch, 1), in [0, 1].
        log_prob : torch.Tensor
            Shape (batch,).
        entropy : torch.Tensor
            Shape (batch,).
        value : torch.Tensor
            Shape (batch, 1).
        """
        features = self.backbone(obs)

        # Actor: Beta distribution parameters
        ab = self.actor(features)  # (batch, 2)
        # Add 1 to ensure alpha, beta >= 1 (unimodal distribution)
        alpha = ab[:, 0:1] + 1.0
        beta = ab[:, 1:2] + 1.0
        dist = Beta(alpha, beta)

        if action is None:
            action = dist.rsample()  # (batch, 1)

        log_prob = dist.log_prob(action).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1)           # (batch,)
        value = self.critic(features)                  # (batch, 1)

        return action, log_prob, entropy, value


class CircularPad1d(nn.Module):
    """Circular padding for 1D convolutions (LiDAR rays wrap around)."""

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        return F.pad(x, (self.pad, self.pad), mode="circular")


class ActorCriticModular(nn.Module):
    """Modality-aware actor-critic with shared encoders and shared fusion.

    Architecture:
        Gas history (30) --> GRU --> gas_features (GRU_HIDDEN)
        LiDAR (24)        --> CircularConv1D --> lidar_features (CONV_CHANNELS)
        Context (5)       --> pass-through

        [gas_features | lidar_features | context] --> fusion MLP --> shared features
            |
            +-- Actor head --> (alpha, beta) --> Beta dist --> angle
            +-- Critic head --> V(s)
    """

    def __init__(self, obs_dim=cfg.STATE_DIM,
                 gas_len=cfg.GAS_HISTORY_LENGTH * cfg.GAS_FEATURES_PER_STEP,
                 lidar_len=cfg.LIDAR_NUM_RAYS,
                 gru_hidden=cfg.GAS_GRU_HIDDEN,
                 conv_channels=cfg.LIDAR_CONV_CHANNELS,
                 conv_kernel=cfg.LIDAR_CONV_KERNEL,
                 hidden_dim=cfg.HIDDEN_DIM,
                 actor_head_dim=cfg.ACTOR_HEAD_DIM,
                 critic_head_dim=cfg.CRITIC_HEAD_DIM):
        super().__init__()

        self.gas_len = gas_len
        self.lidar_len = lidar_len
        self.gas_steps = gas_len // cfg.GAS_FEATURES_PER_STEP
        self.context_len = obs_dim - gas_len - lidar_len  # pos(2)+wind(2)+time(1) = 5

        # --- Gas encoder: GRU over the spatial detection history ---
        self.gas_gru = nn.GRU(input_size=cfg.GAS_FEATURES_PER_STEP, hidden_size=gru_hidden, batch_first=True)
        gas_out_dim = gru_hidden

        # --- LiDAR encoder: 1D conv with circular padding ---
        pad = conv_kernel // 2
        self.lidar_conv = nn.Sequential(
            CircularPad1d(pad),
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
            CircularPad1d(pad),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
        )
        lidar_out_dim = conv_channels * lidar_len

        # --- Fusion MLP ---
        fusion_in = gas_out_dim + lidar_out_dim + self.context_len
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_in, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # --- Actor head ---
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, actor_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_head_dim, 2), std=0.01),
            nn.Softplus(),
        )

        # --- Critic head ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, critic_head_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_head_dim, 1), std=1.0),
        )

    def _encode(self, obs):
        """Split observation and encode each modality."""
        gas = obs[:, :self.gas_len]                                    # (B, 30)
        lidar = obs[:, self.gas_len:self.gas_len + self.lidar_len]     # (B, 24)
        context = obs[:, self.gas_len + self.lidar_len:]               # (B, 5)

        # GRU: (B, 30) -> (B, 10, 3) -> GRU -> last hidden (B, gru_hidden)
        gas_seq = gas.view(gas.size(0), self.gas_steps, cfg.GAS_FEATURES_PER_STEP)
        _, gas_h = self.gas_gru(gas_seq)         # gas_h: (1, B, gru_hidden)
        gas_feat = gas_h.squeeze(0)              # (B, gru_hidden)

        # Conv1D: (B, 24) -> (B, 1, 24) -> conv -> (B, C, 24) -> flatten
        lidar_seq = lidar.unsqueeze(1)           # (B, 1, 24)
        lidar_feat = self.lidar_conv(lidar_seq)  # (B, C, 24)
        lidar_feat = lidar_feat.flatten(1)       # (B, C*24)

        fused = torch.cat([gas_feat, lidar_feat, context], dim=1)
        return self.fusion(fused)

    def get_value(self, obs):
        return self.critic(self._encode(obs))

    def get_action_and_value(self, obs, action=None):
        features = self._encode(obs)

        ab = self.actor(features)
        alpha = ab[:, 0:1] + 1.0
        beta = ab[:, 1:2] + 1.0
        dist = Beta(alpha, beta)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features)

        return action, log_prob, entropy, value


class ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> ReLU with residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.linear = layer_init(nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.norm(F.relu(self.linear(x)))


class GatedFusion(nn.Module):
    """Soft gating over concatenated modality features."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            layer_init(nn.Linear(dim, dim)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ActorCriticDualBackbone(nn.Module):
    """Modality-aware actor-critic with gated fusion, separate backbones,
    residual connections, LayerNorm, and circular Gaussian action space.

    Architecture:
        Gas history (30) --> GRU (2-layer, shared) --> 64 dims
        LiDAR (24)       --> CircConv1D (shared)   --> 48 dims
        Context (5)      --> pass-through           -->  5 dims
                                    |
                          concat --> GatedFusion (85 dims)
                         /                          \\
            Actor projection [128]          Critic projection [256]
            + LayerNorm + ReLU              + LayerNorm + ReLU
            ResidualBlock x3                ResidualBlock x3
                  |                               |
            Actor head [128]                Critic head [256]
            --> (mean_cos, mean_sin,        --> V(s)
                 log_std_cos, log_std_sin)
                  |
            2D Gaussian --> (cos θ, sin θ) --> atan2 --> angle

    Action: (cos θ, sin θ) -- naturally circular, no wrapping discontinuity.
    """

    ACTION_DIM = 2  # (cos θ, sin θ)

    def __init__(self, obs_dim=cfg.STATE_DIM,
                 gas_len=cfg.GAS_HISTORY_LENGTH * cfg.GAS_FEATURES_PER_STEP,
                 lidar_len=cfg.LIDAR_NUM_RAYS,
                 gru_hidden=cfg.GAS_GRU_HIDDEN,
                 conv_channels=cfg.LIDAR_CONV_CHANNELS,
                 conv_kernel=cfg.LIDAR_CONV_KERNEL,
                 actor_fusion_dim=128,
                 critic_fusion_dim=256,
                 actor_head_dim=cfg.ACTOR_HEAD_DIM,
                 critic_head_dim=256):
        super().__init__()

        self.gas_len = gas_len
        self.lidar_len = lidar_len
        self.gas_steps = gas_len // cfg.GAS_FEATURES_PER_STEP
        self.context_len = obs_dim - gas_len - lidar_len

        # --- Shared encoders ---
        self.gas_gru = nn.GRU(input_size=cfg.GAS_FEATURES_PER_STEP, hidden_size=gru_hidden,
                              num_layers=2, batch_first=True)
        gas_out_dim = gru_hidden

        pad = conv_kernel // 2
        self.lidar_conv = nn.Sequential(
            CircularPad1d(pad),
            nn.Conv1d(1, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
            CircularPad1d(pad),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel),
            nn.ReLU(),
        )
        lidar_out_dim = conv_channels * lidar_len

        fusion_in = gas_out_dim + lidar_out_dim + self.context_len

        # --- Gated fusion ---
        self.gate = GatedFusion(fusion_in)

        # --- Actor path ---
        self.actor_proj = nn.Sequential(
            layer_init(nn.Linear(fusion_in, actor_fusion_dim)),
            nn.LayerNorm(actor_fusion_dim),
            nn.ReLU(),
        )
        self.actor_res = nn.Sequential(
            ResidualBlock(actor_fusion_dim),
            ResidualBlock(actor_fusion_dim),
            ResidualBlock(actor_fusion_dim),
        )
        # Outputs: mean_cos, mean_sin, log_std_cos, log_std_sin
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(actor_fusion_dim, actor_head_dim)),
            nn.LayerNorm(actor_head_dim),
            nn.ReLU(),
            layer_init(nn.Linear(actor_head_dim, 4), std=0.01),
        )

        # --- Critic path (wider: 256) ---
        self.critic_proj = nn.Sequential(
            layer_init(nn.Linear(fusion_in, critic_fusion_dim)),
            nn.LayerNorm(critic_fusion_dim),
            nn.ReLU(),
        )
        self.critic_res = nn.Sequential(
            ResidualBlock(critic_fusion_dim),
            ResidualBlock(critic_fusion_dim),
            ResidualBlock(critic_fusion_dim),
        )
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(critic_fusion_dim, critic_head_dim)),
            nn.LayerNorm(critic_head_dim),
            nn.ReLU(),
            layer_init(nn.Linear(critic_head_dim, 1), std=1.0),
        )

    def _encode_shared(self, obs):
        """Encode observation through shared gas/lidar encoders + gated fusion."""
        gas = obs[:, :self.gas_len]
        lidar = obs[:, self.gas_len:self.gas_len + self.lidar_len]
        context = obs[:, self.gas_len + self.lidar_len:]

        gas_seq = gas.view(gas.size(0), self.gas_steps, cfg.GAS_FEATURES_PER_STEP)
        _, gas_h = self.gas_gru(gas_seq)
        gas_feat = gas_h[-1]

        lidar_seq = lidar.unsqueeze(1)
        lidar_feat = self.lidar_conv(lidar_seq).flatten(1)

        fused = torch.cat([gas_feat, lidar_feat, context], dim=1)
        return self.gate(fused)

    def _actor_dist(self, encoded):
        """Build 2D Gaussian distribution for (cos θ, sin θ)."""
        feat = self.actor_res(self.actor_proj(encoded))
        out = self.actor_head(feat)  # (B, 4)
        mean = out[:, :2]           # (mean_cos, mean_sin)
        log_std = out[:, 2:].clamp(-5, 0.5)
        std = log_std.exp()
        return Normal(mean, std)

    def get_value(self, obs):
        encoded = self._encode_shared(obs)
        feat = self.critic_res(self.critic_proj(encoded))
        return self.critic_head(feat)

    def get_action_and_value(self, obs, action=None):
        encoded = self._encode_shared(obs)

        # Actor
        dist = self._actor_dist(encoded)
        if action is None:
            action = dist.rsample()  # (B, 2): (cos θ, sin θ)

        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)           # (B,)

        # Critic
        critic_feat = self.critic_res(self.critic_proj(encoded))
        value = self.critic_head(critic_feat)

        return action, log_prob, entropy, value
