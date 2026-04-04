"""
PPO Actor-Critic network with shared MLP backbone and Beta-distribution actor.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

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
