"""
MapCNN — 5-layer strided CNN encoder for the ego-centric occupancy crop.

Input : (B, 2, MAP_CANVAS_H, MAP_CANVAS_W)  — channel 0: occupied, channel 1: known
Output: (B, MAP_FEAT_DIM)
"""

import numpy as np
import torch.nn as nn
from .. import config as cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (CleanRL convention)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MapCNN(nn.Module):
    """Encode a 2-channel (150 x 200) occupancy crop into a feature vector.

    Architecture (5 conv layers):
        Layer 1: stride=1  — detects edges, doorways (6-cell minimum gap)
        Layer 2: stride=2  — (32,  75, 100) — local structure
        Layer 3: stride=2  — (64,  38,  50) — room shapes; RF matches LiDAR range
        Layer 4: stride=2  — (64,  19,  25) — room-scale topology
        Layer 5: stride=2  — (64,  10,  13) — cross-room layout
        Flatten + Linear   — (MAP_FEAT_DIM,)

    GroupNorm after each conv stabilises training across 5 layers of
    stride-2 compression. GroupNorm (not BatchNorm) is used deliberately:
    its statistics are per-sample, so the encoder is independent of batch
    size/composition and behaves identically in train() and eval(). This
    matters because the teacher is an on-policy PPO actor — BatchNorm would
    make the policy batch-dependent and corrupt the importance ratio between
    rollout (small batch) and update (large minibatch), and would shift the
    policy when the teacher is frozen with eval() during distillation.
    """

    def __init__(self, map_feat_dim: int = cfg.MAP_FEAT_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            # Layer 1 — stride 1, preserve fine structure
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            # Layer 2 — stride 2 → (32, 75, 100)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            # Layer 3 — stride 2 → (64, 38, 50)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            # Layer 4 — stride 2 → (64, 19, 25)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            # Layer 5 — stride 2 → (64, 10, 13)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        # 64 * 10 * 13 = 8320
        self.proj = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 10 * 13, map_feat_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 2, MAP_CANVAS_H, MAP_CANVAS_W).

        Returns
        -------
        torch.Tensor
            Shape (B, map_feat_dim).
        """
        return self.proj(self.cnn(x))
