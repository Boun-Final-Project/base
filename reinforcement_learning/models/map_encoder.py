"""
Map encoders for the ego-centric occupancy crop.

Input : (B, 2, MAP_CANVAS_H, MAP_CANVAS_W)  — channel 0: occupied, channel 1: known
Output: (B, MAP_FEAT_DIM)

Two encoders share the same 5-layer strided conv stem:
  - MapCNN       : flatten the conv feature map → Linear (fixed pooling).
  - MapCrossAttn : the agent's sensor state queries the conv tokens via
                   multi-head cross-attention (query-conditioned readout).
"""

import numpy as np
import torch
import torch.nn as nn
from .. import config as cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (CleanRL convention)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _group_norm(channels):
    """GroupNorm with ~4 channels per group, guaranteed to divide `channels`."""
    groups = max(1, channels // 4)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def build_cnn_stem():
    """5-layer strided conv stem shared by MapCNN and MapCrossAttn.

    (B, 2, H, W) → (B, C, ceil(H/16), ceil(W/16)) where C = cfg.MAP_STEM_CHANNELS.

    Layer widths scale with C as [C//4, C//2, C, C, C], so C=64 recovers the
    original [16, 32, 64, 64, 64] stem and the default C=16 gives a lean
    [4, 8, 16, 16, 16] stem (occupancy is low-entropy — 2 bits/cell — so a
    narrow stem is usually sufficient).

    Strides 1,2,2,2,2; GroupNorm (per-sample, not BatchNorm) after each conv
    so the encoder is independent of batch size/composition and behaves
    identically in train() and eval() — required for an on-policy PPO actor
    and for a teacher frozen with eval() during distillation.
    """
    c = cfg.MAP_STEM_CHANNELS
    w1, w2 = max(1, c // 4), max(1, c // 2)
    return nn.Sequential(
        # Layer 1 — stride 1, preserve fine structure (doorways, 6-cell gaps)
        nn.Conv2d(2, w1, kernel_size=3, stride=1, padding=1, bias=False),
        _group_norm(w1),
        nn.ReLU(inplace=True),
        # Layer 2 — stride 2 — local structure
        nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False),
        _group_norm(w2),
        nn.ReLU(inplace=True),
        # Layer 3 — stride 2 — room shapes; RF matches LiDAR range
        nn.Conv2d(w2, c, kernel_size=3, stride=2, padding=1, bias=False),
        _group_norm(c),
        nn.ReLU(inplace=True),
        # Layer 4 — stride 2 — room-scale topology
        nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
        _group_norm(c),
        nn.ReLU(inplace=True),
        # Layer 5 — stride 2 — cross-room layout
        nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
        _group_norm(c),
        nn.ReLU(inplace=True),
    )


def _stem_grid():
    """Token grid (h, w) after the conv stem for the configured canvas."""
    _h = (cfg.MAP_CANVAS_H + 15) // 16
    _w = (cfg.MAP_CANVAS_W + 15) // 16
    return _h, _w


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
        self.cnn = build_cnn_stem()
        # H,W after 5 conv layers (strides 1,2,2,2,2): ceil(H/16), ceil(W/16)
        # 150x200 with C=16 -> 16*10*13 = 2080 flattened features
        _h, _w = _stem_grid()
        self.proj = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(cfg.MAP_STEM_CHANNELS * _h * _w, map_feat_dim)),
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


class MapCrossAttn(nn.Module):
    """Query-conditioned map readout via multi-head cross-attention.

    The agent's fused sensor state (gas + lidar + context) forms a single
    query that attends over the conv-stem tokens — "given what I smell and
    see, which parts of the map matter?" — instead of a fixed pooled summary.

    Pipeline (C = cfg.MAP_STEM_CHANNELS):
        map_canvas (B, 2, H, W)
            → conv stem → (B, C, h, w) → h*w tokens of dim C  (+ pos embed)
        agent_feat (B, agent_feat_dim) → query (B, 1, C)
            → MultiheadAttention(query, tokens, tokens) → (B, 1, C)
            → Linear → (B, map_feat_dim)

    Multi-head attention on a single query yields one weighted combination of
    the value vectors *per head*, mitigating the single-query bottleneck. The
    conv tokens (not raw cells) carry local geometry, so the agent can attend
    to structures (openings, corridors) rather than isolated pixels. Token
    count is small (h*w ≈ 35), so cost is negligible.

    A blank (zeroed) map_canvas — produced for map-dropout episodes — yields
    deterministic constant tokens through the stem, so "map off" maps to a
    fixed readout, mirroring MapCNN's behaviour. No separate null token is
    needed, and attention introduces no batch dependence (PPO/eval-safe).
    """

    def __init__(self, agent_feat_dim: int,
                 map_feat_dim: int = cfg.MAP_FEAT_DIM,
                 num_heads: int = cfg.MAP_ATTN_HEADS):
        super().__init__()
        self.cnn = build_cnn_stem()
        _h, _w = _stem_grid()
        self.num_tokens = _h * _w
        embed_dim = cfg.MAP_STEM_CHANNELS  # token dim == conv channels (no token reprojection)
        assert embed_dim % num_heads == 0, (
            f"MAP_ATTN_HEADS ({num_heads}) must divide MAP_STEM_CHANNELS ({embed_dim})"
        )
        self.embed_dim = embed_dim

        # Learned positional embedding over the (fixed, ego-centric) token grid.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Agent sensor state → attention query.
        self.q_proj = layer_init(nn.Linear(agent_feat_dim, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            layer_init(nn.Linear(embed_dim, map_feat_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, map_canvas, agent_feat):
        """
        Parameters
        ----------
        map_canvas : torch.Tensor
            Shape (B, 2, MAP_CANVAS_H, MAP_CANVAS_W).
        agent_feat : torch.Tensor
            Shape (B, agent_feat_dim) — fused gas+lidar+context features.

        Returns
        -------
        torch.Tensor
            Shape (B, map_feat_dim).
        """
        x = self.cnn(map_canvas)                       # (B, C, h, w)
        tokens = x.flatten(2).transpose(1, 2)          # (B, T, C)
        tokens = tokens + self.pos_embed               # (B, T, C)
        query = self.q_proj(agent_feat).unsqueeze(1)   # (B, 1, C)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)  # (B, 1, C)
        return self.proj(attn_out.squeeze(1))          # (B, map_feat_dim)
