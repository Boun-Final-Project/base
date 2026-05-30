"""
Distillation-aware PPO components.

Provides:
  kl_diagonal_gaussian   — closed-form KL(student || teacher) for diagonal Gaussians
  build_map_canvases     — reconstruct map canvases from stored positions + registry
  DistilRolloutBuffer    — RolloutBuffer extended with agent_positions and map_ids
  teacher_ppo_update     — standard PPO update for ActorCriticTeacher (Phase 1)
  distil_ppo_update      — PPO + KL distillation update for student (Phase 2)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from .. import config as cfg
from .ppo import RolloutBuffer, compute_gae


# ---------------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------------

def kl_diagonal_gaussian(
    mu_s: torch.Tensor,
    log_std_s: torch.Tensor,
    mu_t: torch.Tensor,
    log_std_t: torch.Tensor,
) -> torch.Tensor:
    """KL(N(mu_s, sigma_s) || N(mu_t, sigma_t)), summed over action dimensions.

    Closed-form per dimension:
        KL_i = (log_std_t_i - log_std_s_i)
               + (exp(2*log_std_s_i) + (mu_s_i - mu_t_i)^2) / (2*exp(2*log_std_t_i))
               - 0.5

    Parameters
    ----------
    mu_s, log_std_s : (B, 2)  student distribution parameters
    mu_t, log_std_t : (B, 2)  teacher distribution parameters (detached by caller)

    Returns
    -------
    kl : (B,)  non-negative, zero iff student == teacher
    """
    kl_per_dim = (
        (log_std_t - log_std_s)
        + (log_std_s.mul(2).exp() + (mu_s - mu_t).pow(2))
          / (log_std_t.mul(2).exp() * 2.0)
        - 0.5
    )
    return kl_per_dim.sum(dim=-1)   # (B,)


# ---------------------------------------------------------------------------
# Map canvas reconstruction
# ---------------------------------------------------------------------------

def build_map_canvases(
    agent_positions: np.ndarray,   # (B, 2)  float32, world coords
    map_ids: np.ndarray,           # (B,)    int32, keys into map_registry
    map_registry: dict,            # int → np.ndarray (H_ds, W_ds) at MAP_DOWNSAMPLE_RES
    dropped: np.ndarray | None = None,  # (B,) bool — True rows are blanked to zeros
) -> torch.Tensor:                 # (B, 2, MAP_CANVAS_H, MAP_CANVAS_W)
    """Reconstruct ego-centric map canvases for a minibatch on-the-fly.

    Rows flagged in ``dropped`` are returned as all-zero canvases (the
    "map withheld" signal used by teacher map-dropout training).
    """
    B = len(agent_positions)
    canvases = np.zeros(
        (B, 2, cfg.MAP_CANVAS_H, cfg.MAP_CANVAS_W), dtype=np.float32
    )
    ds_res = cfg.MAP_DOWNSAMPLE_RES
    half_w = cfg.MAP_HALF_W
    half_h = cfg.MAP_HALF_H

    for i in range(B):
        rx, ry   = agent_positions[i]
        map_ds   = map_registry[int(map_ids[i])]
        ds_h, ds_w = map_ds.shape

        dst_x0 = int(round((half_w - rx) / ds_res))
        dst_y0 = int(round((half_h - ry) / ds_res))
        dst_x1 = dst_x0 + ds_w
        dst_y1 = dst_y0 + ds_h
        src_x0, src_y0 = 0, 0
        src_x1, src_y1 = ds_w, ds_h

        if dst_x0 < 0:
            src_x0 -= dst_x0; dst_x0 = 0
        if dst_y0 < 0:
            src_y0 -= dst_y0; dst_y0 = 0
        if dst_x1 > cfg.MAP_CANVAS_W:
            src_x1 -= dst_x1 - cfg.MAP_CANVAS_W; dst_x1 = cfg.MAP_CANVAS_W
        if dst_y1 > cfg.MAP_CANVAS_H:
            src_y1 -= dst_y1 - cfg.MAP_CANVAS_H; dst_y1 = cfg.MAP_CANVAS_H

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            canvases[i, 0, dst_y0:dst_y1, dst_x0:dst_x1] = (
                map_ds[src_y0:src_y1, src_x0:src_x1]
            )
            canvases[i, 1, dst_y0:dst_y1, dst_x0:dst_x1] = 1.0

    if dropped is not None:
        canvases[np.asarray(dropped, dtype=bool)] = 0.0
    return torch.from_numpy(canvases)


# ---------------------------------------------------------------------------
# DistilRolloutBuffer
# ---------------------------------------------------------------------------

class DistilRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extended with agent position and map ID tracking."""

    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        super().__init__(num_steps, num_envs, obs_dim, action_dim, device)
        self.agent_positions = np.zeros((num_steps, num_envs, 2), dtype=np.float32)
        self.map_ids         = np.zeros((num_steps, num_envs),    dtype=np.int32)
        self.map_dropped     = np.zeros((num_steps, num_envs),    dtype=bool)
        self._map_registry: dict[int, np.ndarray] = {}
        self._current_map_id = np.zeros(num_envs, dtype=np.int32)
        self._next_id = 0

    def register_map(self, env_idx: int, map_ds: np.ndarray) -> None:
        """Register a (new) downsampled map for env_idx."""
        mid = self._next_id
        self._map_registry[mid] = map_ds
        self._current_map_id[env_idx] = mid
        self._next_id += 1

    def clear_registry(self) -> None:
        """Discard accumulated maps — call before each rollout to prevent growth.

        After calling this, register_map() must be called for every env before
        insert() is called, otherwise insert() will raise AssertionError.
        """
        self._map_registry.clear()
        self._next_id = 0
        self._current_map_id[:] = -1

    def reset(self) -> None:
        super().reset()
        # Do NOT clear the registry here; caller decides when to clear.

    def insert(self, obs, action, log_prob, reward, done, value,
               agent_pos: np.ndarray, map_dropped: np.ndarray | None = None) -> None:
        assert (self._current_map_id >= 0).all(), \
            "register_map() must be called for all envs before insert()"
        self.agent_positions[self.pos] = agent_pos
        self.map_ids[self.pos]         = self._current_map_id
        self.map_dropped[self.pos]     = False if map_dropped is None else map_dropped
        super().insert(obs, action, log_prob, reward, done, value)

    def get_batches(self, advantages, returns, num_minibatches):
        """Yield shuffled minibatches; adds 'agent_positions' and 'map_ids' keys."""
        batch_size = self.num_steps * self.num_envs
        mb_size    = batch_size // num_minibatches

        b_obs       = self.obs.reshape(batch_size, -1)
        b_actions   = self.actions.reshape(batch_size, -1)
        b_log_probs = self.log_probs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns   = returns.reshape(batch_size)
        b_values    = self.values.reshape(batch_size)
        b_agent_pos = self.agent_positions.reshape(batch_size, 2)
        b_map_ids   = self.map_ids.reshape(batch_size)
        b_map_dropped = self.map_dropped.reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, mb_size):
            mb_idx  = indices[start:start + mb_size]
            cpu_idx = mb_idx.cpu().numpy()
            yield {
                "obs":             b_obs[mb_idx],
                "actions":         b_actions[mb_idx],
                "log_probs":       b_log_probs[mb_idx],
                "advantages":      b_advantages[mb_idx],
                "returns":         b_returns[mb_idx],
                "values":          b_values[mb_idx],
                "agent_positions": b_agent_pos[cpu_idx],
                "map_ids":         b_map_ids[cpu_idx],
                "map_dropped":     b_map_dropped[cpu_idx],
            }


# ---------------------------------------------------------------------------
# Phase 1 — Teacher PPO update
# ---------------------------------------------------------------------------

def teacher_ppo_update(
    teacher,
    optimizer,
    buffer: DistilRolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    entropy_coeff: float,
    value_loss_coeff: float,
    max_grad_norm: float,
    update_epochs: int,
    num_minibatches: int,
    device: torch.device,
    target_kl: float | None = None,
) -> dict:
    """Standard PPO update for ActorCriticTeacher.

    Identical to ppo_update() in ppo.py except it reconstructs map canvases
    from the buffer and passes them to the teacher's forward methods.
    """
    adv_flat   = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    totals = defaultdict(float)
    n_updates = 0

    for epoch in range(update_epochs):
        epoch_kl = 0.0
        epoch_batches = 0

        for mb in buffer.get_batches(advantages, returns, num_minibatches):
            canvases = build_map_canvases(
                mb["agent_positions"], mb["map_ids"], buffer._map_registry,
                dropped=mb["map_dropped"],
            ).to(device)

            _, new_log_prob, entropy, new_value = teacher.get_action_and_value(
                mb["obs"], canvases, action=mb["actions"]
            )

            log_ratio = new_log_prob - mb["log_probs"]
            ratio     = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac  = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

            mb_adv = mb["advantages"]
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon),
            ).mean()

            v_loss = 0.5 * torch.max(
                (new_value - mb["returns"]) ** 2,
                (mb["values"] + (new_value - mb["values"]).clamp(
                    -clip_epsilon, clip_epsilon) - mb["returns"]) ** 2,
            ).mean()

            loss = pg_loss - entropy_coeff * entropy.mean() + value_loss_coeff * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += pg_loss.item()
            totals["value_loss"]  += v_loss.item()
            totals["entropy"]     += entropy.mean().item()
            totals["approx_kl"]   += approx_kl.item()
            totals["clipfrac"]    += clipfrac.item()
            n_updates     += 1
            epoch_kl      += approx_kl.item()
            epoch_batches += 1

        if target_kl is not None and epoch_batches > 0:
            if epoch_kl / epoch_batches > target_kl:
                break

    return {k: v / n_updates for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Phase 2 — Student distillation update
# ---------------------------------------------------------------------------

def distil_ppo_update(
    student,
    frozen_teacher,
    optimizer,
    buffer: DistilRolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    entropy_coeff: float,
    value_loss_coeff: float,
    max_grad_norm: float,
    update_epochs: int,
    num_minibatches: int,
    distil_lambda: float,
    device: torch.device,
    target_kl: float | None = None,
    distil_from_map: bool = True,
) -> dict:
    """PPO + KL-distillation update for ActorCriticDualBackbone (student).

    Loss = L_PPO(student) + distil_lambda * KL(student || teacher)

    The teacher is frozen (no gradients computed through it).
    """
    adv_flat   = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    totals = defaultdict(float)
    n_updates = 0

    for epoch in range(update_epochs):
        epoch_kl = 0.0
        epoch_batches = 0

        for mb in buffer.get_batches(advantages, returns, num_minibatches):
            canvases = build_map_canvases(
                mb["agent_positions"], mb["map_ids"], buffer._map_registry
            ).to(device)

            # --- Standard PPO on student ---
            _, new_log_prob, entropy, new_value = student.get_action_and_value(
                mb["obs"], action=mb["actions"]
            )

            log_ratio = new_log_prob - mb["log_probs"]
            ratio     = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac  = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

            mb_adv = mb["advantages"]
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon),
            ).mean()

            v_loss = 0.5 * torch.max(
                (new_value - mb["returns"]) ** 2,
                (mb["values"] + (new_value - mb["values"]).clamp(
                    -clip_epsilon, clip_epsilon) - mb["returns"]) ** 2,
            ).mean()

            ppo_loss = pg_loss - entropy_coeff * entropy.mean() + value_loss_coeff * v_loss

            # --- Distillation: KL(student || teacher) ---
            # map-on: real canvas (privileged target). map-off: blank canvas
            # (target recoverable from the student's own observations).
            teacher_canvas = canvases if distil_from_map else torch.zeros_like(canvases)
            mu_s, log_std_s = student.get_actor_params(mb["obs"])
            with torch.no_grad():
                mu_t, log_std_t = frozen_teacher.get_actor_params(mb["obs"], teacher_canvas)

            kl_loss = kl_diagonal_gaussian(mu_s, log_std_s, mu_t, log_std_t).mean()

            loss = ppo_loss + distil_lambda * kl_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

            totals["ppo_loss"]    += ppo_loss.item()
            totals["policy_loss"] += pg_loss.item()
            totals["value_loss"]  += v_loss.item()
            totals["kl_loss"]     += kl_loss.item()
            totals["entropy"]     += entropy.mean().item()
            totals["approx_kl"]   += approx_kl.item()
            totals["clipfrac"]    += clipfrac.item()
            n_updates     += 1
            epoch_kl      += approx_kl.item()
            epoch_batches += 1

        if target_kl is not None and epoch_batches > 0:
            if epoch_kl / epoch_batches > target_kl:
                break

    return {k: v / n_updates for k, v in totals.items()}
