"""
CleanRL-style PPO implementation for continuous action spaces (Beta distribution).
"""

import numpy as np
import torch
import torch.nn as nn

from .. import config as cfg


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, batch):
        """Update with a batch of values (numpy array)."""
        batch = batch.ravel()
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return max(self.var ** 0.5, 1e-6)


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : torch.Tensor
        Shape (num_steps, num_envs).
    values : torch.Tensor
        Shape (num_steps, num_envs).
    dones : torch.Tensor
        Shape (num_steps, num_envs). 1.0 if episode ended.
    next_value : torch.Tensor
        Shape (num_envs,). V(s_{T+1}) from the critic.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda.

    Returns
    -------
    advantages : torch.Tensor
        Shape (num_steps, num_envs).
    returns : torch.Tensor
        Shape (num_steps, num_envs). advantages + values.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Storage for rollout data collected from vectorized environments."""

    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.pos = 0

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

    def insert(self, obs, action, log_prob, reward, done, value):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1

    def reset(self):
        self.pos = 0

    def get_batches(self, advantages, returns, num_minibatches):
        """Yield shuffled minibatches for PPO update.

        Flattens (num_steps, num_envs) into a single batch dimension,
        then shuffles and splits into minibatches.

        Yields
        ------
        dict with keys: obs, actions, log_probs, advantages, returns, values
        """
        batch_size = self.num_steps * self.num_envs
        mb_size = batch_size // num_minibatches
        assert mb_size > 0

        # Flatten
        b_obs = self.obs.reshape(batch_size, -1)
        b_actions = self.actions.reshape(batch_size, -1)
        b_log_probs = self.log_probs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = self.values.reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, mb_size):
            end = start + mb_size
            mb_idx = indices[start:end]
            yield {
                "obs": b_obs[mb_idx],
                "actions": b_actions[mb_idx],
                "log_probs": b_log_probs[mb_idx],
                "advantages": b_advantages[mb_idx],
                "returns": b_returns[mb_idx],
                "values": b_values[mb_idx],
            }


class SpatialRolloutBuffer:
    """Rollout buffer for spatial (4, G, G) observations with recurrent state.

    Spatial tensor is kept on CPU to avoid large GPU allocations (e.g. for
    G=221: 1024 steps × 8 envs × 4 × 221 × 221 × float32 ≈ 6 GB).
    Minibatches are moved to device on demand inside get_seq_batches().

    Sequences for truncated BPTT are formed by splitting each env's rollout
    into chunks of length SEQ_LEN. The hidden state stored at the start of
    each chunk is the GRU state *before* processing that chunk's first
    observation.
    """

    def __init__(self, num_steps, num_envs, action_dim, device,
                 gru_hidden=None, seq_len=None):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.device    = device
        self.pos       = 0

        G = cfg.SPATIAL_GRID_SIZE
        self._G = G
        self.spatial_obs = torch.zeros((num_steps, num_envs, 4, G, G), dtype=torch.float32)

        self.actions   = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs),             device=device)
        self.rewards   = torch.zeros((num_steps, num_envs),             device=device)
        self.dones     = torch.zeros((num_steps, num_envs),             device=device)
        self.values    = torch.zeros((num_steps, num_envs),             device=device)

        # Recurrent state: h_t stored *before* processing obs[t]
        self.gru_hidden = gru_hidden if gru_hidden is not None else cfg.SPATIAL_GRU_HIDDEN
        self.seq_len    = seq_len    if seq_len    is not None else cfg.SPATIAL_SEQ_LEN
        self.hidden_states = torch.zeros(
            (num_steps, num_envs, self.gru_hidden), device=device
        )

    def insert(self, spatial, action, log_prob, reward, done, value, hidden):
        """hidden : (1, num_envs, gru_hidden) — h_t at the *start* of this step."""
        self.spatial_obs[self.pos]   = spatial.cpu()
        self.actions[self.pos]       = action
        self.log_probs[self.pos]     = log_prob
        self.rewards[self.pos]       = reward
        self.dones[self.pos]         = done
        self.values[self.pos]        = value
        self.hidden_states[self.pos] = hidden.squeeze(0)
        self.pos += 1

    def reset(self):
        self.pos = 0

    def get_seq_batches(self, advantages, returns, num_minibatches):
        """Yield minibatches of sequences for recurrent PPO.

        Each yielded minibatch is a dict whose tensors have leading dims
        (seq_len, B_seq, ...). Hidden state h0 has shape (1, B_seq, gru_hidden).
        """
        T, N, S = self.num_steps, self.num_envs, self.seq_len
        assert T % S == 0, f"rollout_length ({T}) must be divisible by SEQ_LEN ({S})"
        n_chunks_per_env = T // S
        n_seq            = n_chunks_per_env * N
        mb_seqs          = max(1, n_seq // num_minibatches)

        # Reshape (T, N, ...) → (n_chunks, S, N, ...) → flatten chunk×env axes
        def _seq_view(x):
            # x: (T, N, *) → (n_chunks, S, N, *) → (S, n_chunks*N, *)
            new_shape = (n_chunks_per_env, S, N) + tuple(x.shape[2:])
            return x.reshape(new_shape).permute(1, 0, 2, *range(3, x.ndim + 1)).reshape(
                S, n_seq, *x.shape[2:]
            )

        # Hidden states at the start of each chunk: hidden_states[chunk * S]
        # (n_chunks, N, H) → (n_chunks*N, H) → unsqueeze to (1, n_seq, H)
        h0 = self.hidden_states[::S].reshape(n_seq, self.gru_hidden)   # (n_seq, H)

        seq_spatial    = _seq_view(self.spatial_obs)                   # (S, n_seq, 4, G, G) on CPU
        seq_actions    = _seq_view(self.actions)                       # (S, n_seq, A)
        seq_log_probs  = _seq_view(self.log_probs)                     # (S, n_seq)
        seq_dones      = _seq_view(self.dones)                         # (S, n_seq)
        seq_values     = _seq_view(self.values)                        # (S, n_seq)
        seq_advantages = _seq_view(advantages)                         # (S, n_seq)
        seq_returns    = _seq_view(returns)                            # (S, n_seq)

        order = torch.randperm(n_seq, device=self.device)
        for start in range(0, n_seq, mb_seqs):
            idx     = order[start:start + mb_seqs]
            cpu_idx = idx.cpu()
            yield {
                "spatial":    seq_spatial[:, cpu_idx].to(self.device),     # (S, B, 4, G, G)
                "actions":    seq_actions[:, idx],                          # (S, B, A)
                "log_probs":  seq_log_probs[:, idx],                        # (S, B)
                "dones":      seq_dones[:, idx],                            # (S, B)
                "values":     seq_values[:, idx],                           # (S, B)
                "advantages": seq_advantages[:, idx],                       # (S, B)
                "returns":    seq_returns[:, idx],                          # (S, B)
                "h0":         h0[idx].unsqueeze(0).contiguous(),            # (1, B, H)
            }


def spatial_ppo_update(agent, optimizer, buffer, advantages, returns,
                       clip_epsilon, entropy_coeff, value_loss_coeff,
                       max_grad_norm, update_epochs, num_minibatches,
                       target_kl=None):
    """Recurrent PPO update for the spatial architecture (truncated BPTT)."""
    adv_flat   = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    total_pg_loss = total_v_loss = total_entropy = 0.0
    total_clipfrac = total_approx_kl = 0.0
    n_updates = 0

    for epoch in range(update_epochs):
        epoch_kl      = 0.0
        epoch_batches = 0
        for mb in buffer.get_seq_batches(advantages, returns, num_minibatches):
            # mb tensors are (S, B, ...). h0 is (1, B, gru_hidden), detached.
            h0 = mb["h0"].detach()

            new_log_prob, entropy, new_value = agent.evaluate_sequence(
                mb["spatial"], h0, mb["actions"], dones_seq=mb["dones"]
            )

            log_ratio = new_log_prob - mb["log_probs"]
            ratio     = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac  = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

            mb_adv  = mb["advantages"]
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon),
            ).mean()

            v_loss_unclipped = (new_value - mb["returns"]) ** 2
            v_clipped        = mb["values"] + torch.clamp(
                new_value - mb["values"], -clip_epsilon, clip_epsilon
            )
            v_loss = 0.5 * torch.max(
                v_loss_unclipped, (v_clipped - mb["returns"]) ** 2
            ).mean()

            loss = pg_loss - entropy_coeff * entropy.mean() + value_loss_coeff * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss   += pg_loss.item()
            total_v_loss    += v_loss.item()
            total_entropy   += entropy.mean().item()
            total_clipfrac  += clipfrac.item()
            total_approx_kl += approx_kl.item()
            n_updates       += 1
            epoch_kl        += approx_kl.item()
            epoch_batches   += 1

        if target_kl is not None and epoch_batches > 0:
            if epoch_kl / epoch_batches > target_kl:
                break

    return {
        "policy_loss": total_pg_loss   / n_updates,
        "value_loss":  total_v_loss    / n_updates,
        "entropy":     total_entropy   / n_updates,
        "clipfrac":    total_clipfrac  / n_updates,
        "approx_kl":   total_approx_kl / n_updates,
    }


def ppo_update(agent, optimizer, buffer, advantages, returns,
               clip_epsilon, entropy_coeff, value_loss_coeff,
               max_grad_norm, update_epochs, num_minibatches,
               target_kl=None):
    """Run PPO update epochs on the collected rollout.

    Parameters
    ----------
    agent : ActorCritic
    optimizer : torch.optim.Optimizer
    buffer : RolloutBuffer
    advantages : torch.Tensor
        Shape (num_steps, num_envs).
    returns : torch.Tensor
        Shape (num_steps, num_envs).
    clip_epsilon, entropy_coeff, value_loss_coeff, max_grad_norm : float
    update_epochs : int
    num_minibatches : int

    Returns
    -------
    stats : dict
        Training statistics for logging.
    """
    # Normalize advantages
    adv_flat = advantages.reshape(-1)
    adv_mean = adv_flat.mean()
    adv_std = adv_flat.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    total_clipfrac = 0.0
    total_approx_kl = 0.0
    n_updates = 0

    early_stopped = False
    for epoch in range(update_epochs):
        epoch_kl = 0.0
        epoch_batches = 0
        for mb in buffer.get_batches(advantages, returns, num_minibatches):
            # Clamp actions for numerical stability (Beta needs [eps, 1-eps])
            if mb["actions"].shape[-1] == 1:
                mb_actions = mb["actions"].clamp(1e-6, 1.0 - 1e-6)
            else:
                mb_actions = mb["actions"]

            _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                mb["obs"], action=mb_actions
            )

            log_ratio = new_log_prob - mb["log_probs"]
            ratio = log_ratio.exp()

            # Approximate KL for early stopping / monitoring
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

            mb_adv = mb["advantages"]

            # Clipped surrogate objective
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clipped)
            new_value = new_value.squeeze(-1)
            v_loss_unclipped = (new_value - mb["returns"]) ** 2
            v_clipped = mb["values"] + torch.clamp(
                new_value - mb["values"], -clip_epsilon, clip_epsilon
            )
            v_loss_clipped = (v_clipped - mb["returns"]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - entropy_coeff * entropy_loss + value_loss_coeff * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy_loss.item()
            total_clipfrac += clipfrac.item()
            total_approx_kl += approx_kl.item()
            n_updates += 1
            epoch_kl += approx_kl.item()
            epoch_batches += 1

        # KL early stopping: if mean KL over this epoch exceeds threshold, stop
        if target_kl is not None and epoch_batches > 0:
            if epoch_kl / epoch_batches > target_kl:
                early_stopped = True
                break

    return {
        "policy_loss": total_pg_loss / n_updates,
        "value_loss": total_v_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "clipfrac": total_clipfrac / n_updates,
        "approx_kl": total_approx_kl / n_updates,
    }
