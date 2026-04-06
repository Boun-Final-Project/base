"""
CleanRL-style PPO implementation for continuous action spaces (Beta distribution).
"""

import numpy as np
import torch
import torch.nn as nn


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
