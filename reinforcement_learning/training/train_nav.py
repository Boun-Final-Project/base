"""
PPO training for wall-following navigation -- LiDAR validation.

Usage:
    python -m reinforcement_learning.training.train_nav
    python -m reinforcement_learning.training.train_nav --num-envs 8 --total-timesteps 5000000
"""

import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from reinforcement_learning import config as cfg
    from reinforcement_learning.envs.nav_env import NavigationEnv, NAV_OBS_DIM
    from reinforcement_learning.models.nav_actor_critic import NavActorCritic
    from reinforcement_learning.training.ppo import (
        RolloutBuffer, RunningMeanStd, compute_gae, ppo_update
    )
    from reinforcement_learning.training.train import VecEnv, SubprocVecEnv
else:
    from .. import config as cfg
    from ..envs.nav_env import NavigationEnv, NAV_OBS_DIM
    from ..models.nav_actor_critic import NavActorCritic
    from .ppo import RolloutBuffer, RunningMeanStd, compute_gae, ppo_update
    from .train import VecEnv, SubprocVecEnv

NAV_ACTION_DIM = 2


@dataclasses.dataclass
class TrainArgs:
    num_envs: int = 8
    total_timesteps: int = 5_000_000
    rollout_length: int = 1024
    output_dir: str = "runs/nav_lidar"


def _make_nav_env(seed: int, rank: int):
    def _init():
        env = NavigationEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args: TrainArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_fns = [_make_nav_env(seed=42, rank=i) for i in range(args.num_envs)]
    envs = VecEnv(env_fns) if args.num_envs == 1 else SubprocVecEnv(env_fns)
    print(f"VecEnv: {'serial' if args.num_envs == 1 else 'subprocess'} ({args.num_envs} envs)")

    agent = NavActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.LEARNING_RATE, eps=1e-5)
    reward_rms = RunningMeanStd()
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    buffer = RolloutBuffer(
        num_steps=args.rollout_length,
        num_envs=args.num_envs,
        obs_dim=NAV_OBS_DIM,
        action_dim=NAV_ACTION_DIM,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    obs_np, _ = envs.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

    batch_size = args.rollout_length * args.num_envs
    num_updates = max(1, args.total_timesteps // batch_size)
    global_step = 0

    ep_returns, ep_lengths, ep_successes = [], [], []
    ep_return = np.zeros(args.num_envs)
    ep_length = np.zeros(args.num_envs, dtype=int)

    print(f"\nTraining for {args.total_timesteps:,} timesteps "
          f"({num_updates} updates, batch {batch_size:,})\n")

    t_start = time.time()

    for update in range(1, num_updates + 1):
        buffer.reset()

        for _ in range(args.rollout_length):
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs)

            actions_np = action.cpu().numpy()  # (num_envs, 2)
            next_obs_np, rewards_np, dones_np, infos = envs.step(actions_np)
            global_step += args.num_envs

            reward_rms.update(rewards_np)
            rewards_norm = (rewards_np / (reward_rms.std + 1e-8)).astype(np.float32)

            buffer.insert(
                obs,
                action,
                log_prob,
                torch.tensor(rewards_norm, dtype=torch.float32, device=device),
                torch.tensor(dones_np, dtype=torch.float32, device=device),
                value.squeeze(-1),
            )

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

            ep_return += rewards_np
            ep_length += 1
            for i, done in enumerate(dones_np):
                if done:
                    ep_returns.append(float(ep_return[i]))
                    ep_lengths.append(int(ep_length[i]))
                    ep_successes.append(float(infos[i].get("terminated", False)))
                    ep_return[i] = 0.0
                    ep_length[i] = 0

        with torch.no_grad():
            next_value = agent.get_value(obs).squeeze(-1)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, cfg.GAMMA, cfg.GAE_LAMBDA,
        )

        stats = ppo_update(
            agent, optimizer, buffer, advantages, returns,
            cfg.CLIP_EPSILON, cfg.ENTROPY_COEFF, cfg.VALUE_LOSS_COEFF,
            cfg.MAX_GRAD_NORM, cfg.UPDATE_EPOCHS, cfg.NUM_MINIBATCHES,
        )

        if update % 10 == 0 or update == 1:
            n = min(len(ep_returns), 100)
            mean_ret = float(np.mean(ep_returns[-n:])) if n else 0.0
            mean_len = float(np.mean(ep_lengths[-n:])) if n else 0.0
            success_rate = float(np.mean(ep_successes[-n:])) if n else 0.0
            sps = global_step / (time.time() - t_start)
            print(
                f"Update {update:4d}/{num_updates} | "
                f"Step {global_step:>9,} | "
                f"SPS {sps:5.0f} | "
                f"Return {mean_ret:7.1f} | "
                f"EpLen {mean_len:5.0f} | "
                f"Succ {success_rate:4.0%} | "
                f"PgLoss {stats['policy_loss']:7.4f} | "
                f"VLoss {stats['value_loss']:8.2f} | "
                f"Ent {stats['entropy']:6.3f} | "
                f"KL {stats['approx_kl']:.4f} | "
                f"Clip {stats['clipfrac']:.3f}"
            )

        if update % 100 == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint_{update:05d}.pt")
            torch.save({
                "update": update,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    if hasattr(envs, "close"):
        envs.close()

    total_time = time.time() - t_start
    print(f"\nTraining complete. {global_step:,} steps in {total_time:.0f}s "
          f"({global_step / total_time:.0f} SPS)")
    if ep_returns:
        print(f"Final avg return (last 50): {np.mean(ep_returns[-50:]):.1f}")
        print(f"Final avg length (last 50): {np.mean(ep_lengths[-50:]):.0f}")
        print(f"Final success rate (last 50): {np.mean(ep_successes[-50:]):.1%}")

    final = os.path.join(args.output_dir, "final.pt")
    torch.save(agent.state_dict(), final)
    print(f"Model saved to {final}")


def main():
    parser = argparse.ArgumentParser(description="Train wall-following nav agent")
    parser.add_argument("--num-envs",         type=int, default=8)
    parser.add_argument("--total-timesteps",  type=int, default=5_000_000)
    parser.add_argument("--rollout-length",   type=int, default=1024)
    parser.add_argument("--output-dir",       type=str, default="runs/nav_lidar")
    a = parser.parse_args()

    if a.num_envs > 1:
        mp.set_start_method("fork", force=True)

    train(TrainArgs(
        num_envs=a.num_envs,
        total_timesteps=a.total_timesteps,
        rollout_length=a.rollout_length,
        output_dir=a.output_dir,
    ))


if __name__ == "__main__":
    main()
