"""
Phase 1 — Teacher PPO training.

Usage:
    python -m reinforcement_learning.training.train_teacher \\
        --total-timesteps 50000000 --output-dir runs/teacher_001

The teacher (ActorCriticTeacher) is trained with standard PPO.
The only difference from train.py is:
  - Uses serial VecEnv (required for direct env state access)
  - Collects map canvases at each rollout step via env.get_map_canvas()
  - Uses DistilRolloutBuffer and teacher_ppo_update
"""

import argparse
import json
import os
import time

import numpy as np
import torch

# Support both direct execution and module execution
if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from reinforcement_learning import config as cfg
    from reinforcement_learning.envs.gas_source_env import GasSourceEnv
    from reinforcement_learning.models.actor_critic import ActorCriticTeacher
    from reinforcement_learning.training.distil_ppo import DistilRolloutBuffer, teacher_ppo_update
    from reinforcement_learning.training.ppo import RunningMeanStd, compute_gae
    from reinforcement_learning.training.train import (
        VecEnv, get_curriculum_ranges, get_template_curriculum, plot_training_curves)
else:
    from .. import config as cfg
    from ..envs.gas_source_env import GasSourceEnv
    from ..models.actor_critic import ActorCriticTeacher
    from .distil_ppo import DistilRolloutBuffer, teacher_ppo_update
    from .ppo import RunningMeanStd, compute_gae
    from .train import (
        VecEnv, get_curriculum_ranges, get_template_curriculum, plot_training_curves)


def make_env(seed, rank, template_id=None):
    def _init():
        env = GasSourceEnv(template_id=template_id)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    template_id = args.template if args.template >= 0 else None
    env_fns = [make_env(args.seed, i, template_id=template_id)
               for i in range(args.num_envs)]
    # Serial VecEnv required: env state (_robot_pos, _map_ds) accessed directly.
    vec_env = VecEnv(env_fns)
    print(f"VecEnv: serial ({args.num_envs} envs)")

    teacher   = ActorCriticTeacher(obs_dim=cfg.STATE_DIM).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr, eps=1e-5)
    print(f"Parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    start_update = 0
    global_step  = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        teacher.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step  = ckpt["global_step"]
        start_update = ckpt["update"]
        print(f"Resumed from {args.resume} (step {global_step:,})")

    buffer = DistilRolloutBuffer(
        args.rollout_length, args.num_envs, cfg.STATE_DIM, 2, device
    )
    reward_rms = RunningMeanStd()

    num_updates = args.total_timesteps // (args.rollout_length * args.num_envs)
    ep_return_running = np.zeros(args.num_envs)
    ep_length_running = np.zeros(args.num_envs, dtype=int)
    episode_returns, episode_lengths, episode_successes = [], [], []

    # Per-update metrics for plotting
    metrics = {
        "steps": [],
        "mean_return": [],
        "mean_length": [],
        "success_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
        "reward_per_step": [],
        "sps": [],
    }

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nTraining for {args.total_timesteps:,} timesteps "
          f"({num_updates} updates, {args.rollout_length * args.num_envs} steps/update)")
    print(f"Envs: {args.num_envs}, Rollout: {args.rollout_length}, "
          f"Minibatches: {args.num_minibatches}, Epochs: {args.update_epochs}")
    print(f"Map dropout p: {args.map_dropout}")
    print()

    # Save config snapshot (CLI args used for this run)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    obs, _ = vec_env.reset()
    # Register initial maps for all envs.
    for i, env in enumerate(vec_env.envs):
        buffer.register_map(i, env._map_ds)

    # Per-episode map dropout: a fixed fraction of episodes are trained blind
    # (blank canvas) so the teacher stays competent from gas/lidar alone.
    BLANK = np.zeros((2, cfg.MAP_CANVAS_H, cfg.MAP_CANVAS_W), dtype=np.float32)
    map_active = np.random.rand(args.num_envs) > args.map_dropout  # True = map visible

    start_time = time.time()
    prev_max_template = -1

    for update in range(start_update + 1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            for pg in optimizer.param_groups:
                pg["lr"] = max(1e-6, frac * args.lr)

        if args.curriculum:
            progress = global_step / args.total_timesteps
            w_range, h_range = get_curriculum_ranges(progress)
            max_template, tmpl_weights = get_template_curriculum(progress)
            vec_env.set_curriculum(w_range, h_range, max_template, tmpl_weights)
            if max_template != prev_max_template:
                print(f"[Curriculum] Step {global_step:,} — max_template → {max_template}")
                prev_max_template = max_template

        # === Rollout ===
        buffer.reset()
        buffer.clear_registry()
        for i, env in enumerate(vec_env.envs):
            buffer.register_map(i, env._map_ds)

        for step in range(args.rollout_length):
            global_step += args.num_envs

            # Collect canvases BEFORE the step (matches obs_t). Blind episodes
            # get a blank canvas so the teacher cannot use the map this episode.
            canvases_np = np.stack([
                env.get_map_canvas() if map_active[i] else BLANK
                for i, env in enumerate(vec_env.envs)
            ])                                                   # (N, 2, H, W)
            canvases_t = torch.tensor(canvases_np, dtype=torch.float32, device=device)
            obs_t      = torch.tensor(obs, dtype=torch.float32, device=device)
            agent_pos  = np.stack([env._robot_pos for env in vec_env.envs])

            with torch.no_grad():
                action, log_prob, _, value = teacher.get_action_and_value(
                    obs_t, canvases_t
                )

            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = vec_env.step(action_np)

            reward_rms.update(rewards)
            normalized_rewards = rewards / reward_rms.std

            buffer.insert(
                obs_t, action, log_prob,
                torch.tensor(normalized_rewards, device=device),
                torch.tensor(dones, device=device),
                value.squeeze(-1),
                agent_pos=agent_pos,
                map_dropped=~map_active,
            )

            # Register new maps for envs whose episode ended (auto-reset happened)
            # and redraw the per-episode dropout mask for the new episode.
            for i in range(args.num_envs):
                if dones[i]:
                    buffer.register_map(i, vec_env.envs[i]._map_ds)
                    map_active[i] = np.random.rand() > args.map_dropout

            ep_return_running += rewards
            ep_length_running += 1
            for i in range(args.num_envs):
                if dones[i]:
                    episode_returns.append(ep_return_running[i])
                    episode_lengths.append(ep_length_running[i])
                    episode_successes.append(infos[i].get("terminated", False))
                    ep_return_running[i] = 0.0
                    ep_length_running[i] = 0

            obs = next_obs

        # Bootstrap value (honour the active dropout mask)
        with torch.no_grad():
            canvases_np = np.stack([
                env.get_map_canvas() if map_active[i] else BLANK
                for i, env in enumerate(vec_env.envs)
            ])
            canvases_t  = torch.tensor(canvases_np, dtype=torch.float32, device=device)
            next_obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)
            next_value  = teacher.get_value(next_obs_t, canvases_t).squeeze(-1)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, args.gamma, args.gae_lambda,
        )

        # === PPO Update ===
        stats = teacher_ppo_update(
            teacher, optimizer, buffer, advantages, returns,
            args.clip_epsilon, args.entropy_coeff, args.value_loss_coeff,
            args.max_grad_norm, args.update_epochs, args.num_minibatches,
            device, target_kl=args.target_kl,
        )

        # === Logging ===
        elapsed = time.time() - start_time
        sps     = global_step / elapsed
        n_recent = 50
        if episode_returns:
            mean_ret     = np.mean(episode_returns[-n_recent:])
            mean_len     = np.mean(episode_lengths[-n_recent:])
            success_rate = np.mean(episode_successes[-n_recent:])
        else:
            mean_ret = mean_len = success_rate = 0.0

        # Record metrics every update
        metrics["steps"].append(global_step)
        metrics["mean_return"].append(mean_ret)
        metrics["mean_length"].append(mean_len)
        metrics["success_rate"].append(success_rate)
        metrics["policy_loss"].append(stats["policy_loss"])
        metrics["value_loss"].append(stats["value_loss"])
        metrics["entropy"].append(stats["entropy"])
        metrics["approx_kl"].append(stats["approx_kl"])
        metrics["clipfrac"].append(stats["clipfrac"])
        metrics["reward_per_step"].append(mean_ret / max(mean_len, 1))
        metrics["sps"].append(sps)

        if update % args.log_interval == 0 or update == 1:
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

        # === Checkpoint ===
        if update % args.save_interval == 0 or update == num_updates:
            path = os.path.join(ckpt_dir, f"teacher_{global_step}.pt")
            torch.save({
                "model_state_dict":     teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update":      update,
            }, path)
            print(f"  Saved: {path}")

    vec_env.close()

    total_time = time.time() - start_time
    print(f"\nPhase 1 complete. {global_step:,} steps in {total_time:.0f}s "
          f"({global_step / total_time:.0f} SPS)")
    if episode_returns:
        print(f"Final avg return (last 50): {np.mean(episode_returns[-50:]):.1f}")
        print(f"Final avg length (last 50): {np.mean(episode_lengths[-50:]):.0f}")
        print(f"Final success rate (last 50): "
              f"{np.mean(episode_successes[-50:]):.0%}")

    # Save metrics and plot
    metrics_path = os.path.join(args.output_dir, "metrics.npz")
    np.savez(metrics_path, **{k: np.array(v) for k, v in metrics.items()})
    print(f"Metrics saved to {metrics_path}")

    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(metrics, plot_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int,   default=cfg.TOTAL_TIMESTEPS)
    p.add_argument("--lr",              type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--num-envs",        type=int,   default=cfg.NUM_ENVS)
    p.add_argument("--rollout-length",  type=int,   default=cfg.ROLLOUT_LENGTH)
    p.add_argument("--num-minibatches", type=int,   default=cfg.NUM_MINIBATCHES)
    p.add_argument("--update-epochs",   type=int,   default=cfg.UPDATE_EPOCHS)
    p.add_argument("--clip-epsilon",    type=float, default=cfg.CLIP_EPSILON)
    p.add_argument("--entropy-coeff",   type=float, default=cfg.ENTROPY_COEFF)
    p.add_argument("--value-loss-coeff",type=float, default=cfg.VALUE_LOSS_COEFF)
    p.add_argument("--max-grad-norm",   type=float, default=cfg.MAX_GRAD_NORM)
    p.add_argument("--gamma",           type=float, default=cfg.GAMMA)
    p.add_argument("--gae-lambda",      type=float, default=cfg.GAE_LAMBDA)
    p.add_argument("--target-kl",       type=float, default=None)
    p.add_argument("--seed",            type=int,   default=1)
    p.add_argument("--template",        type=int,   default=-1)
    p.add_argument("--curriculum",      action="store_true")
    p.add_argument("--anneal-lr",       action="store_true")
    p.add_argument("--map-dropout",     type=float, default=cfg.MAP_DROPOUT_P)
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--output-dir",      type=str,   default="runs/teacher")
    p.add_argument("--log-interval",    type=int,   default=1)
    p.add_argument("--save-interval",   type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
