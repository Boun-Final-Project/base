"""
Phase 2 — Student distillation training.

Usage:
    python -m reinforcement_learning.training.train_distil \\
        --teacher-ckpt runs/teacher_001/checkpoints/teacher_50000000.pt \\
        --output-dir runs/student_001

The student (ActorCriticDualBackbone) is trained from scratch with:
    L = L_PPO(student) + DISTIL_LAMBDA * KL(student || teacher)

The teacher is loaded from checkpoint, frozen, and used as oracle only.
Map canvases are built on-the-fly during the PPO update from stored
agent positions and a per-episode map registry.
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
    from reinforcement_learning.models.actor_critic import ActorCriticDualBackbone, ActorCriticTeacher
    from reinforcement_learning.training.distil_ppo import DistilRolloutBuffer, distil_ppo_update
    from reinforcement_learning.training.ppo import RunningMeanStd, compute_gae
    from reinforcement_learning.training.train import (
        VecEnv, get_curriculum_ranges, get_template_curriculum, plot_training_curves)
else:
    from .. import config as cfg
    from ..envs.gas_source_env import GasSourceEnv
    from ..models.actor_critic import ActorCriticDualBackbone, ActorCriticTeacher
    from .distil_ppo import DistilRolloutBuffer, distil_ppo_update
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
    vec_env = VecEnv(env_fns)
    print(f"VecEnv: serial ({args.num_envs} envs)")

    # --- Frozen teacher ---
    frozen_teacher = ActorCriticTeacher(obs_dim=cfg.STATE_DIM,
                                        use_attention=args.attention).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    frozen_teacher.load_state_dict(ckpt["model_state_dict"])
    frozen_teacher.eval()
    for p in frozen_teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded teacher from {args.teacher_ckpt}")

    # --- Student (fresh) ---
    student   = ActorCriticDualBackbone(obs_dim=cfg.STATE_DIM).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, eps=1e-5)
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")

    start_update = 0
    global_step  = 0
    if args.resume:
        ckpt_s = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt_s["model_state_dict"])
        optimizer.load_state_dict(ckpt_s["optimizer_state_dict"])
        global_step  = ckpt_s["global_step"]
        start_update = ckpt_s["update"]
        print(f"Resumed student from {args.resume} (step {global_step:,})")

    buffer     = DistilRolloutBuffer(
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
        "kl_loss": [],  # distillation KL(student || teacher)
    }

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nTraining for {args.total_timesteps:,} timesteps "
          f"({num_updates} updates, {args.rollout_length * args.num_envs} steps/update)")
    print(f"Envs: {args.num_envs}, Rollout: {args.rollout_length}, "
          f"Minibatches: {args.num_minibatches}, Epochs: {args.update_epochs}, "
          f"DistilLambda: {args.distil_lambda}, DistilFrom: {args.distil_from}")
    print()

    # Save config snapshot (CLI args used for this run)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    obs, _ = vec_env.reset()
    for i, env in enumerate(vec_env.envs):
        buffer.register_map(i, env._map_ds)

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

        # === Rollout (student acts, canvases still collected for later update) ===
        buffer.reset()
        buffer.clear_registry()
        for i, env in enumerate(vec_env.envs):
            buffer.register_map(i, env._map_ds)

        for step in range(args.rollout_length):
            global_step += args.num_envs

            obs_t     = torch.tensor(obs, dtype=torch.float32, device=device)
            agent_pos = np.stack([env._robot_pos for env in vec_env.envs])

            with torch.no_grad():
                action, log_prob, _, value = student.get_action_and_value(obs_t)

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
            )

            for i in range(args.num_envs):
                if dones[i]:
                    buffer.register_map(i, vec_env.envs[i]._map_ds)

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

        # Bootstrap value (student only, no canvas)
        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            next_value = student.get_value(next_obs_t).squeeze(-1)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, args.gamma, args.gae_lambda,
        )

        # === Distillation PPO Update ===
        stats = distil_ppo_update(
            student, frozen_teacher, optimizer, buffer, advantages, returns,
            args.clip_epsilon, args.entropy_coeff, args.value_loss_coeff,
            args.max_grad_norm, args.update_epochs, args.num_minibatches,
            args.distil_lambda, device, target_kl=args.target_kl,
            distil_from_map=(args.distil_from == "on"),
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
        metrics["kl_loss"].append(stats["kl_loss"])

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
                f"DistKL {stats['kl_loss']:.4f} | "
                f"KL {stats['approx_kl']:.4f} | "
                f"Clip {stats['clipfrac']:.3f}"
            )

        # === Checkpoint ===
        if update % args.save_interval == 0 or update == num_updates:
            path = os.path.join(ckpt_dir, f"student_{global_step}.pt")
            torch.save({
                "model_state_dict":     student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update":      update,
            }, path)
            print(f"  Saved: {path}")

    vec_env.close()

    total_time = time.time() - start_time
    print(f"\nPhase 2 complete. {global_step:,} steps in {total_time:.0f}s "
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
    p.add_argument("--teacher-ckpt",    type=str,   required=True)
    p.add_argument("--attention",       action="store_true",
                   help="Teacher checkpoint uses the cross-attention map readout (MapCrossAttn). "
                        "Must match how the teacher was trained.")
    p.add_argument("--total-timesteps", type=int,   default=cfg.TOTAL_TIMESTEPS)
    p.add_argument("--lr",              type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--distil-lambda",   type=float, default=cfg.DISTIL_LAMBDA)
    p.add_argument("--distil-from",     choices=["on", "off"], default="on",
                   help="query the frozen teacher with the real (on) or blank (off) map")
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
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--output-dir",      type=str,   default="runs/student")
    p.add_argument("--log-interval",    type=int,   default=1)
    p.add_argument("--save-interval",   type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
