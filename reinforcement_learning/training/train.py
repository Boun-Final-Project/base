"""
PPO training entry point for gas source localization.

Usage:
    python training/train.py --template 0 --total-timesteps 2000000
    python -m reinforcement_learning.training.train --template 0
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

# Support both direct execution and module execution
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from reinforcement_learning import config as cfg
    from reinforcement_learning.envs.gas_source_env import GasSourceEnv
    from reinforcement_learning.envs.spatial_obs_wrapper import SpatialObsWrapper
    from reinforcement_learning.models.actor_critic import ActorCritic, ActorCriticModular, ActorCriticDualBackbone
    from reinforcement_learning.models.actor_critic_spatial import ActorCriticSpatial
    from reinforcement_learning.training.ppo import (RolloutBuffer, SpatialRolloutBuffer,
                                                     RunningMeanStd, compute_gae,
                                                     ppo_update, spatial_ppo_update)
else:
    from .. import config as cfg
    from ..envs.gas_source_env import GasSourceEnv
    from ..envs.spatial_obs_wrapper import SpatialObsWrapper
    from ..models.actor_critic import ActorCritic, ActorCriticModular, ActorCriticDualBackbone
    from ..models.actor_critic_spatial import ActorCriticSpatial
    from .ppo import (RolloutBuffer, SpatialRolloutBuffer,
                      RunningMeanStd, compute_gae,
                      ppo_update, spatial_ppo_update)


def make_spatial_env(seed, rank, template_id=None):
    """Create a thunk that returns a seeded SpatialObsWrapper(GasSourceEnv)."""
    def _init():
        env = SpatialObsWrapper(GasSourceEnv(template_id=template_id))
        env.reset(seed=seed + rank)
        return env
    return _init


def make_env(seed, rank, template_id=None):
    """Create a thunk that returns a seeded GasSourceEnv."""
    def _init():
        env = GasSourceEnv(template_id=template_id)
        env.reset(seed=seed + rank)
        return env
    return _init


class VecEnv:
    """Simple synchronous vectorized environment wrapper."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def set_curriculum(self, w_range, h_range, max_template):
        for env in self.envs:
            env.set_room_size_range(w_range, h_range)
            env.set_max_template(max_template)

    def step(self, actions):
        """Step all envs. Auto-resets on termination/truncation.

        Parameters
        ----------
        actions : np.ndarray
            Shape (num_envs, 1).

        Returns
        -------
        obs, rewards, dones, infos
        """
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated

            if done:
                info["terminal_observation"] = obs
                info["terminated"] = terminated  # source found
                info["truncated"] = truncated     # timeout
                obs, _ = env.reset()

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(float(done))
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            info_list,
        )


def _worker(child_conn, parent_conn, env_fn):
    """Subprocess entry point: owns one env, serves step/reset/close commands."""
    parent_conn.close()
    env = env_fn()
    try:
        while True:
            cmd, data = child_conn.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    info["terminal_observation"] = obs
                    info["terminated"] = terminated
                    info["truncated"] = truncated
                    obs, _ = env.reset()
                child_conn.send((obs, np.float32(reward), np.float32(done), info))
            elif cmd == "reset":
                obs, info = env.reset()
                child_conn.send((obs, info))
            elif cmd == "set_curriculum":
                w_range, h_range, max_template = data
                env.set_room_size_range(w_range, h_range)
                env.set_max_template(max_template)
                child_conn.send(None)
            elif cmd == "close":
                break
    finally:
        child_conn.close()


class SubprocVecEnv:
    """Vectorized environment that runs each env in its own subprocess.

    All envs step in parallel across CPU cores.  Communication is via
    ``multiprocessing.Pipe`` using the ``fork`` start method (Linux).
    """

    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        ctx = mp.get_context("fork")

        # Pipe() returns (parent_conn, child_conn)
        pairs = [ctx.Pipe(duplex=True) for _ in range(self.num_envs)]
        self._parents  = [p for p, _ in pairs]
        child_conns    = [c for _, c in pairs]

        self._procs = []
        for child_conn, parent_conn, fn in zip(child_conns, self._parents, env_fns):
            p = ctx.Process(
                target=_worker,
                args=(child_conn, parent_conn, fn),
                daemon=True,
            )
            p.start()
            child_conn.close()   # parent doesn't use the child end
            self._procs.append(p)

    def step(self, actions):
        for conn, action in zip(self._parents, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._parents]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=np.float32),
            list(infos),
        )

    def reset(self):
        for conn in self._parents:
            conn.send(("reset", None))
        results  = [conn.recv() for conn in self._parents]
        obs_list, info_list = zip(*results)
        return np.stack(obs_list), list(info_list)

    def set_curriculum(self, w_range, h_range, max_template):
        for conn in self._parents:
            conn.send(("set_curriculum", (w_range, h_range, max_template)))
        for conn in self._parents:
            conn.recv()

    def close(self):
        for conn in self._parents:
            try:
                conn.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _stack_spatial(obs_list):
    return (np.stack([o[0] for o in obs_list]),   # (N, 4, 98, 98)
            np.stack([o[1] for o in obs_list]))    # (N, 2)


class SpatialVecEnv(VecEnv):
    """Serial VecEnv for environments returning (spatial, wind) tuple observations."""

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return _stack_spatial(obs_list), info_list

    def step(self, actions):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            if done:
                info["terminal_observation"] = obs
                info["terminated"] = terminated
                info["truncated"]  = truncated
                obs, _ = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(float(done))
            info_list.append(info)
        return (
            _stack_spatial(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list,   dtype=np.float32),
            info_list,
        )


class SpatialSubprocVecEnv(SubprocVecEnv):
    """Subprocess VecEnv for environments returning (spatial, wind) tuple observations."""

    def reset(self):
        for conn in self._parents:
            conn.send(("reset", None))
        results = [conn.recv() for conn in self._parents]
        obs_tuples, info_list = zip(*results)
        return _stack_spatial(obs_tuples), list(info_list)

    def step(self, actions):
        for conn, action in zip(self._parents, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._parents]
        obs_tuples, rewards, dones, infos = zip(*results)
        return (
            _stack_spatial(obs_tuples),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=np.float32),
            list(infos),
        )


def get_curriculum_ranges(progress):
    """Compute room size ranges based on training progress (0→1).

    Linearly interpolates from small rooms to full range over
    CURRICULUM_FRACTION of training, then stays at full range.
    """
    t = min(progress / cfg.CURRICULUM_FRACTION, 1.0)
    w_lo = cfg.CURRICULUM_WIDTH_START[0] + t * (cfg.ROOM_WIDTH_RANGE[0] - cfg.CURRICULUM_WIDTH_START[0])
    w_hi = cfg.CURRICULUM_WIDTH_START[1] + t * (cfg.ROOM_WIDTH_RANGE[1] - cfg.CURRICULUM_WIDTH_START[1])
    h_lo = cfg.CURRICULUM_HEIGHT_START[0] + t * (cfg.ROOM_HEIGHT_RANGE[0] - cfg.CURRICULUM_HEIGHT_START[0])
    h_hi = cfg.CURRICULUM_HEIGHT_START[1] + t * (cfg.ROOM_HEIGHT_RANGE[1] - cfg.CURRICULUM_HEIGHT_START[1])
    return (w_lo, w_hi), (h_lo, h_hi)


def get_template_curriculum(progress):
    """Return max template index based on training progress (0→1).

    Uses TEMPLATE_CURRICULUM_STAGES from config: list of (threshold, max_id).
    """
    max_id = cfg.TEMPLATE_CURRICULUM_STAGES[0][1]
    for threshold, tid in cfg.TEMPLATE_CURRICULUM_STAGES:
        if progress >= threshold:
            max_id = tid
    return max_id


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Vectorized environments
    is_spatial  = (args.arch == "spatial")
    template_id = args.template if args.template >= 0 else None
    if is_spatial:
        env_fns = [make_spatial_env(args.seed, i, template_id=template_id) for i in range(args.num_envs)]
    else:
        env_fns = [make_env(args.seed, i, template_id=template_id) for i in range(args.num_envs)]
    if args.serial:
        vec_env = SpatialVecEnv(env_fns) if is_spatial else VecEnv(env_fns)
        print(f"VecEnv: serial ({args.num_envs} envs)")
    else:
        vec_env = SpatialSubprocVecEnv(env_fns) if is_spatial else SubprocVecEnv(env_fns)
        print(f"VecEnv: subprocess ({args.num_envs} envs, {mp.cpu_count()} CPU cores)")
    obs_dim    = cfg.STATE_DIM
    action_dim = 2 if args.arch in ("dual", "spatial") else 1

    # Agent and optimizer
    if args.arch == "spatial":
        agent = ActorCriticSpatial().to(device)
    elif args.arch == "dual":
        agent = ActorCriticDualBackbone(obs_dim=obs_dim).to(device)
    elif args.arch == "modular":
        agent = ActorCriticModular(obs_dim=obs_dim).to(device)
    else:
        agent = ActorCritic(obs_dim=obs_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Resume from checkpoint
    start_update = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Optionally reset learning rate to initial value
        if args.reset_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            print(f"  Learning rate reset to {args.lr}")
        global_step = ckpt["global_step"]
        start_update = ckpt["update"]
        print(f"Resumed from {args.resume} (step {global_step:,}, update {start_update})")

    # Rollout buffer
    if is_spatial:
        buffer = SpatialRolloutBuffer(args.rollout_length, args.num_envs, action_dim, device)
    else:
        buffer = RolloutBuffer(args.rollout_length, args.num_envs, obs_dim, action_dim, device)

    # Reward normalization (divide by running std)
    reward_rms = RunningMeanStd()

    # Tracking
    num_updates = args.total_timesteps // (args.rollout_length * args.num_envs)
    episode_returns = []
    episode_lengths = []
    episode_successes = []  # True if source found, False if timeout
    ep_return_running = np.zeros(args.num_envs)
    ep_length_running = np.zeros(args.num_envs, dtype=int)

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

    # Checkpoint directory
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nTraining for {args.total_timesteps:,} timesteps "
          f"({num_updates} updates, {args.rollout_length * args.num_envs} steps/update)")
    print(f"Envs: {args.num_envs}, Rollout: {args.rollout_length}, "
          f"Minibatches: {args.num_minibatches}, Epochs: {args.update_epochs}")
    if args.curriculum:
        print(f"Curriculum: rooms {cfg.CURRICULUM_WIDTH_START}x{cfg.CURRICULUM_HEIGHT_START} "
              f"→ {cfg.ROOM_WIDTH_RANGE}x{cfg.ROOM_HEIGHT_RANGE} "
              f"over {cfg.CURRICULUM_FRACTION:.0%} of training")
        print(f"Curriculum: templates unlock at "
              + ", ".join(f"{int(t*100)}%→0-{i}" for t, i in cfg.TEMPLATE_CURRICULUM_STAGES))
    print()

    # Save config snapshot: cfg constants with CLI overrides applied
    # Map CLI arg names to their corresponding cfg constant names
    cli_to_cfg = {
        "total_timesteps": "TOTAL_TIMESTEPS", "lr": "LEARNING_RATE",
        "gamma": "GAMMA", "gae_lambda": "GAE_LAMBDA",
        "clip_epsilon": "CLIP_EPSILON", "entropy_coeff": "ENTROPY_COEFF",
        "value_loss_coeff": "VALUE_LOSS_COEFF", "max_grad_norm": "MAX_GRAD_NORM",
        "num_envs": "NUM_ENVS", "rollout_length": "ROLLOUT_LENGTH",
        "num_minibatches": "NUM_MINIBATCHES", "update_epochs": "UPDATE_EPOCHS",
    }
    config_snapshot = {}
    for k in sorted(dir(cfg)):
        if k.isupper():
            v = getattr(cfg, k)
            if isinstance(v, (int, float, str, bool, list, tuple)):
                config_snapshot[k] = v
    # Override with actual CLI values used
    args_dict = vars(args)
    for cli_name, cfg_name in cli_to_cfg.items():
        if cli_name in args_dict:
            config_snapshot[cfg_name] = args_dict[cli_name]
    # Add CLI-only args (no cfg equivalent)
    for k in ["seed", "template", "arch", "curriculum", "anneal_lr", "anneal_start", "target_kl", "output_dir"]:
        if k in args_dict:
            config_snapshot[k] = args_dict[k]
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)
    print(f"Config saved to {config_path}")

    obs, _ = vec_env.reset()
    start_time = time.time()

    for update in range(start_update + 1, num_updates + 1):
        # Learning rate annealing (only after anneal_start fraction of training)
        if args.anneal_lr:
            progress = (update - 1) / num_updates
            if progress < args.anneal_start:
                lr = args.lr
            else:
                frac = 1.0 - (progress - args.anneal_start) / (1.0 - args.anneal_start)
                lr = max(args.min_lr, frac * args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Curriculum: update room size ranges and template availability
        if args.curriculum:
            progress = global_step / args.total_timesteps
            w_range, h_range = get_curriculum_ranges(progress)
            max_template = get_template_curriculum(progress)
            vec_env.set_curriculum(w_range, h_range, max_template)

        # === Rollout ===
        buffer.reset()
        for step in range(args.rollout_length):
            global_step += args.num_envs

            with torch.no_grad():
                if is_spatial:
                    spatial_t = torch.tensor(obs[0], dtype=torch.float32, device=device)
                    wind_t    = torch.tensor(obs[1], dtype=torch.float32, device=device)
                    action, log_prob, _, value = agent.get_action_and_value(spatial_t, wind_t)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    action, log_prob, _, value = agent.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = vec_env.step(action_np)

            reward_rms.update(rewards)
            normalized_rewards = rewards / reward_rms.std

            if is_spatial:
                buffer.insert(
                    spatial_t, wind_t, action, log_prob,
                    torch.tensor(normalized_rewards, device=device),
                    torch.tensor(dones, device=device),
                    value.squeeze(-1),
                )
            else:
                buffer.insert(
                    obs_t, action, log_prob,
                    torch.tensor(normalized_rewards, device=device),
                    torch.tensor(dones, device=device),
                    value.squeeze(-1),
                )

            # Track episode stats
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

        # Bootstrap value for GAE
        with torch.no_grad():
            if is_spatial:
                spatial_t = torch.tensor(obs[0], dtype=torch.float32, device=device)
                wind_t    = torch.tensor(obs[1], dtype=torch.float32, device=device)
                next_value = agent.get_value(spatial_t, wind_t).squeeze(-1)
            else:
                next_obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                next_value = agent.get_value(next_obs_t).squeeze(-1)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, args.gamma, args.gae_lambda,
        )

        # === PPO Update ===
        _update_fn = spatial_ppo_update if is_spatial else ppo_update
        stats = _update_fn(
            agent, optimizer, buffer, advantages, returns,
            args.clip_epsilon, args.entropy_coeff, args.value_loss_coeff,
            args.max_grad_norm, args.update_epochs, args.num_minibatches,
            target_kl=args.target_kl,
        )

        # === Logging ===
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        n_recent = 50
        if episode_returns:
            recent_ret = episode_returns[-min(n_recent, len(episode_returns)):]
            recent_len = episode_lengths[-min(n_recent, len(episode_lengths)):]
            recent_suc = episode_successes[-min(n_recent, len(episode_successes)):]
            mean_ret = np.mean(recent_ret)
            mean_len = np.mean(recent_len)
            success_rate = np.mean(recent_suc)
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
            path = os.path.join(ckpt_dir, f"agent_{global_step}.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
            }, path)
            print(f"  Saved checkpoint: {path}")

    vec_env.close()

    total_time = time.time() - start_time
    print(f"\nTraining complete. {global_step:,} steps in {total_time:.0f}s "
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


def plot_training_curves(metrics, save_path):
    """Plot training metrics and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array(metrics["steps"])

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    panels = [
        ("mean_return",    "Episode Return",     None),
        ("mean_length",    "Episode Length",      None),
        ("success_rate",   "Success Rate",        (0, 1)),
        ("reward_per_step","Reward / Step",       None),
        ("policy_loss",    "Policy Loss",         None),
        ("value_loss",     "Value Loss",          None),
        ("entropy",        "Entropy",             None),
        ("approx_kl",      "Approx KL",           None),
        ("clipfrac",       "Clip Fraction",       (0, 1)),
    ]

    for ax, (key, title, ylim) in zip(axes.flatten(), panels):
        y = np.array(metrics[key])
        ax.plot(steps, y, linewidth=0.8, alpha=0.4)
        # Smoothed line (rolling mean, window=max(1, len/50))
        w = max(1, len(y) // 50)
        if w > 1 and len(y) > w:
            smooth = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(steps[w - 1:], smooth, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for gas source localization")

    # Environment
    parser.add_argument("--num-envs", type=int, default=cfg.NUM_ENVS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--template", type=int, default=-1,
                        help="Map template ID (0-5). -1 = random (default)")
    parser.add_argument("--serial", action="store_true", default=False,
                        help="Use serial VecEnv instead of subprocess (for debugging)")
    parser.add_argument("--arch", type=str, default="mlp",
                        choices=["mlp", "modular", "dual", "spatial"],
                        help="Network: 'mlp' (flat), 'modular' (GRU+Conv shared fusion), "
                             "'dual' (GRU+Conv separate actor/critic), "
                             "'spatial' (dual-stream CNN + FiLM, ego-centric map obs)")

    # PPO
    parser.add_argument("--total-timesteps", type=int, default=cfg.TOTAL_TIMESTEPS)
    parser.add_argument("--rollout-length", type=int, default=cfg.ROLLOUT_LENGTH)
    parser.add_argument("--num-minibatches", type=int, default=cfg.NUM_MINIBATCHES)
    parser.add_argument("--update-epochs", type=int, default=cfg.UPDATE_EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=cfg.GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=cfg.GAE_LAMBDA)
    parser.add_argument("--clip-epsilon", type=float, default=cfg.CLIP_EPSILON)
    parser.add_argument("--entropy-coeff", type=float, default=cfg.ENTROPY_COEFF)
    parser.add_argument("--value-loss-coeff", type=float, default=cfg.VALUE_LOSS_COEFF)
    parser.add_argument("--max-grad-norm", type=float, default=cfg.MAX_GRAD_NORM)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anneal-start", type=float, default=0.5,
                        help="Fraction of training before LR annealing begins (default: 0.5)")
    parser.add_argument("--min-lr", type=float, default=1e-4,
                        help="Minimum learning rate floor during annealing (default: 1e-4)")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="KL early stopping threshold (e.g. 0.03). None = disabled")
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="Enable room size curriculum (small → large)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--reset-lr", action="store_true", default=False,
                        help="Reset learning rate to initial value when resuming")

    # Logging & output
    parser.add_argument("--output-dir", type=str, default="runs/ppo_gsl")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
