"""
Evaluate a trained NavActorCritic agent from a checkpoint with epsilon-greedy actions.

Usage
-----
    python3 test_nav_agent.py \
        --checkpoint ../../runs/nav_lidar/checkpoint_00500.pt \
        --episodes 20 --epsilon 0.05 --seed 0

Greedy action = distribution mean (2D Gaussian over (cos theta, sin theta)).
With probability epsilon a uniformly random action is chosen instead.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.nav_env import NavigationEnv
from reinforcement_learning.models.nav_actor_critic import NavActorCritic

WALL_TARGET_DIST = 0.50   # metres — the reward peak distance


def load_agent(checkpoint_path, device):
    """Load NavActorCritic from a checkpoint file.

    Accepts both final.pt (plain state_dict) and checkpoint_*.pt (wrapped dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent = NavActorCritic()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded NavActorCritic from {checkpoint_path}")
        print(f"  global_step={ckpt.get('global_step', '?')}, "
              f"update={ckpt.get('update', '?')}")
    else:
        agent.load_state_dict(ckpt)
        print(f"Loaded NavActorCritic from {checkpoint_path}  (final.pt)")
    agent.to(device).eval()
    return agent


def greedy_action(agent, obs_t):
    """Return the deterministic (mean) action for a given observation tensor."""
    with torch.no_grad():
        features = agent._encode(obs_t)
        dist = agent._actor_dist(features)
        return dist.mean.cpu().numpy()[0]   # (2,): (cos theta, sin theta)


def random_action(rng):
    """Uniform random heading encoded as (cos theta, sin theta)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def run_episode(agent, env, epsilon, rng, device, render=False, obs=None):
    """Run one episode and return a result dict.

    If *obs* is None, env.reset() is called internally to obtain the initial
    observation.  Pass *obs* when the caller has already reset the env (e.g.
    to seed the layout and capture the initial frame before calling here).
    """
    if obs is None:
        obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    success = False
    wall_dist_sum = 0.0

    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if rng.random() < epsilon:
            action = random_action(rng)
        else:
            action = greedy_action(agent, obs_t)

        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        total_reward += reward
        steps += 1

        lidar_obs = obs[:cfg.LIDAR_NUM_RAYS]
        wall_dist_sum += float(lidar_obs.min()) * cfg.LIDAR_MAX_RANGE

        if terminated:
            success = True
            break
        if truncated:
            break

    avg_wall_dist = wall_dist_sum / steps if steps > 0 else 0.0
    return {
        "reward":        total_reward,
        "steps":         steps,
        "success":       success,
        "avg_wall_dist": avg_wall_dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Probability of taking a uniform random action")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Save per-step PNG frames under this directory "
                             "(one subfolder per episode)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto-detect)")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    agent = load_agent(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)

    results = []
    for ep in range(args.episodes):
        viz_dir = (os.path.join(args.viz_dir, f"episode_{ep:03d}")
                   if args.viz_dir else None)
        env = NavigationEnv(viz_output_dir=viz_dir)
        obs, _ = env.reset(seed=args.seed + ep)
        if viz_dir:
            env.render()
        out = run_episode(agent, env, args.epsilon, rng, device,
                          render=(viz_dir is not None), obs=obs)
        results.append(out)

        flag = "OK" if out["success"] else "--"
        print(f"  ep {ep:3d} [{flag}] reward={out['reward']:+8.1f}  "
              f"steps={out['steps']:3d}  wall={out['avg_wall_dist']:.2f}m")

    rewards    = np.array([r["reward"]        for r in results])
    steps      = np.array([r["steps"]         for r in results])
    successes  = np.array([r["success"]       for r in results])
    wall_dists = np.array([r["avg_wall_dist"] for r in results])

    print()
    print(f"Episodes:           {len(results)}")
    print(f"Success rate:       {successes.mean() * 100:.1f}%  "
          f"({int(successes.sum())}/{len(results)})")
    print(f"Reward:             {rewards.mean():+.1f} ± {rewards.std():.1f}")
    print(f"Steps:              {steps.mean():.1f} ± {steps.std():.1f}")
    print(f"Avg wall dist:      {wall_dists.mean():.2f} ± {wall_dists.std():.2f} m"
          f"   (target {WALL_TARGET_DIST:.2f} m)")

    if args.viz_dir:
        print(f"Frames saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
