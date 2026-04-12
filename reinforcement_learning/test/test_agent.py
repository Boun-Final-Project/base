"""
Evaluate a trained PPO agent from a checkpoint with epsilon-greedy actions.

Usage
-----
    python3 -m reinforcement_learning.test_agent \
        --checkpoint runs/ppo_dual-010/checkpoints/agent_150000000.pt \
        --episodes 20 --epsilon 0.05 --template 4 --seed 0

    # With visualization (one PNG per step under <viz-dir>/episode_XXX/):
    python3 -m reinforcement_learning.test_agent \
        --checkpoint runs/ppo_dual-010/checkpoints/agent_150000000.pt \
        --episodes 3 --epsilon 0.0 --viz-dir runs/ppo_dual-010/eval

The checkpoint's sibling ``config.json`` (one level up from ``checkpoints/``)
is read to determine the architecture (``flat`` / ``modular`` / ``dual``).

Greedy action = distribution mean:
  - Beta(alpha, beta)        ->  alpha / (alpha + beta)        (flat, modular)
  - Normal(mean, std) on R^2 ->  mean                          (dual)

With probability epsilon a uniformly random action is chosen instead.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv
from reinforcement_learning.models.actor_critic import (
    ActorCritic,
    ActorCriticDualBackbone,
    ActorCriticModular,
)


def load_agent(checkpoint_path, device):
    """Load an agent + its arch from a checkpoint path.

    Looks for ``config.json`` two levels up (i.e. the run directory) to read
    the architecture name.
    """
    ckpt_path = Path(checkpoint_path)
    run_dir = ckpt_path.parent.parent  # runs/<run>/checkpoints/agent_X.pt -> runs/<run>
    config_path = run_dir / "config.json"

    arch = "flat"
    if config_path.exists():
        with open(config_path) as f:
            run_cfg = json.load(f)
        arch = run_cfg.get("arch", "flat")
    else:
        print(f"[warn] no config.json at {config_path}, defaulting arch=flat")

    if arch == "dual":
        agent = ActorCriticDualBackbone(obs_dim=cfg.STATE_DIM)
    elif arch == "modular":
        agent = ActorCriticModular(obs_dim=cfg.STATE_DIM)
    else:
        agent = ActorCritic(obs_dim=cfg.STATE_DIM)

    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.to(device).eval()

    print(f"Loaded {arch} agent from {ckpt_path}")
    print(f"  global_step={ckpt.get('global_step', '?')}, "
          f"update={ckpt.get('update', '?')}")
    return agent, arch


def greedy_action(agent, arch, obs_t):
    """Return the deterministic (mean) action for a given observation."""
    with torch.no_grad():
        if arch == "dual":
            encoded = agent._encode_shared(obs_t)
            dist = agent._actor_dist(encoded)
            return dist.mean  # (1, 2) -> (cos, sin)
        else:
            # Beta scalar in [0, 1]
            if arch == "modular":
                features = agent.fusion(agent._encode(obs_t))
            else:
                features = agent.backbone(obs_t)
            ab = agent.actor(features)
            alpha = ab[:, 0:1] + 1.0
            beta = ab[:, 1:2] + 1.0
            return alpha / (alpha + beta)  # Beta mean, shape (1, 1)


def random_action(arch, rng):
    """Uniform random action matching the agent's action shape."""
    if arch == "dual":
        # Random unit vector (cos θ, sin θ)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([[np.cos(theta), np.sin(theta)]], dtype=np.float32)
    return np.array([[rng.uniform(0.0, 1.0)]], dtype=np.float32)


def run_episode(agent, arch, env, epsilon, rng, device, render=False):
    obs, info = env.reset()
    if render:
        env.render()

    total_reward = 0.0
    detections = 0
    collisions = 0
    steps = 0
    success = False

    while True:
        if rng.random() < epsilon:
            action = random_action(arch, rng)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = greedy_action(agent, arch, obs_t).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action[0])
        total_reward += reward
        steps += 1
        detections += int(info.get("gas_reading", 0))
        collisions += int(info.get("collision", False))

        if render:
            env.render()

        if terminated:
            success = True
            break
        if truncated:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "detections": detections,
        "collisions": collisions,
        "final_distance": info["distance_to_source"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to agent_*.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Probability of taking a uniform random action")
    parser.add_argument("--template", type=int, default=-1,
                        help="Map template id (0-5). -1 = random per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="If set, save per-step PNG frames under this directory "
                             "(one subfolder per episode)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    agent, arch = load_agent(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)
    template_id = args.template if args.template >= 0 else None

    results = []
    for ep in range(args.episodes):
        viz_dir = None
        if args.viz_dir is not None:
            viz_dir = os.path.join(args.viz_dir, f"episode_{ep:03d}")

        env = GasSourceEnv(
            seed=args.seed + ep,
            template_id=template_id,
            viz_output_dir=viz_dir,
        )
        out = run_episode(
            agent, arch, env, args.epsilon, rng, device,
            render=(viz_dir is not None),
        )
        results.append(out)

        flag = "OK" if out["success"] else "--"
        print(f"  ep {ep:3d} [{flag}] reward={out['reward']:+8.1f}  "
              f"steps={out['steps']:3d}  dist={out['final_distance']:.2f}m  "
              f"detections={out['detections']:3d}  collisions={out['collisions']:3d}")

    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    successes = np.array([r["success"] for r in results])
    dists = np.array([r["final_distance"] for r in results])

    print()
    print(f"Episodes:        {len(results)}")
    print(f"Success rate:    {successes.mean() * 100:.1f}%  "
          f"({int(successes.sum())}/{len(results)})")
    print(f"Reward:          {rewards.mean():+.2f} ± {rewards.std():.2f}")
    print(f"Steps:           {steps.mean():.1f} ± {steps.std():.1f}")
    print(f"Final distance:  {dists.mean():.2f} ± {dists.std():.2f} m")


if __name__ == "__main__":
    main()
