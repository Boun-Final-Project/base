"""
Evaluate spatial-arch checkpoints on the fixed test_envs.json.

Mirrors eval_checkpoints.py but uses ActorCriticSpatial + SpatialObsWrapper and
greedy action = distribution mean of the Gaussian policy.

Usage:
    python3 eval_checkpoints_spatial.py --run-dirs ../../runs/ppo_spatial_ent_...
    python3 eval_checkpoints_spatial.py --run-dirs <dir> --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_5_channel import config as cfg
from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.envs.spatial_obs_wrapper import SpatialObsWrapper
from rl_5_channel.models.actor_critic_spatial import ActorCriticSpatial

_DEFAULT_TEST_ENVS  = os.path.join(_SCRIPT_DIR, "test_envs.json")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_results")


def load_run_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    print(f"  [warn] config.json not found at {config_path}")
    return {}


def load_agent(checkpoint_path, device):
    agent = ActorCriticSpatial()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.to(device).eval()
    return agent


def greedy_action(agent, spatial_t, wind_t):
    with torch.no_grad():
        shared = agent._encode(spatial_t, wind_t)
        dist = agent._actor_dist(shared)
        return dist.mean.cpu().numpy()  # (1, 2)


def run_episode(agent, env_spec, device):
    env = SpatialObsWrapper(GasSourceEnv(template_id=env_spec["template_id"]))
    (spatial, wind), _ = env.reset(seed=env_spec["seed"])
    total_return = 0.0
    success = False
    while True:
        spatial_t = torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0)
        wind_t    = torch.as_tensor(wind,    dtype=torch.float32, device=device).unsqueeze(0)
        action = greedy_action(agent, spatial_t, wind_t)[0]
        (spatial, wind), reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        if terminated:
            success = True
            break
        if truncated:
            break
    return total_return, success


def eval_run(run_dir, test_envs, device, output_dir, latest_only=False):
    run_dir  = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    run_cfg = load_run_config(run_dir)

    ckpt_files = sorted(
        ckpt_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpt_files:
        print(f"  No checkpoints found in {ckpt_dir}, skipping.")
        return
    if latest_only:
        ckpt_files = ckpt_files[-1:]

    n_envs  = len(test_envs)
    n_ckpts = len(ckpt_files)
    print(f"\nRun: {run_dir.name}  |  {n_ckpts} checkpoints × {n_envs} environments")

    steps     = np.zeros(n_ckpts, dtype=np.int64)
    returns   = np.zeros((n_ckpts, n_envs), dtype=np.float32)
    successes = np.zeros((n_ckpts, n_envs), dtype=bool)

    orig = {k: getattr(cfg, k) for k in ("LIDAR_NUM_RAYS", "STATE_DIM")}
    cfg.LIDAR_NUM_RAYS = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
    cfg.STATE_DIM      = run_cfg.get("STATE_DIM",      cfg.STATE_DIM)
    try:
        for ci, ckpt_path in enumerate(ckpt_files):
            step      = int(ckpt_path.stem.split("_")[1])
            steps[ci] = step

            agent = load_agent(ckpt_path, device)

            for ei, env_spec in enumerate(test_envs):
                ret, succ         = run_episode(agent, env_spec, device)
                returns[ci, ei]   = ret
                successes[ci, ei] = succ

            print(f"  [{ci+1:3d}/{n_ckpts}]  step={step:>10,}  "
                  f"return={returns[ci].mean():7.1f} ± {returns[ci].std():5.1f}  "
                  f"success={successes[ci].mean():.1%}")
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_dir.name}.npz")
    template_ids = np.array([e["template_id"] for e in test_envs], dtype=np.int64)
    np.savez(
        out_path,
        steps        = steps,
        returns      = returns,
        successes    = successes,
        template_ids = template_ids,
    )
    print(f"  Saved → {out_path}")

    # Per-template breakdown for the last checkpoint
    print(f"\n  Per-template success (final checkpoint, step={steps[-1]:,}):")
    template_names = ["empty", "single_wall", "u_shape", "three_walls", "complex_maze", "multi_room"]
    for t in range(6):
        mask = template_ids == t
        if mask.sum() == 0:
            continue
        succ_t = successes[-1, mask].mean()
        ret_t  = returns[-1, mask].mean()
        print(f"    template {t} ({template_names[t]:>12}, n={mask.sum():>2}): "
              f"success={succ_t:.1%}  return={ret_t:7.1f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--test-envs", type=str, default=_DEFAULT_TEST_ENVS)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--latest-only", action="store_true", default=False,
                        help="Only evaluate the newest checkpoint per run")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    with open(args.test_envs) as f:
        test_envs = json.load(f)["envs"]
    print(f"Loaded {len(test_envs)} test environments from {args.test_envs}")

    for run_dir in args.run_dirs:
        eval_run(run_dir, test_envs, device, args.output_dir,
                 latest_only=args.latest_only)


if __name__ == "__main__":
    main()
