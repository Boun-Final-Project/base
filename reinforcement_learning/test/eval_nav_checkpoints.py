"""
Evaluate all checkpoints of one or more navigation training runs on fixed test
environments.

Loads nav_test_envs.json, scans each run's root directory for checkpoint files
in step order, runs each checkpoint greedily on all environments, and saves
per-run results to eval_results/<run_name>.npz.

Checkpoint naming convention (in run root, NOT a checkpoints/ subdir):
    checkpoint_00100.pt   keys: model_state_dict, optimizer_state_dict,
                                global_step, update
    checkpoint_00200.pt
    ...
    final.pt              skipped (no step metadata)

Usage:
    python3 eval_nav_checkpoints.py --run-dirs ../../runs/nav_lidar
    python3 eval_nav_checkpoints.py --run-dirs ../../runs/run_a ../../runs/run_b
    python3 eval_nav_checkpoints.py --run-dirs ../../runs/nav_lidar --device cpu
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

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.nav_env import NavigationEnv
from reinforcement_learning.models.nav_actor_critic import NavActorCritic

_DEFAULT_TEST_ENVS  = os.path.join(_SCRIPT_DIR, "nav_test_envs.json")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_results")

_SIZE_CAT_MAP = {"small": 0, "medium": 1, "large": 2, "extra-large": 3}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _discover_checkpoints(run_dir: Path):
    """Return checkpoint paths sorted by integer step in the stem.

    Skips ``final.pt`` and any file whose stem cannot be parsed as
    ``checkpoint_<int>``.
    """
    ckpt_files = []
    for p in run_dir.glob("checkpoint_*.pt"):
        try:
            int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        ckpt_files.append(p)
    return sorted(ckpt_files, key=lambda p: int(p.stem.split("_")[1]))


def _step_from_ckpt(ckpt: dict, path: Path) -> int:
    """Return global step stored in checkpoint, falling back to update * 8192."""
    if "global_step" in ckpt:
        return int(ckpt["global_step"])
    if "update" in ckpt:
        return int(ckpt["update"]) * 8192
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def load_agent(checkpoint_path: Path, device: torch.device):
    """Load a NavActorCritic from checkpoint_path and return (agent, ckpt)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent = NavActorCritic()
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.to(device).eval()
    return agent, ckpt


def greedy_action(agent: NavActorCritic, obs_t: torch.Tensor) -> np.ndarray:
    """Return deterministic action (distribution mean) as a (2,) numpy array."""
    with torch.no_grad():
        features = agent._encode(obs_t)
        dist = agent._actor_dist(features)
        return dist.mean.cpu().numpy()[0]   # (2,): (cos theta, sin theta)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(agent: NavActorCritic, env_spec: dict,
                device: torch.device) -> dict:
    """Run one greedy episode on a fixed environment spec."""
    env = NavigationEnv()
    obs, _ = env.reset(seed=env_spec["seed"])

    total_return = 0.0
    wall_dists = []
    steps = 0
    success = False

    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = greedy_action(agent, obs_t)
        obs, reward, terminated, truncated, info = env.step(action)

        wall_dists.append(
            float(np.min(obs[:cfg.LIDAR_NUM_RAYS]) * cfg.LIDAR_MAX_RANGE)
        )
        total_return += reward
        steps += 1

        if terminated:
            success = True
            break
        if truncated:
            break

    return {
        "return":        total_return,
        "steps":         steps,
        "success":       success,
        "avg_wall_dist": float(np.mean(wall_dists)) if wall_dists else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def eval_run(run_dir: Path, test_envs: list, device: torch.device,
             output_dir: str) -> None:
    """Evaluate all checkpoints in run_dir and save results to output_dir."""
    ckpt_files = _discover_checkpoints(run_dir)
    if not ckpt_files:
        print(f"  No checkpoints found in {run_dir}, skipping.")
        return

    n_ckpts = len(ckpt_files)
    n_envs  = len(test_envs)
    print(f"\nRun: {run_dir.name}  |  {n_ckpts} checkpoints × {n_envs} environments")

    steps          = np.zeros(n_ckpts, dtype=np.int64)
    returns        = np.zeros((n_ckpts, n_envs), dtype=np.float32)
    successes      = np.zeros((n_ckpts, n_envs), dtype=bool)
    ep_lengths     = np.zeros((n_ckpts, n_envs), dtype=np.int32)
    avg_wall_dists = np.zeros((n_ckpts, n_envs), dtype=np.float32)

    size_cats = np.array(
        [_SIZE_CAT_MAP.get(e.get("size_cat", "small"), 0) for e in test_envs],
        dtype=np.int8,
    )

    for ci, ckpt_path in enumerate(ckpt_files):
        agent, ckpt = load_agent(ckpt_path, device)
        step = _step_from_ckpt(ckpt, ckpt_path)
        steps[ci] = step

        for ei, env_spec in enumerate(test_envs):
            result = run_episode(agent, env_spec, device)
            returns[ci, ei]        = result["return"]
            successes[ci, ei]      = result["success"]
            ep_lengths[ci, ei]     = result["steps"]
            avg_wall_dists[ci, ei] = result["avg_wall_dist"]

        print(
            f"  [{ci+1:3d}/{n_ckpts}]  "
            f"step={step:>10,}  "
            f"return={returns[ci].mean():7.1f} ± {returns[ci].std():5.1f}  "
            f"success={successes[ci].mean():5.1%}  "
            f"wall={avg_wall_dists[ci].mean():.2f}m"
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_dir.name}.npz")
    np.savez(
        out_path,
        steps          = steps,
        returns        = returns,
        successes      = successes,
        ep_lengths     = ep_lengths,
        avg_wall_dists = avg_wall_dists,
        size_cats      = size_cats,
    )
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="One or more run root directories containing checkpoint_*.pt files",
    )
    parser.add_argument(
        "--test-envs", type=str, default=_DEFAULT_TEST_ENVS,
        help="Path to nav_test_envs.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help="Directory for result .npz files (default: %(default)s)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    with open(args.test_envs) as f:
        test_data = json.load(f)
    test_envs = test_data["envs"]
    print(f"Loaded {len(test_envs)} test environments from {args.test_envs}")

    for run_dir in args.run_dirs:
        eval_run(Path(run_dir), test_envs, device, args.output_dir)


if __name__ == "__main__":
    main()
