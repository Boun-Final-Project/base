"""
Evaluate all checkpoints of one or more training runs on fixed test environments.

Loads test_envs.json, scans each run's checkpoints/ directory in step order,
evaluates each checkpoint on all 100 environments in parallel (one process per
checkpoint), and saves per-run results to eval_results/<run_name>.npz.

Usage:
    python3 eval_checkpoints.py --run-dirs ../../runs/run1 ../../runs/run2
    python3 eval_checkpoints.py --run-dirs ../../runs/run1 --workers 8
    python3 eval_checkpoints.py --run-dirs ../../runs/run1 --eval-device cpu --workers 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.gas_source_env import GasSourceEnv
from reinforcement_learning.envs.spatial_obs_wrapper import SpatialObsWrapper
from reinforcement_learning.models.actor_critic import (
    ActorCritic,
    ActorCriticModular,
    ActorCriticDualBackbone,
)
from reinforcement_learning.models.actor_critic_spatial import ActorCriticSpatial

_DEFAULT_TEST_ENVS  = os.path.join(_SCRIPT_DIR, "test_envs.json")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_results")


def load_run_config(run_dir):
    """Read config.json from a run directory. Returns dict (may be empty)."""
    config_path = Path(run_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    print(f"  [warn] config.json not found at {config_path}")
    return {}


def load_agent(checkpoint_path, run_cfg, device):
    """Load agent from checkpoint using the run's saved config values."""
    ckpt_path = Path(checkpoint_path)

    arch      = run_cfg.get("arch", "mlp")
    obs_dim   = run_cfg.get("STATE_DIM", cfg.STATE_DIM)
    lidar_len = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
    gas_len   = (run_cfg.get("GAS_HISTORY_LENGTH", cfg.GAS_HISTORY_LENGTH)
                 * run_cfg.get("GAS_FEATURES_PER_STEP", cfg.GAS_FEATURES_PER_STEP))

    kwargs = dict(obs_dim=obs_dim, gas_len=gas_len, lidar_len=lidar_len)

    if arch == "spatial":
        agent = ActorCriticSpatial()
    elif arch == "dual":
        agent = ActorCriticDualBackbone(**kwargs)
    elif arch == "modular":
        agent = ActorCriticModular(**kwargs)
    else:
        agent = ActorCritic(obs_dim=obs_dim)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.to(device).eval()
    return agent, arch


def greedy_action(agent, arch, obs_t, h=None):
    """Deterministic action: distribution mean for all architectures.

    For spatial arch returns (action, new_h); other archs return action only.
    obs_t is a spatial tensor for spatial arch, a flat tensor otherwise.
    """
    with torch.no_grad():
        if arch == "spatial":
            shared, h_new = agent._encode(obs_t, h)
            dist = agent._actor_dist(shared)
            return dist.mean.cpu().numpy(), h_new   # (1, 2): (cos θ, sin θ)
        elif arch == "dual":
            dist = agent._actor_dist(agent._encode_shared(obs_t))
            return dist.mean.cpu().numpy()          # (1, 2): (cos θ, sin θ)
        else:
            features = (agent._encode(obs_t) if arch == "modular"
                        else agent.backbone(obs_t))
            ab    = agent.actor(features)
            alpha = ab[:, 0:1] + 1.0
            beta  = ab[:, 1:2] + 1.0
            return (alpha / (alpha + beta)).cpu().numpy()  # (1, 1)


def _obs_to_tensors(obs, arch, device):
    """Convert environment observation to tensor(s) for the given arch."""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def run_episode(agent, arch, env_spec, device):
    """Run one greedy episode on a fixed environment spec."""
    base_env = GasSourceEnv(template_id=env_spec["template_id"])
    env      = SpatialObsWrapper(base_env) if arch == "spatial" else base_env
    obs, _   = env.reset(seed=env_spec["seed"])
    total_return = 0.0
    success = False
    h = agent.initial_hidden(1, device) if arch == "spatial" else None
    while True:
        if arch == "spatial":
            action, h = greedy_action(agent, arch, _obs_to_tensors(obs, arch, device), h)
            action = action[0]
        else:
            action = greedy_action(agent, arch, _obs_to_tensors(obs, arch, device))[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        if terminated:
            success = True
            break
        if truncated:
            break
    return total_return, success


def _eval_one_ckpt(args):
    """Worker: evaluate one checkpoint across all test environments.

    Runs in a subprocess — patches cfg locally (safe, subprocess-isolated).
    Returns (ci, step, returns_array, successes_array).
    """
    ckpt_path_str, run_cfg, test_envs, device_str, ci = args
    torch.set_num_threads(1)
    device    = torch.device(device_str)
    ckpt_path = Path(ckpt_path_str)
    step      = int(ckpt_path.stem.split("_")[1])

    orig = {k: getattr(cfg, k) for k in ("LIDAR_NUM_RAYS", "STATE_DIM")}
    try:
        cfg.LIDAR_NUM_RAYS = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
        cfg.STATE_DIM      = run_cfg.get("STATE_DIM",      cfg.STATE_DIM)

        agent, arch = load_agent(ckpt_path, run_cfg, device)

        n_envs    = len(test_envs)
        returns   = np.zeros(n_envs, dtype=np.float32)
        successes = np.zeros(n_envs, dtype=bool)

        for ei, env_spec in enumerate(test_envs):
            returns[ei], successes[ei] = run_episode(agent, arch, env_spec, device)
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)

    return ci, step, returns, successes


def eval_run(run_dir, test_envs, output_dir, eval_device="cpu", n_workers=None):
    run_dir  = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    run_cfg = load_run_config(run_dir)
    print(f"  STATE_DIM={run_cfg.get('STATE_DIM', cfg.STATE_DIM)}  "
          f"LIDAR_NUM_RAYS={run_cfg.get('LIDAR_NUM_RAYS', cfg.LIDAR_NUM_RAYS)}")

    ckpt_files = sorted(
        ckpt_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpt_files:
        print(f"  No checkpoints found in {ckpt_dir}, skipping.")
        return

    n_envs  = len(test_envs)
    n_ckpts = len(ckpt_files)
    print(f"\nRun: {run_dir.name}  |  {n_ckpts} checkpoints × {n_envs} environments")

    steps     = np.zeros(n_ckpts, dtype=np.int64)
    returns   = np.zeros((n_ckpts, n_envs), dtype=np.float32)
    successes = np.zeros((n_ckpts, n_envs), dtype=bool)

    actual_workers = n_workers if n_workers is not None else min(os.cpu_count() or 1, n_ckpts)
    print(f"  Using {actual_workers} worker(s) on device={eval_device}")

    worker_args = [
        (str(ckpt_path), run_cfg, test_envs, eval_device, ci)
        for ci, ckpt_path in enumerate(ckpt_files)
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_eval_one_ckpt, a): a for a in worker_args}
        for fut in as_completed(futures):
            ckpt_path_str = futures[fut][0]
            try:
                ci, step, rets, succs = fut.result()
            except Exception as exc:
                raise RuntimeError(f"Worker failed on {ckpt_path_str}") from exc
            steps[ci]       = step
            returns[ci]     = rets
            successes[ci]   = succs
            completed += 1
            print(f"  [{completed:3d}/{n_ckpts}]  step={step:>10,}  "
                  f"return={rets.mean():7.1f} ± {rets.std():5.1f}  "
                  f"success={succs.mean():.1%}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_dir.name}.npz")
    np.savez(
        out_path,
        steps        = steps,
        returns      = returns,
        successes    = successes,
        template_ids = np.array([e["template_id"] for e in test_envs], dtype=np.int64),
    )
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="One or more run directories containing checkpoints/")
    parser.add_argument("--test-envs", type=str, default=_DEFAULT_TEST_ENVS,
                        help="Path to test_envs.json")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
                        help="Directory for result .npz files")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument(
        "--eval-device", type=str, default="cpu",
        help="Device used inside worker processes for model inference (default: cpu)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: min(cpu_count, n_checkpoints))",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Eval workers device: {args.eval_device}")

    with open(args.test_envs) as f:
        test_envs = json.load(f)["envs"]
    print(f"Loaded {len(test_envs)} test environments from {args.test_envs}")

    for run_dir in args.run_dirs:
        eval_run(run_dir, test_envs, args.output_dir,
                 eval_device=args.eval_device, n_workers=args.workers)


if __name__ == "__main__":
    main()
