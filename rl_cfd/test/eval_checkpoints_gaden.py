"""
Evaluate spatial-arch checkpoints on the GADEN maps from base/gaden_maps/.

Mirrors eval_checkpoints_spatial.py but swaps the procedurally-generated
test envs for hand-curated GADEN layouts: walls rasterized from the CFD
mesh, source / robot from gaden_maps/recommended_configs.yaml, wind
advection driven by GADEN's spatial wind field. Policy ctx vector still
sees a single (speed, cos, sin, step) — the field's spatial mean.

Usage:
    python -m rl_cfd.test.eval_checkpoints_gaden --run-dirs <run> [--latest-only]
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

from rl_cfd import config as cfg
from rl_cfd.envs.gas_source_env import GasSourceEnv
from rl_cfd.envs.spatial_obs_wrapper import SpatialObsWrapper
from rl_cfd.models.actor_critic_spatial import ActorCriticSpatial
from rl_cfd.test.gaden_loader import (
    DEFAULT_MAP_KEYS,
    load_full_map,
)


_DEFAULT_GADEN_ROOT = Path(_SCRIPT_DIR).parents[1] / "gaden_maps"
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_results_gaden")


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


def run_episode(agent, full_map, episode_seed, device):
    map_data = {k: full_map[k] for k in ("grid", "source_pos", "robot_pos",
                                          "width", "height")}
    env = SpatialObsWrapper(GasSourceEnv())
    (spatial, wind), _ = env.reset(
        seed=episode_seed,
        options={"map_data": map_data, "wind_field": full_map["wind_field"]},
    )
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


def load_maps(gaden_root: Path, map_keys):
    yaml_path = gaden_root / "recommended_configs.yaml"
    print(f"Loading {len(map_keys)} GADEN maps from {gaden_root}")
    full_maps = []
    for key in map_keys:
        fm = load_full_map(gaden_root, yaml_path, key)
        spd, dirn = fm["wind_field"].spatial_mean()
        H, W = fm["grid"].grid.shape
        occ = (fm["grid"].grid != 0).mean()
        print(f"  {key:18s}  {W}×{H}c  walls={occ:5.1%}  "
              f"wind_mean={spd:.2f}m/s @ {np.degrees(dirn):+5.0f}°")
        full_maps.append(fm)
    return full_maps


def eval_run(run_dir, full_maps, map_keys, episodes_per_map, device,
             output_dir, latest_only=False):
    run_dir  = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_cfg  = load_run_config(run_dir)

    ckpt_files = sorted(
        ckpt_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpt_files:
        print(f"  No checkpoints found in {ckpt_dir}, skipping.")
        return
    if latest_only:
        ckpt_files = ckpt_files[-1:]

    n_maps  = len(full_maps)
    n_ckpts = len(ckpt_files)
    n_eps   = episodes_per_map
    print(f"\nRun: {run_dir.name}  |  {n_ckpts} ckpts × {n_maps} maps × {n_eps} eps")

    steps     = np.zeros(n_ckpts, dtype=np.int64)
    returns   = np.zeros((n_ckpts, n_maps, n_eps), dtype=np.float32)
    successes = np.zeros((n_ckpts, n_maps, n_eps), dtype=bool)

    orig = {k: getattr(cfg, k) for k in ("LIDAR_NUM_RAYS", "STATE_DIM")}
    cfg.LIDAR_NUM_RAYS = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
    cfg.STATE_DIM      = run_cfg.get("STATE_DIM",      cfg.STATE_DIM)
    try:
        for ci, ckpt_path in enumerate(ckpt_files):
            step      = int(ckpt_path.stem.split("_")[1])
            steps[ci] = step
            agent = load_agent(ckpt_path, device)

            for mi, fm in enumerate(full_maps):
                for ei in range(n_eps):
                    seed = 1000 * mi + ei
                    ret, succ = run_episode(agent, fm, seed, device)
                    returns[ci, mi, ei]   = ret
                    successes[ci, mi, ei] = succ

            print(f"  [{ci+1:3d}/{n_ckpts}]  step={step:>10,}  "
                  f"return={returns[ci].mean():7.1f} ± {returns[ci].std():5.1f}  "
                  f"success={successes[ci].mean():.1%}")
    finally:
        for k, v in orig.items():
            setattr(cfg, k, v)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_dir.name}.npz")
    np.savez(
        out_path,
        steps     = steps,
        returns   = returns,
        successes = successes,
        map_keys  = np.array(map_keys, dtype=object),
    )
    print(f"  Saved → {out_path}")

    # Per-map breakdown for the last checkpoint
    print(f"\n  Per-map success (final checkpoint, step={steps[-1]:,}):")
    for mi, key in enumerate(map_keys):
        succ_m = successes[-1, mi].mean()
        ret_m  = returns[-1, mi].mean()
        print(f"    {key:18s}  success={succ_m:5.1%}  return={ret_m:7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs",         nargs="+", required=True)
    parser.add_argument("--gaden-root",       type=Path, default=_DEFAULT_GADEN_ROOT)
    parser.add_argument("--maps",             nargs="+", default=DEFAULT_MAP_KEYS)
    parser.add_argument("--episodes-per-map", type=int, default=5)
    parser.add_argument("--output-dir",       type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device",           type=str, default=None)
    parser.add_argument("--latest-only",      action="store_true", default=False,
                        help="Only evaluate the newest checkpoint per run")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    full_maps = load_maps(args.gaden_root, args.maps)

    for run_dir in args.run_dirs:
        eval_run(run_dir, full_maps, args.maps, args.episodes_per_map,
                 device, args.output_dir, latest_only=args.latest_only)


if __name__ == "__main__":
    main()
