"""
Run a checkpoint on each GADEN map and plot the robot's trajectory on the
occupancy grid.

For each map: runs ``--episodes-per-map`` greedy episodes, overlays each
trajectory as a line with a viridis colour gradient (early steps cool, late
steps warm), shows the source (red star), the start position (blue circle),
and the final position with a marker that encodes success.

Saves one PNG per map under ``rl_5_channel/test/gaden_trajectories/<run-name>/``.

Usage
-----
    python -m rl_5_channel.test.visualize_gaden_trajectories \\
        --run-dir base/rl_5_channel/runs/ppo_spatial_ent_20260424_164042_job4475 \\
        [--episodes-per-map 1] [--maps 4rooms uleft]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_5_channel import config as cfg
from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.envs.spatial_obs_wrapper import SpatialObsWrapper
from rl_5_channel.models.actor_critic_spatial import ActorCriticSpatial
from rl_5_channel.test.gaden_loader import (
    DEFAULT_MAP_KEYS,
    load_full_map,
)


_DEFAULT_GADEN_ROOT = Path(_SCRIPT_DIR).parents[1] / "gaden_maps"
_DEFAULT_OUT_ROOT   = Path(_SCRIPT_DIR) / "gaden_trajectories"


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(
        (run_dir / "checkpoints").glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No agent_*.pt found in {run_dir}/checkpoints")
    return ckpts[-1]


def load_run_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_agent(checkpoint_path: Path, device) -> ActorCriticSpatial:
    agent = ActorCriticSpatial()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.load_state_dict(ckpt["model_state_dict"])
    return agent.to(device).eval()


def greedy_action(agent, spatial_t, wind_t):
    with torch.no_grad():
        shared = agent._encode(spatial_t, wind_t)
        dist = agent._actor_dist(shared)
        return dist.mean.cpu().numpy()


def rollout(agent, full_map, episode_seed, device):
    """Run one greedy episode, return (success, total_return, trajectory)."""
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
    traj = np.asarray(env._env._trajectory, dtype=np.float64)
    return success, float(total_return), traj


def _draw_gradient_line(ax, traj: np.ndarray, cmap_name: str, alpha: float):
    """Plot a single trajectory as a line whose colour fades along time."""
    if len(traj) < 2:
        return
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.15, 0.95, len(segments)))
    colors[:, 3] = alpha
    lc = LineCollection(segments, colors=colors, linewidths=1.8)
    ax.add_collection(lc)


def visualize_one_map(agent, full_map, key: str, n_eps: int, device,
                      out_path: Path, quiver_stride: int = 10) -> dict:
    grid_arr = full_map["grid"].grid
    res      = full_map["grid"].resolution
    H, W     = grid_arr.shape
    source   = full_map["source_pos"]
    start    = full_map["robot_pos"]
    field    = full_map["wind_field"].field
    mean_spd, mean_dir = full_map["wind_field"].spatial_mean()

    fig, ax = plt.subplots(figsize=(W * res * 0.6 + 2, H * res * 0.6 + 2))
    ax.imshow(grid_arr, origin="lower", cmap="gray_r",
              extent=[0, W * res, 0, H * res])

    # Wind quiver underlay (low alpha, so trajectories pop)
    ys, xs = np.mgrid[0:H:quiver_stride, 0:W:quiver_stride]
    ax.quiver(xs * res + res / 2, ys * res + res / 2,
              field[ys, xs, 0], field[ys, xs, 1],
              color="tab:green", alpha=0.35, scale=20, width=0.003,
              zorder=2)

    # Run the rollouts and overlay each trajectory
    cmaps = ["viridis", "plasma", "cividis", "magma", "inferno"]
    successes = []
    rewards   = []
    lengths   = []
    for ei in range(n_eps):
        seed = 1000 * hash(key) % 100000 + ei
        succ, ret, traj = rollout(agent, full_map, seed, device)
        successes.append(succ)
        rewards.append(ret)
        lengths.append(len(traj))
        cmap = cmaps[ei % len(cmaps)]
        _draw_gradient_line(ax, traj, cmap, alpha=0.9 if n_eps == 1 else 0.7)
        # End marker per episode
        end = traj[-1]
        marker = "P" if succ else "x"
        color  = "limegreen" if succ else "red"
        ax.plot(end[0], end[1], marker=marker, color=color, markersize=12,
                markeredgewidth=2, zorder=5,
                label=f"ep{ei}: {'success' if succ else 'fail'} ({len(traj)} steps, R={ret:+.0f})")

    # Source + start
    ax.plot(source[0], source[1], "r*", markersize=24, zorder=4, label="source")
    ax.plot(start[0],  start[1],  "bo", markersize=12, zorder=4, label="start")

    succ_rate = np.mean(successes)
    ax.set_title(
        f"{key}  ({W}×{H} cells, {W*res:.1f}×{H*res:.1f} m, walls={(grid_arr!=0).mean():.1%})\n"
        f"mean wind {mean_spd:.2f} m/s @ {np.degrees(mean_dir):+.0f}°  |  "
        f"{n_eps} ep · success {succ_rate:.0%} · "
        f"mean steps {np.mean(lengths):.0f} · mean R {np.mean(rewards):+.0f}",
        fontsize=11,
    )
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, W * res); ax.set_ylim(0, H * res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"successes": successes, "rewards": rewards, "lengths": lengths}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir",          type=Path, required=True)
    parser.add_argument("--checkpoint",       type=Path, default=None,
                        help="Specific checkpoint .pt; defaults to latest in run-dir")
    parser.add_argument("--gaden-root",       type=Path, default=_DEFAULT_GADEN_ROOT)
    parser.add_argument("--maps",             nargs="+", default=DEFAULT_MAP_KEYS)
    parser.add_argument("--episodes-per-map", type=int, default=1)
    parser.add_argument("--output-dir",       type=Path, default=None,
                        help="Defaults to test/gaden_trajectories/<run-name>/")
    parser.add_argument("--device",           type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    run_dir = args.run_dir
    ckpt    = args.checkpoint or find_latest_checkpoint(run_dir)
    print(f"Run:        {run_dir.name}")
    print(f"Checkpoint: {ckpt.name}")

    # Apply run-config overrides for cfg.LIDAR_NUM_RAYS / STATE_DIM, mirroring
    # the eval scripts.
    run_cfg = load_run_config(run_dir)
    cfg.LIDAR_NUM_RAYS = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
    cfg.STATE_DIM      = run_cfg.get("STATE_DIM",      cfg.STATE_DIM)

    agent = load_agent(ckpt, device)

    out_root = args.output_dir or (_DEFAULT_OUT_ROOT / run_dir.name)
    print(f"Output:     {out_root}")

    yaml_path = args.gaden_root / "recommended_configs.yaml"
    for key in args.maps:
        full_map = load_full_map(args.gaden_root, yaml_path, key)
        out_path = out_root / f"{key}.png"
        stats = visualize_one_map(agent, full_map, key,
                                  args.episodes_per_map, device, out_path)
        ok = sum(stats["successes"])
        print(f"  {key:18s}  {ok}/{args.episodes_per_map} success  "
              f"mean steps={np.mean(stats['lengths']):.0f}  "
              f"mean R={np.mean(stats['rewards']):+.0f}  -> {out_path}")


if __name__ == "__main__":
    main()
