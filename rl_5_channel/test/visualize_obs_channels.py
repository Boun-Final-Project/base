"""
Visualize agent input channels + compare OLD vs NEW is_wall.

Runs a short episode in SpatialObsWrapper, takes snapshots at a few steps,
and saves per-channel + side-by-side wall comparison images.

OLD is_wall simulates the previous behavior (center-only GT lookup).
NEW is_wall uses the fixed dense-wall precomputation (any-subcell).

Usage:
    python3 visualize_obs_channels.py --template 4 --steps 60 --seed 42
    python3 visualize_obs_channels.py --all-templates
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_5_channel import config as cfg
from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.envs.spatial_obs_wrapper import SpatialObsWrapper

_OUT_DIR = os.path.join(_SCRIPT_DIR, "obs_viz")

TEMPLATE_NAMES = ["empty", "single_wall", "u_shape", "three_walls",
                  "complex_maze", "multi_room"]

CHANNELS = ["is_known", "is_wall", "gas", "recency", "det_count"]


def compute_old_is_wall(env_grid, map_h_cells, map_w_cells, cell_res, true_res):
    """Replicate the OLD center-only is_wall computation for comparison."""
    rows = np.arange(map_h_cells)
    cols = np.arange(map_w_cells)
    tx = (cols + 0.5) * cell_res
    ty = (rows + 0.5) * cell_res
    WX, WY = np.meshgrid(tx, ty)
    cx = np.floor(WX / true_res).astype(np.int32)
    cy = np.floor(WY / true_res).astype(np.int32)
    gh, gw = env_grid.shape
    cx_safe = np.clip(cx, 0, gw - 1)
    cy_safe = np.clip(cy, 0, gh - 1)
    centre_in = (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
    return (env_grid[cy_safe, cx_safe] != 0) & centre_in


def save_channel_figure(spatial, robot_pos_cell, out_path, title):
    """4 small subplots for is_known/is_wall/gas/recency and one for det_count."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for i, name in enumerate(CHANNELS):
        ax = axes[i]
        ax.imshow(spatial[i], origin="lower", cmap="viridis",
                  vmin=spatial[i].min(), vmax=max(spatial[i].max(), 1e-6))
        # Draw robot at center of 98x98 ego window (49, 49)
        ax.plot(49, 49, marker="o", color="red", markersize=5)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_wall_comparison(gt_grid, old_wall, new_wall, template_name, out_path):
    """Side-by-side GT walls, OLD is_wall, NEW is_wall."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Wall representation: {template_name}",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(gt_grid != 0, origin="lower", cmap="gray_r")
    axes[0].set_title(f"GT walls (0.1 m grid, {gt_grid.shape})", fontsize=10)

    axes[1].imshow(old_wall, origin="lower", cmap="gray_r")
    axes[1].set_title(f"OLD is_wall (centre-only, {old_wall.shape})\n"
                      f"walls detected: {old_wall.sum()}", fontsize=10)

    axes[2].imshow(new_wall, origin="lower", cmap="gray_r")
    axes[2].set_title(f"NEW is_wall (any-subcell, {new_wall.shape})\n"
                      f"walls detected: {new_wall.sum()}", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run_one_template(template_id, seed, snapshot_steps, output_dir):
    template_name = TEMPLATE_NAMES[template_id]
    print(f"\nTemplate {template_id} ({template_name}), seed={seed}")

    env = SpatialObsWrapper(GasSourceEnv(template_id=template_id))
    (spatial, wind), _ = env.reset(seed=seed)

    # --- Wall-only comparison (OLD vs NEW, full map) ---
    gt_grid = env._env._grid.grid
    true_res = env._env._grid.resolution
    old_wall = compute_old_is_wall(gt_grid, env._map_h_cells, env._map_w_cells,
                                    env.CELL_RES, true_res)
    new_wall = env._dense_wall

    save_wall_comparison(
        gt_grid, old_wall, new_wall, template_name,
        os.path.join(output_dir, f"t{template_id}_{template_name}_walls.png"),
    )

    # --- Channel snapshots across a short episode ---
    # Use a simple forward-biased random action to explore the map.
    rng = np.random.default_rng(seed)
    step = 0
    for target_step in snapshot_steps:
        while step < target_step:
            # Random direction, biased forward
            action = rng.normal(0.5, 0.3, size=2)
            (spatial, wind), _, terminated, truncated, _ = env.step(action)
            step += 1
            if terminated or truncated:
                break
        if terminated or truncated:
            break

        save_channel_figure(
            spatial, env._env._robot_pos,
            os.path.join(output_dir,
                         f"t{template_id}_{template_name}_step{step}.png"),
            title=f"Template {template_id} ({template_name}), step {step}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=int, default=4,
                        help="Template id 0-5")
    parser.add_argument("--all-templates", action="store_true",
                        help="Run for every template 0-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60,
                        help="Max steps to run per template")
    parser.add_argument("--snapshots", type=int, nargs="+",
                        default=[1, 15, 30, 60],
                        help="Step numbers at which to snapshot channels")
    parser.add_argument("--output-dir", type=str, default=_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    templates = list(range(6)) if args.all_templates else [args.template]
    for t in templates:
        run_one_template(t, args.seed, args.snapshots, args.output_dir)

    print(f"\nDone. Images in {args.output_dir}/")


if __name__ == "__main__":
    main()
