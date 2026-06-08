"""
Side-by-side visualization: YOUR wrapper vs FRIEND's wrapper on the same seed.

Two identical envs + identical action sequences. Captures observations at
specified timesteps and saves one comparison figure per snapshot.

YOUR channels  (5): is_known, is_wall, gas, recency, det_count
FRIEND channels (4): occupancy (-1/0/1), gas (-1/0/1), recency, det_count
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

from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.envs.spatial_obs_wrapper import SpatialObsWrapper as Mine
from rl_5_channel.envs.spatial_obs_wrapper_friend import SpatialObsWrapper as Friend

_OUT_DIR = os.path.join(_SCRIPT_DIR, "obs_viz")
TEMPLATE_NAMES = ["empty", "single_wall", "u_shape", "three_walls",
                  "complex_maze", "multi_room"]


def step_matched(mine_env, friend_env, action):
    mine_out   = mine_env.step(action)
    friend_out = friend_env.step(action)
    return mine_out, friend_out


def plot_side_by_side(mine_spatial, friend_spatial, gt_wall, step_idx,
                      template_name, out_path):
    # Panels: GT walls | YOUR is_known | YOUR is_wall | YOUR gas | YOUR rec | YOUR det
    #                  | FRIEND occ    | FRIEND gas   | FRIEND rec | FRIEND det | (empty)
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.25, wspace=0.15)

    # Row 0: GT (spans both rows in the first column) + YOUR channels
    ax_gt = fig.add_subplot(gs[:, 0])
    ax_gt.imshow(gt_wall != 0, origin="lower", cmap="gray_r",
                 interpolation="nearest")
    ax_gt.set_title(f"GT walls @ 0.1m\n({gt_wall.shape})", fontsize=10)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    mine_titles   = ["is_known", "is_wall", "gas", "recency", "det_count"]
    friend_titles = ["occupancy\n(-1=wall, 0=unknown, 1=free)",
                     "gas\n(-1=no det, 0=unvisited, 1=det)",
                     "recency", "det_count"]

    for i, name in enumerate(mine_titles):
        ax = fig.add_subplot(gs[0, i + 1])
        im = ax.imshow(mine_spatial[i], origin="lower", cmap="viridis",
                       interpolation="nearest")
        ax.plot(49, 49, "ro", markersize=4)
        ax.set_title(f"YOURS · {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for i, name in enumerate(friend_titles):
        ax = fig.add_subplot(gs[1, i + 1])
        # For occupancy: use a 3-way colormap centered on 0
        if i == 0 or i == 1:  # occupancy or gas (3-valued)
            im = ax.imshow(friend_spatial[i], origin="lower",
                           cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")
        else:
            im = ax.imshow(friend_spatial[i], origin="lower", cmap="viridis",
                           interpolation="nearest")
        ax.plot(49, 49, "ro", markersize=4)
        ax.set_title(f"FRIEND · {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Template {template_name} — step {step_idx}",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run_template(template_id, seed, snapshots, output_dir):
    name = TEMPLATE_NAMES[template_id]
    print(f"\nTemplate {template_id} ({name}), seed={seed}")

    mine_env   = Mine(GasSourceEnv(template_id=template_id))
    friend_env = Friend(GasSourceEnv(template_id=template_id))

    (m_spatial, _), _ = mine_env.reset(seed=seed)
    (f_spatial, _), _ = friend_env.reset(seed=seed)
    gt_wall = mine_env._env._grid.grid

    rng = np.random.default_rng(seed)
    step = 0
    # snapshot at step 0 (right after reset)
    if 0 in snapshots:
        plot_side_by_side(
            m_spatial, f_spatial, gt_wall, 0, name,
            os.path.join(output_dir, f"compare_t{template_id}_{name}_step0.png"),
        )

    for target in sorted([s for s in snapshots if s > 0]):
        while step < target:
            action = rng.normal(0.5, 0.3, size=2)
            (m_spatial, _), _, m_term, m_trunc, _ = mine_env.step(action)
            (f_spatial, _), _, f_term, f_trunc, _ = friend_env.step(action)
            step += 1
            if m_term or m_trunc or f_term or f_trunc:
                break
        if m_term or m_trunc or f_term or f_trunc:
            print(f"  Episode ended at step {step}")
            break
        plot_side_by_side(
            m_spatial, f_spatial, gt_wall, step, name,
            os.path.join(output_dir,
                         f"compare_t{template_id}_{name}_step{step}.png"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=int, default=4)
    parser.add_argument("--all-templates", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshots", type=int, nargs="+",
                        default=[0, 15, 40, 80])
    parser.add_argument("--output-dir", type=str, default=_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    templates = list(range(6)) if args.all_templates else [args.template]
    for t in templates:
        run_template(t, args.seed, args.snapshots, args.output_dir)
    print(f"\nDone. Images in {args.output_dir}/")


if __name__ == "__main__":
    main()
