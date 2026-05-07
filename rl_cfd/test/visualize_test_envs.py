"""
Visually inspect the 100 fixed test environments from test_envs.json.

Produces one figure per template, arranged as a grid of maps. Each map
shows the occupancy grid, the gas source (red star), and the robot start
position (blue circle).

Requires test_envs.json to exist (run generate_test_envs.py first).

Usage:
    python3 visualize_test_envs.py
    python3 visualize_test_envs.py --test-envs test_envs.json --output-dir test_env_inspection/
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from reinforcement_learning.envs.gas_source_env import GasSourceEnv

_DEFAULT_TEST_ENVS  = os.path.join(_SCRIPT_DIR, "test_envs.json")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "test_env_inspection")

TEMPLATE_NAMES = [
    "Empty", "Single Wall", "U-Shape",
    "Three Walls", "Complex Maze", "Multi-Room",
    "Dead-End Corridor", "Serpentine", "Dense Multi-Room",
    "Hybrid",
]

# Grid layout: (rows, cols) per template matching the env counts
GRID_LAYOUT = {0: (2, 5), 1: (2, 5), 2: (3, 5), 3: (3, 5), 4: (5, 5), 5: (5, 5),
               6: (3, 5), 7: (3, 5), 8: (3, 5), 9: (2, 5)}


def render_template(envs_for_template, template_id, output_dir):
    rows, cols = GRID_LAYOUT[template_id]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    fig.suptitle(
        f"Template {template_id}: {TEMPLATE_NAMES[template_id]} "
        f"({len(envs_for_template)} environments)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes = np.array(axes).flatten()

    for ax, env_spec in zip(axes, envs_for_template):
        # Reconstruct the environment with the fixed seed
        env = GasSourceEnv(template_id=env_spec["template_id"])
        env.reset(seed=env_spec["seed"])

        grid   = env._grid
        source = env._source_pos
        robot  = env._robot_pos

        # Occupancy grid: walls dark, free space white
        ax.imshow(
            grid.grid,
            origin="lower",
            extent=[0, grid.width, 0, grid.height],
            cmap="Greys", vmin=0, vmax=1,
            aspect="equal",
            interpolation="nearest",
        )

        ax.plot(*source, "r*", markersize=9,  zorder=5, label="Source")
        ax.plot(*robot,  "bo", markersize=6,  zorder=5, label="Robot")

        ax.set_title(
            f"env {env_spec['env_id']}  "
            f"{grid.width:.0f}×{grid.height:.0f} m",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused axes (shouldn't happen with exact counts, but just in case)
    for ax in axes[len(envs_for_template):]:
        ax.set_visible(False)

    # Single shared legend in the top-left axis
    axes[0].legend(loc="upper right", fontsize=6, markerscale=0.8)

    fig.tight_layout()
    path = os.path.join(output_dir, f"template_{template_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Template {template_id} ({TEMPLATE_NAMES[template_id]:>14}): {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-envs",  type=str, default=_DEFAULT_TEST_ENVS)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    with open(args.test_envs) as f:
        data = json.load(f)
    envs = data["envs"]
    print(f"Loaded {len(envs)} environments from {args.test_envs}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Group by template
    by_template = {}
    for e in envs:
        by_template.setdefault(e["template_id"], []).append(e)

    print(f"\nRendering figures → {args.output_dir}/")
    for tid in sorted(by_template):
        render_template(by_template[tid], tid, args.output_dir)

    print(f"\nDone. {len(by_template)} figures saved.")


if __name__ == "__main__":
    main()
