#!/usr/bin/env python3
"""
Visualize 5 random map samples from each of the 10 map templates (T0-T9).

Usage:
    python reinforcement_learning/visualization_scripts/visualize_map_templates.py
    python reinforcement_learning/visualization_scripts/visualize_map_templates.py --seed 42
    python reinforcement_learning/visualization_scripts/visualize_map_templates.py --out maps.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from reinforcement_learning.envs.map_generator import MapGenerator

TEMPLATE_NAMES = [
    "T0: Empty",
    "T1: Single Wall",
    "T2: U-Shape",
    "T3: Three Walls",
    "T4: Complex Maze",
    "T5: Multi-Room",
    "T6: Dead-End Corridor",
    "T7: Serpentine Corridor",
    "T8: Dense Multi-Room",
    "T9: Hybrid",
]

N_TEMPLATES = 10
N_SAMPLES = 5


def render_map(ax, grid, source_pos, robot_pos, title=""):
    """Render a single map onto an axes."""
    # grid: 2D numpy array, 1=wall, 0=free
    # Flip vertically so row-0 is at bottom (y increases upward)
    img = np.flipud(grid)

    # Custom colormap: 0=free (white), 1=wall (dark grey)
    cmap = ListedColormap(["#f5f5f5", "#2d2d2d"])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")

    h, w = grid.shape

    def to_pixel(pos_m):
        """Convert metric (x, y) to pixel (col, row) with y-flip."""
        from reinforcement_learning import config as cfg
        col = pos_m[0] / cfg.GRID_RESOLUTION
        row = (h - 1) - pos_m[1] / cfg.GRID_RESOLUTION  # flip
        return col, row

    # Source: red star
    sx, sy = to_pixel(source_pos)
    ax.plot(sx, sy, marker="*", color="#e74c3c", markersize=9,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    # Robot: blue circle
    rx, ry = to_pixel(robot_pos)
    ax.plot(rx, ry, marker="o", color="#3498db", markersize=7,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualize map templates")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gen = MapGenerator(rng=rng)

    fig, axes = plt.subplots(
        N_TEMPLATES, N_SAMPLES,
        figsize=(N_SAMPLES * 2.6, N_TEMPLATES * 2.2),
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )

    for t_id in range(N_TEMPLATES):
        for s_idx in range(N_SAMPLES):
            ax = axes[t_id, s_idx]
            result = gen.generate(template_id=t_id)

            title = TEMPLATE_NAMES[t_id] if s_idx == 0 else ""
            subtitle = f"sample {s_idx + 1}"
            full_title = f"{title}\n{subtitle}" if title else subtitle

            render_map(
                ax,
                result["grid"].grid,  # OccupancyGrid → numpy array
                result["source_pos"],
                result["robot_pos"],
                title=full_title,
            )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2d2d2d", label="Wall"),
        mpatches.Patch(color="#f5f5f5", label="Free"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
                   markersize=10, label="Source"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                   markersize=9, label="Robot"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Map Templates T0–T9 — 5 samples each  (seed={args.seed})",
        fontsize=13, y=1.002,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if args.out:
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
