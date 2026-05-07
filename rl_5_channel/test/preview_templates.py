"""
Quick visual preview of all 11 map-generator templates.

Renders 4 random samples per template (44 maps total) as a single
multi-panel figure. Calls MapGenerator directly — does not require
test_envs.json or any GasSourceEnv setup.

Usage:
    python -m base.rl_5_channel.test.preview_templates
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

from rl_5_channel.envs.map_generator import MapGenerator

TEMPLATE_NAMES = [
    "Empty",            # 0
    "Single Wall",      # 1
    "U-Shape",          # 2
    "Three Walls",      # 3
    "Complex Maze",     # 4
    "Multi-Room",       # 5
    "Dead-End Corridor",   # 6
    "Serpentine",          # 7
    "Dense Multi-Room",    # 8
    "Hybrid",              # 9
]

_DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "template_preview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-template", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    n_templates = len(TEMPLATE_NAMES)
    n_samples = args.samples_per_template

    rng = np.random.default_rng(args.seed)
    mg = MapGenerator(rng=rng)

    fig, axes = plt.subplots(
        n_templates, n_samples,
        figsize=(n_samples * 3.0, n_templates * 2.4),
    )
    fig.suptitle(
        f"All {n_templates} map-generator templates "
        f"({n_samples} samples each)",
        fontsize=14, fontweight="bold", y=1.0,
    )

    for tid in range(n_templates):
        for s in range(n_samples):
            ax = axes[tid, s]
            try:
                result = mg.generate(template_id=tid)
                grid = result["grid"]
                source = result["source_pos"]
                robot = result["robot_pos"]
                ax.imshow(
                    grid.grid,
                    origin="lower",
                    extent=[0, grid.width, 0, grid.height],
                    cmap="Greys", vmin=0, vmax=1,
                    aspect="equal",
                    interpolation="nearest",
                )
                ax.plot(*source, "r*", markersize=11, zorder=5)
                ax.plot(*robot,  "bo", markersize=6, zorder=5)
                size_label = f"{grid.width:.0f}×{grid.height:.0f} m"
            except Exception as e:
                ax.text(0.5, 0.5, f"FAIL:\n{type(e).__name__}",
                        ha="center", va="center", fontsize=8, color="red",
                        transform=ax.transAxes)
                size_label = "—"

            if s == 0:
                ax.set_ylabel(
                    f"T{tid}\n{TEMPLATE_NAMES[tid]}",
                    fontsize=9, fontweight="bold",
                )
            ax.set_title(size_label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
