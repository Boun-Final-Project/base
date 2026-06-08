"""
Visual sanity check for GADEN map / wind / source loading.

For each map, plots the rasterized occupancy grid, the gas source (red star),
the recommended robot start (blue circle), and a subsampled wind quiver.
Saves a PNG per map under rl_cfd/test/gaden_viz/.

Run once after touching gaden_loader.py to verify walls look right and the
wind direction matches GADEN's CFD output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_cfd.test.gaden_loader import (
    DEFAULT_MAP_KEYS,
    load_full_map,
)


_DEFAULT_GADEN_ROOT = Path(_SCRIPT_DIR).parents[1] / "gaden_maps"
_DEFAULT_OUT_DIR    = Path(_SCRIPT_DIR) / "gaden_viz"


def visualize_map(gaden_root: Path, yaml_path: Path, map_key: str,
                  out_dir: Path, quiver_stride: int = 8) -> None:
    md = load_full_map(gaden_root, yaml_path, map_key)
    grid_arr = md["grid"].grid
    res      = md["grid"].resolution
    H, W     = grid_arr.shape
    src      = md["source_pos"]
    rob      = md["robot_pos"]
    field    = md["wind_field"].field
    speed, direction = md["wind_field"].spatial_mean()

    fig, ax = plt.subplots(figsize=(W * res * 0.7 + 2, H * res * 0.7 + 2))
    ax.imshow(grid_arr, origin="lower", cmap="gray_r",
              extent=[0, W * res, 0, H * res])

    # Subsampled quiver
    ys, xs = np.mgrid[0:H:quiver_stride, 0:W:quiver_stride]
    us = field[ys, xs, 0]
    vs = field[ys, xs, 1]
    ax.quiver(xs * res + res / 2, ys * res + res / 2, us, vs,
              color="tab:green", alpha=0.7, scale=20, width=0.003)

    ax.plot(src[0], src[1], "r*", markersize=22, label="source")
    ax.plot(rob[0], rob[1], "bo", markersize=12, label="robot start")

    occ = (grid_arr != 0).mean()
    ax.set_title(
        f"{map_key}  ({W}×{H} cells, {W*res:.1f}×{H*res:.1f} m, walls={occ:.1%})\n"
        f"mean wind: {speed:.3f} m/s @ {np.degrees(direction):+.0f}°  "
        f"max speed: {md['wind_field'].peak_speed():.2f} m/s",
        fontsize=11,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{map_key}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  {map_key:18s}  walls={occ:5.1%}  wind={speed:.3f}m/s  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaden-root", type=Path, default=_DEFAULT_GADEN_ROOT)
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAP_KEYS)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--quiver-stride", type=int, default=8)
    args = parser.parse_args()

    yaml_path = args.gaden_root / "recommended_configs.yaml"
    print(f"GADEN root: {args.gaden_root}")
    print(f"Output:     {args.output_dir}")
    print(f"Maps:       {args.maps}")
    for key in args.maps:
        visualize_map(args.gaden_root, yaml_path, key, args.output_dir,
                      quiver_stride=args.quiver_stride)


if __name__ == "__main__":
    main()
