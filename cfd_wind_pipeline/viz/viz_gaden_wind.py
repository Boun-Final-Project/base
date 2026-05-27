"""Visualize an actual GADEN wind field for comparison with our CFD output.
Uses the same plot style as viz_wind.py so they're directly comparable."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RL_PACKAGE_PATH, GADEN_MAPS_ROOT, DATA_ROOT
sys.path.insert(0, RL_PACKAGE_PATH)
from reinforcement_learning.test.gaden_loader import DEFAULT_MAP_KEYS, load_full_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', required=True, help='GADEN map key e.g. many_rooms')
    p.add_argument('--out', default=None)
    p.add_argument('--gaden-root', default=GADEN_MAPS_ROOT)
    args = p.parse_args()

    gaden_root = Path(args.gaden_root)
    yaml_path = gaden_root / 'recommended_configs.yaml'
    fm = load_full_map(gaden_root, yaml_path, args.map)
    wind_field = fm['wind_field']
    grid = fm['grid']

    # field is (H, W, 2) where [..., 0]=Ux, [..., 1]=Uy
    field = wind_field.field.astype(np.float64)
    res = wind_field.resolution
    occ = grid.grid  # (H, W), nonzero = wall
    H, W = field.shape[:2]

    map_w = W * res
    map_h = H * res
    speed = np.linalg.norm(field, axis=-1)
    free_mask = (occ == 0)

    print(f"GADEN map: {args.map}")
    print(f"  shape: {field.shape}, cell={res}, map={map_w:.2f}x{map_h:.2f} m")
    print(f"  free cells: {free_mask.sum()} of {free_mask.size}")
    print(f"  free-cell |U|: mean={speed[free_mask].mean():.3f}, std={speed[free_mask].std():.3f}")
    dirs = np.arctan2(field[..., 1], field[..., 0])
    free_dirs = dirs[free_mask]
    sin_m = np.mean(np.sin(free_dirs)); cos_m = np.mean(np.cos(free_dirs))
    R = np.sqrt(sin_m**2 + cos_m**2)
    circ_std = float(np.sqrt(-2 * np.log(R))) if R > 0 else float('inf')
    print(f"  direction circular_std = {circ_std:.3f} rad")

    fig, ax = plt.subplots(figsize=(12, max(4, 12 * map_h / map_w)))
    # Walls as grey
    ax.imshow(occ, origin='lower', extent=(0, map_w, 0, map_h),
              cmap='Greys', alpha=0.6, vmin=0, vmax=2)
    # |U| as colormap
    speed_disp = speed.copy()
    speed_disp[~free_mask] = np.nan
    im = ax.imshow(speed_disp, origin='lower', extent=(0, map_w, 0, map_h),
                   cmap='viridis', alpha=0.7, vmin=0, vmax=max(0.1, speed[free_mask].max()))
    plt.colorbar(im, ax=ax, label='|U| [m/s]')
    # Quiver — subsample
    skip = max(1, H // 30)
    xs = (np.arange(W) + 0.5) * res
    ys = (np.arange(H) + 0.5) * res
    Xs, Ys = np.meshgrid(xs[::skip], ys[::skip], indexing='xy')
    Ux = field[::skip, ::skip, 0]
    Uy = field[::skip, ::skip, 1]
    # mask walls
    mask = (occ[::skip, ::skip] == 0)
    Ux = np.where(mask, Ux, 0)
    Uy = np.where(mask, Uy, 0)
    ax.quiver(Xs, Ys, Ux, Uy, color='red', alpha=0.85, scale=20)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"GADEN map='{args.map}' — mean|U|={speed[free_mask].mean():.3f}, "
                 f"speed_std={speed[free_mask].std():.3f}, dir_std={circ_std:.2f} rad")
    ax.set_aspect('equal')

    out = args.out or os.path.join(DATA_ROOT, f'gaden_{args.map}_wind.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
