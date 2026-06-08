"""Visualize the extracted wind field as a quiver plot over the map."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    case = Path(args.case_dir)
    wind = np.load(case / 'wind_field.npz')
    grid = np.load(case / 'grid.npz')
    field = wind['field']  # (H, W, 2)
    cell = float(wind['cell_size'])
    occ = grid['grid']  # (H, W)
    H, W = field.shape[:2]
    print(f"field shape: {field.shape}, cell={cell}, H={H} W={W}")

    fig, ax = plt.subplots(figsize=(12, 8))
    # Walls
    ax.imshow(occ, origin='lower', extent=(0, W*cell, 0, H*cell),
              cmap='Greys', alpha=0.6, vmin=0, vmax=2)
    # Wind speed magnitude as background
    speed = np.linalg.norm(field, axis=-1)
    im = ax.imshow(speed, origin='lower', extent=(0, W*cell, 0, H*cell),
                   cmap='viridis', alpha=0.5, vmin=0, vmax=max(0.1, speed.max()))
    plt.colorbar(im, ax=ax, label='|U| [m/s]')
    # Quiver — subsample for clarity
    skip = max(1, H // 30)
    xs = (np.arange(W) + 0.5) * cell
    ys = (np.arange(H) + 0.5) * cell
    Xs, Ys = np.meshgrid(xs[::skip], ys[::skip], indexing='xy')
    Ux = field[::skip, ::skip, 0]
    Uy = field[::skip, ::skip, 1]
    ax.quiver(Xs, Ys, Ux, Uy, color='red', alpha=0.85, scale=20)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"Wind field — case={case.name}, mean|U|={speed[occ==0].mean():.3f} m/s, "
                 f"std={speed[occ==0].std():.3f}")
    ax.set_aspect('equal')

    out = args.out or str(case / 'wind_viz.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
