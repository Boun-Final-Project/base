"""Contact sheet visualization of CFD wind library cases.

Renders a grid of N×M panels, each showing one case's walls + wind quiver.
Use to spot-check the library after a build run.

Usage:
    python viz/viz_library.py --library-dir $CFD_DATA_ROOT/library_v1
    python viz/viz_library.py --library-dir $CFD_DATA_ROOT/library_v1 \\
        --max-cases 24 --cols 6 --out /tmp/library_sheet.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_one(ax, case_dir: Path):
    """Draw walls + wind quiver for a single case on the given axis."""
    try:
        wind = np.load(case_dir / 'wind_field.npz')
        grid = np.load(case_dir / 'grid.npz')
    except FileNotFoundError:
        ax.text(0.5, 0.5, f'{case_dir.name}\n(no data)', ha='center', va='center',
                transform=ax.transAxes, color='red')
        ax.set_xticks([]); ax.set_yticks([])
        return
    field = wind['field']
    cell = float(wind['cell_size'])
    occ = grid['grid']
    H, W = field.shape[:2]
    speed = np.linalg.norm(field, axis=-1)

    ax.imshow(occ, origin='lower', extent=(0, W*cell, 0, H*cell),
              cmap='Greys', alpha=0.65, vmin=0, vmax=2)
    ax.imshow(speed, origin='lower', extent=(0, W*cell, 0, H*cell),
              cmap='viridis', alpha=0.45, vmin=0, vmax=max(0.1, speed.max()))
    skip = max(1, H // 18)
    xs = (np.arange(W) + 0.5) * cell
    ys = (np.arange(H) + 0.5) * cell
    Xs, Ys = np.meshgrid(xs[::skip], ys[::skip], indexing='xy')
    Ux = field[::skip, ::skip, 0]
    Uy = field[::skip, ::skip, 1]
    ax.quiver(Xs, Ys, Ux, Uy, color='red', alpha=0.85, scale=18, width=0.005)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    mean_u = speed[occ == 0].mean()
    title = f"{case_dir.name}  |U|={mean_u:.2f}"
    ax.set_title(title, fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library-dir', required=True)
    p.add_argument('--max-cases', type=int, default=24,
                   help='Cap on cases to show (default 24)')
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--out', default=None,
                   help='Output png (default: <library-dir>/library_contact_sheet.png)')
    p.add_argument('--only-complete', action='store_true', default=True,
                   help='Skip cases without wind_field.npz (default true)')
    args = p.parse_args()

    lib = Path(args.library_dir)
    manifest_path = lib / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json at {lib}")
    manifest = json.loads(manifest_path.read_text())

    # Filter to completed cases (with wind_field.npz)
    cases = []
    for entry in manifest:
        cd = Path(entry['case_dir'])
        if args.only_complete and not (cd / 'wind_field.npz').exists():
            continue
        cases.append(cd)
        if len(cases) >= args.max_cases:
            break

    if not cases:
        print(f"No completed cases found in {lib}")
        return

    n = len(cases)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for i, cd in enumerate(cases):
        r, c = i // cols, i % cols
        plot_one(axes[r, c], cd)

    fig.suptitle(f"{lib.name} — {n} cases", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = args.out or str(lib / 'library_contact_sheet.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f"Saved {out} ({n} cases, {rows}×{cols})")


if __name__ == '__main__':
    main()
