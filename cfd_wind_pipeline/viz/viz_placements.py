"""Generate a contact sheet of procedural maps with their inlet/outlet placements
so the user can review the placement strategy BEFORE running CFD."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

# Add parent dir to path so we can import placement, gen_case, config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT
from placement import sample_map_with_openings
from gen_case import punch_openings


def plot_one(ax, map_data, openings, title, wall_thick_m=0.6):
    grid = np.array(map_data['grid'].grid, dtype=np.int8)
    cell = map_data['grid'].resolution
    map_w = map_data['width']; map_h = map_data['height']
    # Apply the punch so viz shows the POST-PUNCH state
    wall_thick_cells = max(1, int(round(wall_thick_m / cell)))
    openings_dict = {'west': [], 'east': [], 'south': [], 'north': []}
    for o in openings:
        openings_dict[o.side].append((o.lo, o.hi))
    punch_openings(grid, cell, wall_thick_cells, openings_dict)
    ax.imshow(grid, origin='lower', extent=(0, map_w, 0, map_h),
              cmap='Greys', alpha=0.7, vmin=0, vmax=2)
    # Map boundary
    ax.add_patch(mpatches.Rectangle((0, 0), map_w, map_h, fill=False,
                                     edgecolor='black', linewidth=0.5))
    # Source + robot
    sx, sy = map_data['source_pos']; rx, ry = map_data['robot_pos']
    ax.scatter([sx], [sy], c='lime', s=40, marker='*', zorder=5,
               edgecolors='black', linewidth=0.5, label='source')
    ax.scatter([rx], [ry], c='cyan', s=30, marker='s', zorder=5,
               edgecolors='black', linewidth=0.5, label='robot')
    # Openings — draw as outlined rectangles (so we don't obscure the punched-out region)
    for o in openings:
        color = 'blue' if o.role == 'inlet' else 'red'
        if o.side == 'west':
            ax.add_patch(mpatches.Rectangle((0, o.lo), wall_thick_m, o.hi-o.lo,
                                              fill=False, edgecolor=color, linewidth=2.0))
            cx, cy = wall_thick_m, 0.5*(o.lo+o.hi)
            dx = 0.8 if o.role == 'inlet' else -0.6
            ax.arrow(cx, cy, dx, 0, head_width=0.35, head_length=0.25,
                     fc=color, ec=color, alpha=0.95, linewidth=2)
        elif o.side == 'east':
            ax.add_patch(mpatches.Rectangle((map_w-wall_thick_m, o.lo), wall_thick_m, o.hi-o.lo,
                                              fill=False, edgecolor=color, linewidth=2.0))
            cx, cy = map_w-wall_thick_m, 0.5*(o.lo+o.hi)
            dx = -0.8 if o.role == 'inlet' else 0.6
            ax.arrow(cx, cy, dx, 0, head_width=0.35, head_length=0.25,
                     fc=color, ec=color, alpha=0.95, linewidth=2)
        elif o.side == 'south':
            ax.add_patch(mpatches.Rectangle((o.lo, 0), o.hi-o.lo, wall_thick_m,
                                              fill=False, edgecolor=color, linewidth=2.0))
            cx, cy = 0.5*(o.lo+o.hi), wall_thick_m
            dy = 0.8 if o.role == 'inlet' else -0.6
            ax.arrow(cx, cy, 0, dy, head_width=0.35, head_length=0.25,
                     fc=color, ec=color, alpha=0.95, linewidth=2)
        else:  # north
            ax.add_patch(mpatches.Rectangle((o.lo, map_h-wall_thick_m), o.hi-o.lo, wall_thick_m,
                                              fill=False, edgecolor=color, linewidth=2.0))
            cx, cy = 0.5*(o.lo+o.hi), map_h-wall_thick_m
            dy = -0.8 if o.role == 'inlet' else 0.6
            ax.arrow(cx, cy, 0, dy, head_width=0.35, head_length=0.25,
                     fc=color, ec=color, alpha=0.95, linewidth=2)
    ax.set_xlim(-1, map_w + 1)
    ax.set_ylim(-1, map_h + 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-maps', type=int, default=40)
    p.add_argument('--cols', type=int, default=5)
    p.add_argument('--templates', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--out', default=os.path.join(DATA_ROOT, 'placement_contact_sheet.png'))
    args = p.parse_args()

    samples = []
    rng = np.random.default_rng(0)
    attempts = 0
    while len(samples) < args.n_maps and attempts < args.n_maps * 4:
        seed = args.seed_start + attempts
        template_id = int(rng.choice(args.templates))
        map_data, ops = sample_map_with_openings(template_id, seed)
        attempts += 1
        if map_data is None:
            continue
        samples.append((template_id, seed, map_data, ops))

    print(f"Got {len(samples)} valid placements in {attempts} attempts "
          f"(reject rate {100*(1-len(samples)/attempts):.1f}%)")

    rows = (len(samples) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 3.5, rows * 2.8))
    axes = np.atleast_2d(axes)
    for idx, (tid, seed, map_data, ops) in enumerate(samples):
        r, c = idx // args.cols, idx % args.cols
        n_in = sum(1 for o in ops if o.role == 'inlet')
        n_out = sum(1 for o in ops if o.role == 'outlet')
        title = f"T{tid} seed={seed} ({n_in}in/{n_out}out)"
        plot_one(axes[r, c], map_data, ops, title)
    for idx in range(len(samples), rows * args.cols):
        r, c = idx // args.cols, idx % args.cols
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
