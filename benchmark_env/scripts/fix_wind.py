#!/usr/bin/env python3
"""
fix_wind.py — make a GADEN wind CSV "z-safe" so stock GADEN (no FillEmptyWindCells)
never reads zero wind at the anemometer height.

The CFD point cloud is binned onto GADEN's cell grid; columns meshed coarser than
cell_size leave vertical gaps (the "sawtooth"), so the z=0.5 cell can be empty and
the anemometer reads 0. This tool bins to the grid, fills every empty cell in a
populated (x,y) column from the nearest populated cell in that SAME column
(vertical only — no cross-wall bleed), and re-emits one point per cell at its
center. Re-binning this on stock GADEN yields a gap-free grid.

Usage:
    fix_wind.py <in.csv> <out.csv> [--cell 0.1] [--sample-z 0.5]
Prints a before/after comparison (dead-at-sample-z columns).
"""
import sys
import csv
import argparse
from collections import defaultdict


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = [h.strip().strip('"') for h in next(r)]
        idx = {name: hdr.index(name) for name in ("Points:0", "Points:1", "Points:2",
                                                  "U:0", "U:1", "U:2")}
        pts = []
        for row in r:
            if not row:
                continue
            try:
                x, y, z = (float(row[idx["Points:0"]]), float(row[idx["Points:1"]]), float(row[idx["Points:2"]]))
                u, v, w = (float(row[idx["U:0"]]), float(row[idx["U:1"]]), float(row[idx["U:2"]]))
            except (ValueError, IndexError):
                continue
            pts.append((x, y, z, u, v, w))
    return pts


def bin_grid(pts, cell):
    """Return (grid dict (ix,iy,iz)->(u,v,w) averaged, origin, dims)."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    ox, oy, oz = min(xs), min(ys), min(zs)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for x, y, z, u, v, w in pts:
        ix = int((x - ox) / cell); iy = int((y - oy) / cell); iz = int((z - oz) / cell)
        a = acc[(ix, iy, iz)]
        a[0] += u; a[1] += v; a[2] += w; a[3] += 1
    grid = {k: (a[0] / a[3], a[1] / a[3], a[2] / a[3]) for k, a in acc.items()}
    dz = max(iz for _, _, iz in grid) + 1
    return grid, (ox, oy, oz), dz


def dead_at_z(grid, origin, cell, sample_z):
    """Count populated (x,y) columns whose sample_z cell is empty."""
    oz = origin[2]
    iz_s = int((sample_z - oz) / cell)
    cols = defaultdict(set)
    for (ix, iy, iz) in grid:
        cols[(ix, iy)].add(iz)
    dead = sum(1 for izs in cols.values() if iz_s not in izs)
    return dead, len(cols), iz_s


def vertical_fill(grid, dz):
    """Fill every empty cell in a populated column from nearest populated cell (same column)."""
    cols = defaultdict(dict)   # (ix,iy) -> {iz: (u,v,w)}
    for (ix, iy, iz), wind in grid.items():
        cols[(ix, iy)][iz] = wind
    filled = dict(grid)
    added = 0
    for (ix, iy), zmap in cols.items():
        for iz in range(dz):
            if iz in zmap:
                continue
            # nearest populated cell in this column
            best = None
            for d in range(1, dz):
                if iz + d in zmap:
                    best = zmap[iz + d]; break
                if iz - d in zmap:
                    best = zmap[iz - d]; break
            if best is not None:
                filled[(ix, iy, iz)] = best
                added += 1
    return filled, added


def write(path, grid, origin, cell):
    ox, oy, oz = origin
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Points:0", "Points:1", "Points:2", "U:0", "U:1", "U:2"])
        for (ix, iy, iz), (u, v, w) in sorted(grid.items()):
            x = ox + (ix + 0.5) * cell; y = oy + (iy + 0.5) * cell; z = oz + (iz + 0.5) * cell
            wr.writerow([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}", f"{u:.6g}", f"{v:.6g}", f"{w:.6g}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--cell", type=float, default=0.1)
    ap.add_argument("--sample-z", type=float, default=0.5)
    a = ap.parse_args()

    pts = load(a.infile)
    grid0, origin, dz = bin_grid(pts, a.cell)
    d0, ncol, iz_s = dead_at_z(grid0, origin, a.cell, a.sample_z)

    grid1, added = vertical_fill(grid0, dz)
    d1, ncol1, _ = dead_at_z(grid1, origin, a.cell, a.sample_z)

    write(a.outfile, grid1, origin, a.cell)

    print(f"  cells: {len(grid0):>7d} -> {len(grid1):>7d}  (+{added} filled)")
    print(f"  populated (x,y) columns: {ncol}")
    print(f"  dead-at-z={a.sample_z} columns: {d0} -> {d1}"
          + ("  ✅" if d1 == 0 else "  ⚠️ still gaps"))


if __name__ == "__main__":
    main()
