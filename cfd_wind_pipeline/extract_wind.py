"""Extract wind field from an OpenFOAM case as a (H, W, 2) numpy grid.

Strategy:
1. Add a `sample` function object to the case that writes a horizontal slice
   at z=wall_height/2 as a raw (x y z Ux Uy Uz) text file.
2. Run `postProcess -func sample -latestTime` in the container to produce it.
3. Read the raw file, interpolate onto our (H, W) grid using nearest-cell or
   bilinear via scipy.
4. Save the wind field as wind_field.npz alongside the case.

Run AFTER the case has been solved (simpleFoam done).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


SAMPLE_DICT = """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM auto-generated wind-slice sampler                                  |
\\*---------------------------------------------------------------------------*/
type            surfaces;
libs            (sampling);
writeControl    onEnd;
surfaceFormat   raw;
fields          (U);
interpolationScheme cellPoint;

surfaces
(
    midSlice
    {{
        type cuttingPlane;
        planeType pointAndNormal;
        pointAndNormalDict
        {{
            point  (0 0 {z_slice});
            normal (0 0 1);
        }}
        interpolate true;
    }}
);
"""


def add_sample_dict(case_dir: Path, z_slice: float):
    (case_dir / 'system' / 'sample').write_text(
        SAMPLE_DICT.format(z_slice=z_slice)
    )


def parse_raw_surface_file(path: Path):
    """Parse OpenFOAM raw surface output.
    Format: header comment lines + 'x y z Ux Uy Uz' per row.
    Returns: (xy, uxy) arrays."""
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xy = data[:, :2]  # x, y (z is constant on the slice)
    uxy = data[:, 3:5]  # Ux, Uy (ignore Uz)
    return xy, uxy


def grid_interpolate(xy, uxy, map_width, map_height, cell_size,
                     occupancy_grid):
    """Interpolate (x,y)→(Ux,Uy) onto a regular grid matching our cells."""
    H, W = occupancy_grid.shape
    # Grid cell centers
    cx = (np.arange(W) + 0.5) * cell_size  # x coordinates of cell centers
    cy = (np.arange(H) + 0.5) * cell_size  # y coordinates
    gx, gy = np.meshgrid(cx, cy, indexing='xy')  # (H, W)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Use nearest-neighbor for robustness near walls; linear for smooth interior
    Ux = griddata(xy, uxy[:, 0], pts, method='linear', fill_value=0.0)
    Uy = griddata(xy, uxy[:, 1], pts, method='linear', fill_value=0.0)
    # Backfill NaN regions (outside convex hull) with nearest
    Ux_nn = griddata(xy, uxy[:, 0], pts, method='nearest')
    Uy_nn = griddata(xy, uxy[:, 1], pts, method='nearest')
    Ux = np.where(np.isfinite(Ux), Ux, Ux_nn)
    Uy = np.where(np.isfinite(Uy), Uy, Uy_nn)

    field = np.stack([Ux.reshape(H, W), Uy.reshape(H, W)], axis=-1)
    # Zero out wind in wall cells
    field[occupancy_grid != 0] = 0.0
    return field


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir', required=True)
    p.add_argument('--write-dict-only', action='store_true',
                   help='Just write the sample dict; do not parse')
    p.add_argument('--parse-only', action='store_true',
                   help='Skip dict writing; only parse existing sample output')
    args = p.parse_args()

    case = Path(args.case_dir)
    meta = json.loads((case / 'meta.json').read_text())
    z_slice = meta['wall_height_m'] / 2

    if not args.parse_only:
        add_sample_dict(case, z_slice)
        print(f"Wrote system/sample with z_slice={z_slice}")

    if args.write_dict_only:
        return

    # Find the raw surface file
    pp_dir = case / 'postProcessing' / 'sample'
    if not pp_dir.exists():
        raise FileNotFoundError(
            f"{pp_dir} not found. Run `postProcess -func sample -latestTime` "
            "in the container first."
        )
    # Latest time subdir
    times = sorted([p for p in pp_dir.iterdir() if p.is_dir()],
                   key=lambda p: float(p.name))
    if not times:
        raise FileNotFoundError(f"No time directories in {pp_dir}")
    last_time = times[-1]
    raw_files = list(last_time.glob('U_midSlice.raw')) + \
                list(last_time.glob('midSlice*.raw')) + \
                list(last_time.glob('*.raw'))
    if not raw_files:
        raise FileNotFoundError(f"No .raw files in {last_time}")
    raw_path = raw_files[0]
    print(f"Reading {raw_path}")

    xy, uxy = parse_raw_surface_file(raw_path)
    print(f"  {len(xy)} sample points, Ux range [{uxy[:,0].min():.3f}, {uxy[:,0].max():.3f}], "
          f"Uy range [{uxy[:,1].min():.3f}, {uxy[:,1].max():.3f}]")

    # Load original grid for binning + wall masking
    grid_npz = np.load(case / 'grid.npz')
    occ_grid = grid_npz['grid']
    cell_size = float(grid_npz['cell_size'])
    map_w = float(grid_npz['map_width'])
    map_h = float(grid_npz['map_height'])

    field = grid_interpolate(xy, uxy, map_w, map_h, cell_size, occ_grid)
    H, W = field.shape[:2]
    print(f"Wind field: shape {field.shape}, "
          f"|U| range [{np.linalg.norm(field, axis=-1).min():.3f}, "
          f"{np.linalg.norm(field, axis=-1).max():.3f}], "
          f"mean |U| = {np.linalg.norm(field, axis=-1).mean():.3f}")
    # Free-cell stats
    free_mask = (occ_grid == 0)
    speeds = np.linalg.norm(field, axis=-1)
    free_speeds = speeds[free_mask]
    print(f"Free cells: n={free_mask.sum()}, "
          f"mean |U| = {free_speeds.mean():.3f}, "
          f"std |U| = {free_speeds.std():.3f} (spatial variance — was 0 in uniform training!)")
    # Direction stats
    dirs = np.arctan2(field[..., 1], field[..., 0])
    free_dirs = dirs[free_mask]
    # Circular std
    sin_mean = np.mean(np.sin(free_dirs))
    cos_mean = np.mean(np.cos(free_dirs))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circ_std = np.sqrt(-2 * np.log(R)) if R > 0 else float('inf')
    print(f"Free-cell direction: circular_std = {circ_std:.3f} rad "
          f"(GADEN range was 1.02-2.00)")

    out_path = case / 'wind_field.npz'
    np.savez_compressed(out_path, field=field.astype(np.float32),
                        cell_size=cell_size,
                        map_width=map_w, map_height=map_h)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
