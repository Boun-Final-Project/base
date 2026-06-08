"""Load a GADEN map's geometry and run OUR CFD pipeline on it.
Then compare the resulting wind to GADEN's original wind on the same geometry.

Usage:
  python gen_case_from_gaden.py --map many_rooms --out-dir case_gaden_many_rooms
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RL_PACKAGE_PATH, GADEN_MAPS_ROOT
sys.path.insert(0, RL_PACKAGE_PATH)
from reinforcement_learning.test.gaden_loader import load_full_map

from gen_case import (
    map_to_walls_stl, parse_opening_list, punch_openings,
    write_block_mesh_dict, write_snappy_dict, write_control_dict,
    write_fv_schemes, write_fv_solution, write_transport_props,
    write_turbulence_props,
    write_U_field, write_p_field, write_k_field, write_epsilon_field,
    write_nut_field, find_in_mesh_point,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--map', required=True, help='GADEN map key e.g. many_rooms')
    p.add_argument('--gaden-root', default=GADEN_MAPS_ROOT)
    p.add_argument('--height', type=float, default=3.0)
    p.add_argument('--inlet-speed', type=float, default=0.1)
    p.add_argument('--bg-cells-per-meter', type=float, default=4.0)
    p.add_argument('--end-time', type=int, default=200)
    p.add_argument('--strip-margin', type=float, default=0.6,
                   help='outer margin to clear so bg-mesh inlet/outlet have open faces')
    p.add_argument('--opening-width', type=float, default=None,
                   help='GADEN-style: keep outer walls, punch a single opening this wide on E+W (meters)')
    p.add_argument('--openings-west', type=str, default='',
                   help='List of y-ranges as "1.4-2.9,6.1-7.6"')
    p.add_argument('--openings-east', type=str, default='')
    p.add_argument('--openings-south', type=str, default='')
    p.add_argument('--openings-north', type=str, default='')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / 'system').mkdir(parents=True)
    (out_dir / 'constant' / 'triSurface').mkdir(parents=True)
    (out_dir / '0.orig').mkdir()

    # 1. Load GADEN geometry
    gaden_root = Path(args.gaden_root)
    yaml_path = gaden_root / 'recommended_configs.yaml'
    fm = load_full_map(gaden_root, yaml_path, args.map)
    grid_obj = fm['grid']
    grid_arr = np.array(grid_obj.grid, dtype=np.int8)  # may be wall=1
    cell_size = grid_obj.resolution
    H, W = grid_arr.shape
    map_w = W * cell_size
    map_h = H * cell_size
    print(f"GADEN map: {args.map}, shape={grid_arr.shape}, cell={cell_size}, "
          f"map={map_w:.2f}x{map_h:.2f} m, walls={(grid_arr != 0).sum()}")

    # 2. Boundary openings
    openings = {
        'west':  parse_opening_list(args.openings_west),
        'east':  parse_opening_list(args.openings_east),
        'south': parse_opening_list(args.openings_south),
        'north': parse_opening_list(args.openings_north),
    }
    if any(openings.values()):
        wall_thick = max(1, int(round(args.strip_margin / cell_size)))
        punch_openings(grid_arr, cell_size, wall_thick, openings)
        n_open = sum(len(v) for v in openings.values())
        print(f"Punched {n_open} openings (W={openings['west']}, E={openings['east']}, "
              f"S={openings['south']}, N={openings['north']}); walls = {(grid_arr != 0).sum()}")
    elif args.opening_width is not None:
        ow_cells = max(1, int(round(args.opening_width / cell_size)))
        wall_thick = max(1, int(round(args.strip_margin / cell_size)))
        mid_y = H // 2
        half = ow_cells // 2
        grid_arr[max(0, mid_y - half):mid_y + half, :wall_thick] = 0
        grid_arr[max(0, mid_y - half):mid_y + half, -wall_thick:] = 0
        # Update openings dict for boundary-blocking
        y_lo = (mid_y - half) * cell_size; y_hi = (mid_y + half) * cell_size
        openings['west'] = [(y_lo, y_hi)]
        openings['east'] = [(y_lo, y_hi)]
        print(f"Punched {ow_cells*cell_size:.2f}m middle E+W openings, walls = {(grid_arr != 0).sum()}")
    else:
        margin = max(1, int(round(args.strip_margin / cell_size)))
        grid_arr[:, :margin] = 0
        grid_arr[:, -margin:] = 0
        grid_arr[:margin, :] = 0
        grid_arr[-margin:, :] = 0
        print(f"Stripped {margin}-cell outer band, walls now = {(grid_arr != 0).sum()}")

    # 3. Walls → STL (with boundary-blocking panels for true GADEN-style flow)
    pad = 0.5
    stl_path = out_dir / 'constant' / 'triSurface' / 'walls.stl'
    n_facets = map_to_walls_stl(grid_arr, cell_size, args.height, str(stl_path),
                                bg_pad=pad, openings=openings)
    print(f"STL: {n_facets} facets → {stl_path}")

    # 4. Background mesh
    x_min, x_max = -pad, map_w + pad
    y_min, y_max = -pad, map_h + pad
    z_min, z_max = 0.0, args.height
    nx = max(8, int((x_max - x_min) * args.bg_cells_per_meter))
    ny = max(8, int((y_max - y_min) * args.bg_cells_per_meter))
    nz = max(4, int((z_max - z_min) * args.bg_cells_per_meter))
    print(f"Background mesh: {nx}x{ny}x{nz}")
    write_block_mesh_dict(out_dir / 'system' / 'blockMeshDict',
                          x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)

    in_mesh_pt = find_in_mesh_point(grid_arr, cell_size, args.height / 2)
    print(f"locationInMesh = {in_mesh_pt}")
    write_snappy_dict(out_dir / 'system' / 'snappyHexMeshDict', in_mesh_pt)
    write_control_dict(out_dir / 'system' / 'controlDict', end_time=args.end_time)
    write_fv_schemes(out_dir / 'system' / 'fvSchemes')
    write_fv_solution(out_dir / 'system' / 'fvSolution')
    write_transport_props(out_dir / 'constant' / 'transportProperties')
    write_turbulence_props(out_dir / 'constant' / 'turbulenceProperties')
    write_U_field(out_dir / '0.orig' / 'U', args.inlet_speed)
    write_p_field(out_dir / '0.orig' / 'p')
    write_k_field(out_dir / '0.orig' / 'k')
    write_epsilon_field(out_dir / '0.orig' / 'epsilon')
    write_nut_field(out_dir / '0.orig' / 'nut')

    meta = {
        'source': 'gaden',
        'gaden_map': args.map,
        'map_width_m': float(map_w),
        'map_height_m': float(map_h),
        'cell_size_m': float(cell_size),
        'grid_shape': list(grid_arr.shape),
        'inlet_speed': float(args.inlet_speed),
        'wall_height_m': float(args.height),
        'bg_mesh_nx_ny_nz': [nx, ny, nz],
        'locationInMesh': list(in_mesh_pt),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out_dir / 'grid.npz',
                        grid=grid_arr.astype(np.uint8),
                        cell_size=cell_size,
                        map_width=map_w,
                        map_height=map_h)
    print(f"Case ready at {out_dir}")


if __name__ == '__main__':
    main()
