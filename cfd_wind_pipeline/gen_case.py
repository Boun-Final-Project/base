"""Generate OpenFOAM case for a procedural map.

Pipeline:
1. Generate one procedural map using MapGenerator (or a small hand-crafted one)
2. Write the walls as STL (cubes for each wall cell) into constant/triSurface/walls.stl
3. Generate all required OpenFOAM dict files (blockMeshDict, snappyHexMeshDict,
   controlDict, etc.) into system/
4. Generate 0.orig/{U,p,k,epsilon,nut} with inlet on x=0 face, outlet on x=W face
5. Save metadata (grid, dims, inlet/outlet face names) for the extractor later

Output: a complete OpenFOAM case directory at OUT_DIR ready for
  blockMesh + snappyHexMesh + simpleFoam.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Local import (this file is meant to be run from the cfd_wind_pipeline/ dir,
# or with PYTHONPATH including it)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RL_PACKAGE_PATH
sys.path.insert(0, RL_PACKAGE_PATH)
from reinforcement_learning.envs.map_generator import MapGenerator


# ---------------------------------------------------------------------------
# STL writing (ASCII format)
# ---------------------------------------------------------------------------

def _box_facets(x0, y0, z0, x1, y1, z1):
    """Return 12 triangle facets (each = (normal, [v1,v2,v3])) for an axis-aligned box."""
    # vertices
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    # faces: 6 quads × 2 triangles each, each with outward normal
    quads = [
        ((0, 0, -1), [0, 3, 2, 1]),  # -z (bottom)
        ((0, 0,  1), [4, 5, 6, 7]),  # +z (top)
        ((-1, 0, 0), [0, 4, 7, 3]),  # -x
        ((1,  0, 0), [1, 2, 6, 5]),  # +x
        ((0, -1, 0), [0, 1, 5, 4]),  # -y
        ((0,  1, 0), [2, 3, 7, 6]),  # +y
    ]
    facets = []
    for normal, q in quads:
        # split quad into two triangles (q[0], q[1], q[2]) and (q[0], q[2], q[3])
        facets.append((normal, (v[q[0]], v[q[1]], v[q[2]])))
        facets.append((normal, (v[q[0]], v[q[2]], v[q[3]])))
    return facets


def write_stl(facets, path, name='walls'):
    """Write ASCII STL file."""
    with open(path, 'w') as f:
        f.write(f"solid {name}\n")
        for normal, (v1, v2, v3) in facets:
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in (v1, v2, v3):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


# ---------------------------------------------------------------------------
# Procedural map → STL conversion
# ---------------------------------------------------------------------------

def map_to_walls_stl(grid_arr, cell_size, height, out_path,
                     bg_pad=0.0, openings=None):
    """For each wall cell in the 2D grid, emit a cube STL.

    grid_arr: (H, W) int, nonzero = wall
    cell_size: m per cell
    height: vertical extrusion in m
    bg_pad: if > 0, also add wall panels that extend outer walls to x=-bg_pad / y=-bg_pad
            for each side, EXCEPT at y/x positions inside an opening.
    openings: dict with keys 'west', 'east', 'south', 'north', each a list of
              (lo, hi) tuples — y-ranges for west/east, x-ranges for south/north.
              If a y/x falls in any range for that side, no blocking wall is added there.

    Returns the total facet count.
    """
    facets = []
    H, W = grid_arr.shape
    map_w = W * cell_size
    map_h = H * cell_size

    # 1. Wall cells from grid
    for r in range(H):
        for c in range(W):
            if grid_arr[r, c] != 0:
                x0 = c * cell_size
                y0 = r * cell_size
                facets.extend(_box_facets(x0, y0, 0.0,
                                           x0 + cell_size, y0 + cell_size, height))

    # 2. Boundary-blocking wall panels (extend outer walls to bg-mesh boundary)
    if bg_pad > 0:
        openings = openings or {}
        def in_opening(side, coord):
            return any(lo <= coord <= hi for lo, hi in openings.get(side, []))
        # West (x=-bg_pad to x=0), iterate over y-cells
        for r in range(H):
            y_c = (r + 0.5) * cell_size
            if not in_opening('west', y_c):
                facets.extend(_box_facets(-bg_pad, r*cell_size, 0,
                                           0, (r+1)*cell_size, height))
        # East
        for r in range(H):
            y_c = (r + 0.5) * cell_size
            if not in_opening('east', y_c):
                facets.extend(_box_facets(map_w, r*cell_size, 0,
                                           map_w + bg_pad, (r+1)*cell_size, height))
        # South (y=-bg_pad to y=0)
        for c in range(W):
            x_c = (c + 0.5) * cell_size
            if not in_opening('south', x_c):
                facets.extend(_box_facets(c*cell_size, -bg_pad, 0,
                                           (c+1)*cell_size, 0, height))
        # North
        for c in range(W):
            x_c = (c + 0.5) * cell_size
            if not in_opening('north', x_c):
                facets.extend(_box_facets(c*cell_size, map_h, 0,
                                           (c+1)*cell_size, map_h + bg_pad, height))

    write_stl(facets, out_path, name='walls')
    return len(facets)


def parse_opening_list(s):
    """Parse '0.5-1.5,3.0-4.0' into [(0.5,1.5), (3.0,4.0)]."""
    if not s:
        return []
    result = []
    for chunk in s.split(','):
        lo, hi = chunk.split('-')
        result.append((float(lo), float(hi)))
    return result


def punch_openings(grid_arr, cell_size, wall_thick_cells, openings):
    """Clear cells in the outer wall band corresponding to the openings."""
    H, W = grid_arr.shape
    for y_lo, y_hi in openings.get('west', []):
        r_lo, r_hi = int(y_lo / cell_size), int(np.ceil(y_hi / cell_size))
        grid_arr[r_lo:r_hi, :wall_thick_cells] = 0
    for y_lo, y_hi in openings.get('east', []):
        r_lo, r_hi = int(y_lo / cell_size), int(np.ceil(y_hi / cell_size))
        grid_arr[r_lo:r_hi, -wall_thick_cells:] = 0
    for x_lo, x_hi in openings.get('south', []):
        c_lo, c_hi = int(x_lo / cell_size), int(np.ceil(x_hi / cell_size))
        grid_arr[:wall_thick_cells, c_lo:c_hi] = 0
    for x_lo, x_hi in openings.get('north', []):
        c_lo, c_hi = int(x_lo / cell_size), int(np.ceil(x_hi / cell_size))
        grid_arr[-wall_thick_cells:, c_lo:c_hi] = 0


# ---------------------------------------------------------------------------
# OpenFOAM dict writers — minimal viable for snappyHexMesh + simpleFoam k-epsilon
# ---------------------------------------------------------------------------

_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM auto-generated for procedural-map CFD                              |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def write_block_mesh_dict(path, x_min, x_max, y_min, y_max, z_min, z_max,
                          nx, ny, nz):
    """Background mesh as a single hex block with named patches."""
    txt = _HEADER.format(cls='dictionary', obj='blockMeshDict')
    txt += f"""
scale 1;

vertices
(
    ({x_min} {y_min} {z_min})
    ({x_max} {y_min} {z_min})
    ({x_max} {y_max} {z_min})
    ({x_min} {y_max} {z_min})
    ({x_min} {y_min} {z_max})
    ({x_max} {y_min} {z_max})
    ({x_max} {y_max} {z_max})
    ({x_min} {y_max} {z_max})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    sides
    {{
        type wall;
        faces ((0 1 5 4) (3 7 6 2));
    }}
    floor
    {{
        type wall;
        faces ((0 3 2 1));
    }}
    ceiling
    {{
        type wall;
        faces ((4 5 6 7));
    }}
);

mergePatchPairs
(
);
"""
    Path(path).write_text(txt)


def write_snappy_dict(path, in_mesh_point, refinement_level=2):
    txt = _HEADER.format(cls='dictionary', obj='snappyHexMeshDict')
    px, py, pz = in_mesh_point
    txt += f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    walls
    {{
        type triSurfaceMesh;
        file "walls.stl";
    }}
}}

castellatedMeshControls
{{
    maxLocalCells     1000000;
    maxGlobalCells    5000000;
    minRefinementCells 0;
    maxLoadUnbalance  0.1;
    nCellsBetweenLevels 1;

    features
    (
    );

    refinementSurfaces
    {{
        walls
        {{
            level ({refinement_level} {refinement_level});
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
    }}

    locationInMesh ({px} {py} {pz});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance    2.0;
    nSolveIter   30;
    nRelaxIter   5;
}}

addLayersControls
{{
    relativeSizes       true;
    layers              {{}}
    expansionRatio      1.0;
    finalLayerThickness 0.3;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       -1e30;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}}

mergeTolerance 1e-6;
"""
    Path(path).write_text(txt)


def write_control_dict(path, end_time=200, write_interval=50, delta_t=1):
    txt = _HEADER.format(cls='dictionary', obj='controlDict')
    txt += f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {delta_t};
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    Path(path).write_text(txt)


def write_fv_schemes(path):
    txt = _HEADER.format(cls='dictionary', obj='fvSchemes')
    txt += """
ddtSchemes      { default steadyState; }
gradSchemes
{
    default        Gauss linear;
    grad(U)        cellLimited Gauss linear 1;
}
divSchemes
{
    default                 none;
    div(phi,U)              bounded Gauss linearUpwind grad(U);
    div(phi,k)              bounded Gauss upwind;
    div(phi,epsilon)        bounded Gauss upwind;
    div(phi,omega)          bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""
    Path(path).write_text(txt)


def write_fv_solution(path):
    txt = _HEADER.format(cls='dictionary', obj='fvSolution')
    txt += """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|epsilon|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl
    {
        p               1e-3;
        U               1e-4;
        "(k|epsilon|omega)" 1e-4;
    }
}
relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.7;
    }
}
"""
    Path(path).write_text(txt)


def write_transport_props(path, nu=1.5e-5):
    txt = _HEADER.format(cls='dictionary', obj='transportProperties')
    txt += f"""
transportModel  Newtonian;
nu              {nu};
"""
    Path(path).write_text(txt)


def write_turbulence_props(path):
    txt = _HEADER.format(cls='dictionary', obj='turbulenceProperties')
    txt += """
simulationType  RAS;
RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""
    Path(path).write_text(txt)


# ---- 0.orig fields ----

def _field_header(cls, obj):
    return _HEADER.format(cls=cls, obj=obj)


def write_U_field(path, inlet_speed):
    """U = velocity. Inlet: fixed velocity. Outlet: zeroGradient.
    Walls (sides, floor, ceiling, walls): no-slip (uniform 0).
    """
    txt = _field_header('volVectorField', 'U')
    txt += f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    inlet     {{ type fixedValue;  value uniform ({inlet_speed} 0 0); }}
    outlet    {{ type zeroGradient; }}
    sides     {{ type noSlip; }}
    floor     {{ type noSlip; }}
    ceiling   {{ type noSlip; }}
    walls     {{ type noSlip; }}
}}
"""
    Path(path).write_text(txt)


def write_p_field(path):
    txt = _field_header('volScalarField', 'p')
    txt += """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet     { type zeroGradient; }
    outlet    { type fixedValue; value uniform 0; }
    sides     { type zeroGradient; }
    floor     { type zeroGradient; }
    ceiling   { type zeroGradient; }
    walls     { type zeroGradient; }
}
"""
    Path(path).write_text(txt)


def write_k_field(path, k_init=0.1):
    txt = _field_header('volScalarField', 'k')
    txt += f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_init};

boundaryField
{{
    inlet     {{ type fixedValue; value uniform {k_init}; }}
    outlet    {{ type zeroGradient; }}
    sides     {{ type kqRWallFunction; value uniform {k_init}; }}
    floor     {{ type kqRWallFunction; value uniform {k_init}; }}
    ceiling   {{ type kqRWallFunction; value uniform {k_init}; }}
    walls     {{ type kqRWallFunction; value uniform {k_init}; }}
}}
"""
    Path(path).write_text(txt)


def write_epsilon_field(path, eps_init=0.01):
    txt = _field_header('volScalarField', 'epsilon')
    txt += f"""
dimensions      [0 2 -3 0 0 0 0];
internalField   uniform {eps_init};

boundaryField
{{
    inlet     {{ type fixedValue; value uniform {eps_init}; }}
    outlet    {{ type zeroGradient; }}
    sides     {{ type epsilonWallFunction; value uniform {eps_init}; }}
    floor     {{ type epsilonWallFunction; value uniform {eps_init}; }}
    ceiling   {{ type epsilonWallFunction; value uniform {eps_init}; }}
    walls     {{ type epsilonWallFunction; value uniform {eps_init}; }}
}}
"""
    Path(path).write_text(txt)


def write_nut_field(path):
    txt = _field_header('volScalarField', 'nut')
    txt += """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet     { type calculated; value uniform 0; }
    outlet    { type calculated; value uniform 0; }
    sides     { type nutkWallFunction; value uniform 0; }
    floor     { type nutkWallFunction; value uniform 0; }
    ceiling   { type nutkWallFunction; value uniform 0; }
    walls     { type nutkWallFunction; value uniform 0; }
}
"""
    Path(path).write_text(txt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_in_mesh_point(grid_arr, cell_size, z):
    """Find a free-cell location to act as locationInMesh (the seed point
    snappyHexMesh uses to identify the fluid region)."""
    H, W = grid_arr.shape
    # Try center first, then sweep
    centers = [(H // 2, W // 2)]
    for r in range(H):
        for c in range(W):
            centers.append((r, c))
    for r, c in centers:
        if grid_arr[r, c] == 0:
            return ((c + 0.5) * cell_size, (r + 0.5) * cell_size, z)
    raise RuntimeError("No free cell found for locationInMesh")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--template-id', type=int, default=2,
                   help='Map template ID (0=empty, 1=single_wall, 2=u_shape, '
                        '3=three_walls, 4=complex_maze, 5=multi_room)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--height', type=float, default=3.0, help='wall height [m]')
    p.add_argument('--inlet-speed', type=float, default=1.0, help='m/s')
    p.add_argument('--bg-cells-per-meter', type=float, default=4.0,
                   help='background mesh resolution')
    p.add_argument('--end-time', type=int, default=200,
                   help='simpleFoam steady-state iterations to run')
    p.add_argument('--strip-outer', action='store_true', default=True,
                   help='Strip outer wall band so inlet/outlet are open')
    p.add_argument('--no-strip-outer', dest='strip_outer', action='store_false')
    p.add_argument('--strip-margin', type=float, default=0.6,
                   help='Margin to strip from outer walls [m]')
    p.add_argument('--strip-all-sides', action='store_true', default=True,
                   help='Strip N/S sides too (in addition to inlet/outlet E/W)')
    p.add_argument('--opening-width', type=float, default=None,
                   help='GADEN-style single-opening (centered) — punches one opening this wide on E+W')
    p.add_argument('--openings-west', type=str, default='',
                   help='Explicit list of y-ranges on west wall, e.g. "1.4-2.9,6.1-7.6"')
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

    # 1. Generate map
    rng = np.random.default_rng(args.seed)
    gen = MapGenerator(rng=rng)
    map_data = gen.generate(template_id=args.template_id)
    grid_obj = map_data['grid']
    # Grid resolution per map_generator: cells are 0.1m by default
    grid_arr = grid_obj.grid                # (H, W) — 0=free, nonzero=wall
    cell_size = grid_obj.resolution
    map_width = map_data['width']
    map_height = map_data['height']
    print(f"Map: template={args.template_id}, size={map_width:.2f}×{map_height:.2f} m, "
          f"grid={grid_arr.shape}, cell={cell_size:.3f} m, walls={(grid_arr != 0).sum()}")

    # 2. Boundary opening mode
    H, W = grid_arr.shape
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
        print(f"Punched {n_open} explicit openings; walls = {(grid_arr != 0).sum()}")
    elif args.opening_width is not None:
        ow_cells = max(1, int(round(args.opening_width / cell_size)))
        wall_thick = max(1, int(round(args.strip_margin / cell_size)))
        mid_y = H // 2
        half = ow_cells // 2
        grid_arr[max(0, mid_y - half):mid_y + half, :wall_thick] = 0
        grid_arr[max(0, mid_y - half):mid_y + half, -wall_thick:] = 0
        y_lo = (mid_y - half) * cell_size; y_hi = (mid_y + half) * cell_size
        openings['west'] = [(y_lo, y_hi)]
        openings['east'] = [(y_lo, y_hi)]
        print(f"Punched centered {ow_cells*cell_size:.2f}m E+W openings; "
              f"walls = {(grid_arr != 0).sum()}")
    elif args.strip_outer:
        margin = max(1, int(round(args.strip_margin / cell_size)))
        grid_arr[:, :margin] = 0
        grid_arr[:, -margin:] = 0
        if args.strip_all_sides:
            grid_arr[:margin, :] = 0
            grid_arr[-margin:, :] = 0
        print(f"Stripped outer walls (margin {margin} cells = {margin*cell_size:.2f}m); "
              f"new wall count = {(grid_arr != 0).sum()}")

    # 3. Walls → STL (with boundary-blocking panels if explicit openings supplied)
    pad = 0.5
    stl_path = out_dir / 'constant' / 'triSurface' / 'walls.stl'
    if any(openings.values()):
        n_facets = map_to_walls_stl(grid_arr, cell_size, args.height, str(stl_path),
                                     bg_pad=pad, openings=openings)
    else:
        n_facets = map_to_walls_stl(grid_arr, cell_size, args.height, str(stl_path))
    print(f"STL: {n_facets} facets → {stl_path}")

    # 3. blockMeshDict (background bounding box, slightly larger than map)
    x_min, x_max = -pad, map_width + pad
    y_min, y_max = -pad, map_height + pad
    z_min, z_max = 0.0, args.height
    nx = max(8, int((x_max - x_min) * args.bg_cells_per_meter))
    ny = max(8, int((y_max - y_min) * args.bg_cells_per_meter))
    nz = max(4, int((z_max - z_min) * args.bg_cells_per_meter))
    print(f"Background mesh: {nx}×{ny}×{nz} cells over {x_max-x_min:.1f}×{y_max-y_min:.1f}×{z_max-z_min:.1f} m")
    write_block_mesh_dict(out_dir / 'system' / 'blockMeshDict',
                          x_min, x_max, y_min, y_max, z_min, z_max,
                          nx, ny, nz)

    # 4. snappyHexMeshDict
    in_mesh_pt = find_in_mesh_point(grid_arr, cell_size, args.height / 2)
    print(f"locationInMesh = {in_mesh_pt}")
    write_snappy_dict(out_dir / 'system' / 'snappyHexMeshDict', in_mesh_pt)

    # 5. Other system files
    write_control_dict(out_dir / 'system' / 'controlDict', end_time=args.end_time)
    write_fv_schemes(out_dir / 'system' / 'fvSchemes')
    write_fv_solution(out_dir / 'system' / 'fvSolution')

    # 6. constant
    write_transport_props(out_dir / 'constant' / 'transportProperties')
    write_turbulence_props(out_dir / 'constant' / 'turbulenceProperties')

    # 7. 0.orig fields
    write_U_field(out_dir / '0.orig' / 'U', args.inlet_speed)
    write_p_field(out_dir / '0.orig' / 'p')
    write_k_field(out_dir / '0.orig' / 'k')
    write_epsilon_field(out_dir / '0.orig' / 'epsilon')
    write_nut_field(out_dir / '0.orig' / 'nut')

    # 8. Save map metadata
    meta = {
        'template_id': int(args.template_id),
        'seed': int(args.seed),
        'map_width_m': float(map_width),
        'map_height_m': float(map_height),
        'cell_size_m': float(cell_size),
        'grid_shape': list(grid_arr.shape),
        'inlet_speed': float(args.inlet_speed),
        'inlet_face': 'inlet (x=0 face)',
        'outlet_face': f'outlet (x={x_max} face)',
        'wall_height_m': float(args.height),
        'bg_mesh_nx_ny_nz': [nx, ny, nz],
        'locationInMesh': list(in_mesh_pt),
        'source_pos': map_data['source_pos'],
        'robot_pos': map_data['robot_pos'],
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out_dir / 'grid.npz',
                        grid=grid_arr.astype(np.uint8),
                        cell_size=cell_size,
                        map_width=map_width,
                        map_height=map_height)

    print(f"\nCase ready at {out_dir}")
    print("Run with:")
    print(f"  cd {out_dir}")
    print("  blockMesh && snappyHexMesh -overwrite && cp -r 0.orig 0 && simpleFoam")


if __name__ == '__main__':
    main()
