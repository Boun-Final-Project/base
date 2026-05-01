"""
Loaders that turn a GADEN map directory into RL-env compatible inputs:
occupancy grid, spatially-varying wind field, and recommended source / robot
positions from gaden_maps/recommended_configs.yaml.

Walls come from the STL CAD models (``cad_models/walls.stl`` ∪
``cad_models/inner.stl``), sliced at z=0.5 m (matches GADEN sim source
heights) and scanline-filled. ``doors.stl`` is treated as openings and not
rasterized.

The wind field still comes from ``wind_at_cell_centers_0.csv`` and is
z-collapsed: per (x, y) cell we average all (Ux, Uy) entries across the CFD
points falling in it. Walls (per the STL raster) have their wind zeroed.
"""

from __future__ import annotations

from pathlib import Path
import csv
import struct

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from ..envs.occupancy_grid import OccupancyGrid


# Robot-height z-slice for STL rasterization. Matches GADEN sim source
# heights (z=0.5 in the bundled sim.yaml files).
_STL_Z_SLICE = 0.5

# GADEN STL convention (verified by visualizing each file's fill on
# 10x6_u_left): walls.stl is the outer wall MATERIAL (its fill is wall),
# inner.stl is the FLUID ENVELOPE inside the building (its fill is free,
# its complement is the inner walls). doors.stl marks openings; ignore.
_SOLID_STL_NAMES = ("walls.stl",)
_FLUID_STL_NAMES = ("inner.stl",)


# User-facing map names → folder names (folder typo on "labrinth" preserved)
MAP_NAME_ALIASES = {
    "4rooms":          "4_rooms",
    "uleft":           "10x6_u_left",
    "uright":          "10x6_u_right",
    "labyrinth_left":  "curved_labrinth_left",
    "labyrinth_right": "curved_labrinth_right",
    "many_rooms":      "many_rooms",
    "ultimate":        "ultimate",
}

DEFAULT_MAP_KEYS = list(MAP_NAME_ALIASES.keys())

_WIND_CSV_REL  = Path("wind_simulations/1ms/wind_at_cell_centers_0.csv")
_CONFIG_REL    = Path("environment_configurations/config1/config.yaml")


def resolve_map_dir(gaden_root: Path, map_key: str) -> Path:
    folder = MAP_NAME_ALIASES.get(map_key, map_key)
    map_dir = Path(gaden_root) / folder
    if not map_dir.is_dir():
        raise FileNotFoundError(
            f"GADEN map '{map_key}' (folder '{folder}') not found at {map_dir}"
        )
    return map_dir


def _read_cell_size(map_dir: Path) -> float:
    cfg_path = map_dir / _CONFIG_REL
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["cell_size"])


def _read_wind_csv(map_dir: Path):
    """Stream the wind CSV once, returning x, y, ux, uy as float arrays."""
    csv_path = map_dir / _WIND_CSV_REL
    xs, ys, uxs, uys = [], [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected: Points:0, Points:1, Points:2, U:0, U:1, U:2
        ix = header.index("Points:0")
        iy = header.index("Points:1")
        iu = header.index("U:0")
        iv = header.index("U:1")
        for row in reader:
            xs.append(float(row[ix]))
            ys.append(float(row[iy]))
            uxs.append(float(row[iu]))
            uys.append(float(row[iv]))
    return (np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(uxs, dtype=np.float64),
            np.asarray(uys, dtype=np.float64))


def _bounds(xs: np.ndarray, ys: np.ndarray, cell_size: float):
    """Bounding box snapped to cell_size. Returns (origin_x, origin_y, W, H)."""
    margin = 1  # one-cell padding so origin isn't tight against wind cloud
    x_min = np.floor(xs.min() / cell_size) * cell_size - margin * cell_size
    y_min = np.floor(ys.min() / cell_size) * cell_size - margin * cell_size
    x_max = np.ceil (xs.max() / cell_size) * cell_size + margin * cell_size
    y_max = np.ceil (ys.max() / cell_size) * cell_size + margin * cell_size
    W = int(round((x_max - x_min) / cell_size))
    H = int(round((y_max - y_min) / cell_size))
    return float(x_min), float(y_min), W, H


def _parse_stl_binary(path: Path) -> np.ndarray:
    """Return the triangle list of a binary STL file, shape (N, 3, 3)."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        dt = np.dtype([("n", "3f4"), ("v", "9f4"), ("a", "u2")])
        data = np.frombuffer(f.read(50 * n), dtype=dt, count=n)
    return data["v"].reshape(-1, 3, 3).astype(np.float64)


def _triangle_z_segments(tris: np.ndarray, z: float) -> np.ndarray:
    """Compute the 2D line segments where each triangle crosses plane z.

    Returns shape ``(M, 2, 2)`` — for each triangle that properly crosses the
    plane, the two ``(x, y)`` endpoints of its intersection segment.
    Tangent / coplanar / non-crossing triangles are dropped.
    """
    zs = tris[..., 2]
    above = zs > z
    below = zs < z
    n_above = above.sum(axis=1)
    n_below = below.sum(axis=1)
    crossing = (n_above >= 1) & (n_below >= 1)
    tris = tris[crossing]
    if len(tris) == 0:
        return np.zeros((0, 2, 2))
    above = above[crossing]
    n_above_c = above.sum(axis=1)

    # 'odd' = the singleton-side vertex; 'others' = the two on the opposite side
    odd_above = n_above_c == 1
    odd_idx   = np.where(odd_above, above.argmax(axis=1), (~above).argmax(axis=1))

    # Index helpers
    rows = np.arange(len(tris))
    other_mask = np.ones((len(tris), 3), dtype=bool)
    other_mask[rows, odd_idx] = False
    others = np.argsort(other_mask.astype(np.int8), axis=1)[:, 1:]   # (M, 2)

    odd_v   = tris[rows, odd_idx]                                     # (M, 3)
    other_v = tris[rows[:, None], others]                             # (M, 2, 3)

    # Linear interpolation along edges odd→other_v[:, k]
    dz = other_v[..., 2] - odd_v[:, None, 2]                          # (M, 2)
    # crossing implies dz != 0 on both edges (singleton on one side)
    t  = (z - odd_v[:, None, 2]) / dz                                 # (M, 2)
    xy = odd_v[:, None, :2] + t[..., None] * (other_v[..., :2] - odd_v[:, None, :2])
    return xy                                                         # (M, 2, 2)


def _scanline_fill(segments: np.ndarray, H: int, W: int,
                   cell_size: float, origin_xy: tuple[float, float]) -> np.ndarray:
    """Even-odd scanline polygon fill from a soup of 2D line segments.

    segments : ``(M, 2, 2)`` world-frame ``(x, y)`` endpoints.
    Returns a ``(H, W)`` boolean wall mask.
    """
    if len(segments) == 0:
        return np.zeros((H, W), dtype=bool)

    ox, oy = origin_xy
    seg = np.empty_like(segments)
    seg[..., 0] = (segments[..., 0] - ox) / cell_size
    seg[..., 1] = (segments[..., 1] - oy) / cell_size

    x1 = seg[:, 0, 0]; y1 = seg[:, 0, 1]
    x2 = seg[:, 1, 0]; y2 = seg[:, 1, 1]
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)
    dy    = y2 - y1

    walls = np.zeros((H, W), dtype=bool)
    for r in range(H):
        y_row  = r + 0.5
        active = (y_min < y_row) & (y_row <= y_max)
        if not active.any():
            continue
        x_ints = x1[active] + (y_row - y1[active]) * (x2[active] - x1[active]) / dy[active]
        x_ints.sort()
        # Pair consecutive intersections — interior of even-odd contour.
        for i in range(0, len(x_ints) - 1, 2):
            c0 = max(int(np.ceil (x_ints[i]     - 0.5)),     0)
            c1 = min(int(np.floor(x_ints[i + 1] - 0.5)) + 1, W)
            if c1 > c0:
                walls[r, c0:c1] = True
    return walls


def _rasterize_stl_walls(map_dir: Path, H: int, W: int, cell_size: float,
                         origin_xy: tuple[float, float],
                         z_slice: float = _STL_Z_SLICE) -> np.ndarray:
    """Rasterize the GADEN CAD geometry into an HxW bool wall mask.

    Combines two STL files with opposite semantics:
      walls.stl  — solid: filled cells are wall material
      inner.stl  — fluid envelope: filled cells are free, complement is wall
    """
    cad_dir = map_dir / "cad_models"

    def _fill(name):
        path = cad_dir / name
        if not path.exists():
            return None
        segs = _triangle_z_segments(_parse_stl_binary(path), z_slice)
        return _scanline_fill(segs, H, W, cell_size, origin_xy)

    solid = np.zeros((H, W), dtype=bool)
    for name in _SOLID_STL_NAMES:
        m = _fill(name)
        if m is not None:
            solid |= m

    fluid = None
    for name in _FLUID_STL_NAMES:
        m = _fill(name)
        if m is not None:
            fluid = m if fluid is None else (fluid | m)

    if fluid is None:
        if not solid.any():
            raise FileNotFoundError(
                f"No usable STL geometry in {cad_dir}; expected at least one of "
                f"{_SOLID_STL_NAMES} or {_FLUID_STL_NAMES}."
            )
        return solid
    # walls = solid (outer wall material) ∪ ¬fluid (everything outside inner.stl envelope)
    return solid | (~fluid)


def load_gaden_grid(map_dir: Path):
    """Rasterize a GADEN map's wind cloud into a 2D occupancy grid.

    Returns
    -------
    grid : OccupancyGrid
        ``grid.grid`` is an int8 array of shape (H, W); 0 = free, 1 = wall.
        ``grid.resolution`` matches the GADEN config's cell_size.
    origin_xy : tuple[float, float]
        World-frame coordinates (in metres) of cell (0, 0)'s lower corner.
        Subtract this from any GADEN absolute position to get env-frame xy.
    """
    map_dir = Path(map_dir)
    cell_size = _read_cell_size(map_dir)
    # CFD bbox sets the grid's footprint so the wind field rasterization can
    # use the same origin / shape. Walls themselves come from the STL CAD.
    xs, ys, _, _ = _read_wind_csv(map_dir)
    origin_x, origin_y, W, H = _bounds(xs, ys, cell_size)

    walls = _rasterize_stl_walls(map_dir, H, W, cell_size, (origin_x, origin_y))

    grid = OccupancyGrid(W * cell_size, H * cell_size, cell_size)
    grid.grid = walls.astype(np.int8)
    return grid, (origin_x, origin_y)


class GadenWindField:
    """Spatially-varying wind, queried at world (env-frame) positions.

    Parameters
    ----------
    field : np.ndarray
        Shape ``(H, W, 2)``. ``field[r, c, 0] = Ux``, ``field[r, c, 1] = Uy``.
    resolution : float
        Cell size in metres.
    occupancy : np.ndarray
        Shape ``(H, W)`` bool; True where the cell is wall (zero wind).
    """

    def __init__(self, field: np.ndarray, resolution: float, occupancy: np.ndarray):
        self.field      = field
        self.resolution = float(resolution)
        self.H, self.W, _ = field.shape
        self._free_mask = ~occupancy
        speeds = np.linalg.norm(field, axis=2)
        self._peak_speed = float(speeds[self._free_mask].max()) if self._free_mask.any() else 0.0

        # Drop-in compat with rl_cfd's WindField: per-cell components + ctx state.
        self.Ux = field[..., 0].astype(np.float32)
        self.Uy = field[..., 1].astype(np.float32)
        self.max_speed = 2.0  # ctx normalization, must match cfg.WIND_MAX_SPEED at training time
        spd, dirn = self.spatial_mean()
        self.speed = spd
        self.direction = dirn

    def query(self, positions: np.ndarray) -> np.ndarray:
        """Nearest-cell lookup. positions: (N, 2) -> (N, 2) (Ux, Uy)."""
        if positions.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        cols = np.clip(np.floor(positions[:, 0] / self.resolution).astype(np.int64),
                       0, self.W - 1)
        rows = np.clip(np.floor(positions[:, 1] / self.resolution).astype(np.int64),
                       0, self.H - 1)
        return self.field[rows, cols].astype(np.float64)

    def local_batch(self, positions: np.ndarray) -> np.ndarray:
        """rl_cfd's plume calls this; identical to query()."""
        return self.query(positions).astype(np.float32)

    def spatial_mean(self) -> tuple[float, float]:
        """(speed, direction) of the mean wind vector across non-wall cells."""
        if not self._free_mask.any():
            return 0.0, 0.0
        mean = self.field[self._free_mask].mean(axis=0)
        return float(np.linalg.norm(mean)), float(np.arctan2(mean[1], mean[0]))

    def peak_speed(self) -> float:
        """Maximum local wind speed across the field. (Method name kept distinct
        from the ``max_speed`` ctx-normalization attribute.)"""
        return self._peak_speed

    # rl_cfd-style ctx queries (mirror WindField.get_observation_*) ------------
    def get_observation(self):
        return (self.speed / self.max_speed, self.direction / (2.0 * np.pi))

    def get_observation_spatial(self):
        return (self.speed / self.max_speed,
                float(np.cos(self.direction)),
                float(np.sin(self.direction)))

    def get_dispersion_offset(self, dispersion_factor: float):
        s, d = self.speed, self.direction
        return np.array([s * dispersion_factor * np.cos(d),
                         s * dispersion_factor * np.sin(d)], dtype=np.float64)


def load_gaden_wind_field(map_dir: Path, grid: OccupancyGrid,
                          origin_xy: tuple[float, float]) -> GadenWindField:
    """Build the (H, W, 2) z-collapsed wind field for a GADEN map.

    Each (x, y) cell receives the mean of all wind-CSV rows that fall in it
    across all z-layers. Cells with no wind data (walls) receive zero wind.
    """
    map_dir = Path(map_dir)
    H, W = grid.grid.shape
    res = grid.resolution
    ox, oy = origin_xy

    xs, ys, uxs, uys = _read_wind_csv(map_dir)
    cols = np.clip(np.floor((xs - ox) / res).astype(np.int64), 0, W - 1)
    rows = np.clip(np.floor((ys - oy) / res).astype(np.int64), 0, H - 1)

    sum_field = np.zeros((H, W, 2), dtype=np.float64)
    counts    = np.zeros((H, W),     dtype=np.int64)
    np.add.at(sum_field[..., 0], (rows, cols), uxs)
    np.add.at(sum_field[..., 1], (rows, cols), uys)
    np.add.at(counts, (rows, cols), 1)

    field = np.zeros_like(sum_field)
    nonzero = counts > 0
    field[nonzero, 0] = sum_field[nonzero, 0] / counts[nonzero]
    field[nonzero, 1] = sum_field[nonzero, 1] / counts[nonzero]

    occupancy = (grid.grid != 0)
    free      = ~occupancy
    # The CFD mesh is sparser than 0.1 m in open regions, so many free cells
    # ended up with no CFD point and would otherwise read (0, 0) — making the
    # plume act as if there was no wind in those gaps. Fill each such cell
    # with the wind from the nearest free cell that does have data.
    has_data = nonzero & free
    needs    = free & ~has_data
    if needs.any() and has_data.any():
        # distance_transform_edt with return_indices=True gives, for every
        # cell, the (row, col) of the nearest True cell.
        _, (nr, nc) = distance_transform_edt(~has_data, return_indices=True)
        field[needs] = field[nr[needs], nc[needs]]

    # Walls: force zero wind regardless.
    field[occupancy] = 0.0
    return GadenWindField(field, res, occupancy)


def load_gaden_spec(yaml_path: Path, map_key: str, origin_xy: tuple[float, float],
                    sim_id: str = "sim1") -> dict:
    """Read source/robot positions for one (map, sim) pair from the yaml.

    Returns
    -------
    dict
        ``{map_key, folder, source_pos (np.ndarray, env-frame), robot_pos
        (np.ndarray, env-frame), start_time, notes}``
    """
    folder = MAP_NAME_ALIASES.get(map_key, map_key)
    with open(yaml_path) as f:
        all_specs = yaml.safe_load(f)
    if folder not in all_specs:
        raise KeyError(f"map '{folder}' (key '{map_key}') not in {yaml_path}")
    if sim_id not in all_specs[folder]:
        raise KeyError(f"{sim_id} not in {folder} ({list(all_specs[folder])})")
    sim = all_specs[folder][sim_id]

    if sim.get("source") is None:
        raise ValueError(f"{folder}/{sim_id}: source is null")
    if sim.get("robot_start") is None:
        raise ValueError(f"{folder}/{sim_id}: robot_start is null")

    ox, oy = origin_xy
    src = np.array(sim["source"],      dtype=np.float64) - np.array([ox, oy])
    rob = np.array(sim["robot_start"], dtype=np.float64) - np.array([ox, oy])
    return {
        "map_key":    map_key,
        "folder":     folder,
        "sim_id":     sim_id,
        "source_pos": src,
        "robot_pos":  rob,
        "start_time": int(sim.get("start_time", 0)),
        "notes":      sim.get("notes", ""),
    }


def load_full_map(gaden_root: Path, yaml_path: Path, map_key: str,
                  sim_id: str = "sim1") -> dict:
    """One-shot loader: returns everything an env reset needs.

    Returns
    -------
    dict
        ``{grid, source_pos, robot_pos, width, height, wind_field, origin_xy,
        spec}`` where the first five keys exactly match what
        ``MapGenerator.generate()`` returns and can be passed directly as
        ``options["map_data"]`` to ``GasSourceEnv.reset()``.
    """
    map_dir = resolve_map_dir(gaden_root, map_key)
    grid, origin = load_gaden_grid(map_dir)
    wind_field   = load_gaden_wind_field(map_dir, grid, origin)
    spec         = load_gaden_spec(yaml_path, map_key, origin, sim_id=sim_id)
    return {
        "grid":       grid,
        "source_pos": spec["source_pos"],
        "robot_pos":  spec["robot_pos"],
        "width":      grid.width,
        "height":     grid.height,
        "wind_field": wind_field,
        "origin_xy":  origin,
        "spec":       spec,
    }
