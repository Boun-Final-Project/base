"""
5-channel spatial observation builder for GADEN deployment (image arch).

Mirrors rl_new/envs/spatial_obs_wrapper.py so the same ActorCriticSpatial
weights apply. Returns ``(spatial, ctx)`` per call to :py:meth:`build`:

    spatial : (5, 98, 98) float32 — [is_known, is_wall, gas, recency, det_count]
    ctx     : (4,)        float32 — [speed/max, cos(dir), sin(dir), step/MAX_STEPS]

Occupancy is revealed from the GADEN ground-truth grid (obtained at startup
from ``/gaden_environment/occupancyMap3D``) with a per-cell visibility check
— physically equivalent to a LiDAR-built map but at wrapper resolution, with
no quantisation-overwrite or dotted-wall artefacts.

Lifecycle (one call each per episode):
    reset()                  clear world grids
    load_wind_from_file(csv) compute mean wind speed and direction
    update_gas(raw)          every GasSensor callback (latches binary)
    record_step()            once per policy step (before build())
    build()                  returns the tuple above
"""

from typing import Optional, Tuple

import numpy as np

import sys
_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.sensor_model import BinarySensorModel


class SpatialObsBuilder:
    """5-channel ego-centric spatial observation assembler (image arch)."""

    GRID_SIZE = cfg.SPATIAL_GRID_SIZE         # 98 cells — ego half-width = 49
    CELL_RES  = cfg.VISITED_CELL_RESOLUTION   # 0.5 m per wrapper cell

    def __init__(self, occ_map, map_width: float, map_height: float):
        """
        Parameters
        ----------
        occ_map
            ``efe_igdm.mapping.OccupancyGridMap`` — the GADEN ground truth
            already loaded by the node (``self._occ_map``). Exposes
            ``.grid``, ``.resolution``, ``.grid_width/height``, ``.origin_x/y``.
        map_width, map_height
            Map extents in metres (``occ_map.real_world_width/height``).
        """
        self._occ_map = occ_map
        self.map_width = map_width
        self.map_height = map_height
        self._decay = float(np.exp(-cfg.SPATIAL_LAMBDA))

        self._map_h_cells = int(np.floor(map_height / self.CELL_RES))
        self._map_w_cells = int(np.floor(map_width  / self.CELL_RES))

        # World-space channels (allocated in reset())
        self._known_world: Optional[np.ndarray] = None
        self._wall_world:  Optional[np.ndarray] = None
        self._gas_world:   Optional[np.ndarray] = None
        self._rec_world:   Optional[np.ndarray] = None
        self._det_world:   Optional[np.ndarray] = None
        self._max_det: int = 0

        # Live sensor state (set by the node)
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self._latest_binary: int = 0
        self._initialized: bool = False      # gas threshold seeded

        # Wind lock (constant CFD field per GADEN map)
        self._wind_locked: bool = False
        self._locked_wind_speed: float = 0.0
        self._locked_wind_dir: float = 0.0

        self._step_count: int = 0
        self._seeded: bool = False

        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
            threshold_decay=cfg.SENSOR_THRESHOLD_DECAY,
        )

        # Pre-compute ray sample distances at true-grid resolution (for the
        # visibility check). ``_reveal_radius_m`` matches training's 3 m.
        self._reveal_radius_m = float(cfg.LIDAR_MAX_RANGE)
        self._reveal_r_cells  = int(np.ceil(self._reveal_radius_m / self.CELL_RES))
        true_res = float(occ_map.resolution)
        self._true_res = true_res
        n_steps = int(np.ceil(self._reveal_radius_m / true_res))
        self._ray_samples_t = np.arange(1, n_steps + 1) * true_res  # (S,)

        self.reset()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self):
        h, w = self._map_h_cells, self._map_w_cells
        self._known_world = np.zeros((h, w), dtype=np.float32)
        self._wall_world  = np.zeros((h, w), dtype=np.float32)
        self._gas_world   = np.zeros((h, w), dtype=np.float32)
        self._rec_world   = np.zeros((h, w), dtype=np.float32)
        self._det_world   = np.zeros((h, w), dtype=np.float32)
        self._max_det = 0
        self._step_count = 0
        self._seeded = False
        self._initialized = False
        self._latest_binary = 0
        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
            threshold_decay=cfg.SENSOR_THRESHOLD_DECAY,
        )
        # wind lock persists across episodes (constant field)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def update_gas(self, raw_concentration: float):
        if not self._initialized:
            self._sensor.initialize_threshold(raw_concentration)
            self._latest_binary = 0
            self._initialized = True
        else:
            self._sensor.update_threshold(raw_concentration)
            self._latest_binary = int(
                self._sensor.get_binary_measurement(raw_concentration)
            )

    def update_lidar(self, msg):
        """Accept live laser scans for API compatibility with the 4-ch builder.

        The image arch reveals occupancy from the ground-truth grid, so live
        scans aren't needed to build the observation. Kept as a no-op so the
        node can call this unconditionally and the builder's ``ready`` check
        is simpler.
        """
        pass

    def load_wind_from_file(self, wind_csv_path: str):
        """CSV-mean mode: compute wind (speed, direction) as the mean over
        all cells of a GADEN CFD file. Set once at startup; never updates.
        """
        import csv
        ux_vals, uy_vals = [], []
        with open(wind_csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ux_vals.append(float(row['U:0']))
                uy_vals.append(float(row['U:1']))
        mean_ux = float(np.mean(ux_vals))
        mean_uy = float(np.mean(uy_vals))
        self._locked_wind_speed = float(np.sqrt(mean_ux ** 2 + mean_uy ** 2))
        self._locked_wind_dir   = float(np.arctan2(mean_uy, mean_ux)) % (2.0 * np.pi)
        self._wind_locked = True

    def update_wind_live(self, speed: float, direction: float):
        """Live mode: overwrite the wind context every anemometer reading.

        The policy has no recurrence, so each step's action is computed from
        the most-recently-received wind. Matches training semantics ("what
        the robot feels at this moment") since training's wind was uniform
        across space → local wind = global mean there anyway.
        """
        self._locked_wind_speed = float(speed)
        self._locked_wind_dir = float(direction) % (2.0 * np.pi)
        self._wind_locked = True

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def record_step(self):
        """Called once per policy step.

        First call seeds the world grids from the initial reveal with binary=0
        (matches ``SpatialObsWrapper.reset()``). Subsequent calls decay
        recency, re-reveal from the new pose, and splat the latest binary
        (matches ``SpatialObsWrapper.step()``).
        """
        if self.robot_x is None or self.robot_y is None:
            return

        if not self._seeded:
            # First call is the training-equivalent of wrapper.reset(): reveal
            # and mark start cell with binary=0, time_frac stays at 0 for the
            # first observation (matches env._current_step = 0 at reset).
            self._reveal((self.robot_x, self.robot_y))
            self._update_robot_cell((self.robot_x, self.robot_y), binary=0)
            self._seeded = True
        else:
            # Subsequent calls are wrapper.step(): env.step has incremented
            # _current_step, then wrapper does decay → reveal → update.
            self._rec_world *= self._decay
            self._reveal((self.robot_x, self.robot_y))
            self._update_robot_cell(
                (self.robot_x, self.robot_y), binary=self._latest_binary
            )
            self._step_count += 1

    # ------------------------------------------------------------------
    # Reveal + robot-cell update (world-frame, mirrors training wrapper)
    # ------------------------------------------------------------------

    def _reveal(self, robot_pos):
        """Reveal wrapper cells within reveal_radius using GT grid + occlusion.

        For each 0.5 m wrapper cell around the robot, cast a ray from the
        robot to the cell centre through the true 0.1 m GT grid; if any wall
        sits strictly between them, the cell stays unknown. Otherwise stamp
        ``is_known=1`` and ``is_wall = GT-wall-at-centre``.
        """
        rx, ry = robot_pos
        rc = self._reveal_r_cells
        cx_cell = int(np.floor(rx / self.CELL_RES))
        cy_cell = int(np.floor(ry / self.CELL_RES))

        rows = np.arange(cy_cell - rc, cy_cell + rc + 1)  # (H,)
        cols = np.arange(cx_cell - rc, cx_cell + rc + 1)  # (W,)

        tx = (cols + 0.5) * self.CELL_RES  # (W,)
        ty = (rows + 0.5) * self.CELL_RES  # (H,)
        WX, WY = np.meshgrid(tx, ty)        # (H, W)

        dx = WX - rx
        dy = WY - ry
        dist = np.sqrt(dx * dx + dy * dy)

        in_radius = dist <= self._reveal_radius_m
        in_map = ((rows >= 0) & (rows < self._map_h_cells))[:, None] & \
                 ((cols >= 0) & (cols < self._map_w_cells))[None, :]
        candidate = in_radius & in_map

        # Occlusion check on the true grid.
        true_res = self._true_res
        gw = self._occ_map.grid_width
        gh = self._occ_map.grid_height
        ox = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
        oy = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
        t = self._ray_samples_t  # (S,)

        safe_d = np.where(dist > 0, dist, 1.0)
        udx = dx / safe_d
        udy = dy / safe_d

        sx = rx + udx[..., None] * t[None, None, :]  # (H, W, S)
        sy = ry + udy[..., None] * t[None, None, :]

        before_target = t[None, None, :] < dist[..., None]

        gx = np.floor((sx - ox) / true_res).astype(np.int32)
        gy = np.floor((sy - oy) / true_res).astype(np.int32)
        in_bounds = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx_safe = np.clip(gx, 0, gw - 1)
        gy_safe = np.clip(gy, 0, gh - 1)

        # A cell is a wall only if it's explicitly CELL_OCCUPIED (1). Unknown
        # cells (-1, from SLAM-built maps) and outlets (2) don't occlude — we
        # don't yet know they're walls. For GT maps this is equivalent to the
        # old `!= 0` check since GT only contains {0, 1}.
        sample_wall = (self._occ_map.grid[gy_safe, gx_safe] == 1) & in_bounds

        # Mask out samples inside the target's own wrapper cell (otherwise a
        # wall-centred target occludes itself).
        sample_cell_r = np.floor(sy / self.CELL_RES).astype(np.int32)
        sample_cell_c = np.floor(sx / self.CELL_RES).astype(np.int32)
        same_as_target = (sample_cell_r == rows[:, None, None]) & \
                         (sample_cell_c == cols[None, :, None])

        occluded = np.any(sample_wall & before_target & ~same_as_target, axis=2)
        visible = candidate & ~occluded

        cx = np.floor((WX - ox) / true_res).astype(np.int32)
        cy = np.floor((WY - oy) / true_res).astype(np.int32)
        cx_safe = np.clip(cx, 0, gw - 1)
        cy_safe = np.clip(cy, 0, gh - 1)
        centre_in = (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
        cell_val = self._occ_map.grid[cy_safe, cx_safe]
        cell_is_wall     = (cell_val == 1) & centre_in
        # Observed = SLAM/GT has a concrete value (not -1 unknown). For GT
        # maps this is equivalent to centre_in.
        cell_is_observed = (cell_val != -1) & centre_in

        # Only stamp cells that are visible AND SLAM has actually observed.
        stampable = visible & cell_is_observed
        vr, vc = np.where(stampable)
        world_r = rows[vr]
        world_c = cols[vc]
        # is_known is cumulative: once observed, stays observed.
        self._known_world[world_r, world_c] = 1.0
        self._wall_world [world_r, world_c] = cell_is_wall[vr, vc].astype(np.float32)

        # Wall-edge propagation: a wrapper cell whose underlying GT/SLAM grid
        # contains ANY wall sub-cell, AND that's adjacent to an observed-free
        # wrapper cell, gets stamped wall=1. This fills the dotted-wall pattern
        # caused by per-step center-only stamping and ray self-occlusion.
        # Compute the any-subcell-is-wall mask at wrapper resolution on the fly.
        sub = int(round(self.CELL_RES / true_res))
        gh_pad = ((gh + sub - 1) // sub) * sub
        gw_pad = ((gw + sub - 1) // sub) * sub
        if gh_pad != gh or gw_pad != gw:
            padded = np.zeros((gh_pad, gw_pad), dtype=self._occ_map.grid.dtype)
            padded[:gh, :gw] = self._occ_map.grid
        else:
            padded = self._occ_map.grid
        wall_subs = (padded == 1)
        dense_wall = wall_subs.reshape(gh_pad // sub, sub,
                                       gw_pad // sub, sub).any(axis=(1, 3))
        dense_wall = dense_wall[:self._map_h_cells, :self._map_w_cells]

        free_known = (self._known_world > 0) & (~dense_wall)
        adj = np.zeros_like(free_known)
        adj[1:, :]  |= free_known[:-1, :]
        adj[:-1, :] |= free_known[1:, :]
        adj[:, 1:]  |= free_known[:, :-1]
        adj[:, :-1] |= free_known[:, 1:]
        edge = adj & dense_wall & (self._known_world == 0)
        if edge.any():
            self._known_world[edge] = 1.0
            self._wall_world [edge] = 1.0

    def _update_robot_cell(self, robot_pos, binary: int):
        """Update gas / recency / det_count; on detection splat to centre+4-nbrs."""
        row = int(np.floor(robot_pos[1] / self.CELL_RES))
        col = int(np.floor(robot_pos[0] / self.CELL_RES))
        if not (0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells):
            return

        mapped = 1.0 if binary else -1.0
        if self._gas_world[row, col] != 1.0:
            self._gas_world[row, col] = mapped

        self._rec_world[row, col] = 1.0

        if binary:
            # 5-cell plus splat (matches training).
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = row + dr, col + dc
                if 0 <= r < self._map_h_cells and 0 <= c < self._map_w_cells:
                    self._gas_world[r, c] = 1.0
                    self._det_world[r, c] += 1.0
                    val = int(self._det_world[r, c])
                    if val > self._max_det:
                        self._max_det = val

    # ------------------------------------------------------------------
    # Build observation
    # ------------------------------------------------------------------

    def build(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.robot_x is None or self.robot_y is None:
            return None
        if not self._wind_locked:
            return None
        if not self._initialized:
            return None
        if not self._seeded:
            return None

        robot_col = int(np.floor(self.robot_x / self.CELL_RES))
        robot_row = int(np.floor(self.robot_y / self.CELL_RES))
        ego_r0 = self.GRID_SIZE // 2 - robot_row
        ego_c0 = self.GRID_SIZE // 2 - robot_col

        def _embed(world_buf, fill=0.0):
            out = np.full(
                (self.GRID_SIZE, self.GRID_SIZE), fill, dtype=np.float32
            )
            dr0 = max(0, ego_r0)
            dc0 = max(0, ego_c0)
            dr1 = min(self.GRID_SIZE, ego_r0 + self._map_h_cells)
            dc1 = min(self.GRID_SIZE, ego_c0 + self._map_w_cells)
            if dr1 <= dr0 or dc1 <= dc0:
                return out
            sr0 = dr0 - ego_r0
            sc0 = dc0 - ego_c0
            out[dr0:dr1, dc0:dc1] = world_buf[
                sr0:sr0 + (dr1 - dr0), sc0:sc0 + (dc1 - dc0)
            ]
            return out

        known = _embed(self._known_world, fill=0.0)
        wall  = _embed(self._wall_world,  fill=0.0)
        gas   = _embed(self._gas_world,   fill=0.0)
        rec   = _embed(self._rec_world,   fill=0.0)
        if self._max_det > 0:
            det = _embed(self._det_world / self._max_det, fill=0.0)
        else:
            det = _embed(self._det_world, fill=0.0)

        spatial = np.stack([known, wall, gas, rec, det], axis=0)

        time_frac = min(self._step_count / cfg.MAX_STEPS, 1.0)
        ctx = np.array([
            self._locked_wind_speed / cfg.WIND_MAX_SPEED,
            float(np.cos(self._locked_wind_dir)),
            float(np.sin(self._locked_wind_dir)),
            time_frac,
        ], dtype=np.float32)

        return spatial, ctx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def ready(self) -> bool:
        """True once gas and wind are both initialised. Occupancy reveal
        doesn't need live sensor data — it uses the GT grid."""
        return (
            self.robot_x is not None and
            self._wind_locked and
            self._initialized
        )
