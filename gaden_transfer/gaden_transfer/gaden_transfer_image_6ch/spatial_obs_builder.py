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
    """6-channel ego-centric spatial observation assembler (image arch).

    Channels: [is_known, is_wall, gas, recency, det_count, motion].
    The trailing ``motion`` channel is a fast-decay (≈0.6/step) trail of the
    robot's recent positions. For backward-compat with 5-channel checkpoints
    (efe_0_2_wall_*), the channel is the LAST one — drop it via ``[:5]``
    slicing if loading the old architecture.
    """

    GRID_SIZE = cfg.SPATIAL_GRID_SIZE   # 98 cells — ego half-width = 49
    # rl_osl tightens wrapper-cell resolution from 0.5 m → 0.2 m. Hard-coded
    # here so the in-tree shared cfg.VISITED_CELL_RESOLUTION can stay at 0.5
    # for the older gaden_transfer_image_5ch (rl_new) path.
    CELL_RES  = 0.2
    MOTION_DECAY = 0.6  # per-policy-step decay for the motion-trail channel

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
        self._known_world:  Optional[np.ndarray] = None
        self._wall_world:   Optional[np.ndarray] = None
        self._gas_world:    Optional[np.ndarray] = None
        self._rec_world:    Optional[np.ndarray] = None
        self._det_world:    Optional[np.ndarray] = None
        self._motion_world: Optional[np.ndarray] = None  # 6th channel: motion trail
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

        # CELL_RES must be an integer multiple of the GT grid resolution so we
        # can max-pool sub-cells into wrapper cells without rounding.
        ratio = self.CELL_RES / true_res
        self._sub = int(round(ratio))
        if abs(ratio - self._sub) > 1e-6:
            raise ValueError(
                f"CELL_RES ({self.CELL_RES}) must be an integer multiple of "
                f"grid resolution ({true_res}); got ratio {ratio}"
            )

        # _dense_wall: precomputed any-subcell-is-wall map at wrapper
        # resolution. With CELL_RES=0.2 m and GT=0.1 m, each wrapper cell
        # pools a 2×2 GT block; if any sub-cell is occupied (==1), the
        # wrapper cell is a wall. This replaces the old center-only check
        # which missed thin walls aligned off-center.
        self._dense_wall: Optional[np.ndarray] = None
        # Cache invalidator for SLAM mode: hash of the grid at last
        # precomputation. Recompute when it changes.
        self._dense_wall_grid_id: int = -1

        self.reset()
        self._refresh_dense_wall()  # initial pass over occ_map

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self):
        h, w = self._map_h_cells, self._map_w_cells
        self._known_world  = np.zeros((h, w), dtype=np.float32)
        self._wall_world   = np.zeros((h, w), dtype=np.float32)
        self._gas_world    = np.zeros((h, w), dtype=np.float32)
        self._rec_world    = np.zeros((h, w), dtype=np.float32)
        self._det_world    = np.zeros((h, w), dtype=np.float32)
        self._motion_world = np.zeros((h, w), dtype=np.float32)
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
            self._update_motion((self.robot_x, self.robot_y))
            self._seeded = True
        else:
            # Subsequent calls are wrapper.step(): env.step has incremented
            # _current_step, then wrapper does decay → reveal → update.
            self._rec_world *= self._decay
            self._reveal((self.robot_x, self.robot_y))
            self._update_robot_cell(
                (self.robot_x, self.robot_y), binary=self._latest_binary
            )
            self._update_motion((self.robot_x, self.robot_y))
            self._step_count += 1

    # ------------------------------------------------------------------
    # Reveal + robot-cell update (world-frame, mirrors training wrapper)
    # ------------------------------------------------------------------

    def _refresh_dense_wall(self):
        """Recompute the wrapper-resolution wall map from the current GT/SLAM
        grid via max-pooling. Each wrapper cell is marked as wall if ANY of
        its sub-cells is occupied (==1). Cheap: O(gw·gh).

        The cache key is `id(grid) | grid.tobytes-hash` — for in-place
        SLAM grid mutations the array id stays the same, so we hash a
        coarse digest. In practice we just recompute every call (it's <1 ms
        for typical 104×64 grids).
        """
        gt = self._occ_map.grid
        sub = self._sub
        h = self._map_h_cells
        w = self._map_w_cells
        gh, gw = gt.shape
        gh_pad = ((gh + sub - 1) // sub) * sub
        gw_pad = ((gw + sub - 1) // sub) * sub
        if gh_pad != gh or gw_pad != gw:
            padded = np.zeros((gh_pad, gw_pad), dtype=gt.dtype)
            padded[:gh, :gw] = gt
        else:
            padded = gt
        pooled = padded.reshape(gh_pad // sub, sub, gw_pad // sub, sub).max(axis=(1, 3))
        # SLAM grid uses -1 unknown / 0 free / 1 wall / 2 outlet.
        # For walls only, check == 1. (max() of {-1, 0, 1, 2} → up to 2; we
        # specifically want the "is there a wall sub-cell" answer.)
        # Use np.any along sub-axes for "any sub-cell == 1":
        wall_subs = (padded == 1)
        pooled_wall = wall_subs.reshape(gh_pad // sub, sub, gw_pad // sub, sub).any(axis=(1, 3))
        self._dense_wall = pooled_wall[:h, :w]

    def _reveal(self, robot_pos):
        """Reveal wrapper cells within reveal_radius using GT grid + occlusion.

        For each 0.5 m wrapper cell around the robot, cast a ray from the
        robot to the cell centre through the true 0.1 m GT grid; if any wall
        sits strictly between them, the cell stays unknown. Otherwise stamp
        ``is_known=1`` and ``is_wall = GT-wall-at-centre``.
        """
        rx, ry = robot_pos
        rc = self._reveal_r_cells
        # Wrapper-cell indices are in GT-grid frame (matches _dense_wall and
        # the world buffers). Subtract origin so non-zero GADEN origins don't
        # silently shift walls relative to the robot.
        ox_w = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
        oy_w = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
        cx_cell = int(np.floor((rx - ox_w) / self.CELL_RES))
        cy_cell = int(np.floor((ry - oy_w) / self.CELL_RES))

        rows = np.arange(cy_cell - rc, cy_cell + rc + 1)  # (H,)
        cols = np.arange(cx_cell - rc, cx_cell + rc + 1)  # (W,)

        # Cell centres in world frame (origin-shifted back).
        tx = ox_w + (cols + 0.5) * self.CELL_RES  # (W,)
        ty = oy_w + (rows + 0.5) * self.CELL_RES  # (H,)
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
        ox = ox_w
        oy = oy_w
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
        # wall-centred target occludes itself). Wrapper indices use GT-grid
        # frame, so subtract origin.
        sample_cell_r = np.floor((sy - oy_w) / self.CELL_RES).astype(np.int32)
        sample_cell_c = np.floor((sx - ox_w) / self.CELL_RES).astype(np.int32)
        same_as_target = (sample_cell_r == rows[:, None, None]) & \
                         (sample_cell_c == cols[None, :, None])

        occluded = np.any(sample_wall & before_target & ~same_as_target, axis=2)
        visible = candidate & ~occluded

        # --- Wall labelling: use precomputed dense_wall (any-subcell-is-wall)
        # at wrapper resolution. Replaces the old center-cell check which
        # missed thin walls aligned off-center.
        self._refresh_dense_wall()
        rows_safe = np.clip(rows, 0, self._map_h_cells - 1)
        cols_safe = np.clip(cols, 0, self._map_w_cells - 1)
        rows_valid = (rows >= 0) & (rows < self._map_h_cells)
        cols_valid = (cols >= 0) & (cols < self._map_w_cells)
        cell_is_wall = self._dense_wall[rows_safe[:, None], cols_safe[None, :]] \
                       & (rows_valid[:, None] & cols_valid[None, :])

        # --- Observed mask: any-subcell semantics so thin walls aren't
        # filtered out. SLAM ray-tracing typically marks only one sub-cell
        # of a wrapper cell as wall (==1) while leaving its neighbors at -1
        # (unknown). The center-only check would then say "not observed" and
        # drop the wall stamp entirely. Pool max(grid != -1) over sub-cells:
        # if any sub-cell has been observed, the wrapper cell is observed.
        gt = self._occ_map.grid
        sub = self._sub
        gh_pad = ((gh + sub - 1) // sub) * sub
        gw_pad = ((gw + sub - 1) // sub) * sub
        if gh_pad != gh or gw_pad != gw:
            obs_padded = np.full((gh_pad, gw_pad), -1, dtype=gt.dtype)
            obs_padded[:gh, :gw] = gt
        else:
            obs_padded = gt
        observed_subs = (obs_padded != -1)
        observed_pooled = observed_subs.reshape(
            gh_pad // sub, sub, gw_pad // sub, sub
        ).any(axis=(1, 3))[:self._map_h_cells, :self._map_w_cells]
        cell_is_observed = observed_pooled[rows_safe[:, None], cols_safe[None, :]] \
                           & (rows_valid[:, None] & cols_valid[None, :])

        stampable = visible & cell_is_observed
        vr, vc = np.where(stampable)
        world_r = rows[vr]
        world_c = cols[vc]
        # is_known is cumulative: once observed, stays observed.
        self._known_world[world_r, world_c] = 1.0
        self._wall_world [world_r, world_c] = cell_is_wall[vr, vc].astype(np.float32)

        # Wall-edge propagation: a `_dense_wall` cell adjacent to any observed
        # free cell is implicitly known too — the policy can otherwise see the
        # near face of a U-wall but the far face stays unstamped because each
        # wall cell occludes its own neighbours along the line of sight.
        # Dilate the (known & free) region by one wrapper cell into dense_wall.
        free_known = (self._known_world > 0) & (~self._dense_wall)
        adj = np.zeros_like(free_known)
        adj[1:, :]  |= free_known[:-1, :]
        adj[:-1, :] |= free_known[1:, :]
        adj[:, 1:]  |= free_known[:, :-1]
        adj[:, :-1] |= free_known[:, 1:]
        edge = adj & self._dense_wall & (self._known_world == 0)
        if edge.any():
            self._known_world[edge] = 1.0
            self._wall_world [edge] = 1.0

    def _update_motion(self, robot_pos):
        """Decay motion-trail channel and stamp the current cell. Mirror of
        SpatialObsWrapper._update_motion. Origin-aware indexing (matches
        the rest of this file post-origin-fix)."""
        self._motion_world *= self.MOTION_DECAY
        ox_w = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
        oy_w = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
        row = int(np.floor((robot_pos[1] - oy_w) / self.CELL_RES))
        col = int(np.floor((robot_pos[0] - ox_w) / self.CELL_RES))
        if 0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells:
            self._motion_world[row, col] = 1.0

    def _update_robot_cell(self, robot_pos, binary: int):
        """Update gas / recency / det_count; on detection splat to centre+4-nbrs."""
        ox_w = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
        oy_w = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
        row = int(np.floor((robot_pos[1] - oy_w) / self.CELL_RES))
        col = int(np.floor((robot_pos[0] - ox_w) / self.CELL_RES))
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

        ox_w = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
        oy_w = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
        robot_col = int(np.floor((self.robot_x - ox_w) / self.CELL_RES))
        robot_row = int(np.floor((self.robot_y - oy_w) / self.CELL_RES))
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

        known  = _embed(self._known_world,  fill=0.0)
        wall   = _embed(self._wall_world,   fill=0.0)
        gas    = _embed(self._gas_world,    fill=0.0)
        rec    = _embed(self._rec_world,    fill=0.0)
        if self._max_det > 0:
            det = _embed(self._det_world / self._max_det, fill=0.0)
        else:
            det = _embed(self._det_world, fill=0.0)
        # NOTE: motion channel disabled to match the 5-ch checkpoint
        # (efe_0_2_wall_*). _motion_world is still maintained for forward-compat.
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
