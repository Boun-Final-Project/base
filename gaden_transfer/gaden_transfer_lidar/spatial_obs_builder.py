"""
Ego-centric spatial observation builder for GADEN deployment.

Mirrors `envs/spatial_obs_wrapper.py::SpatialObsWrapper` so the same
ActorCriticSpatial weights apply directly. Returns a ``(spatial, wind)``
tuple per call to :py:meth:`build`:

    spatial : (4, 98, 98) float32 — channels [occupancy, gas, recency, det]
    wind    : (2,)        float32 — [speed/max_speed, direction/2π]

Lifecycle (must match training order exactly):

    reset()                  once per episode
    update_lidar(msg)        every LaserScan callback (latches data only)
    update_gas(raw)          every GasSensor callback (latches + binary)
    record_step()            once per policy step, called before build()
    build()                  returns the tuple above
"""

import numpy as np
from typing import Optional, Tuple

from sensor_msgs.msg import LaserScan

import sys
_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.sensor_model import BinarySensorModel
from .lidar_resampler import resample_scan


class SpatialObsBuilder:
    """Stateful spatial observation assembler for GADEN deployment."""

    GRID_SIZE = cfg.SPATIAL_GRID_SIZE           # 98
    CELL_RES  = cfg.VISITED_CELL_RESOLUTION     # 0.5 m/cell

    def __init__(self, map_width: float, map_height: float,
                 origin_x: float = 0.0, origin_y: float = 0.0):
        self.map_width = map_width
        self.map_height = map_height
        # Wrapper-cell indices live in GT-grid frame; subtract the GADEN
        # occupancy origin so non-zero origins don't shift walls relative to
        # the robot. Mirrors the patch in gaden_transfer_image_6ch.
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self._decay = float(np.exp(-cfg.SPATIAL_LAMBDA))

        self._map_h_cells = int(np.floor(map_height / self.CELL_RES))
        self._map_w_cells = int(np.floor(map_width  / self.CELL_RES))

        # World-space grids (shape = [h_cells, w_cells]); allocated in reset()
        self._occ_world: Optional[np.ndarray] = None
        self._gas_world: Optional[np.ndarray] = None
        self._rec_world: Optional[np.ndarray] = None
        self._det_world: Optional[np.ndarray] = None
        self._max_det: int = 0

        # Latest sensor state (set by callbacks on the node)
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_theta: float = 0.0
        self._latest_lidar_norm: Optional[np.ndarray] = None    # sensor-frame, [0,1]
        self._latest_binary: int = 0
        self._initialized: bool = False                          # gas threshold seeded

        # Wind is locked once from the CFD CSV (constant per GADEN map)
        self._wind_locked: bool = False
        self._locked_wind_speed: float = 0.0
        self._locked_wind_dir: float = 0.0

        # Per-episode step count + whether initial scan has been folded in
        self._step_count: int = 0
        self._seeded: bool = False

        # Binary gas sensor (adaptive threshold — same as training)
        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
            threshold_decay=cfg.SENSOR_THRESHOLD_DECAY,
        )

        # Precompute sensor-frame ray angles and per-step offsets (distances)
        self._n_rays: int = cfg.LIDAR_NUM_RAYS
        self._sensor_angles: np.ndarray = np.linspace(
            0.0, 2.0 * np.pi, self._n_rays, endpoint=False
        )
        # Sample along each ray at sub-cell granularity so we don't skip cells
        self._step_res: float = self.CELL_RES / 2.0              # 0.25 m
        self._n_steps: int = int(np.ceil(cfg.LIDAR_MAX_RANGE / self._step_res))
        self._t: np.ndarray = np.arange(1, self._n_steps + 1) * self._step_res  # (S,)

        self.reset()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self):
        """Call at the start of each episode — clears grids and gas threshold."""
        h, w = self._map_h_cells, self._map_w_cells
        self._occ_world = np.zeros((h, w), dtype=np.float32)
        self._gas_world = np.zeros((h, w), dtype=np.float32)
        self._rec_world = np.zeros((h, w), dtype=np.float32)
        self._det_world = np.zeros((h, w), dtype=np.float32)
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
        # wind lock persists across episodes (constant CFD field)

    # ------------------------------------------------------------------
    # Sensor callbacks (called by the node)
    # ------------------------------------------------------------------

    def update_lidar(self, msg: LaserScan):
        """Latch latest scan — occupancy grids update in record_step()."""
        self._latest_lidar_norm = resample_scan(
            msg, self._n_rays, cfg.LIDAR_MAX_RANGE
        )

    def update_gas(self, raw_concentration: float):
        """Update adaptive threshold + latest binary (mirrors training reset+step)."""
        if not self._initialized:
            self._sensor.initialize_threshold(raw_concentration)
            self._latest_binary = 0
            self._initialized = True
        else:
            self._sensor.update_threshold(raw_concentration)
            self._latest_binary = int(
                self._sensor.get_binary_measurement(raw_concentration)
            )

    def load_wind_from_file(self, wind_csv_path: str):
        """Compute mean wind speed / direction from GADEN CFD CSV.

        CSV columns: Points:0, Points:1, Points:2, U:0, U:1, U:2 — we average
        U:0 and U:1 as in the existing ObservationBuilder.
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
        self._locked_wind_dir = float(np.arctan2(mean_uy, mean_ux)) % (2.0 * np.pi)
        self._wind_locked = True

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def record_step(self):
        """Called once per policy step.

        First call seeds the world grids from the initial scan with binary=0,
        matching training's ``SpatialObsWrapper.reset()``. Subsequent calls
        decay recency, fold in the latest scan, and mark the robot cell with
        the latest binary reading — matching ``SpatialObsWrapper.step()``.
        """
        if self._latest_lidar_norm is None or self.robot_x is None \
                or self.robot_y is None:
            return

        if not self._seeded:
            self._update_occupancy_world(self._latest_lidar_norm)
            self._update_robot_cell(binary=0)
            self._seeded = True
        else:
            self._rec_world *= self._decay
            self._update_occupancy_world(self._latest_lidar_norm)
            self._update_robot_cell(binary=self._latest_binary)

        self._step_count += 1

    # ------------------------------------------------------------------
    # World-grid updates
    # ------------------------------------------------------------------

    def _update_occupancy_world(self, distances_norm_sensor_frame: np.ndarray):
        """Reveal free cells along each ray and mark walls at hit distances.

        Rays arrive in *sensor* frame (index 0 = robot's forward). We add
        ``robot_theta`` to each sensor angle to get the world-frame direction,
        then rasterise the ray into the world grid.
        """
        rx, ry = float(self.robot_x), float(self.robot_y)
        world_angles = self.robot_theta + self._sensor_angles              # (R,)
        cos_a = np.cos(world_angles)                                        # (R,)
        sin_a = np.sin(world_angles)                                        # (R,)

        dx = cos_a[:, None] * self._t[None, :]                              # (R, S)
        dy = sin_a[:, None] * self._t[None, :]
        wx = rx + dx
        wy = ry + dy

        wc = np.floor((wx - self.origin_x) / self.CELL_RES).astype(np.int32)  # (R, S)
        wr = np.floor((wy - self.origin_y) / self.CELL_RES).astype(np.int32)

        in_map = (wc >= 0) & (wc < self._map_w_cells) & \
                 (wr >= 0) & (wr < self._map_h_cells)

        # Number of free sub-steps per ray (distance_norm scaled to step count)
        n_free = (distances_norm_sensor_frame * self._n_steps).astype(np.int32)  # (R,)
        free_mask = np.arange(self._n_steps)[None, :] < n_free[:, None]          # (R, S)
        free_valid = free_mask & in_map

        fr, fc = wr[free_valid], wc[free_valid]
        not_wall = self._occ_world[fr, fc] != -1.0
        self._occ_world[fr[not_wall], fc[not_wall]] = 1.0

        # Wall marking: first hit step per ray (only if the ray actually hit)
        hit_mask = distances_norm_sensor_frame < 1.0                             # (R,)
        hit_s = np.clip(n_free, 0, self._n_steps - 1)                            # (R,)
        ray_idx = np.arange(self._n_rays)
        hit_wc = wc[ray_idx, hit_s]
        hit_wr = wr[ray_idx, hit_s]
        hit_valid = hit_mask & \
                    (hit_wc >= 0) & (hit_wc < self._map_w_cells) & \
                    (hit_wr >= 0) & (hit_wr < self._map_h_cells)
        self._occ_world[hit_wr[hit_valid], hit_wc[hit_valid]] = -1.0

    def _update_robot_cell(self, binary: int):
        """Update gas, recency, and detection-count grids at the robot's cell."""
        row = int(np.floor((self.robot_y - self.origin_y) / self.CELL_RES))
        col = int(np.floor((self.robot_x - self.origin_x) / self.CELL_RES))
        if not (0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells):
            return

        mapped = 1.0 if binary else -1.0
        if self._gas_world[row, col] != 1.0:
            self._gas_world[row, col] = mapped

        self._rec_world[row, col] = 1.0

        if binary:
            self._det_world[row, col] += 1.0
            val = int(self._det_world[row, col])
            if val > self._max_det:
                self._max_det = val

    # ------------------------------------------------------------------
    # Build observation
    # ------------------------------------------------------------------

    def build(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (spatial[4,98,98], wind[2]) or None if not ready."""
        if self.robot_x is None or self.robot_y is None:
            return None
        if self._latest_lidar_norm is None:
            return None
        if not self._wind_locked:
            return None
        if not self._initialized:
            return None
        if not self._seeded:
            # record_step() hasn't been called yet — no world data to embed
            return None

        robot_col = int(np.floor((self.robot_x - self.origin_x) / self.CELL_RES))
        robot_row = int(np.floor((self.robot_y - self.origin_y) / self.CELL_RES))
        ego_r0 = self.GRID_SIZE // 2 - robot_row
        ego_c0 = self.GRID_SIZE // 2 - robot_col

        def _embed(world_buf: np.ndarray, fill: float = 0.0) -> np.ndarray:
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

        occ = _embed(self._occ_world, fill=0.0)
        gas = _embed(self._gas_world, fill=0.0)
        rec = _embed(self._rec_world, fill=0.0)
        if self._max_det > 0:
            det = _embed(self._det_world / self._max_det, fill=0.0)
        else:
            det = _embed(self._det_world, fill=0.0)

        spatial = np.stack([occ, gas, rec, det], axis=0)  # (4, 98, 98)
        wind = np.array([
            self._locked_wind_speed / cfg.WIND_MAX_SPEED,
            self._locked_wind_dir / (2.0 * np.pi),
        ], dtype=np.float32)

        return spatial, wind

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def ready(self) -> bool:
        """True once all sensors have delivered at least one reading."""
        return (
            self.robot_x is not None and
            self._latest_lidar_norm is not None and
            self._wind_locked and
            self._initialized
        )
