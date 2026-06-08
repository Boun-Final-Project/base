"""
Builds the 107-dim observation vector from live ROS2 sensor data.

Mirrors gas_source_env.py::_build_observation() exactly so the same
pretrained weights apply without any re-normalisation.

Observation layout (all values in [0, 1]):
    [0:30]    gas_history  — 10 steps × (rel_x, rel_y, binary)
    [30:102]  lidar        — 72 normalised ray distances
    [102:104] position     — (x/W, y/H)
    [104:106] wind         — (speed/max_speed, direction/2π)
    [106]     time         — step / max_steps
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple

import sys, os
_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.sensor_model import BinarySensorModel
from .lidar_resampler import resample_scan
from sensor_msgs.msg import LaserScan


class ObservationBuilder:
    """Stateful observation assembler.  One instance per episode.

    Call ``reset()`` at the start of each episode, then ``build()`` after
    every sensor update to get the latest 107-dim vector.
    """

    def __init__(self, map_width: float, map_height: float):
        """
        Parameters
        ----------
        map_width, map_height : float
            Real-world dimensions of the map in metres.  Used to normalise
            robot position and gas-history relative positions.  Obtained
            from the occupancy grid service at startup.
        """
        self.map_width = map_width
        self.map_height = map_height

        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
        )

        # Rolling buffer: stores (abs_x, abs_y, binary) tuples
        self._gas_history: deque = deque(
            [(None, None, 0)] * cfg.GAS_HISTORY_LENGTH,
            maxlen=cfg.GAS_HISTORY_LENGTH,
        )

        # Latest sensor readings (set by the node's callbacks)
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_theta: float = 0.0  # radians, world-frame heading; set by node but not consumed by build()
        self.lidar_norm: Optional[np.ndarray] = None   # shape (72,), already [0,1], SENSOR frame
        self.wind_speed: Optional[float] = None
        self.wind_direction: Optional[float] = None    # radians from +x

        self._step_count: int = 0
        self._initialized: bool = False

        # Latest gas binary (set every callback, but only appended to history
        # once per policy step in `record_step()` — matches training cadence)
        self._latest_binary: int = 0

        self._wind_locked: bool = False
        self._locked_wind_speed: float = 0.0
        self._locked_wind_dir: float = 0.0

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self):
        """Call at the start of each episode."""
        self._gas_history = deque(
            [(None, None, 0)] * cfg.GAS_HISTORY_LENGTH,
            maxlen=cfg.GAS_HISTORY_LENGTH,
        )
        self._step_count = 0
        self._initialized = False
        self._sensor = BinarySensorModel(
            alpha=cfg.SENSOR_ALPHA,
            sigma_env=cfg.SENSOR_SIGMA_ENV,
            threshold_weight=cfg.SENSOR_THRESHOLD_WEIGHT,
        )
        # Wind lock persists across episodes (same wind field)

    # ------------------------------------------------------------------
    # Called by the node's sensor callbacks
    # ------------------------------------------------------------------

    def update_lidar(self, msg: LaserScan):
        """Process a raw LaserScan into the normalised 72-ray array."""
        self.lidar_norm = resample_scan(msg, cfg.LIDAR_NUM_RAYS, cfg.LIDAR_MAX_RANGE)

    def update_gas(self, raw_concentration: float):
        """Update adaptive threshold + cache latest binary reading.

        Does NOT append to gas_history — that happens once per policy step
        via ``record_step()`` to match the training cadence (1 entry/step).
        """
        if not self._initialized:
            self._sensor.initialize_threshold(raw_concentration)
            self._latest_binary = 0
            self._initialized = True
            # Seed history with initial entry, like training's reset()
            if self.robot_x is not None and self.robot_y is not None:
                self._gas_history.append((self.robot_x, self.robot_y, 0))
        else:
            self._sensor.update_threshold(raw_concentration)
            self._latest_binary = self._sensor.get_binary_measurement(raw_concentration)

    def load_wind_from_file(self, wind_csv_path: str):
        """Compute mean wind speed and direction from a GADEN wind CSV file.

        The CSV has columns: Points:0, Points:1, Points:2, U:0, U:1, U:2
        where U:0 and U:1 are the x/y wind velocity components (m/s).
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
        self._locked_wind_speed = float(np.sqrt(mean_ux**2 + mean_uy**2))
        self._locked_wind_dir = float(np.arctan2(mean_uy, mean_ux)) % (2.0 * np.pi)
        self._wind_locked = True

    def record_step(self):
        """Append latest gas reading + increment step counter.

        Call once per policy step (after teleport, before next obs build).
        Mirrors training, which appends one (pos, binary) tuple per env.step().
        """
        if self.robot_x is not None and self.robot_y is not None:
            self._gas_history.append(
                (self.robot_x, self.robot_y, self._latest_binary)
            )
        self._step_count += 1

    # ------------------------------------------------------------------
    # Build observation
    # ------------------------------------------------------------------

    def build(self) -> Optional[np.ndarray]:
        """Assemble and return the 107-dim observation vector.

        Returns ``None`` if any required sensor has not yet received its
        first message (robot pose, lidar, wind).
        """
        if self.robot_x is None or self.robot_y is None:
            return None
        if self.lidar_norm is None:
            return None
        if not self._wind_locked:
            return None

        # --- Gas history (30 dims) ---
        gas_entries = []
        for ax, ay, b in self._gas_history:
            if ax is None:
                gas_entries.extend([0.5, 0.5, 0.0])
            else:
                rel_x = 0.5 + (ax - self.robot_x) / (2.0 * self.map_width)
                rel_y = 0.5 + (ay - self.robot_y) / (2.0 * self.map_height)
                gas_entries.extend([
                    float(np.clip(rel_x, 0.0, 1.0)),
                    float(np.clip(rel_y, 0.0, 1.0)),
                    float(b),
                ])
        gas = np.array(gas_entries, dtype=np.float32)  # (30,)

        # --- LiDAR (72 dims) — sensor-frame, matches training convention ---
        # lidar_norm slot-0 = robot forward, matching gas_source_env's
        # LidarSim.scan(heading=robot_heading) convention.
        lidar = self.lidar_norm.astype(np.float32)

        # --- Position (2 dims) ---
        pos = np.array([
            self.robot_x / self.map_width,
            self.robot_y / self.map_height,
        ], dtype=np.float32)

        # --- Wind (2 dims) — use locked mean from episode start ---
        wind = np.array([
            self._locked_wind_speed / cfg.WIND_MAX_SPEED,
            self._locked_wind_dir / (2.0 * np.pi),
        ], dtype=np.float32)

        # --- Time (1 dim) ---
        time_frac = np.array(
            [self._step_count / cfg.MAX_STEPS], dtype=np.float32
        )

        obs = np.concatenate([gas, lidar, pos, wind, time_frac])  # (107,)
        return np.clip(obs, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def ready(self) -> bool:
        """True once all sensors have provided at least one reading."""
        return (self.robot_x is not None and
                self.lidar_norm is not None and
                self._wind_locked and
                self._initialized)
