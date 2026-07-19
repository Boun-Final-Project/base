"""Latest-value cache for the four sensor topics; snapshot() mints an Observation.

Async sensor rates are hidden from the agent: every callback overwrites the latest
value here, and the decision loop takes one coherent snapshot per step.
"""
import math
import time
from typing import Optional

import numpy as np

from gsl_bench.agent import MapInfo, Observation


class SensorCache:
    """Plain (non-node) holder of the freshest reading from each sensor."""

    def __init__(self, n_rays: int = 72, lidar_max: float = 3.0):
        self.n_rays = int(n_rays)
        self.lidar_max = float(lidar_max)

        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.theta: float = 0.0

        self.gas_ppm: float = 0.0
        self._gas_seen: bool = False
        self.wind_speed: float = 0.0
        self.wind_direction: float = 0.0

        self.lidar: Optional[np.ndarray] = None
        self.lidar_msg = None
        self.lidar_min: Optional[float] = None
        self.last_scan_stamp_ns: int = 0

        # Wall-clock liveness beat for the STEADY_TIME watchdog (never sim time:
        # a dead simulator freezes /clock, which is exactly what we must detect).
        self.last_pose_wall: Optional[float] = None

        self.travel_distance_m: float = 0.0
        self._last_xy: Optional[tuple] = None

    # -- callbacks -------------------------------------------------------

    def update_pose(self, msg, set_position=True):
        """geometry_msgs/PoseWithCovarianceStamped from /<ns>/ground_truth.

        In SLAM/TF mode (pose_source == 'tf') the caller passes set_position=False
        so ground-truth world coordinates never leak into self.x/y.  Bookkeeping
        fields (last_pose_wall, theta, travel_distance) are always updated.
        """
        self.last_pose_wall = time.monotonic()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        if set_position:
            self.theta = math.atan2(siny_cosp, cosy_cosp)

        # Travelled path length. The d < 1.0 guard skips the one initial-placement
        # jump, which is a teleport, not travel.
        if self._last_xy is not None:
            d = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if d < 1.0:
                self.travel_distance_m += d
        self._last_xy = (x, y)

        if set_position:
            self.x = x
            self.y = y

    def update_gas(self, msg):
        """olfaction_msgs/GasSensor from /fake_pid/Sensor_reading."""
        self.gas_ppm = float(msg.raw)
        self._gas_seen = True

    def update_wind(self, msg):
        """olfaction_msgs/Anemometer from /fake_anemometer/WindSensor_reading.

        fake_anemometer.cpp publishes wind_speed = u*u + v*v — the SQUARED
        magnitude, it never takes the sqrt. Training and Python eval feed the true
        magnitude, so the correction happens here, once, for every agent.
        """
        self.wind_speed = math.sqrt(max(0.0, float(msg.wind_speed)))
        self.wind_direction = float(msg.wind_direction)

    def update_lidar(self, msg):
        """sensor_msgs/LaserScan from /<ns>/laser_scanner."""
        st = msg.header.stamp
        self.last_scan_stamp_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
        rng = np.asarray(msg.ranges, dtype=np.float32)
        finite = np.isfinite(rng)
        self.lidar_min = float(rng[finite].min()) if finite.any() else None
        rng = np.where(finite, rng, self.lidar_max)
        self.lidar = np.clip(rng, 0.0, self.lidar_max)
        self.lidar_msg = msg

    # -- snapshot --------------------------------------------------------

    def ready(self) -> bool:
        """All sensors needed to build an Observation have reported at least once."""
        return self.x is not None and self.lidar is not None and self._gas_seen

    def snapshot(self, step: int, sim_time: float, map_info: MapInfo) -> Observation:
        return Observation(
            x=float(self.x),
            y=float(self.y),
            theta=float(self.theta),
            gas_ppm=float(self.gas_ppm),
            wind_speed=float(self.wind_speed),
            wind_direction=float(self.wind_direction),
            lidar=self.lidar.copy(),
            lidar_max_range=self.lidar_max,
            step=int(step),
            sim_time=float(sim_time),
            map=map_info,
            lidar_msg=self.lidar_msg,
        )
