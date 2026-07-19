"""Public agent API for gsl_bench. Third parties import ONLY from this module.

Implement observe() and act(); the harness owns simulation, motion execution,
timeouts, collision handling and scoring.

    from gsl_bench.agent import GSLAgent, Observation, Waypoint

    class MyAgent(GSLAgent):
        def observe(self, obs): self.obs = obs
        def act(self):          return Waypoint(self.obs.x + 0.5, self.obs.y)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class MapInfo:
    """Static occupancy map, given once at reset and repeated in every Observation."""

    grid: np.ndarray          # (H, W) uint8: 0 = free, non-zero = occupied
    resolution: float         # metres per cell
    origin_x: float           # world x of grid[0, 0]'s cell corner
    origin_y: float
    width_m: float            # real-world width  (W * resolution)
    height_m: float           # real-world height (H * resolution)

    def is_free(self, x: float, y: float) -> bool:
        """True if world point (x, y) is inside the map and in a free cell."""
        i = int((y - self.origin_y) / self.resolution)
        j = int((x - self.origin_x) / self.resolution)
        if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:
            return self.grid[i, j] == 0
        return False


@dataclass
class ScenarioInfo:
    """Episode context handed to reset(). NEVER contains the source position."""

    name: str                 # e.g. "4_rooms_start_a"
    map: MapInfo
    start_x: float
    start_y: float
    max_steps: int            # step budget (e.g. 600)
    step_size_hint: float     # the harness caps hops at ~2x this (default 0.5 m)


@dataclass
class Observation:
    """Snapshot of every sensor at one decision step."""

    x: float                  # robot pose, map frame
    y: float
    theta: float              # rad, world frame, from the configured pose source
    gas_ppm: float            # latest PID reading (0.0 until the first message)
    wind_speed: float         # m/s — already sqrt-corrected by the harness
    wind_direction: float     # rad from world +x
    lidar: np.ndarray         # (72,) metres, ray 0 = sensor frame 0 deg, non-finite -> max range
    lidar_max_range: float    # 3.0
    step: int                 # decision-step index, 0-based
    sim_time: float           # seconds since episode start
    map: MapInfo              # the same object as ScenarioInfo.map
    # Raw sensor_msgs/LaserScan behind `lidar`. Present when running under the ROS
    # harness, None otherwise (e.g. unit tests). Agents that only need distances
    # should use `lidar`; this exists for agents whose observation code already
    # consumes the ROS message (angle_min/angle_increment/ranges).
    lidar_msg: Optional[Any] = None


@dataclass
class Waypoint:
    """Absolute map-frame target the harness will drive to."""

    x: float
    y: float
    theta: Optional[float] = None   # arrival heading; None = face the direction of travel


class GSLAgent(ABC):
    """Implement observe() and act(). Everything else has a working default.

    Lifecycle:  initialize() -> reset(scenario) -> [observe(obs) -> act()]* -> (episode ends)

    The harness guarantees:
      * observe() and act() are called back-to-back, exactly once per decision step.
      * Waypoints are clamped into free space (radius-aware) and capped at max_hop;
        the agent is not told. The next observe() simply shows where the robot
        really ended up.
      * If the motion backend cannot reach the waypoint within drive_timeout
        seconds, it is cancelled and the next decision step starts from wherever
        the robot stopped.
      * The agent never receives the source position.
    """

    # Motion capabilities. Existing agents inherit stop-go semantics unchanged.
    motion_mode: str = 'stop_go'
    decision_rate_hz: Optional[float] = None
    max_goal_distance: float = 1.0
    required_pose_source: Optional[str] = None
    required_map_source: Optional[str] = None
    required_motion: Optional[str] = None
    required_nav_profile: Optional[str] = None

    def initialize(self) -> None:
        """Called once per process, before the first episode. Load weights here."""

    def reset(self, scenario: ScenarioInfo) -> None:
        """Called at the start of every episode."""

    @abstractmethod
    def observe(self, obs: Observation) -> None:
        """Ingest the current sensor snapshot (update belief / filters / history)."""

    @abstractmethod
    def act(self) -> Waypoint:
        """Return the next target position. Called immediately after observe()."""

    def name(self) -> str:
        return type(self).__name__

    def metadata(self) -> dict:
        """Implementation-specific provenance included in result.json."""
        return {}
