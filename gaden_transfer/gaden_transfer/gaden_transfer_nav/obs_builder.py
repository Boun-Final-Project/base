"""
Builds the 77-dim observation vector from live ROS2 sensor data.

Mirrors reinforcement_learning.envs.nav_env::_build_obs() exactly so the same
pretrained weights apply without any re-normalisation.

Observation layout (77 dims total):
    [0:72]   lidar        — 72 normalised ray distances (normalised by update_lidar)
    [72:74]  position     — (robot_x / map_width, robot_y / map_height)
    [74:76]  goal_dir     — (dx / dist_to_goal, dy / dist_to_goal)
    [76]     goal_dist    — dist_to_goal / map_diagonal
"""

import numpy as np
from typing import Optional

from sensor_msgs.msg import LaserScan
from reinforcement_learning import config as cfg
from .lidar_resampler import resample_scan


class NavObsBuilder:
    """Stateful observation assembler for navigation task.

    One instance per episode. Call ``reset()`` at the start of each episode,
    then ``build()`` after every sensor update to get the latest 77-dim vector.
    """

    def __init__(self, map_width: float, map_height: float, goal_x: float, goal_y: float):
        """
        Parameters
        ----------
        map_width, map_height : float
            Real-world dimensions of the map in metres. Used to normalise
            robot position and goal distance. Obtained from the occupancy
            grid service at startup.
        goal_x, goal_y : float
            Goal position in world frame (metres).
        """
        self.map_width = map_width
        self.map_height = map_height
        self.goal_x = goal_x
        self.goal_y = goal_y

        # Precompute map diagonal for goal distance normalisation
        self._map_diagonal = float(np.sqrt(map_width ** 2 + map_height ** 2))

        # Latest sensor readings (set by the node's callbacks)
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self._latest_lidar: Optional[np.ndarray] = None  # shape (72,), already [0,1]

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self, goal_x: Optional[float] = None, goal_y: Optional[float] = None):
        """Call at the start of each episode.

        Clears robot pose and lidar so that ``ready`` returns False until fresh
        callbacks arrive. Optionally updates the goal position if both coordinates
        are provided.

        Parameters
        ----------
        goal_x, goal_y : float, optional
            New goal position. Both must be provided to update; if only one is
            provided, raises ValueError. If neither is provided, use the existing goal.
        """
        # Clear per-episode sensor state
        self.robot_x = None
        self.robot_y = None
        self._latest_lidar = None

        if goal_x is not None and goal_y is not None:
            self.goal_x = goal_x
            self.goal_y = goal_y
        elif goal_x is not None or goal_y is not None:
            raise ValueError("reset() requires both goal_x and goal_y or neither")

    # ------------------------------------------------------------------
    # Called by the node's sensor callbacks
    # ------------------------------------------------------------------

    def update_lidar(self, msg: LaserScan):
        """Process a raw LaserScan into the normalised 72-ray array.

        Parameters
        ----------
        msg : sensor_msgs.msg.LaserScan
            Raw laser scan message from the robot.
        """
        self._latest_lidar = resample_scan(msg, cfg.LIDAR_NUM_RAYS, cfg.LIDAR_MAX_RANGE)

    def update_pose(self, x: float, y: float) -> None:
        """Update robot pose from odometry or localization callback.

        Parameters
        ----------
        x, y : float
            Robot position in world frame (metres).
        """
        self.robot_x = x
        self.robot_y = y

    # ------------------------------------------------------------------
    # Build observation
    # ------------------------------------------------------------------

    def build(self) -> Optional[np.ndarray]:
        """Assemble and return the 77-dim observation vector.

        Returns ``None`` if robot pose or lidar have not yet received their
        first message.

        Returns
        -------
        obs : np.ndarray or None
            Shape (77,), dtype float32. Layout:
            [0:72]   lidar (normalised [0, 1])
            [72:74]  robot position (normalised by map dims)
            [74:76]  goal direction (dx/dist_to_goal, dy/dist_to_goal)
            [76]     goal distance (normalised by map diagonal)
        """
        # Check readiness
        if self.robot_x is None or self.robot_y is None:
            return None
        if self._latest_lidar is None:
            return None

        # --- LiDAR (72 dims) — already normalised by update_lidar ---
        lidar = self._latest_lidar.astype(np.float32)

        # --- Position (2 dims) ---
        pos_norm = np.array([
            self.robot_x / self.map_width,
            self.robot_y / self.map_height,
        ], dtype=np.float32)

        # --- Goal direction and distance ---
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = float(np.sqrt(dx ** 2 + dy ** 2))

        # When dist is very small, use (0, 0) for goal direction and 0 for goal distance
        # Otherwise, normalize direction by distance
        goal_dir = np.array(
            [dx / (dist + 1e-8), dy / (dist + 1e-8)],
            dtype=np.float32
        )
        goal_dist = np.array([dist / self._map_diagonal], dtype=np.float32)

        # --- Concatenate and return ---
        obs = np.concatenate([lidar, pos_norm, goal_dir, goal_dist])  # (77,)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once robot pose and lidar have both provided at least one reading."""
        return (self.robot_x is not None and
                self.robot_y is not None and
                self._latest_lidar is not None)
