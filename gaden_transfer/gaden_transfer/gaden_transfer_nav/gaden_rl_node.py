"""
NavRLNode — deploy a pretrained NavActorCritic PPO agent inside GADEN.

Subscribes to ground-truth pose, LiDAR, and an optional dynamic goal topic,
builds the 77-dim observation vector via NavObsBuilder, runs the policy, and
moves the robot by either direct teleport or Nav2 PoseStamped goal.

Usage
-----
    ros2 run gaden_transfer_nav gaden_rl_node \
        --ros-args \
        -p checkpoint:=/path/to/agent_10000000.pt \
        -p goal_x:=5.0 \
        -p goal_y:=5.0 \
        -p max_steps:=600
"""

import os
import sys
import math
from typing import Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

# RL package imports: add src/base/ to sys.path so both the installed entry
# point and direct execution can find the reinforcement_learning package.
_SRC_BASE = '/home/hdd/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning.models.nav_actor_critic import NavActorCritic
from reinforcement_learning import config as cfg

# efe_igdm/__init__.py transitively imports numba, which may not be installed
# in the system Python used by ros2 run.  We only need the mapping submodule,
# so stub out the top-level package before importing it to skip __init__.py.
import importlib.util as _ilu, types as _types
if 'efe_igdm' not in sys.modules:
    _stub = _types.ModuleType('efe_igdm')
    _s = _ilu.find_spec('efe_igdm')
    if _s is not None:
        _stub.__path__ = list(_s.submodule_search_locations)
    sys.modules['efe_igdm'] = _stub
del _ilu, _types

from efe_igdm.mapping.occupancy_grid import (
    load_3d_occupancy_grid_from_service, OccupancyGridMap
)
from .obs_builder import NavObsBuilder


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class NavRLNode(Node):
    """ROS2 node that runs a pretrained NavActorCritic PPO agent inside GADEN."""

    def __init__(self):
        super().__init__('gaden_nav_rl_node')

        self._declare_parameters()
        self._load_parameters()
        self._init_state()
        self._load_occupancy_map()
        self._load_agent()
        self._init_ros_interfaces()

        self.get_logger().info(
            f'NavRLNode ready. Checkpoint: {self._checkpoint_path}'
        )
        self.get_logger().info(
            f'Device: {self._device} | Max steps: {self._max_steps} | '
            f'Goal: ({self._goal_x:.2f}, {self._goal_y:.2f}) ± {self._goal_tolerance:.2f} m'
        )
        self.get_logger().info(
            f'Map: {self._map_width:.2f}x{self._map_height:.2f} m | '
            f'use_nav2={self._use_nav2}'
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', 5.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('max_steps', 600)
        self.declare_parameter('use_nav2', False)
        self.declare_parameter('step_size', 0.5)
        self.declare_parameter('num_episodes', 1)
        self.declare_parameter('step_delay', 0.5)
        self.declare_parameter('start_x', -999.0)
        self.declare_parameter('start_y', -999.0)
        self.declare_parameter('occupancy_service', '/gaden_environment/occupancyMap3D')
        self.declare_parameter('occupancy_z_level', 5)
        self.declare_parameter('occupancy_timeout', 60.0)
        self.declare_parameter('step_log_every', 1)
        self.declare_parameter('publish_markers', True)

    def _load_parameters(self):
        self._checkpoint_path: str = self.get_parameter('checkpoint').value
        _dev_str: str = self.get_parameter('device').value
        self._device = torch.device(
            _dev_str if torch.cuda.is_available() or _dev_str == 'cpu' else 'cpu'
        )
        self._goal_x: float = float(self.get_parameter('goal_x').value)
        self._goal_y: float = float(self.get_parameter('goal_y').value)
        self._goal_tolerance: float = float(self.get_parameter('goal_tolerance').value)
        self._max_steps: int = int(self.get_parameter('max_steps').value)
        self._use_nav2: bool = bool(self.get_parameter('use_nav2').value)
        self._step_size: float = float(self.get_parameter('step_size').value)
        self._num_episodes: int = int(self.get_parameter('num_episodes').value)
        self._step_delay: float = float(self.get_parameter('step_delay').value)
        self._start_x: float = float(self.get_parameter('start_x').value)
        self._start_y: float = float(self.get_parameter('start_y').value)
        self._occ_service: str = self.get_parameter('occupancy_service').value
        self._occ_z: int = int(self.get_parameter('occupancy_z_level').value)
        self._occ_timeout: float = float(self.get_parameter('occupancy_timeout').value)
        self._step_log_every: int = int(self.get_parameter('step_log_every').value)
        self._publish_markers: bool = bool(self.get_parameter('publish_markers').value)

    def _init_state(self):
        self._robot_x: Optional[float] = None
        self._robot_y: Optional[float] = None

        self._episode: int = 0
        self._step_in_episode: int = 0

        self._search_complete: bool = False
        self._start_teleport_done: bool = False
        self._last_step_time_ns: int = 0

        # Fresh-scan gate: make _take_step wait for a laser scan whose stamp
        # post-dates the most recent teleport, so the obs builder never
        # sees pre-teleport lidar data from the wrong position.
        self._latest_scan_stamp_ns: int = 0
        self._teleport_wait_stamp_ns: Optional[int] = None

        self._latest_lidar_min: Optional[float] = None

        self._obs_builder: Optional[NavObsBuilder] = None

    def _load_occupancy_map(self):
        """Fetch the 2D occupancy grid from GADEN and derive map dimensions."""
        self.get_logger().info(
            f'Waiting up to {self._occ_timeout:.0f}s for {self._occ_service} ...'
        )
        grid_2d, _outlet_mask, params = load_3d_occupancy_grid_from_service(
            self,
            z_level=self._occ_z,
            service_name=self._occ_service,
            timeout_sec=self._occ_timeout,
        )
        self._occ_map = OccupancyGridMap(grid_2d, params)
        if self._occ_map.origin_x == 0.0 and self._occ_map.origin_y == 0.0:
            self._occ_map.origin_x = -0.2
            self._occ_map.origin_y = -0.2

        self._map_width: float = self._occ_map.real_world_width
        self._map_height: float = self._occ_map.real_world_height
        self.get_logger().info(
            f'Occupancy map: {self._occ_map.width}x{self._occ_map.height} cells, '
            f'{self._map_width:.2f}x{self._map_height:.2f} m'
        )

    def _load_agent(self):
        if not self._checkpoint_path:
            raise ValueError(
                "No checkpoint path provided. Set the 'checkpoint' ROS2 parameter:\n"
                "  ros2 run gaden_transfer_nav gaden_rl_node \\\n"
                "      --ros-args -p checkpoint:=/path/to/agent_STEP.pt"
            )
        if not os.path.isfile(self._checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self._checkpoint_path}"
            )

        ckpt = torch.load(self._checkpoint_path, map_location=self._device)
        self._agent = NavActorCritic()
        self._agent.load_state_dict(ckpt['model_state_dict'])
        self._agent.to(self._device)
        self._agent.eval()

        n_params = sum(p.numel() for p in self._agent.parameters())
        self.get_logger().info(f'Loaded NavActorCritic: {n_params:,} parameters')

        self._obs_builder = NavObsBuilder(
            map_width=self._map_width,
            map_height=self._map_height,
            goal_x=self._goal_x,
            goal_y=self._goal_y,
        )

    def _init_ros_interfaces(self):
        """Set up subscribers, publishers."""
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/PioneerP3DX/ground_truth',
            self._pose_callback, 10)

        self._lidar_sub = self.create_subscription(
            LaserScan,
            '/PioneerP3DX/laser_scanner',
            self._lidar_callback, 10)

        self._nav_goal_sub = self.create_subscription(
            PoseStamped,
            '/nav_goal',
            self._nav_goal_callback, 10)

        self._teleport_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/PioneerP3DX/initialpose', 10)

        self._goal_marker_pub = self.create_publisher(
            Marker, '/gaden_nav_rl/goal_marker', 1)

        self._action_marker_pub = self.create_publisher(
            Marker, '/gaden_nav_rl/action_marker', 1)

        if self._use_nav2:
            self._nav2_goal_pub = self.create_publisher(
                PoseStamped, '/goal_pose', 10)

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self._robot_x = x
        self._robot_y = y

        if self._obs_builder is None:
            return

        # Update obs builder pose
        self._obs_builder.update_pose(x, y)

        if not self._obs_builder.ready:
            return

        # Initial: teleport to start position once
        if not self._start_teleport_done:
            self._start_teleport_done = True
            if self._start_x > -998.0 and self._start_y > -998.0:
                self.get_logger().info(
                    f'Teleporting to start position '
                    f'({self._start_x:.2f}, {self._start_y:.2f})'
                )
                self._teleport_to(self._start_x, self._start_y)
                self._last_step_time_ns = self.get_clock().now().nanoseconds
                return

        # Fresh-scan gate: wait for a scan whose stamp is strictly newer than
        # the last teleport.
        if self._teleport_wait_stamp_ns is not None and \
                self._latest_scan_stamp_ns <= self._teleport_wait_stamp_ns:
            return

        # Step-rate gate
        now_ns = self.get_clock().now().nanoseconds
        delay_ns = int(self._step_delay * 1e9)
        if now_ns - self._last_step_time_ns < delay_ns:
            return
        self._last_step_time_ns = now_ns
        self._take_step()

    def _lidar_callback(self, msg: LaserScan):
        # Record scan stamp for the fresh-scan gate
        st = msg.header.stamp
        self._latest_scan_stamp_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)

        if self._obs_builder is None:
            return

        # Track min range for logging
        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        self._latest_lidar_min = min(finite_ranges) if finite_ranges else None

        self._obs_builder.update_lidar(msg)

    def _nav_goal_callback(self, msg: PoseStamped):
        # Goal is swapped in place — the episode continues from the current step count.
        # Step counter and start-teleport are NOT reset; only the obs_builder goal changes.
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self._obs_builder.reset(goal_x=self._goal_x, goal_y=self._goal_y)
        self.get_logger().info(
            f'New nav goal: ({self._goal_x:.2f}, {self._goal_y:.2f})'
        )

    # ------------------------------------------------------------------
    # Core control loop
    # ------------------------------------------------------------------

    def _take_step(self):
        """Run one policy step: build obs → infer action → move robot."""
        if self._search_complete:
            return
        if self._robot_x is None or self._robot_y is None:
            return
        if not self._obs_builder.ready:
            return

        # --- Check goal reached ---
        dist_to_goal = math.hypot(
            self._robot_x - self._goal_x,
            self._robot_y - self._goal_y,
        )
        if dist_to_goal < self._goal_tolerance:
            self.get_logger().info(
                f'[Episode {self._episode}] Goal reached at step '
                f'{self._step_in_episode}!'
            )
            self._end_episode(success=True)
            return

        # --- Episode timeout check ---
        if self._step_in_episode >= self._max_steps:
            self.get_logger().warn(
                f'[Episode {self._episode}] Max steps ({self._max_steps}) exceeded.'
            )
            self._end_episode(success=False)
            return

        # --- Build observation ---
        obs = self._obs_builder.build()
        if obs is None:
            return

        # --- Policy inference ---
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self._agent.get_action_and_value(obs_t)
        action_np = action.cpu().numpy().flatten()  # (2,)

        # --- Decode heading: action is (cos theta, sin theta) ---
        theta = math.atan2(float(action_np[1]), float(action_np[0]))
        target_x = self._robot_x + self._step_size * math.cos(theta)
        target_y = self._robot_y + self._step_size * math.sin(theta)

        # --- Collision clamp ---
        target_x, target_y, collided = self._clamp_to_free(
            self._robot_x, self._robot_y, target_x, target_y, theta
        )
        if collided:
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'COLLISION — clamped to ({target_x:.2f},{target_y:.2f})'
            )

        # --- Step logging ---
        if self._step_log_every <= 1 or (self._step_in_episode % self._step_log_every == 0):
            lidar_text = (
                f'{self._latest_lidar_min:.2f}' if self._latest_lidar_min is not None
                else 'n/a'
            )
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'Pos ({self._robot_x:.2f},{self._robot_y:.2f}) '
                f'θ={math.degrees(theta):.1f}deg → '
                f'Target ({target_x:.2f},{target_y:.2f}) | '
                f'd2goal={dist_to_goal:.2f}m lidar_min={lidar_text}m'
            )

        # --- Publish markers ---
        if self._publish_markers:
            self._publish_goal_marker()
            self._publish_action_marker(target_x, target_y)

        # --- Execute motion ---
        if self._use_nav2:
            nav2_msg = PoseStamped()
            nav2_msg.header.stamp = self.get_clock().now().to_msg()
            nav2_msg.header.frame_id = 'map'
            nav2_msg.pose.position.x = float(target_x)
            nav2_msg.pose.position.y = float(target_y)
            nav2_msg.pose.orientation.w = 1.0
            self._nav2_goal_pub.publish(nav2_msg)
        else:
            self._teleport_to(target_x, target_y)

        self._step_in_episode += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_to_free(self, rx: float, ry: float,
                       tx: float, ty: float, theta: float) -> tuple:
        """Validate target; if blocked, walk back along the ray until free.

        Returns
        -------
        (clamped_x, clamped_y, collided) where ``collided`` is True iff the
        original target was not free.
        """
        if not hasattr(self, '_occ_map'):
            return tx, ty, False

        if self._occ_map.is_valid((tx, ty), radius=cfg.ROBOT_RADIUS):
            return tx, ty, False

        # Original target blocked — walk back along the ray
        step = self._step_size - 0.05
        while step >= 0.1:
            cx = rx + step * math.cos(theta)
            cy = ry + step * math.sin(theta)
            if self._occ_map.is_valid((cx, cy), radius=cfg.ROBOT_RADIUS):
                return cx, cy, True
            step -= 0.05

        # Nowhere to go — stay in place
        return rx, ry, True

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _end_episode(self, success: bool):
        """Log episode result and either shutdown or reset for the next one."""
        result = 'SUCCESS' if success else 'FAILURE'
        self.get_logger().info(
            f'[Episode {self._episode}] {result} — '
            f'{self._step_in_episode} steps'
        )
        self._episode += 1

        if self._episode >= self._num_episodes:
            self.get_logger().info(
                f'Completed {self._num_episodes} episode(s). Shutting down.'
            )
            self._search_complete = True
            os._exit(0)

        # Prepare next episode
        self._step_in_episode = 0
        self._obs_builder.reset(goal_x=self._goal_x, goal_y=self._goal_y)

        if self._start_x > -998.0 and self._start_y > -998.0:
            self.get_logger().info(
                f'Starting episode {self._episode} — teleporting to '
                f'({self._start_x:.2f}, {self._start_y:.2f})'
            )
            self._start_teleport_done = False

        self._last_step_time_ns = self.get_clock().now().nanoseconds

    # ------------------------------------------------------------------
    # Teleport
    # ------------------------------------------------------------------

    def _teleport_to(self, x: float, y: float):
        """Publish a PoseWithCovarianceStamped to teleport the robot."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance = [0.0] * 36
        self._teleport_pub.publish(msg)
        # Re-arm fresh-scan gate
        self._teleport_wait_stamp_ns = self._latest_scan_stamp_ns

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------

    def _publish_goal_marker(self):
        """Publish a green SPHERE marker at the goal position."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gaden_nav_rl_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self._goal_x)
        marker.pose.position.y = float(self._goal_y)
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.1
        marker.color.a = 0.8
        self._goal_marker_pub.publish(marker)

    def _publish_action_marker(self, target_x: float, target_y: float):
        """Publish a yellow ARROW marker from robot position to target."""
        if self._robot_x is None or self._robot_y is None:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gaden_nav_rl_action'
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.a = 0.95
        marker.color.r = 0.95
        marker.color.g = 0.75
        marker.color.b = 0.05
        marker.points = [
            Point(x=float(self._robot_x), y=float(self._robot_y), z=0.12),
            Point(x=float(target_x), y=float(target_y), z=0.12),
        ]
        self._action_marker_pub.publish(marker)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NavRLNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
