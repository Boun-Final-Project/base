"""
NavRLNode — deploy a pretrained NavActorCritic PPO agent inside GADEN.

Subscribes to ground-truth pose, LiDAR, and an optional dynamic goal topic,
builds the 77-dim observation vector via NavObsBuilder, runs the policy, and
moves the robot with a Twist cmd_vel P-controller (IDLE/MOVING state machine).

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

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist, Point
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
            f'Map: {self._map_width:.2f}x{self._map_height:.2f} m'
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
        self.declare_parameter('step_size', 0.5)
        self.declare_parameter('num_episodes', 1)
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed_gain', 2.0)
        self.declare_parameter('arrival_tolerance', 0.15)
        self.declare_parameter('cmd_vel_rate', 10.0)
        self.declare_parameter('waypoint_timeout', 10.0)
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
        self._step_size: float = float(self.get_parameter('step_size').value)
        self._num_episodes: int = int(self.get_parameter('num_episodes').value)
        self._linear_speed: float = float(self.get_parameter('linear_speed').value)
        self._angular_speed_gain: float = float(self.get_parameter('angular_speed_gain').value)
        self._arrival_tolerance: float = float(self.get_parameter('arrival_tolerance').value)
        self._cmd_vel_rate: float = float(self.get_parameter('cmd_vel_rate').value)
        self._waypoint_timeout: float = float(self.get_parameter('waypoint_timeout').value)
        self._occ_service: str = self.get_parameter('occupancy_service').value
        self._occ_z: int = int(self.get_parameter('occupancy_z_level').value)
        self._occ_timeout: float = float(self.get_parameter('occupancy_timeout').value)
        self._step_log_every: int = int(self.get_parameter('step_log_every').value)
        self._step_log_every = max(1, self._step_log_every)
        self._publish_markers: bool = bool(self.get_parameter('publish_markers').value)

    def _init_state(self):
        self._robot_x: Optional[float] = None
        self._robot_y: Optional[float] = None

        self._episode: int = 0
        self._step_in_episode: int = 0

        self._search_complete: bool = False

        self._moving: bool = False           # True while tracking a waypoint
        self._waypoint_x: Optional[float] = None
        self._waypoint_y: Optional[float] = None
        self._waypoint_start_ns: int = 0
        self._robot_theta: Optional[float] = None   # yaw extracted from ground_truth

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

        self._cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)

        self._goal_marker_pub = self.create_publisher(
            Marker, '/gaden_nav_rl/goal_marker', 1)

        self._action_marker_pub = self.create_publisher(
            Marker, '/gaden_nav_rl/action_marker', 1)

        self._cmd_vel_timer = self.create_timer(
            1.0 / self._cmd_vel_rate, self._cmd_vel_tick
        )

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_x = x
        self._robot_y = y
        self._robot_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        if self._obs_builder is None:
            return
        self._obs_builder.update_pose(x, y)
        # Trigger inference only when IDLE and obs ready
        if not self._moving and not self._search_complete and self._obs_builder.ready:
            self._take_step()

    def _lidar_callback(self, msg: LaserScan):
        if self._obs_builder is None:
            return

        # Track min range for logging
        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        self._latest_lidar_min = min(finite_ranges) if finite_ranges else None

        self._obs_builder.update_lidar(msg)

    def _nav_goal_callback(self, msg: PoseStamped):
        if self._obs_builder is None:
            return
        # Goal is swapped in place — the episode continues from the current step count.
        # Only the obs_builder goal changes; step counter is not reset.
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

        # Set waypoint and enter MOVING state
        self._waypoint_x = target_x
        self._waypoint_y = target_y
        self._waypoint_start_ns = self.get_clock().now().nanoseconds
        self._moving = True

    # ------------------------------------------------------------------
    # cmd_vel control loop
    # ------------------------------------------------------------------

    def _cmd_vel_tick(self):
        # Single-executor assumed: _moving and _waypoint_x/y are written only in
        # _take_step (pose callback) and read/cleared here — no lock needed.
        if not self._moving:
            return
        if self._robot_x is None or self._robot_theta is None:
            return

        dx = self._waypoint_x - self._robot_x
        dy = self._waypoint_y - self._robot_y
        dist = math.hypot(dx, dy)

        elapsed_ns = self.get_clock().now().nanoseconds - self._waypoint_start_ns
        timed_out = elapsed_ns > self._waypoint_timeout * 1e9

        if dist < self._arrival_tolerance or timed_out:
            # Stop robot
            self._cmd_vel_pub.publish(Twist())
            self._step_in_episode += 1
            self._moving = False
            if timed_out and dist >= self._arrival_tolerance:
                self.get_logger().warn(
                    f'[Ep {self._episode} Step {self._step_in_episode}] '
                    f'Waypoint timeout (dist={dist:.2f}m)'
                )
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - self._robot_theta
        # Wrap to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        twist = Twist()
        twist.linear.x = self._linear_speed
        twist.angular.z = self._angular_speed_gain * heading_error
        self._cmd_vel_pub.publish(twist)

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
            rclpy.shutdown()
            return

        # Prepare next episode
        self._step_in_episode = 0
        self._obs_builder.reset(goal_x=self._goal_x, goal_y=self._goal_y)

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
