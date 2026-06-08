"""
GadenRLNodeImage — deploy the 5-channel spatial PPO agent inside GADEN.

Loads an ActorCriticSpatial checkpoint trained against the 5-channel
observation ([is_known, is_wall, gas, recency, det_count]) and 4-dim context
([speed/max, cos(dir), sin(dir), step/MAX_STEPS]). Uses GADEN's
``/gaden_environment/occupancyMap3D`` service as the ground-truth grid for
the reveal, mirroring training's SpatialObsWrapper exactly.

Usage
-----
    ros2 run reinforcement_learning gaden_rl_node_image --ros-args \
        -p checkpoint:=/path/to/agent_41779200.pt \
        -p wind_file:=/path/to/wind_at_cell_centers_0.csv \
        -p true_source_x:=... -p true_source_y:=... \
        -p start_x:=... -p start_y:=... \
        -p max_steps:=600 -p num_episodes:=1
"""

import os
import sys
import math
from typing import Optional, Tuple

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from olfaction_msgs.msg import GasSensor, Anemometer
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty

_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning import config as cfg
from .actor_critic_spatial import ActorCriticSpatial
from .spatial_obs_builder import SpatialObsBuilder

from geometry_msgs.msg import PoseWithCovarianceStamped as PoseWCS
from efe_igdm.mapping.occupancy_grid import (
    load_3d_occupancy_grid_from_service, OccupancyGridMap,
    create_empty_occupancy_map,
)
from efe_igdm.planning.navigator import Navigator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_agent(checkpoint_path: str, device: torch.device) -> ActorCriticSpatial:
    agent = ActorCriticSpatial()
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt['model_state_dict'])
    agent.to(device)
    agent.eval()
    return agent


def _action_to_target(robot_x: float, robot_y: float,
                      action: np.ndarray, step_size: float) -> Tuple[float, float, float]:
    """Action is Normal-sampled (cos θ, sin θ); decode via atan2."""
    cos_t, sin_t = float(action[0]), float(action[1])
    theta = math.atan2(sin_t, cos_t)
    target_x = robot_x + step_size * math.cos(theta)
    target_y = robot_y + step_size * math.sin(theta)
    return target_x, target_y, theta


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GadenRLNodeImage(Node):

    def __init__(self):
        super().__init__('gaden_rl_node_image')

        self._declare_parameters()
        self._load_parameters()
        self._init_state()
        self._load_occupancy_map()
        self._load_agent()
        self._init_ros_interfaces()

        self.get_logger().info(f'GadenRLNodeImage ready. Checkpoint: {self._checkpoint_path}')
        self.get_logger().info(
            f'Device: {self._device} | Max steps: {self._max_steps} | '
            f'Num episodes: {self._num_episodes}'
        )
        self.get_logger().info(f'Map normalisation: {self._map_width:.1f} x {self._map_height:.1f} m')
        self.get_logger().info(f'True source: ({self._true_source_x:.2f}, {self._true_source_y:.2f})')

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 4.5)
        self.declare_parameter('max_steps', cfg.MAX_STEPS)
        self.declare_parameter('num_episodes', 1)
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('occupancy_service', '/gaden_environment/occupancyMap3D')
        self.declare_parameter('occupancy_z_level', 5)
        self.declare_parameter('occupancy_timeout', 60.0)
        self.declare_parameter('wind_file', '')
        self.declare_parameter('step_delay', 0.5)
        self.declare_parameter('start_x', -999.0)
        self.declare_parameter('start_y', -999.0)
        self.declare_parameter('step_log_every', 1)
        self.declare_parameter('publish_action_marker', True)
        self.declare_parameter('action_marker_topic', '/model_free_gsl/target')
        # --- SLAM integration ---
        # If true, the observation builder uses a live SLAM-built grid
        # (subscribed from slam_map_topic) instead of the GADEN ground truth.
        # Collision checks still run against the GT occupancy map for safety.
        self.declare_parameter('use_slam_map', False)
        self.declare_parameter('slam_map_topic', '/slam_node/slam_map')
        # --- Debug image dumps ---
        # If non-empty, save a PNG of the 5 spatial channels + world grid
        # every `debug_dump_every` steps to this directory.
        self.declare_parameter('debug_dump_dir', '')
        self.declare_parameter('debug_dump_every', 20)
        # --- Nav2 motion ---
        # If true, replace per-step teleports with Nav2 goal-based navigation
        # so the robot physically drives to each waypoint. Makes travel
        # distance / travel time directly comparable to ADSM-style baselines.
        self.declare_parameter('use_nav2', False)
        # 0.1 m — tight enough that each 0.5 m policy step actually covers
        # ~0.4 m of physical travel. 0.3 m (Nav2's typical default) cancels
        # too early → only ~0.2 m per step.
        self.declare_parameter('nav_goal_tolerance', 0.1)
        # --- Nav2 goal pipelining ---
        # If true (and use_nav2 true): trigger policy steps on a fixed timer
        # and send each new goal without waiting for the previous one to
        # finish. Nav2's BT preempts the active goal → robot never stops.
        # Matches ADSM's 1 Hz "send new goal each second" pattern.
        # Trade-off: observations are from mid-motion poses (OOD vs training).
        self.declare_parameter('pipelined_goals', False)
        self.declare_parameter('pipeline_period_s', 1.0)
        # --- Wind source ---
        # 'csv_mean' (default): lock wind once from the mean of a GADEN CFD
        #     CSV. Matches training's constant-per-episode wind semantics.
        # 'live': update wind every anemometer callback from the live sensor.
        #     Treats the observation as "what the robot feels now", which is
        #     what training actually gave the policy (training's wind was
        #     uniform across space, so local = global there).
        self.declare_parameter('wind_source', 'csv_mean')

    def _load_parameters(self):
        self._checkpoint_path: str = self.get_parameter('checkpoint').value
        _dev_str: str = self.get_parameter('device').value
        self._device = torch.device(
            _dev_str if torch.cuda.is_available() or _dev_str == 'cpu' else 'cpu'
        )
        self._true_source_x: float = self.get_parameter('true_source_x').value
        self._true_source_y: float = self.get_parameter('true_source_y').value
        self._max_steps: int       = int(self.get_parameter('max_steps').value)
        self._num_episodes: int    = int(self.get_parameter('num_episodes').value)
        self._xy_tolerance: float  = self.get_parameter('xy_goal_tolerance').value
        self._occ_service: str     = self.get_parameter('occupancy_service').value
        self._occ_z: int           = self.get_parameter('occupancy_z_level').value
        self._occ_timeout: float   = float(self.get_parameter('occupancy_timeout').value)
        self._wind_file: str       = self.get_parameter('wind_file').value
        self._step_delay: float    = float(self.get_parameter('step_delay').value)
        self._start_x: float       = self.get_parameter('start_x').value
        self._start_y: float       = self.get_parameter('start_y').value
        self._step_log_every: int  = int(self.get_parameter('step_log_every').value)
        self._publish_action_marker: bool = bool(self.get_parameter('publish_action_marker').value)
        self._action_marker_topic: str = self.get_parameter('action_marker_topic').value
        self._use_slam_map: bool   = bool(self.get_parameter('use_slam_map').value)
        self._slam_map_topic: str  = self.get_parameter('slam_map_topic').value
        self._debug_dump_dir: str  = self.get_parameter('debug_dump_dir').value
        self._debug_dump_every: int = int(self.get_parameter('debug_dump_every').value)
        self._use_nav2: bool        = bool(self.get_parameter('use_nav2').value)
        self._nav_goal_tolerance: float = float(self.get_parameter('nav_goal_tolerance').value)
        self._pipelined_goals: bool = bool(self.get_parameter('pipelined_goals').value)
        self._pipeline_period_s: float = float(self.get_parameter('pipeline_period_s').value)
        self._wind_source: str = str(self.get_parameter('wind_source').value)
        if self._wind_source not in ('csv_mean', 'live'):
            self.get_logger().warn(
                f'wind_source={self._wind_source!r} is not one of (csv_mean, live); '
                f'falling back to csv_mean.'
            )
            self._wind_source = 'csv_mean'

    def _init_state(self):
        self._robot_x: Optional[float] = None
        self._robot_y: Optional[float] = None
        self._current_theta: Optional[float] = None

        self._episode: int = 0
        self._step_in_episode: int = 0

        self._search_complete: bool = False
        self._start_teleport_done: bool = False
        self._last_step_time_ns: int = 0
        self._latest_gas_raw: Optional[float] = None
        self._latest_wind_speed: Optional[float] = None
        self._latest_wind_dir: Optional[float] = None
        self._latest_lidar_min: Optional[float] = None
        # Sim-time stamp of the most recent laser scan we've seen.
        # Used by the "wait for fresh scan after teleport" gate so we don't
        # build observations with a scan that predates the current pose
        # (which was the cause of the splattered-walls SLAM artefact).
        self._latest_scan_stamp_ns: int = 0
        # Stamp we had at the moment of the last teleport; _take_step only
        # proceeds once _latest_scan_stamp_ns has moved past it.
        self._teleport_wait_stamp_ns: Optional[int] = None

        # Nav2 state
        self._is_moving: bool = False
        self._travel_distance_m: float = 0.0
        self._last_pose_xy: Optional[Tuple[float, float]] = None
        self._navigator: Optional[Navigator] = None

        self._obs_builder: Optional[SpatialObsBuilder] = None

    def _load_occupancy_map(self):
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

        self._map_width: float  = self._occ_map.real_world_width
        self._map_height: float = self._occ_map.real_world_height
        self.get_logger().info(
            f'Occupancy map: {self._occ_map.width}x{self._occ_map.height} cells, '
            f'{self._map_width:.2f}x{self._map_height:.2f} m'
        )

    def _load_agent(self):
        if not self._checkpoint_path:
            raise ValueError(
                "No checkpoint path provided. Set the 'checkpoint' ROS2 parameter."
            )
        if not os.path.isfile(self._checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self._checkpoint_path}"
            )

        self._agent = _load_agent(self._checkpoint_path, self._device)
        n_params = sum(p.numel() for p in self._agent.parameters())
        self.get_logger().info(
            f'Loaded checkpoint: {self._checkpoint_path} ({n_params:,} parameters)'
        )

        # The builder uses either GT (fast, bit-exact with training) or a
        # SLAM-built grid (stricter sim-to-real-ish test). _clamp_to_free
        # stays on the GT map regardless, so collisions don't depend on
        # SLAM's exploration progress.
        if self._use_slam_map:
            self._slam_grid_map = create_empty_occupancy_map(self._occ_map)
            self.get_logger().info(
                f'Observation grid: SLAM-built (subscribing to {self._slam_map_topic})'
            )
            self._obs_builder = SpatialObsBuilder(
                self._slam_grid_map, self._map_width, self._map_height
            )
        else:
            self._slam_grid_map = None
            self.get_logger().info('Observation grid: GADEN ground truth')
            self._obs_builder = SpatialObsBuilder(
                self._occ_map, self._map_width, self._map_height
            )

        # Debug image dump directory
        if self._debug_dump_dir:
            os.makedirs(self._debug_dump_dir, exist_ok=True)
            self.get_logger().info(
                f'Debug dumps every {self._debug_dump_every} steps → {self._debug_dump_dir}'
            )

        if self._wind_source == 'csv_mean':
            if self._wind_file:
                self._obs_builder.load_wind_from_file(self._wind_file)
                self.get_logger().info(
                    f'Wind source: csv_mean → '
                    f'speed={self._obs_builder._locked_wind_speed:.3f} m/s '
                    f'dir={np.degrees(self._obs_builder._locked_wind_dir):.1f} deg'
                )
            else:
                self.get_logger().warn(
                    'wind_source=csv_mean but no wind_file provided — '
                    'wind context will stay zero until locked'
                )
        else:  # live
            self.get_logger().info(
                'Wind source: live → anemometer reading updates wind every callback'
            )

        if self._use_nav2:
            mode = 'Nav2 pipelined' if self._pipelined_goals else 'Nav2 stop-go'
            self.get_logger().info(
                f'Motion mode: {mode} (goal tolerance {self._nav_goal_tolerance} m). '
                f'Waiting for /PioneerP3DX/navigate_to_pose action server...'
            )
            self._navigator = Navigator(self, on_complete_callback=self._on_nav_complete)
            self.get_logger().info('Navigator ready.')
            if self._pipelined_goals:
                self.get_logger().info(
                    f'Pipeline timer: period={self._pipeline_period_s} s. '
                    f'Steps triggered by timer; Nav2 goals preempt the active one.'
                )
                self._pipeline_timer = self.create_timer(
                    self._pipeline_period_s, self._pipeline_tick
                )
        else:
            self.get_logger().info('Motion mode: teleport (use_nav2=false).')

    def _init_ros_interfaces(self):
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/PioneerP3DX/ground_truth',
            self._pose_callback, 10)
        self._gas_sub = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self._gas_callback, 10)
        self._lidar_sub = self.create_subscription(
            LaserScan,
            '/PioneerP3DX/laser_scanner',
            self._lidar_callback, 10)
        self._wind_sub = self.create_subscription(
            Anemometer,
            '/fake_anemometer/WindSensor_reading',
            self._wind_callback, 10)

        self._teleport_pub = self.create_publisher(
            PoseWCS, '/PioneerP3DX/initialpose', 10)
        self._next_action_pub = self.create_publisher(Marker, self._action_marker_topic, 1)

        if self._use_slam_map:
            # SlamNode publishes with TRANSIENT_LOCAL so subscribers pick up
            # the latest map even if they join late.
            slam_qos = QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
            )
            self._slam_sub = self.create_subscription(
                OccupancyGrid, self._slam_map_topic,
                self._slam_map_callback, slam_qos)
            self._slam_map_count = 0
            # Reset publisher: tells SlamNode to wipe its grid + pose buffer
            # immediately after our start-pose teleport so the map never
            # contains pre-teleport data.
            self._slam_reset_pub = self.create_publisher(
                Empty, '/slam_node/reset_map', 1
            )
            self._slam_reset_sent: bool = False

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_theta = math.atan2(siny_cosp, cosy_cosp)

        # Accumulate travel distance — integrate |Δpose| between frames so
        # the summary reports the robot's true path length, not the step
        # count × STEP_SIZE (which over-counts when Nav2 takes indirect paths).
        if self._last_pose_xy is not None:
            dx = x - self._last_pose_xy[0]
            dy = y - self._last_pose_xy[1]
            d = math.hypot(dx, dy)
            # Guard against the initial teleport jump (tens of metres in one
            # frame) polluting the distance accumulator.
            if d < 1.0:
                self._travel_distance_m += d
        self._last_pose_xy = (x, y)

        self._robot_x = x
        self._robot_y = y
        if self._obs_builder is not None:
            self._obs_builder.robot_x = x
            self._obs_builder.robot_y = y

        if self._obs_builder is None or not self._obs_builder.ready:
            return

        # Nav2 stop-go gate: skip stepping while Nav2 is driving the robot
        # (on_complete re-fires stepping). The pipelined-mode return is
        # moved below so initial-teleport + fresh-scan + SLAM-reset still
        # run through pose_callback as normal setup.
        if self._use_nav2 and not self._pipelined_goals and self._is_moving:
            return

        if not self._start_teleport_done:
            self._start_teleport_done = True
            if self._start_x > -998.0 and self._start_y > -998.0:
                self.get_logger().info(
                    f'Teleporting to start position ({self._start_x:.2f}, {self._start_y:.2f})'
                )
                self._teleport_to(self._start_x, self._start_y)
                self._last_step_time_ns = self.get_clock().now().nanoseconds
                return

        # Fresh-scan gate: after any teleport, wait for a laser scan whose
        # stamp post-dates the teleport. Otherwise we'd step with a scan from
        # the PRE-teleport position, which is both what the policy sees AND
        # what SlamNode builds into the map → the splatter artefact.
        if self._teleport_wait_stamp_ns is not None and \
                self._latest_scan_stamp_ns <= self._teleport_wait_stamp_ns:
            return

        # SLAM reset: just AFTER the start-pose teleport has settled (fresh
        # scan has arrived), wipe SlamNode's grid so it doesn't contain any
        # scans from the default GADEN spawn. Re-arm the fresh-scan gate so
        # the first step waits for a scan that post-dates the reset.
        if self._use_slam_map and not self._slam_reset_sent:
            self.get_logger().info('Resetting SLAM map after start-pose teleport.')
            self._slam_reset_pub.publish(Empty())
            self._slam_reset_sent = True
            self._teleport_wait_stamp_ns = self._latest_scan_stamp_ns
            return

        # Pipelined mode: setup (teleport + fresh scan + SLAM reset) is done;
        # from here on, the pipeline timer drives stepping. pose_callback
        # only keeps pose + travel-distance state up to date.
        if self._use_nav2 and self._pipelined_goals:
            return

        now_ns = self.get_clock().now().nanoseconds
        delay_ns = int(self._step_delay * 1e9)
        if now_ns - self._last_step_time_ns < delay_ns:
            return
        self._last_step_time_ns = now_ns
        self._take_step()

    def _gas_callback(self, msg: GasSensor):
        if self._obs_builder is None:
            return
        self._latest_gas_raw = float(msg.raw)
        self._obs_builder.update_gas(msg.raw)

    def _lidar_callback(self, msg: LaserScan):
        # Logging: track closest obstacle for step logs.
        finite = [r for r in msg.ranges if math.isfinite(r)]
        self._latest_lidar_min = min(finite) if finite else None
        # Fresh-scan gate: remember the sim-time stamp of the latest scan so
        # _pose_callback can tell whether this scan post-dates the teleport.
        st = msg.header.stamp
        self._latest_scan_stamp_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)

    def _wind_callback(self, msg: Anemometer):
        self._latest_wind_speed = float(msg.wind_speed)
        self._latest_wind_dir   = float(msg.wind_direction)
        # Live-wind mode: feed each reading into the observation builder so
        # build() uses the most recent anemometer sample in the ctx vector.
        if self._wind_source == 'live' and self._obs_builder is not None:
            self._obs_builder.update_wind_live(
                msg.wind_speed, msg.wind_direction
            )

    def _teleport_to(self, x: float, y: float):
        msg = PoseWCS()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance = [0.0] * 36
        self._teleport_pub.publish(msg)
        # Remember the latest scan stamp at the moment of teleport.
        # _take_step will block until a scan with a strictly later stamp
        # arrives, guaranteeing the policy and SLAM see the post-teleport scan.
        self._teleport_wait_stamp_ns = self._latest_scan_stamp_ns

    def _on_nav_complete(self):
        """Called by Navigator when a goal finishes (success or abort).

        In stop-go mode: clear the moving flag; the next pose message fires
        _take_step through _pose_callback.
        In pipelined mode: no-op — the pipeline timer is the only step driver.
        """
        self._is_moving = False

    def _pipeline_tick(self):
        """Timer-driven step for pipelined Nav2 mode.

        Fires every _pipeline_period_s seconds. Each tick computes a new
        goal from the robot's current pose and sends it via Navigator —
        Nav2's BT preempts the active goal → robot never stops between
        policy decisions.
        """
        if self._search_complete:
            return
        if not self._start_teleport_done:
            return
        if self._robot_x is None or self._robot_y is None:
            return
        if self._obs_builder is None or not self._obs_builder.ready:
            return
        # Wait for a fresh scan after the start-pose teleport before the
        # first tick (same gate as stop-go mode uses).
        if self._teleport_wait_stamp_ns is not None and \
                self._latest_scan_stamp_ns <= self._teleport_wait_stamp_ns:
            return
        self._take_step()

    def _slam_map_callback(self, msg: OccupancyGrid):
        """Decode ROS OccupancyGrid (-1/0/100/50) into the builder's grid
        (-1/0/1/2) and mutate in place so the next reveal sees the update."""
        if self._slam_grid_map is None:
            return
        data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        internal = np.full_like(data, -1, dtype=np.int8)
        internal[data == 0]   = 0   # CELL_FREE
        internal[data == 100] = 1   # CELL_OCCUPIED
        internal[data == 50]  = 2   # CELL_OUTLET
        if internal.shape != self._slam_grid_map.grid.shape:
            self.get_logger().warn(
                f'SLAM map shape {internal.shape} != builder grid '
                f'{self._slam_grid_map.grid.shape} — ignoring update.'
            )
            return
        # Mutate in place so the obs builder's reference stays valid.
        np.copyto(self._slam_grid_map.grid, internal)
        self._slam_map_count += 1
        if self._slam_map_count == 1:
            n_known = int((internal != -1).sum())
            self.get_logger().info(
                f'First SLAM map received: {n_known}/{internal.size} cells observed.'
            )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _take_step(self):
        if self._search_complete:
            return
        if self._robot_x is None or self._robot_y is None:
            return
        if self._obs_builder is None or not self._obs_builder.ready:
            return

        if self._step_in_episode >= self._max_steps:
            self.get_logger().warn(
                f'[Episode {self._episode}] Max steps ({self._max_steps}) reached. '
                f'Episode failed.'
            )
            self._end_episode(success=False)
            return

        dist_to_source = math.hypot(
            self._robot_x - self._true_source_x,
            self._robot_y - self._true_source_y,
        )
        if dist_to_source < cfg.D_SUCCESS:
            self.get_logger().info(
                f'[Episode {self._episode}] Source found at step {self._step_in_episode}! '
                f'Distance: {dist_to_source:.3f} m'
            )
            self._end_episode(success=True)
            return

        # Per-step update (reveals occupancy + splats detection + decays recency).
        self._obs_builder.record_step()

        built = self._obs_builder.build()
        if built is None:
            return
        spatial_np, ctx_np = built

        # Optional debug dump — spatial channels + world grid snapshot.
        if self._debug_dump_dir and self._debug_dump_every > 0 and \
           (self._step_in_episode % self._debug_dump_every == 0):
            self._dump_debug_image(spatial_np, ctx_np)

        spatial_t = torch.tensor(spatial_np, dtype=torch.float32, device=self._device).unsqueeze(0)
        ctx_t     = torch.tensor(ctx_np,     dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self._agent.get_action_and_value(spatial_t, ctx_t)
        action_np = action.cpu().numpy().flatten()  # (2,)

        target_x, target_y, theta = _action_to_target(
            self._robot_x, self._robot_y, action_np, cfg.STEP_SIZE
        )

        if self._step_log_every <= 1 or (self._step_in_episode % self._step_log_every == 0):
            action_text = np.array2string(action_np, precision=3, separator=',')
            gas_text   = f'{self._latest_gas_raw:.3f}'   if self._latest_gas_raw   is not None else 'n/a'
            wind_s_txt = f'{self._latest_wind_speed:.2f}' if self._latest_wind_speed is not None else 'n/a'
            wind_d_txt = f'{math.degrees(self._latest_wind_dir):.1f}' if self._latest_wind_dir is not None else 'n/a'
            lidar_txt  = f'{self._latest_lidar_min:.2f}'  if self._latest_lidar_min  is not None else 'n/a'
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'Pos ({self._robot_x:.2f},{self._robot_y:.2f}) '
                f'Action {action_text} θ={math.degrees(theta):.1f}deg → '
                f'Target ({target_x:.2f},{target_y:.2f}) | '
                f'd2src={dist_to_source:.2f}m gas={gas_text} '
                f'wind=({wind_s_txt}m/s,{wind_d_txt}deg) lidar_min={lidar_txt}m'
            )

        target_x, target_y, collided = self._clamp_to_free(
            self._robot_x, self._robot_y, target_x, target_y, theta
        )
        if collided:
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'COLLISION — clamped to ({target_x:.2f},{target_y:.2f})'
            )

        if self._publish_action_marker:
            self._publish_next_action_marker(target_x, target_y)

        if self._use_nav2 and self._navigator is not None:
            # Send Nav2 goal aligned with the commanded motion direction.
            # Without `use_orientation=True`, Navigator defaults goal yaw to
            # 0 (facing +x), forcing the robot to rotate-to-east at the end
            # of every step — a big wall-time tax. Aligning goal yaw to
            # `theta` means the robot drives in a straight line, arrives
            # facing the motion direction, and the next step often begins
            # already pointed roughly correctly.
            self._is_moving = True
            self._navigator.send_goal(
                target_x, target_y,
                yaw=theta, use_orientation=True,
                tolerance=self._nav_goal_tolerance,
            )
        else:
            # Teleport path (original behaviour)
            msg = PoseWCS()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x = float(target_x)
            msg.pose.pose.position.y = float(target_y)
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance = [0.0] * 36
            self._teleport_pub.publish(msg)

        self._step_in_episode += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_valid_slam(self, position, radius=cfg.ROBOT_RADIUS):
        """SLAM-aware collision predicate.

        `OccupancyGridMap.is_valid` treats any non-zero cell as blocked, which
        is wrong for a SLAM grid containing {-1 unknown, 0 free, 1 wall,
        2 outlet} — unknowns (-1) would be treated as walls, blocking most of
        the map before the robot has explored. Instead, only count cells
        explicitly marked CELL_OCCUPIED (1) as walls. Nav2's lidar-built
        costmap still protects against actual walls revealed mid-motion.
        """
        gmap = self._slam_grid_map
        if gmap is None:
            return True
        ox = float(getattr(gmap, 'origin_x', 0.0) or 0.0)
        oy = float(getattr(gmap, 'origin_y', 0.0) or 0.0)
        gx = int((position[0] - ox) / gmap.resolution)
        gy = int((position[1] - oy) / gmap.resolution)
        if not (0 <= gx < gmap.grid_width and 0 <= gy < gmap.grid_height):
            return False  # out-of-map is blocked
        rc = int(np.ceil(radius / gmap.resolution))
        rc_sq = rc * rc
        for ddx in range(-rc, rc + 1):
            for ddy in range(-rc, rc + 1):
                if ddx * ddx + ddy * ddy > rc_sq:
                    continue
                cx, cy = gx + ddx, gy + ddy
                if 0 <= cx < gmap.grid_width and 0 <= cy < gmap.grid_height:
                    if gmap.grid[cy, cx] == 1:
                        return False
        return True

    def _clamp_to_free(self, rx: float, ry: float,
                       tx: float, ty: float, theta: float) -> tuple:
        """If the target is invalid (blocked), the robot stays in place
        (mirrors training's ``if not collision: self._robot_pos = new_pos``).

        Uses the SLAM map when ``use_slam_map`` is enabled so the pre-check
        only knows about walls the robot has actually observed — matching the
        observation-side cheating-level. Falls back to GT occupancy otherwise.
        """
        if self._use_slam_map and self._slam_grid_map is not None:
            valid = self._is_valid_slam((tx, ty), radius=cfg.ROBOT_RADIUS)
        elif hasattr(self, '_occ_map'):
            valid = self._occ_map.is_valid((tx, ty), radius=cfg.ROBOT_RADIUS)
        else:
            valid = True
        if valid:
            return tx, ty, False
        return rx, ry, True

    def _end_episode(self, success: bool):
        result = 'SUCCESS' if success else 'FAILURE'
        # Log travel distance on the same line that summarizer greps for
        # "SUCCESS/FAILURE" so the batch aggregate can pick it up without
        # changing the multi-line parse.
        self.get_logger().info(
            f'[Episode {self._episode}] {result} — {self._step_in_episode} steps '
            f'travel_distance_m={self._travel_distance_m:.2f}'
        )
        self._episode += 1

        if self._episode >= self._num_episodes:
            self.get_logger().info(
                f'Completed {self._num_episodes} episode(s). Shutting down.'
            )
            self._search_complete = True
            # os._exit() FIRST: calling rclpy.shutdown() from inside a
            # subscription callback can deadlock on the C++ executor (we're
            # mid-callback and shutdown tries to tear down the very thing
            # that's running us), which would prevent os._exit() from being
            # reached. Bypass rclpy entirely.
            os._exit(0)

        self._step_in_episode = 0
        self._travel_distance_m = 0.0
        self._last_pose_xy = None  # discard the upcoming teleport jump
        self._is_moving = False     # clear any pending Nav2 state
        self._obs_builder.reset()
        self._obs_builder.robot_x = self._robot_x
        self._obs_builder.robot_y = self._robot_y

        if self._start_x > -998.0 and self._start_y > -998.0:
            self.get_logger().info(
                f'Starting episode {self._episode} — teleporting to '
                f'({self._start_x:.2f}, {self._start_y:.2f})'
            )
            self._start_teleport_done = False

        self._last_step_time_ns = self.get_clock().now().nanoseconds

    def _dump_debug_image(self, spatial_np: np.ndarray, ctx_np: np.ndarray):
        """Save a 2×3 PNG: 5 spatial channels + the world grid the obs came from.

        Layout:
            [is_known] [is_wall] [gas]
            [recency ] [det_ct ] [world grid w/ robot]
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            self.get_logger().warn(f'matplotlib unavailable, skipping dump: {e}')
            return

        ch_names = ['is_known', 'is_wall', 'gas', 'recency', 'det_count']
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for i, name in enumerate(ch_names):
            ax = axes.flat[i]
            data = spatial_np[i]
            vmin, vmax = (-1.0, 1.0) if name in ('gas',) else (0.0, 1.0)
            im = ax.imshow(data, origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax)
            ax.set_title(f'{name}')
            ax.axvline(49, color='r', lw=0.3, alpha=0.6)
            ax.axhline(49, color='r', lw=0.3, alpha=0.6)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # World grid (source of reveal): -1 unknown, 0 free, 1 wall, 2 outlet
        ax = axes.flat[5]
        src_grid = (
            self._slam_grid_map.grid if self._use_slam_map else self._occ_map.grid
        )
        cmap = plt.get_cmap('gray_r')
        # Render unknowns as neutral gray, walls as black, free as white
        disp = np.full(src_grid.shape, 0.5, dtype=np.float32)   # unknown = 0.5
        disp[src_grid == 0] = 0.0                                # free = white
        disp[src_grid == 1] = 1.0                                # wall = black
        disp[src_grid == 2] = 0.8                                # outlet = dark-ish
        ax.imshow(disp, origin='lower', cmap=cmap, vmin=0, vmax=1)
        # Robot marker
        try:
            ox = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
            oy = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
            res = float(self._occ_map.resolution)
            gx = (self._robot_x - ox) / res
            gy = (self._robot_y - oy) / res
            ax.plot(gx, gy, 'ro', markersize=6)
        except Exception:
            pass
        which = 'SLAM' if self._use_slam_map else 'GT'
        n_known = int((src_grid != -1).sum())
        n_walls = int((src_grid == 1).sum())
        ax.set_title(f'{which} grid — known={n_known} walls={n_walls}')
        ax.axis('off')

        fig.suptitle(
            f'Ep {self._episode} Step {self._step_in_episode:04d}   '
            f'pos=({self._robot_x:.2f},{self._robot_y:.2f})   '
            f'ctx=[speed={ctx_np[0]:.2f}, cosθ={ctx_np[1]:.2f}, sinθ={ctx_np[2]:.2f}, '
            f't={ctx_np[3]:.3f}]',
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fn = os.path.join(
            self._debug_dump_dir,
            f'ep{self._episode:02d}_step{self._step_in_episode:04d}.png',
        )
        fig.savefig(fn, dpi=80)
        plt.close(fig)

    def _publish_next_action_marker(self, target_x: float, target_y: float):
        if self._robot_x is None or self._robot_y is None:
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gaden_rl_image_next_action'
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.a = 0.95
        marker.color.r = 0.25
        marker.color.g = 0.70
        marker.color.b = 0.95
        marker.points = [
            Point(x=float(self._robot_x), y=float(self._robot_y), z=0.12),
            Point(x=float(target_x),      y=float(target_y),      z=0.12),
        ]
        self._next_action_pub.publish(marker)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GadenRLNodeImage()
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
