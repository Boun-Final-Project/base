"""The generic gsl_bench harness node: one episode, any agent.

This is gaden_rl_node.py with the policy carved out and a GSLAgent plugged in.
It owns everything that is not method-specific:

    sensors -> Observation -> agent.observe()/act() -> Waypoint
            -> hop cap -> clamp to free space -> Nav2 drive (per-step timeout)
            -> success / max_steps / env-dead termination -> result.json

Run one episode and exit:

    ros2 run gsl_bench runner_node --ros-args \
        -p agent:=random_walk \
        -p true_source_x:=2.0 -p true_source_y:=4.5 \
        -p start_x:=8.1 -p start_y:=1.5 \
        -p result_file:=/tmp/run/result.json
"""
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import rclpy
import yaml
from rclpy.clock import Clock, ClockType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from olfaction_msgs.msg import Anemometer, GasSensor
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import tf2_ros

from efe_igdm.mapping.occupancy_grid import (
    OccupancyGridMap, load_3d_occupancy_grid_from_service,
)
from efe_igdm.planning.navigator import Navigator

from gsl_bench.agent import MapInfo, ScenarioInfo
from gsl_bench.harness.obs_cache import SensorCache
from gsl_bench.registry import load_agent_class

# The escape planner lives in gaden_transfer (src/base). Imported lazily in
# _setup_escape so the harness runs without it when escape is off.
_SRC_BASE = '/home/efe/ros2_ws/src/base'
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)


class RunnerNode(Node):
    """Runs exactly one episode of one agent, then writes result.json and exits."""

    def __init__(self):
        super().__init__('gsl_bench_runner')

        self._declare_parameters()
        self._load_parameters()
        self._init_state()

        if self._map_source == 'service':
            self._load_occupancy_map()
        else:
            # SLAM mode: fetch the TRUE map extent once, HERE in __init__ (before
            # the executor spins, so load_3d_occupancy_grid_from_service's own
            # spin_until_future_complete is safe — calling it from a live spinning
            # callback deadlocks). The live SLAM grid still owns the occupancy
            # CONTENTS; we only need the true width/height so the policy's
            # position + gas-history features normalize by the real map size the
            # checkpoint was trained on, not the tiny initial SLAM patch.
            self._true_map_wh = self._fetch_true_map_size()

        self._load_agent()

        if self._map_source == 'service':
            self._setup_escape()

        self._init_ros_interfaces()

        if self._map_source == 'service':
            self._agent.reset(ScenarioInfo(
                name=self._scenario_name,
                map=self._map_info,
                start_x=self._start_x,
                start_y=self._start_y,
                max_steps=self._max_steps,
                step_size_hint=self._step_size_hint,
            ))
            self._agent_reset_done = True

            self.get_logger().info(
                f'gsl_bench runner ready | agent={self._agent_spec} '
                f'({self._agent.name()}) | scenario={self._scenario_name} | '
                f'max_steps={self._max_steps} '
                f'(dist_budget={self._max_travel_distance_m:.1f}m, '
                f'time_budget={self._max_sim_time_s:.1f}s) | '
                f'escape={"ON" if self._escape else "OFF"} | '
                f'pose={self._pose_source} | map={self._map_source}')
            self.get_logger().info(
                f'Map: {self._map_info.width_m:.1f} x {self._map_info.height_m:.1f} m | '
                f'start=({self._start_x:.2f},{self._start_y:.2f})')
        else:
            self.get_logger().info(
                f'gsl_bench runner waiting for SLAM map on {self._slam_map_topic} ...')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('agent', '')
        self.declare_parameter('agent_config', '')       # YAML path -> dict -> agent __init__
        self.declare_parameter('scenario_name', '')
        self.declare_parameter('sim_name', 'sim1')

        # Success check only. NEVER handed to the agent.
        self.declare_parameter('true_source_x', 0.0)
        self.declare_parameter('true_source_y', 0.0)
        # One-time initial placement via /initialpose.
        self.declare_parameter('start_x', -999.0)
        self.declare_parameter('start_y', -999.0)

        self.declare_parameter('max_steps', 600)
        # Oracle-derived per-scenario budgets (gsl_bench.eval.oracle). 0.0 = disabled,
        # same sentinel convention as drive_timeout's eff <= 0.0.
        self.declare_parameter('max_travel_distance_m', 0.0)
        self.declare_parameter('max_sim_time_s', 0.0)
        self.declare_parameter('step_delay', 0.5)
        self.declare_parameter('success_radius', 0.5)
        self.declare_parameter('nav_goal_tolerance', 0.10)
        self.declare_parameter('drive_timeout', 30.0)
        self.declare_parameter('nav2_server_timeout', 300.0)
        self.declare_parameter('pose_stale_timeout', 120.0)
        self.declare_parameter('max_hop', 1.0)
        self.declare_parameter('robot_radius', 0.25)
        self.declare_parameter('step_size_hint', 0.5)

        self.declare_parameter('occupancy_service', '/gaden_environment/occupancyMap3D')
        self.declare_parameter('occupancy_z_level', 5)
        self.declare_parameter('occupancy_timeout', 300.0)

        self.declare_parameter('namespace', 'PioneerP3DX')
        self.declare_parameter('escape', False)
        self.declare_parameter('publish_action_marker', True)
        self.declare_parameter('result_file', '')
        self.declare_parameter('pose_source', 'ground_truth')
        self.declare_parameter('map_source', 'service')
        self.declare_parameter('slam_map_topic', '')
        self.declare_parameter('motion', 'nav2')
        self.declare_parameter('nav_profile', 'standard')

    def _load_parameters(self):
        g = self.get_parameter
        self._agent_spec: str = g('agent').value
        self._agent_config_path: str = g('agent_config').value
        self._scenario_name: str = g('scenario_name').value
        self._sim_name: str = g('sim_name').value

        self._true_source_x: float = float(g('true_source_x').value)
        self._true_source_y: float = float(g('true_source_y').value)
        self._start_x: float = float(g('start_x').value)
        self._start_y: float = float(g('start_y').value)

        self._max_steps: int = int(g('max_steps').value)
        self._max_travel_distance_m: float = float(g('max_travel_distance_m').value)
        self._max_sim_time_s: float = float(g('max_sim_time_s').value)
        self._step_delay: float = float(g('step_delay').value)
        self._success_radius: float = float(g('success_radius').value)
        self._nav_goal_tolerance: float = float(g('nav_goal_tolerance').value)
        self._drive_timeout: float = float(g('drive_timeout').value)
        self._pose_stale_timeout: float = float(g('pose_stale_timeout').value)
        self._max_hop: float = float(g('max_hop').value)
        self._robot_radius: float = float(g('robot_radius').value)
        self._step_size_hint: float = float(g('step_size_hint').value)

        self._occ_service: str = g('occupancy_service').value
        self._occ_z: int = int(g('occupancy_z_level').value)
        self._occ_timeout: float = float(g('occupancy_timeout').value)

        self._ns: str = g('namespace').value
        self._escape_enabled: bool = bool(g('escape').value)
        self._publish_marker: bool = bool(g('publish_action_marker').value)
        self._result_file: str = g('result_file').value

        self._pose_source: str = g('pose_source').value
        self._map_source: str = g('map_source').value
        self._slam_map_topic: str = g('slam_map_topic').value
        self._motion: str = g('motion').value
        self._nav_profile: str = g('nav_profile').value
        if not self._slam_map_topic:
            self._slam_map_topic = f'/{self._ns}/map'

        if not self._agent_spec:
            raise ValueError("Parameter 'agent' is required (e.g. -p agent:=random_walk).")
        if not self._result_file:
            raise ValueError("Parameter 'result_file' is required.")

    def _init_state(self):
        self._cache = SensorCache(lidar_max=3.0)
        self._step: int = 0
        self._done: bool = False
        self._placed: bool = False          # initial /initialpose published
        self._is_moving: bool = False
        self._drive_goal_time = None        # node clock at goal send (sim-side timeout)
        self._drive_goal_wall: Optional[float] = None   # monotonic mirror (wall backstop)
        self._drive_canceling: bool = False
        self._teleport_wait_stamp_ns: Optional[int] = None
        self._last_step_time_ns: int = 0
        self._episode_start_ns: Optional[int] = None
        self._n_escapes: int = 0
        self._n_clamped: int = 0
        self._n_drive_timeouts: int = 0
        self._n_action_starts: int = 0
        self._n_goal_updates: int = 0
        self._navigator = None
        self._escape = None
        self._slam_enabled = False
        self._tf_buffer = None
        self._tf_listener = None
        self._gt_x: Optional[float] = None
        self._gt_y: Optional[float] = None
        self._gt_theta: Optional[float] = None
        self._agent_reset_done: bool = False
        self._slam_map_sub = None
        self._tf_fallback_count: int = 0
        self._true_map_wh: Optional[tuple] = None
        self._motion_mode: str = 'stop_go'
        self._decision_period_s: Optional[float] = None
        self._goal_update_pub = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _fetch_true_map_size(self):
        """Query GADEN's ground-truth occupancy service for the real map
        width/height only. MUST be called from __init__ (before the executor
        spins): load_3d_occupancy_grid_from_service spins the node itself, which
        deadlocks if the node is already being spun by the executor. Returns
        (w, h) in meters, or None if unreachable (falls back to SLAM patch)."""
        try:
            _grid, _outlet, params = load_3d_occupancy_grid_from_service(
                self, z_level=self._occ_z, service_name=self._occ_service,
                timeout_sec=self._occ_timeout)
            w = float(params['env_max'][0] - params['env_min'][0])
            h = float(params['env_max'][1] - params['env_min'][1])
            self.get_logger().info(f'True map size from service: {w:.1f}x{h:.1f} m')
            return (w, h)
        except Exception as exc:
            self.get_logger().warn(
                f'Could not fetch true map size ({exc}); '
                f'position/gas features will use the SLAM patch size.')
            return None

    def _load_occupancy_map(self):
        self.get_logger().info(
            f'Waiting up to {self._occ_timeout:.0f}s for {self._occ_service} ...')
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

        self._map_info = MapInfo(
            grid=self._occ_map.grid,
            resolution=float(self._occ_map.resolution),
            origin_x=float(self._occ_map.origin_x),
            origin_y=float(self._occ_map.origin_y),
            width_m=float(self._occ_map.real_world_width),
            height_m=float(self._occ_map.real_world_height),
        )
        self.get_logger().info(
            f'Occupancy map: {self._occ_map.width}x{self._occ_map.height} cells, '
            f'{self._map_info.width_m:.2f}x{self._map_info.height_m:.2f} m')

    def _load_agent(self):
        self._agent_config = {}
        if self._agent_config_path:
            with open(self._agent_config_path) as f:
                self._agent_config = yaml.safe_load(f) or {}
        cls = load_agent_class(self._agent_spec)
        try:
            self._agent = cls(self._agent_config)
        except TypeError:
            # Agents with a no-arg __init__ are allowed.
            self._agent = cls()
        self._agent.initialize()
        self._motion_mode = getattr(self._agent, 'motion_mode', 'stop_go')
        if self._motion_mode not in ('stop_go', 'continuous'):
            raise ValueError(f'Unsupported agent motion_mode={self._motion_mode!r}')
        rate = getattr(self._agent, 'decision_rate_hz', None)
        self._decision_period_s = (1.0 / float(rate)
                                   if self._motion_mode == 'continuous' and rate else None)
        requirements = {
            'pose_source': (getattr(self._agent, 'required_pose_source', None), self._pose_source),
            'map_source': (getattr(self._agent, 'required_map_source', None), self._map_source),
            'motion': (getattr(self._agent, 'required_motion', None), self._motion),
            'nav_profile': (getattr(self._agent, 'required_nav_profile', None), self._nav_profile),
        }
        bad = [f'{name}={actual!r} (requires {required!r})'
               for name, (required, actual) in requirements.items()
               if required is not None and required != actual]
        if bad:
            raise ValueError(f'{self._agent.name()} incompatible harness: ' + ', '.join(bad))
        if self._motion_mode == 'continuous' and self._escape_enabled:
            raise ValueError('Canonical continuous ADSM runs require escape:=false')
        requested_reach = float(getattr(self._agent, 'max_goal_distance', self._max_hop))
        self._effective_max_hop = max(self._max_hop, requested_reach)

    def _setup_escape(self):
        """Opt-in stuck-escape: SLAM the lidar, and when the agent is circling or
        the map stops growing, drive to an unexplored frontier in short hops.

        Default OFF so every method is scored on its own behaviour; when ON it is
        recorded in result.json (it historically moves the 7-map score 28 -> 32).
        """
        if not self._escape_enabled:
            return
        try:
            from gaden_transfer.gaden_transfer_lidar.escape_planner import CirclingEscape
        except ImportError as exc:
            self.get_logger().error(
                f'escape:=true but the escape planner is not importable ({exc}) — disabled.')
            return
        try:
            env_flag = os.environ.get('OSL_ESCAPE_USE_LIVE_GRID', None)
            if env_flag is not None:
                use_live = env_flag == '1'
            else:
                use_live = (self._map_source == 'slam')

            live_provider = (lambda: self._occ_map) if use_live else None

            self._escape = CirclingEscape(
                self._occ_map,
                robot_radius=self._robot_radius,
                logger=self.get_logger(),
                win=int(os.environ.get('OSL_ESCAPE_WIN', '25')),
                ratio=float(os.environ.get('OSL_ESCAPE_RATIO', '0.2')),
                streak=int(os.environ.get('OSL_ESCAPE_STREAK', '35')),
                cooldown=int(os.environ.get('OSL_ESCAPE_COOLDOWN', '40')),
                min_dist=float(os.environ.get('OSL_ESCAPE_MINDIST', '3.0')),
                grow_win=int(os.environ.get('OSL_ESCAPE_GROW_WIN', '120')),
                grow_min=int(os.environ.get('OSL_ESCAPE_GROW_MIN', '250')),
                frontier_min_cells=int(os.environ.get('OSL_ESCAPE_FRONTIER_MIN', '12')),
                target_mode=os.environ.get('OSL_ESCAPE_TARGET', 'largest'),
                use_live_grid=use_live,
                live_map_provider=live_provider,
            )
            mode_txt = ('live SLAM grid' if use_live else 'private LidarMapper grid')
            self.get_logger().info(
                f'Stuck-escape ENABLED (recorded in result.json). '
                f'Map: {mode_txt}.')
        except Exception as exc:
            self.get_logger().error(f'Failed to init escape: {exc} — disabled.')
            self._escape = None

    def _init_ros_interfaces(self):
        ns = self._ns
        # Heavy sensor/step callbacks (pose lookup, policy inference, goal send) go
        # in their own MutuallyExclusive group so they DON'T share the node's default
        # group with rclpy's internal /clock subscription (added by use_sim_time).
        # Otherwise the busy /ground_truth callback monopolises the default group and
        # /clock is never processed under the MultiThreadedExecutor -> the sim clock
        # FREEZES (see the watchdog comment below), the TF pose lookup then requests a
        # stale frozen time and throws "extrapolation into the past" every tick, and
        # the runner holds its origin placement pose and drives to wrong-frame goals.
        # Isolating them keeps /clock (default group) serviceable on the 2nd thread.
        self._cbg = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            PoseWithCovarianceStamped, f'/{ns}/ground_truth', self._pose_callback, 10,
            callback_group=self._cbg)
        self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self._gas_callback, 10,
            callback_group=self._cbg)
        self.create_subscription(
            Anemometer, '/fake_anemometer/WindSensor_reading', self._wind_callback, 10,
            callback_group=self._cbg)
        self.create_subscription(
            LaserScan, f'/{ns}/laser_scanner', self._lidar_callback, 10,
            callback_group=self._cbg)

        self._place_pub = self.create_publisher(
            PoseWithCovarianceStamped, f'/{ns}/initialpose', 10)
        self._marker_pub = self.create_publisher(Marker, '/gsl_bench/target', 1)
        # Always-on goal pose (frame map) for visualization: the Nav2 goal itself
        # goes out as an action (not RViz-visualizable), so we mirror every target
        # here — normal steps, escape hops, and continuous retargets alike.
        self._goal_pub = self.create_publisher(PoseStamped, '/gsl_bench/goal', 1)
        if self._motion_mode == 'continuous':
            self._goal_update_pub = self.create_publisher(
                PoseStamped, f'/{ns}/goal_update', 1)

        if self._pose_source == 'tf':
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self.get_logger().info(
                'Pose source: TF lookup (map -> {ns}_base_link) — noisy SLAM pose')

        if self._map_source == 'slam':
            self._slam_map_sub = self.create_subscription(
                OccupancyGrid, self._slam_map_topic,
                self._slam_map_callback, 10, callback_group=self._cbg)
            self.get_logger().info(
                f'Map source: SLAM topic ({self._slam_map_topic})')

        # Nav2 pre-flight: Navigator.__init__ waits on the action server with NO
        # timeout, so confirm it is up here and fail fast instead of hanging.
        nav_wait = float(self.get_parameter('nav2_server_timeout').value)
        pre = ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
        self.get_logger().info(
            f'Waiting up to {nav_wait:.0f}s for Nav2 action server /{ns}/navigate_to_pose ...')
        if not pre.wait_for_server(timeout_sec=nav_wait):
            self.get_logger().error(
                f'Nav2 action server /{ns}/navigate_to_pose not available after '
                f'{nav_wait:.0f}s — aborting.')
            self._write_result(success=False, status='env_dead')
            os._exit(2)
        pre.destroy()
        self._navigator = Navigator(self, on_complete_callback=self._on_nav_complete)
        self.get_logger().info(
            f'Motion: Nav2 physical drive ({self._motion_mode}, '
            f'tolerance={self._nav_goal_tolerance:.2f} m, '
            f'max_goal={self._effective_max_hop:.2f} m)')

        # Liveness watchdog. MUST run on a STEADY_TIME clock: a default timer follows
        # the node clock, which freezes with the simulator — exactly the scenario this
        # watchdog exists to catch.
        self._watchdog = self.create_timer(
            2.0, self._wall_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'Watchdog armed: env-dead abort after {self._pose_stale_timeout:.0f}s '
            'without a pose; wall backstop on the per-step drive timeout.')

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _gas_callback(self, msg: GasSensor):
        self._cache.update_gas(msg)

    def _wind_callback(self, msg: Anemometer):
        self._cache.update_wind(msg)

    def _lidar_callback(self, msg: LaserScan):
        self._cache.update_lidar(msg)
        # Fold the scan into the escape planner's online map, but only once the
        # episode is really running: before the first step the pose can still be the
        # pre-placement spawn, which would map a phantom region.
        if self._escape is not None and self._slam_enabled:
            self._escape.update_scan(msg, self._cache.x, self._cache.y, self._cache.theta)

    def _slam_map_callback(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        # Threshold at 50, not "== 0": slam_toolbox publishes an already
        # binarized {0, 100, -1}, so this is a no-op for it, but cartographer's
        # occupancy_grid_node publishes a continuous 0-100 probability (any
        # cell short of absolute certainty reads e.g. 24, not exactly 0) —
        # treating "== 0" as the only free value made every cell "occupied"
        # and boxed the robot in at spawn under cartographer.
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        grid = np.where(grid < 0, -1, np.where(grid >= 50, 1, 0)).astype(np.int8)

        params = {
            'env_min': [ox, oy, 0.0],
            'env_max': [ox + w * res, oy + h * res, 0.0],
            'num_cells': [w, h, 1],
            'cell_size': res,
            'z_level': 0,
            'z_height': 0.0,
        }
        self._occ_map = OccupancyGridMap(grid, params)
        self._map_info = MapInfo(
            grid=self._occ_map.grid,
            resolution=float(res),
            origin_x=float(ox),
            origin_y=float(oy),
            width_m=float(w * res),
            height_m=float(h * res),
        )

        if not self._agent_reset_done:
            self._agent_reset_done = True
            self._setup_escape()
            # The agent gets the TRUE map size for its position/gas normalization
            # (matching the training convention) while _map_info keeps the SLAM
            # patch size for escape/collision. Grid stays the SLAM grid either way.
            agent_map = self._map_info
            if self._true_map_wh is not None:
                agent_map = MapInfo(
                    grid=self._map_info.grid, resolution=self._map_info.resolution,
                    origin_x=self._map_info.origin_x, origin_y=self._map_info.origin_y,
                    width_m=self._true_map_wh[0], height_m=self._true_map_wh[1])
            self._agent.reset(ScenarioInfo(
                name=self._scenario_name,
                map=agent_map,
                start_x=self._start_x,
                start_y=self._start_y,
                max_steps=self._max_steps,
                step_size_hint=self._step_size_hint,
            ))
            self.get_logger().info(
                f'SLAM map received ({w}x{h}, {w*res:.1f}x{h*res:.1f}m) — agent ready | '
                f'agent={self._agent_spec} ({self._agent.name()}) | '
                f'scenario={self._scenario_name} | '
                f'max_steps={self._max_steps} '
                f'(dist_budget={self._max_travel_distance_m:.1f}m, '
                f'time_budget={self._max_sim_time_s:.1f}s) | '
                f'escape={"ON" if self._escape else "OFF"} | '
                f'pose={self._pose_source} | map={self._map_source}')

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """The tick. Everything downstream is driven by pose arrivals."""
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y
        gq = msg.pose.pose.orientation
        self._gt_theta = math.atan2(
            2.0 * (gq.w * gq.z + gq.x * gq.y),
            1.0 - 2.0 * (gq.y * gq.y + gq.z * gq.z))
        self._cache.update_pose(msg, set_position=(self._pose_source != 'tf'))

        if self._pose_source == 'tf' and self._tf_buffer is not None:
            try:
                tf = self._tf_buffer.lookup_transform(
                    'map', f'{self._ns}_base_link', rclpy.time.Time())
                self._cache.x = tf.transform.translation.x
                self._cache.y = tf.transform.translation.y
                q = tf.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self._cache.theta = math.atan2(siny_cosp, cosy_cosp)
                self._tf_fallback_count = 0
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                self._tf_fallback_count += 1
                if self._tf_fallback_count <= 3 or self._tf_fallback_count % 50 == 0:
                    self.get_logger().warn(
                        f'TF lookup map->{self._ns}_base_link failed '
                        f'(#{self._tf_fallback_count}): {exc}. '
                        f'Holding last SLAM pose.')

        if self._done:
            return

        # (1) Driving: hold the step loop until Nav2 completes, with a per-step timeout.
        if self._is_moving and self._motion_mode == 'stop_go':
            self._check_drive_timeout()
            return

        # (2) One-time initial placement.
        if not self._placed:
            self._placed = True
            if self._start_x > -998.0 and self._start_y > -998.0:
                self.get_logger().info(
                    f'Placing robot at start ({self._start_x:.2f}, {self._start_y:.2f})')
                self._place_robot(self._start_x, self._start_y)
                self._last_step_time_ns = self.get_clock().now().nanoseconds
                return

        # (3) Fresh-scan gate: without it, step 0's lidar is from the pre-placement pose.
        if (self._teleport_wait_stamp_ns is not None
                and self._cache.last_scan_stamp_ns <= self._teleport_wait_stamp_ns):
            return

        # (4) Rate gate.
        now_ns = self.get_clock().now().nanoseconds
        delay = (self._decision_period_s if self._motion_mode == 'continuous'
                 else self._step_delay)
        if now_ns - self._last_step_time_ns < int(delay * 1e9):
            return
        self._last_step_time_ns = now_ns

        if self._episode_start_ns is None:
            self._episode_start_ns = now_ns

        self._take_step()

    def _check_drive_timeout(self):
        """Cancel a wedged Nav2 goal so the agent re-decides instead of waiting ~150s
        for the behaviour tree to give up."""
        eff = self._drive_timeout
        if self._escape is not None and self._escape.escaping:
            eff = min(self._drive_timeout, 10.0)   # escape hops are short; fail them fast
        # Snapshot: _on_nav_complete nulls _drive_goal_time from another executor
        # thread, so re-reading it after the guard can yield None mid-subtraction.
        goal_time = self._drive_goal_time
        if eff <= 0.0 or goal_time is None or self._drive_canceling:
            return
        elapsed = (self.get_clock().now() - goal_time).nanoseconds * 1e-9
        if elapsed > eff:
            self._drive_canceling = True
            self._n_drive_timeouts += 1
            self.get_logger().warn(
                f'[Step {self._step:3d}] DRIVE TIMEOUT — goal not reached in '
                f'{elapsed:.1f}s (> {eff:.1f}s); canceling, re-deciding.')
            if self._navigator is not None:
                self._navigator.cancel_current_goal()

    def _on_nav_complete(self):
        """Nav2 goal finished (success / abort / reject / cancel)."""
        self._is_moving = False
        self._drive_goal_time = None
        self._drive_goal_wall = None
        self._drive_canceling = False

    def _wall_watchdog(self):
        """The only thing here that keeps ticking when the simulator is gone.

        Every other timeout is measured on the node clock and the step loop runs off
        pose callbacks — both driven by the sim. When the sim dies they die with it,
        and the run hangs until the batch wall timeout (observed: a single step
        blocking 6.3 h).
        """
        now_w = time.monotonic()

        # (1) Simulator liveness.
        if (self._pose_stale_timeout > 0.0 and self._cache.last_pose_wall is not None
                and not self._done):
            stale = now_w - self._cache.last_pose_wall
            if stale > self._pose_stale_timeout:
                self.get_logger().error(
                    f'ENV DEAD — no pose for {stale:.0f}s wall '
                    f'(> {self._pose_stale_timeout:.0f}s). Aborting.')
                self._write_result(success=False, status='env_dead')
                os._exit(3)

        # (2) Wall backstop for the per-step drive timeout: only fires when the
        # sim-time path cannot run at all. Deliberately slack so it never pre-empts it.
        if self._motion_mode == 'continuous':
            # A continuously retargeted action may legitimately span the episode;
            # Nav2's progress checker handles controller stalls in this mode.
            return
        if not self._is_moving or self._drive_canceling:
            return
        goal_wall = self._drive_goal_wall
        if goal_wall is None:
            return
        eff = self._drive_timeout
        if self._escape is not None and self._escape.escaping:
            eff = min(self._drive_timeout, 10.0)
        if eff <= 0.0:
            return
        limit = max(eff * 3.0, eff + 30.0)
        if now_w - goal_wall > limit:
            self._drive_canceling = True
            self._n_drive_timeouts += 1
            self.get_logger().warn(
                f'[Step {self._step:3d}] WALL-CLOCK DRIVE TIMEOUT — '
                f'{now_w - goal_wall:.1f}s wall (> {limit:.1f}s) with no goal completion; '
                'canceling, re-deciding.')
            if self._navigator is not None:
                self._navigator.cancel_current_goal()

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def _take_step(self):
        if self._done or not self._cache.ready():
            return
        if self._map_source == 'slam' and not self._agent_reset_done:
            return

        rx, ry = self._cache.x, self._cache.y

        eval_x = self._gt_x if self._gt_x is not None else rx
        eval_y = self._gt_y if self._gt_y is not None else ry

        # --- termination checks ---
        dist_to_source = math.hypot(eval_x - self._true_source_x, eval_y - self._true_source_y)
        if dist_to_source < self._success_radius:
            # The old drivers grep for this exact line; keep it verbatim.
            self.get_logger().info(
                f'Source found at step {self._step}!  Distance: {dist_to_source:.3f} m')
            self._end_episode(success=True, status='success')
            return
        if self._step >= self._max_steps:
            self.get_logger().warn(f'Max steps ({self._max_steps}) reached. Episode failed.')
            self._end_episode(success=False, status='max_steps')
            return
        if (self._max_travel_distance_m > 0.0
                and self._cache.travel_distance_m >= self._max_travel_distance_m):
            self.get_logger().warn(
                f'Max travel distance ({self._max_travel_distance_m:.1f} m) reached '
                f'({self._cache.travel_distance_m:.1f} m travelled). Episode failed.')
            self._end_episode(success=False, status='max_travel_distance')
            return
        if self._max_sim_time_s > 0.0 and self._sim_time() >= self._max_sim_time_s:
            self.get_logger().warn(
                f'Max sim time ({self._max_sim_time_s:.1f} s) reached '
                f'({self._sim_time():.1f} s elapsed). Episode failed.')
            self._end_episode(success=False, status='max_sim_time')
            return

        # --- optional escape (harness-level recovery) ---
        if self._escape is not None:
            self._slam_enabled = True   # pose is confirmed post-placement
            og = 1 if self._cache.gas_ppm > self._escape.src_gas_eps else 0
            self._escape.record_step(
                rx, ry, og, self._cache.wind_speed, self._cache.wind_direction)
            if self._escape.escaping:
                wp = self._escape.next_waypoint(rx, ry)
                if wp is not None:
                    self._drive(float(wp[0]), float(wp[1]), theta=None)
                    self._step += 1
                    return
                # path exhausted -> fall through and let the agent decide again
            else:
                info = self._escape.start_escape_if_stuck(rx, ry, self._step)
                if info is not None:
                    tx, ty, fsize, fdist, nwp = info
                    self._n_escapes += 1
                    self.get_logger().warn(
                        f'[Step {self._step:3d}] STUCK ({self._escape.stuck_reason}) → '
                        f'escape to ({tx:.2f},{ty:.2f}) via {nwp} hops '
                        f'[frontier {fsize} cells, {fdist:.1f}m away]')
                    wp = self._escape.next_waypoint(rx, ry)
                    if wp is not None:
                        self._drive(float(wp[0]), float(wp[1]), theta=None)
                        self._step += 1
                        return

        # --- the agent ---
        obs = self._cache.snapshot(self._step, self._sim_time(), self._map_info)
        try:
            self._agent.observe(obs)
            wp = self._agent.act()
        except Exception as exc:
            import traceback
            self.get_logger().error(f'Agent raised: {exc}\n{traceback.format_exc()}')
            self._end_episode(success=False, status='agent_error')
            return

        if wp is None:
            self.get_logger().error('Agent returned None from act(); expected a Waypoint.')
            self._end_episode(success=False, status='agent_error')
            return

        tx, ty = float(wp.x), float(wp.y)
        theta = wp.theta if wp.theta is None else float(wp.theta)

        # --- sanitize: cap the hop, then clamp into free space ---
        tx, ty = self._cap_hop(rx, ry, tx, ty)
        ray = math.atan2(ty - ry, tx - rx) if (tx, ty) != (rx, ry) else (theta or 0.0)
        tx, ty, collided = self._clamp_to_free(rx, ry, tx, ty, ray)
        if collided:
            self._n_clamped += 1

        lm = self._cache.lidar_min
        lidar_text = 'n/a' if lm is None else f'{lm:.2f}'
        # yaw diagnostic: the lidar array is rolled by _cache.theta (SLAM/TF yaw
        # in tf mode), while the physical scan is cast at the TRUE body yaw
        # (_gt_theta). Any gap here rotates the whole lidar pattern into wrong
        # obs slots. err wrapped to [-180,180].
        yaw_slam = math.degrees(self._cache.theta)
        if self._gt_theta is not None:
            yaw_err = math.degrees(math.atan2(
                math.sin(self._cache.theta - self._gt_theta),
                math.cos(self._cache.theta - self._gt_theta)))
            yaw_text = (f' yaw(slam={yaw_slam:.1f},gt={math.degrees(self._gt_theta):.1f},'
                        f'err={yaw_err:.1f}deg)')
        else:
            yaw_text = f' yaw(slam={yaw_slam:.1f}deg)'
        self.get_logger().info(
            f'[Step {self._step:3d}] Pos ({rx:.2f},{ry:.2f}) → Target ({tx:.2f},{ty:.2f}) | '
            f'd2src={dist_to_source:.2f}m gas={self._cache.gas_ppm:.3f} '
            f'wind=({self._cache.wind_speed:.2f}m/s,'
            f'{math.degrees(self._cache.wind_direction):.1f}deg) '
            f'lidar_min={lidar_text}m'
            f'{yaw_text}'
            f'{" COLLISION — clamped" if collided else ""}')

        if self._publish_marker:
            self._publish_target_marker(rx, ry, tx, ty)

        self._drive(tx, ty, theta if theta is not None else ray)
        self._step += 1

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _publish_goal_pose(self, x: float, y: float, theta: Optional[float]):
        """Mirror the current drive target on /gsl_bench/goal for RViz (frame map)."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        yaw = 0.0 if theta is None else float(theta)
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(msg)

    def _drive(self, x: float, y: float, theta: Optional[float]):
        """Physical Nav2 drive to (x, y). Never a teleport — that is eval policy."""
        self._publish_goal_pose(x, y, theta)
        if self._motion_mode == 'continuous' and self._is_moving:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            yaw = 0.0 if theta is None else float(theta)
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            self._goal_update_pub.publish(msg)
            self._n_goal_updates += 1
            return
        self._is_moving = True
        self._drive_canceling = False
        self._drive_goal_time = self.get_clock().now()
        self._drive_goal_wall = time.monotonic()
        if theta is None:
            self._navigator.send_goal(
                x, y, use_orientation=False, tolerance=self._nav_goal_tolerance)
        else:
            self._navigator.send_goal(
                x, y, yaw=float(theta), use_orientation=True,
                tolerance=self._nav_goal_tolerance)
        self._n_action_starts += 1

    def _place_robot(self, x: float, y: float):
        """The one teleport allowed per episode: initial placement."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance = [0.0] * 36
        self._place_pub.publish(msg)
        # Arm the fresh-scan gate: the first step must wait for a scan stamped
        # after this placement.
        self._teleport_wait_stamp_ns = self._cache.last_scan_stamp_ns

    def _cap_hop(self, rx, ry, tx, ty):
        """Shorten an over-long request along its own ray; agents can't outrun the sim."""
        d = math.hypot(tx - rx, ty - ry)
        if d <= self._effective_max_hop or d < 1e-9:
            return tx, ty
        s = self._effective_max_hop / d
        return rx + (tx - rx) * s, ry + (ty - ry) * s

    def _clamp_to_free(self, rx, ry, tx, ty, theta):
        """Validate the target; if blocked, walk back along the ray until free.

        Mirrors the training-time collision logic: the robot would simply not move
        if the target were inside a wall.

        In SLAM mode (map_source == 'slam') unknown cells are treated as passable
        so the robot can drive into unmapped territory and expand the map.
        """
        valid = (self._occ_map.is_valid_traversable((tx, ty), radius=self._robot_radius)
                 if self._map_source == 'slam'
                 else self._occ_map.is_valid((tx, ty), radius=self._robot_radius))
        if valid:
            return tx, ty, False
        step = math.hypot(tx - rx, ty - ry) - 0.05
        while step >= 0.1:
            cx = rx + step * math.cos(theta)
            cy = ry + step * math.sin(theta)
            valid = (self._occ_map.is_valid_traversable((cx, cy), radius=self._robot_radius)
                     if self._map_source == 'slam'
                     else self._occ_map.is_valid((cx, cy), radius=self._robot_radius))
            if valid:
                return cx, cy, True
            step -= 0.05
        return rx, ry, True   # boxed in: stay put

    def _publish_target_marker(self, rx, ry, tx, ty):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'gsl_bench_target'
        m.id = 1
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = 0.06, 0.12, 0.12
        m.color.a, m.color.r, m.color.g, m.color.b = 0.95, 0.95, 0.75, 0.05
        m.points = [Point(x=float(rx), y=float(ry), z=0.12),
                    Point(x=float(tx), y=float(ty), z=0.12)]
        self._marker_pub.publish(m)

    # ------------------------------------------------------------------
    # Episode end
    # ------------------------------------------------------------------

    def _sim_time(self) -> float:
        if self._episode_start_ns is None:
            return 0.0
        return (self.get_clock().now().nanoseconds - self._episode_start_ns) * 1e-9

    def _write_result(self, success: bool, status: str):
        rx = self._cache.x if self._cache.x is not None else float('nan')
        ry = self._cache.y if self._cache.y is not None else float('nan')
        eval_x = self._gt_x if self._gt_x is not None else rx
        eval_y = self._gt_y if self._gt_y is not None else ry
        final_d = (math.hypot(eval_x - self._true_source_x, eval_y - self._true_source_y)
                   if eval_x is not None else None)
        result = {
            'schema': 1,
            'scenario': self._scenario_name,
            'sim': self._sim_name,
            'agent': self._agent_spec,
            'agent_config': self._agent_config,
            'status': status,
            'success': bool(success),
            'steps': int(self._step),
            'sim_time_s': round(self._sim_time(), 2),
            'wall_time_s': None,               # filled in by episode_runner
            'travel_distance_m': round(self._cache.travel_distance_m, 3),
            'final_distance_m': None if final_d is None else round(final_d, 3),
            'source': [self._true_source_x, self._true_source_y],
            'start': [self._start_x, self._start_y],
            'escapes': self._n_escapes,
            'clamped_steps': self._n_clamped,
            'drive_timeouts': self._n_drive_timeouts,
            'nav_action_starts': self._n_action_starts,
            'goal_updates': self._n_goal_updates,
            'tf_fallback_count': self._tf_fallback_count,
            'agent_metadata': self._agent.metadata(),
            'harness': {
                'success_radius': self._success_radius,
                'max_steps': self._max_steps,
                'max_travel_distance_m': self._max_travel_distance_m,
                'max_sim_time_s': self._max_sim_time_s,
                'step_delay': self._step_delay,
                'nav_goal_tolerance': self._nav_goal_tolerance,
                'drive_timeout': self._drive_timeout,
                'escape': bool(self._escape is not None),
                'max_hop': self._max_hop,
                'effective_max_hop': self._effective_max_hop,
                'motion_mode': self._motion_mode,
                'decision_rate_hz': getattr(self._agent, 'decision_rate_hz', None),
                'robot_radius': self._robot_radius,
                'lidar': '360deg/72ray/3m',
                'motion': self._motion,
                'nav_profile': self._nav_profile,
                'pose_source': self._pose_source,
                'map_source': self._map_source,
            },
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._result_file)), exist_ok=True)
            with open(self._result_file, 'w') as f:
                json.dump(result, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            self.get_logger().error(f'Failed to write {self._result_file}: {exc}')

    def _end_episode(self, success: bool, status: str):
        self._done = True
        self.get_logger().info(
            f'{"SUCCESS" if success else "FAILURE"} ({status}) — {self._step} steps, '
            f'{self._cache.travel_distance_m:.1f} m travelled')
        self._write_result(success, status)
        # os._exit() and not rclpy.shutdown(): we are inside a subscription callback,
        # and shutdown here can deadlock tearing down the executor that is running us.
        # The result file is already flushed and fsynced.
        os._exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RunnerNode()
        # >=3 threads: one is free to service rclpy's /clock subscription (default
        # callback group) while the heavy sensor callbacks (their own _cbg group)
        # and the Nav2 action feedback run — otherwise the sim clock starves/freezes.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
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
