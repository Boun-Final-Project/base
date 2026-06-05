"""
GadenRLNode — deploy a pretrained PPO agent inside GADEN.

Subscribes to the same four topics used by efe_igdm, builds the 107-dim
observation vector, runs the policy, and sends Nav2 goals.

Usage
-----
    ros2 run reinforcement_learning gaden_rl_node \
        --ros-args \
        -p checkpoint:=/path/to/agent_10000000.pt \
        -p arch:=mlp \
        -p true_source_x:=2.0 \
        -p true_source_y:=4.5 \
        -p max_steps:=600

Checkpoint loading
------------------
The checkpoint path is set via the ROS2 parameter ``checkpoint``.  Pass it
on the command line (shown above), in a params.yaml file loaded by the
launch file, or by editing the default value below.  The checkpoint .pt
file is the one saved by training/train.py:

    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": ...,
        "update": ...,
    }, path)

Only ``model_state_dict`` is loaded here — the optimiser state is ignored.
"""

import os
import sys
import math
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from olfaction_msgs.msg import GasSensor, Anemometer
from gaden_msgs.srv import WindPosition
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

# RL package imports: add src/base/ to sys.path so both the installed entry
# point and direct execution can find the reinforcement_learning package.
_SRC_BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
if _SRC_BASE not in sys.path:
    sys.path.insert(0, _SRC_BASE)

from reinforcement_learning import config as cfg
from reinforcement_learning.models.actor_critic import (
    ActorCritic, ActorCriticModular, ActorCriticDualBackbone
)
from reinforcement_learning.models.actor_critic_spatial import ActorCriticSpatial
from .obs_builder import ObservationBuilder
from .spatial_obs_builder import SpatialObsBuilder

from geometry_msgs.msg import PoseWithCovarianceStamped as PoseWCS
from efe_igdm.mapping.occupancy_grid import (
    load_3d_occupancy_grid_from_service, OccupancyGridMap
)
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from efe_igdm.planning.navigator import Navigator

from .escape_planner import CirclingEscape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_agent(checkpoint_path: str, arch: str, device: torch.device,
                run_cfg: dict = None):
    """Load a trained agent from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Absolute path to a .pt checkpoint produced by training/train.py.
    arch : str
        Architecture tag used when the checkpoint was saved: 'mlp',
        'modular', or 'dual'.  Must match the original training flag.
    device : torch.device
        Where to place the model (cpu / cuda).
    run_cfg : dict, optional
        Values from config.json; used to set obs/lidar/gas dims correctly.

    Returns
    -------
    agent : nn.Module
        Loaded model in eval mode.
    """
    if run_cfg is None:
        run_cfg = {}
    lidar_len = run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)
    if ("GAS_HISTORY_LENGTH" in run_cfg and "GAS_FEATURES_PER_STEP" in run_cfg):
        gas_len = run_cfg["GAS_HISTORY_LENGTH"] * run_cfg["GAS_FEATURES_PER_STEP"]
    else:
        gas_len = cfg.GAS_HISTORY_LENGTH * cfg.GAS_FEATURES_PER_STEP
    obs_dim = run_cfg.get("STATE_DIM", cfg.STATE_DIM)

    if arch == 'spatial':
        agent = ActorCriticSpatial()
    elif arch == 'dual':
        agent = ActorCriticDualBackbone(obs_dim=obs_dim, gas_len=gas_len, lidar_len=lidar_len)
    elif arch == 'modular':
        agent = ActorCriticModular(obs_dim=obs_dim, gas_len=gas_len, lidar_len=lidar_len)
    else:
        agent = ActorCritic(obs_dim=obs_dim)

    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt['model_state_dict'])
    agent.to(device)
    agent.eval()
    return agent


def _load_run_config(checkpoint_path: str) -> dict:
    """Read config.json from the checkpoint's parent run directory.

    Looks for config.json two levels up from the checkpoint file:
      .../runs/<run-name>/checkpoints/agent_STEP.pt
                         ^^^^^^^^^^^ config.json lives here
    Returns an empty dict if not found.
    """
    run_dir = Path(checkpoint_path).parent.parent
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _action_to_target(robot_x: float, robot_y: float,
                       action: np.ndarray, arch: str,
                       step_size: float) -> Tuple[float, float, float]:
    """Convert policy output to a (x, y) goal 0.5 m from the robot.

    Beta arch (mlp / modular): action is shape (1,), scalar in [0, 1].
        theta = action[0] * 2π

    Dual / spatial arch: action is shape (2,), (cos θ, sin θ).
        theta = atan2(sin, cos)
    """
    if arch in ('dual', 'spatial'):
        cos_t, sin_t = float(action[0]), float(action[1])
        theta = math.atan2(sin_t, cos_t)
    else:
        theta = float(action[0]) * 2.0 * math.pi

    target_x = robot_x + step_size * math.cos(theta)
    target_y = robot_y + step_size * math.sin(theta)
    return target_x, target_y, theta


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class GadenRLNode(Node):
    """ROS2 node that runs a pretrained PPO agent inside GADEN."""

    def __init__(self):
        super().__init__('gaden_rl_node')

        self._declare_parameters()
        self._load_parameters()
        self._init_state()
        self._load_occupancy_map()
        self._load_agent()
        self._setup_escape()
        self._init_ros_interfaces()

        self.get_logger().info(f'GadenRLNode ready. Checkpoint: {self._checkpoint_path}')
        self.get_logger().info(
            f'Architecture: {self._arch} | Device: {self._device} | Max steps: {self._max_steps}'
        )
        self.get_logger().info(f'Map normalisation: {self._map_width:.1f} x {self._map_height:.1f} m')
        self.get_logger().info(f'True source: ({self._true_source_x:.2f}, {self._true_source_y:.2f})')

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        # Checkpoint (required — no sensible default)
        self.declare_parameter('checkpoint', '')
        # Architecture must match the training run: mlp | modular | dual
        self.declare_parameter('arch', 'mlp')
        # PyTorch device
        self.declare_parameter('device', 'cpu')
        # True source position (for success detection and logging)
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 4.5)
        # Episode length cap (override cfg.MAX_STEPS if needed)
        self.declare_parameter('max_steps', cfg.MAX_STEPS)
        # Number of episodes to run before the process exits (batch eval)
        self.declare_parameter('num_episodes', 1)
        # Nav2 tolerance
        self.declare_parameter('xy_goal_tolerance', 0.3)
        # Occupancy service (used to auto-derive map dimensions)
        self.declare_parameter('occupancy_service', '/gaden_environment/occupancyMap3D')
        self.declare_parameter('occupancy_z_level', 5)
        self.declare_parameter('occupancy_timeout', 60.0)
        # Path to GADEN wind CSV (e.g. .../wind_simulations/1ms/wind_at_cell_centers_0.csv)
        self.declare_parameter('wind_file', '')
        # Delay between teleport steps (seconds)
        self.declare_parameter('step_delay', 0.5)
        # Start position (empty string = use GADEN's default spawn)
        self.declare_parameter('start_x', -999.0)
        self.declare_parameter('start_y', -999.0)
        # Logging and visualization
        self.declare_parameter('step_log_every', 1)
        self.declare_parameter('publish_action_marker', True)
        self.declare_parameter('action_marker_topic', '/model_free_gsl/target')
        # Debug image dump (only for arch=spatial — flat obs isn't 2D)
        self.declare_parameter('debug_dump_dir', '')
        self.declare_parameter('debug_dump_every', 20)
        # Lidar frame the policy was trained against:
        #   "world"   → current branch (ray 0 = world +x), needs roll at deploy
        #   "heading" → feature/rl* training (ray 0 = robot forward), no roll
        self.declare_parameter('lidar_frame', 'world')
        # Motion mode: False = teleport between steps (default, unchanged);
        # True = drive to each step's target via Nav2 (stop-go). Driving gives the
        # robot a real heading, handled by the obs builder's drive_mode roll.
        self.declare_parameter('use_nav2', False)
        # Nav2 arrival tolerance (m): the robot stops within this of each target.
        self.declare_parameter('nav_goal_tolerance', 0.1)
        # Per-step drive timeout (s, sim time): if Nav2 hasn't reached the target
        # within this, cancel the goal and let the policy re-predict — avoids the
        # ~150 s recovery-loop stalls when DWB gets wedged. 0 = disabled.
        self.declare_parameter('drive_timeout', 8.0)

    def _load_parameters(self):
        self._checkpoint_path: str = self.get_parameter('checkpoint').value
        self._arch: str = self.get_parameter('arch').value
        _dev_str: str = self.get_parameter('device').value
        self._device = torch.device(
            _dev_str if torch.cuda.is_available() or _dev_str == 'cpu' else 'cpu'
        )
        self._true_source_x: float = self.get_parameter('true_source_x').value
        self._true_source_y: float = self.get_parameter('true_source_y').value
        self._max_steps: int = self.get_parameter('max_steps').value
        self._num_episodes: int = int(self.get_parameter('num_episodes').value)
        self._xy_tolerance: float = self.get_parameter('xy_goal_tolerance').value
        self._occ_service: str = self.get_parameter('occupancy_service').value
        self._occ_z: int = self.get_parameter('occupancy_z_level').value
        self._occ_timeout: float = float(self.get_parameter('occupancy_timeout').value)
        self._wind_file: str = self.get_parameter('wind_file').value
        self._step_delay: float = float(self.get_parameter('step_delay').value)
        self._start_x: float = self.get_parameter('start_x').value
        self._start_y: float = self.get_parameter('start_y').value
        self._step_log_every: int = int(self.get_parameter('step_log_every').value)
        self._publish_action_marker: bool = bool(self.get_parameter('publish_action_marker').value)
        self._action_marker_topic: str = self.get_parameter('action_marker_topic').value
        self._debug_dump_dir: str = self.get_parameter('debug_dump_dir').value
        self._debug_dump_every: int = int(self.get_parameter('debug_dump_every').value)
        self._lidar_frame: str = self.get_parameter('lidar_frame').value
        self._use_nav2: bool = bool(self.get_parameter('use_nav2').value)
        self._nav_goal_tolerance: float = float(self.get_parameter('nav_goal_tolerance').value)
        self._drive_timeout: float = float(self.get_parameter('drive_timeout').value)

    def _init_state(self):
        self._robot_x: Optional[float] = None
        self._robot_y: Optional[float] = None
        self._current_theta: Optional[float] = None

        self._episode: int = 0
        self._step_in_episode: int = 0

        self._is_moving: bool = False
        # Per-step drive-timeout bookkeeping (drive mode only)
        self._drive_goal_time = None        # rclpy Time when current goal was sent
        self._drive_canceling: bool = False  # cancel issued, awaiting completion
        self._search_complete: bool = False
        self._start_teleport_done: bool = False
        self._last_step_time_ns: int = 0
        self._latest_gas_raw: Optional[float] = None
        self._latest_wind_speed: Optional[float] = None
        self._latest_wind_dir: Optional[float] = None
        self._latest_lidar_min: Optional[float] = None
        # Fresh-scan gate: make _take_step wait for a laser scan whose stamp
        # post-dates the most recent teleport, so the spatial builder never
        # rasterises live lidar from the PRE-teleport position into cells
        # that belong to the post-teleport frame (splatter bug).
        self._latest_scan_stamp_ns: int = 0
        self._teleport_wait_stamp_ns: Optional[int] = None

        self._obs_builder: Optional[ObservationBuilder] = None

        # Nav2 driving (use_nav2=True): Navigator drives the robot to each target
        # and clears _is_moving on completion (stop-go). None in teleport mode.
        self._navigator = None
        # Actual travelled path length (m), integrated from pose deltas; the
        # d<1.0 guard skips teleport jumps so it is meaningful in both modes.
        self._last_pose_xy: Optional[tuple] = None
        self._travel_distance_m: float = 0.0

        # Wind z-snap (local-wind policies only). The anemometer reads wind at a
        # fixed TF height z=0.5, but the CFD wind field has irregular z-banding:
        # at many (x,y) cells z=0.5 lands in an empty layer and returns 0 m/s.
        # When active, we query /wind_value at a few z-offsets in ONE service
        # call and use the first nonzero reading. Runs under a MultiThreaded
        # executor + reentrant group so the per-step call cannot deadlock.
        # Default OFF: the GADEN preprocessing vertical-fill makes the served
        # grid dense at z=0.5, so the anemometer topic already reads real wind
        # and the per-step service z-snap is redundant. Set OSL_WIND_ZSNAP=1 to
        # re-enable it (e.g. for maps preprocessed before the fill).
        self._wind_zsnap: bool = os.environ.get("OSL_WIND_ZSNAP", "0") == "1"
        self._wind_z_base: float = 0.5
        self._wind_z_offsets = (0.0, -0.05, 0.30, -0.30, 0.60, 0.15, -0.15)
        self._wind_cli = None
        self._wind_cbgroup = None

    def _load_occupancy_map(self):
        """Fetch the 2D occupancy grid from GADEN and derive map dimensions.

        Also stored for collision lookups in `_take_step`.
        """
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
                "No checkpoint path provided.  Set the 'checkpoint' ROS2 parameter:\n"
                "  ros2 run reinforcement_learning gaden_rl_node \\\n"
                "      --ros-args -p checkpoint:=/path/to/agent_STEP.pt"
            )
        if not os.path.isfile(self._checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self._checkpoint_path}"
            )

        run_cfg = _load_run_config(self._checkpoint_path)
        self._agent = _load_agent(self._checkpoint_path, self._arch, self._device, run_cfg)
        self.get_logger().info(
            f'Run config: LIDAR_NUM_RAYS={run_cfg.get("LIDAR_NUM_RAYS", cfg.LIDAR_NUM_RAYS)} '
            f'GAS_HISTORY_LENGTH={run_cfg.get("GAS_HISTORY_LENGTH", cfg.GAS_HISTORY_LENGTH)} '
            f'STATE_DIM={run_cfg.get("STATE_DIM", cfg.STATE_DIM)} '
            f'({"from config.json" if run_cfg else "cfg defaults — config.json not found"})'
        )
        n_params = sum(p.numel() for p in self._agent.parameters())
        self.get_logger().info(
            f'Loaded checkpoint: {self._checkpoint_path} '
            f'({n_params:,} parameters, arch={self._arch})'
        )

        if self._arch == 'spatial':
            self._obs_builder = SpatialObsBuilder(
                self._map_width, self._map_height,
                origin_x=self._occ_map.origin_x,
                origin_y=self._occ_map.origin_y,
            )
        else:
            self._obs_builder = ObservationBuilder(
                self._map_width, self._map_height, lidar_frame=self._lidar_frame,
                drive_mode=self._use_nav2,
            )
            self.get_logger().info(f'Lidar frame: {self._lidar_frame}')

        if self._wind_file:
            self._obs_builder.load_wind_from_file(self._wind_file)
            self.get_logger().info(
                f'Wind from file: speed={self._obs_builder._locked_wind_speed:.3f} m/s '
                f'dir={np.degrees(self._obs_builder._locked_wind_dir):.1f} deg'
            )
        else:
            self.get_logger().warn('No wind_file provided — wind obs will be zero until locked')

        if self._debug_dump_dir and self._arch == 'spatial':
            os.makedirs(self._debug_dump_dir, exist_ok=True)
            self.get_logger().info(
                f'Debug dumps every {self._debug_dump_every} steps → {self._debug_dump_dir}'
            )

    def _setup_escape(self):
        """Optionally enable the SLAM-based circling-escape (OSL_ESCAPE=1).

        Builds an online occupancy map from LiDAR and, when the policy is stuck
        circling, drives to the largest unexplored frontier. Default OFF so the
        baseline behaviour is unchanged unless explicitly enabled.
        """
        self._escape = None
        if os.environ.get('OSL_ESCAPE', '0') != '1':
            return
        try:
            self._escape = CirclingEscape(
                self._occ_map,
                robot_radius=cfg.ROBOT_RADIUS,
                logger=self.get_logger(),
                win=int(os.environ.get('OSL_ESCAPE_WIN', '25')),
                ratio=float(os.environ.get('OSL_ESCAPE_RATIO', '0.2')),
                streak=int(os.environ.get('OSL_ESCAPE_STREAK', '35')),
                cooldown=int(os.environ.get('OSL_ESCAPE_COOLDOWN', '40')),
                min_dist=float(os.environ.get('OSL_ESCAPE_MINDIST', '3.0')),
                cov_res=float(os.environ.get('OSL_ESCAPE_COV_RES', '0.5')),
                cov_win=int(os.environ.get('OSL_ESCAPE_COV_WIN', '40')),
                cov_k=int(os.environ.get('OSL_ESCAPE_COV_K', '3')),
                cov_streak=int(os.environ.get('OSL_ESCAPE_COV_STREAK', '25')),
            )
            self.get_logger().info(
                'SLAM stuck-escape ENABLED — trigger on EITHER signal: '
                f'circling (eff-streak>={self._escape.streak_thr}, win={self._escape.win}, '
                f'ratio={self._escape.ratio}) OR loitering (coverage-stagnation>='
                f'{self._escape.cov_streak_thr}: <{self._escape.cov_k} new '
                f'{self._escape.cov_res}m-cells in {self._escape.cov_win} steps); '
                f'cooldown={self._escape.cooldown}, min_dist={self._escape.min_dist}m'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to init circling-escape: {exc} — disabled.')
            self._escape = None

    def _init_ros_interfaces(self):
        """Set up subscribers and the Navigator (reused from efe_igdm)."""
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

        if self._wind_zsnap:
            # Reentrant group so the synchronous per-step service call can be
            # serviced by another executor thread instead of deadlocking the
            # one running _take_step (single-threaded executor was the bug).
            self._wind_cbgroup = ReentrantCallbackGroup()
            self._wind_cli = self.create_client(
                WindPosition, '/wind_value', callback_group=self._wind_cbgroup)
            if self._wind_cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().info(
                    'Wind z-snap enabled: querying /wind_value at z-offsets '
                    f'{self._wind_z_offsets} around z={self._wind_z_base}')
            else:
                self.get_logger().warn(
                    'Wind z-snap requested but /wind_value unavailable — '
                    'falling back to anemometer topic only.')
                self._wind_cli = None

        # --- Motion mode setup ---
        if self._use_nav2:
            # Pre-flight guard: Navigator.__init__ waits on the Nav2 action server
            # with NO timeout, so confirm it is up here (60 s) and fail fast
            # instead of hanging forever if the Nav2 stack didn't come up.
            _pre = ActionClient(self, NavigateToPose, '/PioneerP3DX/navigate_to_pose')
            if not _pre.wait_for_server(timeout_sec=60.0):
                self.get_logger().error(
                    'use_nav2=True but Nav2 action server '
                    '/PioneerP3DX/navigate_to_pose not available after 60 s — aborting.')
                os._exit(2)
            _pre.destroy()
            self._navigator = Navigator(self, on_complete_callback=self._on_nav_complete)
            self.get_logger().info(
                f'Motion mode: DRIVE via Nav2 (stop-go, '
                f'tolerance={self._nav_goal_tolerance:.2f} m)')
        else:
            self.get_logger().info('Motion mode: TELEPORT (instant)')


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

        # Integrate actual travelled distance (skip teleport jumps with d<1.0).
        if self._last_pose_xy is not None:
            d = math.hypot(x - self._last_pose_xy[0], y - self._last_pose_xy[1])
            if d < 1.0:
                self._travel_distance_m += d
        self._last_pose_xy = (x, y)

        self._robot_x = x
        self._robot_y = y
        if self._obs_builder is not None:
            self._obs_builder.robot_x = x
            self._obs_builder.robot_y = y
            self._obs_builder.robot_theta = self._current_theta

        if self._obs_builder is None or not self._obs_builder.ready:
            return

        # Stop-go drive gate: while Nav2 is driving the robot to the current
        # target, don't start another step. _on_nav_complete clears _is_moving
        # when the goal finishes (success/abort/reject), and the next pose
        # callback then proceeds. No-op in teleport mode (_use_nav2 False).
        if self._use_nav2 and self._is_moving:
            # Per-step drive timeout: if DWB gets wedged (recovery-loop stall),
            # cancel the goal so the policy re-predicts in ~drive_timeout s instead
            # of waiting ~150 s for Nav2's behavior tree to give up.
            if (self._drive_timeout > 0.0 and self._drive_goal_time is not None
                    and not self._drive_canceling):
                elapsed = (self.get_clock().now()
                           - self._drive_goal_time).nanoseconds * 1e-9
                if elapsed > self._drive_timeout:
                    self._drive_canceling = True
                    self.get_logger().warn(
                        f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                        f'DRIVE TIMEOUT — goal not reached in {elapsed:.1f}s '
                        f'(> {self._drive_timeout:.1f}s); canceling, re-predicting.')
                    if self._navigator is not None:
                        self._navigator.cancel_current_goal()
            return

        # Initial: teleport to start position once
        if not self._start_teleport_done:
            self._start_teleport_done = True
            if self._start_x > -998.0 and self._start_y > -998.0:
                self.get_logger().info(
                    f'Teleporting to start position ({self._start_x:.2f}, {self._start_y:.2f})'
                )
                self._teleport_to(self._start_x, self._start_y)
                self._last_step_time_ns = self.get_clock().now().nanoseconds
                return

        # Fresh-scan gate: wait for a laser scan whose header.stamp is
        # strictly newer than the last teleport. Guarantees the spatial
        # observation builder rasterises only post-teleport lidar data.
        if self._teleport_wait_stamp_ns is not None and \
                self._latest_scan_stamp_ns <= self._teleport_wait_stamp_ns:
            return

        # Gate step rate by elapsed sim time since last step
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
        # Record scan stamp for the fresh-scan gate regardless of obs_builder.
        st = msg.header.stamp
        self._latest_scan_stamp_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
        if self._obs_builder is None:
            return
        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        self._latest_lidar_min = min(finite_ranges) if finite_ranges else None
        self._obs_builder.update_lidar(msg)

        # Fold the same scan into the online SLAM map for the circling-escape.
        if self._escape is not None:
            self._escape.update_scan(
                msg, self._robot_x, self._robot_y, self._current_theta)

    def _wind_callback(self, msg: Anemometer):
        # Mean wind is loaded from the CFD file at startup (used by mean-wind
        # policies). For local point-wind policies (OSL_LOCAL_WIND_OBS=1) the
        # obs builder uses these LIVE anemometer readings at the robot cell.
        #
        # BUG WORKAROUND: simulated_anemometer publishes wind_speed = u*u + v*v
        # (squared magnitude — it never takes the sqrt; fake_anemometer.cpp:138).
        # Training/Python eval feed the TRUE magnitude sqrt(u^2+v^2), so without
        # correcting here the local-wind policy sees systematically crushed wind
        # (e.g. 0.7 m/s -> 0.49, 0.1 -> 0.01), which nullifies the local-wind
        # signal — observed as many_rooms collapsing 0% on ROS2 vs 90% in Python.
        speed = float(np.sqrt(max(0.0, msg.wind_speed)))
        self._latest_wind_speed = speed
        self._latest_wind_dir = float(msg.wind_direction)
        if self._obs_builder is not None:
            self._obs_builder.update_live_wind(speed, msg.wind_direction)

    def _query_wind_zsnap(self, x: float, y: float):
        """Query /wind_value at several z-offsets in ONE call; return the first
        nonzero (u,v) as (speed, direction_rad), or None.

        Works around the CFD z-banding: the anemometer's fixed z=0.5 often lands
        in an empty layer (0 m/s). Probing a few heights recovers the real wind
        for that (x,y) column. Safe to call from _take_step because the client
        sits on a reentrant callback group under a MultiThreadedExecutor.
        """
        if self._wind_cli is None:
            return None
        req = WindPosition.Request()
        n = len(self._wind_z_offsets)
        req.x = [float(x)] * n
        req.y = [float(y)] * n
        req.z = [self._wind_z_base + dz for dz in self._wind_z_offsets]
        fut = self._wind_cli.call_async(req)
        # Bounded wait; the reentrant group lets another executor thread fill
        # the future. time.sleep yields the GIL so that thread can run.
        end = self.get_clock().now().nanoseconds + int(0.5e9)
        while not fut.done() and self.get_clock().now().nanoseconds < end:
            time.sleep(0.001)
        if not fut.done() or fut.result() is None:
            return None
        res = fut.result()
        for u, v in zip(res.u, res.v):
            sp = math.hypot(u, v)
            if sp > 1e-4:
                return sp, math.atan2(v, u)
        return None

    def _teleport_to(self, x: float, y: float):
        msg = PoseWCS()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance = [0.0] * 36
        self._teleport_pub.publish(msg)
        # Re-arm fresh-scan gate: next _take_step must wait for a scan whose
        # sim-time stamp is strictly newer than this.
        self._teleport_wait_stamp_ns = self._latest_scan_stamp_ns

    def _on_nav_complete(self):
        """Nav2 goal finished (success / abort / reject / cancel). Clear the moving
        flag so the next ground_truth pose callback takes the next step (stop-go)."""
        self._is_moving = False
        self._drive_goal_time = None
        self._drive_canceling = False

    # ------------------------------------------------------------------
    # Core control loop
    # ------------------------------------------------------------------

    def _take_step(self):
        """Run one policy step: build obs → infer action → send goal."""
        if self._search_complete:
            return
        if self._robot_x is None or self._robot_y is None:
            return
        if self._obs_builder is None or not self._obs_builder.ready:
            return

        # --- Episode timeout check ---
        if self._step_in_episode >= self._max_steps:
            self.get_logger().warn(
                f'[Episode {self._episode}] Max steps ({self._max_steps}) reached. '
                f'Episode failed.'
            )
            self._end_episode(success=False)
            return

        # --- Check if source has been found ---
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

        # --- SLAM circling detection + frontier-exploration escape (opt-in) ---
        # When the policy is stuck circling (sustained low displacement-efficiency),
        # drive to the largest unexplored frontier of the online SLAM map instead
        # of issuing another policy step, then resume the policy from there.
        if self._escape is not None:
            self._escape.record_step(self._robot_x, self._robot_y)
            tgt = self._escape.maybe_escape_target(
                self._robot_x, self._robot_y, self._step_in_episode)
            if tgt is not None:
                tx, ty, fsize, fdist = tgt
                self.get_logger().warn(
                    f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                    f'STUCK ({self._escape.stuck_reason}) → frontier escape '
                    f'#{self._escape.n_escapes} to ({tx:.2f},{ty:.2f}) '
                    f'[frontier {fsize} cells, {fdist:.1f}m away, '
                    f'mapped={self._escape.mapped_fraction()*100:.0f}%]')
                if self._use_nav2 and self._navigator is not None:
                    self._is_moving = True
                    self._drive_canceling = False
                    self._drive_goal_time = self.get_clock().now()
                    self._navigator.send_goal(
                        tx, ty, use_orientation=False,
                        tolerance=self._nav_goal_tolerance)
                else:
                    self._teleport_to(tx, ty)
                self._step_in_episode += 1
                return

        # --- Per-step state update ---
        # Flat obs: step 0 is seeded by the first gas callback, so skip.
        # Spatial obs: step 0 must seed the world grids from the first scan,
        # so always call record_step().

        # --- Wind z-snap (local-wind policies) ---
        # Override the anemometer feed with the first nonzero reading found
        # across a few z-offsets at the robot column. Must run before build()
        # so the corrected wind lands in this step's observation.
        if self._wind_cli is not None:
            snap = self._query_wind_zsnap(self._robot_x, self._robot_y)
            if snap is not None:
                sp, dir_rad = snap
                self._latest_wind_speed = sp
                self._latest_wind_dir = dir_rad
                self._obs_builder.update_live_wind(sp, dir_rad)

        if self._arch == 'spatial':
            self._obs_builder.record_step()
        elif self._step_in_episode > 0:
            self._obs_builder.record_step()

        # --- Build observation ---
        built = self._obs_builder.build()
        if built is None:
            return

        # --- Optional debug dump (spatial only) ---
        if self._arch == 'spatial' and self._debug_dump_dir \
                and self._debug_dump_every > 0 \
                and (self._step_in_episode % self._debug_dump_every == 0):
            spatial_np, wind_np = built
            self._dump_debug_image(spatial_np, wind_np)

        # --- Policy inference ---
        if self._arch == 'spatial':
            spatial_np, wind_np = built
            spatial_t = torch.tensor(
                spatial_np, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            wind_t = torch.tensor(
                wind_np, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = self._agent.get_action_and_value(
                    spatial_t, wind_t
                )
            action_np = action.cpu().numpy().flatten()  # (2,)
        else:
            obs = built
            # Sanity-check observation range (flat obs is normalised to [0,1])
            if not (np.all(obs >= 0.0) and np.all(obs <= 1.0)):
                self.get_logger().warn(
                    f'Observation out of [0,1] range: '
                    f'min={obs.min():.3f} max={obs.max():.3f} — clipping.'
                )
                obs = np.clip(obs, 0.0, 1.0)
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            with torch.no_grad():
                if self._arch == 'dual':
                    encoded = self._agent._encode_shared(obs_t)
                    action = self._agent._actor_dist(encoded).mean
                else:
                    action, _, _, _ = self._agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy().flatten()  # (1,) or (2,)

        # --- Optional debug dump (dual/flat arch): full obs vector + raw action
        # per step, so an offline proxy can be checked against the EXACT inputs the
        # deployed policy saw (open-loop action comparison). Writes one .npz/step. ---
        if self._arch != 'spatial' and self._debug_dump_dir \
                and self._debug_dump_every > 0 \
                and (self._step_in_episode % self._debug_dump_every == 0):
            os.makedirs(self._debug_dump_dir, exist_ok=True)
            np.savez(
                os.path.join(self._debug_dump_dir,
                             f'obs_{self._step_in_episode:04d}.npz'),
                step=self._step_in_episode,
                obs=obs.astype(np.float32),
                action=action_np.astype(np.float32),
                robot_xy=np.array([self._robot_x, self._robot_y], dtype=np.float32),
            )

        # --- Compute target position ---
        target_x, target_y, theta = _action_to_target(
            self._robot_x, self._robot_y,
            action_np, self._arch, cfg.STEP_SIZE
        )

        # Heading-frame agents need ray 0 of next obs to align with the direction
        # they just commanded — track it here so the next obs is built correctly.
        # (Physical yaw is always 0 due to identity-quaternion teleport.)
        if self._obs_builder is not None and hasattr(self._obs_builder, 'last_action_theta'):
            self._obs_builder.last_action_theta = float(theta)

        if self._step_log_every <= 1 or (self._step_in_episode % self._step_log_every == 0):
            action_text = np.array2string(action_np, precision=3, separator=',')
            gas_text = f'{self._latest_gas_raw:.3f}' if self._latest_gas_raw is not None else 'n/a'
            wind_s_text = f'{self._latest_wind_speed:.2f}' if self._latest_wind_speed is not None else 'n/a'
            wind_d_text = f'{math.degrees(self._latest_wind_dir):.1f}' if self._latest_wind_dir is not None else 'n/a'
            lidar_text = f'{self._latest_lidar_min:.2f}' if self._latest_lidar_min is not None else 'n/a'
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'Pos ({self._robot_x:.2f},{self._robot_y:.2f}) '
                f'Action {action_text} θ={math.degrees(theta):.1f}deg → '
                f'Target ({target_x:.2f},{target_y:.2f}) | '
                f'd2src={dist_to_source:.2f}m gas={gas_text} wind=({wind_s_text}m/s,{wind_d_text}deg) '
                f'lidar_min={lidar_text}m'
            )

        # --- Collision check + clamp to free cell ---
        # If basic_sim would reject this teleport (target inside a wall), the
        # robot wouldn't move at all. Walk back along the ray until we land in
        # a free cell, exactly like the training collision logic.
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

        # --- Move robot: drive via Nav2 (stop-go) or teleport ---
        if self._use_nav2 and self._navigator is not None:
            # Drive to the (collision-clamped) target. yaw=theta + use_orientation
            # makes the robot arrive facing the commanded direction so the next
            # step's heading-frame obs aligns; the drive_mode roll corrects any
            # residual yaw error. _is_moving gates further steps until arrival.
            self._is_moving = True
            self._drive_canceling = False
            self._drive_goal_time = self.get_clock().now()  # for per-step timeout
            self._navigator.send_goal(
                target_x, target_y, yaw=theta,
                use_orientation=True, tolerance=self._nav_goal_tolerance,
            )
        else:
            # --- Teleport robot ---
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
        step = cfg.STEP_SIZE - 0.05
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
        """Log episode result and either shutdown or re-teleport for the next one."""
        result = 'SUCCESS' if success else 'FAILURE'
        self.get_logger().info(
            f'[Episode {self._episode}] {result} — '
            f'{self._step_in_episode} steps'
        )
        self._episode += 1

        # Stop after the configured number of episodes. rclpy.shutdown() from
        # inside a subscription callback does not reliably make rclpy.spin()
        # return — spin can hang until an external kill. Force-exit: DDS shm
        # is wiped between runs by the batch runner, so the abrupt exit is
        # safe for eval use.
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

        # Prepare next episode: clear per-episode state and teleport back to
        # the configured start pose. Resetting ``_start_teleport_done`` makes
        # the pose callback re-issue the teleport on its next firing, which
        # guarantees the observation builder sees the new pose before any
        # step is taken.
        self._step_in_episode = 0
        self._is_moving = False  # clear any lingering drive flag before next episode
        self._obs_builder.reset()
        self._obs_builder.robot_x = self._robot_x
        self._obs_builder.robot_y = self._robot_y
        # Wind values persist across episodes (no episode boundary in GADEN wind)

        if self._start_x > -998.0 and self._start_y > -998.0:
            self.get_logger().info(
                f'Starting episode {self._episode} — teleporting to '
                f'({self._start_x:.2f}, {self._start_y:.2f})'
            )
            self._start_teleport_done = False

        self._last_step_time_ns = self.get_clock().now().nanoseconds

    def _dump_debug_image(self, spatial_np: np.ndarray, wind_np: np.ndarray):
        """Dump a 2×3 PNG: 4 spatial channels (4-ch old arch) + GT world grid.

        Channels: [occupancy, gas, recency, det_count] where occupancy ∈ {-1,0,1}.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            self.get_logger().warn(f'matplotlib unavailable, skipping dump: {e}')
            return

        names = ['occupancy', 'gas', 'recency', 'det_count']
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for i, name in enumerate(names):
            ax = axes.flat[i]
            data = spatial_np[i]
            vmin, vmax = (-1.0, 1.0) if name in ('occupancy', 'gas') else (0.0, 1.0)
            im = ax.imshow(data, origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax)
            ax.set_title(name)
            ax.axvline(49, color='r', lw=0.3, alpha=0.6)
            ax.axhline(49, color='r', lw=0.3, alpha=0.6)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # GT world grid (the source of reveal for the old 4-ch builder)
        ax = axes.flat[4]
        grid = self._occ_map.grid
        disp = np.full(grid.shape, 0.5, dtype=np.float32)
        disp[grid == 0] = 0.0
        disp[grid == 1] = 1.0
        ax.imshow(disp, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        try:
            ox = float(getattr(self._occ_map, 'origin_x', 0.0) or 0.0)
            oy = float(getattr(self._occ_map, 'origin_y', 0.0) or 0.0)
            res = float(self._occ_map.resolution)
            gx = (self._robot_x - ox) / res
            gy = (self._robot_y - oy) / res
            ax.plot(gx, gy, 'ro', markersize=6)
        except Exception:
            pass
        ax.set_title(f'GT grid — walls={int((grid != 0).sum())}')
        ax.axis('off')

        axes.flat[5].axis('off')

        fig.suptitle(
            f'Ep {self._episode} Step {self._step_in_episode:04d}   '
            f'pos=({self._robot_x:.2f},{self._robot_y:.2f})   '
            f'wind=[{wind_np[0]:.2f}, {wind_np[1]:.2f}]',
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
        """Publish current planned next action as an RViz arrow marker."""
        if self._robot_x is None or self._robot_y is None:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gaden_rl_next_action'
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
        self._next_action_pub.publish(marker)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GadenRLNode()
        # MultiThreadedExecutor: lets the /wind_value z-snap call (reentrant
        # group) be serviced by a second thread instead of deadlocking the
        # thread running _take_step. Harmless for non-local-wind policies.
        executor = MultiThreadedExecutor(num_threads=2)
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
