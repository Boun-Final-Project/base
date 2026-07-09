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
import time
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
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path as NavPath, OccupancyGrid
from visualization_msgs.msg import Marker

# RL package imports: add src/base/ to sys.path so both the installed entry
# point and direct execution can find the reinforcement_learning package.
_SRC_BASE = '/home/efe/ros2_ws/src/base'
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

def _load_agent(checkpoint_path: str, arch: str, device: torch.device):
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

    Returns
    -------
    agent : nn.Module
        Loaded model in eval mode.
    """
    if arch == 'spatial':
        agent = ActorCriticSpatial()
    elif arch == 'dual':
        agent = ActorCriticDualBackbone(obs_dim=cfg.STATE_DIM)
    elif arch == 'modular':
        agent = ActorCriticModular(obs_dim=cfg.STATE_DIM)
    else:
        agent = ActorCritic(obs_dim=cfg.STATE_DIM)

    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt['model_state_dict'])
    agent.to(device)
    agent.eval()
    return agent


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
        self._slam_enabled: bool = False   # gate SLAM updates to post-teleport (set in _take_step)
        self._latest_plan = []             # latest Nav2 global plan (debug)
        self._capture_plan_for = None      # step# to save the next plan for (debug)
        self._escape_last_target = (0.0, 0.0)
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

        # Deterministic cmd_vel drive (OSL_DET_DRIVE=1): bypass Nav2/DWB and drive
        # each target via a proportional rotate-then-go controller publishing
        # /PioneerP3DX/cmd_vel directly. DWB skims walls (robot modelled 5 cm) and
        # wedges on near-wall 0.5 m steps; this drives a straight line and lands ON
        # the commanded point. Same mechanism a sibling RL node uses successfully.
        self._det_drive: bool = os.environ.get('OSL_DET_DRIVE', '0') == '1'
        self._det_linear: float = float(os.environ.get('OSL_DET_LINEAR', '0.3'))
        self._det_gain: float = float(os.environ.get('OSL_DET_GAIN', '2.0'))
        self._det_arrive_tol: float = float(os.environ.get('OSL_DET_ARRIVE_TOL', '0.15'))
        self._det_timeout: float = float(os.environ.get('OSL_DET_TIMEOUT', '12.0'))
        self._cmd_vel_pub = None
        self._waypoint_x: Optional[float] = None
        self._waypoint_y: Optional[float] = None
        self._waypoint_start_ns: int = 0
        # HYBRID: det-drive (cmd_vel) drives ONLY escape hops; policy steps stay on
        # Nav2/DWB (which avoids obstacles — det-drive has none and collides in tight
        # maps). _cmdvel_active is True only while a cmd_vel escape hop is executing,
        # so _cmd_vel_tick never publishes during a Nav2 policy drive.
        self._cmdvel_active: bool = False

        # DirectDrive (OSL_DIRECT_DRIVE=1): a THIRD per-step motion mode. Physically
        # drives the robot to each clamped 0.5 m policy goal via the same closed-loop
        # cmd_vel controller as det-drive, but adds a per-tick lidar+occupancy safety
        # gate so the center provably never enters ROBOT_RADIUS of a wall. Replaces
        # Nav2-DRIVE (whose global planner hugs walls / DWB tracking error trips the
        # 0.28 m collision boundary) while staying physical (unlike teleport) and
        # faithful (lands where teleport would, so the policy obs is unchanged).
        # Precedence: DirectDrive > Nav2 (use_nav2 param) > teleport. Mutually
        # exclusive with OSL_DET_DRIVE and use_nav2 — leave both unset/False.
        self._direct_drive: bool = os.environ.get('OSL_DIRECT_DRIVE', '0') == '1'
        self._dd_linear: float = float(os.environ.get('OSL_DD_LINEAR', '0.3'))
        self._dd_gain: float = float(os.environ.get('OSL_DD_GAIN', '2.0'))
        self._dd_wmax: float = float(os.environ.get('OSL_DD_WMAX', '1.5'))
        self._dd_arrive_tol: float = float(os.environ.get('OSL_DD_ARRIVE_TOL', '0.10'))
        self._dd_timeout: float = float(os.environ.get('OSL_DD_TIMEOUT', '12.0'))
        self._dd_lookahead: float = float(os.environ.get('OSL_DD_LOOKAHEAD', '0.20'))
        self._dd_lidar_margin: float = float(os.environ.get('OSL_DD_LIDAR_MARGIN', '0.33'))
        self._dd_slow_r: float = float(os.environ.get('OSL_DD_SLOW_R', '0.30'))
        self._dd_cone_half: float = float(os.environ.get('OSL_DD_CONE_HALF', '0.79'))
        self._dd_stuck_s: float = float(os.environ.get('OSL_DD_STUCK_S', '2.0'))
        # DirectDrive runtime state (separate _dd_active flag so the validated
        # det-drive HYBRID escape path stays byte-identical):
        self._dd_active: bool = False
        self._dd_best_dist: float = float('inf')
        self._dd_progress_ns: int = 0
        self._dd_cone_min_sensor: Optional[float] = None

        # CENTERLINE-PURSUIT (OSL_PURSUIT=1, requires OSL_DIRECT_DRIVE=1 + OSL_ROUTE=1):
        # follow a static-map, radius-consistent, centerline-biased A* path to each
        # (unchanged) clamped policy goal with a SMOOTH lookahead controller whose speed
        # tapers CONTINUOUSLY with forward clearance (never a binary freeze => no wedge),
        # and whose stall-recovery pulls toward higher EDT clearance + the GOAL (never the
        # raw heading => no ceiling-slide/wander). Rebuilds Nav2's plan+follow honestly.
        import threading as _threading
        self._pursuit: bool = os.environ.get('OSL_PURSUIT', '0') == '1'
        self._pp_vmax: float = float(os.environ.get('OSL_PP_VMAX', '0.9'))
        self._pp_gain: float = float(os.environ.get('OSL_PP_GAIN', '1.5'))
        self._pp_wmax: float = float(os.environ.get('OSL_PP_WMAX', '2.0'))
        self._pp_lookahead: float = float(os.environ.get('OSL_PP_LOOKAHEAD', '0.30'))
        self._pp_advance: float = float(os.environ.get('OSL_PP_ADVANCE', '0.15'))
        self._pp_arrive_tol: float = float(os.environ.get('OSL_PP_ARRIVE_TOL', '0.10'))
        self._pp_slow_r: float = float(os.environ.get('OSL_PP_SLOW_R', '0.30'))
        self._pp_he_cut: float = float(os.environ.get('OSL_PP_HE_CUT', '1.0'))
        self._pp_spin: float = float(os.environ.get('OSL_PP_SPIN', '1.2'))
        self._pp_cone_stop: float = float(os.environ.get('OSL_PP_CONE_STOP', '0.18'))
        self._pp_cone_free: float = float(os.environ.get('OSL_PP_CONE_FREE', '0.40'))
        self._pp_stuck_s: float = float(os.environ.get('OSL_PP_STUCK_S', '2.5'))
        self._pp_timeout: float = float(os.environ.get('OSL_PP_TIMEOUT', '8.0'))
        self._pp_stall_ticks: int = int(os.environ.get('OSL_PP_STALL_TICKS', '15'))
        self._pp_clear_w: float = float(os.environ.get('OSL_PP_CLEAR_W', '2.0'))
        self._pp_active: bool = False
        self._pp_path = None
        self._pp_idx: int = 0
        self._pp_goal = None
        self._pp_best: float = float('inf')
        self._pp_progress_ns: int = 0
        self._pp_stall_n: int = 0
        self._pp_recover_ns: int = 0
        self._pp_lock = _threading.Lock()

        # ClearanceRoute (OSL_ROUTE=1, requires OSL_DIRECT_DRIVE=1): before driving,
        # SNAP each clamped goal up a cached clearance (distance-transform) field to the
        # nearest corridor-center cell. _clamp_to_free's straight walk-back parks the
        # 0.28 body ON the wall-grazing is_valid boundary; from there ~half the policy's
        # 0.5 m headings point into the wall -> clamp stay-put -> circle (measured 53%).
        # Snapping the landing pose to the corridor centerline (like Nav2's inflation
        # kept it central, but with no Nav2 and no wedging) collapses that cascade.
        self._route_on: bool = os.environ.get('OSL_ROUTE', '0') == '1'
        self._route_snap_r: float = float(os.environ.get('OSL_ROUTE_SNAP_R', '0.4'))   # snap search disk (m)
        self._route_snap_cap: float = float(os.environ.get('OSL_ROUTE_SNAP_CAP', '0.30'))  # max lateral displacement (m)
        # Snap to the MINIMUM displacement that reaches this clearance FLOOR (off the
        # wall), not the max-clearance cell — keeps the landing near the policy's goal
        # and avoids over-centering into wide pockets / away from narrow maze throats.
        self._route_target_clear: float = float(os.environ.get('OSL_ROUTE_TARGET_CLEAR', '0.40'))
        # A* router: when the policy's straight step is blocked (clamp advances < minstep),
        # plan a centerline-biased path AROUND the wall toward the policy's raw target and
        # take its first ~0.5 m waypoint. W_CLEAR pulls the path onto the EDT ridge.
        self._route_wclear: float = float(os.environ.get('OSL_ROUTE_WCLEAR', '2.0'))
        self._route_minstep: float = float(os.environ.get('OSL_ROUTE_MINSTEP', '0.15'))
        self._route_horizon: float = float(os.environ.get('OSL_ROUTE_HORIZON', '5.0'))  # m, free-space flood reach
        self._route_goal_radius: float = float(os.environ.get('OSL_ROUTE_GOAL_RADIUS', '0.25'))  # plan the GOAL like 0.05 (0.25)
        # Goal-clamp radius: the radius the policy's 0.5m target is validated/walked-back
        # at. Default 0.28 (= physical body). Set OSL_GOAL_CLAMP_RADIUS=0.05 to compute the
        # goal EXACTLY as the 0.05-era runs did (they used 0.25; 0.05 = treat the body as a
        # point for planning). The 0.28 body is still kept safe downstream, so a wall-closer
        # goal just lets the body drive further along the ray before BasicSim stops it.
        self._goal_clamp_radius: float = float(os.environ.get('OSL_GOAL_CLAMP_RADIUS', str(cfg.ROBOT_RADIUS)))
        # Deterministic eval: use the policy's MEAN action instead of sampling from its
        # high-variance Gaussian (std up to 1.65). Sampling is noise-dominated at deploy
        # (the headings wander); the mean is the clean learned heading. OSL_DETERMINISTIC=1.
        self._deterministic: bool = os.environ.get('OSL_DETERMINISTIC', '0') == '1'
        # GOAL-REPLAY diagnostic (definitive goal-following test): when OSL_REPLAY_GOALS
        # points at a file of "x y" world points (e.g. a recorded 0.05 success path), the
        # POLICY is bypassed and the 0.28 body is driven to each recorded goal in turn via
        # the SAME clamp+route+DirectDrive pipeline. Tests whether the executor can follow
        # the exact 0.05 goals to the source (isolates movement from the policy).
        self._replay_path: str = os.environ.get('OSL_REPLAY_GOALS', '')
        self._replay_goals_list = None
        self._replay_idx: int = 0
        if self._replay_path:
            pts = []
            try:
                with open(self._replay_path) as _f:
                    for _ln in _f:
                        _ln = _ln.strip()
                        if not _ln or _ln.startswith('#'):
                            continue
                        _a, _b = _ln.split()[:2]
                        pts.append((float(_a), float(_b)))
                self._replay_goals_list = pts
            except Exception as _e:
                self.get_logger().error(f'OSL_REPLAY_GOALS load failed: {_e}')
        self._route_freemasks = None   # inflation-eroded free masks (3,2,1,0 cells), built at map load
        self._dd_debug: bool = os.environ.get('OSL_DD_DEBUG', '0') == '1'   # per-step goal-chain trace
        self._clear_cells = None    # EDT clearance field (cells-to-nearest-wall), built at map load
        if self._route_on and not self._direct_drive:
            self.get_logger().warn(
                'OSL_ROUTE=1 requires OSL_DIRECT_DRIVE=1 (it layers on the DirectDrive '
                'executor) — disabling ClearanceRoute.')
            self._route_on = False
        if self._pursuit and not (self._direct_drive and self._route_on):
            self.get_logger().warn(
                'OSL_PURSUIT=1 requires OSL_DIRECT_DRIVE=1 + OSL_ROUTE=1 (cmd_vel + cone-min '
                '+ freemasks) — disabling CenterlinePursuit.')
            self._pursuit = False

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

        # ClearanceRoute: cache the Euclidean distance transform of the free space
        # once (cells-to-nearest-wall; *resolution = meters). Its local maxima are the
        # corridor centerlines that _snap_to_clearance climbs toward. Few-ms, one-time.
        if self._route_on:
            try:
                from scipy.ndimage import distance_transform_edt
                self._clear_cells = distance_transform_edt(self._occ_map.grid == 0)
                # Inflation-eroded free masks (robot radius ~3 cells), relaxed 3->2->1->0
                # so narrow doorways stay plannable; the A* router uses these.
                occ = (self._occ_map.grid != 0)
                self._route_freemasks = []
                # Radius-consistent inflation: the PRIMARY (first) mask must clear exactly
                # the body radius so A* paths are clearance-true at cfg.ROBOT_RADIUS, not a
                # hardcoded 0.30 m. r_cells = round(radius/res): 0.20/0.10 -> 2, 0.28 -> 3
                # (so the old (3,2,1,0) is reproduced for the 0.28 body = no regression).
                _r_cells = max(1, int(round(cfg.ROBOT_RADIUS / self._occ_map.resolution)))
                _infl_levels = sorted({_r_cells, _r_cells - 1, max(0, _r_cells - 2), 0}, reverse=True)
                for infl in _infl_levels:
                    block = occ.copy()
                    for _ in range(infl):
                        d = block.copy()
                        d[1:, :] |= block[:-1, :]; d[:-1, :] |= block[1:, :]
                        d[:, 1:] |= block[:, :-1]; d[:, :-1] |= block[:, 1:]
                        block = d
                    self._route_freemasks.append((self._occ_map.grid == 0) & (~block))
                self.get_logger().info(
                    f'ClearanceRoute ENABLED: EDT clearance field + A* free masks built '
                    f'(snap_r={self._route_snap_r} m, cap={self._route_snap_cap} m, '
                    f'wclear={self._route_wclear}, minstep={self._route_minstep} m). '
                    f'max clearance={self._clear_cells.max()*self._occ_map.resolution:.2f} m')
            except Exception as e:
                self._clear_cells = None
                self._route_freemasks = None
                self.get_logger().warn(f'ClearanceRoute: EDT build failed ({e}) — snap/A* disabled.')

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

        self._agent = _load_agent(self._checkpoint_path, self._arch, self._device)
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
                drive_mode=(self._use_nav2 or self._direct_drive),
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
        # Opt-in radial escape from _clamp_to_free dead-ends (pocket/edge traps where
        # every policy goal is rejected -> robot would otherwise freeze in place).
        # Independent of OSL_ESCAPE (the circling/map-stall escape).
        self._clamp_escape = os.environ.get('OSL_CLAMP_ESCAPE', '0') == '1'
        self._escape_dump_every = int(os.environ.get('OSL_ESCAPE_DUMP_EVERY', '0'))
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
                grow_win=int(os.environ.get('OSL_ESCAPE_GROW_WIN', '120')),
                grow_min=int(os.environ.get('OSL_ESCAPE_GROW_MIN', '250')),
                frontier_min_cells=int(os.environ.get('OSL_ESCAPE_FRONTIER_MIN', '12')),
                target_mode=os.environ.get('OSL_ESCAPE_TARGET', 'largest'),
            )
            self.get_logger().info(
                'SLAM stuck-escape ENABLED — trigger on EITHER signal: '
                f'circling (eff-streak>={self._escape.streak_thr}, win={self._escape.win}, '
                f'ratio={self._escape.ratio}) OR map-growth stall '
                f'(<{self._escape.grow_min} new cells in {self._escape.grow_win} steps); '
                f'frontier gate >= {self._escape.frontier_min_cells} cells; '
                f'cooldown={self._escape.cooldown}, min_dist={self._escape.min_dist}m, '
                f'target={self._escape.target_mode}'
            )
            # Publish the escape's online SLAM map (the robot's actual perception)
            # so an RViz video can show it instead of the ground-truth map.
            slam_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                  reliability=QoSReliabilityPolicy.RELIABLE)
            self._slam_map_pub = self.create_publisher(OccupancyGrid, '/rlaika/slam_map', slam_qos)
            self._slam_map_timer = self.create_timer(0.5, self._publish_slam_map)
            self.get_logger().info('Publishing online SLAM map on /rlaika/slam_map (2 Hz).')
        except Exception as exc:
            self.get_logger().error(f'Failed to init circling-escape: {exc} — disabled.')
            self._escape = None

    def _publish_slam_map(self):
        """Publish the escape's online occupancy grid (robot's perception) as a
        nav_msgs/OccupancyGrid for visualization. -1 unknown, 0 free, 100 occ, 50 outlet."""
        if self._escape is None:
            return
        sm = self._escape.slam_map
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = sm.resolution
        msg.info.width = sm.width
        msg.info.height = sm.height
        msg.info.origin.position.x = sm.origin_x
        msg.info.origin.position.y = sm.origin_y
        msg.info.origin.orientation.w = 1.0
        g = sm.grid.flatten()
        ros = np.full_like(g, -1, dtype=np.int8)
        ros[g == 0] = 0
        ros[g == 1] = 100
        ros[g == 2] = 50
        msg.data = ros.tolist()
        self._slam_map_pub.publish(msg)

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

        # Debug: Nav2 global plan (to inspect the path an escape goal produces).
        self._plan_sub = self.create_subscription(
            NavPath, '/PioneerP3DX/plan', self._plan_callback, 10)

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

        if self._det_drive or self._direct_drive:
            self._cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)
            if self._direct_drive and self._pursuit:
                tick_cb = self._pursuit_tick
            elif self._direct_drive:
                tick_cb = self._direct_drive_tick
            else:
                tick_cb = self._cmd_vel_tick
            self._cmd_vel_timer = self.create_timer(0.05, tick_cb)
            if self._pursuit:
                self.get_logger().info(
                    'Motion mode: CENTERLINE-PURSUIT (A* path + smooth lookahead, '
                    f'continuous clearance-tapered speed; vmax={self._pp_vmax} m/s, '
                    f'gain={self._pp_gain}, wmax={self._pp_wmax}, lookahead={self._pp_lookahead} m, '
                    f'arrive_tol={self._pp_arrive_tol} m, cone_stop={self._pp_cone_stop}/'
                    f'cone_free={self._pp_cone_free} m, spin={self._pp_spin} rad, '
                    f'clear_w={self._pp_clear_w}, stuck={self._pp_stuck_s}s, timeout={self._pp_timeout}s)')
            elif self._direct_drive:
                self.get_logger().info(
                    'Motion mode: DIRECT DRIVE (closed-loop cmd_vel, lidar+occupancy '
                    f'safety-gated; linear={self._dd_linear} m/s, gain={self._dd_gain}, '
                    f'wmax={self._dd_wmax} rad/s, arrive_tol={self._dd_arrive_tol} m, '
                    f'lookahead={self._dd_lookahead} m, lidar_margin={self._dd_lidar_margin} m, '
                    f'stuck={self._dd_stuck_s}s, timeout={self._dd_timeout}s)')
            else:
                self.get_logger().info(
                    'HYBRID drive: ESCAPE HOPS via deterministic cmd_vel '
                    f'(linear={self._det_linear} m/s, gain={self._det_gain}, '
                    f'arrive_tol={self._det_arrive_tol} m, timeout={self._det_timeout}s); '
                    'POLICY steps stay on Nav2/DWB (obstacle-avoiding)')


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
        if (self._use_nav2 or self._direct_drive) and self._is_moving:
            # DirectDrive owns its own arrival/timeout/stuck logic inside the tick,
            # so under it this gate simply waits (returns) while moving. The Nav2
            # per-step drive timeout below applies only to Nav2.
            if self._use_nav2:
                # Per-step drive timeout: if DWB gets wedged (recovery-loop stall),
                # cancel the goal so the policy re-predicts in ~drive_timeout s instead
                # of waiting ~150 s for Nav2's behavior tree to give up.
                # Escape hops are short (~0.5 m line-of-sight); fail them fast (10 s)
                # so a wedged hop is skipped quickly instead of burning 30 s.
                eff_timeout = self._drive_timeout
                if self._escape is not None and self._escape.escaping:
                    eff_timeout = min(self._drive_timeout, 10.0)
                _goal_t = self._drive_goal_time
                if (eff_timeout > 0.0 and _goal_t is not None
                        and not self._drive_canceling):
                    elapsed = (self.get_clock().now()
                               - _goal_t).nanoseconds * 1e-9
                    if elapsed > eff_timeout:
                        self._drive_canceling = True
                        self.get_logger().warn(
                            f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                            f'DRIVE TIMEOUT — goal not reached in {elapsed:.1f}s '
                            f'(> {eff_timeout:.1f}s); canceling, re-predicting.')
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

        # Forward-cone min (DirectDrive safety gate): nearest return within a
        # +/-_dd_cone_half arc about robot forward. Computed in the SENSOR frame
        # (ray i -> angle_min + i*increment, ray 0 = robot forward), so it is
        # heading-invariant and needs no yaw correction. Using the forward arc
        # (not the global min) avoids a wall beside/behind falsely holding the
        # gate and stalling forward motion in a wide corridor.
        if self._direct_drive:
            cone = None
            ang = msg.angle_min
            for r in msg.ranges:
                if math.isfinite(r):
                    a = math.atan2(math.sin(ang), math.cos(ang))  # wrap to [-pi,pi]
                    if abs(a) <= self._dd_cone_half and (cone is None or r < cone):
                        cone = r
                ang += msg.angle_increment
            self._dd_cone_min_sensor = cone

        self._obs_builder.update_lidar(msg)

        # Fold the same scan into the online SLAM map for the circling-escape —
        # but ONLY once the episode is actually running (_slam_enabled, set in the
        # first _take_step). The robot's GADEN spawn pose is often near the source
        # (e.g. many_rooms spawns at [1.35,2.05], ~1 m from it). Gating on scan
        # stamps alone is NOT enough: the first post-teleport scan can arrive while
        # self._robot_x/y is still the stale spawn pose, mapping a phantom region
        # onto the source. _take_step only runs once the pose is confirmed at the
        # start, so keying off it guarantees every integrated scan uses a real pose.
        if self._escape is not None and self._slam_enabled:
            self._escape.update_scan(
                msg, self._robot_x, self._robot_y, self._current_theta)

    def _plan_callback(self, msg: NavPath):
        """Store Nav2's latest global plan; save it if an escape just requested capture."""
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._latest_plan = pts
        if self._capture_plan_for is not None and self._escape is not None \
                and self._escape.dump_dir and pts:
            try:
                np.savez(os.path.join(self._escape.dump_dir,
                                      f'plan_step{self._capture_plan_for}.npz'),
                         path=np.array(pts, dtype=np.float32),
                         target=np.array(self._escape_last_target, dtype=np.float32))
                self.get_logger().warn(
                    f'[escape] captured Nav2 plan ({len(pts)} pts, '
                    f'end=({pts[-1][0]:.2f},{pts[-1][1]:.2f})) for escape@{self._capture_plan_for}')
            except Exception as exc:
                self.get_logger().warn(f'[escape] plan capture failed: {exc}')
            self._capture_plan_for = None

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

    def _start_cmdvel_move(self, x, y):
        """Begin a deterministic cmd_vel drive to (x,y). _cmd_vel_tick drives it
        and clears _is_moving on arrival/timeout (mirrors _on_nav_complete).
        Leaves _drive_goal_time None so the Nav2 per-step timeout path stays inert."""
        self._waypoint_x = float(x)
        self._waypoint_y = float(y)
        self._waypoint_start_ns = self.get_clock().now().nanoseconds
        self._is_moving = True
        self._cmdvel_active = True
        self._drive_goal_time = None

    def _cmd_vel_tick(self):
        """Proportional rotate-then-go controller: turn to face the target, then
        drive straight to it. Bypasses Nav2/DWB so it never wedges on near-wall
        hops and lands ON the commanded point. Stops on arrival or timeout."""
        if not self._cmdvel_active or self._waypoint_x is None:
            return                                  # only drive during an escape hop
        if self._robot_x is None or self._current_theta is None:
            return
        dx = self._waypoint_x - self._robot_x
        dy = self._waypoint_y - self._robot_y
        dist = math.hypot(dx, dy)
        timed_out = ((self.get_clock().now().nanoseconds - self._waypoint_start_ns)
                     > self._det_timeout * 1e9)
        if dist < self._det_arrive_tol or timed_out:
            self._cmd_vel_pub.publish(Twist())        # stop
            self._cmdvel_active = False
            self._is_moving = False                   # -> next pose cb takes a step
            return
        heading_error = math.atan2(dy, dx) - self._current_theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        tw = Twist()
        # rotate in place when badly misaligned, else drive forward while steering
        tw.linear.x = 0.0 if abs(heading_error) > 0.8 else self._det_linear
        tw.angular.z = self._det_gain * heading_error
        self._cmd_vel_pub.publish(tw)

    # ------------------------------------------------------------------
    # DirectDrive: physical, lidar-gated closed-loop drive to each goal
    # ------------------------------------------------------------------
    def _start_direct_move(self, x, y):
        """Begin a DirectDrive physical move to the (already-clamped) goal (x,y).
        _direct_drive_tick drives it and clears _is_moving on arrival/timeout/stuck.
        Uses a SEPARATE _dd_active flag so the det-drive HYBRID escape path is
        untouched; leaves _drive_goal_time None so the Nav2 timeout path stays inert."""
        self._waypoint_x = float(x)
        self._waypoint_y = float(y)
        self._waypoint_start_ns = self.get_clock().now().nanoseconds
        self._dd_progress_ns = self._waypoint_start_ns
        self._dd_best_dist = float('inf')
        self._is_moving = True
        self._dd_active = True
        self._drive_goal_time = None

    def _dd_forward_cone_min(self):
        """Nearest lidar return within +/-_dd_cone_half of robot forward (meters),
        or None if no scan yet. Cached per-scan in _lidar_callback (sensor frame,
        heading-invariant)."""
        return self._dd_cone_min_sensor

    def _direct_drive_tick(self):
        """Proportional rotate-then-go controller with a per-tick safety gate.
        Rotation in place cannot translate the center toward a wall, so only
        FORWARD translation is gated: it is published only when (a) the body-heading
        occupancy lookahead is free at radius ROBOT_RADIUS and (b) the forward-cone
        lidar min clears _dd_lidar_margin. Otherwise linear.x=0 (rotate/recover in
        place). Goals are pre-clamped clear, so the gate is skipped within lookahead
        range of the goal to avoid a false-block on the final approach. Stops on
        arrival (within _dd_arrive_tol), hard timeout, or no-progress (stuck), then
        clears _is_moving so the next _pose_callback advances the step."""
        if not self._dd_active or self._waypoint_x is None:
            return
        if self._robot_x is None or self._current_theta is None:
            return
        rx, ry, th = self._robot_x, self._robot_y, self._current_theta
        dx = self._waypoint_x - rx
        dy = self._waypoint_y - ry
        dist = math.hypot(dx, dy)
        now_ns = self.get_clock().now().nanoseconds

        # Heading error to goal, wrapped to [-pi, pi] (computed first so the watchdog
        # can tell turning-in-place apart from being stuck).
        he = math.atan2(dy, dx) - th
        he = math.atan2(math.sin(he), math.cos(he))
        rotating = abs(he) > 0.8                         # rotate-in-place gate (~46 deg)

        # Progress watchdog: fast recovery (re-predict) vs the slow hard timeout.
        # Rotation in place makes no translation progress but is NOT stuck — the robot
        # is pivoting to face the goal — so reset the clock while rotating. Without
        # this, any pivot longer than _dd_stuck_s (e.g. a ~180-240 deg turn-around in a
        # dead-end) is aborted mid-turn and the robot can NEVER reverse direction. This
        # was the corner-stuck bug behind every "froze at the wall" we saw.
        if dist < self._dd_best_dist - 0.02:
            self._dd_best_dist = dist
            self._dd_progress_ns = now_ns
        if rotating:
            self._dd_progress_ns = now_ns
        no_progress = (now_ns - self._dd_progress_ns) > self._dd_stuck_s * 1e9
        timed_out = (now_ns - self._waypoint_start_ns) > self._dd_timeout * 1e9

        if dist < self._dd_arrive_tol or timed_out or no_progress:
            self._cmd_vel_pub.publish(Twist())          # zero Twist = stop
            if self._dd_debug:
                reason = 'arrive' if dist < self._dd_arrive_tol else ('timeout' if timed_out else 'stuck')
                dur = (now_ns - self._waypoint_start_ns) * 1e-9
                self.get_logger().info(
                    f'[DD end] reason={reason} dist_left={dist:.3f} dur={dur:.2f}s '
                    f'pos=({rx:.2f},{ry:.2f}) he={math.degrees(he):.0f}deg rotating={rotating} '
                    f'goal=({self._waypoint_x:.2f},{self._waypoint_y:.2f}) best={self._dd_best_dist:.3f}')
            self._dd_active = False
            self._is_moving = False                     # -> next pose cb advances step
            self._waypoint_x = self._waypoint_y = None
            return

        # Safety gate (forward translation only; skipped near a pre-clamped goal).
        forward_ok = True
        if not rotating and dist > self._dd_lookahead:
            look_x = rx + self._dd_lookahead * math.cos(th)
            look_y = ry + self._dd_lookahead * math.sin(th)
            occ_ok = (not hasattr(self, '_occ_map')) or \
                self._occ_map.is_valid((look_x, look_y), radius=cfg.ROBOT_RADIUS)
            cone_min = self._dd_forward_cone_min()
            lid_ok = (cone_min is None) or (cone_min > self._dd_lidar_margin)
            forward_ok = occ_ok and lid_ok

        tw = Twist()
        if rotating or not forward_ok:
            tw.linear.x = 0.0                            # freeze translation, keep steering
        else:
            tw.linear.x = self._dd_linear * min(1.0, dist / self._dd_slow_r)  # near-goal ramp
        w = self._dd_gain * he
        tw.angular.z = max(-self._dd_wmax, min(self._dd_wmax, w))  # explicit clamp
        self._cmd_vel_pub.publish(tw)

    def _drive_escape_waypoint(self, wp):
        """Drive one short escape-path waypoint (DirectDrive, Nav2 stop-go, cmd_vel, or teleport)."""
        wx, wy = float(wp[0]), float(wp[1])
        if self._direct_drive:
            self._start_direct_move(wx, wy)
            return
        if self._det_drive:
            self._start_cmdvel_move(wx, wy)
            return
        if self._use_nav2 and self._navigator is not None:
            self._is_moving = True
            self._drive_canceling = False
            self._drive_goal_time = self.get_clock().now()
            self._navigator.send_goal(
                wx, wy, use_orientation=False, tolerance=self._nav_goal_tolerance)
        else:
            self._teleport_to(wx, wy)

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
        # When the policy is stuck (circling or loitering), plan a free-space path
        # to a reachable unexplored frontier and DRIVE IT IN SHORT ~0.5 m HOPS.
        # Short line-of-sight hops are what the DWB controller handles reliably;
        # one long cross-doorway goal makes it wedge on the wall.
        if self._escape is not None:
            self._slam_enabled = True   # pose is now confirmed post-teleport
            _og = 1 if (self._latest_gas_raw is not None
                        and self._latest_gas_raw > self._escape.src_gas_eps) else 0
            _v = self._latest_wind_speed if self._latest_wind_speed is not None else 0.0
            _phi = self._latest_wind_dir if self._latest_wind_dir is not None else 0.0
            self._escape.record_step(self._robot_x, self._robot_y, _og, _v, _phi)
            if self._escape_dump_every and \
                    self._step_in_episode % self._escape_dump_every == 0:
                self._escape.dump_map(self._step_in_episode,
                                      self._robot_x, self._robot_y)
            # (a) continue an in-progress escape: drive the next path waypoint
            if self._escape.escaping:
                wp = self._escape.next_waypoint(self._robot_x, self._robot_y)
                if wp is not None:
                    self._drive_escape_waypoint(wp)
                    self._step_in_episode += 1
                    return
                # path exhausted → fall through to resume the policy
            else:
                # (b) start an escape if stuck (plans the path, enters escaping)
                info = self._escape.start_escape_if_stuck(
                    self._robot_x, self._robot_y, self._step_in_episode)
                if info is not None:
                    tx, ty, fsize, fdist, nwp = info
                    self.get_logger().warn(
                        f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                        f'STUCK ({self._escape.stuck_reason}) → escape to '
                        f'({tx:.2f},{ty:.2f}) via {nwp} hops '
                        f'[frontier {fsize} cells, {fdist:.1f}m away, '
                        f'mapped={self._escape.mapped_fraction()*100:.0f}%]')
                    wp = self._escape.next_waypoint(self._robot_x, self._robot_y)
                    if wp is not None:
                        self._drive_escape_waypoint(wp)
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
                action, _, _, _ = self._agent.get_action_and_value(
                    obs_t, deterministic=self._deterministic)
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

        # --- GOAL-REPLAY override: ignore the policy, drive to the next recorded goal. ---
        if self._replay_goals_list is not None:
            if self._replay_idx >= len(self._replay_goals_list):
                self.get_logger().info(
                    f'[Episode {self._episode}] Replay goals exhausted at step '
                    f'{self._step_in_episode} (final d2src={dist_to_source:.3f} m).')
                self._end_episode(success=(dist_to_source < cfg.D_SUCCESS))
                return
            gx, gy = self._replay_goals_list[self._replay_idx]
            self._replay_idx += 1
            theta = math.atan2(gy - self._robot_y, gx - self._robot_x)
            target_x, target_y = float(gx), float(gy)

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
            slam_text = (f' slam_grow={self._escape.slam_growth} mapped={self._escape.mapped_now}'
                         if self._escape is not None else '')
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'Pos ({self._robot_x:.2f},{self._robot_y:.2f}) '
                f'Action {action_text} θ={math.degrees(theta):.1f}deg → '
                f'Target ({target_x:.2f},{target_y:.2f}) | '
                f'd2src={dist_to_source:.2f}m gas={gas_text} wind=({wind_s_text}m/s,{wind_d_text}deg) '
                f'lidar_min={lidar_text}m{slam_text}'
            )

        # --- Collision check + clamp to free cell ---
        # If basic_sim would reject this teleport (target inside a wall), the
        # robot wouldn't move at all. Walk back along the ray until we land in
        # a free cell, exactly like the training collision logic.
        raw_tx, raw_ty = target_x, target_y    # policy's intended target (pre-clamp), for A* routing
        target_x, target_y, collided = self._clamp_to_free(
            self._robot_x, self._robot_y, target_x, target_y, theta,
            radius=self._goal_clamp_radius
        )
        if collided:
            self.get_logger().info(
                f'[Ep {self._episode} Step {self._step_in_episode:3d}] '
                f'COLLISION — clamped to ({target_x:.2f},{target_y:.2f})'
            )

        if self._publish_action_marker:
            self._publish_next_action_marker(target_x, target_y)

        # --- Move robot: DirectDrive (physical, lidar-gated), Nav2 (stop-go), or
        # teleport. Target is already collision-clamped above; do NOT re-clamp. ---
        if self._direct_drive:
            if self._pursuit:
                # CENTERLINE-PURSUIT: build a static-map path to the policy's goal, then
                # _pursuit_tick follows it smoothly. Rebuilt every policy step (~0.5 m).
                gx, gy = target_x, target_y
                clamp_step = math.hypot(gx - self._robot_x, gy - self._robot_y)
                if collided and clamp_step < self._route_minstep:
                    # BLOCKED: the clamp gave no forward progress (policy points into a wall,
                    # stay-put). Route toward the policy's RAW intent (raw_tx,raw_ty ~ the
                    # heading) with a clearance bias -> go AROUND the wall toward the corridor
                    # CENTER (the clearance term), NOT sliding along the wall/ceiling (which
                    # was the wander root cause). This replaces the old raw-heading flood.
                    g = self._route_clearance_step(self._robot_x, self._robot_y, raw_tx, raw_ty)
                    path = ([(self._robot_x, self._robot_y), g] if g is not None
                            else [(self._robot_x, self._robot_y), (self._robot_x, self._robot_y)])
                elif self._seg_is_valid(self._robot_x, self._robot_y, gx, gy):
                    path = [(self._robot_x, self._robot_y), (gx, gy)]
                else:
                    path = self._route_astar(self._robot_x, self._robot_y, gx, gy)
                    if not path or len(path) < 2:
                        g = self._route_clearance_step(self._robot_x, self._robot_y, gx, gy)
                        path = ([(self._robot_x, self._robot_y), g] if g is not None
                                else [(self._robot_x, self._robot_y), (self._robot_x, self._robot_y)])
                self._start_pursuit(path)
            elif self._route_on:
                # MOVE LIKE THE 0.05 RUNS: drive to the policy's (clamped, valid) goal.
                # The goal is exactly what a 0.05-radius run targets (clamp radii 0.25 vs
                # 0.28 differ on 0% of steps). The ONLY difference is the 0.28 body, so the
                # ONLY job here is MOVEMENT: reach that goal collision-free. Drive straight
                # if the segment is clear; otherwise plan a short A* path AROUND the wall to
                # the SAME goal (Nav2's global planner rebuilt honestly: 0.28-inflated grid,
                # centerline cost, string-pulled to one <=0.5 m hop). The reverse-capable
                # controller (below) backs into goals behind it instead of wedging on a 180
                # deg spin (the thing Nav2's forward-only DWB could not do). No gas, no
                # heading-flood, no snap — purely getting the body to the 0.05 goal.
                gx, gy = target_x, target_y
                clamp_step = math.hypot(gx - self._robot_x, gy - self._robot_y)
                blocked = collided and clamp_step < self._route_minstep
                if blocked:
                    # The clamp only walks STRAIGHT back along the policy ray; when that ray
                    # is walled for the 0.28 body it gives up (stay-put) and the robot
                    # deadlocks (a 0.05 body would have squeezed forward). Go AROUND the
                    # wall toward the policy's intent instead: flood the reachable free
                    # space and step toward the cell with most progress along the heading
                    # (capped 0.5 m). This is the planner the bigger body needs — pure
                    # movement, the "better planner" that lets 0.28 reach 0.05's goals.
                    g = self._route_toward_heading(self._robot_x, self._robot_y, theta)
                    if g is not None:
                        target_x, target_y = g
                    # else: truly no reachable progress -> stay, re-predict next step.
                elif not self._seg_is_valid(self._robot_x, self._robot_y, gx, gy):
                    # Forward goal is valid but the straight segment grazes a wall -> A*
                    # a short path AROUND the wall to the SAME 0.05 goal.
                    wp = self._route_step_goal(self._robot_x, self._robot_y, gx, gy)
                    if wp is not None:
                        target_x, target_y = wp
                # else: straight line to the 0.05 goal is clear -> drive straight to it.
            if not self._pursuit:
                # Physically drive there via the closed-loop cmd_vel controller (0.28-collision-safe).
                self._start_direct_move(target_x, target_y)
        elif self._use_nav2 and self._navigator is not None:
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
                       tx: float, ty: float, theta: float, radius: float = None) -> tuple:
        """Validate target; if blocked, walk back along the ray until free.

        ``radius`` defaults to cfg.ROBOT_RADIUS (the 0.28 body). Pass a smaller value
        (e.g. OSL_GOAL_CLAMP_RADIUS=0.05) to compute the GOAL exactly like the 0.05-era
        runs (which clamped at 0.25); the 0.28 body is still kept collision-safe by the
        route/snap/DirectDrive gate + the BasicSim 0.28 backstop, so a goal closer to a
        wall just means the body drives further along the ray before it physically stops.

        Returns
        -------
        (clamped_x, clamped_y, collided) where ``collided`` is True iff the
        original target was not free.
        """
        rad = cfg.ROBOT_RADIUS if radius is None else radius
        if not hasattr(self, '_occ_map'):
            return tx, ty, False

        if self._occ_map.is_valid((tx, ty), radius=rad):
            return tx, ty, False

        # Original target blocked — walk back along the ray
        step = cfg.STEP_SIZE - 0.05
        while step >= 0.1:
            cx = rx + step * math.cos(theta)
            cy = ry + step * math.sin(theta)
            if self._occ_map.is_valid((cx, cy), radius=rad):
                return cx, cy, True
            step -= 0.05

        # Nowhere to go along the policy ray. Optionally sweep OTHER headings for any
        # valid nearby target so the robot escapes pocket/edge traps instead of
        # freezing in place for the rest of the episode (OSL_CLAMP_ESCAPE=1). Prefer
        # larger hops, and never re-try the (blocked) policy heading itself.
        if getattr(self, '_clamp_escape', False):
            for esc_r in (cfg.STEP_SIZE, 0.35, 0.25, 0.15):
                for j in range(1, 16):
                    ang = theta + j * (2.0 * math.pi / 16.0)
                    ex = rx + esc_r * math.cos(ang)
                    ey = ry + esc_r * math.sin(ang)
                    if self._occ_map.is_valid((ex, ey), radius=rad):
                        return ex, ey, True

        # Nowhere to go — stay in place
        return rx, ry, True

    # ------------------------------------------------------------------
    # ClearanceRoute: snap a clamped goal up to the corridor centerline
    # ------------------------------------------------------------------
    def _clear_at(self, wx: float, wy: float) -> float:
        """Clearance (m to nearest wall) at world point, 0.0 if out of bounds/no field."""
        if self._clear_cells is None:
            return 0.0
        gx, gy = self._occ_map.world_to_grid(wx, wy)
        if 0 <= gy < self._clear_cells.shape[0] and 0 <= gx < self._clear_cells.shape[1]:
            return float(self._clear_cells[gy, gx]) * self._occ_map.resolution
        return 0.0

    def _seg_is_valid(self, ax: float, ay: float, bx: float, by: float) -> bool:
        """True iff the 0.28-body straight segment a->b stays free (sampled <cell res)."""
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d / 0.05))
        for k in range(n + 1):
            t = k / n
            if not self._occ_map.is_valid((ax + t * (bx - ax), ay + t * (by - ay)),
                                          radius=cfg.ROBOT_RADIUS):
                return False
        return True

    def _goal_like_005(self, rx, ry, tx, ty, theta, radius):
        """Calculate the next GOAL exactly like the r=0.05 run: the first point along
        the policy ray that is `radius`-valid (0.05 used radius 0.25), walking back from
        the full 0.5 m target. This is the goal the 0.05 setup would aim at. Returns
        (rx,ry) only if nothing on the ray is valid."""
        if self._occ_map.is_valid((tx, ty), radius=radius):
            return tx, ty
        step = cfg.STEP_SIZE - 0.05
        while step >= 0.1:
            cx = rx + step * math.cos(theta)
            cy = ry + step * math.sin(theta)
            if self._occ_map.is_valid((cx, cy), radius=radius):
                return cx, cy
            step -= 0.05
        return rx, ry

    def _snap_to_clearance(self, rx: float, ry: float, cx: float, cy: float) -> tuple:
        """Nudge the clamped goal (cx,cy) just off the wall: the MINIMUM-displacement
        valid+reachable cell whose clearance reaches the floor (_route_target_clear).
        This un-grazes the landing pose (kills the stay-put cascade) while staying close
        to the policy's goal and NOT over-centering into wide pockets (which would dodge
        narrow maze throats). If the clamp goal is already above the floor, no move. If
        no cell reaches the floor within the disk (truly tight spot), fall back to the
        max-clearance cell as a best-effort un-graze."""
        if self._clear_cells is None:
            return cx, cy
        if self._clear_at(cx, cy) >= self._route_target_clear:
            return cx, cy
        res = self._occ_map.resolution
        snap_r = max(1, int(round(self._route_snap_r / res)))
        gx0, gy0 = self._occ_map.world_to_grid(cx, cy)
        H, W = self._clear_cells.shape
        floor_x = floor_y = None
        floor_disp = float('inf')        # nearest cell reaching the clearance floor
        fb_x, fb_y = cx, cy              # fallback: best-effort max clearance
        fb_c = self._clear_at(cx, cy)
        for dgy in range(-snap_r, snap_r + 1):
            for dgx in range(-snap_r, snap_r + 1):
                if dgx * dgx + dgy * dgy > snap_r * snap_r:
                    continue
                gx, gy = gx0 + dgx, gy0 + dgy
                if not (0 <= gy < H and 0 <= gx < W):
                    continue
                c = float(self._clear_cells[gy, gx]) * res
                if c <= fb_c and c < self._route_target_clear:
                    continue
                wx, wy = self._occ_map.grid_to_world(gx, gy)
                disp = math.hypot(wx - cx, wy - cy)
                if disp > self._route_snap_cap:
                    continue
                if not self._occ_map.is_valid((wx, wy), radius=cfg.ROBOT_RADIUS):
                    continue
                if not self._seg_is_valid(cx, cy, wx, wy):
                    continue
                if c >= self._route_target_clear:
                    if disp < floor_disp:        # prefer the closest off-wall cell
                        floor_disp, floor_x, floor_y = disp, wx, wy
                elif c > fb_c:
                    fb_x, fb_y, fb_c = wx, wy, c  # track best-effort fallback
        if floor_x is not None:
            return floor_x, floor_y
        return fb_x, fb_y

    @staticmethod
    def _nearest_free_cell(cell, free, r=6):
        cx, cy = cell
        if free(cx, cy):
            return (cx, cy)
        for rad in range(1, r + 1):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if free(cx + dx, cy + dy):
                        return (cx + dx, cy + dy)
        return None

    def _route_astar(self, sx, sy, gx, gy):
        """Centerline-biased A* on the static map: route the 0.28 body AROUND a wall
        toward (gx,gy) when the straight step is blocked. 8-connected, obstacle-inflated
        (robot radius, relaxed 3->2->1->0 so doorways stay plannable), no diagonal
        corner-cut, EDT-clearance cost to hug corridor centers. Returns ~0.5 m world
        waypoints (start first) each re-validated with is_valid(0.28), or None."""
        if self._route_freemasks is None or not hasattr(self, '_occ_map'):
            return None
        import heapq
        og = self._occ_map
        H, W = og.grid.shape
        res = og.resolution
        ox, oy = og.origin_x, og.origin_y
        clr = self._clear_cells
        wclr = self._route_wclear
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        for freemask in self._route_freemasks:        # already relaxed 3->2->1->0
            def free(cx, cy):
                return 0 <= cx < W and 0 <= cy < H and freemask[cy, cx]
            start = self._nearest_free_cell((int((sx - ox) / res), int((sy - oy) / res)), free)
            goal = self._nearest_free_cell((int((gx - ox) / res), int((gy - oy) / res)), free)
            if start is None or goal is None:
                continue

            def hcost(a):
                return math.hypot(a[0] - goal[0], a[1] - goal[1])

            openq = [(hcost(start), 0.0, start)]
            gsc = {start: 0.0}
            came = {}
            reached = None
            while openq:
                _, gc, cur = heapq.heappop(openq)
                if cur == goal or hcost(cur) <= 1.0:
                    reached = cur
                    break
                if gc > gsc.get(cur, 1e18):
                    continue
                for dx, dy, cost in nbrs:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if not free(nx, ny):
                        continue
                    if dx and dy and not (free(cur[0] + dx, cur[1]) and
                                          free(cur[0], cur[1] + dy)):
                        continue
                    cc = float(clr[ny, nx]) if clr is not None else 99.0
                    ng = gc + cost * (1.0 + wclr / (cc + 1.0))   # centerline bias
                    if ng < gsc.get((nx, ny), 1e18):
                        gsc[(nx, ny)] = ng
                        came[(nx, ny)] = cur
                        heapq.heappush(openq, (ng + hcost((nx, ny)), ng, (nx, ny)))
            if reached is None:
                continue
            cells = [reached]
            while cells[-1] in came:
                cells.append(came[cells[-1]])
            cells.reverse()
            out = []                                       # full-resolution world path
            for c in cells:
                wx, wy = og.grid_to_world(*c)
                if not og.is_valid((wx, wy), radius=cfg.ROBOT_RADIUS):
                    rep = self._nearest_free_cell(c, free)
                    if rep is None:
                        continue
                    wx, wy = og.grid_to_world(*rep)
                out.append((wx, wy))
            return out if len(out) >= 2 else None
        return None

    def _route_step_goal(self, rx, ry, gx, gy):
        """A* around the wall toward (gx,gy), then STRING-PULL: return the furthest
        point on the path still in straight line-of-sight from (rx,ry), capped at one
        STEP_SIZE. This stops at the corner (so the rotate-then-go controller never
        tries to drive straight through a wall) while making real progress around it.
        Returns (x,y) or None."""
        path = self._route_astar(rx, ry, gx, gy)
        if not path:
            return None
        goal = None
        for (wx, wy) in path[1:]:
            if math.hypot(wx - rx, wy - ry) > cfg.STEP_SIZE + 0.05:
                break                                      # cap step length
            if not self._seg_is_valid(rx, ry, wx, wy):
                break                                      # corner — stop at last clear
            goal = (wx, wy)
        return goal

    def _route_toward_heading(self, rx, ry, theta):
        """Direction-following local planner (the maze solver). Flood the reachable
        free space from the robot (Dijkstra on the inflation-eroded mask, so cells are
        >=0.3 m off walls), pick the reachable cell with MAXIMUM progress along the
        policy heading, backtrack the path, then string-pull to the first line-of-sight
        step (cap STEP_SIZE). This follows the corridor toward where the policy wants to
        go and BACKS OUT of dead-ends (the escape route's far cells outscore the pocket),
        instead of routing toward a target buried in a wall. Returns (x,y) or None."""
        if self._route_freemasks is None or not hasattr(self, '_occ_map'):
            return None
        import heapq
        og = self._occ_map
        res = og.resolution
        ox, oy = og.origin_x, og.origin_y
        H, W = og.grid.shape
        ct, st = math.cos(theta), math.sin(theta)
        horizon = self._route_horizon / res
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        for freemask in self._route_freemasks:
            def free(cx, cy):
                return 0 <= cx < W and 0 <= cy < H and freemask[cy, cx]
            start = self._nearest_free_cell((int((rx - ox) / res), int((ry - oy) / res)), free)
            if start is None:
                continue
            distc = {start: 0.0}
            came = {}
            pq = [(0.0, start)]
            best_cell = start
            best_score = 0.0
            while pq:
                d, cur = heapq.heappop(pq)
                if d > distc.get(cur, 1e18):
                    continue
                wx, wy = og.grid_to_world(*cur)
                score = (wx - rx) * ct + (wy - ry) * st        # progress along the heading
                if score > best_score:
                    best_score = score
                    best_cell = cur
                for dx, dy, cost in nbrs:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if not free(nx, ny):
                        continue
                    if dx and dy and not (free(cur[0] + dx, cur[1]) and free(cur[0], cur[1] + dy)):
                        continue
                    nd = d + cost
                    if nd <= horizon and nd < distc.get((nx, ny), 1e18):
                        distc[(nx, ny)] = nd
                        came[(nx, ny)] = cur
                        heapq.heappush(pq, (nd, (nx, ny)))
            if best_cell == start or best_score <= 0.05:
                continue                                        # no forward progress at this infl; relax
            cells = [best_cell]
            while cells[-1] in came:
                cells.append(came[cells[-1]])
            cells.reverse()
            goal = None
            for c in cells[1:]:
                wx, wy = og.grid_to_world(*c)
                if math.hypot(wx - rx, wy - ry) > cfg.STEP_SIZE + 0.05:
                    break
                if not self._seg_is_valid(rx, ry, wx, wy):
                    break
                goal = (wx, wy)
            if goal is not None:
                return goal
        return None

    def _route_clearance_step(self, rx, ry, gx, gy):
        """Clearance-seeking recovery (the anti-wander fix). Flood the reachable free
        space (Dijkstra on the inflation-eroded mask), score each cell by NET PROGRESS
        TOWARD THE GOAL plus an EDT-clearance bonus, pick the max, string-pull to the
        first line-of-sight step (cap STEP_SIZE). Unlike _route_toward_heading this follows
        the GOAL (not the raw policy heading) AND prefers higher clearance, so at a wall pin
        it pulls toward the corridor CENTER / goal instead of sliding along the wall/ceiling
        (the slide into y>5.5 off-distribution space was the wander root cause). Returns
        (x,y) or None."""
        if self._route_freemasks is None or not hasattr(self, '_occ_map'):
            return None
        import heapq
        og = self._occ_map
        res = og.resolution
        ox, oy = og.origin_x, og.origin_y
        H, W = og.grid.shape
        _dgx, _dgy = gx - rx, gy - ry
        _n = math.hypot(_dgx, _dgy)
        ugx, ugy = (_dgx / _n, _dgy / _n) if _n > 1e-6 else (0.0, 0.0)
        clr = self._clear_cells
        horizon = self._route_horizon / res
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        for freemask in self._route_freemasks:
            def free(cx, cy):
                return 0 <= cx < W and 0 <= cy < H and freemask[cy, cx]
            start = self._nearest_free_cell((int((rx - ox) / res), int((ry - oy) / res)), free)
            if start is None:
                continue
            distc = {start: 0.0}
            came = {}
            pq = [(0.0, start)]
            best_cell = start
            best_score = -1e18
            while pq:
                d, cur = heapq.heappop(pq)
                if d > distc.get(cur, 1e18):
                    continue
                wx, wy = og.grid_to_world(*cur)
                cc = (float(clr[cur[1], cur[0]]) * res) if clr is not None else 0.0
                score = (wx - rx) * ugx + (wy - ry) * ugy + self._pp_clear_w * cc
                if score > best_score:
                    best_score = score
                    best_cell = cur
                for dx, dy, cost in nbrs:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if not free(nx, ny):
                        continue
                    if dx and dy and not (free(cur[0] + dx, cur[1]) and free(cur[0], cur[1] + dy)):
                        continue
                    nd = d + cost
                    if nd <= horizon and nd < distc.get((nx, ny), 1e18):
                        distc[(nx, ny)] = nd
                        came[(nx, ny)] = cur
                        heapq.heappush(pq, (nd, (nx, ny)))
            if best_cell == start:
                continue
            cells = [best_cell]
            while cells[-1] in came:
                cells.append(came[cells[-1]])
            cells.reverse()
            goal = None
            for c in cells[1:]:
                wx, wy = og.grid_to_world(*c)
                if math.hypot(wx - rx, wy - ry) > cfg.STEP_SIZE + 0.05:
                    break
                if not self._seg_is_valid(rx, ry, wx, wy):
                    break
                goal = (wx, wy)
            if goal is not None:
                return goal
        return None

    def _start_pursuit(self, path):
        """Begin following a static-map path (list of (x,y), start first) with the smooth
        lookahead controller in _pursuit_tick. _pp_goal = the path's final point; the tick
        clears _is_moving on arrival/timeout/no-progress so the next policy step runs."""
        self._pp_path = path
        self._pp_idx = 0
        self._pp_goal = path[-1]
        now = self.get_clock().now().nanoseconds
        self._waypoint_start_ns = now
        self._pp_progress_ns = now
        self._pp_recover_ns = now
        self._pp_best = float('inf')
        self._pp_stall_n = 0
        self._is_moving = True
        self._pp_active = True
        self._drive_goal_time = None
        if self._dd_debug:
            self.get_logger().info(
                f'[PP plan] n={len(path)} goal=({self._pp_goal[0]:.2f},{self._pp_goal[1]:.2f}) '
                f'from=({path[0][0]:.2f},{path[0][1]:.2f})')

    def _pursuit_tick(self):
        """Smooth pure-pursuit follower of self._pp_path with CONTINUOUS clearance-tapered
        speed (creeps through 0.30 m throats, never a binary freeze => no wedge) and a
        clearance-seeking stall recovery (pulls toward goal+center, never the raw heading
        => no ceiling-slide). The lookahead is always ON the pre-validated path, so curving
        toward it is collision-safe. Clears _is_moving on arrival/timeout/no-progress."""
        if not self._pp_lock.acquire(blocking=False):
            return
        try:
            if not self._pp_active or self._pp_path is None:
                return
            if self._robot_x is None or self._current_theta is None:
                return
            rx, ry, th = self._robot_x, self._robot_y, self._current_theta
            path = self._pp_path
            gx, gy = self._pp_goal
            dist_goal = math.hypot(gx - rx, gy - ry)
            now = self.get_clock().now().nanoseconds

            # advance the path index past waypoints already reached
            while self._pp_idx < len(path) - 1 and \
                    math.hypot(path[self._pp_idx][0] - rx, path[self._pp_idx][1] - ry) < self._pp_advance:
                self._pp_idx += 1

            # lookahead point: walk forward from the index accumulating arc-length
            lx, ly = path[-1]
            acc = 0.0
            px, py = rx, ry
            for k in range(self._pp_idx, len(path)):
                wx, wy = path[k]
                seg = math.hypot(wx - px, wy - py)
                if acc + seg >= self._pp_lookahead:
                    t = (self._pp_lookahead - acc) / seg if seg > 1e-6 else 1.0
                    lx, ly = px + t * (wx - px), py + t * (wy - py)
                    break
                acc += seg
                px, py = wx, wy
                lx, ly = wx, wy

            # steer toward the lookahead
            he = math.atan2(ly - ry, lx - rx) - th
            he = math.atan2(math.sin(he), math.cos(he))
            w = max(-self._pp_wmax, min(self._pp_wmax, self._pp_gain * he))

            # continuous speed = binding constraint of heading-align x forward-clearance,
            # ramped down near the goal. Never a binary freeze (only slows in a throat).
            cone = self._dd_forward_cone_min()
            heading_f = max(0.0, 1.0 - abs(he) / self._pp_he_cut)
            if cone is None:
                cone_f = 1.0
            else:
                cone_f = (cone - self._pp_cone_stop) / max(1e-6, self._pp_cone_free - self._pp_cone_stop)
                cone_f = max(0.0, min(1.0, cone_f))
            goal_ramp = min(1.0, dist_goal / self._pp_slow_r)
            v = self._pp_vmax * min(heading_f, cone_f) * goal_ramp

            aligning = abs(he) > self._pp_spin       # turning in place to face the goal = progress
            rotate_in_place = aligning or (cone is not None and cone <= self._pp_cone_stop)

            # progress watchdog (a productive turn is NOT a stall: reset the clock while aligning)
            if dist_goal < self._pp_best - 0.02:
                self._pp_best = dist_goal
                self._pp_progress_ns = now
            if aligning:
                self._pp_progress_ns = now
                self._pp_stall_n = 0
            no_progress = (now - self._pp_progress_ns) > self._pp_stuck_s * 1e9
            timed_out = (now - self._waypoint_start_ns) > self._pp_timeout * 1e9

            if dist_goal < self._pp_arrive_tol or timed_out or no_progress:
                if self._dd_debug:
                    _rsn = 'arrive' if dist_goal < self._pp_arrive_tol else ('timeout' if timed_out else 'no_progress')
                    self.get_logger().info(
                        f'[PP end] {_rsn} d2goal={dist_goal:.2f} pos=({rx:.2f},{ry:.2f}) '
                        f'cone={(cone if cone is not None else -1):.2f} he={math.degrees(he):.0f} stall={self._pp_stall_n}')
                self._cmd_vel_pub.publish(Twist())
                self._pp_active = False
                self._is_moving = False
                self._pp_path = None
                return

            tw = Twist()
            tw.linear.x = 0.0 if rotate_in_place else v
            tw.angular.z = w
            self._cmd_vel_pub.publish(tw)

            # clearance-seeking stall recovery: truly pinned (not translating, not aligning)
            if v >= 0.03 and not rotate_in_place:
                self._pp_stall_n = 0
            elif (not aligning) and dist_goal > self._pp_arrive_tol:
                self._pp_stall_n += 1
            if self._pp_stall_n >= self._pp_stall_ticks and (now - self._pp_recover_ns) > 0.75e9:
                self._pp_recover_ns = now
                g = self._route_clearance_step(rx, ry, gx, gy)
                if self._dd_debug:
                    self.get_logger().info(
                        f'[PP recover] pinned ({rx:.2f},{ry:.2f}) cone={(cone if cone is not None else -1):.2f} '
                        f'he={math.degrees(he):.0f} -> clearance g={g}')
                if g is not None:
                    self._pp_path = [(rx, ry), g]
                    self._pp_goal = g
                    self._pp_idx = 0
                    self._pp_progress_ns = now
                    self._pp_best = float('inf')
                    self._pp_stall_n = 0
        finally:
            self._pp_lock.release()

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
        self._replay_idx = 0  # restart the recorded goal sequence for the new episode
        self._is_moving = False  # clear any lingering drive flag before next episode
        if self._direct_drive and self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(Twist())  # stop any in-flight DirectDrive / pursuit
            self._dd_active = False
            self._pp_active = False
            self._pp_path = None
            self._waypoint_x = self._waypoint_y = None
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
