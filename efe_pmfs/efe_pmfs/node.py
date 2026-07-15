"""
Top-level ROS 2 node for efe_pmfs.

Dual-mode PMFS GSL with online SLAM and a known mean wind vector.

Pipeline:
  LiDAR scan  -> OccupancyGridMap (SLAM)
  Gas sample  -> threshold to hit/miss -> HitMap.update(...)
  Every N stops -> SourceBelief.refresh(...) runs filament sims for each valid
                   lattice candidate, recomputes p(S|H) via hit-map comparison.
  Plan:
    LOCAL  : RRT-Infotaxis. BI computed against SourceBelief's cached f^{S_k}.
             BI* below adaptive threshold → switch to GLOBAL.
    GLOBAL : Frontier-based PRM with PMFS-MI utility. MI recovery → LOCAL.
  Declare on variance of p(S|H) below threshold.
"""
from __future__ import annotations
from math import atan2
from typing import List, Optional, Tuple
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from olfaction_msgs.msg import Anemometer, GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

from .mapping.occupancy_grid import create_empty_occupancy_map_from_dims
from .mapping.lidar_mapper import LidarMapper
from .utils.scenario import load_scenario
from .pmfs.constant_wind import ConstantWindField
from .pmfs.hit_map import HitMap, HitMapSettings
from .pmfs.filament_sim import FilamentSimulator, FilamentSettings
from .pmfs.source_belief import SourceBelief
from .planning.rrt import RRT
from .planning.global_planner import GlobalPlanner
from .planning.dead_end_detector import DeadEndDetector
from .planning.navigator import Navigator
from .visualization.markers import MarkerVisualizer
from .utils.logger import ExperimentLogger


class EfePMFSNode(Node):
    def __init__(self):
        super().__init__('efe_pmfs_node')
        self._init_parameters()
        self._init_state()
        self._init_components()
        self._init_ros_interfaces()
        self.node_initialized = True
        self.get_logger().info('efe_pmfs node initialized.')

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_parameters(self):
        # Scenario (optional). If set (e.g. "env_b" or "env_b/sim3") the map
        # bounding box, resolution, mean wind, and true source are auto-loaded
        # from install/test_env and the corresponding manual params are ignored.
        self.declare_parameter('scenario', '')
        self.declare_parameter('override_cell_size_m', 0.0)
        # Map
        self.declare_parameter('map_width_m', 24.0)
        self.declare_parameter('map_height_m', 16.0)
        self.declare_parameter('map_origin_x', 0.0)
        self.declare_parameter('map_origin_y', 0.0)
        self.declare_parameter('cell_size_m', 0.2)
        # Wind
        self.declare_parameter('wind_u_mean', 0.0)
        self.declare_parameter('wind_v_mean', 0.0)
        # Hit map
        self.declare_parameter('hit_threshold_ppm', 0.3)
        self.declare_parameter('hit_prior', 0.3)
        self.declare_parameter('hit_kernel_sigma', 1.5)
        self.declare_parameter('hit_kernel_stretch', 1.0)
        self.declare_parameter('hit_local_window', 20)
        self.declare_parameter('confidence_sigma_spatial', 1.0)
        self.declare_parameter('confidence_measurement_weight', 1.0)
        # Filament sim
        self.declare_parameter('filament_noise_stddev', 0.2)
        self.declare_parameter('filament_dt', 0.1)
        self.declare_parameter('filament_min_warmup', 20)
        self.declare_parameter('filament_max_warmup', 300)
        self.declare_parameter('filament_record_steps', 100)
        self.declare_parameter('filament_per_step', 5)
        # Source belief
        self.declare_parameter('candidate_stride', 5)
        self.declare_parameter('source_discrimination_power', 0.2)
        self.declare_parameter('source_update_every_n_stops', 3)
        self.declare_parameter('declaration_std_threshold', 0.7)  # metres
        # RRT
        self.declare_parameter('n_tn', 50)
        self.declare_parameter('delta', 0.7)
        self.declare_parameter('max_depth', 4)
        self.declare_parameter('positive_weight', 0.8)  # discount factor γ
        # Dual-mode
        self.declare_parameter('enable_global_planner', True)
        self.declare_parameter('dead_end_epsilon', 0.6)
        self.declare_parameter('dead_end_initial_threshold', 0.1)
        self.declare_parameter('switch_back_threshold', 1.5)
        # Global planner
        self.declare_parameter('prm_samples', 300)
        self.declare_parameter('prm_connection_radius', 5.0)
        self.declare_parameter('frontier_min_size', 3)
        self.declare_parameter('lambda_p', 0.1)
        self.declare_parameter('lambda_s', 0.05)
        # Navigation
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('robot_radius', 0.25)
        # Logging / ground truth
        self.declare_parameter('true_source_x', -999.0)
        self.declare_parameter('true_source_y', -999.0)

        g = self.get_parameter
        self.P = {
            'scenario': g('scenario').value,
            'override_cell_size_m': g('override_cell_size_m').value,
            'map_width_m': g('map_width_m').value,
            'map_height_m': g('map_height_m').value,
            'map_origin_x': g('map_origin_x').value,
            'map_origin_y': g('map_origin_y').value,
            'cell_size_m': g('cell_size_m').value,
            'wind_u_mean': g('wind_u_mean').value,
            'wind_v_mean': g('wind_v_mean').value,
            'hit_threshold_ppm': g('hit_threshold_ppm').value,
            'hit_prior': g('hit_prior').value,
            'hit_kernel_sigma': g('hit_kernel_sigma').value,
            'hit_kernel_stretch': g('hit_kernel_stretch').value,
            'hit_local_window': g('hit_local_window').value,
            'confidence_sigma_spatial': g('confidence_sigma_spatial').value,
            'confidence_measurement_weight': g('confidence_measurement_weight').value,
            'filament_noise_stddev': g('filament_noise_stddev').value,
            'filament_dt': g('filament_dt').value,
            'filament_min_warmup': g('filament_min_warmup').value,
            'filament_max_warmup': g('filament_max_warmup').value,
            'filament_record_steps': g('filament_record_steps').value,
            'filament_per_step': g('filament_per_step').value,
            'candidate_stride': g('candidate_stride').value,
            'source_discrimination_power': g('source_discrimination_power').value,
            'source_update_every_n_stops': g('source_update_every_n_stops').value,
            'declaration_std_threshold': g('declaration_std_threshold').value,
            'n_tn': g('n_tn').value,
            'delta': g('delta').value,
            'max_depth': g('max_depth').value,
            'positive_weight': g('positive_weight').value,
            'enable_global_planner': g('enable_global_planner').value,
            'dead_end_epsilon': g('dead_end_epsilon').value,
            'dead_end_initial_threshold': g('dead_end_initial_threshold').value,
            'switch_back_threshold': g('switch_back_threshold').value,
            'prm_samples': g('prm_samples').value,
            'prm_connection_radius': g('prm_connection_radius').value,
            'frontier_min_size': g('frontier_min_size').value,
            'lambda_p': g('lambda_p').value,
            'lambda_s': g('lambda_s').value,
            'xy_goal_tolerance': g('xy_goal_tolerance').value,
            'robot_radius': g('robot_radius').value,
            'true_source_x': g('true_source_x').value,
            'true_source_y': g('true_source_y').value,
        }

    def _init_state(self):
        self.sensor_raw_value: Optional[float] = None
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_theta: Optional[float] = None
        self.sensor_initialized = False
        self.node_initialized = False
        self.search_complete = False
        self.planning_pending = False

        self.step_count = 0
        self.measurement_count = 0
        self.source_updates = 0

        self.planner_mode = 'LOCAL'
        self.global_path: List[Tuple[float, float]] = []
        self.global_path_index = 0
        self.settling_start_time = None

        self.laser_scan_count = 0
        self.total_travel_distance = 0.0
        self.previous_position = None
        self.computation_times = []

        self.start_time = None  # set after init
        self.last_sensor_ppm = 0.0
        self.last_is_hit = False
        self.last_debug_info = {}
        self.last_bi_optimal = 0.0

    def _apply_scenario_overrides(self):
        """If `scenario` param is set, load map/wind/source from test_env.

        The scenario's native cell size is used UNLESS the user explicitly
        set `cell_size_m` to a non-default value — a cheap override that
        lets us run at 0.2 m on a 0.1 m scenario (4× faster filament sim)
        without editing the scenario files.
        """
        scen = self.P.get('scenario', '') or ''
        if not scen:
            return
        try:
            info = load_scenario(scen)
        except Exception as e:
            self.get_logger().error(f'[SCENARIO] failed to load "{scen}": {e}')
            raise
        self.P['map_origin_x'] = info.origin_x
        self.P['map_origin_y'] = info.origin_y
        self.P['map_width_m'] = info.width_m
        self.P['map_height_m'] = info.height_m
        # By default inherit the scenario's native cell size. Set the ROS
        # param `override_cell_size_m > 0` to force a different resolution
        # (e.g. 0.2 on a 0.1-m scenario for a 4× filament-sim speedup).
        override = self.P.get('override_cell_size_m', 0.0)
        self.P['cell_size_m'] = float(override) if override and override > 0 else info.cell_size_m
        self.P['wind_u_mean'] = info.wind_u_mean
        self.P['wind_v_mean'] = info.wind_v_mean
        if info.source_x is not None:
            self.P['true_source_x'] = info.source_x
            self.P['true_source_y'] = info.source_y
        tag = f'@{info.wind_tag}' if info.wind_tag else ''
        self.get_logger().info(
            f'[SCENARIO] "{info.scenario}/{info.sim}{tag}" → '
            f'map [{info.origin_x:.2f},{info.origin_y:.2f}]+'
            f'{info.width_m:.1f}×{info.height_m:.1f}m @ {info.cell_size_m:.2f}m, '
            f'wind=({info.wind_u_mean:+.3f},{info.wind_v_mean:+.3f}) m/s '
            f'(|w|={info.wind_speed_mean:.3f}), '
            f'true_source=({info.source_x},{info.source_y}), '
            f'wind_csv={info.wind_csv.name}'
        )

    def _init_components(self):
        self._apply_scenario_overrides()

        # 1. Empty SLAM grid from known dimensions
        self.slam_map = create_empty_occupancy_map_from_dims(
            width_m=self.P['map_width_m'],
            height_m=self.P['map_height_m'],
            cell_size_m=self.P['cell_size_m'],
            origin_x=self.P['map_origin_x'],
            origin_y=self.P['map_origin_y'],
        )

        # 2. Constant wind
        self.wind = ConstantWindField(
            grid_shape=(self.slam_map.height, self.slam_map.width),
            u_mean=self.P['wind_u_mean'],
            v_mean=self.P['wind_v_mean'],
        )

        # 3. Hit map
        hit_settings = HitMapSettings(
            prior=self.P['hit_prior'],
            kernel_sigma=self.P['hit_kernel_sigma'],
            kernel_stretch_constant=self.P['hit_kernel_stretch'],
            local_estimation_window=int(self.P['hit_local_window']),
            confidence_sigma_spatial=self.P['confidence_sigma_spatial'],
            confidence_measurement_weight=self.P['confidence_measurement_weight'],
        )
        self.hit_map = HitMap(
            grid_shape=(self.slam_map.height, self.slam_map.width),
            settings=hit_settings,
            wind_speed=self.wind.speed,
            wind_downwind_angle=self.wind.downwind_angle,
        )

        # 4. Filament simulator
        fil_settings = FilamentSettings(
            noise_stddev=self.P['filament_noise_stddev'],
            delta_time=self.P['filament_dt'],
            min_warmup_iterations=int(self.P['filament_min_warmup']),
            max_warmup_iterations=int(self.P['filament_max_warmup']),
            iterations_to_record=int(self.P['filament_record_steps']),
            filaments_per_step=int(self.P['filament_per_step']),
        )
        self.filament_sim = FilamentSimulator(self.slam_map, self.wind, fil_settings)

        # 5. Source belief
        self.source_belief = SourceBelief(
            grid_shape=(self.slam_map.height, self.slam_map.width),
            stride=int(self.P['candidate_stride']),
            source_discrimination_power=self.P['source_discrimination_power'],
        )

        # 6. Planners
        self.rrt = RRT(
            occupancy_grid=self.slam_map,
            N_tn=int(self.P['n_tn']),
            R_range=self.P['n_tn'] * self.P['delta'],
            delta=self.P['delta'],
            max_depth=int(self.P['max_depth']),
            discount_factor=self.P['positive_weight'],
            robot_radius=self.P['robot_radius'],
        )
        self.global_planner = GlobalPlanner(
            occupancy_grid=self.slam_map,
            robot_radius=self.P['robot_radius'],
            prm_samples=int(self.P['prm_samples']),
            prm_connection_radius=self.P['prm_connection_radius'],
            frontier_min_size=int(self.P['frontier_min_size']),
            lambda_p=self.P['lambda_p'],
            lambda_s=self.P['lambda_s'],
        )
        self.dead_end_detector = DeadEndDetector(
            epsilon=self.P['dead_end_epsilon'],
            initial_threshold=self.P['dead_end_initial_threshold'],
        )

        # 7. SLAM updater + visualizers + logger
        self.lidar_mapper = LidarMapper(self.slam_map, outlet_mask=None)
        self.marker_viz = MarkerVisualizer(self, self.slam_map)
        self.logger = ExperimentLogger()
        self.get_logger().info(f'Data logging to: {self.logger.log_filename}')

        # 8. Navigator — requires Nav2 action server
        self.navigator = Navigator(self, on_complete_callback=self._on_navigation_complete)

    def _init_ros_interfaces(self):
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/PioneerP3DX/ground_truth',
            self.pose_callback, 10,
        )
        self.sensor_sub = self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self.sensor_callback, 10,
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/PioneerP3DX/laser_scanner', self.laser_callback, 10,
        )
        self.wind_sub = self.create_subscription(
            Anemometer, '/fake_anemometer/WindSensor_reading',
            self.wind_callback, 10,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)

        # Periodic viz pushes
        self.viz_timer = self.create_timer(1.0, self._publish_overlays)
        self.start_time = self.get_clock().now()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = atan2(siny_cosp, cosy_cosp)

        if self.previous_position is not None:
            self.total_travel_distance += float(np.hypot(
                x - self.previous_position[0], y - self.previous_position[1]
            ))

        self.current_position = (x, y)
        self.current_theta = theta
        self.previous_position = (x, y)

        if self.planning_pending and not self.navigator.is_moving:
            self.planning_pending = False
            self.take_step()

        if (self.node_initialized and not self.navigator.initial_spin_done
                and not self.navigator.is_moving and self.sensor_raw_value is not None
                and not self.planning_pending):
            self.take_step()

    def sensor_callback(self, msg: GasSensor):
        self.sensor_raw_value = msg.raw
        if (self.node_initialized and not self.navigator.initial_spin_done
                and not self.navigator.is_moving and self.current_position is not None
                and not self.planning_pending):
            self.take_step()

    def laser_callback(self, msg: LaserScan):
        if not self.node_initialized or self.current_position is None or self.current_theta is None:
            return
        self.lidar_mapper.update_from_scan(
            msg, self.current_position[0], self.current_position[1], self.current_theta
        )
        self.laser_scan_count += 1

    def wind_callback(self, msg: Anemometer):
        # efe_pmfs uses a known mean wind; anemometer is ignored by the algorithm.
        # Kept as a subscription so the launch wiring mirrors efe_igdm.
        pass

    def _on_navigation_complete(self):
        self.planning_pending = True

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def take_step(self):
        if self.search_complete or self.sensor_raw_value is None or self.current_position is None:
            return
        step_start = time.time()
        if not self.sensor_initialized:
            self.sensor_initialized = True

        if not self.navigator.initial_spin_done:
            self.navigator.perform_initial_spin(self.current_position, self.current_theta)
            return
        if self.navigator.is_moving:
            return
        if self.settling_start_time is not None:
            self._handle_settling_complete()
            return

        # 1. Fold gas sample into hit map (binary hit/miss)
        is_hit = self.sensor_raw_value > self.P['hit_threshold_ppm']
        gx, gy = self.slam_map.world_to_grid(*self.current_position)
        self.hit_map.update(self.slam_map.grid, (gx, gy), is_hit)
        self.measurement_count += 1
        self.last_sensor_ppm = float(self.sensor_raw_value)
        self.last_is_hit = bool(is_hit)

        # 2. Periodically refresh source belief (filament sims are expensive)
        every_n = max(int(self.P['source_update_every_n_stops']), 1)
        if self.measurement_count % every_n == 0:
            t0 = time.time()
            self.source_belief.refresh(
                self.slam_map.grid, self.filament_sim,
                self.hit_map.hit_prob, self.hit_map.confidence,
                log=lambda s: self.get_logger().info(s),
            )
            self.source_updates += 1
            self.get_logger().info(
                f'[SourceBelief] refresh #{self.source_updates} '
                f'took {time.time()-t0:.2f}s'
            )

        # 3. Planning
        next_pos = None
        debug_info = {}
        dead_end = False
        bi_optimal = 0.0

        if self.planner_mode == 'GLOBAL':
            next_pos, should_return = self._run_global_planning()
            if should_return:
                return
        else:  # LOCAL
            next_pos, debug_info, dead_end, bi_optimal = self._run_local_planning()

        self.last_debug_info = debug_info
        self.last_bi_optimal = bi_optimal

        # 4. Visualization + log
        self._update_visualizations(debug_info, bi_optimal, dead_end)
        step_time = time.time() - step_start
        self.computation_times.append(step_time)
        self._log_step(bi_optimal, dead_end, step_time)

        # 5. Convergence
        if self._check_convergence():
            return

        # 6. Execute
        if next_pos is not None:
            # Skip sending a zero-distance goal — it returns SUCCEEDED instantly
            # and re-enters the planner, creating a livelock. Trigger global
            # planning directly instead.
            cur_cell = self.slam_map.world_to_grid(*self.current_position)
            next_cell = self.slam_map.world_to_grid(*next_pos)
            if cur_cell == next_cell:
                self.get_logger().warn(
                    '[STUCK] next_pos == current cell — forcing GLOBAL replan'
                )
                self._handle_dead_end_transition()
                if self.planner_mode == 'GLOBAL' and self.global_path:
                    next_pos = self.global_path[self.global_path_index]
                else:
                    self.planning_pending = True
                    return
            self.get_logger().info(f'→ moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
            self.navigator.send_goal(next_pos[0], next_pos[1],
                                     tolerance=self.P['xy_goal_tolerance'])

    # ------------------------------------------------------------------
    # Planning submodes
    # ------------------------------------------------------------------
    def _run_local_planning(self):
        debug_info = self.rrt.get_next_move_debug(self.current_position, self.source_belief)
        next_pos = debug_info['next_position']

        # Cell-level stuck detection: if the planner wants to stay in the same
        # cell, count it as a wasted step no matter how the metric distance
        # looks with odometry jitter.
        cur_cell = self.slam_map.world_to_grid(*self.current_position)
        next_cell = self.slam_map.world_to_grid(*next_pos)
        move_dist = np.hypot(next_pos[0] - self.current_position[0],
                              next_pos[1] - self.current_position[1])
        if cur_cell == next_cell or move_dist < 0.05:
            self.navigator.consecutive_failures += 1
            self.get_logger().warn(
                f'[STUCK] planner picked same cell {next_cell} '
                f'(streak {self.navigator.consecutive_failures}/'
                f'{self.navigator.max_failures_tolerance})'
            )
            if self.navigator.consecutive_failures >= self.navigator.max_failures_tolerance:
                self._trigger_recovery()
                return None, debug_info, False, 0.0
        else:
            if self.navigator.consecutive_failures > 0:
                self.navigator.consecutive_failures -= 1

        bi_optimal = debug_info.get('best_utility', 0.0)
        dead_end = False
        if self.P['enable_global_planner']:
            dead_end = self.dead_end_detector.is_dead_end(bi_optimal)
            if dead_end:
                self._handle_dead_end_transition()
                if self.planner_mode == 'GLOBAL' and self.global_path:
                    next_pos = self.global_path[self.global_path_index]
        return next_pos, debug_info, dead_end, bi_optimal

    def _run_global_planning(self):
        if not self.global_path or self.global_path_index >= len(self.global_path):
            self.get_logger().warn('[GLOBAL] path exhausted → settle')
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        while self.global_path_index < len(self.global_path):
            wp = self.global_path[self.global_path_index]
            d = np.hypot(wp[0] - self.current_position[0], wp[1] - self.current_position[1])
            if d < self.P['xy_goal_tolerance']:
                self.global_path_index += 1
            else:
                break
        if self.global_path_index >= len(self.global_path):
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        waypoint = self.global_path[self.global_path_index]
        gx, gy = self.slam_map.world_to_grid(*waypoint)
        mi = self.source_belief.mutual_information_at((gx, gy))
        self.dead_end_detector.update_threshold(max(mi, 0.0))
        thresh = self.P['switch_back_threshold'] * self.dead_end_detector.get_status()['bi_threshold']
        if mi > thresh:
            self.get_logger().info(f'[SWITCH→LOCAL] MI recovered at waypoint ({mi:.4f}).')
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        self.marker_viz.visualize_global_path(self.global_path)
        return waypoint, False

    def _handle_dead_end_transition(self):
        frontier_cells = self.global_planner.detect_frontiers()
        if not frontier_cells:
            self.get_logger().info('[DEAD END] no frontiers — stay LOCAL.')
            return
        self.get_logger().warn(f'[DEAD END] {len(frontier_cells)} frontier cells → planning GLOBAL.')
        result = self.global_planner.plan(self.current_position, self.source_belief)
        if result.get('success'):
            self.planner_mode = 'GLOBAL'
            self.global_path = result['best_global_path']
            self.global_path_index = 1
            self.marker_viz.visualize_frontier_cells(result['frontier_cells'])
            self.marker_viz.visualize_frontier_centroids(result['frontier_clusters'])
            self.marker_viz.visualize_prm_graph(result['prm_vertices'], self.global_planner.vertex_dict)
            self.marker_viz.visualize_global_path(self.global_path)
        else:
            self.get_logger().info('[DEAD END] global plan failed — stay LOCAL.')

    def _handle_settling_complete(self):
        self.get_logger().info('[MODE] settle complete → LOCAL')
        self.settling_start_time = None
        # Fold the current sample once (robot just stopped)
        is_hit = self.sensor_raw_value > self.P['hit_threshold_ppm']
        gx, gy = self.slam_map.world_to_grid(*self.current_position)
        self.hit_map.update(self.slam_map.grid, (gx, gy), is_hit)
        self.measurement_count += 1
        self.planner_mode = 'LOCAL'
        self.global_path = []
        self.global_path_index = 0
        self.planning_pending = True

    def _trigger_recovery(self):
        success = self.navigator.attempt_teleport_recovery(
            self.current_position, self.slam_map, self.dead_end_detector
        )
        if success:
            self.current_position = None
            return
        self.get_logger().warn('Teleport failed, trying global plan fallback.')
        self.planner_mode = 'GLOBAL'
        res = self.global_planner.plan(self.current_position, self.source_belief)
        if res.get('success'):
            self.global_path = res['best_global_path']
            self.global_path_index = 1
            self.marker_viz.visualize_global_path(self.global_path)
            self.planning_pending = True
        else:
            self.get_logger().error('Global recovery failed — stay LOCAL.')
            self.planner_mode = 'LOCAL'
            self.planning_pending = True

    # ------------------------------------------------------------------
    # Convergence + declaration
    # ------------------------------------------------------------------
    def _check_convergence(self):
        std_x, std_y = self.source_belief.std_world(self.slam_map.grid_to_world)
        sigma_p = max(std_x, std_y)
        if self.source_updates >= 1 and sigma_p < self.P['declaration_std_threshold']:
            self.get_logger().info(f'[DECLARE] σ_p = {sigma_p:.3f} < threshold')
            self._finish_and_summarize()
            return True
        return False

    def _finish_and_summarize(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        est_x, est_y = self.source_belief.estimate_world(self.slam_map.grid_to_world)
        err = -1.0
        if self.P['true_source_x'] != -999.0:
            err = float(np.hypot(est_x - self.P['true_source_x'],
                                  est_y - self.P['true_source_y']))
        avg = float(np.mean(self.computation_times)) if self.computation_times else 0.0
        self.logger.save_summary(
            self.step_count, self.total_travel_distance, elapsed, avg,
            est_x, est_y, err,
        )
        self.get_logger().info(
            f'Done. steps={self.step_count}, est=({est_x:.2f},{est_y:.2f}), err={err:.3f}m'
        )
        self.search_complete = True

    # ------------------------------------------------------------------
    # Visualization + logging
    # ------------------------------------------------------------------
    def _update_visualizations(self, debug_info, bi_optimal, dead_end):
        self.marker_viz.visualize_planner_mode(self.planner_mode)
        est_x, est_y = self.source_belief.estimate_world(self.slam_map.grid_to_world)
        self.marker_viz.visualize_estimated_source(est_x, est_y)
        self.marker_viz.visualize_current_position(self.current_position)
        self.marker_viz.publish_source_belief(
            self.source_belief.candidate_cells,
            self.source_belief.p_S,
            self.source_belief.candidate_valid,
        )
        if self.planner_mode == 'LOCAL' and debug_info:
            self.marker_viz.visualize_all_paths(
                debug_info.get('all_paths', []),
                debug_info.get('all_utilities', None),
            )
            self.marker_viz.visualize_best_path(debug_info.get('best_path', []))

    def _publish_overlays(self):
        if not self.node_initialized:
            return
        self.marker_viz.publish_slam_map()
        self.marker_viz.publish_hit_prob(self.hit_map.hit_prob)
        self.marker_viz.publish_confidence(self.hit_map.confidence)
        self.marker_viz.visualize_wind_arrow(self.P['wind_u_mean'], self.P['wind_v_mean'])

    def _log_step(self, bi_optimal, dead_end, step_time):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        est_x, est_y = self.source_belief.estimate_world(self.slam_map.grid_to_world)
        std_x, std_y = self.source_belief.std_world(self.slam_map.grid_to_world)
        entropy = self.source_belief.entropy()
        bi_threshold = self.dead_end_detector.get_status()['bi_threshold']
        gpl = len(self.global_path) if self.planner_mode == 'GLOBAL' else 0
        gpi = self.global_path_index if self.planner_mode == 'GLOBAL' else 0
        self.logger.log_step(
            step=self.step_count, elapsed=elapsed, mode=self.planner_mode,
            sensor_ppm=self.last_sensor_ppm, is_hit=self.last_is_hit,
            robot_pos=self.current_position, est_pos=(est_x, est_y),
            est_std=(std_x, std_y), entropy=entropy,
            bi_optimal=bi_optimal, bi_threshold=bi_threshold, dead_end=dead_end,
            num_branches=self.last_debug_info.get('num_branches', 0),
            global_path_len=gpl, global_path_idx=gpi,
            source_updates=self.source_updates, step_time_s=step_time,
        )
        self.step_count += 1

    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.close()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EfePMFSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
