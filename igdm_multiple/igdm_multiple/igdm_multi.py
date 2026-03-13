"""
Multi-Source Gas Source Localization Node.

Extends the basic RRT-Infotaxis node to handle multiple gas sources using:
- Multi-layer particle filter with peak suppression (PSPF)
- OIC/RSC/REC exploration corrections (ADE-PSPF)
- Source verification against measurement history

Based on:
- Kim et al., IEEE RA-L 2025 (base architecture)
- Gao et al., Sensors 2018 (PSPF)
- Bai et al., RAS 2023 (ADE-PSPF)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# Messages
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

# Custom Modules
from .mapping.occupancy_grid import load_3d_occupancy_grid_from_service, create_empty_occupancy_map, OccupancyGridMap
from .estimation.sensor_model import ContinuousGaussianSensorModel
from .estimation.igdm_gas_model import IndoorGaussianDispersionModel
from .estimation.multi_layer_pf import MultiLayerParticleFilter
from .estimation.source_verifier import SourceVerifier
from .planning.rrt_multi import MultiSourceRRT
from .planning.global_planner import GlobalPlanner
from .visualization.text_visualizer import TextVisualizer
from .planning.dead_end_detector import DeadEndDetector

# Helper modules
from .visualization.marker_visualizer import MarkerVisualizer
from .planning.navigator import Navigator
from .mapping.lidar_mapper import LidarMapper
from .utils.experiment_logger import ExperimentLogger

import numpy as np
import time
from typing import Tuple, List, Optional


class MultiSourceNode(Node):
    """
    Multi-source gas source localization using multi-layer particle filter.
    """

    def __init__(self):
        super().__init__('multi_source_node')

        self._init_parameters()
        self._init_state_variables()
        self._setup_data_logging()
        self._init_ros_interfaces()
        self._init_models_and_planners()

        self.node_initialized = True
        self.get_logger().info(
            f'Multi-Source Node initialized ({self.params["num_layers"]} layers, '
            f'{self.params["particles_per_layer"]} particles/layer)'
        )

    def _init_parameters(self):
        """Declare and cache parameters."""
        # Base parameters (same as basic node)
        self.declare_parameter('sigma_m', 1.5)
        self.declare_parameter('n_tn', 50)
        self.declare_parameter('delta', 0.7)
        self.declare_parameter('max_depth', 4)
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('robot_radius', 0.25)
        self.declare_parameter('sigma_threshold', 0.5)
        self.declare_parameter('stop_and_measure_time', 0.5)
        self.declare_parameter('positive_weight', 0.5)
        self.declare_parameter('dead_end_epsilon', 0.6)
        self.declare_parameter('dead_end_initial_threshold', 0.1)
        self.declare_parameter('enable_global_planner', True)
        self.declare_parameter('prm_samples', 300)
        self.declare_parameter('prm_connection_radius', 5.0)
        self.declare_parameter('frontier_min_size', 3)
        self.declare_parameter('lambda_p', 0.1)
        self.declare_parameter('lambda_s', 0.05)
        self.declare_parameter('switch_back_threshold', 1.5)
        self.declare_parameter('resample_threshold', 0.5)
        self.declare_parameter('sensor_alpha', 0.1)
        self.declare_parameter('sensor_sigma_env', 1.5)
        self.declare_parameter('sensor_num_levels', 10)
        self.declare_parameter('max_concentration', 120.0)

        # Multi-source specific parameters
        self.declare_parameter('num_layers', 4)
        self.declare_parameter('particles_per_layer', 500)
        self.declare_parameter('peak_suppression_radius', 1.0)
        self.declare_parameter('verification_threshold', 5.0)
        self.declare_parameter('oic_beta', 0.5)
        self.declare_parameter('rsc_radius', 1.0)
        self.declare_parameter('rec_radius', 2.0)

        # True source positions for evaluation (comma-separated)
        self.declare_parameter('true_source_x', '2.0')
        self.declare_parameter('true_source_y', '4.5')

        # Cache values
        self.params = {
            'sigma_m': self.get_parameter('sigma_m').value,
            'sigma_threshold': self.get_parameter('sigma_threshold').value,
            'xy_goal_tolerance': self.get_parameter('xy_goal_tolerance').value,
            'robot_radius': self.get_parameter('robot_radius').value,
            'dead_end_initial_threshold': self.get_parameter('dead_end_initial_threshold').value,
            'enable_global_planner': self.get_parameter('enable_global_planner').value,
            'switch_back_threshold': self.get_parameter('switch_back_threshold').value,
            'n_tn': self.get_parameter('n_tn').value,
            'delta': self.get_parameter('delta').value,
            'max_depth': self.get_parameter('max_depth').value,
            'positive_weight': self.get_parameter('positive_weight').value,
            'sensor_alpha': self.get_parameter('sensor_alpha').value,
            'sensor_sigma_env': self.get_parameter('sensor_sigma_env').value,
            'sensor_num_levels': self.get_parameter('sensor_num_levels').value,
            'max_concentration': self.get_parameter('max_concentration').value,
            # Multi-source
            'num_layers': self.get_parameter('num_layers').value,
            'particles_per_layer': self.get_parameter('particles_per_layer').value,
            'peak_suppression_radius': self.get_parameter('peak_suppression_radius').value,
            'verification_threshold': self.get_parameter('verification_threshold').value,
            'oic_beta': self.get_parameter('oic_beta').value,
            'rsc_radius': self.get_parameter('rsc_radius').value,
            'rec_radius': self.get_parameter('rec_radius').value,
        }

        # Parse true source positions (support multiple via comma-separated strings)
        true_x_str = self.get_parameter('true_source_x').value
        true_y_str = self.get_parameter('true_source_y').value
        self.true_sources = []
        try:
            xs = [float(x.strip()) for x in str(true_x_str).split(',')]
            ys = [float(y.strip()) for y in str(true_y_str).split(',')]
            for x, y in zip(xs, ys):
                self.true_sources.append((x, y))
        except (ValueError, AttributeError):
            self.true_sources = []

    def _init_state_variables(self):
        """Initialize internal state."""
        self.sensor_raw_value: Optional[float] = None
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_theta: Optional[float] = None
        self.sensor_initialized = False
        self.node_initialized = False
        self.step_count = 0
        self.search_complete = False
        self.planning_pending = False

        # Stop-and-measure
        self._measure_wait_start = None
        self._fresh_reading = False

        # Dual-mode planner state
        self.planner_mode = 'LOCAL'
        self.global_path: List[Tuple[float, float]] = []
        self.global_path_index = 0
        self.settling_start_time = None

        # SLAM / Map
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

        # Performance tracking
        self.total_travel_distance = 0.0
        self.previous_position = None
        self.computation_times = []

        # Multi-source: measurement positions history for RSC
        self.measurement_positions: List[Tuple[float, float]] = []

    def _setup_data_logging(self):
        self.logger = ExperimentLogger()
        self.get_logger().info(f'Data logging to: {self.logger.log_filename}')
        self.start_time = self.get_clock().now()

    def _init_ros_interfaces(self):
        # Subscriptions
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, '/PioneerP3DX/ground_truth', self.pose_callback, 10)
        self.sensor_subscription = self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self.sensor_callback, 10)
        self.laser_subscription = self.create_subscription(
            LaserScan, '/PioneerP3DX/laser_scanner', self.laser_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        # SLAM map publisher
        map_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.slam_map_pub = self.create_publisher(OccupancyGrid, '/rrt_infotaxis/slam_map', map_qos)
        self.slam_map_timer = self.create_timer(0.5, self._publish_slam_map_timer)

    def _init_models_and_planners(self):
        try:
            grid_2d, self.outlet_mask, params = load_3d_occupancy_grid_from_service(
                self, z_level=5, service_name='/gaden_environment/occupancyMap3D', timeout_sec=10.0
            )
            self.occupancy_map = OccupancyGridMap(grid_2d, params)

            if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
                self.occupancy_map.origin_x = -0.2
                self.occupancy_map.origin_y = -0.2

            self.slam_map = create_empty_occupancy_map(self.occupancy_map)
            self.get_logger().info(f'Outlet mask loaded: {int(np.sum(self.outlet_mask))} outlet cells')
        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        # Helper modules
        self.marker_viz = MarkerVisualizer(self, self.slam_map)
        self.navigator = Navigator(self, on_complete_callback=self._on_navigation_complete)
        self.lidar_mapper = LidarMapper(self.slam_map, outlet_mask=self.outlet_mask)
        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")

        # Multi-source specific publishers
        self.multi_source_pub = self.create_publisher(
            MarkerArray, '/rrt_infotaxis/source_estimates', 10
        )

        # Models
        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.params['sigma_m'],
            occupancy_grid=self.slam_map,
            wind_alpha=0.0
        )
        self.sensor_model = ContinuousGaussianSensorModel(
            alpha=self.params['sensor_alpha'],
            sigma_env=self.params['sensor_sigma_env'],
            num_levels=self.params['sensor_num_levels'],
            max_concentration=self.params['max_concentration']
        )

        # Multi-layer particle filter
        search_bounds = {
            "x": (0, self.slam_map.real_world_width),
            "y": (0, self.slam_map.real_world_height),
            "Q": (0, self.params['max_concentration'])
        }
        self.multi_pf = MultiLayerParticleFilter(
            num_layers=self.params['num_layers'],
            num_particles_per_layer=self.params['particles_per_layer'],
            search_bounds=search_bounds,
            sensor_model=self.sensor_model,
            dispersion_model=self.dispersion_model,
            peak_suppression_radius=self.params['peak_suppression_radius'],
            convergence_sigma=self.params['sigma_threshold'],
            verification_threshold=self.params['verification_threshold']
        )

        # Source verifier
        self.source_verifier = SourceVerifier(
            dispersion_model=self.dispersion_model,
            rmse_threshold=self.params['verification_threshold']
        )

        # Multi-source RRT
        self.rrt = MultiSourceRRT(
            occupancy_grid=self.slam_map,
            N_tn=self.params['n_tn'],
            R_range=self.params['n_tn'] * self.params['delta'],
            delta=self.params['delta'],
            max_depth=self.params['max_depth'],
            robot_radius=self.params['robot_radius'],
            positive_weight=self.params['positive_weight'],
            oic_beta=self.params['oic_beta'],
            rsc_radius=self.params['rsc_radius'],
            rec_radius=self.params['rec_radius']
        )

        self.dead_end_detector = DeadEndDetector(
            epsilon=self.get_parameter('dead_end_epsilon').value,
            initial_threshold=self.params['dead_end_initial_threshold']
        )

        self.global_planner = GlobalPlanner(
            occupancy_grid=self.slam_map,
            robot_radius=self.params['robot_radius'],
            prm_samples=self.get_parameter('prm_samples').value,
            prm_connection_radius=self.get_parameter('prm_connection_radius').value,
            frontier_min_size=self.get_parameter('frontier_min_size').value,
            lambda_p=self.get_parameter('lambda_p').value,
            lambda_s=self.get_parameter('lambda_s').value
        )

    def _on_navigation_complete(self):
        self.planning_pending = True

    # =========================================================================
    # CORE LOGIC
    # =========================================================================

    def take_step(self):
        if self.search_complete or self.sensor_raw_value is None or self.current_position is None:
            return

        step_start_time = time.time()

        if not self.sensor_initialized:
            self.sensor_initialized = True
            self.get_logger().info(f'Sensor initialized')

        if not self.navigator.initial_spin_done:
            self.navigator.perform_initial_spin(self.current_position, self.current_theta)
            return

        MIN_LASER_SCANS = 5
        if self.laser_scan_count < MIN_LASER_SCANS:
            return

        if self.navigator.is_moving:
            return

        if self.settling_start_time is not None:
            self._handle_settling_complete()
            return

        # Stop-and-measure
        measure_time = self.get_parameter('stop_and_measure_time').value
        if measure_time > 0:
            if self._measure_wait_start is None:
                self._measure_wait_start = self.get_clock().now()
                self._fresh_reading = False
                return
            elapsed = (self.get_clock().now() - self._measure_wait_start).nanoseconds / 1e9
            if elapsed < measure_time or not self._fresh_reading:
                return
            self._measure_wait_start = None

        self.get_logger().info(
            f'[STEP {self.step_count}] Pos: ({self.current_position[0]:.2f}, '
            f'{self.current_position[1]:.2f}) | Sensor: {self.sensor_raw_value:.4f}'
        )

        # Record measurement position for RSC
        self.measurement_positions.append(self.current_position)

        # Update multi-layer particle filter
        self.multi_pf.update(self.sensor_raw_value, self.current_position)

        # Check for newly converged layers
        self._check_layer_convergence()

        # Get summary for visualization
        all_estimates = self.multi_pf.get_all_estimates()
        confirmed = self.multi_pf.get_confirmed_sources()
        n_confirmed = len(confirmed)
        n_converged = sum(1 for e in all_estimates if e['converged'])
        n_sources = self.multi_pf.get_source_count_estimate()

        self.get_logger().info(
            f'  Sources: {n_sources} estimated, {n_confirmed} confirmed, '
            f'{n_converged} converged'
        )

        # Plan next move
        next_pos = None
        debug_info = {}
        dead_end_detected = False
        bi_optimal = 0.0

        if self.planner_mode == 'GLOBAL':
            next_pos, should_return = self._run_global_planning()
            if should_return:
                return
        elif self.planner_mode == 'LOCAL':
            next_pos, debug_info, dead_end_detected, bi_optimal = self._run_local_planning()

        # Visualization
        self._update_visualizations(all_estimates, debug_info, bi_optimal, dead_end_detected)
        self._log_step_data_multi(all_estimates, debug_info, bi_optimal, dead_end_detected)

        # Check convergence
        if self._check_convergence(all_estimates):
            return

        # Execute move
        if next_pos is not None:
            step_computation_time = time.time() - step_start_time
            self.computation_times.append(step_computation_time)
            self.get_logger().info(f'Moving to: ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
            self.navigator.send_goal(next_pos[0], next_pos[1],
                                     tolerance=self.params['xy_goal_tolerance'])

    def _check_layer_convergence(self):
        """Check and verify newly converged layers."""
        for layer in self.multi_pf.layers:
            if layer.converged and not layer.confirmed:
                est = layer.estimate
                verified = self.source_verifier.verify(
                    candidate=est,
                    confirmed_sources=self.multi_pf.confirmed_sources,
                    measurement_history=self.multi_pf.measurement_history,
                    min_distance=self.params['peak_suppression_radius']
                )
                if verified:
                    layer.confirmed = True
                    self.multi_pf.confirmed_sources.append(dict(est))
                    self.get_logger().info(
                        f'SOURCE CONFIRMED (layer {layer.id}): '
                        f'({est["x"]:.2f}, {est["y"]:.2f}), Q={est["Q"]:.1f}'
                    )

    # =========================================================================
    # PLANNING
    # =========================================================================

    def _handle_settling_complete(self):
        self.get_logger().info('[MODE SWITCH] Settling complete. Switching to LOCAL.')
        self.settling_start_time = None
        self.multi_pf.update(self.sensor_raw_value, self.current_position)
        self.planner_mode = 'LOCAL'
        self.global_path = []
        self.global_path_index = 0
        self.marker_viz.clear_global_planner_visualizations()
        self.dead_end_detector.reset()
        self.planning_pending = True

    def _run_global_planning(self) -> Tuple[Optional[Tuple[float, float]], bool]:
        self.get_logger().info('[GLOBAL MODE] Following frontier path...')

        if not self.global_path or self.global_path_index >= len(self.global_path):
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        while self.global_path_index < len(self.global_path):
            wp = self.global_path[self.global_path_index]
            dist = np.hypot(wp[0] - self.current_position[0], wp[1] - self.current_position[1])
            if dist < self.params['xy_goal_tolerance']:
                self.global_path_index += 1
            else:
                break

        if self.global_path_index >= len(self.global_path):
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        waypoint = self.global_path[self.global_path_index]

        if not self.is_path_to_waypoint_valid(waypoint):
            self.global_path_index += 1
            self.planning_pending = True
            return None, True

        # Use planning layer for entropy check
        planning_layer = self.multi_pf.get_best_layer_for_planning()
        if planning_layer is not None:
            pf = planning_layer.pf
            current_entropy = pf.get_entropy()
            expected_entropy = pf.compute_expected_entropy(waypoint)
            mutual_info = current_entropy - expected_entropy
            detector_status = self.dead_end_detector.get_status()
            thresh = self.params['switch_back_threshold'] * detector_status["bi_threshold"]

            if mutual_info > thresh:
                self.get_logger().info(f'[SWITCH] High MI ({mutual_info:.4f}). Settling.')
                self.settling_start_time = self.get_clock().now()
                self.planning_pending = True
                return None, True

        self.marker_viz.visualize_global_path(self.global_path)
        return waypoint, False

    def _run_local_planning(self) -> Tuple[Optional[Tuple[float, float]], dict, bool, float]:
        debug_info = self.rrt.get_next_move_multi(
            self.current_position,
            self.multi_pf,
            measurement_positions=self.measurement_positions
        )
        next_pos = debug_info["next_position"]

        move_dist = np.hypot(next_pos[0] - self.current_position[0],
                             next_pos[1] - self.current_position[1])

        if move_dist < 0.05:
            self.navigator.consecutive_failures += 1
            if self.navigator.consecutive_failures >= self.navigator.max_failures_tolerance:
                self.trigger_recovery()
                return None, debug_info, False, 0.0
        else:
            if self.navigator.consecutive_failures > 0:
                self.navigator.consecutive_failures -= 1

        bi_optimal = debug_info.get("best_utility", 0.0)
        dead_end_detected = False

        if self.params['enable_global_planner']:
            dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)
            if dead_end_detected:
                self._handle_dead_end_transition()
                if self.planner_mode == 'GLOBAL' and self.global_path:
                    next_pos = self.global_path[self.global_path_index]

        return next_pos, debug_info, dead_end_detected, bi_optimal

    def _handle_dead_end_transition(self):
        frontier_cells = self.global_planner.detect_frontiers()
        if not frontier_cells:
            self.get_logger().info('[DEAD END] No frontiers. Staying LOCAL.')
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
            return

        # Use planning layer for global planner
        planning_layer = self.multi_pf.get_best_layer_for_planning()
        pf = planning_layer.pf if planning_layer else self.multi_pf.layers[0].pf

        self.get_logger().warn(f'[DEAD END] Found {len(frontier_cells)} frontier cells.')
        result = self.global_planner.plan(self.current_position, pf)

        if result['success']:
            self.marker_viz.clear_global_planner_visualizations()
            self.planner_mode = 'GLOBAL'
            self.global_path = result['best_global_path']
            self.global_path_index = 1

            self.marker_viz.visualize_frontier_cells(result['frontier_cells'])
            self.marker_viz.visualize_frontier_centroids(result['frontier_clusters'])
            self.marker_viz.visualize_prm_graph(result['prm_vertices'],
                                                self.global_planner.vertex_dict)
            self.marker_viz.visualize_global_path(self.global_path)
        else:
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])

    def trigger_recovery(self):
        success = self.navigator.attempt_teleport_recovery(
            self.current_position, self.slam_map, self.dead_end_detector
        )
        if success:
            self.current_position = None
            return

        self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
        self.planner_mode = 'GLOBAL'

        planning_layer = self.multi_pf.get_best_layer_for_planning()
        pf = planning_layer.pf if planning_layer else self.multi_pf.layers[0].pf

        res = self.global_planner.plan(self.current_position, pf)
        if res['success']:
            self.global_path = res['best_global_path']
            self.global_path_index = 1
            self.marker_viz.clear_global_planner_visualizations()
            self.marker_viz.visualize_global_path(self.global_path)
            self.planning_pending = True
        else:
            self.planner_mode = 'LOCAL'
            self.planning_pending = True

    # =========================================================================
    # CONVERGENCE
    # =========================================================================

    def _check_convergence(self, all_estimates):
        """
        Multi-source convergence: all layers either confirmed or fully dispersed.
        """
        confirmed = self.multi_pf.get_confirmed_sources()
        if not confirmed:
            return False

        # Check if all non-confirmed layers are dispersed
        all_resolved = True
        for est in all_estimates:
            if est['confirmed']:
                continue
            if est['sigma_p'] < self.params['sigma_threshold'] * 3:
                # This layer is still somewhat concentrated — not resolved
                all_resolved = False
                break

        if all_resolved:
            self.get_logger().info(
                f'ALL SOURCES RESOLVED! {len(confirmed)} sources confirmed.'
            )
            self._save_summary_and_finish()
            return True

        return False

    def _save_summary_and_finish(self):
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        confirmed = self.multi_pf.get_confirmed_sources()

        summary_lines = [
            f'=== MULTI-SOURCE GSL SUMMARY ===',
            f'Total Steps: {self.step_count}',
            f'Total Distance: {self.total_travel_distance:.1f} m',
            f'Total Time: {elapsed_time:.1f} s',
            f'Sources Found: {len(confirmed)}',
        ]

        for i, src in enumerate(confirmed):
            summary_lines.append(
                f'  Source {i+1}: ({src["x"]:.2f}, {src["y"]:.2f}), Q={src["Q"]:.1f}'
            )

        # Compute errors against true sources if available
        if self.true_sources:
            summary_lines.append(f'True Sources: {len(self.true_sources)}')
            for i, (tx, ty) in enumerate(self.true_sources):
                # Find closest confirmed source
                if confirmed:
                    dists = [np.hypot(tx - s['x'], ty - s['y']) for s in confirmed]
                    min_dist = min(dists)
                    summary_lines.append(
                        f'  True {i+1}: ({tx:.2f}, {ty:.2f}), '
                        f'closest est error: {min_dist:.2f} m'
                    )

        summary = '\n'.join(summary_lines)
        self.get_logger().info('\n' + summary)

        # Save to file
        try:
            summary_path = self.logger.log_filename.replace('.csv', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)
        except Exception:
            pass

        self.search_complete = True

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        from math import atan2

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = atan2(siny_cosp, cosy_cosp)

        if self.previous_position is not None:
            dist = np.hypot(x - self.previous_position[0], y - self.previous_position[1])
            self.total_travel_distance += dist

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

    def sensor_callback(self, msg):
        self.sensor_raw_value = msg.raw
        if self._measure_wait_start is not None:
            self._fresh_reading = True
            if not self.navigator.is_moving:
                self.take_step()
            return
        if (self.node_initialized and not self.navigator.initial_spin_done
                and not self.navigator.is_moving and self.current_position
                and not self.planning_pending):
            self.take_step()

    def laser_callback(self, msg: LaserScan):
        if not self.node_initialized or self.current_position is None or self.current_theta is None:
            return

        obstacles_found = self.lidar_mapper.update_from_scan(
            msg, self.current_position[0], self.current_position[1], self.current_theta
        )
        self.laser_scan_count += 1
        self.total_obstacles_marked += obstacles_found

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def _update_visualizations(self, all_estimates, debug_info, bi_optimal, dead_end_detected):
        self.marker_viz.visualize_planner_mode(self.planner_mode)

        # Visualize particles from all layers with different colors
        self._visualize_multi_layer_particles()

        # Visualize estimated sources
        self._visualize_multi_source_estimates(all_estimates)

        self.marker_viz.visualize_current_position(self.current_position)

        if self.planner_mode == 'LOCAL' and debug_info:
            self.marker_viz.visualize_all_paths(
                debug_info.get("all_paths", []),
                debug_info.get("all_utilities", None)
            )
            self.marker_viz.visualize_best_path(debug_info.get("best_path", []))

        # Text info (use first layer for compatibility)
        planning_layer = self.multi_pf.get_best_layer_for_planning()
        if planning_layer:
            current_entropy = planning_layer.pf.get_entropy()
            est = planning_layer.estimate or planning_layer.pf.get_estimate()[0]
            std = planning_layer.std or planning_layer.pf.get_estimate()[1]
            sigma_p = max(std['x'], std['y'])
        else:
            current_entropy = 0.0
            est = {'x': 0, 'y': 0}
            sigma_p = 0.0

        confirmed = self.multi_pf.get_confirmed_sources()
        n_sources = self.multi_pf.get_source_count_estimate()

        bin_width = self.sensor_model.max_concentration / self.sensor_model.num_levels
        current_bin = min(int(self.sensor_raw_value / bin_width), self.sensor_model.num_levels - 1)

        self.text_visualizer.publish_source_info(
            timestamp=self.get_clock().now().to_msg(),
            predicted_x=est['x'], predicted_y=est['y'], predicted_z=0.5,
            std_dev=sigma_p, search_complete=self.search_complete,
            sensor_value=self.sensor_raw_value, binary_value=current_bin,
            max_concentration=self.sensor_model.max_concentration,
            num_levels=self.sensor_model.num_levels, threshold=0.0,
            num_branches=debug_info.get("num_branches", 0) if debug_info else 0,
            best_utility=debug_info.get("best_utility", 0.0) if debug_info else 0.0,
            best_entropy_gain=debug_info.get("best_entropy_gain", 0.0) if debug_info else 0.0,
            best_travel_cost=debug_info.get("best_travel_cost", 0.0) if debug_info else 0.0,
            num_tree_nodes=debug_info.get("num_tree_nodes", 0) if debug_info else 0,
            entropy=current_entropy, bi_optimal=bi_optimal,
            bi_threshold=self.dead_end_detector.get_status()["bi_threshold"],
            dead_end_detected=dead_end_detected
        )
        self.publish_slam_map()

    def _visualize_multi_layer_particles(self):
        """Visualize particles from all layers with distinct colors."""
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        # Layer colors: blue, red, green, yellow, magenta, cyan
        layer_colors = [
            (0.2, 0.4, 1.0),  # Blue
            (1.0, 0.2, 0.2),  # Red
            (0.2, 1.0, 0.2),  # Green
            (1.0, 1.0, 0.2),  # Yellow
            (1.0, 0.2, 1.0),  # Magenta
            (0.2, 1.0, 1.0),  # Cyan
        ]

        marker_array = MarkerArray()

        for layer in self.multi_pf.layers:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"particles_layer_{layer.id}"
            marker.id = layer.id
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = 0.08
            marker.pose.orientation.w = 1.0

            particles, weights = layer.pf.get_particles()
            color = layer_colors[layer.id % len(layer_colors)]

            norm_w = weights / weights.max() if weights.max() > 0 else weights

            for p_val, w in zip(particles, norm_w):
                p = Point()
                p.x, p.y, p.z = float(p_val[0]), float(p_val[1]), 0.5
                marker.points.append(p)
                c = ColorRGBA()
                intensity = 0.3 + 0.7 * float(w)
                c.r = float(color[0] * intensity)
                c.g = float(color[1] * intensity)
                c.b = float(color[2] * intensity)
                c.a = 0.6 if not layer.confirmed else 0.3
                marker.colors.append(c)

            marker_array.markers.append(marker)

        self.marker_viz.particle_pub.publish(marker_array)

    def _visualize_multi_source_estimates(self, all_estimates):
        """Visualize estimated source positions for all layers."""
        from visualization_msgs.msg import Marker, MarkerArray
        from std_msgs.msg import ColorRGBA

        marker_array = MarkerArray()
        for est in all_estimates:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "source_estimates"
            marker.id = est['layer_id']
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(est['estimate']['x'])
            marker.pose.position.y = float(est['estimate']['y'])
            marker.pose.position.z = 0.6

            if est['confirmed']:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.5
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red = confirmed
            elif est['converged']:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.4
                marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.9)  # Orange = converged
            else:
                marker.scale.x = marker.scale.y = marker.scale.z = 0.25
                marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)  # Gray = tracking

            marker_array.markers.append(marker)

        self.multi_source_pub.publish(marker_array)

    def _log_step_data_multi(self, all_estimates, debug_info, bi_optimal, dead_end_detected):
        """Log step data. Uses planning layer for compatibility with ExperimentLogger."""
        planning_layer = self.multi_pf.get_best_layer_for_planning()
        if planning_layer:
            pf = planning_layer.pf
        else:
            pf = self.multi_pf.layers[0].pf

        bi_threshold = self.dead_end_detector.get_status()["bi_threshold"]
        self.logger.log_step(
            self.step_count, pf, self.sensor_raw_value,
            self.current_position, self.params, debug_info,
            bi_optimal, bi_threshold, dead_end_detected,
            self.planner_mode,
            len(self.global_path) if self.planner_mode == 'GLOBAL' else 0,
            self.global_path_index if self.planner_mode == 'GLOBAL' else 0
        )
        self.step_count += 1

    # =========================================================================
    # HELPERS
    # =========================================================================

    def is_path_to_waypoint_valid(self, waypoint: tuple) -> bool:
        if self.current_position is None:
            return False
        return self._is_segment_valid(self.current_position, waypoint)

    def _is_valid_optimistic(self, position: tuple) -> bool:
        gx, gy = self.slam_map.world_to_grid(*position)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
            return False
        radius_cells = int(np.ceil(self.params['robot_radius'] / self.slam_map.resolution))
        radius_sq = radius_cells ** 2
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_sq:
                    continue
                check_gx, check_gy = gx + dx, gy + dy
                if 0 <= check_gx < self.slam_map.width and 0 <= check_gy < self.slam_map.height:
                    if self.slam_map.grid[check_gy, check_gx] > 0:
                        return False
        return True

    def _is_segment_valid(self, start: tuple, end: tuple) -> bool:
        pos1, pos2 = np.array(start), np.array(end)
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return self._is_valid_optimistic(tuple(pos1))
        num_samples = max(int(np.ceil(dist / (self.slam_map.resolution * 0.5))), 2)
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            if not self._is_valid_optimistic((sample_pos[0], sample_pos[1])):
                return False
        return True

    def publish_slam_map(self):
        if not hasattr(self, 'slam_map'):
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.slam_map.resolution
        msg.info.width = self.slam_map.width
        msg.info.height = self.slam_map.height
        msg.info.origin.position.x = self.slam_map.origin_x
        msg.info.origin.position.y = self.slam_map.origin_y
        msg.info.origin.orientation.w = 1.0
        grid_data = self.slam_map.grid.flatten()
        ros_grid = np.full_like(grid_data, -1, dtype=np.int8)
        ros_grid[grid_data == 0] = 0
        ros_grid[grid_data == 1] = 100
        ros_grid[grid_data == 2] = 50
        msg.data = ros_grid.tolist()
        self.slam_map_pub.publish(msg)

    def _publish_slam_map_timer(self):
        self.publish_slam_map()

    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.close()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MultiSourceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
