"""
RRT-Infotaxis Node for Gas Source Localization in GADEN.

Simplified LOCAL-only mode with:
- Discrete 5-level sensor model
- Updated RRT with exploration penalty
- Weighted movement strategy
- No global planner complexity
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
from .mapping.occupancy_grid import create_occupancy_map_from_service, create_empty_occupancy_map
from .estimation.sensor_model import ContinuousGaussianSensorModel
from .estimation.sensor_model_discrete import DiscreteSensorModel
from .estimation.particle_filter import ParticleFilter
from .estimation.igdm_gas_model import IndoorGaussianDispersionModel
from .planning.rrt_updated import RRTInfotaxisUpdated
from .visualization.text_visualizer import TextVisualizer
from .visualization.marker_visualizer import MarkerVisualizer
from .planning.navigator import Navigator
from .mapping.lidar_mapper import LidarMapper
from .utils.experiment_logger import ExperimentLogger

import numpy as np
import time
from typing import Tuple, Optional


class RRTInfotaxisNode(Node):
    """
    RRT Infotaxis node for GADEN (Simplified LOCAL-only mode).
    Uses discrete sensor and exploration penalty for improved performance.
    """

    def __init__(self):
        super().__init__('rrt_infotaxis_node')

        self._init_parameters()
        self._init_state_variables()
        self._setup_data_logging()
        self._init_maps()  # Initialize maps FIRST
        self._init_helper_modules()  # Initialize helpers BEFORE subscriptions
        self._init_models_and_planners()
        self._init_ros_interfaces()  # Initialize ROS interfaces LAST

        self.node_initialized = True
        sensor_type = "Discrete 5-level" if self.params['use_discrete_sensor'] else "Continuous Gaussian"
        self.get_logger().info(f'Node initialized successfully with {sensor_type} sensor')
        self.get_logger().info('Waiting for sensor and pose data...')

    def _init_parameters(self):
        """Declare and cache parameters."""
        # Core parameters
        self.declare_parameter('sigma_m', 1.5)
        self.declare_parameter('number_of_particles', 1000)
        self.declare_parameter('n_tn', 20)
        self.declare_parameter('delta', 1.0)
        self.declare_parameter('max_depth', 2)
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('robot_radius', 0.35)
        self.declare_parameter('sigma_threshold', 0.3)
        self.declare_parameter('success_distance', 0.5)
        self.declare_parameter('positive_weight', 0.6)
        self.declare_parameter('resample_threshold', 0.42)
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 5.0)

        # NEW: Sensor type selection
        self.declare_parameter('use_discrete_sensor', True)
        self.declare_parameter('sensor_alpha', 0.1)
        self.declare_parameter('sensor_sigma_env', 0.1)
        self.declare_parameter('sensor_threshold_weight', 0.5)

        # NEW: Coarse grid optimization
        self.declare_parameter('coarse_resolution', 0.5)

        # NEW: Exploration penalty
        self.declare_parameter('penalty_radius', 1.0)
        self.declare_parameter('penalty_max_steps', 5)

        # NEW: Weighted movement
        self.declare_parameter('weighted_utility_threshold', 0.5)

        # Map usage
        self.declare_parameter('use_slam', False)  # Use ground truth by default

        # Cache values
        self.params = {
            'sigma_m': self.get_parameter('sigma_m').value,
            'sigma_threshold': self.get_parameter('sigma_threshold').value,
            'xy_goal_tolerance': self.get_parameter('xy_goal_tolerance').value,
            'robot_radius': self.get_parameter('robot_radius').value,
            'true_source_x': self.get_parameter('true_source_x').value,
            'true_source_y': self.get_parameter('true_source_y').value,
            'n_tn': self.get_parameter('n_tn').value,
            'delta': self.get_parameter('delta').value,
            'max_depth': self.get_parameter('max_depth').value,
            'positive_weight': self.get_parameter('positive_weight').value,
            'number_of_particles': self.get_parameter('number_of_particles').value,
            'resample_threshold': self.get_parameter('resample_threshold').value,
            'use_discrete_sensor': self.get_parameter('use_discrete_sensor').value,
            'sensor_alpha': self.get_parameter('sensor_alpha').value,
            'sensor_sigma_env': self.get_parameter('sensor_sigma_env').value,
            'sensor_threshold_weight': self.get_parameter('sensor_threshold_weight').value,
            'coarse_resolution': self.get_parameter('coarse_resolution').value,
            'penalty_radius': self.get_parameter('penalty_radius').value,
            'penalty_max_steps': self.get_parameter('penalty_max_steps').value,
            'weighted_utility_threshold': self.get_parameter('weighted_utility_threshold').value,
            'use_slam': self.get_parameter('use_slam').value,
            'success_distance': self.get_parameter('success_distance').value,
        }

    def _init_state_variables(self):
        """Initialize internal state."""
        self.sensor_raw_value: Optional[float] = None
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_theta: Optional[float] = None
        self.sensor_initialized = False
        self.node_initialized = False
        self.step_count = 0
        self.search_complete = False

        # NEW: Visited positions for exploration penalty
        self.visited_positions = []  # List of (position, step) tuples

        # SLAM / Map logic
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

        # Performance tracking
        self.total_travel_distance = 0.0
        self.previous_position = None
        self.computation_times = []

    def _setup_data_logging(self):
        """Initialize Experiment Logger."""
        self.logger = ExperimentLogger()
        self.get_logger().info(f'Data logging to: {self.logger.log_filename}')
        self.start_time = self.get_clock().now()

    def _init_maps(self):
        """Load maps FIRST (before any other initialization)."""
        try:
            self.occupancy_map = create_occupancy_map_from_service(
                self, z_level=5, service_name='/gaden_environment/occupancyMap3D', timeout_sec=10.0
            )
            # FIX: GADEN origin correction
            if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
                self.occupancy_map.origin_x = -0.2
                self.occupancy_map.origin_y = -0.2

            if self.params['use_slam']:
                self.slam_map = create_empty_occupancy_map(self.occupancy_map)
                self.active_map = self.slam_map
                self.get_logger().info('Using SLAM map')
            else:
                self.active_map = self.occupancy_map
                self.slam_map = self.occupancy_map  # For compatibility
                self.get_logger().info('Using ground truth map directly')

        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

    def _init_helper_modules(self):
        """Initialize helper modules BEFORE ROS subscriptions."""
        self.marker_viz = MarkerVisualizer(self, self.active_map)
        self.navigator = Navigator(self, on_complete_callback=self._on_navigation_complete)
        self.lidar_mapper = LidarMapper(self.slam_map) if self.params['use_slam'] else None
        self.text_info_pub = None  # Will be set in _init_ros_interfaces
        self.text_visualizer = None  # Will be initialized after publisher is created

    def _init_models_and_planners(self):
        """Initialize estimation and planning components."""
        # 1. Initialize IGDM with coarse grid optimization
        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.params['sigma_m'],
            occupancy_grid=self.active_map,
            coarse_resolution=self.params['coarse_resolution']
        )

        # 2. Initialize sensor model (discrete or continuous)
        if self.params['use_discrete_sensor']:
            self.sensor_model = DiscreteSensorModel(
                alpha=self.params['sensor_alpha'],
                sigma_env=self.params['sensor_sigma_env'],
                threshold_weight=self.params['sensor_threshold_weight']
            )
            self.get_logger().info('Using Discrete 5-level sensor model')
        else:
            self.sensor_model = ContinuousGaussianSensorModel(
                alpha=0.1, sigma_env=1.5, num_levels=10, max_concentration=120.0
            )
            self.get_logger().info('Using Continuous Gaussian sensor model')

        # 3. Initialize particle filter
        self.particle_filter = ParticleFilter(
            num_particles=self.params['number_of_particles'],
            search_bounds={
                "x": (0, self.active_map.real_world_width),
                "y": (0, self.active_map.real_world_height),
                "Q": (0, 120.0)
            },
            sensor_model=self.sensor_model,
            dispersion_model=self.dispersion_model,
            resample_threshold=self.params['resample_threshold']
        )

        # 4. Initialize updated RRT with exploration penalty
        self.rrt = RRTInfotaxisUpdated(
            occupancy_grid=self.active_map,
            N_tn=self.params['n_tn'],
            R_range=self.params['n_tn'] * self.params['delta'],
            delta=self.params['delta'],
            max_depth=self.params['max_depth'],
            robot_radius=self.params['robot_radius'],
            positive_weight=self.params['positive_weight'],
            visited_positions=self.visited_positions,
            current_step=0,
            penalty_radius=self.params['penalty_radius']
        )
        self.get_logger().info(f'Initialized RRT: N_tn={self.params["n_tn"]}, delta={self.params["delta"]}, max_depth={self.params["max_depth"]}')

    def _init_ros_interfaces(self):
        """Initialize Publishers and Subscribers LAST."""
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        # Now initialize text visualizer with the publisher
        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")

        # SLAM map publisher
        map_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                            reliability=QoSReliabilityPolicy.RELIABLE)
        self.slam_map_pub = self.create_publisher(OccupancyGrid, '/rrt_infotaxis/slam_map', map_qos)
        self.slam_map_timer = self.create_timer(0.5, self._publish_slam_map_timer)

        # Subscriptions (LAST to avoid callbacks before initialization)
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, '/PioneerP3DX/ground_truth', self.pose_callback, 10)
        self.sensor_subscription = self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self.sensor_callback, 10)

        if self.params['use_slam']:
            self.laser_subscription = self.create_subscription(
                LaserScan, '/PioneerP3DX/laser_scanner', self.laser_callback, 10)

    def _on_navigation_complete(self):
        """Callback from Navigator when a move finishes."""
        # Trigger next planning step
        self.take_step()

    # =========================================================================
    # CORE LOGIC: Take Step
    # =========================================================================

    def take_step(self):
        """Main control loop iteration (Simplified LOCAL-only)."""
        if self.search_complete or self.sensor_raw_value is None or self.current_position is None:
            return

        step_start_time = time.time()

        # 0. Initialize sensor threshold on first measurement
        if not self.sensor_initialized:
            if self.params['use_discrete_sensor']:
                self.sensor_model.initialize_threshold(self.sensor_raw_value)
                self.get_logger().info(f'Sensor initialized with threshold: {self.sensor_model.threshold:.4f}')
            self.sensor_initialized = True
            return

        # 1. Handle initial spin
        if not self.navigator.initial_spin_done:
            self.navigator.perform_initial_spin(self.current_position, self.current_theta)
            return

        # 2. Block if moving
        if self.navigator.is_moving:
            return

        self.step_count += 1
        self.get_logger().info(f'[STEP {self.step_count}] Pos: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}) | Sensor: {self.sensor_raw_value:.4f}')

        # 3. MEASURE: Process sensor reading
        if self.params['use_discrete_sensor']:
            self.sensor_model.update_threshold(self.sensor_raw_value)
            discrete_measurement = self.sensor_model.get_discrete_measurement(self.sensor_raw_value)
            level_names = ["Very Low (0)", "Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
            self.get_logger().info(f'Discrete level: {discrete_measurement} ({level_names[discrete_measurement]})')
            measurement = discrete_measurement
        else:
            measurement = self.sensor_raw_value

        # 4. UPDATE: Particle filter
        self.particle_filter.update(measurement, self.current_position)
        current_means, current_stds = self.particle_filter.get_estimate()
        est_x, est_y = current_means["x"], current_means["y"]

        # Debug: Check effective sample size
        N_eff = self.particle_filter._effective_sample_size()
        weight_var = np.var(self.particle_filter.weights)
        self.get_logger().info(f'N_eff={N_eff:.1f}/{self.particle_filter.N}, weight_var={weight_var:.6f}')

        self.get_logger().info(f'Estimate: x={est_x:.2f}±{current_stds["x"]:.2f}, y={est_y:.2f}±{current_stds["y"]:.2f}, Q={current_means["Q"]:.2f}±{current_stds["Q"]:.2f}')

        # 5. CHECK CONVERGENCE (early check)
        sigma_p = max(current_stds['x'], current_stds['y'])
        dist_to_true = np.linalg.norm(
            np.array(self.current_position) -
            np.array([self.params['true_source_x'], self.params['true_source_y']])
        )

        self.get_logger().info(f'Convergence: sigma_p={sigma_p:.3f}, threshold={self.params["sigma_threshold"]:.3f}')
        self.get_logger().info(f'Distance to true source: {dist_to_true:.3f}m (threshold: {self.params["success_distance"]:.3f}m)')

        if dist_to_true < self.params['success_distance']:
            self.get_logger().info('✓ SUCCESS! Reached true source!')
            self._finalize_search()
            return

        if sigma_p < self.params['sigma_threshold']:
            self.get_logger().info('✓ CONVERGED! Estimation converged!')
            self._finalize_search()
            return

        # 6. PLAN: RRT with exploration penalty
        self.rrt.visited_positions = self.visited_positions
        self.rrt.current_step = self.step_count

        self.get_logger().info('[PLAN] Building RRT with exploration penalty...')
        debug_info = self.rrt.get_next_move_debug(self.current_position, self.particle_filter)

        # Log planning results
        self.get_logger().info(f'[PLAN] Best utility: {debug_info["best_utility"]:.4f}')
        self.get_logger().info(f'[PLAN] J1 (Info gain): {debug_info["best_information_gain"]:.4f} (norm), Original: {debug_info["best_information_gain_original"]:.4f}')
        self.get_logger().info(f'[PLAN] J2 (Travel cost): {debug_info["best_travel_cost"]:.4f} (norm), Original: {debug_info["best_travel_cost_original"]:.4f}')
        self.get_logger().info(f'[PLAN] Paths analyzed: {debug_info["total_paths"]}, With penalties: {debug_info["paths_with_penalties"]}')

        if debug_info['best_penalty_applied']:
            self.get_logger().info(f'[PLAN] ⚠️ PENALTY APPLIED to best path')

        # 7. MOVE: Weighted movement strategy
        next_pos, move_info = self._calculate_weighted_move(debug_info, self.current_position)

        if move_info['strategy'] == 'weighted':
            self.get_logger().info(f'[MOVE] ✓ WEIGHTED MOVEMENT ({move_info["num_paths"]} paths)')
        else:
            self.get_logger().info(f'[MOVE] Greedy selection: {move_info.get("reason", "")}')

        # 8. VISUALIZE & LOG
        self._update_visualizations(est_x, est_y, current_stds, debug_info, measurement)
        self._log_step_data(current_means, current_stds, debug_info, measurement)

        # 9. EXECUTE MOVE
        step_computation_time = time.time() - step_start_time
        self.computation_times.append(step_computation_time)

        self.get_logger().info(f'[MOVE] Moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
        self.navigator.send_goal(next_pos[0], next_pos[1], tolerance=self.params['xy_goal_tolerance'])

        # Track visited position for next step's penalty
        self.visited_positions.append((self.current_position, self.step_count))

    def _calculate_weighted_move(self, debug_info, current_pos):
        """Calculate weighted movement position based on top utility paths."""
        all_utilities = debug_info.get('all_utilities', [])
        path_metadata = debug_info.get('path_metadata', [])

        if not all_utilities or not path_metadata:
            return debug_info['next_position'], {'strategy': 'fallback', 'reason': 'no_utilities'}

        # Find paths with utility > threshold
        valid_paths = [(idx, utility) for idx, utility in enumerate(all_utilities)
                       if utility > self.params['weighted_utility_threshold']]
        valid_paths.sort(key=lambda x: x[1], reverse=True)

        # If we have at least 2 valid paths, use weighted strategy
        if len(valid_paths) >= 2:
            top_indices = [valid_paths[i][0] for i in range(2)]
            top_utilities = [valid_paths[i][1] for i in range(2)]

            # Extract path endpoints
            path_endpoints = []
            for idx in top_indices:
                path = path_metadata[idx]['path']
                if path and len(path) > 1:
                    endpoint = tuple(path[-1].position)
                    path_endpoints.append(endpoint)
                else:
                    return debug_info['next_position'], {'strategy': 'fallback', 'reason': 'invalid_path'}

            # Normalize utilities to get weights
            utility_sum = sum(top_utilities)
            weights = [u / utility_sum for u in top_utilities]

            # Calculate weighted direction using unit vectors
            dx_vectors = []
            dy_vectors = []
            distances = []

            for endpoint in path_endpoints:
                dx = endpoint[0] - current_pos[0]
                dy = endpoint[1] - current_pos[1]
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(dist)

                if dist > 0:
                    dx_vectors.append(dx / dist)
                    dy_vectors.append(dy / dist)
                else:
                    dx_vectors.append(0)
                    dy_vectors.append(0)

            # Calculate weighted direction
            weighted_dx = sum(dx_vectors[i] * weights[i] for i in range(len(weights)))
            weighted_dy = sum(dy_vectors[i] * weights[i] for i in range(len(weights)))

            # Calculate weighted step size
            weighted_step_size = sum(distances[i] * weights[i] for i in range(len(weights)))

            # Normalize the direction
            weighted_dir_norm = np.sqrt(weighted_dx**2 + weighted_dy**2)
            if weighted_dir_norm > 1e-6:
                weighted_dx /= weighted_dir_norm
                weighted_dy /= weighted_dir_norm

                # Calculate final position
                final_x = current_pos[0] + weighted_dx * weighted_step_size
                final_y = current_pos[1] + weighted_dy * weighted_step_size

                return (final_x, final_y), {
                    'strategy': 'weighted',
                    'num_paths': len(top_utilities),
                    'utilities': top_utilities,
                    'weights': weights,
                    'step_size': weighted_step_size
                }
            else:
                return debug_info['next_position'], {'strategy': 'fallback', 'reason': 'zero_direction'}
        else:
            return debug_info['next_position'], {
                'strategy': 'greedy',
                'reason': f'only_{len(valid_paths)}_paths_exceed_threshold',
                'valid_paths': len(valid_paths)
            }

    def _finalize_search(self):
        """Finalize search when converged or succeeded."""
        self.search_complete = True
        current_means, current_stds = self.particle_filter.get_estimate()

        error = np.sqrt(
            (current_means['x'] - self.params['true_source_x'])**2 +
            (current_means['y'] - self.params['true_source_y'])**2
        )

        self.get_logger().info('='*70)
        self.get_logger().info('SEARCH COMPLETE')
        self.get_logger().info(f'True source: ({self.params["true_source_x"]:.2f}, {self.params["true_source_y"]:.2f})')
        self.get_logger().info(f'Estimated: ({current_means["x"]:.2f}, {current_means["y"]:.2f})')
        self.get_logger().info(f'Localization error: {error:.3f}m')
        self.get_logger().info(f'Steps: {self.step_count}')
        self.get_logger().info('='*70)

        # Save summary
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        avg_comp = np.mean(self.computation_times) if self.computation_times else 0.0

        self.logger.save_summary(
            step_count=self.step_count,
            total_dist=self.total_travel_distance,
            elapsed_time=elapsed_time,
            avg_comp_time=avg_comp,
            est_x=current_means['x'],
            est_y=current_means['y'],
            est_error=error
        )

    # =========================================================================
    # VISUALIZATION & LOGGING
    # =========================================================================

    def _update_visualizations(self, est_x, est_y, current_stds, debug_info, measurement):
        """Update all visualizations."""
        # Visualize particles
        particles, weights = self.particle_filter.get_particles()
        self.marker_viz.visualize_particles(particles, weights)

        # Visualize estimates
        self.marker_viz.visualize_estimated_source(est_x, est_y)
        self.marker_viz.visualize_current_position(self.current_position)

        # Visualize RRT
        if 'rrt_nodes' in debug_info and debug_info['rrt_nodes']:
            all_paths = debug_info.get('all_paths', [])
            all_utilities = debug_info.get('all_utilities', [])
            self.marker_viz.visualize_all_paths(all_paths, all_utilities)

            # Find the best path by finding the index with max utility
            path_metadata = debug_info.get('path_metadata', [])
            if path_metadata and all_utilities:
                best_idx = max(range(len(all_utilities)), key=lambda i: all_utilities[i])
                best_path = path_metadata[best_idx].get('path', [])
                if best_path:
                    self.marker_viz.visualize_best_path(best_path)

        # Text visualization
        sigma_p = max(current_stds['x'], current_stds['y'])
        if self.params['use_discrete_sensor']:
            threshold_value = self.sensor_model.threshold if hasattr(self.sensor_model, 'threshold') else 0.0
            binary_value = self.sensor_model.get_discrete_measurement(self.sensor_raw_value)
            num_levels = 5  # Discrete sensor has 5 levels
        else:
            threshold_value = 0.0
            binary_value = 0
            num_levels = 10  # Continuous sensor discretized to 10 bins

        self.text_visualizer.publish_source_info(
            timestamp=self.get_clock().now().to_msg(),
            predicted_x=est_x,
            predicted_y=est_y,
            predicted_z=0.5,  # Fixed height for visualization
            std_dev=sigma_p,
            search_complete=self.search_complete,
            sensor_value=self.sensor_raw_value,
            binary_value=binary_value,
            threshold=threshold_value,
            max_concentration=120.0,
            num_levels=num_levels,
            entropy=self.particle_filter.get_entropy(),
            bi_optimal=debug_info.get('best_utility', 0.0),
            bi_threshold=0.0,
            dead_end_detected=False,
            num_branches=debug_info.get('total_paths', 0),
            best_utility=debug_info.get('best_utility', 0.0),
            best_entropy_gain=debug_info.get('best_information_gain', 0.0),
            best_travel_cost=debug_info.get('best_travel_cost', 0.0),
            num_tree_nodes=len(debug_info.get('rrt_nodes', []))
        )

    def _log_step_data(self, current_means, current_stds, debug_info, measurement):
        """Log step data to CSV."""
        # Adapt debug_info to match what logger expects
        adapted_debug_info = {
            'num_branches': debug_info.get('total_paths', 0),
            'best_utility': debug_info.get('best_utility', 0.0),
            'best_entropy_gain': debug_info.get('best_information_gain', 0.0),
            'best_travel_cost': debug_info.get('best_travel_cost', 0.0),
        }

        self.logger.log_step(
            step_count=self.step_count,
            particle_filter=self.particle_filter,
            sensor_value=self.sensor_raw_value,
            current_pos=self.current_position,
            params=self.params,
            debug_info=adapted_debug_info,
            bi_optimal=debug_info.get('best_utility', 0.0),
            dead_end_detected=False,
            planner_mode='LOCAL',
            global_path_len=0,
            global_path_index=0
        )

    # =========================================================================
    # ROS CALLBACKS
    # =========================================================================

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Process robot pose update."""
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_theta = np.arctan2(siny_cosp, cosy_cosp)

        # Track travel distance
        if self.previous_position is not None:
            dist = np.linalg.norm(np.array(self.current_position) - np.array(self.previous_position))
            self.total_travel_distance += dist
        self.previous_position = self.current_position

        # Trigger step if not moving
        if self.node_initialized and not self.navigator.is_moving and self.sensor_raw_value is not None:
            self.take_step()

    def sensor_callback(self, msg: GasSensor):
        """Process gas sensor measurement."""
        self.sensor_raw_value = msg.raw

    def laser_callback(self, msg: LaserScan):
        """Process LiDAR scan for SLAM."""
        if not self.params['use_slam'] or self.lidar_mapper is None:
            return

        if self.current_position is None or self.current_theta is None:
            return

        self.laser_scan_count += 1

        # Update SLAM map
        obstacles_found = self.lidar_mapper.update_from_scan(
            msg,
            self.current_position[0],
            self.current_position[1],
            self.current_theta
        )
        self.total_obstacles_marked += obstacles_found

        if self.laser_scan_count % 10 == 0:
            self.get_logger().info(
                f'SLAM Update #{self.laser_scan_count}: +{obstacles_found} obstacles (total: {self.total_obstacles_marked})'
            )

    def _publish_slam_map_timer(self):
        """Publish SLAM map periodically."""
        if self.params['use_slam']:
            msg = self.slam_map.to_ros_msg(frame_id='map', timestamp=self.get_clock().now().to_msg())
            self.slam_map_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = RRTInfotaxisNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
