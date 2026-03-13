import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# Messages
from olfaction_msgs.msg import GasSensor, Anemometer
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

# Custom Modules
from .mapping.occupancy_grid import create_occupancy_map_from_service, create_empty_occupancy_map, load_3d_occupancy_grid_from_service, OccupancyGridMap
from .estimation.sensor_model import ContinuousGaussianSensorModel
from .estimation.particle_filter import ParticleFilter
from .estimation.igdm_gas_model import IndoorGaussianDispersionModel
from .planning.rrt import RRT
from .planning.global_planner import GlobalPlanner
from .visualization.text_visualizer import TextVisualizer
from .planning.dead_end_detector import DeadEndDetector

# --- NEW MODULES ---
from .visualization.marker_visualizer import MarkerVisualizer
from .planning.navigator import Navigator
from .mapping.lidar_mapper import LidarMapper
from .mapping.wind_map import WindMap
from .utils.experiment_logger import ExperimentLogger

import numpy as np
import time
from typing import Tuple, List, Optional

class RRTInfotaxisNode(Node):
    """
    RRT Infotaxis node (Refactored Coordinator) - ADVANCED MODE with Wind Mapping.
    Delegates Visualization, Navigation, Mapping, and Logging to helper classes.
    Includes GMRF-based wind field estimation and wind-aware dispersion modeling.
    """

    def __init__(self):
        super().__init__('rrt_infotaxis_node')
        
        self._init_parameters()
        self._init_state_variables()
        self._setup_data_logging()
        self._init_ros_interfaces()
        self._init_models_and_planners()
        
        self.node_initialized = True
        self.get_logger().info('Advanced Node initialized successfully (with wind mapping), waiting for data...')

    def _init_parameters(self):
        """Declare and cache parameters."""
        self.declare_parameter('use_fast_rrt', True)
        self.declare_parameter('sigma_m', 1.5)
        self.declare_parameter('number_of_particles', 1000)
        self.declare_parameter('n_tn', 50)
        self.declare_parameter('delta', 0.7)
        self.declare_parameter('max_depth', 4)
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('robot_radius', 0.25)
        self.declare_parameter('sigma_threshold', 0.5)
        self.declare_parameter('success_distance', 0.5)
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
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 4.5)
        self.declare_parameter('wind_alpha', 0.5)
        self.declare_parameter('sensor_alpha', 0.1)
        self.declare_parameter('sensor_sigma_env', 1.5)
        self.declare_parameter('sensor_num_levels', 10)
        self.declare_parameter('max_concentration', 120.0)

        # Cache values
        self.params = {
            'sigma_m': self.get_parameter('sigma_m').value,
            'sigma_threshold': self.get_parameter('sigma_threshold').value,
            'xy_goal_tolerance': self.get_parameter('xy_goal_tolerance').value,
            'robot_radius': self.get_parameter('robot_radius').value,
            'dead_end_initial_threshold': self.get_parameter('dead_end_initial_threshold').value,
            'enable_global_planner': self.get_parameter('enable_global_planner').value,
            'switch_back_threshold': self.get_parameter('switch_back_threshold').value,
            'true_source_x': self.get_parameter('true_source_x').value,
            'true_source_y': self.get_parameter('true_source_y').value,
            'n_tn': self.get_parameter('n_tn').value,
            'delta': self.get_parameter('delta').value,
            'max_depth': self.get_parameter('max_depth').value,
            'positive_weight': self.get_parameter('positive_weight').value,
            'use_fast_rrt': self.get_parameter('use_fast_rrt').value,
            'number_of_particles': self.get_parameter('number_of_particles').value,
            'use_gmrf': True,  # Hardcoded enable for GMRF
            'wind_alpha': self.get_parameter('wind_alpha').value,
            'sensor_alpha': self.get_parameter('sensor_alpha').value,
            'sensor_sigma_env': self.get_parameter('sensor_sigma_env').value,
            'sensor_num_levels': self.get_parameter('sensor_num_levels').value,
            'max_concentration': self.get_parameter('max_concentration').value,
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
        self.planning_pending = False

        # Dual-mode planner state
        self.planner_mode = 'LOCAL'
        self.global_path: List[Tuple[float, float]] = []
        self.global_path_index = 0
        self.settling_start_time = None

        # SLAM / Map logic
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

        # Wind sensor state
        self.wind_speed: Optional[float] = None
        self.wind_direction: Optional[float] = None
        self.wind_x: float = 0.0  # Cartesian x-component
        self.wind_y: float = 0.0  # Cartesian y-component

        # Performance tracking
        self.total_travel_distance = 0.0
        self.previous_position = None
        self.computation_times = []

    def _setup_data_logging(self):
        """Initialize Experiment Logger."""
        self.logger = ExperimentLogger()
        self.get_logger().info(f'Data logging to: {self.logger.log_filename}')
        self.start_time = self.get_clock().now()

    def _init_ros_interfaces(self):
        """Initialize Publishers and Subscribers."""
        # Subscriptions
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, '/PioneerP3DX/ground_truth', self.pose_callback, 10)
        self.sensor_subscription = self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self.sensor_callback, 10)
        self.laser_subscription = self.create_subscription(
            LaserScan, '/PioneerP3DX/laser_scanner', self.laser_callback, 10)
        self.wind_subscription = self.create_subscription(
            Anemometer, '/fake_anemometer/WindSensor_reading', self.wind_callback, 10)

        # Ground Truth Wind Subscription
        self.gt_wind_sub = self.create_subscription(
            MarkerArray, '/wind_vectors', self.gt_wind_callback, 10)
        self.gt_wind_data = None

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        # SLAM map publisher
        map_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.slam_map_pub = self.create_publisher(OccupancyGrid, '/rrt_infotaxis/slam_map', map_qos)
        self.slam_map_timer = self.create_timer(0.5, self._publish_slam_map_timer)
        self.wind_map_timer = self.create_timer(5.0, self._publish_wind_map_timer)

    def _init_models_and_planners(self):
        """Load maps and initialize components."""
        try:
            grid_2d, self.outlet_mask, params = load_3d_occupancy_grid_from_service(
                self, z_level=5, service_name='/gaden_environment/occupancyMap3D', timeout_sec=10.0
            )
            self.occupancy_map = OccupancyGridMap(grid_2d, params)

            # FIX: GADEN origin correction
            if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
                self.occupancy_map.origin_x = -0.2
                self.occupancy_map.origin_y = -0.2

            self.slam_map = create_empty_occupancy_map(self.occupancy_map)
            self.get_logger().info(f'Outlet mask loaded: {int(np.sum(self.outlet_mask))} outlet cells')
        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        # 1. Initialize Helper Modules
        self.marker_viz = MarkerVisualizer(self, self.slam_map)
        self.navigator = Navigator(self, on_complete_callback=self._on_navigation_complete)
        self.lidar_mapper = LidarMapper(self.slam_map, outlet_mask=self.outlet_mask)
        self.wind_map = WindMap(
            width=self.slam_map.width,
            height=self.slam_map.height,
            resolution=self.slam_map.resolution,
            origin_x=self.slam_map.origin_x,
            origin_y=self.slam_map.origin_y
        )
        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")

        # 2. Initialize Models
        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.params['sigma_m'], occupancy_grid=self.slam_map,
            wind_alpha=self.params['wind_alpha']
        )
        self.sensor_model = ContinuousGaussianSensorModel(
            alpha=self.params['sensor_alpha'],
            sigma_env=self.params['sensor_sigma_env'],
            num_levels=self.params['sensor_num_levels'],
            max_concentration=self.params['max_concentration']
        )

        self.particle_filter = ParticleFilter(
            num_particles=self.params['number_of_particles'],
            search_bounds={"x": (0, self.slam_map.real_world_width), "y": (0, self.slam_map.real_world_height), "Q": (0, self.params['max_concentration'])},
            sensor_model=self.sensor_model,
            dispersion_model=self.dispersion_model
        )

        self.rrt = RRT(
            occupancy_grid=self.slam_map,
            N_tn=self.params['n_tn'],
            R_range=self.params['n_tn'] * self.params['delta'],
            delta=self.params['delta'],
            max_depth=self.params['max_depth'],
            robot_radius=self.params['robot_radius'],
            positive_weight=self.params['positive_weight']
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
        """Callback from Navigator when a move finishes."""
        self.planning_pending = True

    # =========================================================================
    # CORE LOGIC: Take Step
    # =========================================================================

    def take_step(self):
        """Main control loop iteration."""
        if self.search_complete or self.sensor_raw_value is None or self.current_position is None:
            return

        step_start_time = time.time()

        if not self.sensor_initialized:
            self.sensor_initialized = True
            self.get_logger().info(f'Sensor initialized (α={self.sensor_model.alpha})')

        # 1. Handle Initialization Spin (Delegated to Navigator)
        if not self.navigator.initial_spin_done:
            self.navigator.perform_initial_spin(self.current_position, self.current_theta)
            return

        # 2. Block if moving (Delegated to Navigator)
        if self.navigator.is_moving:
            return

        # 3. Handle Settling State (Mode Switching)
        if self.settling_start_time is not None:
            self._handle_settling_complete()
            return

        self.get_logger().info(f'[STEP {self.step_count}] Pos: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}) | Sensor: {self.sensor_raw_value:.4f}')

        # 4. Update Estimates
        self.particle_filter.update(self.sensor_raw_value, self.current_position)
        current_means, current_stds = self.particle_filter.get_estimate()
        est_x, est_y = current_means["x"], current_means["y"]

        # 5. Plan Next Move
        next_pos = None
        debug_info = {}
        dead_end_detected = False
        bi_optimal = 0.0

        if self.planner_mode == 'GLOBAL':
            next_pos, should_return = self._run_global_planning(current_means)
            if should_return: return
            
        elif self.planner_mode == 'LOCAL':
            next_pos, debug_info, dead_end_detected, bi_optimal = self._run_local_planning()

        # 6. Visualization & Logging (Delegated)
        self._update_visualizations(est_x, est_y, current_stds, debug_info, bi_optimal, dead_end_detected)
        self._log_step_data(current_means, current_stds, debug_info, bi_optimal, dead_end_detected)
        
        # 7. Check Convergence
        if self._check_convergence(current_stds):
            return

        # 8. Execute Move (Delegated to Navigator)
        if next_pos is not None:
            step_computation_time = time.time() - step_start_time
            self.computation_times.append(step_computation_time)
            self.get_logger().info(f'Moving to: ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
            self.navigator.send_goal(next_pos[0], next_pos[1], tolerance=self.params['xy_goal_tolerance'])

    # =========================================================================
    # PLANNING LOGIC
    # =========================================================================

    def _handle_settling_complete(self):
        self.get_logger().info('[MODE SWITCH] Settling complete. Switching to LOCAL.')
        self.settling_start_time = None
        self.particle_filter.update(self.sensor_raw_value, self.current_position)
        
        self.planner_mode = 'LOCAL'
        self.global_path = []
        self.global_path_index = 0
        self.marker_viz.clear_global_planner_visualizations()
        self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
        self.planning_pending = True

    def _run_global_planning(self, current_means) -> Tuple[Optional[Tuple[float, float]], bool]:
        """Returns: (Next Position, Should Return/Stop)"""
        self.get_logger().info('[GLOBAL MODE] Following frontier path...')
        
        if not self.global_path or self.global_path_index >= len(self.global_path):
            self.get_logger().warn('[GLOBAL MODE] Path exhausted. Triggering STOP & SETTLE.')
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True

        # Advance waypoint if reached
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
        
        # Collision Check
        if not self.is_path_to_waypoint_valid(waypoint):
            self.get_logger().warn(f'[GLOBAL MODE] Path blocked to {waypoint}.')
            self.global_path_index += 1
            self.planning_pending = True
            return None, True

        # Entropy Check (Using Optimized Vectorized Call)
        current_entropy = self.particle_filter.get_entropy()
        expected_entropy = self.particle_filter.compute_expected_entropy(waypoint)
        
        mutual_info = current_entropy - expected_entropy
        detector_status = self.dead_end_detector.get_status()
        thresh = self.params['switch_back_threshold'] * detector_status["bi_threshold"]

        if mutual_info > thresh:
            self.get_logger().info(f'[SWITCH] High MI found ({mutual_info:.4f}). Stopping to Settle.')
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True
            
        self.marker_viz.visualize_global_path(self.global_path)
        return waypoint, False

    def _run_local_planning(self) -> Tuple[Optional[Tuple[float, float]], dict, bool, float]:
        """Returns: (Next Position, Debug Info, Dead End Detected, Bi Optimal)"""
        debug_info = self.rrt.get_next_move_debug(self.current_position, self.particle_filter)
        next_pos = debug_info["next_position"]
        
        move_dist = np.hypot(next_pos[0] - self.current_position[0], next_pos[1] - self.current_position[1])
        
        if move_dist < 0.05:
            self.navigator.consecutive_failures += 1
            if self.navigator.consecutive_failures >= self.navigator.max_failures_tolerance:
                self.trigger_recovery()
                return None, debug_info, False, 0.0
        else:
            if self.navigator.consecutive_failures > 0:
                self.navigator.consecutive_failures -= 1
        
        bi_optimal = debug_info.get("best_utility", debug_info.get("best_entropy_gain", 0.0))
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

        self.get_logger().warn(f'[DEAD END] Found {len(frontier_cells)} frontier cells. Planning Global...')
        result = self.global_planner.plan(self.current_position, self.particle_filter)

        if result['success']:
            num_clusters = len(result.get('frontier_clusters', []))
            num_evaluated = result.get('num_reachable', 0)
            self.get_logger().info(f'[GLOBAL MODE] Activated. {num_clusters} clusters, {num_evaluated} evaluated.')
            self.marker_viz.clear_global_planner_visualizations()
            self.planner_mode = 'GLOBAL'
            self.global_path = result['best_global_path']
            self.global_path_index = 1
            
            # Viz
            self.marker_viz.visualize_frontier_cells(result['frontier_cells'])
            self.marker_viz.visualize_frontier_centroids(result['frontier_clusters'])
            self.marker_viz.visualize_prm_graph(result['prm_vertices'], self.global_planner.vertex_dict)
            self.marker_viz.visualize_global_path(self.global_path)
        else:
            self.get_logger().info('[DEAD END] Global plan failed. Staying LOCAL.')
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])

    def trigger_recovery(self):
        # 1. Attempt Teleport (Delegated to Navigator)
        success = self.navigator.attempt_teleport_recovery(
            self.current_position, self.slam_map, self.dead_end_detector
        )
        if success:
            self.current_position = None # Force update from pose callback
            return

        # 2. Fallback to Global Planner
        self.get_logger().warn('Teleport failed. Trying Global Planner fallback.')
        self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
        self.planner_mode = 'GLOBAL'
        
        res = self.global_planner.plan(self.current_position, self.particle_filter)
        if res['success']:
            self.global_path = res['best_global_path']
            self.global_path_index = 1
            self.marker_viz.clear_global_planner_visualizations()
            self.marker_viz.visualize_global_path(self.global_path)
            self.planning_pending = True
        else:
            self.get_logger().error('Global recovery failed. Staying LOCAL.')
            self.planner_mode = 'LOCAL'
            self.planning_pending = True

    def _check_convergence(self, current_stds):
        sigma_p = max(current_stds["x"], current_stds["y"])
        if sigma_p < self.params['sigma_threshold']:
            self.get_logger().info(f'Converged! σ_p = {sigma_p:.3f}')
            self._save_summary_and_finish()
            return True
        return False

    def _save_summary_and_finish(self):
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        means, _ = self.particle_filter.get_estimate()
        est_x, est_y = means["x"], means["y"]
        
        true_x, true_y = self.params['true_source_x'], self.params['true_source_y']
        est_error = -1.0
        if true_x != -999.0:
            est_error = np.hypot(est_x - true_x, est_y - true_y)

        avg_comp_time = np.mean(self.computation_times) if self.computation_times else 0.0

        # Delegate summary writing to Logger
        summary_text = self.logger.save_summary(
            self.step_count, self.total_travel_distance, elapsed_time, 
            avg_comp_time, est_x, est_y, est_error
        )
        self.get_logger().info('\n' + summary_text)
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
        
        # Initial trigger
        if (self.node_initialized and not self.navigator.initial_spin_done 
            and not self.navigator.is_moving and self.sensor_raw_value is not None 
            and not self.planning_pending):
            self.take_step()

    def sensor_callback(self, msg):
        self.sensor_raw_value = msg.raw
        if (self.node_initialized and not self.navigator.initial_spin_done 
            and not self.navigator.is_moving and self.current_position 
            and not self.planning_pending):
            self.take_step()

    def laser_callback(self, msg: LaserScan):
        """Process laser scan via LidarMapper."""
        if not self.node_initialized or self.current_position is None or self.current_theta is None:
            return

        obstacles_found = self.lidar_mapper.update_from_scan(
            msg, 
            self.current_position[0], 
            self.current_position[1], 
            self.current_theta
        )
        self.laser_scan_count += 1
        self.total_obstacles_marked += obstacles_found

    def wind_callback(self, msg: Anemometer):
        """Process wind sensor data from anemometer."""
        if not self.node_initialized:
            return
        self.wind_speed = msg.wind_speed
        self.wind_direction = msg.wind_direction
        # Convert polar to Cartesian
        self.wind_x = msg.wind_speed * np.cos(msg.wind_direction)
        self.wind_y = msg.wind_speed * np.sin(msg.wind_direction)

        # Record measurement in wind map at current robot position
        if self.current_position is not None:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.wind_map.add_measurement(
                self.current_position[0], self.current_position[1],
                msg.wind_speed, msg.wind_direction,
                timestamp=timestamp
            )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def is_path_to_waypoint_valid(self, waypoint: tuple) -> bool:
        """Check valid path using internal simple checker."""
        if self.current_position is None: return False
        return self._is_segment_valid(self.current_position, waypoint)

    def _is_valid_optimistic(self, position: tuple) -> bool:
        """Lightweight optimistic check used by RRT/GlobalPlanner integration."""
        gx, gy = self.slam_map.world_to_grid(*position)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height: return False
        
        radius_cells = int(np.ceil(self.params['robot_radius'] / self.slam_map.resolution))
        radius_sq = radius_cells ** 2

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_sq: continue
                check_gx, check_gy = gx + dx, gy + dy
                if 0 <= check_gx < self.slam_map.width and 0 <= check_gy < self.slam_map.height:
                    if self.slam_map.grid[check_gy, check_gx] > 0: 
                        return False
        return True

    def _is_segment_valid(self, start: tuple, end: tuple) -> bool:
        """Simple line check."""
        pos1, pos2 = np.array(start), np.array(end)
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6: return self._is_valid_optimistic(tuple(pos1))
        
        num_samples = max(int(np.ceil(dist / (self.slam_map.resolution * 0.5))), 2)
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            if not self._is_valid_optimistic((sample_pos[0], sample_pos[1])):
                return False
        return True

    def _update_visualizations(self, est_x, est_y, current_stds, debug_info, bi_optimal, dead_end_detected):
        self.marker_viz.visualize_planner_mode(self.planner_mode)
        self.marker_viz.visualize_particles(self.particle_filter.particles, self.particle_filter.weights)
        self.marker_viz.visualize_estimated_source(est_x, est_y)
        self.marker_viz.visualize_current_position(self.current_position)
        
        if self.planner_mode == 'LOCAL' and debug_info:
            self.marker_viz.visualize_all_paths(debug_info.get("all_paths", []), debug_info.get("all_utilities", None))
            self.marker_viz.visualize_best_path(debug_info.get("best_path", []))

        # Text Viz
        sigma_p = max(current_stds["x"], current_stds["y"])
        current_entropy = self.particle_filter.get_entropy()
        
        bin_width = self.sensor_model.max_concentration / self.sensor_model.num_levels
        current_bin = min(int(self.sensor_raw_value / bin_width), self.sensor_model.num_levels - 1)

        self.text_visualizer.publish_source_info(
            timestamp=self.get_clock().now().to_msg(),
            predicted_x=est_x, predicted_y=est_y, predicted_z=0.5,
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

    def _log_step_data(self, means, stds, debug_info, bi_optimal, dead_end_detected):
        bi_threshold = self.dead_end_detector.get_status()["bi_threshold"]
        self.logger.log_step(
            self.step_count, self.particle_filter, self.sensor_raw_value,
            self.current_position, self.params, debug_info,
            bi_optimal, bi_threshold, dead_end_detected,
            self.planner_mode,
            len(self.global_path) if self.planner_mode == 'GLOBAL' else 0,
            self.global_path_index if self.planner_mode == 'GLOBAL' else 0
        )
        self.step_count += 1

    def publish_slam_map(self):
        if not hasattr(self, 'slam_map'): return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.slam_map.resolution
        msg.info.width = self.slam_map.width
        msg.info.height = self.slam_map.height
        msg.info.origin.position.x = self.slam_map.origin_x
        msg.info.origin.position.y = self.slam_map.origin_y
        msg.info.origin.orientation.w = 1.0
        # Explicit mapping: -1→-1, 0→0, 1→100, 2→50 (outlet)
        grid_data = self.slam_map.grid.flatten()
        ros_grid = np.full_like(grid_data, -1, dtype=np.int8)
        ros_grid[grid_data == 0] = 0
        ros_grid[grid_data == 1] = 100
        ros_grid[grid_data == 2] = 50
        msg.data = ros_grid.tolist()
        self.slam_map_pub.publish(msg)

    def _publish_slam_map_timer(self):
        self.publish_slam_map()

    def gt_wind_callback(self, msg: MarkerArray):
        """Store ground truth wind vectors for validation."""
        self.gt_wind_data = msg

    def _publish_wind_map_timer(self):
        if not hasattr(self, 'wind_map') or not hasattr(self, 'marker_viz'):
            return

        # Fill enclosed unknown regions (wall interiors) with occupied
        filled = self.slam_map.fill_enclosed_unknown()
        if filled > 0:
            self.get_logger().info(f'Filled {filled} enclosed unknown cells as wall')

        # Re-solve wind field
        # GMRF does not strictly require outlets, just measurements + grid
        if self.params.get('use_gmrf', False):
             self.wind_map.solve_gmrf(self.slam_map.grid)
        else:
            # Re-solve potential flow using current SLAM grid
            has_outlets = np.any(self.slam_map.grid == 2)
            if has_outlets:
                solved = self.wind_map.solve_potential_flow(self.slam_map.grid)
                if solved and not getattr(self, '_pf_solve_logged', False):
                    n = self.wind_map.pf_estimator.num_clusters
                    self.get_logger().info(f'Potential flow solved with {n} outlet clusters')
                    self._pf_solve_logged = True

        # Update dispersion model with latest wind field
        self.dispersion_model.set_wind_field(
            self.wind_map.estimated_vx, self.wind_map.estimated_vy
        )

        self.marker_viz.visualize_wind_map(self.wind_map)

        # Validation
        if self.gt_wind_data and self.step_count % 10 == 0:
            # Basic check: compare N vectors?
            # For now, just log presence.
            # Real validation code would go here.
            self.get_logger().info(f'[VALIDATION] GT Vectors available: {len(self.gt_wind_data.markers)}')

    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.close()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RRTInfotaxisNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())