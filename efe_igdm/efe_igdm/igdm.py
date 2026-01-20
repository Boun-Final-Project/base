import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.task import Future

# Messages
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

# Custom Modules
from .mapping.occupancy_grid import create_occupancy_map_from_service, create_empty_occupancy_map
from .estimation.sensor_model import ContinuousGaussianSensorModel
from .estimation.particle_filter_optimized import ParticleFilterOptimized as ParticleFilter
from .models.igdm_gas_model import IndoorGaussianDispersionModel
from .planning.rrt import RRT
from .planning.global_planner import GlobalPlanner
from .visualization.text_visualizer import TextVisualizer
from .planning.dead_end_detector import DeadEndDetector

import numpy as np
import csv
from datetime import datetime
import os
import time
from typing import Tuple, List, Optional, Dict, Any

class RRTInfotaxisNode(Node):
    """RRT Infotaxis node that receives occupancy map and plans gas search."""

    def __init__(self):
        super().__init__('rrt_infotaxis_node')
        
        self._init_parameters()
        self._init_state_variables()
        self._setup_data_logging()
        self._init_ros_interfaces()
        self._init_models_and_planners()
        
        self.node_initialized = True
        self.get_logger().info('Node initialized successfully, waiting for sensor and pose data...')

    def _init_parameters(self):
        """Declare and cache parameters to avoid repeated lookups."""
        # Declarations
        self.declare_parameter('use_fast_rrt', True)
        self.declare_parameter('sigma_m', 1.5)
        self.declare_parameter('number_of_particles', 1000)
        self.declare_parameter('n_tn', 50)
        self.declare_parameter('delta', 0.7)
        self.declare_parameter('max_depth', 4)
        self.declare_parameter('xy_goal_tolerance', 0.3)
        self.declare_parameter('robot_radius', 0.05)
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
        self.declare_parameter('global_fstep_size', 1.5)
        self.declare_parameter('resample_threshold', 0.5)
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 4.5)

        # Cache values for performance
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
        }

    def _init_state_variables(self):
        """Initialize internal state variables."""
        self.sensor_raw_value: Optional[float] = None
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_theta: Optional[float] = None
        self.sensor_initialized = False
        self.node_initialized = False
        self.step_count = 0
        self.is_moving = False
        self.goal_handle = None
        self.goal_position = None
        self.search_complete = False
        self.current_dead_end_status = False
        self.planning_pending = False

        # Dual-mode planner state
        self.planner_mode = 'LOCAL'
        self.global_path: List[Tuple[float, float]] = []
        self.global_path_index = 0
        self.settling_start_time = None

        # Recovery state
        self.consecutive_failures = 0
        self.max_failures_tolerance = 3
        self.in_recovery = False
        self.initial_spin_done = False
        self.initial_spin_goal_handle = None

        # SLAM / Map logic
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

        # Performance tracking
        self.total_travel_distance = 0.0
        self.previous_position = None
        self.computation_times = []
        self.prev_num_paths = 0

    def _init_ros_interfaces(self):
        """Initialize Publishers, Subscribers, and Action Clients."""
        # Subscriptions
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, '/PioneerP3DX/ground_truth', self.pose_callback, 10)
        self.sensor_subscription = self.create_subscription(
            GasSensor, '/fake_pid/Sensor_reading', self.sensor_callback, 10)
        self.laser_subscription = self.create_subscription(
            LaserScan, '/PioneerP3DX/laser_scanner', self.laser_callback, 10)

        # Publishers
        self.particle_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/particles', 10)
        self.all_paths_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/all_paths', 10)
        self.best_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/best_path', 10)
        self.estimated_source_pub = self.create_publisher(Marker, '/rrt_infotaxis/estimated_source', 10)
        self.current_pos_pub = self.create_publisher(Marker, '/rrt_infotaxis/current_position', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/PioneerP3DX/cmd_vel', 10)
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/PioneerP3DX/initialpose', 10)

        # Global planner publishers
        self.frontier_cells_pub = self.create_publisher(Marker, '/rrt_infotaxis/frontier_cells', 10)
        self.frontier_centroids_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/frontier_centroids', 10)
        self.prm_graph_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/prm_graph', 10)
        self.global_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/global_path', 10)
        self.planner_mode_pub = self.create_publisher(Marker, '/rrt_infotaxis/planner_mode', 10)

        # SLAM map publisher
        map_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        self.slam_map_pub = self.create_publisher(OccupancyGrid, '/rrt_infotaxis/slam_map', map_qos)
        self.slam_map_timer = self.create_timer(0.5, self._publish_slam_map_timer)

        # Nav2 Client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/PioneerP3DX/navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Nav2 action server available!')

    def _init_models_and_planners(self):
        """Load maps and initialize computational models."""
        try:
            self.occupancy_map = create_occupancy_map_from_service(
                self, z_level=5, service_name='/gaden_environment/occupancyMap3D', timeout_sec=10.0
            )
            # FIX: GADEN service returns origin (0,0) but actual map starts at (-0.2, -0.2)
            if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
                self.get_logger().warn('GADEN origin correction applied.')
                self.occupancy_map.origin_x = -0.2
                self.occupancy_map.origin_y = -0.2

            self.slam_map = create_empty_occupancy_map(self.occupancy_map)
            self.get_logger().info(f'SLAM initialized: res={self.slam_map.resolution}m')
        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.params['sigma_m'], occupancy_grid=self.slam_map
        )

        self.sensor_model = ContinuousGaussianSensorModel(alpha=0.1, sigma_env=1.5, num_levels=10, max_concentration=120.0)
        
        self.particle_filter = ParticleFilter(
            num_particles=self.params['number_of_particles'],
            search_bounds={"x": (0, self.slam_map.real_world_width), "y": (0, self.slam_map.real_world_height), "Q": (0, 120.0)},
            binary_sensor_model=self.sensor_model,
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

        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")
        
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

    def _setup_data_logging(self):
        log_dir = os.path.expanduser('~/igdm_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'igdm_log_{timestamp}.csv')
        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'step', 'elapsed_time', 'entropy', 'std_dev_x', 'std_dev_y', 'std_dev_Q',
            'est_x', 'est_y', 'est_Q', 'sensor_value', 'continuous_measurement', 'threshold',
            'num_branches', 'best_utility', 'J1_entropy_gain', 'J2_travel_cost',
            'robot_x', 'robot_y', 'sigma_m', 'bi_optimal', 'bi_threshold', 'dead_end_detected',
            'planner_mode', 'global_path_length', 'global_waypoint_index'
        ])
        self.log_file.flush()
        self.get_logger().info(f'Data logging to: {log_filename}')
        self.start_time = self.get_clock().now()

    # =========================================================================
    # CORE LOGIC: Take Step (Refactored)
    # =========================================================================

    def take_step(self):
        """Main control loop iteration."""
        if self.search_complete or self.sensor_raw_value is None or self.current_position is None:
            return

        step_start_time = time.time()

        if not self.sensor_initialized:
            self.sensor_initialized = True
            self.get_logger().info(f'Sensor initialized (α={self.sensor_model.alpha})')

        # 1. Handle Initialization Spin
        if not self.initial_spin_done:
            self._handle_initial_spin()
            return

        # 2. Block if moving
        if self.is_moving:
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
            if should_return: return # Logic dictated a return (e.g., stopping to settle)
            
        elif self.planner_mode == 'LOCAL':
            next_pos, debug_info, dead_end_detected, bi_optimal = self._run_local_planning()

        # 6. Visualization & Logging
        self._update_visualizations(est_x, est_y, current_stds, debug_info, bi_optimal, dead_end_detected)
        self._log_step_data(current_means, current_stds, debug_info, bi_optimal, dead_end_detected)
        
        # 7. Check Convergence
        if self._check_convergence(current_stds):
            return

        # 8. Execute Move
        if next_pos is not None:  # <--- CHANGED from "if next_pos:"
            step_computation_time = time.time() - step_start_time
            self.computation_times.append(step_computation_time)
            self.get_logger().info(f'Moving to: ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
            self.send_nav_goal(next_pos[0], next_pos[1])

    # =========================================================================
    # PLANNING HELPERS
    # =========================================================================

    def _handle_initial_spin(self):
        if not self.is_moving:
            self.get_logger().info('[STARTUP] Starting initial 360° sensor sweep...')
            target_yaw = self.current_theta + 3.14
            self.send_nav_goal_pose(self.current_position[0], self.current_position[1], target_yaw, is_spin=True)

    def _handle_settling_complete(self):
        self.get_logger().info('[MODE SWITCH] Settling complete. Switching to LOCAL.')
        self.settling_start_time = None
        self.particle_filter.update(self.sensor_raw_value, self.current_position)
        
        self.planner_mode = 'LOCAL'
        self.global_path = []
        self.global_path_index = 0
        self.clear_global_planner_visualizations()
        self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
        self.planning_pending = True

    def _run_global_planning(self, current_means) -> Tuple[Optional[Tuple[float, float]], bool]:
        """
        Execute Global planning logic. 
        Returns: (Next Position, Should Return/Stop)
        """
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
            self.global_path_index += 1 # Try next
            if self.global_path_index >= len(self.global_path):
                 self.settling_start_time = self.get_clock().now()
                 self.planning_pending = True
                 return None, True
            self.planning_pending = True
            return None, True

        # Entropy / Information Gain Check
        # Calculate expected entropy reduction if we were to measure at the waypoint
        current_entropy = self.particle_filter.get_entropy()
        expected_entropy = 0.0
        for m in range(self.sensor_model.num_levels):
            prob_z = self.particle_filter.predict_measurement_probability(waypoint, m)
            hyp_entropy = self.particle_filter.compute_hypothetical_entropy(m, waypoint)
            expected_entropy += prob_z * hyp_entropy
        
        mutual_info = current_entropy - expected_entropy
        detector_status = self.dead_end_detector.get_status()
        thresh = self.params['switch_back_threshold'] * detector_status["bi_threshold"]

        if mutual_info > thresh:
            self.get_logger().info(f'[SWITCH] High MI found ({mutual_info:.4f}). Stopping to Settle.')
            self.settling_start_time = self.get_clock().now()
            self.planning_pending = True
            return None, True
            
        self.visualize_global_path(self.global_path)
        return waypoint, False

    def _run_local_planning(self) -> Tuple[Optional[Tuple[float, float]], dict, bool, float]:
        """
        Execute Local RRT planning.
        Returns: (Next Position, Debug Info, Dead End Detected, Bi Optimal)
        """
        debug_info = self.rrt.get_next_move_debug(self.current_position, self.particle_filter)
        next_pos = debug_info["next_position"]
        
        move_dist = np.hypot(next_pos[0] - self.current_position[0], next_pos[1] - self.current_position[1])
        
        if move_dist < 0.05:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures_tolerance:
                self.trigger_recovery()
                # Recovery triggers teleport/plan, so we return None here to stop this step
                return None, debug_info, False, 0.0
        else:
            if self.consecutive_failures > 0:
                self.consecutive_failures -= 1
        
        bi_optimal = debug_info.get("best_utility", debug_info.get("best_entropy_gain", 0.0))
        dead_end_detected = False

        if self.params['enable_global_planner']:
            dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)
            if dead_end_detected:
                self._handle_dead_end_transition()
                # If transition successful, next_pos might change in next loop, 
                # but for now we follow the local path or wait for logic to update
                if self.planner_mode == 'GLOBAL' and self.global_path:
                    next_pos = self.global_path[self.global_path_index]

        return next_pos, debug_info, dead_end_detected, bi_optimal

    def _handle_dead_end_transition(self):
        frontier_cells = self.global_planner.detect_frontiers()
        if not frontier_cells:
            self.get_logger().info('[DEAD END] No frontiers. Staying LOCAL.')
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
            return

        self.get_logger().warn(f'[DEAD END] Found {len(frontier_cells)} frontiers. Planning Global...')
        result = self.global_planner.plan(self.current_position, self.particle_filter)
        
        if result['success']:
            self.get_logger().info('[GLOBAL MODE] Activated.')
            self.clear_global_planner_visualizations()
            self.planner_mode = 'GLOBAL'
            self.global_path = result['best_global_path']
            self.global_path_index = 1
            
            # Viz
            self.visualize_frontier_cells(result['frontier_cells'])
            self.visualize_frontier_centroids(result['frontier_clusters'])
            self.visualize_prm_graph(result['prm_vertices'])
            self.visualize_global_path(self.global_path)
        else:
            self.get_logger().info('[DEAD END] Global plan failed. Staying LOCAL.')
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])

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

        summary = (
            f"ST: {self.step_count} steps\n"
            f"TD: {self.total_travel_distance:.2f} m\n"
            f"Time: {elapsed_time:.2f} s\n"
            f"Avg Comp: {avg_comp_time:.4f} s\n"
            f"Error: {est_error:.3f} m\n"
            f"Est Source: ({est_x:.3f}, {est_y:.3f})"
        )
        self.get_logger().info('\n' + summary)
        
        summary_filename = self.log_file.name.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write(summary)
        
        self.search_complete = True

    # =========================================================================
    # RECOVERY & MOVEMENT
    # =========================================================================

    def trigger_recovery(self):
        self.get_logger().warn('!!! STUCK DETECTED - RECOVERING !!!')
        self.consecutive_failures = 0
        self.in_recovery = False
        
        safe_pos = self.find_safe_position_away_from_wall(self.current_position, distance=0.5)
        
        if safe_pos != self.current_position:
            self.teleport_robot(safe_pos[0], safe_pos[1])
            time.sleep(1.0)
            self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
            self.planning_pending = True
            return

        self.get_logger().warn('Teleport failed. Trying Global Planner fallback.')
        self.dead_end_detector.reset(initial_threshold=self.params['dead_end_initial_threshold'])
        self.planner_mode = 'GLOBAL'
        
        res = self.global_planner.plan(self.current_position, self.particle_filter)
        if res['success']:
            self.global_path = res['best_global_path']
            self.global_path_index = 1
            self.clear_global_planner_visualizations()
            self.visualize_global_path(self.global_path)
            self.planning_pending = True
        else:
            self.get_logger().error('Global recovery failed. Staying LOCAL.')
            self.planner_mode = 'LOCAL'
            self.planning_pending = True

    def send_nav_goal(self, x, y):
        # Helper that doesn't care about orientation
        self.send_nav_goal_pose(x, y, 0.0, use_orientation=False)

    def send_nav_goal_pose(self, x, y, yaw, is_spin=False, use_orientation=True):
        self.goal_position = (float(x), float(y))
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        
        if use_orientation or is_spin:
            from math import sin, cos
            qz = sin(yaw / 2.0)
            qw = cos(yaw / 2.0)
            goal_msg.pose.pose.orientation.z = float(qz)
            goal_msg.pose.pose.orientation.w = float(qw)
        else:
            goal_msg.pose.pose.orientation.w = 1.0

        self.is_moving = True
        future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self.nav_feedback_callback)
        
        if is_spin:
            future.add_done_callback(self.initial_spin_response_callback)
        else:
            future.add_done_callback(self.nav_goal_response_callback)

    def nav_feedback_callback(self, feedback_msg):
        if self.goal_position is None or self.goal_handle is None: return
        
        current_pose = feedback_msg.feedback.current_pose.pose.position
        dx = current_pose.x - self.goal_position[0]
        dy = current_pose.y - self.goal_position[1]
        
        if np.hypot(dx, dy) <= self.params['xy_goal_tolerance']:
            self.get_logger().debug('XY goal reached via feedback, canceling for smooth stop.')
            self.goal_handle.cancel_goal_async().add_done_callback(self._goal_cancel_callback)

    def nav_goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Goal rejected!')
            self.is_moving = False
            return
        self.goal_handle.get_result_async().add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        status = future.result().status
        if status == 4: # SUCCEEDED
            self.consecutive_failures = 0
            self.in_recovery = False
        elif status == 6: # ABORTED
            if not self.in_recovery:
                self.consecutive_failures += 1
        
        self.is_moving = False
        self.goal_handle = None
        self.planning_pending = True

    def initial_spin_response_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self.initial_spin_done = True
            self.is_moving = False
            return
        self.initial_spin_goal_handle = gh
        gh.get_result_async().add_done_callback(lambda f: self._finish_initial_spin())

    def _finish_initial_spin(self):
        self.is_moving = False
        self.initial_spin_done = True
        self.get_logger().info('Initial spin complete.')
        self.planning_pending = True
    
    def _goal_cancel_callback(self, future):
        # Triggered when we manually cancel because we are close enough
        pass

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        from math import atan2
        
        # Orientation quaternion to yaw
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

        if self.planning_pending and not self.is_moving:
            self.planning_pending = False
            self.take_step()
        
        # Initial trigger
        if self.node_initialized and not self.initial_spin_done and not self.is_moving and self.sensor_raw_value is not None and not self.planning_pending:
            self.take_step()

    def sensor_callback(self, msg):
        self.sensor_raw_value = msg.raw
        if self.node_initialized and not self.initial_spin_done and not self.is_moving and self.current_position and not self.planning_pending:
            self.take_step()

    # =========================================================================
    # VISUALIZATION & LOGGING
    # =========================================================================

    def _update_visualizations(self, est_x, est_y, current_stds, debug_info, bi_optimal, dead_end_detected):
        self.visualize_planner_mode()
        self.visualize_particles(self.particle_filter.particles, self.particle_filter.weights)
        self.visualize_estimated_source(est_x, est_y)
        self.visualize_current_position(self.current_position)
        
        if self.planner_mode == 'LOCAL' and debug_info:
            self.visualize_all_paths(debug_info.get("all_paths", []), debug_info.get("all_utilities", None))
            self.visualize_best_path(debug_info.get("best_path", []))

        # Text Viz
        sigma_p = max(current_stds["x"], current_stds["y"])
        current_entropy = self.particle_filter.get_entropy()
        
        # Calculate visualization bins
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
        debug_info = debug_info or {}
        row = [
            self.step_count, 
            0, # elapsed time placeholder
            f'{self.particle_filter.get_entropy():.4f}',
            f'{stds["x"]:.4f}', f'{stds["y"]:.4f}', f'{stds["Q"]:.4f}',
            f'{means["x"]:.4f}', f'{means["y"]:.4f}', f'{means["Q"]:.4f}',
            f'{self.sensor_raw_value:.4f}', f'{self.sensor_raw_value:.4f}', '0.0',
            debug_info.get("num_branches", 0),
            f'{debug_info.get("best_utility", 0.0):.4f}',
            f'{debug_info.get("best_entropy_gain", 0.0):.4f}',
            f'{debug_info.get("best_travel_cost", 0.0):.4f}',
            f'{self.current_position[0]:.4f}', f'{self.current_position[1]:.4f}',
            f'{self.params["sigma_m"]:.4f}',
            f'{bi_optimal:.4f}',
            f'{self.dead_end_detector.get_status()["bi_threshold"]:.4f}',
            1 if dead_end_detected else 0,
            self.planner_mode,
            len(self.global_path) if self.planner_mode == 'GLOBAL' else 0,
            self.global_path_index if self.planner_mode == 'GLOBAL' else 0
        ]
        self.csv_writer.writerow(row)
        self.log_file.flush()
        self.step_count += 1

    # =========================================================================
    # PRESERVED LASER & SLAM LOGIC (Unchanged)
    # =========================================================================

    def laser_callback(self, msg: LaserScan):
        """Process laser scan and update SLAM map using ground truth pose."""
        if not hasattr(self, 'slam_map') or self.current_position is None or self.current_theta is None:
            return

        # Corrected: Allow map updates while moving to prevent data loss
        robot_x = self.current_position[0]
        robot_y = self.current_position[1]
        robot_theta = self.current_theta
        obstacles_this_scan = 0

        for i, range_val in enumerate(msg.ranges):
            if not np.isfinite(range_val):
                continue
            range_val = min(range_val, msg.range_max)
            angle = msg.angle_min + i * msg.angle_increment
            hit_obstacle = (range_val >= msg.range_min and range_val < msg.range_max)

            local_x = range_val * np.cos(angle)
            local_y = range_val * np.sin(angle)
            end_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
            end_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)

            self._mark_ray_as_free(robot_x, robot_y, end_x, end_y)

            if hit_obstacle:
                if self._mark_obstacle_in_slam_map(end_x, end_y):
                    obstacles_this_scan += 1

        self.laser_scan_count += 1
        self.total_obstacles_marked += obstacles_this_scan

    def _mark_obstacle_in_slam_map(self, world_x: float, world_y: float) -> bool:
        gx, gy = self.slam_map.world_to_grid(world_x, world_y)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
            return False
        self.slam_map.grid[gy, gx] = 1
        return True

    def _mark_ray_as_free(self, x0, y0, x1, y1):
        gx0, gy0 = self.slam_map.world_to_grid(x0, y0)
        gx1, gy1 = self.slam_map.world_to_grid(x1, y1)
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        x = gx0
        y = gy0
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        while True:
            if 0 <= x < self.slam_map.width and 0 <= y < self.slam_map.height:
                if x == gx1 and y == gy1:
                    break
                # BUGFIX: Don't overwrite previously detected obstacles
                if self.slam_map.grid[y, x] != 1:
                    self.slam_map.grid[y, x] = 0

            if x == gx1 and y == gy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

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
        flat_grid = (self.slam_map.grid.flatten() * 100).astype(np.int8)
        msg.data = flat_grid.tolist()
        self.slam_map_pub.publish(msg)

    def _publish_slam_map_timer(self):
        self.publish_slam_map()

    # =========================================================================
    # HELPERS (Calculations, Viz)
    # =========================================================================

    def find_safe_position_away_from_wall(self, current_pos, distance=0.5):
        if current_pos is None: return current_pos
        cx, cy = current_pos
        num_samples = 16
        max_search_distance = 2.0
        best_position = None
        max_clearance = 0

        for i in range(num_samples):
            angle = (2 * np.pi * i) / num_samples
            for test_dist in np.linspace(distance, max_search_distance, 5):
                test_x = cx + test_dist * np.cos(angle)
                test_y = cy + test_dist * np.sin(angle)
                test_pos = (test_x, test_y)
                if self._is_valid_optimistic(test_pos):
                    clearance = self._calculate_clearance(test_pos)
                    if clearance > max_clearance:
                        max_clearance = clearance
                        best_position = test_pos

        if best_position is not None and max_clearance >= distance:
            return best_position
        else:
            self.get_logger().warn(f'Could not find safe position {distance}m from walls')
            return current_pos

    def _calculate_clearance(self, position):
        gx, gy = self.slam_map.world_to_grid(*position)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height: return 0.0
        max_radius = int(2.0 / self.slam_map.resolution)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius: continue
                    check_gx = gx + dx
                    check_gy = gy + dy
                    if 0 <= check_gx < self.slam_map.width and 0 <= check_gy < self.slam_map.height:
                        if self.slam_map.grid[check_gy, check_gx] > 0:
                            return radius * self.slam_map.resolution
        return max_radius * self.slam_map.resolution

    def teleport_robot(self, target_x, target_y):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(target_x)
        msg.pose.pose.position.y = float(target_y)
        msg.pose.pose.orientation.w = 1.0
        if self.current_theta is not None:
            from math import sin, cos
            msg.pose.pose.orientation.z = float(sin(self.current_theta / 2.0))
            msg.pose.pose.orientation.w = float(cos(self.current_theta / 2.0))
        msg.pose.covariance = [0.0] * 36
        self.initialpose_pub.publish(msg)
        self.get_logger().info(f'Teleported robot to ({target_x:.2f}, {target_y:.2f})')
        self.current_position = (target_x, target_y)

    def is_path_to_waypoint_valid(self, waypoint: tuple) -> bool:
        if self.current_position is None: return False
        return self._is_segment_valid(self.current_position, waypoint)

    def _is_valid_optimistic(self, position: tuple) -> bool:
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

    # --- Visualization Wrappers (Condensed) ---
    def visualize_particles(self, particles, weights):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "particles", 0, Marker.POINTS, Marker.ADD
        marker.scale.x = marker.scale.y = 0.1
        marker.pose.orientation.w = 1.0
        
        norm_w = weights / weights.max() if (len(weights) > 0 and weights.max() > 0) else weights
        
        for p_val, w in zip(particles, norm_w):
            p = Point()
            p.x, p.y, p.z = float(p_val[0]), float(p_val[1]), 0.5
            marker.points.append(p)
            c = ColorRGBA()
            c.r, c.g, c.b, c.a = float(w), float(w * 0.8), float(1.0 - w), 0.8
            marker.colors.append(c)
        
        marker_array.markers.append(marker)
        self.particle_pub.publish(marker_array)

    def visualize_all_paths(self, all_paths, all_utilities=None):
        marker_array = MarkerArray()
        # Delete old
        for i in range(self.prev_num_paths):
            m = Marker()
            m.action = Marker.DELETE
            m.ns, m.id = "all_paths", i
            marker_array.markers.append(m)
        
        norm_utils = None
        if all_utilities and len(all_utilities) > 0:
            utils = np.array(all_utilities)
            rng = utils.max() - utils.min()
            norm_utils = (utils - utils.min()) / rng if rng > 1e-6 else np.ones_like(utils)*0.5

        count = 0
        for i, path in enumerate(all_paths):
            if len(path) < 2: continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id, marker.type, marker.action = "all_paths", i, Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.08
            for node in path:
                p = Point()
                p.x, p.y, p.z = float(node.position[0]), float(node.position[1]), 0.5
                marker.points.append(p)
            
            c = ColorRGBA()
            if norm_utils is not None and i < len(norm_utils):
                val = norm_utils[i] ** 0.5
                c.r = 1.0 if val < 0.5 else float(2.0 * (1.0 - val))
                c.g = float(2.0 * val) if val < 0.5 else 1.0
                c.b = 0.0
                c.a = 0.9
            else:
                c.r, c.g, c.b, c.a = 0.6, 0.6, 0.6, 0.5
            marker.color = c
            marker_array.markers.append(marker)
            count += 1
        self.prev_num_paths = count
        self.all_paths_pub.publish(marker_array)

    def visualize_best_path(self, best_path):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "best_path", 0
        marker.header.stamp = self.get_clock().now().to_msg()
        if len(best_path) < 2:
            marker.action = Marker.DELETE
        else:
            marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.20
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
            for node in best_path:
                p = Point()
                p.x, p.y, p.z = float(node.position[0]), float(node.position[1]), 0.6
                marker.points.append(p)
        self.best_path_pub.publish(marker)

    def visualize_estimated_source(self, est_x, est_y):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "estimated_source", 0
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(est_x), float(est_y), 0.5
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0)
        self.estimated_source_pub.publish(marker)

    def visualize_current_position(self, position):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "current_position", 0
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(position[0]), float(position[1]), 0.5
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.current_pos_pub.publish(marker)

    def visualize_frontier_cells(self, frontier_cells):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "frontier_cells", 0
        marker.type, marker.action = Marker.CUBE_LIST, Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
        for gx, gy in frontier_cells:
            wx, wy = self.slam_map.grid_to_world(gx, gy)
            p = Point()
            p.x, p.y, p.z = wx, wy, 0.1
            marker.points.append(p)
        self.frontier_cells_pub.publish(marker)

    def visualize_frontier_centroids(self, frontier_clusters):
        marker_array = MarkerArray()
        for i, cluster in enumerate(frontier_clusters):
            marker = Marker()
            marker.header.frame_id, marker.ns, marker.id = "map", "frontier_centroids", i
            marker.type, marker.action = Marker.SPHERE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = cluster.centroid_world[0], cluster.centroid_world[1], 0.3
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            marker_array.markers.append(marker)
        self.frontier_centroids_pub.publish(marker_array)

    def visualize_prm_graph(self, prm_vertices):
        # Implementation identical to original, just ensuring correct indent/imports
        marker_array = MarkerArray()
        # Vertices
        v_marker = Marker()
        v_marker.header.frame_id, v_marker.ns, v_marker.id = "map", "prm_vertices", 0
        v_marker.type, v_marker.action = Marker.SPHERE_LIST, Marker.ADD
        v_marker.scale.x = v_marker.scale.y = v_marker.scale.z = 0.15
        v_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
        
        # Edges
        e_marker = Marker()
        e_marker.header.frame_id, e_marker.ns, e_marker.id = "map", "prm_edges", 1
        e_marker.type, e_marker.action = Marker.LINE_LIST, Marker.ADD
        e_marker.scale.x = 0.02
        e_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3)
        
        added_edges = set()
        for v in prm_vertices:
            p = Point()
            p.x, p.y, p.z = v.position[0], v.position[1], 0.2
            v_marker.points.append(p)
            
            for nid, _ in v.neighbors:
                edge = tuple(sorted([v.id, nid]))
                if edge not in added_edges:
                    added_edges.add(edge)
                    neighbor = self.global_planner.vertex_dict[nid]
                    p2 = Point()
                    p2.x, p2.y, p2.z = neighbor.position[0], neighbor.position[1], 0.2
                    e_marker.points.append(p)
                    e_marker.points.append(p2)
        
        marker_array.markers.append(v_marker)
        marker_array.markers.append(e_marker)
        self.prm_graph_pub.publish(marker_array)

    def visualize_global_path(self, global_path):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "global_path", 0
        marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
        marker.scale.x = 0.15
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for pos in global_path:
            p = Point()
            p.x, p.y, p.z = pos[0], pos[1], 0.4
            marker.points.append(p)
        self.global_path_pub.publish(marker)

    def visualize_planner_mode(self):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "planner_mode", 0
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position.x = self.slam_map.origin_x + 1.0
        marker.pose.position.y = self.slam_map.origin_y + self.slam_map.real_world_height - 1.0
        marker.pose.position.z = 2.0
        marker.scale.z = 0.5
        if self.planner_mode == 'LOCAL':
            marker.text = "MODE: LOCAL (RRT-Infotaxis)"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            marker.text = "MODE: GLOBAL (Frontier Exploration)"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        self.planner_mode_pub.publish(marker)

    def clear_global_planner_visualizations(self):
        # Simple delete triggers
        m = Marker()
        m.action = Marker.DELETE
        m.header.frame_id = "map"
        
        m.ns = "frontier_cells"
        self.frontier_cells_pub.publish(m)
        m.ns = "global_path"
        self.global_path_pub.publish(m)
        
        ma = MarkerArray()
        for ns in ["prm_vertices", "prm_edges", "frontier_vertices", "frontier_centroids"]:
            for i in range(100):
                dm = Marker()
                dm.header.frame_id = "map"
                dm.ns, dm.id, dm.action = ns, i, Marker.DELETE
                ma.markers.append(dm)
        self.prm_graph_pub.publish(ma)
        self.frontier_centroids_pub.publish(ma)

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()

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