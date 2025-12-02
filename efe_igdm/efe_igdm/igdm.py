import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav2_msgs.action import NavigateToPose
from .occupancy_grid import create_occupancy_map_from_service
from .sensor_model import DiscreteSensorModel
# from .particle_filter import ParticleFilter
from .particle_filter_optimized import ParticleFilterOptimized as ParticleFilter
from .igdm_gas_model import IndoorGaussianDispersionModel
from .rrt import RRT
from .unit_conversion import GasUnitConverter
from .text_visualizer import TextVisualizer
from .dead_end_detector import DeadEndDetector
import numpy as np
import csv
from datetime import datetime
import os


class RRTInfotaxisNode(Node):
    """RRT Infotaxis node that receives occupancy map."""

    def __init__(self):
        super().__init__('rrt_infotaxis_node')

        # Parameters
        self.declare_parameter('use_fast_rrt', True)  # Use optimized RRT implementation
        self.declare_parameter('sigma_m', 1.5)  # IGDM constant std dev (meters)
        self.declare_parameter('number_of_particles', 1000)
        self.declare_parameter('n_tn', 50)
        self.declare_parameter('delta', 0.7)
        self.declare_parameter('xy_goal_tolerance', 0.3)  # XY distance tolerance in meters
        self.declare_parameter('robot_radius', 0.35)  # Robot footprint radius for collision checking
        self.declare_parameter('sigma_threshold', 0.35)  # Std dev threshold for estimation convergence (LOWERED to prevent early convergence)
        self.declare_parameter('success_distance', 0.5)  # Distance threshold for source localization success (meters)
        self.declare_parameter('positive_weight', 0.5)  # Weight of information gain compared to travel cost

        # Dead end detection parameters (Equations 20-21 from paper)
        self.declare_parameter('dead_end_epsilon', 0.85)  # Weight for threshold update (ε in Eq. 21)
        self.declare_parameter('dead_end_initial_threshold', 0.1)  # Initial BI threshold

        # Particle filter diversity parameters (prevent early convergence)
        self.declare_parameter('resample_threshold', 0.5)  # Effective sample size threshold for resampling
        self.declare_parameter('min_steps_before_convergence', 10)  # Minimum steps before allowing convergence
        self.declare_parameter('block_convergence_in_dead_end', True)  # Don't converge when dead end detected

        # Sensor readings
        self.sensor_raw_value = None
        self.current_position = None
        self.sensor_initialized = False

        # State management for measure-plan-move loop
        self.is_moving = False
        self.goal_handle = None
        self.goal_position = None  # Store target goal position
        self.search_complete = False  # Flag to indicate if source search is finished
        self.current_dead_end_status = False  # Track if currently in a dead end

        # Data logging for entropy tracking
        self.step_count = 0
        self.start_time = None
        self.log_file = None
        self.csv_writer = None
        self._setup_data_logging()

        # Track previous marker counts for proper cleanup
        self.prev_num_paths = 0

        # Subscriptions
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/PioneerP3DX/ground_truth',
            self.pose_callback,
            10
        )

        self.sensor_subscription = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self.sensor_callback,
            10
        )

        # Publishers for visualization
        self.particle_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/particles', 10)
        self.all_paths_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/all_paths', 10)
        self.best_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/best_path', 10)
        self.estimated_source_pub = self.create_publisher(Marker, '/rrt_infotaxis/estimated_source', 10)
        self.current_pos_pub = self.create_publisher(Marker, '/rrt_infotaxis/current_position', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        # Nav2 action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/PioneerP3DX/navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Nav2 action server available!')

        # Load occupancy map from service
        try:
            self.occupancy_map = create_occupancy_map_from_service(
                self,
                z_level=5,
                service_name='/gaden_environment/occupancyMap3D',
                timeout_sec=10.0
            )
            self.get_logger().info('Successfully received occupancy map!')

        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        # Initialize Indoor Gaussian Dispersion Model (IGDM)
        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.get_parameter('sigma_m').value,
            occupancy_grid=self.occupancy_map
        )
        self.get_logger().info(f'IGDM initialized with σ_m={self.get_parameter("sigma_m").value}m')

        # Initialize sensor model and particle filter
        # Using DiscreteSensorModel with 6 levels (empirical three-sigma rule)
        self.sensor_model = DiscreteSensorModel(alpha=0.1, sigma_env=1.5, num_levels=6)
        self.particle_filter = ParticleFilter(
            num_particles=self.get_parameter('number_of_particles').value,
            search_bounds={"x": (0, self.occupancy_map.real_world_width), "y": (0, self.occupancy_map.real_world_height), "Q": (0, 200)},
            binary_sensor_model=self.sensor_model,
            dispersion_model=self.dispersion_model
        )
        self.get_logger().info('Particle filter initialized')

        # Initialize RRT (choose fast or standard implementation)
        use_fast = self.get_parameter('use_fast_rrt').value
        RRTClass = RRT

        self.rrt = RRTClass(
            occupancy_grid=self.occupancy_map,
            N_tn=self.get_parameter('n_tn').value,
            R_range=self.get_parameter('n_tn').value * self.get_parameter('delta').value,
            delta=self.get_parameter('delta').value,
            robot_radius=self.get_parameter('robot_radius').value,
            positive_weight=self.get_parameter('positive_weight').value
        )
        self.ppm_converter = GasUnitConverter()
        rrt_type = "Fast RRT" if use_fast else "Standard RRT"
        self.get_logger().info(f'{rrt_type} initialized')

        # Initialize text visualizer
        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")
        self.get_logger().info('Text visualizer initialized')

        # Initialize dead end detector (Equations 20-21 from paper)
        self.dead_end_detector = DeadEndDetector(
            epsilon=self.get_parameter('dead_end_epsilon').value,
            initial_threshold=self.get_parameter('dead_end_initial_threshold').value
        )
        self.get_logger().info(f'Dead end detector initialized (ε={self.get_parameter("dead_end_epsilon").value})')

        # Timer for taking steps (start after all components are initialized)
        self.timer = self.create_timer(0.2, self.take_step)
        self.get_logger().info('Node initialized successfully, starting measure-plan-move loop')

    def _setup_data_logging(self):
        """Setup CSV file for logging entropy and other metrics."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.expanduser('~/igdm_logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'igdm_log_{timestamp}.csv')

        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        # Write header
        self.csv_writer.writerow([
            'step', 'elapsed_time', 'entropy', 'std_dev_x', 'std_dev_y', 'std_dev_Q',
            'est_x', 'est_y', 'est_Q', 'sensor_value', 'discrete_level', 'max_threshold',
            'num_branches', 'best_utility', 'J1_entropy_gain', 'J2_travel_cost',
            'robot_x', 'robot_y', 'sigma_m', 'bi_optimal', 'bi_threshold', 'dead_end_detected'
        ])
        self.log_file.flush()

        self.get_logger().info(f'Data logging to: {log_filename}')
        self.start_time = self.get_clock().now()

    def pose_callback(self, msg):
        """Callback for robot pose updates."""
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().debug(f'Robot position: {self.current_position}')

    def sensor_callback(self, msg):
        """Callback for gas sensor readings."""
        self.sensor_raw_value = msg.raw
        self.get_logger().debug(f'Received sensor reading: {self.sensor_raw_value}')

    def visualize_particles(self, particles, weights):
        """Visualize particles with weights as colored spheres."""
        marker_array = MarkerArray()

        # Normalize weights for coloring
        if len(weights) > 0 and weights.max() > 0:
            normalized_weights = weights / weights.max()
        else:
            normalized_weights = weights

        for i, (particle, weight) in enumerate(zip(particles, normalized_weights)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(particle[0])
            marker.pose.position.y = float(particle[1])
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Color based on weight (viridis-like: blue to yellow)
            marker.color = ColorRGBA()
            marker.color.r = float(weight)
            marker.color.g = float(weight * 0.8)
            marker.color.b = float(1.0 - weight)
            marker.color.a = 0.6

            marker_array.markers.append(marker)

        self.particle_pub.publish(marker_array)

    def visualize_all_paths(self, all_paths, all_utilities=None):
        """
        Visualize all RRT paths with color coding based on utility.

        Color scheme:
        - RED (low utility) -> YELLOW (medium) -> GREEN (high utility)
        - Best path is highlighted separately in blue
        """
        marker_array = MarkerArray()

        # Explicitly delete old markers from previous iteration
        # DELETEALL doesn't work reliably when mixed with ADD in same MarkerArray
        for i in range(self.prev_num_paths):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "all_paths"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Normalize utilities for color mapping if provided
        if all_utilities and len(all_utilities) > 0:
            utilities = np.array(all_utilities)
            min_util = utilities.min()
            max_util = utilities.max()

            self.get_logger().info(f'[VIZ] Path utilities: min={min_util:.4f}, max={max_util:.4f}, range={max_util-min_util:.4f}')

            # Avoid division by zero
            if max_util - min_util > 1e-6:
                normalized_utilities = (utilities - min_util) / (max_util - min_util)
            else:
                self.get_logger().warn('[VIZ] All utilities are identical - using uniform coloring')
                normalized_utilities = np.ones_like(utilities) * 0.5
        else:
            self.get_logger().warn('[VIZ] No utilities provided - using gray coloring')
            normalized_utilities = None

        # Add new path markers
        num_valid_paths = 0
        for i, path in enumerate(all_paths):
            if len(path) < 2:
                continue

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "all_paths"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            for node in path:
                point = Point()
                point.x = float(node.position[0])
                point.y = float(node.position[1])
                point.z = 0.5
                marker.points.append(point)

            marker.scale.x = 0.08  # Line width (increased for visibility)

            # Color based on normalized utility
            marker.color = ColorRGBA()
            if normalized_utilities is not None and i < len(normalized_utilities):
                # Color map: Red (0.0) -> Yellow (0.5) -> Green (1.0)
                # Apply power transformation for better contrast
                norm_util = normalized_utilities[i]
                # Square the value for more contrast (emphasize differences)
                norm_util_enhanced = norm_util ** 0.5  # Spread out low values more

                if norm_util_enhanced < 0.5:
                    # Red to Yellow transition
                    marker.color.r = 1.0
                    marker.color.g = float(2.0 * norm_util_enhanced)
                    marker.color.b = 0.0
                else:
                    # Yellow to Green transition
                    marker.color.r = float(2.0 * (1.0 - norm_util_enhanced))
                    marker.color.g = 1.0
                    marker.color.b = 0.0

                marker.color.a = 0.9  # Even higher opacity for better visibility
            else:
                # Fallback to gray if no utilities provided
                marker.color.r = 0.6
                marker.color.g = 0.6
                marker.color.b = 0.6
                marker.color.a = 0.5

            marker_array.markers.append(marker)
            num_valid_paths += 1

        # Track how many markers we published for next iteration
        self.prev_num_paths = num_valid_paths

        self.all_paths_pub.publish(marker_array)

    def visualize_best_path(self, best_path):
        """Visualize the best path in blue (highlighted)."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_path"
        marker.id = 0

        # If no valid path, delete old marker
        if len(best_path) < 2:
            marker.action = Marker.DELETE
            self.best_path_pub.publish(marker)
            return

        # Valid path exists - visualize it
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        for node in best_path:
            point = Point()
            point.x = float(node.position[0])
            point.y = float(node.position[1])
            point.z = 0.6  # Slightly higher than other paths
            marker.points.append(point)

        marker.scale.x = 0.20  # Thicker line for best path visibility

        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 0.5  # Add some green for cyan color
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.best_path_pub.publish(marker)

    def visualize_estimated_source(self, est_x, est_y):
        """Visualize estimated source location as orange sphere."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "estimated_source"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(est_x)
        marker.pose.position.y = float(est_y)
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.65
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.estimated_source_pub.publish(marker)

    def visualize_current_position(self, position):
        """Visualize current robot position as green sphere."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_position"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.current_pos_pub.publish(marker)

    def is_estimation_converged(self):
        """
        Check if particle filter estimation has converged (std dev below threshold).

        Prevents early convergence by requiring:
        1. Standard deviation below threshold
        2. Minimum number of steps completed
        3. Not currently in a dead end (optional, configurable)
        """
        current_means, current_stds = self.particle_filter.get_estimate()
        sigma_x = current_stds["x"]
        sigma_y = current_stds["y"]

        # Use maximum of x and y standard deviations
        sigma_p = max(sigma_x, sigma_y)

        sigma_threshold = self.get_parameter('sigma_threshold').value
        min_steps = self.get_parameter('min_steps_before_convergence').value
        block_dead_end = self.get_parameter('block_convergence_in_dead_end').value

        # Check all conditions
        std_dev_converged = sigma_p < sigma_threshold
        enough_steps = self.step_count >= min_steps
        not_in_dead_end = not self.current_dead_end_status

        # Apply dead end blocking if enabled
        if block_dead_end:
            converged = std_dev_converged and enough_steps and not_in_dead_end
        else:
            converged = std_dev_converged and enough_steps

        # Log convergence status every step for debugging
        self.get_logger().debug(
            f'Convergence check: σ_p={sigma_p:.3f}, σ_t={sigma_threshold:.3f}, '
            f'steps={self.step_count}/{min_steps}, dead_end={self.current_dead_end_status}, converged={converged}'
        )

        # Warn if convergence blocked due to various reasons
        if std_dev_converged and not enough_steps:
            self.get_logger().warn(
                f'Early convergence prevented: σ_p={sigma_p:.3f} < {sigma_threshold:.3f} '
                f'but only {self.step_count} steps (need {min_steps})'
            )

        if block_dead_end and std_dev_converged and enough_steps and self.current_dead_end_status:
            self.get_logger().warn(
                f'Convergence blocked: In dead end! σ_p={sigma_p:.3f} < {sigma_threshold:.3f}, '
                f'steps={self.step_count}>={min_steps}, but BI* too low. Continuing search...'
            )

        if converged:
            self.get_logger().info(
                f'Estimation converged! σ_p = {sigma_p:.3f} < σ_t = {sigma_threshold:.3f} '
                f'after {self.step_count} steps'
            )

        return converged

    def is_source_reached(self):
        """Check if robot reached the estimated source location."""
        if self.current_position is None:
            return False

        # Get estimated source location
        current_means, _ = self.particle_filter.get_estimate()
        est_x = current_means["x"]
        est_y = current_means["y"]

        # Calculate distance to estimated source
        dx = self.current_position[0] - est_x
        dy = self.current_position[1] - est_y
        distance = (dx**2 + dy**2)**0.5

        success_distance = self.get_parameter('success_distance').value
        reached = distance <= success_distance

        if reached:
            self.get_logger().info(f'Source reached! Distance = {distance:.3f}m <= {success_distance:.3f}m')
            self.get_logger().info(f'Estimated source location: ({est_x:.2f}, {est_y:.2f})')

        return reached

    def is_goal_reached(self):
        """Check if robot is close enough to goal (XY only)."""
        if self.goal_position is None or self.current_position is None:
            return False

        dx = self.current_position[0] - self.goal_position[0]
        dy = self.current_position[1] - self.goal_position[1]
        distance = (dx**2 + dy**2)**0.5

        xy_tolerance = self.get_parameter('xy_goal_tolerance').value
        return distance <= xy_tolerance

    def send_nav_goal(self, x, y):
        """Send navigation goal to Nav2."""
        self.goal_position = (float(x), float(y))

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Orientation (facing forward by default)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        # Set behavior tree to skip final rotation
        # This tells Nav2 to ignore orientation tolerance
        goal_msg.behavior_tree = ''  # Use default behavior tree

        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f}) [orientation ignored]')

        self.is_moving = True
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_goal_response_callback)

    def nav_goal_response_callback(self, future):
        """Handle Nav2 goal response."""
        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.get_logger().warn('Navigation goal was rejected!')
            self.is_moving = False
            return

        self.get_logger().info('Navigation goal accepted')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        """Handle Nav2 feedback (optional)."""
        # You can log navigation progress here if needed
        pass

    def nav_result_callback(self, future):
        """Handle Nav2 result."""
        result = future.result().result
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation goal reached!')
        elif status == 5:  # CANCELED
            self.get_logger().warn('Navigation goal was canceled')
        elif status == 6:  # ABORTED
            self.get_logger().warn('Navigation goal was aborted')

        self.is_moving = False
        self.goal_handle = None

    def take_step(self):
        # Check if search is already complete
        if self.search_complete:
            self.get_logger().debug('Search complete, no further action needed')
            return

        # Check if we have necessary data
        if self.sensor_raw_value is None or self.current_position is None:
            self.get_logger().debug('Waiting for sensor and position data...')
            return

        # Initialize sensor thresholds with first measurement
        if not self.sensor_initialized:
            current_measurement = self.ppm_converter.ppm_to_ug_m3(self.sensor_raw_value)
            self.sensor_model.initialize_thresholds(current_measurement)
            self.sensor_initialized = True
            self.get_logger().info(f'Sensor initialized with thresholds based on measurement: {self.sensor_raw_value} ppm = {current_measurement:.2f} μg/m³')
            return

        # Check if robot is currently moving to a goal
        if self.is_moving:
            # Check if we're close enough to the goal (XY only)
            if self.is_goal_reached():
                self.get_logger().info('Goal reached (XY tolerance met), canceling orientation alignment')
                # Cancel the goal to stop orientation alignment
                if self.goal_handle is not None:
                    cancel_future = self.goal_handle.cancel_goal_async()
                    cancel_future.add_done_callback(lambda _: None)
                self.is_moving = False
                self.goal_handle = None
                self.goal_position = None
                return
            else:
                self.get_logger().debug('Robot is moving to goal, waiting...')
                return

        # MEASURE-PLAN-MOVE LOOP

        # 1. MEASURE: Take sensor measurement and update particle filter
        current_measurement_ppm = self.sensor_raw_value
        current_measurement = self.ppm_converter.ppm_to_ug_m3(current_measurement_ppm)

        # Update thresholds adaptively, then compute discrete measurement level
        self.sensor_model.update_thresholds(current_measurement)
        discrete_measurement = self.sensor_model.get_discrete_measurement(current_measurement)

        self.get_logger().info(f'[MEASURE] Sensor reading: {current_measurement:.4f} μg/m³, Discrete level: {discrete_measurement}/{self.sensor_model.num_levels-1}')
        self.particle_filter.update(discrete_measurement, self.current_position)

        # Estimate source location
        current_means, current_stds = self.particle_filter.get_estimate()
        est_x, est_y, est_Q = current_means["x"], current_means["y"], current_means["Q"]

        # 2. PLAN: Use RRT to find best next position
        self.get_logger().info('[PLAN] Computing RRT paths...')
        debug_info = self.rrt.get_next_move_debug(self.current_position, self.particle_filter)
        next_pos = debug_info["next_position"]
        best_path = debug_info["best_path"]
        all_paths = debug_info["all_paths"]

        # Dead end detection (Equations 20-21 from paper)
        # BI* = max(BI) for all branches (Eq. 20)
        bi_optimal = debug_info.get("best_utility", debug_info.get("best_entropy_gain", 0.0))

        # Check if dead end is detected (Eq. 21)
        dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)
        detector_status = self.dead_end_detector.get_status()

        # Update current dead end status (used for convergence blocking)
        self.current_dead_end_status = dead_end_detected

        if dead_end_detected:
            self.get_logger().warn(
                f'[DEAD END DETECTED] BI*={bi_optimal:.4f} < BI_thresh={detector_status["bi_threshold"]:.4f} '
                f'→ Convergence blocked, continuing search'
            )
            # TODO: Switch to global planner when implemented
            # For now, blocks convergence and continues local search
        else:
            self.get_logger().debug(
                f'[LOCAL PLANNER] BI*={bi_optimal:.4f} >= BI_thresh={detector_status["bi_threshold"]:.4f}'
            )

        # Visualize everything in RViz2
        self.visualize_particles(self.particle_filter.particles, self.particle_filter.weights)
        self.visualize_all_paths(all_paths, debug_info.get("all_utilities", None))
        self.visualize_best_path(best_path)
        self.visualize_estimated_source(est_x, est_y)
        self.visualize_current_position(self.current_position)

        # Visualize text info box
        current_stds = self.particle_filter.get_estimate()[1]
        sigma_p = max(current_stds["x"], current_stds["y"])
        current_entropy = self.particle_filter.get_entropy()
        self.text_visualizer.publish_source_info(
            timestamp=self.get_clock().now().to_msg(),
            predicted_x=est_x,
            predicted_y=est_y,
            predicted_z=0.5,  # Assuming source at 0.5m height
            std_dev=sigma_p,
            search_complete=self.search_complete,
            sensor_value=current_measurement,
            binary_value=discrete_measurement,  # Now using discrete level (0 to num_levels-1)
            threshold=self.sensor_model.thresholds[-1] if self.sensor_model.initialized else 0.0,  # Use max threshold
            num_branches=debug_info.get("num_branches", 0),
            best_utility=debug_info.get("best_utility", 0.0),
            best_entropy_gain=debug_info.get("best_entropy_gain", 0.0),
            best_travel_cost=debug_info.get("best_travel_cost", 0.0),
            num_tree_nodes=debug_info.get("num_tree_nodes", 0),
            entropy=current_entropy,
            bi_optimal=bi_optimal,
            bi_threshold=detector_status["bi_threshold"],
            dead_end_detected=dead_end_detected
        )

        # Log data to CSV
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.csv_writer.writerow([
            self.step_count,
            f'{elapsed_time:.2f}',
            f'{current_entropy:.4f}',
            f'{current_stds["x"]:.4f}',
            f'{current_stds["y"]:.4f}',
            f'{current_stds["Q"]:.4f}',
            f'{est_x:.4f}',
            f'{est_y:.4f}',
            f'{est_Q:.4f}',
            f'{current_measurement:.4f}',
            discrete_measurement,  # Discrete level (0 to num_levels-1)
            f'{self.sensor_model.thresholds[-1]:.4f}' if self.sensor_model.initialized else '0.0',  # Max threshold
            debug_info.get("num_branches", 0),
            f'{debug_info.get("best_utility", 0.0):.4f}',
            f'{debug_info.get("best_entropy_gain", 0.0):.4f}',
            f'{debug_info.get("best_travel_cost", 0.0):.4f}',
            f'{self.current_position[0]:.4f}',
            f'{self.current_position[1]:.4f}',
            f'{self.get_parameter("sigma_m").value:.4f}',
            f'{bi_optimal:.4f}',
            f'{detector_status["bi_threshold"]:.4f}',
            1 if dead_end_detected else 0
        ])
        self.log_file.flush()
        self.step_count += 1

        self.get_logger().info(f'[PLAN] Next position: {next_pos}, Estimated source: ({est_x:.2f}, {est_y:.2f}, {est_Q:.2f})')

        # Check finishing conditions (Algorithm 1, line 22 in the paper)
        # Paper only requires estimation convergence (σ_p < σ_t)
        if self.is_estimation_converged():
            self.get_logger().info('='*60)
            self.get_logger().info('SOURCE SEARCH COMPLETED SUCCESSFULLY!')
            self.get_logger().info(f'Final estimated source: ({est_x:.2f}, {est_y:.2f})')
            self.get_logger().info(f'Final estimated release rate: {est_Q:.2f}')
            self.get_logger().info(f'Robot position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})')

            # Calculate distance to estimated source for reference
            if self.current_position is not None:
                dx = self.current_position[0] - est_x
                dy = self.current_position[1] - est_y
                distance = (dx**2 + dy**2)**0.5
                self.get_logger().info(f'Distance to estimated source: {distance:.2f}m')

            self.get_logger().info('='*60)
            self.search_complete = True
            return

        # 3. MOVE: Send goal to Nav2
        self.get_logger().info(f'[MOVE] Sending navigation goal to ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
        self.send_nav_goal(next_pos[0], next_pos[1])

    def __del__(self):
        """Cleanup: Close log file when node is destroyed."""
        if self.log_file is not None:
            self.log_file.close()
            self.get_logger().info('Closed IGDM log file')

def main(args=None):
    """Main function."""
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
