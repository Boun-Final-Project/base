import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from .occupancy_grid import create_occupancy_map_from_service, create_empty_occupancy_map
from .sensor_model import ContinuousGaussianSensorModel
# from .particle_filter import ParticleFilter
from .particle_filter_optimized import ParticleFilterOptimized as ParticleFilter
from .igdm_gas_model import IndoorGaussianDispersionModel
from .rrt import RRT
from .global_planner import GlobalPlanner
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

        # Global planner parameters (Section IV.B.3 from paper)
        self.declare_parameter('prm_samples', 200)  # Number of PRM vertices to sample
        self.declare_parameter('prm_connection_radius', 2.0)  # Max distance for PRM edges (meters)
        self.declare_parameter('frontier_min_size', 3)  # Min cells for valid frontier cluster
        self.declare_parameter('lambda_p', 0.1)  # Weight for path cost in global utility (Eq. 22)
        self.declare_parameter('lambda_s', 0.05)  # Weight for source distance in global utility (Eq. 22)
        self.declare_parameter('switch_back_threshold', 1.5)  # Threshold for switching back to local planner

        # Particle filter diversity parameters
        self.declare_parameter('resample_threshold', 0.5)  # Effective sample size threshold for resampling

        # Sensor readings
        self.sensor_raw_value = None
        self.current_position = None
        self.current_theta = None  # Robot orientation (yaw) for laser transformation
        self.sensor_initialized = False

        # Node initialization flag (prevent callbacks from triggering during __init__)
        self.node_initialized = False

        # State management for measure-plan-move loop
        self.is_moving = False
        self.goal_handle = None
        self.goal_position = None  # Store target goal position
        self.search_complete = False  # Flag to indicate if source search is finished
        self.current_dead_end_status = False  # Track if currently in a dead end
        
        # Synchronization flag
        self.planning_pending = False  # Flag to trigger planning on next fresh pose

        # Dual-mode planner state
        self.planner_mode = 'LOCAL'  # 'LOCAL' or 'GLOBAL'
        self.global_path = []  # Current global path being executed
        self.global_path_index = 0  # Current waypoint index in global path

        # Data logging for entropy tracking
        self.step_count = 0
        self.start_time = None
        self.log_file = None
        self.csv_writer = None
        self._setup_data_logging()

        # Track previous marker counts for proper cleanup
        self.prev_num_paths = 0

        # SLAM map tracking
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

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

        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/PioneerP3DX/laser_scanner',
            self.laser_callback,
            10
        )

        # Publishers for visualization
        self.particle_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/particles', 10)
        self.all_paths_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/all_paths', 10)
        self.best_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/best_path', 10)
        self.estimated_source_pub = self.create_publisher(Marker, '/rrt_infotaxis/estimated_source', 10)
        self.current_pos_pub = self.create_publisher(Marker, '/rrt_infotaxis/current_position', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        # Global planner visualization publishers
        self.frontier_cells_pub = self.create_publisher(Marker, '/rrt_infotaxis/frontier_cells', 10)
        self.frontier_centroids_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/frontier_centroids', 10)
        self.prm_graph_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/prm_graph', 10)
        self.global_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/global_path', 10)
        self.planner_mode_pub = self.create_publisher(Marker, '/rrt_infotaxis/planner_mode', 10)

        # SLAM map publisher with QoS profile compatible with RViz Map display
        map_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Keep latest message for late subscribers
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.slam_map_pub = self.create_publisher(OccupancyGrid, '/rrt_infotaxis/slam_map', map_qos)

        self.consecutive_failures = 0
        self.max_failures_tolerance = 3  # How many fails before triggering recovery
        self.in_recovery = False
        self.recovery_step = 0  # 0: rotate left, 1: rotate right, 2: backup (optional)
        self.initial_spin_done = False
        self.initial_spin_goal_handle = None

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

            # Create empty SLAM map with same dimensions and resolution
            self.slam_map = create_empty_occupancy_map(self.occupancy_map)
            self.get_logger().info(
                f'Empty SLAM map initialized '
                f'(resolution={self.slam_map.resolution}m, '
                f'size={self.slam_map.width}x{self.slam_map.height})'
            )

        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        # Initialize Indoor Gaussian Dispersion Model (IGDM)
        # NOW USING SLAM MAP for obstacle-aware distance calculations
        self.dispersion_model = IndoorGaussianDispersionModel(
            sigma_m=self.get_parameter('sigma_m').value,
            occupancy_grid=self.slam_map
        )
        self.get_logger().info(f'IGDM initialized with σ_m={self.get_parameter("sigma_m").value}m (using SLAM map)')

        # Initialize sensor model and particle filter
        # Using ContinuousGaussianSensorModel (paper's Equation 3)
        self.sensor_model = ContinuousGaussianSensorModel(alpha=0.1, sigma_env=1.0, num_levels=10, max_concentration=20.0)
        self.particle_filter = ParticleFilter(
            num_particles=self.get_parameter('number_of_particles').value,
            search_bounds={"x": (0, self.slam_map.real_world_width), "y": (0, self.slam_map.real_world_height), "Q": (0, 20)},
            binary_sensor_model=self.sensor_model,
            dispersion_model=self.dispersion_model
        )
        self.get_logger().info('Particle filter initialized')

        # Initialize RRT (choose fast or standard implementation)
        use_fast = self.get_parameter('use_fast_rrt').value
        RRTClass = RRT

        self.rrt = RRTClass(
            occupancy_grid=self.slam_map,
            N_tn=self.get_parameter('n_tn').value,
            R_range=self.get_parameter('n_tn').value * self.get_parameter('delta').value,
            delta=self.get_parameter('delta').value,
            robot_radius=self.get_parameter('robot_radius').value,
            positive_weight=self.get_parameter('positive_weight').value
        )
        self.ppm_converter = GasUnitConverter()
        rrt_type = "Fast RRT" if use_fast else "Standard RRT"
        self.get_logger().info(f'{rrt_type} initialized (using SLAM map for collision checking)')

        # Initialize text visualizer
        self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")
        self.get_logger().info('Text visualizer initialized')

        # Initialize dead end detector (Equations 20-21 from paper)
        self.dead_end_detector = DeadEndDetector(
            epsilon=self.get_parameter('dead_end_epsilon').value,
            initial_threshold=self.get_parameter('dead_end_initial_threshold').value
        )
        self.get_logger().info(f'Dead end detector initialized (ε={self.get_parameter("dead_end_epsilon").value})')

        # Initialize global planner (Section IV.B.3 from paper)
        self.global_planner = GlobalPlanner(
            occupancy_grid=self.slam_map,
            robot_radius=self.get_parameter('robot_radius').value,
            prm_samples=self.get_parameter('prm_samples').value,
            prm_connection_radius=self.get_parameter('prm_connection_radius').value,
            frontier_min_size=self.get_parameter('frontier_min_size').value,
            lambda_p=self.get_parameter('lambda_p').value,
            lambda_s=self.get_parameter('lambda_s').value
        )
        self.get_logger().info(f'Global planner initialized (PRM: {self.get_parameter("prm_samples").value} samples)')

        # Event-driven search: take_step() will be called when navigation completes
        # No timer needed - navigation callbacks trigger the next step
        self.get_logger().info('Event-driven search mode: planning triggers on goal completion')

        # Timer for publishing SLAM map at 2 Hz for RViz visualization
        self.slam_map_timer = self.create_timer(0.5, self._publish_slam_map_timer)

        # Timer to check if XY goal is reached (to cancel before final rotation)
        self.goal_check_timer = self.create_timer(0.1, self._check_goal_reached)

        # Mark node as fully initialized
        self.node_initialized = True
        self.get_logger().info('Node initialized successfully, waiting for sensor and pose data...')

    def _setup_data_logging(self):
        """Setup CSV file for logging entropy and other metrics."""
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

    def trigger_recovery(self):
        """Simple recovery behavior: Rotate the robot to clear costmaps or find a new path when stuck."""
        self.in_recovery = True
        self.consecutive_failures = 0
        self.get_logger().warn('!!! STUCK DETECTED - EXECUTING RECOVERY ROTATION !!!')
        
        if self.current_position is None:
            return

        target_yaw = self.current_theta + 1.57  # +90 degrees
        from math import sin, cos
        qz = sin(target_yaw / 2.0)
        qw = cos(target_yaw / 2.0)
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(self.current_position[0])
        goal_msg.pose.pose.position.y = float(self.current_position[1])
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = float(qz)
        goal_msg.pose.pose.orientation.w = float(qw)
        
        self.is_moving = True
        self.get_logger().info('Recovery: Spinning in place...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self.nav_feedback_callback)
        send_goal_future.add_done_callback(self.nav_goal_response_callback)
    
    def pose_callback(self, msg):
        """Callback for robot pose updates."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        from math import atan2
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        theta = atan2(siny_cosp, cosy_cosp)
        self.current_position = (x, y)
        self.current_theta = theta

        if self.planning_pending and not self.is_moving:
            self.get_logger().debug('Pose received, triggering pending plan...')
            self.planning_pending = False
            self.take_step()

        if self.node_initialized and not self.initial_spin_done and not self.is_moving and self.sensor_raw_value is not None and not self.planning_pending:
            self.get_logger().debug('Pose and sensor data ready, triggering initial spin')
            self.take_step()

    def sensor_callback(self, msg):
        """Callback for gas sensor readings."""
        self.sensor_raw_value = msg.raw
        self.get_logger().debug(f'Received sensor reading: {self.sensor_raw_value}')
        if self.node_initialized and not self.initial_spin_done and not self.is_moving and self.current_position is not None and not self.planning_pending:
            self.get_logger().debug('Sensor and pose data ready, triggering initial spin')
            self.take_step()

    def laser_callback(self, msg: LaserScan):
        """Process laser scan and update SLAM map using ground truth pose."""
        if not hasattr(self, 'slam_map') or self.current_position is None or self.current_theta is None:
            return

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

            # Mark free space first (this is where the fix is applied in the helper method)
            self._mark_ray_as_free(robot_x, robot_y, end_x, end_y)

            if hit_obstacle:
                if self._mark_obstacle_in_slam_map(end_x, end_y):
                    obstacles_this_scan += 1

        self.laser_scan_count += 1
        self.total_obstacles_marked += obstacles_this_scan

    def _mark_obstacle_in_slam_map(self, world_x: float, world_y: float) -> bool:
        """Mark obstacle in slam_map with small inflation."""
        gx, gy = self.slam_map.world_to_grid(world_x, world_y)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
            return False
        inflation_cells = 1
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.slam_map.width and 0 <= ny < self.slam_map.height:
                    self.slam_map.grid[ny, nx] = 1
        return True

    def _mark_ray_as_free(self, x0, y0, x1, y1):
        """
        Ray trace from robot to endpoint and mark cells as free.
        FIX: Overwrite obstacles to clear 'ghost walls' caused by timing lag.
        """
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
            # Check bounds
            if 0 <= x < self.slam_map.width and 0 <= y < self.slam_map.height:
                # Stop if we hit the obstacle at the end of the ray
                if x == gx1 and y == gy1:
                    break
                
                # --- CRITICAL FIX ---
                # Always mark as free (0).
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
        """Publish slam_map as OccupancyGrid for RViz visualization."""
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
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        flat_grid = (self.slam_map.grid.flatten() * 100).astype(np.int8)
        msg.data = flat_grid.tolist()
        self.slam_map_pub.publish(msg)

    def _publish_slam_map_timer(self):
        self.publish_slam_map()

    # ... (Rest of the visualization and navigation methods are unchanged) ...
    def visualize_particles(self, particles, weights):
        marker_array = MarkerArray()
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
            marker.color = ColorRGBA()
            marker.color.r = float(weight)
            marker.color.g = float(weight * 0.8)
            marker.color.b = float(1.0 - weight)
            marker.color.a = 0.6
            marker_array.markers.append(marker)
        self.particle_pub.publish(marker_array)

    def visualize_all_paths(self, all_paths, all_utilities=None):
        marker_array = MarkerArray()
        for i in range(self.prev_num_paths):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "all_paths"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        if all_utilities and len(all_utilities) > 0:
            utilities = np.array(all_utilities)
            min_util = utilities.min()
            max_util = utilities.max()
            if max_util - min_util > 1e-6:
                normalized_utilities = (utilities - min_util) / (max_util - min_util)
            else:
                normalized_utilities = np.ones_like(utilities) * 0.5
        else:
            normalized_utilities = None
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
            marker.scale.x = 0.08
            marker.color = ColorRGBA()
            if normalized_utilities is not None and i < len(normalized_utilities):
                norm_util = normalized_utilities[i]
                norm_util_enhanced = norm_util ** 0.5
                if norm_util_enhanced < 0.5:
                    marker.color.r = 1.0
                    marker.color.g = float(2.0 * norm_util_enhanced)
                    marker.color.b = 0.0
                else:
                    marker.color.r = float(2.0 * (1.0 - norm_util_enhanced))
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                marker.color.a = 0.9
            else:
                marker.color.r = 0.6
                marker.color.g = 0.6
                marker.color.b = 0.6
                marker.color.a = 0.5
            marker_array.markers.append(marker)
            num_valid_paths += 1
        self.prev_num_paths = num_valid_paths
        self.all_paths_pub.publish(marker_array)

    def visualize_best_path(self, best_path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_path"
        marker.id = 0
        if len(best_path) < 2:
            marker.action = Marker.DELETE
            self.best_path_pub.publish(marker)
            return
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        for node in best_path:
            point = Point()
            point.x = float(node.position[0])
            point.y = float(node.position[1])
            point.z = 0.6
            marker.points.append(point)
        marker.scale.x = 0.20
        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        self.best_path_pub.publish(marker)

    def visualize_estimated_source(self, est_x, est_y):
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
        current_means, current_stds = self.particle_filter.get_estimate()
        sigma_x = current_stds["x"]
        sigma_y = current_stds["y"]
        sigma_p = max(sigma_x, sigma_y)
        sigma_threshold = self.get_parameter('sigma_threshold').value
        converged = sigma_p < sigma_threshold
        if converged:
            self.get_logger().info(f'Estimation converged! σ_p = {sigma_p:.3f} < σ_t = {sigma_threshold:.3f}')
        return converged

    def is_goal_reached(self):
        if self.goal_position is None or self.current_position is None:
            return False
        dx = self.current_position[0] - self.goal_position[0]
        dy = self.current_position[1] - self.goal_position[1]
        distance = (dx**2 + dy**2)**0.5
        xy_tolerance = self.get_parameter('xy_goal_tolerance').value
        return distance <= xy_tolerance

    def _check_goal_reached(self):
        if not self.is_moving:
            return
        if self.is_goal_reached():
            self.get_logger().info('XY goal reached, canceling navigation to prevent orientation alignment')
            if self.goal_handle is not None:
                cancel_future = self.goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self._goal_cancel_callback)
            else:
                self.is_moving = False
                self.planning_pending = True

    def _goal_cancel_callback(self, future):
        """
        Callback when cancel request is ACCEPTED.
        Do NOT set planning_pending here. We must wait for the Action Server
        to actually stop the robot (handled in nav_result_callback).
        """
        self.get_logger().debug('Cancel request accepted. Waiting for Nav2 to stop...')

        # We can perform global path index update here as it's just bookkeeping
        if self.planner_mode == 'GLOBAL' and self.global_path and self.global_path_index < len(self.global_path):
            self.global_path_index += 1

    def send_nav_goal(self, x, y):
        self.goal_position = (float(x), float(y))
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        from math import sin, cos
        qz = sin(self.current_theta / 2.0)
        qw = cos(self.current_theta / 2.0)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = float(qz)
        goal_msg.pose.pose.orientation.w = float(qw)
        goal_msg.behavior_tree = ''
        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f}) [orientation ignored]')
        self.is_moving = True
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self.nav_feedback_callback)
        send_goal_future.add_done_callback(self.nav_goal_response_callback)

    def nav_goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Navigation goal was rejected!')
            self.is_moving = False
            return
        self.get_logger().info('Navigation goal accepted')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        pass

    def nav_result_callback(self, future):
        """
        Callback when Action Server is completely done (Success, Aborted, or Canceled).
        This guarantees the robot has stopped.
        """
        status = future.result().status
        if status == 4:
            self.get_logger().info('Navigation goal reached!')
            self.consecutive_failures = 0
            self.in_recovery = False
        elif status == 5:
            self.get_logger().warn('Navigation goal was canceled (Robot Stopped)')
        elif status == 6:
            self.get_logger().warn('Navigation goal was aborted')
            if self.in_recovery:
                self.in_recovery = False
            else:
                self.consecutive_failures += 1

        self.is_moving = False
        self.goal_handle = None

        # --- FIX: Only trigger planning here. ---
        # The robot has fully stopped now.
        self.planning_pending = True

    def initial_spin_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Initial spin rejected! Forcing start anyway.')
            self.initial_spin_done = True
            self.is_moving = False
            return
        self.get_logger().info('Initial spin accepted. Sweeping map...')
        self.initial_spin_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.initial_spin_result_callback)

    def initial_spin_result_callback(self, future):
        self.is_moving = False
        self.initial_spin_done = True
        self.get_logger().info('Initial spin complete! Starting gas search.')
        self.planning_pending = True

    def visualize_frontier_cells(self, frontier_cells):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontier_cells"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        for gx, gy in frontier_cells:
            wx, wy = self.slam_map.grid_to_world(gx, gy)
            point = Point()
            point.x = wx
            point.y = wy
            point.z = 0.1
            marker.points.append(point)
        self.frontier_cells_pub.publish(marker)

    def visualize_frontier_centroids(self, frontier_clusters):
        marker_array = MarkerArray()
        for i, cluster in enumerate(frontier_clusters):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontier_centroids"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.centroid_world[0]
            marker.pose.position.y = cluster.centroid_world[1]
            marker.pose.position.z = 0.3
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color = ColorRGBA()
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            marker_array.markers.append(marker)
        self.frontier_centroids_pub.publish(marker_array)

    def visualize_prm_graph(self, prm_vertices):
        marker_array = MarkerArray()
        vertices_marker = Marker()
        vertices_marker.header.frame_id = "map"
        vertices_marker.header.stamp = self.get_clock().now().to_msg()
        vertices_marker.ns = "prm_vertices"
        vertices_marker.id = 0
        vertices_marker.type = Marker.SPHERE_LIST
        vertices_marker.action = Marker.ADD
        vertices_marker.scale.x = 0.15
        vertices_marker.scale.y = 0.15
        vertices_marker.scale.z = 0.15
        vertices_marker.color = ColorRGBA()
        vertices_marker.color.r = 1.0
        vertices_marker.color.g = 1.0
        vertices_marker.color.b = 0.0
        vertices_marker.color.a = 0.5
        for vertex in prm_vertices:
            point = Point()
            point.x = vertex.position[0]
            point.y = vertex.position[1]
            point.z = 0.2
            vertices_marker.points.append(point)
        marker_array.markers.append(vertices_marker)
        edges_marker = Marker()
        edges_marker.header.frame_id = "map"
        edges_marker.header.stamp = self.get_clock().now().to_msg()
        edges_marker.ns = "prm_edges"
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.action = Marker.ADD
        edges_marker.scale.x = 0.02
        edges_marker.color = ColorRGBA()
        edges_marker.color.r = 1.0
        edges_marker.color.g = 1.0
        edges_marker.color.b = 0.0
        edges_marker.color.a = 0.3
        added_edges = set()
        for vertex in prm_vertices:
            for neighbor_id, _ in vertex.neighbors:
                edge = tuple(sorted([vertex.id, neighbor_id]))
                if edge not in added_edges:
                    added_edges.add(edge)
                    p1 = Point()
                    p1.x = vertex.position[0]
                    p1.y = vertex.position[1]
                    p1.z = 0.2
                    neighbor = self.global_planner.vertex_dict[neighbor_id]
                    p2 = Point()
                    p2.x = neighbor.position[0]
                    p2.y = neighbor.position[1]
                    p2.z = 0.2
                    edges_marker.points.append(p1)
                    edges_marker.points.append(p2)
        marker_array.markers.append(edges_marker)
        frontier_vertices_marker = Marker()
        frontier_vertices_marker.header.frame_id = "map"
        frontier_vertices_marker.header.stamp = self.get_clock().now().to_msg()
        frontier_vertices_marker.ns = "frontier_vertices"
        frontier_vertices_marker.id = 2
        frontier_vertices_marker.type = Marker.SPHERE_LIST
        frontier_vertices_marker.action = Marker.ADD
        frontier_vertices_marker.scale.x = 0.25
        frontier_vertices_marker.scale.y = 0.25
        frontier_vertices_marker.scale.z = 0.25
        frontier_vertices_marker.color = ColorRGBA()
        frontier_vertices_marker.color.r = 0.0
        frontier_vertices_marker.color.g = 1.0
        frontier_vertices_marker.color.b = 0.0
        frontier_vertices_marker.color.a = 0.8
        for vertex in prm_vertices:
            if vertex.is_frontier_vertex:
                point = Point()
                point.x = vertex.position[0]
                point.y = vertex.position[1]
                point.z = 0.25
                frontier_vertices_marker.points.append(point)
        marker_array.markers.append(frontier_vertices_marker)
        self.prm_graph_pub.publish(marker_array)

    def visualize_global_path(self, global_path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "global_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        for position in global_path:
            point = Point()
            point.x = position[0]
            point.y = position[1]
            point.z = 0.4
            marker.points.append(point)
        self.global_path_pub.publish(marker)

    def visualize_planner_mode(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planner_mode"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = self.slam_map.origin_x + 1.0
        marker.pose.position.y = self.slam_map.origin_y + self.slam_map.real_world_height - 1.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5
        if self.planner_mode == 'LOCAL':
            marker.text = "MODE: LOCAL (RRT-Infotaxis)"
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.text = "MODE: GLOBAL (Frontier Exploration)"
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        marker.color.a = 1.0
        self.planner_mode_pub.publish(marker)

    def clear_global_planner_visualizations(self):
        """Clear all global planner visualization markers."""
        # Clear frontier cells
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontier_cells"
        marker.id = 0
        marker.action = Marker.DELETE
        self.frontier_cells_pub.publish(marker)

        # Clear frontier centroids
        marker_array = MarkerArray()
        for i in range(100):  # Clear up to 100 centroid markers
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontier_centroids"
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.frontier_centroids_pub.publish(marker_array)

        # Clear PRM graph (vertices, edges, frontier vertices)
        marker_array = MarkerArray()
        for ns in ["prm_vertices", "prm_edges", "frontier_vertices"]:
            for i in range(10):  # Clear up to 10 markers per namespace
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = ns
                marker.id = i
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
        self.prm_graph_pub.publish(marker_array)

        # Clear global path
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "global_path"
        marker.id = 0
        marker.action = Marker.DELETE
        self.global_path_pub.publish(marker)

    def take_step(self):
        if self.search_complete:
            return
        if self.sensor_raw_value is None or self.current_position is None:
            return

        if not self.sensor_initialized:
            self.sensor_initialized = True
            self.get_logger().info(f'Continuous Gaussian sensor model initialized (α={self.sensor_model.alpha}, σ_env={self.sensor_model.sigma_env})')

        if not self.initial_spin_done:
            if not self.is_moving:
                self.get_logger().info('[STARTUP] Starting initial 360° sensor sweep...')
                target_yaw = self.current_theta + 3.14
                from math import sin, cos
                goal_msg = NavigateToPose.Goal()
                goal_msg.pose.header.frame_id = 'map'
                goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
                goal_msg.pose.pose.position.x = float(self.current_position[0])
                goal_msg.pose.pose.position.y = float(self.current_position[1])
                qz = sin(target_yaw / 2.0)
                qw = cos(target_yaw / 2.0)
                goal_msg.pose.pose.orientation.z = float(qz)
                goal_msg.pose.pose.orientation.w = float(qw)
                self.is_moving = True
                future = self.nav_to_pose_client.send_goal_async(goal_msg)
                future.add_done_callback(self.initial_spin_response_callback)
            return

        if self.is_moving:
            return

        self.get_logger().info(f'[STEP {self.step_count}] Robot at ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})')

        current_measurement = self.sensor_raw_value
        self.get_logger().info(f'[MEASURE] Sensor reading: {current_measurement:.4f} ppm (continuous)')
        self.particle_filter.update(current_measurement, self.current_position)
        current_means, current_stds = self.particle_filter.get_estimate()
        est_x, est_y, est_Q = current_means["x"], current_means["y"], current_means["Q"]

        debug_info = None
        best_path = []
        all_paths = []
        bi_optimal = 0.0
        dead_end_detected = False
        next_pos = None

        if self.planner_mode == 'GLOBAL':
            self.get_logger().info('[GLOBAL MODE] Following frontier path...')
            if not self.global_path or self.global_path_index >= len(self.global_path):
                self.get_logger().warn('[GLOBAL MODE] Path exhausted, switching to LOCAL mode')
                self.clear_global_planner_visualizations()
                self.planner_mode = 'LOCAL'
                self.global_path = []
                self.global_path_index = 0
                self.dead_end_detector.reset(initial_threshold=self.get_parameter('dead_end_initial_threshold').value)

            else:
                waypoint = self.global_path[self.global_path_index]
                waypoint_position = tuple(waypoint)
                current_entropy = self.particle_filter.get_entropy()
                expected_entropy = 0.0
                num_measurements = self.sensor_model.num_levels
                for measurement in range(num_measurements):
                    prob_z = self.particle_filter.predict_measurement_probability(waypoint_position, measurement)
                    hyp_entropy = self.particle_filter.compute_hypothetical_entropy(measurement, waypoint_position)
                    expected_entropy += prob_z * hyp_entropy
                mutual_info_waypoint = current_entropy - expected_entropy
                detector_status = self.dead_end_detector.get_status()
                switch_back_threshold = self.get_parameter('switch_back_threshold').value * detector_status["bi_threshold"]

                if mutual_info_waypoint > switch_back_threshold:
                    self.get_logger().info(f'[SWITCH TO LOCAL] Mutual info I={mutual_info_waypoint:.4f} > threshold={switch_back_threshold:.4f}')
                    self.clear_global_planner_visualizations()
                    self.planner_mode = 'LOCAL'
                    self.global_path = []
                    self.global_path_index = 0
                    self.dead_end_detector.reset(initial_threshold=self.get_parameter('dead_end_initial_threshold').value)

                else:
                    next_pos = waypoint
                    self.visualize_global_path(self.global_path)

        if self.planner_mode == 'LOCAL':
            debug_info = self.rrt.get_next_move_debug(self.current_position, self.particle_filter)
            next_pos = debug_info["next_position"]
            best_path = debug_info["best_path"]
            all_paths = debug_info["all_paths"]
            dx_move = next_pos[0] - self.current_position[0]
            dy_move = next_pos[1] - self.current_position[1]
            move_dist = (dx_move**2 + dy_move**2)**0.5
            if move_dist < 0.05:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures_tolerance:
                    self.trigger_recovery()
                    return
            else:
                if self.consecutive_failures > 0:
                    self.consecutive_failures -= 1
            bi_optimal = debug_info.get("best_utility", debug_info.get("best_entropy_gain", 0.0))
            dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)
            if dead_end_detected:
                self.get_logger().warn(f'[DEAD END DETECTED] Switching to GLOBAL mode from ({self.current_position[0]:.3f}, {self.current_position[1]:.3f})')
                self.clear_global_planner_visualizations()
                self.planner_mode = 'GLOBAL'
                global_plan_result = self.global_planner.plan(self.current_position, self.particle_filter)
                if global_plan_result['success']:
                    self.global_path = global_plan_result['best_global_path']
                    self.global_path_index = 0
                    self.visualize_frontier_cells(global_plan_result['frontier_cells'])
                    self.visualize_frontier_centroids(global_plan_result['frontier_clusters'])
                    self.visualize_prm_graph(global_plan_result['prm_vertices'])
                    self.visualize_global_path(self.global_path)
                    next_pos = self.global_path[self.global_path_index]
                    self.global_path_index += 1
                else:
                    self.planner_mode = 'LOCAL'

        self.visualize_planner_mode()
        self.visualize_particles(self.particle_filter.particles, self.particle_filter.weights)
        self.visualize_estimated_source(est_x, est_y)
        self.visualize_current_position(self.current_position)
        if self.planner_mode == 'LOCAL' and debug_info is not None:
            self.visualize_all_paths(all_paths, debug_info.get("all_utilities", None))
            self.visualize_best_path(best_path)

        max_conc = self.sensor_model.max_concentration
        n_levels = self.sensor_model.num_levels
        bin_width = max_conc / n_levels
        current_bin = int(current_measurement / bin_width)
        current_bin = min(current_bin, n_levels - 1)
        sigma_p = max(current_stds["x"], current_stds["y"])
        current_entropy = self.particle_filter.get_entropy()
        detector_status = self.dead_end_detector.get_status()

        self.text_visualizer.publish_source_info(
            timestamp=self.get_clock().now().to_msg(),
            predicted_x=est_x, predicted_y=est_y, predicted_z=0.5,
            std_dev=sigma_p, search_complete=self.search_complete,
            sensor_value=current_measurement, binary_value=current_bin,
            max_concentration=max_conc, num_levels=n_levels, threshold=0.0,
            num_branches=debug_info.get("num_branches", 0) if debug_info else 0,
            best_utility=debug_info.get("best_utility", 0.0) if debug_info else 0.0,
            best_entropy_gain=debug_info.get("best_entropy_gain", 0.0) if debug_info else 0.0,
            best_travel_cost=debug_info.get("best_travel_cost", 0.0) if debug_info else 0.0,
            num_tree_nodes=debug_info.get("num_tree_nodes", 0) if debug_info else 0,
            entropy=current_entropy, bi_optimal=bi_optimal,
            bi_threshold=detector_status["bi_threshold"], dead_end_detected=dead_end_detected
        )
        self.publish_slam_map()
        self.csv_writer.writerow([
            self.step_count, 0, f'{current_entropy:.4f}',
            f'{current_stds["x"]:.4f}', f'{current_stds["y"]:.4f}', f'{current_stds["Q"]:.4f}',
            f'{est_x:.4f}', f'{est_y:.4f}', f'{est_Q:.4f}', f'{current_measurement:.4f}',
            f'{current_measurement:.4f}', '0.0',
            debug_info.get("num_branches", 0) if debug_info else 0,
            f'{debug_info.get("best_utility", 0.0):.4f}' if debug_info else '0.0',
            f'{debug_info.get("best_entropy_gain", 0.0):.4f}' if debug_info else '0.0',
            f'{debug_info.get("best_travel_cost", 0.0):.4f}' if debug_info else '0.0',
            f'{self.current_position[0]:.4f}', f'{self.current_position[1]:.4f}',
            f'{self.get_parameter("sigma_m").value:.4f}', f'{bi_optimal:.4f}',
            f'{detector_status["bi_threshold"]:.4f}', 1 if dead_end_detected else 0,
            self.planner_mode, len(self.global_path) if self.planner_mode == 'GLOBAL' else 0,
            self.global_path_index if self.planner_mode == 'GLOBAL' else 0
        ])
        self.log_file.flush()
        self.step_count += 1
        
        if self.is_estimation_converged():
            self.get_logger().info('SOURCE SEARCH COMPLETED SUCCESSFULLY!')
            self.search_complete = True
            return

        self.get_logger().info(f'Moving to: ({next_pos[0]:.2f}, {next_pos[1]:.2f})')
        self.send_nav_goal(next_pos[0], next_pos[1])

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