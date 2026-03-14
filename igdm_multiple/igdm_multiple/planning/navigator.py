from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.action import ActionClient
import numpy as np
from math import sin, cos
import time

class Navigator:
    """
    Handles robot navigation (Nav2), recovery (teleport), and state management.
    """
    def __init__(self, node, on_complete_callback=None):
        """
        Args:
            node: The ROS 2 node.
            on_complete_callback: Function to call when navigation finishes (success or fail).
                                  Used to trigger the next planning step in the main node.
        """
        self.node = node
        self.on_complete_callback = on_complete_callback
        
        # State
        self.is_moving = False
        self.goal_handle = None
        self.goal_position = None
        self.consecutive_failures = 0
        self.max_failures_tolerance = 5  # Max consecutive failures before recovery
        self.in_recovery = False
        self.initial_spin_done = False
        self.initial_spin_goal_handle = None
        self._spin_timer = None
        self._spin_start_theta = None
        self._spin_accumulated = 0.0
        self._spin_last_theta = None

        # Interfaces
        self.nav_to_pose_client = ActionClient(node, NavigateToPose, '/PioneerP3DX/navigate_to_pose')
        self.initialpose_pub = node.create_publisher(PoseWithCovarianceStamped, '/PioneerP3DX/initialpose', 10)
        
        self.node.get_logger().info('Waiting for Nav2 action server...')
        self.nav_to_pose_client.wait_for_server()
        self.node.get_logger().info('Nav2 action server available!')

    def send_goal(self, x, y, yaw=0.0, use_orientation=False, is_spin=False, tolerance=0.3):
        """
        Send a navigation goal to Nav2.
        """
        self.goal_position = (float(x), float(y))
        self.xy_goal_tolerance = tolerance
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        
        if use_orientation or is_spin:
            qz = sin(yaw / 2.0)
            qw = cos(yaw / 2.0)
            goal_msg.pose.pose.orientation.z = float(qz)
            goal_msg.pose.pose.orientation.w = float(qw)
        else:
            goal_msg.pose.pose.orientation.w = 1.0

        self.is_moving = True
        future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self._nav_feedback_callback)
        
        if is_spin:
            future.add_done_callback(self._initial_spin_response_callback)
        else:
            future.add_done_callback(self._nav_goal_response_callback)

    def attempt_teleport_recovery(self, current_pos, slam_map, dead_end_detector):
        """
        Attempts to recover from a stuck state by finding a safe spot and teleporting.
        Returns:
            bool: True if teleport succeeded, False if failed (requires global planner fallback).
        """
        self.node.get_logger().warn('!!! STUCK DETECTED - ATTEMPTING TELEPORT RECOVERY !!!')
        self.consecutive_failures = 0
        self.in_recovery = False
        
        safe_pos = self._find_safe_position_away_from_wall(current_pos, slam_map, distance=0.5)
        
        if safe_pos != current_pos:
            self._teleport_robot(safe_pos[0], safe_pos[1])
            # Reset logic state
            time.sleep(1.0)
            dead_end_detector.reset()
            # Signal completion to trigger planning
            if self.on_complete_callback:
                self.on_complete_callback()
            return True
        else:
            self.node.get_logger().warn('Teleport failed. Requesting Global Planner fallback.')
            return False

    def perform_initial_spin(self, current_pos, current_theta):
        """Perform a 360° spin using cmd_vel to populate SLAM map from lidar."""
        if self._spin_timer is not None:
            return  # Already spinning
        if not self.is_moving:
            self.node.get_logger().info('[STARTUP] Starting initial 360° sensor sweep via cmd_vel...')
            self.is_moving = True
            self._spin_start_theta = current_theta
            self._spin_last_theta = current_theta
            self._spin_accumulated = 0.0
            # Publish rotation command at ~10 Hz
            self._spin_timer = self.node.create_timer(0.1, self._spin_timer_callback)

    def _spin_timer_callback(self):
        """Timer callback to drive the spin and check completion."""
        # Get current theta from main node
        current_theta = self.node.current_theta
        if current_theta is None:
            return

        # Track accumulated rotation
        dtheta = current_theta - self._spin_last_theta
        # Normalize to [-pi, pi]
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi
        self._spin_accumulated += abs(dtheta)
        self._spin_last_theta = current_theta

        if self._spin_accumulated >= 2 * np.pi:
            # Spin complete - stop rotation
            stop_msg = Twist()
            self.node.cmd_vel_pub.publish(stop_msg)
            self._spin_timer.cancel()
            self.node.destroy_timer(self._spin_timer)
            self._spin_timer = None
            self.is_moving = False
            self.initial_spin_done = True
            self.node.get_logger().info(
                f'Initial spin complete. {self.node.laser_scan_count} laser scans processed.'
            )
            if self.on_complete_callback:
                self.on_complete_callback()
        else:
            # Keep rotating
            twist = Twist()
            twist.angular.z = 3.0  # rad/s — completes full spin in ~2s sim time
            self.node.cmd_vel_pub.publish(twist)

    # --- Internal Logic / Callbacks ---

    def _nav_feedback_callback(self, feedback_msg):
        if self.goal_position is None or self.goal_handle is None: return
        
        current_pose = feedback_msg.feedback.current_pose.pose.position
        dx = current_pose.x - self.goal_position[0]
        dy = current_pose.y - self.goal_position[1]
        
        # Manual distance check for smooth stopping
        if np.hypot(dx, dy) <= self.xy_goal_tolerance:
            self.node.get_logger().debug('XY goal reached via feedback, canceling for smooth stop.')
            self.goal_handle.cancel_goal_async().add_done_callback(self._goal_cancel_callback)

    def _nav_goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.node.get_logger().warn('Goal rejected by Nav2!')
            self.is_moving = False
            self.consecutive_failures += 1
            # Must trigger planning again or the node freezes
            if self.on_complete_callback:
                self.on_complete_callback()
            return
        self.goal_handle.get_result_async().add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future):
        status = future.result().status
        if status == 4: # SUCCEEDED
            self.consecutive_failures = 0
            self.in_recovery = False
        elif status == 6: # ABORTED
            if not self.in_recovery:
                self.consecutive_failures += 1
        
        self.is_moving = False
        self.goal_handle = None
        
        # Trigger the main node's planning cycle
        if self.on_complete_callback:
            self.on_complete_callback()

    def _initial_spin_response_callback(self, future):
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
        self.node.get_logger().info('Initial spin complete.')
        if self.on_complete_callback:
            self.on_complete_callback()

    def _goal_cancel_callback(self, future):
        pass

    def _teleport_robot(self, target_x, target_y):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(target_x)
        msg.pose.pose.position.y = float(target_y)
        msg.pose.pose.orientation.w = 1.0
        # Just use neutral orientation or preserve current if we had access (omitted for simplicity, w=1 is safe)
        msg.pose.covariance = [0.0] * 36
        self.initialpose_pub.publish(msg)
        self.node.get_logger().info(f'Teleported robot to ({target_x:.2f}, {target_y:.2f})')

    def _find_safe_position_away_from_wall(self, current_pos, slam_map, distance=0.5):
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
                
                # We need to access the map's grid check. 
                # Assuming slam_map object has methods like world_to_grid and grid array.
                if self._is_valid_optimistic(test_pos, slam_map):
                    clearance = self._calculate_clearance(test_pos, slam_map)
                    if clearance > max_clearance:
                        max_clearance = clearance
                        best_position = test_pos

        if best_position is not None and max_clearance >= distance:
            return best_position
        else:
            self.node.get_logger().warn(f'Could not find safe position {distance}m from walls')
            return current_pos

    def _is_valid_optimistic(self, position, slam_map, robot_radius=0.2):
        # Re-implementing the helper here to avoid circular dependency
        gx, gy = slam_map.world_to_grid(*position)
        if gx < 0 or gx >= slam_map.width or gy < 0 or gy >= slam_map.height: return False
        
        radius_cells = int(np.ceil(robot_radius / slam_map.resolution))
        radius_sq = radius_cells ** 2

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_sq: continue
                check_gx, check_gy = gx + dx, gy + dy
                if 0 <= check_gx < slam_map.width and 0 <= check_gy < slam_map.height:
                    if slam_map.grid[check_gy, check_gx] > 0: 
                        return False
        return True

    def _calculate_clearance(self, position, slam_map):
        gx, gy = slam_map.world_to_grid(*position)
        if gx < 0 or gx >= slam_map.width or gy < 0 or gy >= slam_map.height: return 0.0
        max_radius = int(2.0 / slam_map.resolution)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius: continue
                    check_gx = gx + dx
                    check_gy = gy + dy
                    if 0 <= check_gx < slam_map.width and 0 <= check_gy < slam_map.height:
                        if slam_map.grid[check_gy, check_gx] > 0:
                            return radius * slam_map.resolution
        return max_radius * slam_map.resolution