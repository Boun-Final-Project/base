#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from gaden_msgs.srv import Occupancy
from olfaction_msgs.msg import GasSensor
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class InfotaxisNode(Node):
    def __init__(self):
        super().__init__('infotaxis_node')

        # Parameters
        self.declare_parameter('z_level', 5)  # Z-level for 2D slice (sensor height)
        self.z_level = self.get_parameter('z_level').get_parameter_value().integer_value

        self.declare_parameter('detection_threshold', 1.0)  # Gas detection threshold (ppm)
        self.detection_threshold = self.get_parameter('detection_threshold').get_parameter_value().double_value

        self.declare_parameter('step_size', 0.5)  # Step size for movement in meters
        self.step_size = self.get_parameter('step_size').get_parameter_value().double_value

        # State variables
        # Gas detection state
        self.gas_detected = False
        self.latest_sensor_reading = 0.0

        # Robot state
        self.robot_position = None  # (x, y, z) in world frame
        self.robot_grid_pos = None  # (i, j) in grid frame
        self.robot_orientation = None  # quaternion

        # Subscribe to gas sensor
        self.gas_sensor_sub = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self.gas_sensor_callback,
            10
        )

        # Subscribe to robot ground truth
        self.ground_truth_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/PioneerP3DX/ground_truth',
            self.ground_truth_callback,
            10
        )

        # Subscribe to keyboard commands
        self.keyboard_sub = self.create_subscription(
            String,
            '/infotaxis/keyboard_command',
            self.keyboard_command_callback,
            10
        )

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/infotaxis/detection_marker', 10)

        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, '/PioneerP3DX/navigate_to_pose')

        # Create service client to get occupancy grid
        self.occupancy_client = self.create_client(Occupancy, '/gaden_environment/occupancyMap3D')

        # Wait for service to be available
        self.get_logger().info('Waiting for occupancy grid service...')
        while not self.occupancy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Occupancy grid service not available, waiting...')

        # Get occupancy grid from GADEN
        self.get_occupancy_grid()

        # Extract 2D slice at z_level
        self.create_2d_slice()

        # Create infotaxis grid based on step_size
        self.create_infotaxis_grid()

        # Initialize uniform probability distribution
        self.initialize_uniform_distribution()

        # Log initialization info
        self.get_logger().info(f'Infotaxis node initialized')
        self.get_logger().info(f'Z-level: {self.z_level}')
        self.get_logger().info(f'Step size: {self.step_size}m')
        self.get_logger().info(f'Infotaxis grid dimensions: {self.grid_shape}')
        self.get_logger().info(f'Grid origin: {self.grid_origin}')
        self.get_logger().info(f'Number of free cells: {self.num_free_cells}')
        if self.num_free_cells > 0:
            self.get_logger().info(f'Uniform probability per cell: {self.probability_dist[self.occupancy_grid == 0][0]:.6f}')

    def get_occupancy_grid(self):
        """Get occupancy grid from gaden_environment service"""
        request = Occupancy.Request()

        future = self.occupancy_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()

            # Store GADEN grid parameters
            self.env_origin = np.array([
                response.origin.x,
                response.origin.y,
                response.origin.z
            ])
            self.gaden_cell_size = response.resolution
            self.num_cells_3d = np.array([
                response.num_cells_x,
                response.num_cells_y,
                response.num_cells_z
            ])

            # Convert occupancy array to 3D grid
            # Values: 0=free space, 1=obstacle, 2=outlet
            occupancy_array = np.array(response.occupancy, dtype=np.uint8)

            # Reshape to 3D grid (z, y, x)
            self.occupancy_grid_3d = occupancy_array.reshape(
                self.num_cells_3d[2],  # z
                self.num_cells_3d[1],  # y
                self.num_cells_3d[0]   # x
            )

            self.get_logger().info(f'Received GADEN occupancy grid from service')
        else:
            self.get_logger().error('Failed to get occupancy grid from service')
            self.occupancy_grid_3d = None

    def create_2d_slice(self):
        """Extract 2D slice from 3D occupancy grid at specified z-level"""
        if self.occupancy_grid_3d is None:
            self.get_logger().error('Cannot create 2D slice: No 3D occupancy grid available')
            return

        # Clamp z_level to valid range
        self.z_level = np.clip(self.z_level, 0, self.num_cells_3d[2] - 1)

        # Extract 2D slice (y, x)
        self.occupancy_grid_2d = self.occupancy_grid_3d[self.z_level, :, :]

        self.get_logger().info(f'Created 2D slice at z-level {self.z_level}')

    def create_infotaxis_grid(self):
        """Create coarser infotaxis grid based on step_size parameter"""
        # Calculate environment bounds
        env_max_x = self.env_origin[0] + self.gaden_cell_size * self.num_cells_3d[0]
        env_max_y = self.env_origin[1] + self.gaden_cell_size * self.num_cells_3d[1]

        # Calculate grid dimensions based on step_size
        grid_nx = int(np.ceil((env_max_x - self.env_origin[0]) / self.step_size))
        grid_ny = int(np.ceil((env_max_y - self.env_origin[1]) / self.step_size))

        self.grid_shape = (grid_nx, grid_ny)
        self.grid_origin = self.env_origin.copy()

        # Create occupancy grid for infotaxis (using step_size cells)
        self.occupancy_grid = np.zeros((grid_ny, grid_nx), dtype=np.uint8)

        # Mark cells as occupied if any of the underlying GADEN cells are occupied
        for j in range(grid_ny):
            for i in range(grid_nx):
                # Get world coordinates for this infotaxis cell
                x_min = self.grid_origin[0] + i * self.step_size
                y_min = self.grid_origin[1] + j * self.step_size
                x_max = x_min + self.step_size
                y_max = y_min + self.step_size

                # Convert to GADEN grid indices
                gaden_i_min = int((x_min - self.env_origin[0]) / self.gaden_cell_size)
                gaden_j_min = int((y_min - self.env_origin[1]) / self.gaden_cell_size)
                gaden_i_max = int((x_max - self.env_origin[0]) / self.gaden_cell_size)
                gaden_j_max = int((y_max - self.env_origin[1]) / self.gaden_cell_size)

                # Clamp to valid range
                gaden_i_min = np.clip(gaden_i_min, 0, self.num_cells_3d[0] - 1)
                gaden_j_min = np.clip(gaden_j_min, 0, self.num_cells_3d[1] - 1)
                gaden_i_max = np.clip(gaden_i_max, 0, self.num_cells_3d[0])
                gaden_j_max = np.clip(gaden_j_max, 0, self.num_cells_3d[1])

                # Check if any underlying GADEN cell is occupied
                region = self.occupancy_grid_2d[gaden_j_min:gaden_j_max, gaden_i_min:gaden_i_max]
                if region.size > 0 and np.any(region != 0):
                    self.occupancy_grid[j, i] = 1

        self.get_logger().info(f'Created infotaxis grid: {self.grid_shape} cells with {self.step_size}m resolution')

    def initialize_uniform_distribution(self):
        """Initialize uniform probability distribution over free cells"""
        # Count free cells (value = 0)
        self.num_free_cells = np.sum(self.occupancy_grid == 0)

        if self.num_free_cells == 0:
            self.get_logger().error('No free cells in infotaxis grid!')
            return

        # Initialize probability distribution
        self.probability_dist = np.zeros_like(self.occupancy_grid, dtype=np.float64)

        # Set uniform probability for free cells
        uniform_prob = 1.0 / self.num_free_cells
        self.probability_dist[self.occupancy_grid == 0] = uniform_prob

        # Verify the distribution sums to 1
        total_prob = np.sum(self.probability_dist)
        self.get_logger().info(f'Total probability: {total_prob:.10f}')

    def get_cell_coordinates(self, i, j):
        """Convert infotaxis grid indices to world coordinates (cell center)"""
        x = self.grid_origin[0] + (i + 0.5) * self.step_size
        y = self.grid_origin[1] + (j + 0.5) * self.step_size
        z = self.grid_origin[2] + (self.z_level + 0.5) * self.gaden_cell_size
        return np.array([x, y, z])

    def get_cell_indices(self, x, y):
        """Convert world coordinates to infotaxis grid indices"""
        i = int((x - self.grid_origin[0]) / self.step_size)
        j = int((y - self.grid_origin[1]) / self.step_size)

        # Clamp to grid bounds
        i = np.clip(i, 0, self.grid_shape[0] - 1)
        j = np.clip(j, 0, self.grid_shape[1] - 1)

        return i, j

    def ground_truth_callback(self, msg):
        """Callback for robot ground truth position"""
        # Extract position
        self.robot_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract orientation
        self.robot_orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        # Convert to grid coordinates (only if infotaxis grid is loaded)
        if hasattr(self, 'grid_origin') and hasattr(self, 'step_size'):
            self.robot_grid_pos = self.get_cell_indices(self.robot_position[0], self.robot_position[1])

    def keyboard_command_callback(self, msg):
        """Callback for keyboard commands"""
        command = msg.data.lower()

        if command == 'up':
            self.move_up()
        elif command == 'down':
            self.move_down()
        elif command == 'left':
            self.move_left()
        elif command == 'right':
            self.move_right()
        elif command == 'print_position':
            self.print_position()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def print_position(self):
        """Print current robot position"""
        if self.robot_position is not None and self.robot_grid_pos is not None:
            self.get_logger().info(f'Robot position: world=({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}), grid=({self.robot_grid_pos[0]}, {self.robot_grid_pos[1]}), gas={self.latest_sensor_reading:.3f} ppm')
        else:
            self.get_logger().warn('Robot position not yet available')

    def gas_sensor_callback(self, msg):
        """Callback for gas sensor readings"""
        self.latest_sensor_reading = msg.raw

        # Check if detection threshold is exceeded
        previous_state = self.gas_detected
        self.gas_detected = msg.raw >= self.detection_threshold

        # Log detection events
        if self.gas_detected and not previous_state:
            self.get_logger().info(f'Gas DETECTED! Reading: {msg.raw:.3f} ppm (threshold: {self.detection_threshold})')
        elif not self.gas_detected and previous_state:
            self.get_logger().info(f'Gas detection lost. Reading: {msg.raw:.3f} ppm')

        # Publish visualization marker
        self.publish_detection_marker(msg.header)

    def publish_detection_marker(self, header):
        """Publish visualization marker for detection status"""
        marker = Marker()
        marker.header = header
        marker.ns = "gas_detection"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position at robot location (same frame as sensor)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5  # Above robot
        marker.pose.orientation.w = 1.0

        # Size
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Color based on detection status
        if self.gas_detected:
            # Red when gas detected
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
        else:
            # Green when no gas
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.3

        marker.lifetime.sec = 0  # 0 means forever

        self.marker_pub.publish(marker)

    def move_to_grid_cell(self, grid_i, grid_j):
        """Move robot to specified grid cell using navigation action"""
        if self.robot_position is None:
            self.get_logger().error('Cannot move: Robot position not yet available')
            return False

        # Check if cell is valid
        if grid_i < 0 or grid_i >= self.grid_shape[0] or \
           grid_j < 0 or grid_j >= self.grid_shape[1]:
            self.get_logger().error(f'Invalid grid cell: ({grid_i}, {grid_j})')
            return False

        # Check if cell is free
        if self.occupancy_grid[grid_j, grid_i] != 0:
            self.get_logger().error(f'Cell ({grid_i}, {grid_j}) is occupied')
            return False

        # Get world coordinates for target cell
        target_pos = self.get_cell_coordinates(grid_i, grid_j)

        # Calculate distance to target
        distance = np.linalg.norm(target_pos[:2] - self.robot_position[:2])
        self.get_logger().info(f'Moving to cell ({grid_i}, {grid_j}) -> ({target_pos[0]:.2f}, {target_pos[1]:.2f}), distance: {distance:.3f}m')

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target_pos[0]
        goal_msg.pose.pose.position.y = target_pos[1]
        goal_msg.pose.pose.position.z = 0.0

        # Use current orientation or default
        if self.robot_orientation is not None:
            goal_msg.pose.pose.orientation.x = self.robot_orientation[0]
            goal_msg.pose.pose.orientation.y = self.robot_orientation[1]
            goal_msg.pose.pose.orientation.z = self.robot_orientation[2]
            goal_msg.pose.pose.orientation.w = self.robot_orientation[3]
        else:
            goal_msg.pose.pose.orientation.w = 1.0

        # Send goal via action
        self.nav_action_client.wait_for_server()
        send_goal_future = self.nav_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self, future):
        """Callback for goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by navigation server')
            return

        self.get_logger().info('Goal accepted by navigation server')

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Callback for goal result"""
        result = future.result().result
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation succeeded!')
        elif status == 5:  # CANCELED
            self.get_logger().warn('Navigation was canceled')
        elif status == 6:  # ABORTED
            self.get_logger().error('Navigation aborted!')
        else:
            self.get_logger().warn(f'Navigation ended with status: {status}')

    def move_left(self):
        """Move robot one step to the left (-x direction)"""
        if self.robot_grid_pos is None:
            self.get_logger().error('Cannot move: Robot grid position not yet available')
            return False

        new_i = self.robot_grid_pos[0] - 1
        new_j = self.robot_grid_pos[1]
        return self.move_to_grid_cell(new_i, new_j)

    def move_right(self):
        """Move robot one step to the right (+x direction)"""
        if self.robot_grid_pos is None:
            self.get_logger().error('Cannot move: Robot grid position not yet available')
            return False

        new_i = self.robot_grid_pos[0] + 1
        new_j = self.robot_grid_pos[1]
        return self.move_to_grid_cell(new_i, new_j)

    def move_up(self):
        """Move robot one step up (+y direction)"""
        if self.robot_grid_pos is None:
            self.get_logger().error('Cannot move: Robot grid position not yet available')
            return False

        new_i = self.robot_grid_pos[0]
        new_j = self.robot_grid_pos[1] + 1
        return self.move_to_grid_cell(new_i, new_j)

    def move_down(self):
        """Move robot one step down (-y direction)"""
        if self.robot_grid_pos is None:
            self.get_logger().error('Cannot move: Robot grid position not yet available')
            return False

        new_i = self.robot_grid_pos[0]
        new_j = self.robot_grid_pos[1] - 1
        return self.move_to_grid_cell(new_i, new_j)


def main(args=None):
    rclpy.init(args=args)
    node = InfotaxisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
