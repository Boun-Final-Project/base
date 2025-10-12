#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from gaden_msgs.srv import Occupancy
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

from .grid_manager import GridManager
from .navigation_controller import NavigationController
from .sensor_handler import SensorHandler


class InfotaxisNode(Node):
    def __init__(self):
        super().__init__('infotaxis_node')

        # Declare parameters
        self.declare_parameter('z_level', 5)
        self.declare_parameter('detection_threshold', 1.0)
        self.declare_parameter('step_size', 0.5)
        self.declare_parameter('robot_namespace', '/PioneerP3DX')

        # Get parameter values
        z_level = self.get_parameter('z_level').get_parameter_value().integer_value
        detection_threshold = self.get_parameter('detection_threshold').get_parameter_value().double_value
        step_size = self.get_parameter('step_size').get_parameter_value().double_value
        robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value

        # Initialize modules
        self.grid_manager = GridManager(self, z_level, step_size)
        self.navigation_controller = NavigationController(self, robot_namespace)
        self.sensor_handler = SensorHandler(self, detection_threshold)

        # Subscribe to gas sensor
        self.gas_sensor_sub = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self._gas_sensor_callback,
            10
        )

        # Subscribe to robot ground truth
        self.ground_truth_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{robot_namespace}/ground_truth',
            self._ground_truth_callback,
            10
        )

        # Subscribe to keyboard commands
        self.keyboard_sub = self.create_subscription(
            String,
            '/infotaxis/keyboard_command',
            self._keyboard_command_callback,
            10
        )

        # Create service client to get occupancy grid
        self.occupancy_client = self.create_client(Occupancy, '/gaden_environment/occupancyMap3D')

        # Wait for service to be available
        self.get_logger().info('Waiting for occupancy grid service...')
        while not self.occupancy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Occupancy grid service not available, waiting...')

        # Initialize grid manager
        if self.grid_manager.initialize(self.occupancy_client):
            self.get_logger().info('Infotaxis node initialized successfully')
        else:
            self.get_logger().error('Failed to initialize infotaxis node')

    def _ground_truth_callback(self, msg):
        """Callback for robot ground truth position."""
        # Extract position
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract orientation
        orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        # Update navigation controller
        self.navigation_controller.update_position(position, orientation)

        # Convert to grid coordinates (only if grid is initialized)
        if self.grid_manager.grid_origin is not None:
            grid_pos = self.grid_manager.get_cell_indices(position[0], position[1])
            self.navigation_controller.update_grid_position(grid_pos)

    def _keyboard_command_callback(self, msg):
        """Callback for keyboard commands."""
        command = msg.data.lower()

        if command in ['up', 'down', 'left', 'right']:
            self.navigation_controller.move_direction(command, self.grid_manager)
        elif command == 'print_position':
            self.navigation_controller.print_position(
                self.sensor_handler.get_latest_reading()
            )
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def _gas_sensor_callback(self, msg):
        """Callback for gas sensor readings."""
        self.sensor_handler.process_sensor_reading(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InfotaxisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
