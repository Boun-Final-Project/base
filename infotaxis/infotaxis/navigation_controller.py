#!/usr/bin/env python3

import numpy as np
from typing import Optional, Tuple
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped


class NavigationController:
    """Handles robot navigation and movement commands."""

    def __init__(self, node: Node, robot_namespace: str = '/PioneerP3DX'):
        """
        Initialize the navigation controller.

        Args:
            node: ROS2 node for logging and action client
            robot_namespace: Namespace for robot topics/actions
        """
        self.node = node
        self.robot_namespace = robot_namespace

        # Robot state
        self.robot_position: Optional[np.ndarray] = None  # (x, y, z) in world frame
        self.robot_grid_pos: Optional[Tuple[int, int]] = None  # (i, j) in grid frame
        self.robot_orientation: Optional[list] = None  # quaternion [x, y, z, w]

        # Action client for navigation
        self.nav_action_client = ActionClient(
            self.node,
            NavigateToPose,
            f'{robot_namespace}/navigate_to_pose'
        )

        # Publisher for teleporting (setting initial pose)
        self.initialpose_publisher = self.node.create_publisher(
            PoseWithCovarianceStamped,
            f'{robot_namespace}/initialpose',
            10
        )

    def update_position(self, position: np.ndarray, orientation: list):
        """
        Update robot position and orientation.

        Args:
            position: World position [x, y, z]
            orientation: Quaternion [x, y, z, w]
        """
        self.robot_position = position
        self.robot_orientation = orientation

    def update_grid_position(self, grid_pos: Tuple[int, int]):
        """
        Update robot grid position.

        Args:
            grid_pos: Grid indices (i, j)
        """
        self.robot_grid_pos = grid_pos

    def teleport_to_cell(self, grid_cell: Tuple[int, int], grid_manager):
        """
        Teleport robot to specified grid cell.

        Args:
            grid_cell: Target grid indices (i, j)
            grid_manager: GridManager instance for coordinate conversion
        """
        grid_i, grid_j = grid_cell

        # Convert grid indices to world coordinates
        world_pos = grid_manager.get_cell_coordinates(grid_i, grid_j)

        # Create initial pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()

        pose_msg.pose.pose.position.x = world_pos[0]
        pose_msg.pose.pose.position.y = world_pos[1]
        pose_msg.pose.pose.position.z = world_pos[2]

        # Default orientation
        pose_msg.pose.pose.orientation.w = 1.0

        # Set covariance
        pose_msg.pose.covariance = [0.0] * 36
        pose_msg.pose.covariance[0] = 0.001
        pose_msg.pose.covariance[7] = 0.001
        pose_msg.pose.covariance[35] = 0.001

        # Publish
        self.initialpose_publisher.publish(pose_msg)

        self.node.get_logger().info(
            f'Teleporting to cell ({grid_i}, {grid_j}) -> world ({world_pos[0]:.2f}, {world_pos[1]:.2f})'
        )

    def print_position(self, latest_sensor_reading: float):
        """
        Print current robot position.

        Args:
            latest_sensor_reading: Latest gas sensor reading in ppm
        """
        if self.robot_position is not None and self.robot_grid_pos is not None:
            self.node.get_logger().info(
                f'Robot position: '
                f'world=({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}), '
                f'grid=({self.robot_grid_pos[0]}, {self.robot_grid_pos[1]}), '
                f'gas={latest_sensor_reading:.3f} ppm'
            )
        else:
            self.node.get_logger().warn('Robot position not yet available')