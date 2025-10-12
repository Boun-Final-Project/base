#!/usr/bin/env python3

import numpy as np
from typing import Optional, Tuple
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


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

    def move_to_grid_cell(self, grid_i: int, grid_j: int,
                         target_world_pos: np.ndarray,
                         is_cell_valid_func, is_cell_free_func) -> bool:
        """
        Move robot to specified grid cell using navigation action.

        Args:
            grid_i: Target grid x-index
            grid_j: Target grid y-index
            target_world_pos: Target world coordinates [x, y, z]
            is_cell_valid_func: Function to check if cell is valid
            is_cell_free_func: Function to check if cell is free

        Returns:
            True if navigation command was sent successfully, False otherwise
        """
        if self.robot_position is None:
            self.node.get_logger().error('Cannot move: Robot position not yet available')
            return False

        # Check if cell is valid
        if not is_cell_valid_func(grid_i, grid_j):
            self.node.get_logger().error(f'Invalid grid cell: ({grid_i}, {grid_j})')
            return False

        # Check if cell is free
        if not is_cell_free_func(grid_i, grid_j):
            self.node.get_logger().error(f'Cell ({grid_i}, {grid_j}) is occupied')
            return False

        # Calculate distance to target
        distance = np.linalg.norm(target_world_pos[:2] - self.robot_position[:2])
        self.node.get_logger().info(
            f'Moving to cell ({grid_i}, {grid_j}) -> '
            f'({target_world_pos[0]:.2f}, {target_world_pos[1]:.2f}), '
            f'distance: {distance:.3f}m'
        )

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target_world_pos[0]
        goal_msg.pose.pose.position.y = target_world_pos[1]
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
        send_goal_future.add_done_callback(self._goal_response_callback)

        return True

    def move_direction(self, direction: str, grid_manager) -> bool:
        """
        Move robot one step in specified direction.

        Args:
            direction: One of 'up', 'down', 'left', 'right'
            grid_manager: GridManager instance for coordinate conversion

        Returns:
            True if navigation command was sent successfully, False otherwise
        """
        if self.robot_grid_pos is None:
            self.node.get_logger().error('Cannot move: Robot grid position not yet available')
            return False

        # Calculate target grid position based on direction
        direction_map = {
            'up': (0, 1),      # +y
            'down': (0, -1),   # -y
            'left': (-1, 0),   # -x
            'right': (1, 0)    # +x
        }

        if direction not in direction_map:
            self.node.get_logger().error(f'Invalid direction: {direction}')
            return False

        di, dj = direction_map[direction]
        new_i = self.robot_grid_pos[0] + di
        new_j = self.robot_grid_pos[1] + dj

        # Get world coordinates
        target_pos = grid_manager.get_cell_coordinates(new_i, new_j)

        # Move to target cell
        return self.move_to_grid_cell(
            new_i, new_j, target_pos,
            grid_manager.is_cell_valid,
            grid_manager.is_cell_free
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

    def _goal_response_callback(self, future):
        """Callback for goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('Goal rejected by navigation server')
            return

        self.node.get_logger().info('Goal accepted by navigation server')

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        """Callback for goal result."""
        result = future.result().result
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.node.get_logger().info('Navigation succeeded!')
        elif status == 5:  # CANCELED
            self.node.get_logger().warn('Navigation was canceled')
        elif status == 6:  # ABORTED
            self.node.get_logger().error('Navigation aborted!')
        else:
            self.node.get_logger().warn(f'Navigation ended with status: {status}')
