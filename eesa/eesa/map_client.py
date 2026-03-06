#!/usr/bin/env python3
"""
MapClient — ROS 2 port of map_client.py from the EESA package.
Subscribes to an OccupancyGrid topic and provides costmap utilities.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from copy import deepcopy


class MapClient:
    """Wraps an OccupancyGrid subscription and provides grid utilities."""

    def __init__(self, node: Node, map_topic: str):
        self.node = node
        self.raw_grid_data = None
        self.grid_data = None
        self.grid_metadata = None
        self.resolution = None
        self.width = None
        self.height = None
        self.origin = None
        self.origin_x = None
        self.origin_y = None
        self.map_frame = None

        # Use transient_local QoS to match typical map publishers (like slam_node)
        slam_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.map_sub = node.create_subscription(
            OccupancyGrid, map_topic, self._map_cb, slam_qos
        )

        # Block until the first map arrives
        self.node.get_logger().info(f'Waiting for map on {map_topic} ...')
        while self.raw_grid_data is None and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.get_logger().info('Map received.')
        self.update_grid_map()

    # ------------------------------------------------------------------
    def _map_cb(self, data: OccupancyGrid):
        self.raw_grid_data = data

    # ------------------------------------------------------------------
    def update_grid_map(self):
        data = deepcopy(self.raw_grid_data)
        self.grid_metadata = data.info
        self.grid_data = np.array(data.data, dtype=np.int8).reshape(
            data.info.height, data.info.width
        )
        self.resolution = self.grid_metadata.resolution
        self.width = self.grid_metadata.width
        self.height = self.grid_metadata.height
        self.origin_x = self.grid_metadata.origin.position.x
        self.origin_y = self.grid_metadata.origin.position.y
        self.origin = (self.origin_x, self.origin_y)
        self.map_frame = data.header.frame_id
        self.node.get_logger().info(
            f'Update grid map: w={self.width} h={self.height} '
            f'origin=({self.origin_x:.2f}, {self.origin_y:.2f})'
        )
        return data

    # ------------------------------------------------------------------
    def get_world_x_y(self, costmap_x, costmap_y):
        world_x = costmap_x * self.resolution + self.origin_x
        world_y = costmap_y * self.resolution + self.origin_y
        return world_x, world_y

    def get_costmap_x_y(self, world_x, world_y):
        costmap_x = int(round((world_x - self.origin_x) / self.resolution))
        costmap_y = int(round((world_y - self.origin_y) / self.resolution))
        return costmap_x, costmap_y

    def get_cost_from_world_x_y(self, x, y):
        cx, cy = self.get_costmap_x_y(x, y)
        return self.get_cost_from_costmap_x_y(cx, cy)

    def get_cost_from_costmap_x_y(self, x, y):
        if self.is_in_gridmap(x, y):
            return self.grid_data[y][x]
        return -10

    def is_in_gridmap(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
