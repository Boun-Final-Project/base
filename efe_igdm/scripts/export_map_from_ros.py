#!/usr/bin/env python3
"""
Export occupancy grid map from ROS2 to PNG image.
This can be used as a background for trajectory plots.

Usage:
    ros2 run igdm export_map
    # or
    python3 export_map_from_ros.py --output my_map.png
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from PIL import Image
import argparse
import sys


class MapExporter(Node):
    """Export occupancy grid map to PNG."""

    def __init__(self, output_file='map.png'):
        super().__init__('map_exporter')
        self.output_file = output_file
        self.map_received = False

        # Subscribe to map topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info('Waiting for map on /map topic...')

    def map_callback(self, msg):
        """Save map when received."""
        if self.map_received:
            return

        self.get_logger().info(f'Received map: {msg.info.width}x{msg.info.height}')
        self.get_logger().info(f'Resolution: {msg.info.resolution} m/pixel')
        self.get_logger().info(f'Origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y})')

        # Convert occupancy grid to image
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))

        # Convert occupancy values to image:
        # -1 (unknown) -> gray (128)
        # 0 (free) -> white (255)
        # 100 (occupied) -> black (0)
        img_data = np.zeros((height, width), dtype=np.uint8)

        img_data[data == -1] = 205  # Unknown = light gray
        img_data[data == 0] = 255   # Free = white
        img_data[data == 100] = 0   # Occupied = black

        # Handle intermediate values (if any)
        mask = (data > 0) & (data < 100)
        img_data[mask] = (255 - data[mask] * 2.55).astype(np.uint8)

        # Flip vertically (ROS convention is bottom-left origin, images are top-left)
        img_data = np.flipud(img_data)

        # Save as PNG
        img = Image.fromarray(img_data, mode='L')
        img.save(self.output_file)

        self.get_logger().info(f'Map saved to {self.output_file}')

        # Save metadata for plotting
        metadata_file = self.output_file.replace('.png', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f'width: {width}\n')
            f.write(f'height: {height}\n')
            f.write(f'resolution: {msg.info.resolution}\n')
            f.write(f'origin_x: {msg.info.origin.position.x}\n')
            f.write(f'origin_y: {msg.info.origin.position.y}\n')

        self.get_logger().info(f'Metadata saved to {metadata_file}')

        self.map_received = True


def main(args=None):
    parser = argparse.ArgumentParser(description='Export ROS2 occupancy grid map to PNG')
    parser.add_argument('--output', '-o', default='map.png',
                       help='Output PNG file path')
    parser.add_argument('--timeout', '-t', type=float, default=10.0,
                       help='Timeout in seconds')

    # Parse known args to allow ROS2 args
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = MapExporter(parsed_args.output)

    # Spin until map received or timeout
    import time
    start_time = time.time()

    while rclpy.ok() and not node.map_received:
        rclpy.spin_once(node, timeout_sec=0.1)

        if time.time() - start_time > parsed_args.timeout:
            node.get_logger().error(f'Timeout waiting for map after {parsed_args.timeout}s')
            break

    if node.map_received:
        node.get_logger().info('Done!')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
