"""
Standalone SLAM node that uses LidarMapper from efe_igdm.

Subscribes to laser scan and pose topics, runs Numba-accelerated Bresenham
ray tracing, and publishes the resulting OccupancyGrid map.

Other algorithms (e.g. ADSM) can subscribe to the published map instead
of implementing their own SLAM.
"""

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid


def euler_from_quaternion_manual(x, y, z, w):
    """Compute (roll, pitch, yaw) from quaternion without tf_transformations."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

from .mapping.occupancy_grid import (
    load_3d_occupancy_grid_from_service,
    create_empty_occupancy_map,
    OccupancyGridMap,
)
from .mapping.lidar_mapper import LidarMapper


class SlamNode(Node):
    """Standalone SLAM node using Python LidarMapper."""

    def __init__(self):
        super().__init__('slam_node')

        # Parameters
        self.declare_parameter('pose_topic', '/PioneerP3DX/ground_truth')
        self.declare_parameter('laser_topic', '/PioneerP3DX/laser_scanner')
        self.declare_parameter('slam_map_topic', '/slam_node/slam_map')
        self.declare_parameter('gaden_occupancy_service', '/gaden_environment/occupancyMap3D')
        self.declare_parameter('z_level', 5)
        self.declare_parameter('publish_rate', 2.0)  # Hz

        pose_topic = self.get_parameter('pose_topic').value
        laser_topic = self.get_parameter('laser_topic').value
        slam_map_topic = self.get_parameter('slam_map_topic').value
        service_name = self.get_parameter('gaden_occupancy_service').value
        z_level = self.get_parameter('z_level').value
        publish_rate = self.get_parameter('publish_rate').value

        # State
        self.current_position = None  # (x, y)
        self.current_theta = None
        self.laser_scan_count = 0
        self.total_obstacles_marked = 0

        # Load map from GADEN occupancy service
        self.get_logger().info('Loading occupancy grid from GADEN service...')
        try:
            grid_2d, self.outlet_mask, params = load_3d_occupancy_grid_from_service(
                self, z_level=z_level, service_name=service_name, timeout_sec=15.0
            )
            self.occupancy_map = OccupancyGridMap(grid_2d, params)

            # GADEN origin correction (matches efe_igdm)
            if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
                self.occupancy_map.origin_x = -0.2
                self.occupancy_map.origin_y = -0.2

            self.slam_map = create_empty_occupancy_map(self.occupancy_map)
            self.get_logger().info(
                f'SLAM map initialized: {self.slam_map.width}x{self.slam_map.height}, '
                f'resolution={self.slam_map.resolution}, '
                f'origin=({self.slam_map.origin_x}, {self.slam_map.origin_y}), '
                f'outlet cells={int(np.sum(self.outlet_mask))}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load occupancy map: {e}')
            raise

        # Initialize LidarMapper
        self.lidar_mapper = LidarMapper(self.slam_map, outlet_mask=self.outlet_mask)

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self.pose_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, laser_topic, self.laser_callback, 10)

        # SLAM map publisher (transient local so late subscribers get last map)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.slam_map_pub = self.create_publisher(OccupancyGrid, slam_map_topic, map_qos)

        # Publish timer
        period = 1.0 / publish_rate
        self.slam_map_timer = self.create_timer(period, self.publish_slam_map)

        self.get_logger().info(
            f'SlamNode ready. Subscribing to pose={pose_topic}, laser={laser_topic}. '
            f'Publishing map on {slam_map_topic} at {publish_rate} Hz.'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion_manual(q.x, q.y, q.z, q.w)
        self.current_position = (p.x, p.y)
        self.current_theta = yaw

    def laser_callback(self, msg: LaserScan):
        if self.current_position is None or self.current_theta is None:
            return

        obstacles_found = self.lidar_mapper.update_from_scan(
            msg,
            self.current_position[0],
            self.current_position[1],
            self.current_theta,
        )
        self.laser_scan_count += 1
        self.total_obstacles_marked += obstacles_found

    # ------------------------------------------------------------------
    # Map publishing (same encoding as efe_igdm)
    # ------------------------------------------------------------------

    def publish_slam_map(self):
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
        msg.info.origin.orientation.w = 1.0

        # Explicit mapping: -1→-1, 0→0, 1→100, 2→50 (outlet)
        grid_data = self.slam_map.grid.flatten()
        ros_grid = np.full_like(grid_data, -1, dtype=np.int8)
        ros_grid[grid_data == 0] = 0
        ros_grid[grid_data == 1] = 100
        ros_grid[grid_data == 2] = 50
        msg.data = ros_grid.tolist()

        self.slam_map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
