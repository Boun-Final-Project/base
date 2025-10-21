#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from gaden_msgs.srv import Occupancy, WindPosition
import numpy as np


class WindVisualizerNode(Node):
    def __init__(self):
        super().__init__('wind_visualizer_node')

        # Callback groups for threading
        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Publisher for wind vectors
        self.wind_pub = self.create_publisher(MarkerArray, '/wind_vectors', 10)

        # Service clients
        self.occupancy_client = self.create_client(
            Occupancy, '/gaden_environment/occupancyMap3D',
            callback_group=self.service_cb_group
        )
        self.wind_client = self.create_client(
            WindPosition, '/wind_value',
            callback_group=self.service_cb_group
        )

        # Parameters
        self.declare_parameter('sampling_resolution', 1.0)  # meters between samples
        self.declare_parameter('arrow_scale', 0.3)  # arrow size multiplier
        self.declare_parameter('update_rate', 1.0)  # Hz
        self.declare_parameter('min_wind_magnitude', 0.001)  # minimum wind to display
        self.declare_parameter('batch_size', 100)  # max points per service call
        self.declare_parameter('sample_height', 0.5)  # z-height to sample wind at

        self.sampling_resolution = self.get_parameter('sampling_resolution').value
        self.arrow_scale = self.get_parameter('arrow_scale').value
        self.update_rate = self.get_parameter('update_rate').value
        self.min_wind_magnitude = self.get_parameter('min_wind_magnitude').value
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.sample_height = self.get_parameter('sample_height').value

        # State
        self.occupancy_data = None
        self.sampling_points = []
        self.pending_wind_request = False
        self.all_wind_data = []  # Store all wind vectors
        self.current_batch_idx = 0  # Track which batch we're processing

        # Wait for services
        self.get_logger().info('Waiting for services...')
        self.wait_for_services()

        # Get occupancy map
        self.get_logger().info('Retrieving occupancy map...')
        self.request_occupancy_map()

    def wait_for_services(self):
        """Wait for required services to be available"""
        while not self.occupancy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for occupancy service...')

        while not self.wind_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for wind service...')

        self.get_logger().info('All services available')

    def request_occupancy_map(self):
        """Request the 3D occupancy map"""
        request = Occupancy.Request()
        future = self.occupancy_client.call_async(request)
        future.add_done_callback(self.occupancy_callback)

    def occupancy_callback(self, future):
        """Handle occupancy map response"""
        try:
            response = future.result()
            self.occupancy_data = {
                'origin': np.array([
                    response.origin.x,
                    response.origin.y,
                    response.origin.z
                ]),
                'resolution': response.resolution,
                'num_cells': np.array([
                    response.num_cells_x,
                    response.num_cells_y,
                    response.num_cells_z
                ]),
                'occupancy': np.array(response.occupancy).reshape(
                    response.num_cells_z,
                    response.num_cells_y,
                    response.num_cells_x
                )
            }
            self.get_logger().info(
                f'Occupancy map retrieved: {self.occupancy_data["num_cells"]} cells, '
                f'resolution: {self.occupancy_data["resolution"]}m'
            )

            # Generate sampling points
            self.get_logger().info('Generating sampling points...')
            self.sampling_points = self.generate_sampling_points()
            self.get_logger().info(f'Generated {len(self.sampling_points)} sampling points')

            # Start timer for periodic updates
            timer_period = 1.0 / self.update_rate
            self.timer = self.create_timer(
                timer_period,
                self.update_wind_visualization,
                callback_group=self.timer_cb_group
            )
            self.get_logger().info('Wind visualizer node initialized')

        except Exception as e:
            self.get_logger().error(f'Failed to retrieve occupancy map: {e}')

    def is_point_free(self, point):
        """Check if a point is in free space (not occupied)"""
        if self.occupancy_data is None:
            return False

        # Convert point to grid indices
        origin = self.occupancy_data['origin']
        resolution = self.occupancy_data['resolution']
        num_cells = self.occupancy_data['num_cells']

        grid_idx = ((point - origin) / resolution).astype(int)

        # Check bounds
        if np.any(grid_idx < 0) or np.any(grid_idx >= num_cells):
            return False

        # Check occupancy (0 = free, >0 = occupied)
        occupancy_value = self.occupancy_data['occupancy'][
            grid_idx[2], grid_idx[1], grid_idx[0]
        ]

        return occupancy_value == 0

    def generate_sampling_points(self):
        """Generate a grid of sampling points in free space"""
        if self.occupancy_data is None:
            return []

        origin = self.occupancy_data['origin']
        resolution = self.occupancy_data['resolution']
        num_cells = self.occupancy_data['num_cells']

        # Calculate world bounds
        max_bounds = origin + num_cells * resolution

        # Generate grid points
        x_points = np.arange(origin[0], max_bounds[0], self.sampling_resolution)
        y_points = np.arange(origin[1], max_bounds[1], self.sampling_resolution)
        z = self.sample_height  # Only sample at specified height

        # Create sampling points at fixed height
        sampling_points = []
        for x in x_points:
            for y in y_points:
                point = np.array([x, y, z])
                if self.is_point_free(point):
                    sampling_points.append(point)

        return sampling_points

    def update_wind_visualization(self):
        """Update wind vector visualization"""
        if len(self.sampling_points) == 0:
            return

        # Skip if we have a pending request
        if self.pending_wind_request:
            return

        # Start from the beginning if we've processed all batches
        if self.current_batch_idx >= len(self.sampling_points):
            self.current_batch_idx = 0

        # Clear wind data when starting a new cycle
        if self.current_batch_idx == 0:
            self.all_wind_data = []

        # Request wind data
        self.pending_wind_request = True

        # Get next batch
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.batch_size, len(self.sampling_points))
        batch = self.sampling_points[start_idx:end_idx]
        batch_indices = (start_idx, end_idx)

        request = WindPosition.Request()
        request.x = [float(p[0]) for p in batch]
        request.y = [float(p[1]) for p in batch]
        request.z = [float(p[2]) for p in batch]

        future = self.wind_client.call_async(request)
        future.add_done_callback(lambda f: self.wind_callback(f, batch, batch_indices))

    def wind_callback(self, future, points, batch_indices):
        """Handle wind data response"""
        self.pending_wind_request = False
        start_idx, end_idx = batch_indices

        try:
            response = future.result()

            # Store wind data for this batch
            for i, point in enumerate(points):
                wind_vector = np.array([
                    response.u[i],
                    response.v[i],
                    response.w[i]
                ])
                self.all_wind_data.append((point, wind_vector))

            # Move to next batch
            self.current_batch_idx = end_idx

            # If we've processed all batches, publish all markers
            if self.current_batch_idx >= len(self.sampling_points):
                self.publish_all_wind_markers()
                self.get_logger().info(
                    f'Collected wind data for all {len(self.all_wind_data)} points',
                    throttle_duration_sec=5.0
                )

        except Exception as e:
            self.get_logger().error(f'Failed to process wind data: {e}')
            self.current_batch_idx = end_idx  # Continue to next batch even on error

    def publish_all_wind_markers(self):
        """Publish all collected wind markers"""
        marker_array = MarkerArray()
        skipped_invalid = 0
        skipped_extreme = 0

        for i, (point, wind_vector) in enumerate(self.all_wind_data):
            # Check for invalid/extreme values
            if np.any(np.isnan(wind_vector)) or np.any(np.isinf(wind_vector)):
                skipped_invalid += 1
                continue

            magnitude = np.linalg.norm(wind_vector)

            # Skip very small wind vectors or extremely large ones (likely errors)
            if magnitude < self.min_wind_magnitude:
                continue

            if magnitude > 10.0:
                self.get_logger().warn(
                    f'Skipping extreme wind vector at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}): '
                    f'magnitude={magnitude:.2f}, vector=({wind_vector[0]:.2f}, {wind_vector[1]:.2f}, {wind_vector[2]:.2f})',
                    throttle_duration_sec=5.0
                )
                skipped_extreme += 1
                continue

            # Create arrow marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "wind_vectors"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Set arrow points (start and end)
            start_point = Point()
            start_point.x = float(point[0])
            start_point.y = float(point[1])
            start_point.z = float(point[2])

            end_point = Point()
            end_point.x = float(point[0] + wind_vector[0])
            end_point.y = float(point[1] + wind_vector[1])
            end_point.z = float(point[2] + wind_vector[2])

            marker.points = [start_point, end_point]

            # Set arrow scale
            marker.scale.x = 0.05 * self.arrow_scale  # shaft diameter
            marker.scale.y = 0.1 * self.arrow_scale   # head diameter
            marker.scale.z = 0.0  # not used for ARROW type

            # Set color based on magnitude (blue to red)
            marker.color = self.wind_magnitude_to_color(magnitude)

            marker.lifetime = rclpy.duration.Duration(seconds=5.0).to_msg()

            marker_array.markers.append(marker)

        # Publish markers
        self.wind_pub.publish(marker_array)

        log_msg = f'Published {len(marker_array.markers)} wind vectors'
        if skipped_invalid > 0:
            log_msg += f' (skipped {skipped_invalid} invalid)'
        if skipped_extreme > 0:
            log_msg += f' (skipped {skipped_extreme} extreme)'

        self.get_logger().info(log_msg, throttle_duration_sec=5.0)

    def wind_magnitude_to_color(self, magnitude):
        """Convert wind magnitude to color (blue = low, red = high)"""
        color = ColorRGBA()

        # Normalize magnitude (adjust these values based on expected wind speeds)
        normalized = min(magnitude / 0.2, 1.0)  # 0.2 m/s = full red

        # Blue to red gradient
        color.r = float(normalized)
        color.g = 0.0
        color.b = float(1.0 - normalized)
        color.a = 0.8  # transparency

        return color


def main(args=None):
    rclpy.init(args=args)

    try:
        node = WindVisualizerNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
