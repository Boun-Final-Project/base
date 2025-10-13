#!/usr/bin/env python3
"""
Visualization script for infotaxis trajectory and source probability distribution.
Subscribes to ROS2 topics and creates plots similar to the test_infotaxis demo.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from olfaction_msgs.msg import GasSensor
from threading import Lock


class InfotaxisVisualizer(Node):
    def __init__(self):
        super().__init__('infotaxis_visualizer')

        # Declare parameters
        self.declare_parameter('robot_namespace', '/PioneerP3DX')
        robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value

        # Data storage
        self.trajectory = []  # List of (x, y) positions
        self.gas_detections = []  # List of detection flags at each position
        self.probability_map = None  # Current probability distribution
        self.map_info = None  # Map metadata
        self.data_lock = Lock()

        # Subscribers
        self.ground_truth_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{robot_namespace}/ground_truth',
            self._ground_truth_callback,
            10
        )

        self.gas_sensor_sub = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self._gas_sensor_callback,
            10
        )

        self.probability_map_sub = self.create_subscription(
            OccupancyGrid,
            '/infotaxis/probability_map',
            self._probability_map_callback,
            10
        )

        # Current gas detection state
        self.current_gas_detected = False
        self.detection_threshold = 1.0

        self.get_logger().info('Infotaxis visualizer initialized. Collecting data...')
        self.get_logger().info('Press Ctrl+C to stop and generate plots.')

    def _ground_truth_callback(self, msg):
        """Store robot position in trajectory."""
        with self.data_lock:
            pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

            # Only add if it's a new position (avoid duplicates)
            if not self.trajectory or np.linalg.norm(np.array(pos) - np.array(self.trajectory[-1])) > 0.01:
                self.trajectory.append(pos)
                self.gas_detections.append(self.current_gas_detected)

    def _gas_sensor_callback(self, msg):
        """Track gas detection state."""
        self.current_gas_detected = msg.raw >= self.detection_threshold

    def _probability_map_callback(self, msg):
        """Store the latest probability map."""
        with self.data_lock:
            # Convert occupancy grid to 2D array
            width = msg.info.width
            height = msg.info.height
            data = np.array(msg.data).reshape((height, width))

            # Convert from 0-100 scale back to probabilities
            self.probability_map = data.astype(float) / 100.0
            self.map_info = msg.info

    def visualize(self):
        """Create visualization plots."""
        with self.data_lock:
            if len(self.trajectory) < 2:
                self.get_logger().error('Not enough trajectory data to visualize')
                return

            traj = np.array(self.trajectory)
            gas_detected = np.array(self.gas_detections)

            self.get_logger().info(f'Visualizing trajectory with {len(traj)} points')
            self.get_logger().info(f'Number of gas detections: {np.sum(gas_detected)}')

            # Create figure
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2, figure=fig)

            # Plot 1: Full trajectory with probability map
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_trajectory_with_probability(ax1, traj, gas_detected)

            # Plot 2: Trajectory only (cleaner view)
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_trajectory_simple(ax2, traj, gas_detected)

            # Plot 3: Probability distribution heatmap
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_probability_heatmap(ax3)

            plt.tight_layout()

            # Save figure
            output_path = '/tmp/infotaxis_visualization.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.get_logger().info(f'Saved visualization to {output_path}')

            plt.show()

    def _plot_trajectory_with_probability(self, ax, traj, gas_detected):
        """Plot trajectory overlaid on probability map."""
        if self.probability_map is not None and self.map_info is not None:
            # Get map extent
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            resolution = self.map_info.resolution
            width = self.map_info.width
            height = self.map_info.height

            extent = [
                origin_x,
                origin_x + width * resolution,
                origin_y,
                origin_y + height * resolution
            ]

            # Plot probability map
            im = ax.imshow(
                self.probability_map,
                origin='lower',
                extent=extent,
                cmap='hot',
                alpha=0.8,
                zorder=0
            )
            plt.colorbar(im, ax=ax, label='Source Probability')

        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'w-', linewidth=2, label='Trajectory', zorder=1)

        # Plot gas detections
        if np.any(gas_detected):
            detection_points = traj[gas_detected]
            ax.scatter(
                detection_points[:, 0],
                detection_points[:, 1],
                marker='D',
                s=50,
                c='cyan',
                edgecolors='blue',
                linewidths=1,
                label='Gas Detected',
                zorder=2
            )

        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=100, c='green',
                  edgecolors='white', linewidths=2, label='Start', zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker='s', s=100, c='red',
                  edgecolors='white', linewidths=2, label='End', zorder=3)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Infotaxis Trajectory with Source Probability Map', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_trajectory_simple(self, ax, traj, gas_detected):
        """Plot trajectory without probability map."""
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.6)

        # Color code by detection
        colors = ['gray' if not det else 'red' for det in gas_detected]
        ax.scatter(traj[:, 0], traj[:, 1], c=colors, s=30, zorder=2)

        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=150, c='green',
                  edgecolors='black', linewidths=2, label='Start', zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker='s', s=150, c='red',
                  edgecolors='black', linewidths=2, label='End', zorder=3)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Trajectory (red = gas detected)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def _plot_probability_heatmap(self, ax):
        """Plot probability distribution as heatmap."""
        if self.probability_map is not None and self.map_info is not None:
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            resolution = self.map_info.resolution
            width = self.map_info.width
            height = self.map_info.height

            extent = [
                origin_x,
                origin_x + width * resolution,
                origin_y,
                origin_y + height * resolution
            ]

            im = ax.imshow(
                self.probability_map,
                origin='lower',
                extent=extent,
                cmap='hot',
                interpolation='bilinear'
            )
            plt.colorbar(im, ax=ax, label='Probability')

            # Find and mark highest probability location
            max_idx = np.unravel_index(np.argmax(self.probability_map), self.probability_map.shape)
            max_y_idx, max_x_idx = max_idx  # Note: array is (height, width)
            max_x = origin_x + max_x_idx * resolution
            max_y = origin_y + max_y_idx * resolution

            ax.scatter(max_x, max_y, marker='*', s=300, c='yellow',
                      edgecolors='black', linewidths=2, label='Most Probable Source', zorder=3)

            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title('Source Probability Distribution', fontsize=12, fontweight='bold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No probability map data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Source Probability Distribution', fontsize=12, fontweight='bold')


def main(args=None):
    rclpy.init(args=args)

    visualizer = InfotaxisVisualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        visualizer.get_logger().info('Stopping data collection and generating plots...')
        visualizer.visualize()
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
