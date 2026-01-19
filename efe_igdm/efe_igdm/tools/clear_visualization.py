#!/usr/bin/env python3
"""
Clear all RRT-Infotaxis visualization markers in RViz.

Publishes empty marker arrays to clear:
- Particles
- All RRT paths
- Best path
- Estimated source
- Current position
- Source info text
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class VisualizationClearer(Node):
    """Node to clear all RRT-Infotaxis visualization markers."""

    def __init__(self):
        super().__init__('visualization_clearer')

        # Create publishers for all visualization topics
        self.particle_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/particles', 10)
        self.all_paths_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/all_paths', 10)
        self.best_path_pub = self.create_publisher(Marker, '/rrt_infotaxis/best_path', 10)
        self.estimated_source_pub = self.create_publisher(Marker, '/rrt_infotaxis/estimated_source', 10)
        self.current_pos_pub = self.create_publisher(Marker, '/rrt_infotaxis/current_position', 10)
        self.text_info_pub = self.create_publisher(MarkerArray, '/rrt_infotaxis/source_info_text', 10)

        self.get_logger().info('Visualization clearer node started')

        # Give publishers time to connect
        self.create_timer(0.5, self.clear_once)
        self.cleared = False

    def clear_once(self):
        """Clear all markers once."""
        if self.cleared:
            return

        self.get_logger().info('Clearing all visualization markers...')

        # Create delete marker for single markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        # Create empty marker array for clearing marker arrays
        empty_array = MarkerArray()
        delete_all_marker = Marker()
        delete_all_marker.action = Marker.DELETEALL
        empty_array.markers = [delete_all_marker]

        # Publish delete commands
        self.particle_pub.publish(empty_array)
        self.all_paths_pub.publish(empty_array)
        self.best_path_pub.publish(delete_marker)
        self.estimated_source_pub.publish(delete_marker)
        self.current_pos_pub.publish(delete_marker)
        self.text_info_pub.publish(empty_array)

        self.get_logger().info('✓ Cleared /rrt_infotaxis/particles')
        self.get_logger().info('✓ Cleared /rrt_infotaxis/all_paths')
        self.get_logger().info('✓ Cleared /rrt_infotaxis/best_path')
        self.get_logger().info('✓ Cleared /rrt_infotaxis/estimated_source')
        self.get_logger().info('✓ Cleared /rrt_infotaxis/current_position')
        self.get_logger().info('✓ Cleared /rrt_infotaxis/source_info_text')
        self.get_logger().info('All visualization markers cleared!')

        self.cleared = True

        # Shutdown after clearing
        self.create_timer(1.0, lambda: rclpy.shutdown())


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    node = VisualizationClearer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
