"""
Simple text box visualizer for RViz to display source estimation information.
"""
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class TextVisualizer:
    """Helper class to publish text information to RViz."""

    def __init__(self, publisher, frame_id="map", position_x=8.0, position_y=5.5, position_z=1.5):
        """
        Initialize the text visualizer.

        Args:
            publisher: ROS2 publisher for Marker/MarkerArray messages
            frame_id: Frame of reference for the text (default: "map")
            position_x: Fixed x position for text box (default: 8.0)
            position_y: Fixed y position for text box (default: 5.5)
            position_z: Fixed z position for text box (default: 1.5)
        """
        self.publisher = publisher
        self.frame_id = frame_id
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = position_z

    def publish_source_info(self, timestamp, predicted_x, predicted_y, predicted_z,
                           std_dev, search_complete, sensor_value, binary_value, threshold,
                           num_branches=0, best_utility=0.0, best_entropy_gain=0.0,
                           best_travel_cost=0.0, num_tree_nodes=0, entropy=0.0,
                           bi_optimal=0.0, bi_threshold=0.0, dead_end_detected=False):
        """
        Publish source estimation information as text in RViz with white background.

        Args:
            timestamp: ROS timestamp
            predicted_x: Predicted x coordinate
            predicted_y: Predicted y coordinate
            predicted_z: Predicted z coordinate
            std_dev: Standard deviation of the estimate
            search_complete: Boolean indicating if search is complete
            sensor_value: Last sensor measurement value
            binary_value: Binary sensor value (0 or 1)
            threshold: Binary sensor threshold value
            num_branches: Number of RRT branches (paths) found
            best_utility: Best utility value (J_total)
            best_entropy_gain: Best entropy gain (J1)
            best_travel_cost: Best travel cost (J2)
            num_tree_nodes: Total number of nodes in RRT tree
            entropy: Shannon entropy of particle distribution
            bi_optimal: Optimal branch information (BI*)
            bi_threshold: Dead end threshold
            dead_end_detected: Whether dead end was detected
        """
        marker_array = MarkerArray()

        # Create white background box
        background = Marker()
        background.header.frame_id = self.frame_id
        background.header.stamp = timestamp
        background.ns = "source_info_background"
        background.id = 0
        background.type = Marker.CUBE
        background.action = Marker.ADD

        # Position in top-right corner (fixed position)
        background.pose.position.x = self.position_x
        background.pose.position.y = self.position_y
        background.pose.position.z = self.position_z
        background.pose.orientation.w = 1.0

        # Background box size (expanded for branch info, entropy, and dead end detection)
        background.scale.x = 2.2  # Width (wider for longer text)
        background.scale.y = 0.05  # Depth (thin)
        background.scale.z = 3.2  # Height (taller to fit all info including dead end)

        # White color with some transparency
        background.color.r = 1.0
        background.color.g = 1.0
        background.color.b = 1.0
        background.color.a = 0.9  # Slightly transparent

        # Create text marker
        text = Marker()
        text.header.frame_id = self.frame_id
        text.header.stamp = timestamp
        text.ns = "source_estimation_info"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD

        # Position same as background (text will be centered)
        text.pose.position.x = self.position_x
        text.pose.position.y = self.position_y
        text.pose.position.z = self.position_z
        text.pose.orientation.w = 1.0

        # Build text content with branch information, entropy, and dead end detection
        # Add visual indicators for clarity
        if search_complete:
            status = "✓ COMPLETE"
        else:
            status = "⟳ SEARCHING"

        dead_end_status = "⚠ DEAD END!" if dead_end_detected else "✓ OK"
        dead_end_margin = bi_optimal - bi_threshold

        text.text = (
            f"Predicted Source:\n"
            f"  x: {predicted_x:.2f} m\n"
            f"  y: {predicted_y:.2f} m\n"
            f"  z: {predicted_z:.2f} m\n"
            f"Std Dev: {std_dev:.3f}\n"
            f"Entropy: {entropy:.3f}\n"
            f"Sensor: {sensor_value:.2f}\n"
            f"Binary: {binary_value}\n"
            f"Threshold: {threshold:.2f}\n"
            f"--- Branch Info (BI) ---\n"
            f"Branches: {num_branches}\n"
            f"Tree Nodes: {num_tree_nodes}\n"
            f"Best Utility: {best_utility:.2f}\n"
            f"  J1 (Entropy): {best_entropy_gain:.2f}\n"
            f"  J2 (Cost): {best_travel_cost:.2f}\n"
            f"--- Dead End Detect ---\n"
            f"BI*: {bi_optimal:.3f}\n"
            f"Threshold: {bi_threshold:.3f}\n"
            f"Margin: {dead_end_margin:+.3f}\n"
            f"Status: {dead_end_status}\n"
            f"Search: {status}"
        )

        # Text appearance - black color
        text.scale.z = 0.2  # Text height in meters
        text.color.r = 0.0
        text.color.g = 0.0
        text.color.b = 0.0
        text.color.a = 1.0  # Fully opaque

        # Add both markers to array and publish
        marker_array.markers = [background, text]
        self.publisher.publish(marker_array)
