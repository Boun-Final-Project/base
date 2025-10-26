"""
Simple text box visualizer for RViz to display source estimation information.
"""
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class TextVisualizer:
    """Helper class to publish text information to RViz."""

    def __init__(self, publisher, frame_id="map", position_x=8.0, position_y=5.5, position_z=2.0):
        """
        Initialize the text visualizer.

        Args:
            publisher: ROS2 publisher for Marker/MarkerArray messages
            frame_id: Frame of reference for the text (default: "map")
            position_x: Fixed x position for text box (default: 8.0)
            position_y: Fixed y position for text box (default: 5.5)
            position_z: Fixed z position for text box (default: 2.0)
        """
        self.publisher = publisher
        self.frame_id = frame_id
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = position_z

    def publish_source_info(self, timestamp, predicted_x, predicted_y, predicted_z,
                           std_dev, search_complete):
        """
        Publish source estimation information as text in RViz with white background.

        Args:
            timestamp: ROS timestamp
            predicted_x: Predicted x coordinate
            predicted_y: Predicted y coordinate
            predicted_z: Predicted z coordinate
            std_dev: Standard deviation of the estimate
            search_complete: Boolean indicating if search is complete
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

        # Background box size
        background.scale.x = 1.8  # Width
        background.scale.y = 0.05  # Depth (thin)
        background.scale.z = 1.2  # Height

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

        # Build text content
        status = "COMPLETE" if search_complete else "SEARCHING"
        text.text = (
            f"Predicted Source:\n"
            f"  x: {predicted_x:.2f} m\n"
            f"  y: {predicted_y:.2f} m\n"
            f"  z: {predicted_z:.2f} m\n"
            f"Std Dev: {std_dev:.3f}\n"
            f"Status: {status}"
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
