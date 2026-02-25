"""
Simple text box visualizer for RViz to display source estimation information.
"""
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math

class TextVisualizer:
    """Helper class to publish text information to RViz."""

    def __init__(self, publisher, frame_id="map", position_x=8.0, position_y=5.5, position_z=1.5):
        self.publisher = publisher
        self.frame_id = frame_id
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = position_z

    def draw_ascii_bar(self, value, max_value, width=10):
        """Creates a text-based progress bar: [■■■□□□□□□□]"""
        if max_value <= 0: 
            return "[]"
        
        # Clamp value
        normalized = min(max(value / max_value, 0.0), 1.0)
        filled_len = int(normalized * width)
        
        bar = "■" * filled_len + "□" * (width - filled_len)
        return f"[{bar}]"

    def publish_source_info(self, timestamp, predicted_x, predicted_y, predicted_z,
                           std_dev, search_complete, sensor_value, binary_value, 
                           threshold, max_concentration=20.0, num_levels=10, # <--- Added these args
                           num_branches=0, best_utility=0.0, best_entropy_gain=0.0,
                           best_travel_cost=0.0, num_tree_nodes=0, entropy=0.0,
                           bi_optimal=0.0, bi_threshold=0.0, dead_end_detected=False):
        """
        Publish source estimation information as text in RViz.
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
        background.pose.position.x = self.position_x
        background.pose.position.y = self.position_y
        background.pose.position.z = self.position_z
        background.pose.orientation.w = 1.0
        background.scale.x = 2.4  # Slightly wider for the bar
        background.scale.y = 0.05
        background.scale.z = 3.5 
        background.color.r = 1.0
        background.color.g = 1.0
        background.color.b = 1.0
        background.color.a = 0.9

        # Create text marker
        text = Marker()
        text.header.frame_id = self.frame_id
        text.header.stamp = timestamp
        text.ns = "source_estimation_info"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = self.position_x
        text.pose.position.y = self.position_y
        text.pose.position.z = self.position_z
        text.pose.orientation.w = 1.0

        # Status Logic
        if search_complete:
            status = "✓ COMPLETE"
        else:
            status = "⟳ SEARCHING"

        dead_end_status = "⚠ DEAD END!" if dead_end_detected else "✓ OK"
        dead_end_margin = bi_optimal - bi_threshold
        
        # --- NEW VISUALIZATION LOGIC ---
        # Generate the discrete bin bar
        bin_bar = self.draw_ascii_bar(binary_value + 1, num_levels, width=10)
        
        # Generate a continuous level bar (relative to max concentration)
        conc_bar = self.draw_ascii_bar(sensor_value, max_concentration, width=10)

        text.text = (
            f"Predicted Source:\n"
            f"  x: {predicted_x:.2f} m\n"
            f"  y: {predicted_y:.2f} m\n"
            f"  z: {predicted_z:.2f} m\n"
            f"Std Dev: {std_dev:.3f}\n"
            f"Entropy: {entropy:.3f}\n"
            f"-----------------------\n"
            f"Sensor: {sensor_value:.2f} / {max_concentration:.0f} ppm\n"
            f"{conc_bar}\n"
            f"Bin: {binary_value} / {num_levels-1}\n"
            f"{bin_bar}\n"
            f"-----------------------\n"
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

        text.scale.z = 0.2
        text.color.r = 0.0
        text.color.g = 0.0
        text.color.b = 0.0
        text.color.a = 1.0

        marker_array.markers = [background, text]
        self.publisher.publish(marker_array)