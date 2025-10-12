#!/usr/bin/env python3

from rclpy.node import Node
from olfaction_msgs.msg import GasSensor
from visualization_msgs.msg import Marker
from std_msgs.msg import Header


class SensorHandler:
    """Handles gas sensor readings and visualization."""

    def __init__(self, node: Node, detection_threshold: float):
        """
        Initialize the sensor handler.

        Args:
            node: ROS2 node for logging and publishing
            detection_threshold: Gas detection threshold in ppm
        """
        self.node = node
        self.detection_threshold = detection_threshold

        # Gas detection state
        self.gas_detected: bool = False
        self.latest_sensor_reading: float = 0.0

        # Publisher for visualization marker
        self.marker_pub = self.node.create_publisher(
            Marker,
            '/infotaxis/detection_marker',
            10
        )

    def process_sensor_reading(self, msg: GasSensor):
        """
        Process gas sensor reading and update detection state.

        Args:
            msg: GasSensor message
        """
        self.latest_sensor_reading = msg.raw

        # Check if detection threshold is exceeded
        previous_state = self.gas_detected
        self.gas_detected = msg.raw >= self.detection_threshold

        # Log detection events
        if self.gas_detected and not previous_state:
            self.node.get_logger().info(
                f'Gas DETECTED! Reading: {msg.raw:.3f} ppm '
                f'(threshold: {self.detection_threshold})'
            )
        elif not self.gas_detected and previous_state:
            self.node.get_logger().info(f'Gas detection lost. Reading: {msg.raw:.3f} ppm')

        # Publish visualization marker
        self.publish_detection_marker(msg.header)

    def publish_detection_marker(self, header: Header):
        """
        Publish visualization marker for detection status.

        Args:
            header: Header from sensor message
        """
        marker = Marker()
        marker.header = header
        marker.ns = "gas_detection"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position at robot location (same frame as sensor)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5  # Above robot
        marker.pose.orientation.w = 1.0

        # Size
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Color based on detection status
        if self.gas_detected:
            # Red when gas detected
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
        else:
            # Green when no gas
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.3

        marker.lifetime.sec = 0  # 0 means forever

        self.marker_pub.publish(marker)

    def get_latest_reading(self) -> float:
        """
        Get the latest gas sensor reading.

        Returns:
            Latest sensor reading in ppm
        """
        return self.latest_sensor_reading

    def is_gas_detected(self) -> bool:
        """
        Check if gas is currently detected.

        Returns:
            True if gas is detected above threshold
        """
        return self.gas_detected
