#!/usr/bin/env python3
"""
RobotClient — ROS 2 port of robot_client.py from the EESA package.
Handles pose (via TF2), sensor subscriptions, and Nav2 navigation.
"""

import math
import copy

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from olfaction_msgs.msg import Anemometer, GasSensor
from nav2_msgs.action import NavigateToPose, ComputePathToPose

import tf2_ros

from eesa.map_client import MapClient


def euler_from_quaternion(q):
    """Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = q
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x, y, z, w]."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


class RobotClient:
    """Manages robot pose, sensors, and Nav2-based navigation."""

    def __init__(
        self,
        node: Node,
        map_frame: str,
        robot_base_frame: str,
        real_pose_topic: str,
        anemometer_topic: str,
        gas_sensor_topic: str,
        nav_action_name: str,
        sensor_window: int = 5,
    ):
        self.node = node
        self.map_frame = map_frame
        self.robot_frame = robot_base_frame
        self.sensor_window = sensor_window

        # Pose state
        self.pose = None
        self.x = None
        self.y = None
        self.z = None
        self.yaw = None
        self.raw_real_pose = None
        self.real_pose = None
        self.real_x = None
        self.real_y = None
        self.real_z = None
        self.real_yaw = None

        # Sensor state
        self.raw_anemometer = []
        self.wind_direction = None
        self.wind_speed = None
        self.raw_gas = []
        self.gas = None

        # Navigation state
        self.move_base_done = True
        self._nav_goal_handle = None

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # Wait for TF
        self.node.get_logger().info(
            f'Waiting for TF {self.map_frame} -> {self.robot_frame} ...'
        )
        while rclpy.ok():
            try:
                self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.robot_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0),
                )
                break
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                continue

        # Real-pose subscriber (PoseWithCovarianceStamped in this environment)
        self.real_pose_sub = self.node.create_subscription(
            PoseWithCovarianceStamped, real_pose_topic, self._real_pose_cb, 50
        )
        self.node.get_logger().info(f'Waiting for real pose on {real_pose_topic} ...')
        while self.raw_real_pose is None and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Anemometer subscriber
        self.anemometer_sub = self.node.create_subscription(
            Anemometer, anemometer_topic, self._anemometer_cb, 10
        )
        self.node.get_logger().info(f'Waiting for anemometer on {anemometer_topic} ...')
        while len(self.raw_anemometer) == 0 and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Gas sensor subscriber
        self.gas_sensor_sub = self.node.create_subscription(
            GasSensor, gas_sensor_topic, self._gas_sensor_cb, 2
        )
        self.node.get_logger().info(f'Waiting for gas sensor on {gas_sensor_topic} ...')
        while len(self.raw_gas) == 0 and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Nav2 NavigateToPose action client
        self._nav_client = ActionClient(self.node, NavigateToPose, nav_action_name)
        self.node.get_logger().info(f'Waiting for Nav2 action server {nav_action_name} ...')
        while not self._nav_client.server_is_ready() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Nav2 ComputePathToPose action client (for plan checking)
        compute_path_name = nav_action_name.rsplit('/', 1)[0] + '/compute_path_to_pose'
        self._compute_path_client = ActionClient(
            self.node, ComputePathToPose, compute_path_name
        )

        self.node.get_logger().info('Robot client initialised.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _real_pose_cb(self, msg: PoseWithCovarianceStamped):
        self.raw_real_pose = msg

    def _anemometer_cb(self, msg: Anemometer):
        self.raw_anemometer.append(msg)
        while len(self.raw_anemometer) > self.sensor_window:
            self.raw_anemometer.pop(0)

    def _gas_sensor_cb(self, msg: GasSensor):
        self.raw_gas.append(msg.raw)
        while len(self.raw_gas) > self.sensor_window:
            self.raw_gas.pop(0)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------
    def update_pose(self):
        """Update robot pose from TF2."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rclpy.time.Time()
            )
            self.x = t.transform.translation.x
            self.y = t.transform.translation.y
            self.z = t.transform.translation.z
            q = t.transform.rotation
            _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.node.get_logger().info(
                f'Robot pose: x={self.x:.2f} y={self.y:.2f} yaw={self.yaw:.2f}'
            )
        except Exception:
            pass
        return self.pose

    def update_real_pose(self):
        self.real_pose = copy.deepcopy(self.raw_real_pose)
        p = self.real_pose.pose.pose
        self.real_x = p.position.x
        self.real_y = p.position.y
        self.real_z = p.position.z
        q = p.orientation
        _, _, self.real_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.node.get_logger().info(
            f'Robot real pose: x={self.real_x:.2f} y={self.real_y:.2f} yaw={self.real_yaw:.2f}'
        )
        return self.real_pose

    def update_anemometer(self):
        anemometer = copy.deepcopy(self.raw_anemometer)
        wd_x = [math.cos(item.wind_direction) for item in anemometer]
        wd_y = [math.sin(item.wind_direction) for item in anemometer]
        self.wind_direction = math.atan2(sum(wd_y), sum(wd_x))
        self.wind_speed = sum(item.wind_speed for item in anemometer) / len(anemometer)
        self.node.get_logger().info(
            f'Anemometer: dir={self.wind_direction:.2f} speed={self.wind_speed:.2f}'
        )

    def update_gas(self):
        gas_data = copy.deepcopy(self.raw_gas)
        self.gas = sum(gas_data) / len(gas_data)
        self.node.get_logger().info(f'Gas sensor: {self.gas:.2f}')

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def check_plan(self, x, y, yaw=None):
        """Check if Nav2 can compute a path to (x, y). Returns True if valid."""
        self.node.get_logger().info(f'Check plan: x={x:.2f} y={y:.2f}')
        if not self._compute_path_client.server_is_ready():
            self.node.get_logger().warn('ComputePathToPose not ready, assuming path valid.')
            return True

        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = PoseStamped()
        goal_msg.start.header.frame_id = self.map_frame
        goal_msg.start.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.start.pose.position.x = float(self.x)
        goal_msg.start.pose.position.y = float(self.y)
        goal_msg.start.pose.orientation.w = 1.0

        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = self.map_frame
        goal_msg.goal.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = float(x)
        goal_msg.goal.pose.position.y = float(y)
        goal_msg.goal.pose.orientation.w = 1.0

        future = self._compute_path_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        if future.result() is None:
            return False
        goal_handle = future.result()
        if not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)
        if result_future.result() is None:
            return False
        result = result_future.result().result
        return len(result.path.poses) > 0

    def send_goal(self, x, y, yaw=None):
        """Send a NavigateToPose goal to Nav2."""
        self.node.get_logger().info(f'Send goal: x={x:.2f} y={y:.2f}')
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        if yaw is None:
            goal_msg.pose.pose.orientation.w = 1.0
        else:
            q = quaternion_from_euler(0.0, 0.0, yaw)
            goal_msg.pose.pose.orientation.x = q[0]
            goal_msg.pose.pose.orientation.y = q[1]
            goal_msg.pose.pose.orientation.z = q[2]
            goal_msg.pose.pose.orientation.w = q[3]

        future = self._nav_client.send_goal_async(
            goal_msg, feedback_callback=self._nav_feedback_cb
        )
        future.add_done_callback(self._nav_goal_response_cb)
        self.move_base_done = False

    def _nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn('Nav2 goal rejected.')
            self.move_base_done = True
            return
        self._nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_cb)

    def _nav_feedback_cb(self, feedback_msg):
        pass

    def _nav_result_cb(self, future):
        self.move_base_done = True
        self._nav_goal_handle = None
        self.node.get_logger().info('Nav2 goal done.')

    def cancel_move(self):
        """Cancel all outstanding Nav2 goals."""
        if self._nav_goal_handle is not None:
            self.node.get_logger().info('Cancelling Nav2 goal ...')
            cancel_future = self._nav_goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=3.0)
        self.move_base_done = True
        self.node.get_logger().info('Cancel done.')
