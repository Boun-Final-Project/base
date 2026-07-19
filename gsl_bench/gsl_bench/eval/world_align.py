"""Align GADEN's world-frame markers into the SLAM map frame for visualization.

The problem: GADEN publishes gas/source markers in a TF frame literally named
`"map"`, but their coordinates are WORLD coordinates (origin at world 0,0). In
realistic mode the SLAM stack's `"map"` frame is anchored at the robot's START
pose, not world 0,0. Same frame name, different origin, so the gas plume would
render offset from the SLAM map by exactly the robot-start translation.

This node bridges the two without renaming the load-bearing SLAM `"map"` frame:

  1. Broadcasts a static TF `map -> gaden_world` that places the world origin at
     `-start` inside the SLAM frame (a pure translation when the start yaw is 0,
     which it is for every benchmark scenario; the general rotated case is handled
     too).
  2. Relays GADEN's `/Gas_Distribution` and `/source_visualization` markers onto
     new topics with `frame_id` rewritten to `gaden_world`, so RViz (fixed frame
     `map`) transforms them by our static TF and draws them at the right spot.

No GADEN rebuild: geometry is untouched, only the header frame is re-stamped, and
the relay publishes to *new* topics so there is no self-loop.

    ros2 run gsl_bench world_align --ros-args \
        -p start_x:=11.0 -p start_y:=2.0 -p yaw:=0.0
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

WORLD_FRAME = 'gaden_world'
SLAM_FRAME = 'map'


class WorldAlign(Node):
    def __init__(self):
        super().__init__('gsl_world_align')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('yaw', 0.0)
        sx = float(self.get_parameter('start_x').value)
        sy = float(self.get_parameter('start_y').value)
        yaw = float(self.get_parameter('yaw').value)

        self._broadcast_static_tf(sx, sy, yaw)

        # Relay GADEN markers (frame "map"/world) -> aligned topics (frame gaden_world).
        self._gas_pub = self.create_publisher(Marker, '/gsl_bench/gas_aligned', 1)
        self._src_pub = self.create_publisher(MarkerArray, '/gsl_bench/source_aligned', 1)
        self.create_subscription(Marker, '/Gas_Distribution', self._on_gas, 1)
        self.create_subscription(MarkerArray, '/source_visualization', self._on_src, 1)
        self.get_logger().info(
            f'world_align: map->{WORLD_FRAME} at start=({sx:.2f},{sy:.2f}) yaw={yaw:.3f}; '
            f'relaying gas/source to /gsl_bench/*_aligned')

    def _broadcast_static_tf(self, sx: float, sy: float, yaw: float):
        # We want a world point p_w to render at p_map = R(-yaw)*(p_w - start).
        # A TF map->gaden_world maps p_map = R_q * p_w + t, so R_q = R(-yaw) and
        # t = -R(-yaw)*start.
        rot = -yaw
        c, s = math.cos(rot), math.sin(rot)
        tx = -(c * sx - s * sy)
        ty = -(s * sx + c * sy)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = SLAM_FRAME
        t.child_frame_id = WORLD_FRAME
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.rotation.z = math.sin(rot / 2.0)
        t.transform.rotation.w = math.cos(rot / 2.0)
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform(t)

    def _on_gas(self, msg: Marker):
        msg.header.frame_id = WORLD_FRAME
        self._gas_pub.publish(msg)

    def _on_src(self, msg: MarkerArray):
        for m in msg.markers:
            m.header.frame_id = WORLD_FRAME
        self._src_pub.publish(msg)


def main(argv=None):
    rclpy.init(args=argv)
    node = WorldAlign()
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
