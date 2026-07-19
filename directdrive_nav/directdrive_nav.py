#!/usr/bin/env python3
"""DirectDrive NavigateToPose server — a drop-in replacement for Nav2's
navigator that drives via rotate-then-go cmd_vel along an A* path on a STATIC
occupancy map. It never issues the wall-hugging curved velocity commands that
BasicSim's all-or-nothing collision model (Robot.cpp: discards any motion whose
endpoint isn't free) freezes on, so it threads tight houses where Nav2+BasicSim
wedges.

ADSM fire-and-forgets NavigateToPose goals at ~1 Hz, so each accepted goal just
(re)plans an A* path to the target and returns immediately; a 10 Hz control
timer continuously drives toward the latest path's next waypoint.

Run (static ground-truth map, ground-truth pose — original mode):
  python3 directdrive_nav.py --ros-args \
     -p map_yaml:=<...>/occupancy.yaml \
     -p action_name:=/PioneerP3DX/dd_navigate_to_pose \
     -p pose_topic:=/PioneerP3DX/ground_truth \
     -p cmd_vel_topic:=/PioneerP3DX/cmd_vel

Run (live SLAM map, noisy TF pose — realistic mode):
  python3 directdrive_nav.py --ros-args \
     -p map_topic:=/PioneerP3DX/map \
     -p pose_source:=tf \
     -p map_frame:=map -p base_frame:=PioneerP3DX_base_link \
     -p action_name:=/PioneerP3DX/navigate_to_pose \
     -p cmd_vel_topic:=/PioneerP3DX/cmd_vel
"""
import os
import math
import heapq
import time
import numpy as np
import yaml
from scipy.ndimage import binary_dilation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import tf2_ros
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DirectDriveNav(Node):
    def __init__(self):
        super().__init__('directdrive_navigator')
        gp = self.declare_parameter
        map_yaml = gp('map_yaml', '').value
        map_topic = gp('map_topic', '').value
        action_name = gp('action_name', '/PioneerP3DX/dd_navigate_to_pose').value
        pose_topic = gp('pose_topic', '/PioneerP3DX/ground_truth').value
        self.pose_source = gp('pose_source', 'topic').value  # 'topic' or 'tf'
        self.map_frame = gp('map_frame', 'map').value
        self.base_frame = gp('base_frame', '').value
        cmd_topic = gp('cmd_vel_topic', '/PioneerP3DX/cmd_vel').value
        self.robot_radius = float(gp('robot_radius', 0.11).value)
        self.linear = float(gp('linear', 0.22).value)
        self.gain = float(gp('gain', 1.5).value)
        self.max_ang = float(gp('max_ang', 1.2).value)
        self.hop_tol = float(gp('hop_tol', 0.18).value)
        self.rotate_thresh = float(gp('rotate_thresh', 0.7).value)
        # Observed real hop completion: 1.6-2.9s. Kept tight (not generous) on
        # purpose: a wedged hop now correctly BLOCKS until this timeout instead
        # of reporting instant false success, so a stuck run pays this cost on
        # every futile step — a loose timeout compounds badly across hundreds
        # of stuck steps.
        self.goal_timeout = float(gp('goal_timeout', 5.0).value)

        self.occ = None
        self._live_map = bool(map_topic)
        if map_yaml and not self._live_map:
            self._load_map(map_yaml)

        self.x = self.y = self.theta = None
        self._path = []
        self._idx = 0

        cbg = ReentrantCallbackGroup()
        if self.pose_source == 'tf':
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        else:
            self.create_subscription(PoseWithCovarianceStamped, pose_topic,
                                     self._pose_cb, 10, callback_group=cbg)
        if self._live_map:
            self.create_subscription(OccupancyGrid, map_topic,
                                     self._map_cb, 10, callback_group=cbg)
        self._cmd = self.create_publisher(Twist, cmd_topic, 10)
        self._as = ActionServer(self, NavigateToPose, action_name,
                                self._execute, callback_group=cbg)
        self.create_timer(0.1, self._control_tick, callback_group=cbg)
        self.create_timer(0.1, self._pose_tick, callback_group=cbg)
        self.get_logger().info(
            f'DirectDrive up: action={action_name} pose_source={self.pose_source} '
            f'map={"live:" + map_topic if self._live_map else map_yaml}')

    # ---- map + A* ----
    def _load_map(self, yaml_path):
        cfg = yaml.safe_load(open(yaml_path))
        pgm = os.path.join(os.path.dirname(yaml_path), cfg['image'])
        t = open(pgm).read().split()
        self.W, self.H = int(t[1]), int(t[2])
        img = np.array(t[4:4 + self.W * self.H], dtype=int).reshape(self.H, self.W)
        self.res = float(cfg['resolution'])
        self.ox, self.oy = float(cfg['origin'][0]), float(cfg['origin'][1])
        occ = (img == 0)  # map_server negate=0: pixel 0 = occupied
        self.occ = binary_dilation(occ, iterations=max(1, int(round(self.robot_radius / self.res))))

    def _map_cb(self, msg: OccupancyGrid):
        """Live SLAM map: rebuild the grid every message. Unknown (-1) cells are
        treated as passable (same convention as efe_igdm's is_valid_traversable)
        so the robot can drive into unmapped territory and expand the map."""
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        occ = np.flipud(grid > 50)  # >50 = occupied; unknown(-1)/free(0) both passable
        self.W, self.H = w, h
        self.res = float(msg.info.resolution)
        self.ox = float(msg.info.origin.position.x)
        self.oy = float(msg.info.origin.position.y)
        self.occ = binary_dilation(occ, iterations=max(1, int(round(self.robot_radius / self.res))))

    def _w2c(self, x, y):
        return (self.H - 1 - int((y - self.oy) / self.res), int((x - self.ox) / self.res))

    def _c2w(self, r, c):
        return (self.ox + (c + 0.5) * self.res, self.oy + (self.H - 1 - r + 0.5) * self.res)

    def _nearest_free(self, r, c):
        if not self.occ[r, c]:
            return r, c
        fr = np.argwhere(~self.occ)
        d = np.hypot(fr[:, 0] - r, fr[:, 1] - c)
        return tuple(fr[d.argmin()])

    def _astar(self, s, g):
        sr, sc = self._w2c(*s)
        gr, gc = self._w2c(*g)
        sr, sc = self._nearest_free(sr, sc)
        gr, gc = self._nearest_free(gr, gc)
        H, W, occ = self.H, self.W, self.occ

        def h(r, c):
            return math.hypot(r - gr, c - gc)
        pq = [(h(sr, sc), 0.0, (sr, sc))]
        came = {(sr, sc): None}
        cost = {(sr, sc): 0.0}
        found = False
        while pq:
            _, g_, cur = heapq.heappop(pq)
            if cur == (gr, gc):
                found = True
                break
            r, c = cur
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not occ[nr, nc]:
                    step = 1.4142 if dr and dc else 1.0
                    ng = g_ + step
                    if ng < cost.get((nr, nc), 1e18):
                        cost[(nr, nc)] = ng
                        came[(nr, nc)] = cur
                        heapq.heappush(pq, (ng + h(nr, nc), ng, (nr, nc)))
        if not found:
            return None
        path = []
        cur = (gr, gc)
        while cur is not None:
            path.append(cur)
            cur = came[cur]
        path = path[::-1]
        return self._simplify(path)

    def _simplify(self, cells):
        """Drop collinear cells; keep line-of-sight-reachable turning points."""
        if len(cells) <= 2:
            return [self._c2w(*c) for c in cells]
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._los(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return [self._c2w(*c) for c in out]

    def _los(self, a, b):
        r0, c0 = a
        r1, c1 = b
        n = int(max(abs(r1 - r0), abs(c1 - c0)))
        if n == 0:
            return True
        for k in range(n + 1):
            r = int(round(r0 + (r1 - r0) * k / n))
            c = int(round(c0 + (c1 - c0) * k / n))
            if self.occ[r, c]:
                return False
        return True

    # ---- ROS callbacks ----
    def _pose_cb(self, msg):
        p = msg.pose.pose
        self.x, self.y = p.position.x, p.position.y
        self.theta = _yaw(p.orientation)

    def _pose_tick(self):
        """TF-based pose (realistic/SLAM mode): holds last known pose on lookup
        failure, same policy as gsl_bench's runner_node."""
        if self.pose_source != 'tf':
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            self.x = tf.transform.translation.x
            self.y = tf.transform.translation.y
            self.theta = _yaw(tf.transform.rotation)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    def _execute(self, gh):
        """Plan, then BLOCK until _control_tick actually drives the robot to
        the goal (or cancel/timeout) before reporting completion.

        Originally this returned right after planning — before any driving
        happened — which was correct for ADSM's fire-and-forget 1 Hz goal
        stream (see module docstring) but wrong for gsl_bench's Navigator
        client, which treats action "success" as "the robot arrived" (that's
        what a real Nav2 action does) and immediately plans the *next* step
        the instant the result future resolves. Because planning is
        near-instant, the harness was advancing to a brand-new target before
        the robot had moved at all, overwriting the in-progress path every
        cycle. Blocking here — running on a ReentrantCallbackGroup thread
        separate from the _control_tick timer that's still driving — makes
        "succeeded" mean what the client assumes it means.
        """
        tgt = gh.request.pose.pose.position
        path = None
        if self.x is not None and self.occ is not None:
            path = self._astar((self.x, self.y), (tgt.x, tgt.y))
        if not path:
            # Nothing to drive (unreachable, or pose/map not ready yet) — fail
            # fast rather than blocking on stale state from a prior goal.
            gh.abort()
            return NavigateToPose.Result()
        self._path = path
        self._idx = 0

        deadline = time.time() + self.goal_timeout
        while self._idx < len(self._path):
            if gh.is_cancel_requested:
                gh.canceled()
                return NavigateToPose.Result()
            if time.time() > deadline:
                gh.abort()
                return NavigateToPose.Result()
            time.sleep(0.05)

        gh.succeed()
        return NavigateToPose.Result()

    def _control_tick(self):
        if self.x is None or not self._path:
            return
        if self._idx >= len(self._path):
            self._cmd.publish(Twist())
            return
        tx, ty = self._path[self._idx]
        if math.hypot(tx - self.x, ty - self.y) < self.hop_tol:
            self._idx += 1
            return
        he = math.atan2(ty - self.y, tx - self.x) - self.theta
        he = math.atan2(math.sin(he), math.cos(he))
        tw = Twist()
        tw.linear.x = 0.0 if abs(he) > self.rotate_thresh else self.linear
        tw.angular.z = max(-self.max_ang, min(self.max_ang, self.gain * he))
        self._cmd.publish(tw)


def main():
    rclpy.init()
    node = DirectDriveNav()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
