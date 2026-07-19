"""The oracle runner: one privileged, honest physical drive straight to the
true source, per scenario. Not a search — it knows the source location (which
GSLAgent deliberately never exposes to real agents) and drives a single A*
path to it over the ground-truth occupancy grid, no exploration, no
uncertainty. What comes back (travel_distance_m, sim_time_s) is meant to be
multiplied by a budget factor and fed back in as a per-scenario termination
cap for real agent runs (see episode_runner's --oracle-budgets).

The path is planned ONCE up front (A* over the dilated ground-truth grid,
simplified to turning points, then resampled to ~--step-size spacing) and
executed as a chain of deterministic cmd_vel hops — a proportional
rotate-then-go controller, ported directly from gaden_rl_node.py's
OSL_DET_DRIVE (`_start_cmdvel_move`/`_cmd_vel_tick`), the same mechanism
already proven in production to drive short (~0.5m) waypoints without
wedging. Two things that do NOT work in this deployment and are deliberately
avoided: (1) sending each hop (or one long goal) as a Nav2 NavigateToPose
action — Nav2's local controller (DWB) + SimpleProgressChecker aborts every
~10s on zero net movement at a doorway, and its recovery behaviors then burn
~150s per hop before giving up, wedging the ENTIRE wall-timeout budget on one
chokepoint (confirmed on 24/29 scenarios in the first oracle sweep,
2026-07-18, which used exactly this Nav2-hop mechanism). Driving cmd_vel
directly has no such recovery/abort loop to get stuck in. (2) aiming each hop
straight at the source (Euclidean, re-decided per hop, no route memory) walks
into the same doorway/wall repeatedly — A* avoids that by construction, it
never proposes a path through occupied cells. Default --launch benchmark_env:
benchmark_env_realistic doesn't start the occupancy service this needs (see
--launch help). Reads the true source position from /ground_truth directly —
this script bypasses the agent abstraction on purpose, real agents never see it.

    ros2 run gsl_bench oracle --suite nav7 --runs 1
"""
import argparse
import atexit
import heapq
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.ndimage import distance_transform_edt

from efe_igdm.mapping.occupancy_grid import (
    OccupancyGridMap, load_3d_occupancy_grid_from_service,
)
from efe_igdm.planning.navigator import Navigator

from gsl_bench.eval import scenario_paths, suites
from gsl_bench.eval.episode_runner import (
    BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    _LIVE_PROCS, _cleanup, _fmt, bake_if_needed, kill_ros, kill_tree,
    patch_lidar_config, patch_nav2_params, patch_no_rviz, patch_playback_loop,
    reexec_under_setsid, start_xvfb, wait_for_pose,
)


class OracleNode(Node):
    """One episode: plan an A* path start->source over the ground-truth grid,
    teleport to start, then drive the path as a chain of deterministic cmd_vel
    hops (rotate-then-go, no Nav2) until within success_radius, recording
    travel distance / sim time / wall time."""

    def __init__(self, ns: str, start_xy, source_xy, success_radius: float,
                 wall_timeout: float, result_file: Path,
                 robot_radius: float = 0.25, step_size: float = 0.5,
                 det_linear: float = 0.3, det_gain: float = 2.0,
                 det_arrive_tol: float = 0.15, det_hop_timeout: float = 12.0,
                 drive_mode: str = 'cmdvel', nav2_goal_tolerance: float = 0.15,
                 plan_margin: float = 0.30, nav2_stride: float = 1.5,
                 occupancy_service: str = '/gaden_environment/occupancyMap3D',
                 occupancy_z_level: int = 5, occupancy_timeout: float = 300.0):
        super().__init__('gsl_bench_oracle')
        self._ns = ns
        self._start_x, self._start_y = start_xy
        self._source_x, self._source_y = source_xy
        self._success_radius = success_radius
        self._wall_timeout = wall_timeout
        self._result_file = result_file
        self._robot_radius = robot_radius
        self._det_linear = det_linear
        self._det_gain = det_gain
        self._det_arrive_tol = det_arrive_tol
        self._det_hop_timeout = det_hop_timeout
        self._drive_mode = drive_mode
        self._nav2_goal_tolerance = nav2_goal_tolerance
        # A* clearance = physical radius + a safety margin, so no planned waypoint
        # sits TANGENT to a wall. The EDT threshold admits cells exactly
        # robot_radius from the nearest obstacle; a waypoint there puts the robot
        # body flush against the wall, and Nav2's DWB (which weaves toward the
        # point rather than tracking a line the way cmd_vel's rotate-then-go does)
        # grazes it — BasicSim's hard-collision then zeroes the motion and the
        # robot FREEZES, wedged, while the per-hop cancel churns past it (observed
        # 2026-07-18 on the 4 tight scenarios: robot stuck at one pose for 14+
        # hops). The margin keeps the path off the walls so Nav2 has room to weave.
        self._plan_clearance = robot_radius + max(0.0, plan_margin)
        # Nav2 is built to navigate metres between goals, not 0.5m micro-hops:
        # fed hops finer than its own goal tolerance it never leaves goal-approach
        # mode. In nav2 mode plan a COARSER path (turning points resampled to
        # ~nav2_stride) so it actually path-navigates each leg; cmd_vel keeps the
        # fine step_size its proportional controller was tuned for.
        self._plan_stride = (max(step_size, nav2_stride)
                             if drive_mode == 'nav2' else step_size)

        self._travel_distance_m = 0.0
        self._last_xy: Optional[tuple] = None
        self._current_theta: Optional[float] = None
        self._placed = False
        self._teleport_wait_stamp_ns: Optional[int] = None
        self._n_hops = 0
        self._episode_start_ns: Optional[int] = None
        self._start_wall: Optional[float] = None
        self._done = False
        self._last_pose_wall: Optional[float] = None

        self._cmdvel_active = False
        self._waypoint_x: Optional[float] = None
        self._waypoint_y: Optional[float] = None
        self._waypoint_start_ns: int = 0
        self._nav2_hop_active = False
        self._nav2_goal_send_ns: Optional[int] = None
        self._nav2_canceling = False
        self._nav2_reject_streak = 0
        self._nav2_max_rejects = 8
        self._nav2_timeout_streak = 0
        self._nav2_max_timeouts = 3
        self._nav2_pending_retry = False
        self._nav2_retry_ns = 0
        self._nav2_last_target: Optional[tuple] = None
        # Action-result futures and pose/timer callbacks may run on different
        # MultiThreadedExecutor workers.  In particular, a result callback can
        # clear _nav2_hop_active while a pose callback simultaneously observes
        # it false; both then enter _maybe_send_next_hop and consume two path
        # entries.  Apart from sending concurrent goals, the second dispatch can
        # exhaust the path and report nav2_failed while the first (final) goal is
        # still driving.  Serialize every path-index/state transition here.
        self._dispatch_lock = threading.RLock()

        self.get_logger().info(
            f'Waiting up to {occupancy_timeout:.0f}s for {occupancy_service} ...')
        grid_2d, _outlet_mask, params = load_3d_occupancy_grid_from_service(
            self, z_level=occupancy_z_level, service_name=occupancy_service,
            timeout_sec=occupancy_timeout)
        self._occ_map = OccupancyGridMap(grid_2d, params)
        if self._occ_map.origin_x == 0.0 and self._occ_map.origin_y == 0.0:
            self._occ_map.origin_x = -0.2
            self._occ_map.origin_y = -0.2

        self._path = self._plan_path(
            (self._start_x, self._start_y), (self._source_x, self._source_y),
            self._plan_stride)
        # Inflation radius is a graded Nav2 cost, not a lethal boundary. Use the
        # matched 0.55m clearance whenever it leaves the map connected, but do not
        # declare a genuinely narrow scenario impossible solely because that soft
        # cost radius seals its doorway. Relax in 0.10m steps, never below the
        # physical robot radius + 0.10m safety clearance.
        if self._path is None and self._drive_mode == 'nav2':
            requested_clearance = self._plan_clearance
            minimum_clearance = self._robot_radius + 0.10
            candidate = requested_clearance - 0.10
            while self._path is None and candidate >= minimum_clearance - 1e-9:
                self._plan_clearance = candidate
                self.get_logger().warn(
                    f'No path at {requested_clearance:.2f}m Nav2 clearance; '
                    f'trying narrow-passage clearance {candidate:.2f}m.')
                self._path = self._plan_path(
                    (self._start_x, self._start_y),
                    (self._source_x, self._source_y), self._plan_stride)
                candidate -= 0.10
        if self._path is not None and self._drive_mode == 'nav2':
            # A* starts at the containing cell centre, which can produce a first
            # waypoint only a few centimetres from the teleported pose. Nav2
            # consistently rejects that no-op goal; it is not route progress and
            # used to waste the entire rejection budget before the real drive.
            while (self._path and math.hypot(
                    self._path[0][0] - self._start_x,
                    self._path[0][1] - self._start_y) <= self._nav2_goal_tolerance):
                self._path.pop(0)
        self._path_idx = 0
        if self._path is None:
            self.get_logger().error(
                'No A* path from start to source over the ground-truth grid — aborting.')
            self._write_result(status='no_path')
            os._exit(2)
        self.get_logger().info(
            f'Planned path: {len(self._path)} waypoints @ ~{self._plan_stride}m '
            f'spacing (clearance {self._plan_clearance:.2f}m).')

        self.create_subscription(
            PoseWithCovarianceStamped, f'/{ns}/ground_truth', self._pose_callback, 10)
        self._place_pub = self.create_publisher(
            PoseWithCovarianceStamped, f'/{ns}/initialpose', 10)

        # Keep one publisher/timer setup for both modes. The timer is a no-op in
        # Nav2 mode; there is deliberately no cmd_vel fallback because oracle
        # budgets must remain achievable by Nav2-driven benchmark agents.
        self._cmd_vel_pub = self.create_publisher(Twist, f'/{ns}/cmd_vel', 10)
        self.create_timer(0.05, self._cmd_vel_tick)
        if self._drive_mode == 'nav2':
            self._navigator = Navigator(self, on_complete_callback=self._on_nav2_hop_complete)
            self.create_timer(1.0, self._nav2_hop_watchdog)
            self.get_logger().info(
                f'Drive mode: NAV2 (active per-hop cancel after '
                f'{self._det_hop_timeout:.0f}s, tolerance={nav2_goal_tolerance:.2f}m; '
                f'retry-on-rejection + skip, final leg targets source @ tol='
                f'{self._success_radius:.2f}m)')
        else:
            self.get_logger().info('Drive mode: cmd_vel (deterministic rotate-then-go)')

        self._watchdog = self.create_timer(
            2.0, self._wall_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_theta = math.atan2(siny_cosp, cosy_cosp)
        self._last_pose_wall = time.monotonic()

        if self._last_xy is not None:
            d = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if d < 1.0:   # skip the initial teleport jump, same guard obs_cache uses
                self._travel_distance_m += d
        self._last_xy = (x, y)

        if self._done:
            return

        msg_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        if not self._placed:
            self._placed = True
            self.get_logger().info(
                f'Placing robot at start ({self._start_x:.2f}, {self._start_y:.2f})')
            self._place_robot(self._start_x, self._start_y)
            # BasicSim's /clock (and every header.stamp derived from it) is elapsed
            # sim time since BasicSim launched, NOT wall-clock epoch time — comparing
            # it against self.get_clock().now() (wall clock, since this node never
            # sets use_sim_time) would make the gate below permanently true. Must
            # compare stamp-vs-stamp, both on BasicSim's own clock, same as
            # runner_node's last_scan_stamp_ns gate.
            self._teleport_wait_stamp_ns = msg_stamp_ns
            return

        # Wait for a pose whose OWN stamp postdates the placement — comparing against
        # "now" at callback time is always true (time only moves forward) and would
        # let the very next pose through immediately, before the teleport is visible.
        if (self._teleport_wait_stamp_ns is not None
                and msg_stamp_ns <= self._teleport_wait_stamp_ns):
            return

        if self._episode_start_ns is None:
            self._episode_start_ns = self.get_clock().now().nanoseconds
            self._start_wall = time.monotonic()

        # _nav2_pending_retry must gate this too: after a goal rejection the retry
        # is deliberately delayed (costmap warmup) and re-issued by the watchdog;
        # without this guard the ~10-30Hz pose stream re-fires the hop immediately,
        # defeating the backoff and torching all retries in a few ms.
        if (not self._cmdvel_active and not self._nav2_hop_active
                and not self._nav2_pending_retry):
            self._maybe_send_next_hop()

    def _plan_path(self, start_xy, goal_xy, step_size: float):
        """A* over the ground-truth grid (dilated by robot_radius), simplified to
        turning points via line-of-sight pruning, then resampled to ~step_size
        spacing. Returns a list of (x, y) waypoints (start excluded, goal
        included), or None if the grid has no free path start->goal. Ported from
        directdrive_nav.py's proven _astar/_simplify/_los onto OccupancyGridMap's
        own (row-major, non-flipped) world_to_grid/grid_to_world convention.
        """
        res = self._occ_map.resolution
        occ = self._occ_map.grid > 0
        # binary_dilation's default structuring element is a 4-connected cross, so
        # N iterations grows a DIAMOND, not a disk: full margin along the axes but
        # only ~margin/sqrt(2) along the diagonals — exactly where A*'s diagonal
        # moves cut doorways/corners, silently shortchanging the intended
        # robot_radius clearance there and causing the path to clip walls the
        # planner believed were clear (confirmed 2026-07-18: raising robot_radius
        # 0.25->0.35 barely helped, since the diagonal shortfall scales the same
        # way regardless of the input radius). distance_transform_edt gives the
        # true Euclidean distance to the nearest occupied cell, so the margin is
        # circular and angle-independent.
        occ = distance_transform_edt(~occ) * res <= self._plan_clearance
        H, W = occ.shape

        def w2c(pt):
            gx, gy = self._occ_map.world_to_grid(*pt)
            return (gy, gx)

        def c2w(rc):
            r, c = rc
            return (self._occ_map.origin_x + (c + 0.5) * res,
                    self._occ_map.origin_y + (r + 0.5) * res)

        def nearest_free(r, c):
            if 0 <= r < H and 0 <= c < W and not occ[r, c]:
                return r, c
            free = np.argwhere(~occ)
            if free.size == 0:
                return r, c
            d = np.hypot(free[:, 0] - r, free[:, 1] - c)
            return tuple(free[d.argmin()])

        def los(a, b):
            r0, c0 = a
            r1, c1 = b
            n = int(max(abs(r1 - r0), abs(c1 - c0)))
            if n == 0:
                return True
            for k in range(n + 1):
                r = int(round(r0 + (r1 - r0) * k / n))
                c = int(round(c0 + (c1 - c0) * k / n))
                if occ[r, c]:
                    return False
            return True

        sr, sc = nearest_free(*w2c(start_xy))
        gr, gc = nearest_free(*w2c(goal_xy))

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
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
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

        cells = []
        cur = (gr, gc)
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells = cells[::-1]

        if len(cells) <= 2:
            simplified = cells
        else:
            simplified = [cells[0]]
            i = 0
            while i < len(cells) - 1:
                j = len(cells) - 1
                while j > i + 1 and not los(cells[i], cells[j]):
                    j -= 1
                simplified.append(cells[j])
                i = j

        turning_points = [c2w(c) for c in simplified]
        waypoints = [start_xy]
        for nxt in turning_points:
            px, py = waypoints[-1]
            seg_d = math.hypot(nxt[0] - px, nxt[1] - py)
            n_steps = max(1, int(math.ceil(seg_d / step_size)))
            for k in range(1, n_steps + 1):
                t = k / n_steps
                waypoints.append((px + (nxt[0] - px) * t, py + (nxt[1] - py) * t))
        return waypoints[1:]   # drop the duplicated start point

    def _maybe_send_next_hop(self):
        with self._dispatch_lock:
            if self._done:
                return
            # Callers normally test these flags themselves, but that test and
            # this method were not atomic under MultiThreadedExecutor.  Make the
            # dispatch method authoritative so a racing callback is harmless.
            if (self._drive_mode == 'nav2'
                    and (self._nav2_hop_active or self._nav2_pending_retry
                         or self._nav2_canceling)):
                return
            if self._drive_mode != 'nav2' and self._cmdvel_active:
                return
            self._dispatch_next_hop_locked()

    def _dispatch_next_hop_locked(self):
        """Dispatch one path entry with ``_dispatch_lock`` held."""
        rx, ry = (self._last_xy if self._last_xy is not None
                  else (self._start_x, self._start_y))
        final_d = math.hypot(rx - self._source_x, ry - self._source_y)
        if final_d < self._success_radius:
            self._end_episode('success', final_d)
            return
        if self._path_idx >= len(self._path):
            self._end_episode('nav2_failed', final_d)
            return

        tx, ty = self._path[self._path_idx]
        self._path_idx += 1
        self._n_hops += 1
        # Final leg (nav2): aim at the SOURCE itself with tolerance = success_radius,
        # not at the precise last A* waypoint (tol 0.15m). Nav2 then drives as close
        # as its costmap allows and reports the goal reached once inside
        # success_radius — exactly the criterion the real agents are scored by, so
        # a success here is one a Nav2-driven agent could reproduce. A tight 0.15m
        # waypoint next to the source instead just gets rejected as too close to the
        # wall the source sits on, stranding the robot metres out.
        is_final = self._drive_mode == 'nav2' and self._path_idx >= len(self._path)
        goal_x, goal_y = (self._source_x, self._source_y) if is_final else (tx, ty)
        goal_tol = self._success_radius if is_final else self._nav2_goal_tolerance
        self.get_logger().info(
            f'[hop {self._n_hops:3d}/{len(self._path)}] ({rx:.2f},{ry:.2f}) -> '
            f'({goal_x:.2f},{goal_y:.2f}) d2src={final_d:.2f}m'
            f'{" [SOURCE, tol=%.2f]" % goal_tol if is_final else ""}')
        if self._drive_mode == 'nav2':
            self._nav2_hop_active = True
            self._nav2_canceling = False
            self._nav2_last_target = (goal_x, goal_y)
            self._nav2_goal_send_ns = self.get_clock().now().nanoseconds
            # Distance-proportional cancel: a ~1.5m nav2 leg needs more than the
            # flat per-hop timeout to plan+drive; scale it so we still abort a
            # genuinely-wedged hop but don't cut short an honest long leg.
            hop_d = math.hypot(goal_x - rx, goal_y - ry)
            self._nav2_hop_timeout = max(self._det_hop_timeout, 6.0 + 5.0 * hop_d)
            self._navigator.send_goal(goal_x, goal_y, tolerance=goal_tol)
        else:
            self._start_cmdvel_move(tx, ty)

    def _on_nav2_hop_complete(self):
        """Navigator's on_complete fires on success, abort, cancel, AND rejection.
        A near-instant completion (< reject_thresh, and not one we canceled) is a
        Nav2 *goal rejection* — almost always because the local/global costmap
        isn't warmed up yet when the first goals go out, so it rejects every one.
        Blindly calling _maybe_send_next_hop then torches the whole planned path in
        a few hundred ms (observed 2026-07-18: 15 hops fired in 0.15s, robot never
        moved). Instead: DON'T advance — step the index back to re-issue the SAME
        waypoint after a short warmup delay, and only skip it after max_rejects
        tries. A watchdog cancellation retries the same required waypoint rather
        than pretending a timed-out leg was completed."""
        with self._dispatch_lock:
            elapsed = (99.0 if self._nav2_goal_send_ns is None
                       else (self.get_clock().now().nanoseconds - self._nav2_goal_send_ns) * 1e-9)
            was_cancel = self._nav2_canceling
            was_reject = (not was_cancel) and elapsed < 1.0
            self._nav2_hop_active = False
            self._nav2_canceling = False
            self._nav2_goal_send_ns = None
            if was_reject:
                self._nav2_reject_streak += 1
                if self._nav2_reject_streak <= self._nav2_max_rejects:
                    self._path_idx = max(0, self._path_idx - 1)   # retry same waypoint
                    self._nav2_retry_ns = (self.get_clock().now().nanoseconds
                                           + int(1.2e9))           # after costmap warmup
                    self._nav2_pending_retry = True
                    self.get_logger().warn(
                        f'[hop {self._n_hops:3d}] NAV2 goal rejected (costmap warming?) '
                        f'— retry {self._nav2_reject_streak}/{self._nav2_max_rejects} in 1.2s')
                    return
            # Persistent rejection (not warmup) means Nav2 genuinely refuses this
            # waypoint — it's inside its costmap inflation. We do NOT fall back to
            # a cmd_vel drive that would thread it anyway: the oracle budget must be
            # achievable by the Nav2-driven agents (gpulaika/adsm) it caps, so the
            # oracle must reach the source the SAME way they do. Cheating with
            # cmd_vel would publish a budget no Nav2 agent could follow and imply a
            # source that is, for a Nav2 robot, unreachable. Skip the waypoint; if
            # the source vicinity as a whole is Nav2-unreachable the run ends
            # nav2_failed — an honest signal about the scenario, not a bug to mask.
                self.get_logger().warn(
                    f'[hop {self._n_hops:3d}] Nav2 rejected {self._nav2_max_rejects}x '
                    f'— skipping waypoint (no cmd_vel cheat).')
                self._nav2_reject_streak = 0
            elif was_cancel:
                # A watchdog cancellation says the robot did NOT reach this
                # waypoint.  Advancing used to strand it on the wrong side of a
                # doorway while later goals marched through the remaining path.
                # Retry the same ground-truth route constraint; if Nav2 cannot
                # execute it repeatedly, fail honestly instead of fabricating
                # progress by skipping it.
                self._nav2_timeout_streak += 1
                if self._nav2_timeout_streak <= self._nav2_max_timeouts:
                    self._path_idx = max(0, self._path_idx - 1)
                    self._nav2_retry_ns = (self.get_clock().now().nanoseconds
                                           + int(1.2e9))
                    self._nav2_pending_retry = True
                    self.get_logger().warn(
                        f'[hop {self._n_hops:3d}] timed-out waypoint not reached '
                        f'— retry {self._nav2_timeout_streak}/'
                        f'{self._nav2_max_timeouts} in 1.2s')
                    return
                rx, ry = (self._last_xy if self._last_xy is not None
                          else (self._start_x, self._start_y))
                final_d = math.hypot(rx - self._source_x, ry - self._source_y)
                self.get_logger().error(
                    f'Nav2 failed the same required waypoint '
                    f'{self._nav2_max_timeouts + 1} times; route is not executable.')
                self._end_episode('nav2_failed', final_d)
                return
            else:
                self._nav2_reject_streak = 0
                self._nav2_timeout_streak = 0
            self._maybe_send_next_hop()

    def _nav2_hop_watchdog(self):
        """Active per-hop cancel: don't wait for Nav2's own SimpleProgressChecker
        + recovery-behavior tree to give up (~150s) — cancel and move to the next
        waypoint after det_hop_timeout, same principle as gaden_rl_node.py's
        per-step drive-timeout cancel in its (non-det-drive) Nav2 path."""
        if self._nav2_pending_retry and not self._nav2_hop_active:
            if self.get_clock().now().nanoseconds >= self._nav2_retry_ns:
                self._nav2_pending_retry = False
                self._maybe_send_next_hop()
            return
        if not self._nav2_hop_active or self._nav2_canceling or self._nav2_goal_send_ns is None:
            return
        elapsed = (self.get_clock().now().nanoseconds - self._nav2_goal_send_ns) * 1e-9
        if elapsed > getattr(self, '_nav2_hop_timeout', self._det_hop_timeout):
            self._nav2_canceling = True
            self.get_logger().warn(
                f'[hop {self._n_hops:3d}] NAV2 HOP TIMEOUT after {elapsed:.1f}s — canceling.')
            self._navigator.cancel_current_goal()

    def _start_cmdvel_move(self, x: float, y: float):
        """Begin a deterministic cmd_vel drive to (x, y). Ported from
        gaden_rl_node.py's _start_cmdvel_move — _cmd_vel_tick drives it and
        clears _cmdvel_active on arrival/timeout."""
        self._waypoint_x = float(x)
        self._waypoint_y = float(y)
        self._waypoint_start_ns = self.get_clock().now().nanoseconds
        self._cmdvel_active = True

    def _cmd_vel_tick(self):
        """Proportional rotate-then-go controller: turn to face the target, then
        drive straight to it. Bypasses Nav2/DWB so it never wedges on near-wall
        hops and lands ON the commanded point. Stops on arrival or per-hop
        timeout. Ported from gaden_rl_node.py's OSL_DET_DRIVE _cmd_vel_tick."""
        if not self._cmdvel_active or self._waypoint_x is None:
            return
        if self._last_xy is None or self._current_theta is None:
            return
        rx, ry = self._last_xy
        dx = self._waypoint_x - rx
        dy = self._waypoint_y - ry
        dist = math.hypot(dx, dy)
        timed_out = ((self.get_clock().now().nanoseconds - self._waypoint_start_ns)
                     > self._det_hop_timeout * 1e9)
        if dist < self._det_arrive_tol or timed_out:
            self._cmd_vel_pub.publish(Twist())   # stop
            self._cmdvel_active = False
            self._maybe_send_next_hop()
            return
        heading_error = math.atan2(dy, dx) - self._current_theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        tw = Twist()
        tw.linear.x = 0.0 if abs(heading_error) > 0.8 else self._det_linear
        tw.angular.z = self._det_gain * heading_error
        self._cmd_vel_pub.publish(tw)

    def _place_robot(self, x: float, y: float):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance = [0.0] * 36
        self._place_pub.publish(msg)

    def _wall_watchdog(self):
        now_w = time.monotonic()
        if self._done:
            return
        if self._last_pose_wall is not None and now_w - self._last_pose_wall > 120.0:
            self.get_logger().error('ENV DEAD — no pose for 120s. Aborting.')
            self._write_result(status='env_dead')
            os._exit(3)
        if (self._start_wall is not None
                and now_w - self._start_wall > self._wall_timeout):
            rx, ry = (self._last_xy if self._last_xy is not None
                      else (self._start_x, self._start_y))
            final_d = math.hypot(rx - self._source_x, ry - self._source_y)
            self.get_logger().warn(
                f'WALL TIMEOUT after {now_w - self._start_wall:.0f}s — giving up.')
            self._end_episode('nav2_timeout', final_d)

    def _sim_time(self) -> float:
        if self._episode_start_ns is None:
            return 0.0
        return (self.get_clock().now().nanoseconds - self._episode_start_ns) * 1e-9

    def _write_result(self, status: str, final_distance_m: Optional[float] = None):
        result = {
            'schema': 1,
            'status': status,
            'source': [self._source_x, self._source_y],
            'start': [self._start_x, self._start_y],
            'travel_distance_m': round(self._travel_distance_m, 3),
            'sim_time_s': round(self._sim_time(), 2),
            'wall_time_s': (round(time.monotonic() - self._start_wall, 1)
                             if self._start_wall is not None else None),
            'final_distance_m': None if final_distance_m is None else round(final_distance_m, 3),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self._result_file)), exist_ok=True)
        with open(self._result_file, 'w') as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _end_episode(self, status: str, final_distance_m: float):
        self._done = True
        if self._drive_mode == 'nav2':
            self._navigator.cancel_current_goal()
        else:
            self._cmd_vel_pub.publish(Twist())
        self.get_logger().info(
            f'{status.upper()} — {self._travel_distance_m:.1f} m travelled, '
            f'{self._sim_time():.1f} s sim time, final_d={final_distance_m:.2f} m')
        self._write_result(status, final_distance_m)
        os._exit(0)


def run_one_oracle(args, scenario: str, sim: str, run_dir: Path,
                    source, start) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / 'oracle_result.json'
    t0 = time.time()

    env_log = open(run_dir / 'env.log', 'wb')
    launch_pkg = args.launch
    launch_file = ('realistic_launch.py' if launch_pkg == 'benchmark_env_realistic'
                   else 'main_simbot_launch.py')
    launch_cmd = ['ros2', 'launch', launch_pkg, launch_file,
                  f'scenario:={scenario}', 'configuration:=config1', f'simulation:={sim}',
                  f'namespace:={args.namespace}']
    if launch_pkg == 'benchmark_env_realistic':
        launch_cmd.append('motion:=nav2')
    env_proc = subprocess.Popen(launch_cmd, stdout=env_log, stderr=subprocess.STDOUT)
    _LIVE_PROCS.append(env_proc)

    print('  waiting for ground_truth...')
    if not wait_for_pose(args.namespace):
        print(f'  {RED}env never published ground_truth — aborting run{RESET}')
        write_stub_oracle_result(result_file, source, start, 'env_dead')
        kill_tree(env_proc)
        _LIVE_PROCS.remove(env_proc)
        kill_ros()
        time.sleep(3)
        return json.loads(result_file.read_text())

    print('  pose online, waiting for Nav2 lifecycle...')
    nav_node = f'/{args.namespace}/bt_navigator'
    nav_deadline = time.monotonic() + args.nav2_startup_timeout
    nav_active = False
    while time.monotonic() < nav_deadline and env_proc.poll() is None:
        try:
            state = subprocess.run(
                ['ros2', 'lifecycle', 'get', nav_node], capture_output=True,
                text=True, timeout=5.0, check=False)
            if state.returncode == 0 and 'active' in state.stdout.lower():
                nav_active = True
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1.0)
    if not nav_active:
        print(f'  {YELLOW}Nav2 did not reach lifecycle state active within '
              f'{args.nav2_startup_timeout:.0f}s — relaunch required{RESET}')
        write_stub_oracle_result(result_file, source, start, 'nav2_unavailable')
        kill_tree(env_proc)
        _LIVE_PROCS.remove(env_proc)
        kill_ros()
        time.sleep(3)
        return json.loads(result_file.read_text())

    print('  Nav2 active, sending oracle to run in-process.')
    # Let costmaps receive their first update after lifecycle activation. Merely
    # discovering the action server is insufficient: an inactive bt_navigator
    # advertises the action but rejects every goal.
    time.sleep(2)

    cmd = [sys.executable, '-u', '-m', 'gsl_bench.eval.oracle', '--_run-node',
           f'--namespace={args.namespace}',
           f'--start-x={_fmt(float(start[0]))}', f'--start-y={_fmt(float(start[1]))}',
           f'--source-x={_fmt(float(source[0]))}', f'--source-y={_fmt(float(source[1]))}',
           f'--success-radius={_fmt(float(args.success_radius))}',
           f'--wall-timeout={_fmt(float(args.wall_timeout))}',
           f'--robot-radius={_fmt(float(args.robot_radius))}',
           f'--step-size={_fmt(float(args.step_size))}',
           f'--det-linear={_fmt(float(args.det_linear))}',
           f'--det-gain={_fmt(float(args.det_gain))}',
           f'--det-arrive-tol={_fmt(float(args.det_arrive_tol))}',
           f'--det-hop-timeout={_fmt(float(args.det_hop_timeout))}',
           f'--drive-mode={args.drive_mode}',
           f'--nav2-goal-tolerance={_fmt(float(args.nav2_goal_tolerance))}',
           f'--plan-margin={_fmt(float(args.plan_margin))}',
           f'--nav2-stride={_fmt(float(args.nav2_stride))}',
           f'--result-file={result_file}']
    node_log = open(run_dir / 'node.log', 'wb')
    node_proc = subprocess.Popen(cmd, stdout=node_log, stderr=subprocess.STDOUT)

    status = 'done'
    while node_proc.poll() is None:
        if time.time() - t0 > args.wall_timeout + 60.0:
            status = 'wall_timeout'
            print(f'  {YELLOW}WALL TIMEOUT{RESET} after {time.time() - t0:.0f}s — killing.')
            node_proc.terminate()
            time.sleep(2)
            node_proc.kill()
            break
        time.sleep(2)

    if not result_file.is_file():
        write_stub_oracle_result(
            result_file, source, start,
            'wall_timeout' if status == 'wall_timeout' else 'crashed')

    res = json.loads(result_file.read_text())
    res['scenario'] = scenario
    res['sim'] = sim
    result_file.write_text(json.dumps(res, indent=2))

    if res.get('status') == 'success':
        print(f'  {GREEN}ORACLE{RESET}  dist={res["travel_distance_m"]}m  '
              f'time={res["sim_time_s"]}s  ({time.time() - t0:.0f}s wall)')
    else:
        print(f'  {YELLOW}oracle did not confirm success{RESET}  status={res["status"]}')

    kill_tree(env_proc)
    _LIVE_PROCS.remove(env_proc)
    kill_ros()
    time.sleep(3)
    return res


def write_stub_oracle_result(path: Path, source, start, status: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'schema': 1, 'status': status, 'source': list(source), 'start': list(start),
            'travel_distance_m': None, 'sim_time_s': None, 'wall_time_s': None,
            'final_distance_m': None,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)


def _run_node_main(argv: List[str]):
    """Entry point for the subprocess spawned by run_one_oracle: one OracleNode,
    one episode, exit. Kept as a `python -m` invocation (not a console_script)
    since it's an internal implementation detail of `oracle`, not a public CLI."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--_run-node', action='store_true')
    ap.add_argument('--namespace', default='PioneerP3DX')
    ap.add_argument('--start-x', type=float, required=True)
    ap.add_argument('--start-y', type=float, required=True)
    ap.add_argument('--source-x', type=float, required=True)
    ap.add_argument('--source-y', type=float, required=True)
    ap.add_argument('--success-radius', type=float, required=True)
    ap.add_argument('--wall-timeout', type=float, required=True)
    ap.add_argument('--robot-radius', type=float, default=0.25)
    ap.add_argument('--step-size', type=float, default=0.5)
    ap.add_argument('--det-linear', type=float, default=0.3)
    ap.add_argument('--det-gain', type=float, default=2.0)
    ap.add_argument('--det-arrive-tol', type=float, default=0.15)
    ap.add_argument('--det-hop-timeout', type=float, default=12.0)
    ap.add_argument('--drive-mode', choices=['cmdvel', 'nav2'], default='cmdvel')
    ap.add_argument('--nav2-goal-tolerance', type=float, default=0.15)
    ap.add_argument('--plan-margin', type=float, default=0.30)
    ap.add_argument('--nav2-stride', type=float, default=1.5)
    ap.add_argument('--result-file', type=Path, required=True)
    a = ap.parse_args(argv)

    rclpy.init()
    node = OracleNode(
        ns=a.namespace, start_xy=(a.start_x, a.start_y),
        source_xy=(a.source_x, a.source_y), success_radius=a.success_radius,
        wall_timeout=a.wall_timeout, result_file=a.result_file,
        robot_radius=a.robot_radius, step_size=a.step_size,
        det_linear=a.det_linear, det_gain=a.det_gain,
        det_arrive_tol=a.det_arrive_tol, det_hop_timeout=a.det_hop_timeout,
        drive_mode=a.drive_mode, nav2_goal_tolerance=a.nav2_goal_tolerance,
        plan_margin=a.plan_margin, nav2_stride=a.nav2_stride)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()


def parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(prog='gsl_bench oracle')
    ap.add_argument('--suite', default=None, help='nav7 | all')
    ap.add_argument('--scenario', default=None,
                    help='scenario name, or a comma-separated list')
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--out', type=Path,
                    default=Path('/home/efe/ros2_ws/Results/oracle_budgets.json'))
    ap.add_argument('--wall-timeout', type=float, default=600.0)
    ap.add_argument('--success-radius', type=float, default=0.5)
    ap.add_argument('--robot-radius', type=float, default=0.25,
                    help='true Euclidean clearance (via distance_transform_edt) used '
                         'to inflate the ground-truth grid before A* planning')
    ap.add_argument('--step-size', type=float, default=0.5,
                    help='waypoint spacing along the planned path, same default as '
                         'episode_runner/runner_node step_size_hint')
    ap.add_argument('--det-linear', type=float, default=0.3,
                    help='cmd_vel drive speed (m/s), same default as OSL_DET_LINEAR')
    ap.add_argument('--det-gain', type=float, default=2.0,
                    help='proportional heading gain, same default as OSL_DET_GAIN')
    ap.add_argument('--det-arrive-tol', type=float, default=0.15,
                    help='per-hop arrival tolerance (m), same default as OSL_DET_ARRIVE_TOL')
    ap.add_argument('--det-hop-timeout', type=float, default=12.0,
                    help='per-hop wall timeout (s) before skipping to the next waypoint '
                         '(cmdvel mode) or canceling the goal (nav2 mode), same default '
                         'as OSL_DET_TIMEOUT')
    ap.add_argument('--drive-mode', choices=['cmdvel', 'nav2'], default='cmdvel',
                    help="cmdvel (default): blind proportional rotate-then-go, no "
                         "obstacle sensing, fails at genuinely tight passages that need "
                         "local avoidance. nav2: real Nav2/DWB local controller per hop "
                         "(can locally deform around tight corners cmdvel cannot thread) "
                         "with an ACTIVE per-hop cancel after det-hop-timeout instead of "
                         "waiting out Nav2's own ~150s SimpleProgressChecker+recovery.")
    ap.add_argument('--nav2-goal-tolerance', type=float, default=0.15,
                    help='Nav2 xy_goal_tolerance per hop, only used in --drive-mode=nav2')
    ap.add_argument('--plan-margin', type=float, default=0.30,
                    help='safety margin (m) added to robot_radius when inflating the '
                         "ground-truth grid for A*; with the default 0.25m robot "
                         "radius this matches Nav2's 0.55m inflation radius")
    ap.add_argument('--nav2-stride', type=float, default=1.5,
                    help='waypoint spacing (m) for the planned path in --drive-mode=nav2 '
                         '(coarser than --step-size so Nav2 path-navigates each leg '
                         'instead of churning 0.5m micro-hops); ignored in cmdvel mode')
    ap.add_argument('--nav2-startup-timeout', type=float, default=60.0,
                    help='seconds to wait for bt_navigator lifecycle state active')
    ap.add_argument('--startup-retries', type=int, default=2,
                    help='automatic environment relaunches after a zero-motion Nav2 '
                         'startup failure (in addition to the requested run)')
    ap.add_argument('--namespace', default='PioneerP3DX')
    ap.add_argument('--skip-bake', action='store_true')
    ap.add_argument('--domain-id', type=int, default=None)
    ap.add_argument('--launch', default='benchmark_env',
                    help='launch package for the environment: benchmark_env (default — '
                         'ground-truth occupancy service, needed for A* planning) or '
                         "benchmark_env_realistic (that launch doesn't start the occupancy "
                         'service this needs, no fallback yet)')
    args = ap.parse_args(argv)
    if not args.suite and not args.scenario:
        args.suite = 'nav7'
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--_run-node':
        _run_node_main(argv)
        return 0

    args = parse_args(argv)

    if args.domain_id is not None:
        os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    elif not os.environ.get('ROS_DOMAIN_ID'):
        print(f'{YELLOW}Warning: ROS_DOMAIN_ID unset.{RESET}')

    reexec_under_setsid()

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))

    scenario_list = ([s.strip() for s in args.scenario.split(',') if s.strip()]
                     if args.scenario else suites.resolve_suite(args.suite))
    share = scenario_paths.install_share_dir()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = args.out.parent / f'oracle_runs_{stamp}'
    run_root.mkdir(parents=True, exist_ok=True)

    print(f'\n{BOLD}{CYAN} gsl_bench oracle{RESET}')
    print(f'  scenarios : {len(scenario_list)} x {args.runs} runs')
    print(f'  launch    : {args.launch}')
    print(f'  manifest  : {BOLD}{args.out}{RESET}\n')

    display = start_xvfb()
    print(f'  {DIM}headless display :{display}{RESET}')

    kill_ros()
    if patch_no_rviz(share):
        print(f'  {DIM}patched install launch: no rviz, no xterm{RESET}')
    if patch_nav2_params(share):
        print(f'  {DIM}patched install nav2_params: tuned (no RotateToGoal){RESET}')
    subprocess.run(['ros2', 'daemon', 'start'], capture_output=True, check=False)
    time.sleep(2)

    manifest = {}
    for scenario in scenario_list:
        if not scenario_paths.scenario_dir(scenario).is_dir():
            print(f'{RED}Unknown scenario: {scenario} — skipping{RESET}')
            continue
        sim = scenario_paths.resolve_sim(scenario)
        print(f'\n{DIM}{"=" * 50}{RESET}\n{BOLD} {scenario}{RESET} ({sim})')

        if not bake_if_needed(scenario, sim, args.skip_bake):
            print(f'{RED}bake failed — skipping {scenario}{RESET}')
            continue
        patch_playback_loop(scenario, sim)
        patch_lidar_config(scenario)
        patch_nav2_params(share)

        source = scenario_paths.source_xy(scenario, sim)
        start = scenario_paths.start_xy(scenario)
        print(f'  source ({source[0]:.2f}, {source[1]:.2f})  '
              f'start ({start[0]:.2f}, {start[1]:.2f})')

        scen_dir = run_root / f'{scenario}_{sim}'
        best = None
        for i in range(1, args.runs + 1):
            print(f'\n{BOLD} {scenario} oracle run {i}/{args.runs}{RESET}')
            attempt = 0
            while True:
                suffix = '' if attempt == 0 else f'_retry_{attempt}'
                res = run_one_oracle(
                    args, scenario, sim, scen_dir / f'run_{i}{suffix}', source, start)
                startup_failure = (
                    res.get('status') == 'nav2_unavailable'
                    or (res.get('status') == 'nav2_failed'
                        and float(res.get('travel_distance_m') or 0.0) == 0.0))
                if not startup_failure or attempt >= args.startup_retries:
                    break
                attempt += 1
                print(f'  {YELLOW}Nav2 startup failure; automatically relaunching '
                      f'(retry {attempt}/{args.startup_retries}){RESET}')
            if res.get('status') == 'success' and (
                    best is None or res['travel_distance_m'] < best['travel_distance_m']):
                best = res
        if best is not None:
            manifest[f'{scenario}_{sim}'] = best
        else:
            print(f'{RED}no successful oracle run for {scenario}_{sim} — '
                  f'omitted from manifest{RESET}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f'\n{BOLD}{GREEN}Oracle manifest: {args.out}{RESET} '
          f'({len(manifest)}/{len(scenario_list)} scenarios){RESET}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
