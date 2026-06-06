"""SLAM-based circling detector + frontier-exploration escape for the RL node.

Why this exists
---------------
champ_far02 is a reactive gas-following policy. On wind-swept maps (notably
many_rooms) the plume peaks several metres *downwind* of the source, so the
policy converges on the plume and circles there forever (0/5). This module gives
the deployment a recovery behaviour that is consistent with the lidar-based
policy: it builds an occupancy map ONLINE from the same LaserScan the policy
uses (LidarMapper — sensed data only, no ground-truth map peeking), detects when
the robot is stuck circling, and on stuck drives it to the largest unexplored
frontier so the policy resumes from a fresh region.

Design decisions (both validated on the recorded many_rooms walk logs):
  * Circling detector = SUSTAINED low displacement-efficiency. Per step we take
    net_displacement / path_length over the last `win` positions; if that stays
    below `ratio` for `streak` consecutive steps the robot is circling. On the
    logs this separated stuck runs (streak 33-61) from clean successes (<=29);
    instantaneous loop/efficiency checks fired on successes too and were rejected.
  * NO upwind bias. Local wind at the actual circling locations did NOT point at
    the source (mean align ~0.03-0.12, negative for one run) — a single wind
    reading is too noisy. So the escape is PURE frontier exploration, which is
    wind-independent and steers the robot into the unexplored rooms over time.

It reuses the efe_igdm SLAM/planning stack (LidarMapper, GlobalPlanner frontier
detection, OccupancyGridMap), so the map and frontiers are exactly the ones the
model-based baselines (ADSM/EESA) build.
"""

import math
import os
from collections import deque

import numpy as np
from efe_igdm.mapping.occupancy_grid import create_empty_occupancy_map
from efe_igdm.mapping.lidar_mapper import LidarMapper
from efe_igdm.planning.global_planner import GlobalPlanner


class CirclingEscape:
    """Online SLAM + circling detection + frontier-exploration escape.

    Parameters
    ----------
    reference_map : OccupancyGridMap
        The GADEN occupancy map — used ONLY to copy grid dimensions/resolution
        for the empty SLAM grid. The SLAM grid itself starts all-unknown and is
        filled solely from LiDAR scans.
    robot_radius : float
        Collision radius for the frontier planner's validity checks.

    Stuck detection combines TWO complementary, observable signals — the robot
    is "stuck" if EITHER fires. Both thresholds are calibrated from the maps
    champ already solves (successful search never exceeds eff-streak 29 or
    coverage-stagnation 14), so a trigger means "behaving worse than any
    successful search did" — not tuned to any one map.
      * Efficiency-streak (catches tight circling): net/path over the last
        `win` steps < `ratio` for `streak` consecutive steps.
      * Coverage-stagnation (catches loitering/oscillation): the set of visited
        `cov_res`-m cells grew by < `cov_k` over the last `cov_win` steps, for
        `cov_streak` consecutive steps.
    cooldown : int
        Minimum steps between escape attempts (also spaces out failed attempts).
    min_dist : float
        Prefer frontier targets at least this far away (to actually leave the
        stuck region); relaxed if no far frontier exists.
    target_mode : "largest" | "nearest"
        How to choose among frontier candidates (all >= min_dist):
        - "largest": biggest unexplored region (size - 0.5*dist). Big jumps that
          can land directly on a far source region (helps many_rooms) but waste
          budget on huge maps (hurts ultimate).
        - "nearest": closest frontier >= min_dist. Incremental local exploration;
          less wasted travel on big maps, reaches far sources via several hops.
    """

    def __init__(self, reference_map, robot_radius=0.25, logger=None,
                 win=25, ratio=0.2, streak=35, cooldown=40, min_dist=3.0,
                 cov_res=0.5, cov_win=40, cov_k=3, cov_streak=25,
                 target_mode='largest'):
        self.slam_map = create_empty_occupancy_map(reference_map)
        self.mapper = LidarMapper(self.slam_map)
        self.gp = GlobalPlanner(self.slam_map, robot_radius=robot_radius,
                                frontier_min_size=3, debug=False)
        self.log = logger

        # efficiency-streak (tight circling)
        self.win = int(win)
        self.ratio = float(ratio)
        self.streak_thr = int(streak)
        # coverage-stagnation (loitering)
        self.cov_res = float(cov_res)
        self.cov_win = int(cov_win)
        self.cov_k = int(cov_k)
        self.cov_streak_thr = int(cov_streak)

        self.cooldown = int(cooldown)
        self.min_dist = float(min_dist)
        self.target_mode = str(target_mode)
        self.dump_dir = os.environ.get('OSL_ESCAPE_DUMP', '').strip()  # debug: dump SLAM grid per escape

        self._pos = deque(maxlen=self.win + 1)
        self.streak = 0                       # efficiency streak counter
        self._visited = set()
        self._cov_hist = deque(maxlen=self.cov_win + 1)  # cumulative visited count
        self.cov_streak = 0                   # coverage-stagnation streak counter
        self.stuck_reason = ''                # which signal fired (for logging)
        self._last_escape_step = -10 ** 9
        self.n_escapes = 0

    # ------------------------------------------------------------------ SLAM
    def update_scan(self, scan_msg, x, y, theta):
        """Fold one LaserScan into the online occupancy map (sensed data only)."""
        if x is None or y is None or theta is None:
            return
        try:
            self.mapper.update_from_scan(scan_msg, float(x), float(y), float(theta))
        except Exception as exc:  # never let SLAM crash the control loop
            if self.log is not None:
                self.log.warn(f'[escape] SLAM update failed: {exc}')

    # ---------------------------------------------------------- stuck detect
    def record_step(self, x, y):
        """Record one policy-step position and update both stuck signals.

        Returns True if the robot is stuck (efficiency-streak OR coverage-
        stagnation has crossed its threshold). Sets ``self.stuck_reason``.
        """
        x = float(x)
        y = float(y)

        # --- (1) efficiency-streak: tight circling -------------------------
        self._pos.append((x, y))
        if len(self._pos) > self.win:
            seg = list(self._pos)
            path = sum(math.hypot(seg[k + 1][0] - seg[k][0], seg[k + 1][1] - seg[k][1])
                       for k in range(len(seg) - 1))
            net = math.hypot(seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1])
            eff = (net / path) if path > 1e-6 else 1.0
            self.streak = self.streak + 1 if eff < self.ratio else 0

        # --- (2) coverage-stagnation: loitering / no new area --------------
        self._visited.add((round(x / self.cov_res), round(y / self.cov_res)))
        self._cov_hist.append(len(self._visited))
        if len(self._cov_hist) > self.cov_win:
            grew = self._cov_hist[-1] - self._cov_hist[0]   # new cells over window
            self.cov_streak = self.cov_streak + 1 if grew < self.cov_k else 0

        eff_stuck = self.streak >= self.streak_thr
        cov_stuck = self.cov_streak >= self.cov_streak_thr
        if eff_stuck and cov_stuck:
            self.stuck_reason = 'circling+loitering'
        elif eff_stuck:
            self.stuck_reason = 'circling'
        elif cov_stuck:
            self.stuck_reason = 'loitering'
        else:
            self.stuck_reason = ''
        return eff_stuck or cov_stuck

    # ------------------------------------------------------- frontier escape
    def maybe_escape_target(self, x, y, step):
        """If stuck (and past cooldown), pick the best unexplored frontier.

        Returns (target_x, target_y, frontier_size, dist_m) or None.
        """
        stuck = (self.streak >= self.streak_thr
                 or self.cov_streak >= self.cov_streak_thr)
        if not stuck:
            return None
        if (step - self._last_escape_step) < self.cooldown:
            return None
        # Space out attempts even if no frontier is found this step.
        self._last_escape_step = step

        self.gp.detect_frontiers()
        clusters = self.gp.cluster_frontiers()
        if not clusters:
            return None

        cands = []
        for c in clusters:
            cx, cy = self.slam_map.grid_to_world(*c.centroid_grid)
            d = math.hypot(cx - x, cy - y)
            cands.append((c.size, d, cx, cy))

        # Among frontiers far enough to leave the basin, choose by target_mode:
        #   nearest -> closest one (incremental exploration, less wasted travel)
        #   largest -> biggest unexplored region (size - 0.5*dist; big jumps)
        far = [c for c in cands if c[1] >= self.min_dist] or cands
        if self.target_mode == 'nearest':
            best = min(far, key=lambda c: c[1])
        else:
            best = max(far, key=lambda c: c[0] - 0.5 * c[1])

        self.streak = 0          # consumed; re-arm both signals after relocation
        self.cov_streak = 0
        self.n_escapes += 1

        if self.dump_dir:        # debug snapshot: the actual SLAM grid + chosen target
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
                fc = np.array(self.gp.frontier_cells, dtype=np.int32) if self.gp.frontier_cells else np.zeros((0, 2), np.int32)
                np.savez(os.path.join(self.dump_dir, f'escape_{self.n_escapes:02d}_step{step}.npz'),
                         grid=self.slam_map.grid.astype(np.int8),
                         origin_x=self.slam_map.origin_x, origin_y=self.slam_map.origin_y,
                         res=self.slam_map.resolution,
                         robot=np.array([x, y]), target=np.array([best[2], best[3]]),
                         frontier_cells=fc)
            except Exception as exc:
                if self.log is not None:
                    self.log.warn(f'[escape] dump failed: {exc}')

        return (best[2], best[3], int(best[0]), round(best[1], 1))

    # --------------------------------------------------------------- helpers
    def dump_map(self, step, x, y):
        """Debug: snapshot the current SLAM grid + frontiers + robot pose."""
        if not self.dump_dir:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            self.gp.detect_frontiers()
            fc = (np.array(self.gp.frontier_cells, dtype=np.int32)
                  if self.gp.frontier_cells else np.zeros((0, 2), np.int32))
            np.savez(os.path.join(self.dump_dir, f'step_{step:04d}.npz'),
                     grid=self.slam_map.grid.astype(np.int8),
                     origin_x=self.slam_map.origin_x, origin_y=self.slam_map.origin_y,
                     res=self.slam_map.resolution,
                     robot=np.array([float(x), float(y)]), frontier_cells=fc)
        except Exception as exc:
            if self.log is not None:
                self.log.warn(f'[escape] map dump failed: {exc}')

    def mapped_fraction(self):
        """Fraction of grid cells that are no longer unknown (for logging)."""
        try:
            g = self.slam_map.grid
            return float((g != -1).sum()) / float(g.size)
        except Exception:
            return 0.0
