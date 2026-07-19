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

import heapq
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
    is "stuck" if EITHER fires.
      * Efficiency-streak (FAST, catches early/tight circling): net/path over
        the last `win` steps < `ratio` for `streak` consecutive steps.
      * Map-growth stall (ROBUST, catches sustained wandering): the LiDAR
        occupancy map gained fewer than `grow_min` new cells over the last
        `grow_win` steps. This is a single CUMULATIVE window total, NOT a
        consecutive streak, so a brief excursion into already-mapped space does
        not reset it (the old streak version did, and missed sustained
        re-covering of known rooms). Calibrated from logs: exploration adds
        ~9000 cells / 120 steps, a stuck wander adds ~50-130 — a ~70x gap.
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
                 grow_win=120, grow_min=250, frontier_min_cells=12,
                 target_mode='largest',
                 use_live_grid=False, live_map_provider=None):
        self._use_live_grid = bool(use_live_grid)
        self._live_map_provider = live_map_provider

        if self._use_live_grid:
            if live_map_provider is None:
                raise ValueError("use_live_grid=True requires live_map_provider")
            live = live_map_provider()
            self.slam_map = (live if live is not None
                             else create_empty_occupancy_map(reference_map))
            self.mapper = None
        else:
            self.slam_map = create_empty_occupancy_map(reference_map)
            self.mapper = LidarMapper(self.slam_map)

        self.gp = GlobalPlanner(self.slam_map, robot_radius=robot_radius,
                                frontier_min_size=3, debug=False)
        self.log = logger

        # efficiency-streak (tight circling)
        self.win = int(win)
        self.ratio = float(ratio)
        self.streak_thr = int(streak)
        # long-horizon map-growth stall (loitering / wandering without discovery)
        self.grow_win = int(grow_win)   # window (steps) for cumulative new-cell count
        self.grow_min = int(grow_min)   # < this many new cells over grow_win => stalled
        self.frontier_min_cells = int(frontier_min_cells)  # gate: ignore tiny frontier slivers

        self.cooldown = int(cooldown)
        self.min_dist = float(min_dist)
        self.target_mode = str(target_mode)
        self.dump_dir = os.environ.get('OSL_ESCAPE_DUMP', '').strip()  # debug: dump SLAM grid per escape
        self.force_every = int(os.environ.get('OSL_ESCAPE_FORCE_EVERY', '0'))  # debug: force an escape every N steps
        self.max_escapes = int(os.environ.get('OSL_ESCAPE_MAX', '5'))         # cap escapes/run (anti-thrash)
        self.sat_thresh = float(os.environ.get('OSL_ESCAPE_SAT', '0.65'))     # stop escaping once map this fraction explored

        # --- ADSM Eq.3 source-term estimator (exploitation) -----------------
        # When stuck AND gas has been seen, drive to the cell that maximizes the
        # accumulated indoor-Gaussian source probability (upwind of the plume),
        # instead of a blind frontier. Falls back to frontier when no gas seen.
        self.use_source_est = os.environ.get('OSL_ESCAPE_SRC_EST', '1') == '1'
        self.src_Q = float(os.environ.get('OSL_SRC_Q', '4'))
        self.src_D = float(os.environ.get('OSL_SRC_D', '1'))
        self.src_l = float(os.environ.get('OSL_SRC_L', '4'))
        self.src_tau = float(os.environ.get('OSL_SRC_TAU', '250'))
        self.src_gas_eps = float(os.environ.get('OSL_SRC_GAS_EPS', '0.2'))
        self.src_min_obs = int(os.environ.get('OSL_SRC_MIN_OBS', '5'))
        # our anemometer wind_dir points DOWNWIND; Eq.3 wants the upwind vector,
        # so flip by pi (validated by checking the estimate points at the source).
        self.src_wind_flip = os.environ.get('OSL_ESCAPE_WIND_FLIP', '1') == '1'
        self._gas_obs = deque(maxlen=120)   # (rx, ry, v, phi) where gas was detected
        self.src_estimate = None            # last computed source estimate (for logging)

        self._pos = deque(maxlen=self.win + 1)
        self.streak = 0                       # efficiency streak counter
        self._cov_hist = deque(maxlen=self.grow_win + 1)  # SLAM mapped-cell count history
        self._grow_stuck = False              # long-horizon map-growth stall flag
        self.mapped_now = 0                   # current SLAM mapped-cell count
        self.slam_growth = 0                  # new SLAM cells over the last grow_win steps
        self.stuck_reason = ''                # which signal fired (for logging)
        self._last_escape_step = -10 ** 9
        self.n_escapes = 0
        # escape-path-following state (drive the planned path in short hops)
        self.escaping = False
        self.escape_path = []
        self.escape_idx = 0
        self._hop_attempts = 0      # tries at the current waypoint
        self._escape_fails = 0      # total wedged-hop attempts this escape

    # ------------------------------------------------------------------ SLAM
    def _refresh_live_map(self):
        """In live-grid mode, pull the latest map from the harness and rebind it
        on self.gp so frontier detection / A* / coverage stats all see the
        current slam_toolbox grid.  No-op in private-map mode."""
        if not self._use_live_grid or self._live_map_provider is None:
            return
        latest = self._live_map_provider()
        if latest is None:
            return
        self.slam_map = latest
        self.gp.occupancy_grid = latest

    def update_scan(self, scan_msg, x, y, theta):
        """Fold one LaserScan into the online occupancy map (sensed data only)."""
        if x is None or y is None or theta is None:
            return
        if self._use_live_grid:
            return   # live slam_toolbox map is maintained externally; nothing to do
        try:
            self.mapper.update_from_scan(scan_msg, float(x), float(y), float(theta))
        except Exception as exc:  # never let SLAM crash the control loop
            if self.log is not None:
                self.log.warn(f'[escape] SLAM update failed: {exc}')

    # ---------------------------------------------------------- stuck detect
    def record_step(self, x, y, og=0, v=0.0, phi=0.0):
        """Record one policy-step position and update both stuck signals.

        og/v/phi: binary gas, airflow speed, airflow direction (rad) at this step
        — buffered (when og=1) for the Eq.3 source estimate.

        Returns True if the robot is stuck (efficiency-streak OR coverage-
        stagnation has crossed its threshold). Sets ``self.stuck_reason``.
        """
        self._refresh_live_map()
        x = float(x)
        y = float(y)
        if og:
            self._gas_obs.append((x, y, float(v), float(phi)))

        # --- (1) efficiency-streak: tight circling -------------------------
        self._pos.append((x, y))
        if len(self._pos) > self.win:
            seg = list(self._pos)
            path = sum(math.hypot(seg[k + 1][0] - seg[k][0], seg[k + 1][1] - seg[k][1])
                       for k in range(len(seg) - 1))
            net = math.hypot(seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1])
            eff = (net / path) if path > 1e-6 else 1.0
            self.streak = self.streak + 1 if eff < self.ratio else 0

        # --- (2) map-growth stall: SLAM map stopped GROWING over a long horizon
        # Cumulative NEW cells the LiDAR added over the last grow_win steps (grid
        # != UNKNOWN), as ONE window total — not a consecutive streak, so a brief
        # excursion into already-mapped space does not reset it (the old streak
        # version did, and missed sustained wandering). Re-covering known space
        # adds ~0 cells; exploring open area adds thousands. Stalled when
        # < grow_min new cells over a full grow_win-step window.
        self.mapped_now = int((self.slam_map.grid != -1).sum())
        self._cov_hist.append(self.mapped_now)
        self.slam_growth = self._cov_hist[-1] - self._cov_hist[0]
        self._grow_stuck = (len(self._cov_hist) > self.grow_win
                            and self.slam_growth < self.grow_min)

        eff_stuck = self.streak >= self.streak_thr
        cov_stuck = self._grow_stuck
        if eff_stuck and cov_stuck:
            self.stuck_reason = 'circling+no-new-area'
        elif eff_stuck:
            self.stuck_reason = 'circling'
        elif cov_stuck:
            self.stuck_reason = 'no-new-area'
        else:
            self.stuck_reason = ''
        return eff_stuck or cov_stuck

    # ------------------------------------------- ADSM Eq.3 source estimator
    def _eq3_logP(self, cx, cy, rx, ry, v, phi):
        """log of ADSM Eq.3: prob the source is at (cx,cy) given the robot at
        (rx,ry) measured gas with airflow (v, phi). Upwind-biased."""
        d = math.hypot(rx - cx, ry - cy)
        if d < 0.05:
            d = 0.05
        if self.src_wind_flip:
            phi = phi + math.pi              # our wind_dir is downwind -> upwind
        wx = v * math.cos(phi)
        wy = v * math.sin(phi)
        D = self.src_D
        lam = (math.sqrt(D * self.src_tau / (1.0 + (v * v * self.src_tau) / (4.0 * D)))
               if v > 1e-6 else math.sqrt(D * self.src_tau))
        dot = (cx - rx) * wx + (cy - ry) * wy   # >0 when source is upwind of robot
        return math.log(self.src_Q * self.src_l / (4.0 * math.pi * D * d)) \
            + dot / (2.0 * D) - d / lam

    def source_estimate_target(self, x, y):
        """ADSM Eq.3 exploitation, realized as a one-shot UPWIND CAST.

        The literal Eq.3 argmax is degenerate for one-shot targeting (the 1/d and
        -d/lambda terms peak it on the gas detections, not upwind), which is why
        ADSM uses it incrementally. Its *effect* is "the source is upwind of the
        gas," so we cast from the gas-observation centroid along the mean upwind
        direction and return the farthest reachable FREE cell toward the source.
        None if too few gas observations. (`_eq3_logP` kept for reference.)
        """
        self._refresh_live_map()
        if not self.use_source_est or len(self._gas_obs) < self.src_min_obs:
            return None
        obs = list(self._gas_obs)
        gcx = sum(o[0] for o in obs) / len(obs)        # gas-observation centroid
        gcy = sum(o[1] for o in obs) / len(obs)
        sgn = -1.0 if self.src_wind_flip else 1.0      # our wind_dir is downwind -> flip to upwind
        ux = sum(sgn * math.cos(o[3]) for o in obs) / len(obs)   # mean upwind direction
        uy = sum(sgn * math.sin(o[3]) for o in obs) / len(obs)
        n = math.hypot(ux, uy)
        if n < 1e-6:
            return None
        ux, uy = ux / n, uy / n
        g = self.slam_map.grid
        H, W = g.shape
        res = self.slam_map.resolution
        ox, oy = self.slam_map.origin_x, self.slam_map.origin_y
        best = None
        for i in range(1, 21):                          # cast up to 10 m upwind
            cx = gcx + i * 0.5 * ux
            cy = gcy + i * 0.5 * uy
            gx = int((cx - ox) / res)
            gy = int((cy - oy) / res)
            if not (0 <= gx < W and 0 <= gy < H):
                break
            if g[gy, gx] == 1:                          # wall -> stop the cast
                break
            if g[gy, gx] == 0:                          # free -> drivable target so far
                best = (cx, cy)
            # unknown (-1): keep going (source region beyond is unmapped)
        self.src_estimate = best
        return best

    # ------------------------------------------------------- frontier escape
    def start_escape_if_stuck(self, x, y, step):
        """If stuck (and past cooldown), pick a REACHABLE frontier and plan a
        free-space path to it. Enters "escaping" mode; the path is then followed
        in short line-of-sight hops via next_waypoint() (so the DWB controller
        never has to track a long curve through a doorway and wedge).

        Returns (target_x, target_y, frontier_size, dist_m, n_waypoints) or None.
        """
        self._refresh_live_map()
        if self.force_every > 0:
            # debug stress mode: force an escape every N steps regardless of stuck
            if (step - self._last_escape_step) < self.force_every:
                return None
            self.stuck_reason = 'forced'
        else:
            stuck = (self.streak >= self.streak_thr or self._grow_stuck)
            if not stuck or (step - self._last_escape_step) < self.cooldown:
                return None
            # anti-thrash cap only. The "explored enough" gate now lives in part
            # (B) as a frontier-existence test (no worthwhile reachable frontier
            # => converged) — a denominator-free measure of remaining reachable
            # space, unlike mapped_fraction over the whole (mostly unreachable) grid.
            if self.n_escapes >= self.max_escapes:
                return None
        self._last_escape_step = step   # space out attempts even if none works

        # (A) EXPLOITATION — ADSM Eq.3: if we've smelled gas, drive to the upwind
        # source estimate (not a blind frontier). This is the piece that targets
        # the wind-offset source champ otherwise can't reach.
        est = self.source_estimate_target(x, y)
        if est is not None:
            path = self._plan_path_astar(x, y, est[0], est[1])
            if path:
                self.escape_path = path
                self.escape_idx = 0
                self.escaping = True
                self._hop_attempts = 0
                self._escape_fails = 0
                self.streak = 0
                self._cov_hist.clear()
                self.n_escapes += 1
                self.stuck_reason = (self.stuck_reason or 'stuck') + '|src-est'
                self._dump(x, y, est)
                return (est[0], est[1], -1, round(math.hypot(est[0] - x, est[1] - y), 1), len(path))

        # (B) EXPLORATION — frontier. GATE: only escape if a worthwhile (big
        # enough) AND reachable frontier still exists. The frontier-cell count is
        # itself the "is there reachable unexplored space left?" measure; when no
        # cluster clears frontier_min_cells we've effectively converged and must
        # NOT yank the robot (let the policy keep gas-searching).
        self.gp.detect_frontiers()
        clusters = self.gp.cluster_frontiers()
        cands = []
        for c in clusters:
            cx, cy = self.slam_map.grid_to_world(*c.centroid_grid)
            d = math.hypot(cx - x, cy - y)
            cands.append((c.size, d, cx, cy))
        if self.log is not None:
            top = sorted(cands, key=lambda c: -c[0])[:5]
            self.log.info(
                '[escape] frontiers(size@dist): '
                + (', '.join(f'{int(s)}@{dd:.1f}m' for s, dd, _, _ in top) or 'none')
                + f'  (gate >= {self.frontier_min_cells} cells)')
        worthwhile = [c for c in cands if c[0] >= self.frontier_min_cells]
        if not worthwhile:
            return None   # no reachable unexplored region worth it -> converged

        # Order survivors by target_mode, take the FIRST with a navigable A* path
        # (reachability check; also rejects frontiers only visible through a gap).
        far = [c for c in worthwhile if c[1] >= self.min_dist] or worthwhile
        if self.target_mode == 'nearest':
            order = sorted(far, key=lambda c: c[1])
        else:
            order = sorted(far, key=lambda c: -(c[0] - 0.5 * c[1]))

        for size, dist, tx, ty in order[:8]:
            path = self._plan_path_astar(x, y, tx, ty)
            if path:
                self.escape_path = path
                self.escape_idx = 0
                self.escaping = True
                self._hop_attempts = 0
                self._escape_fails = 0
                self.streak = 0
                self._cov_hist.clear()
                self.n_escapes += 1
                self._dump(x, y, (tx, ty))
                return (tx, ty, int(size), round(dist, 1), len(path))
        return None

    def next_waypoint(self, x, y, reach_tol=0.45, max_attempts=3, max_total_fails=4):
        """Return the next (wx,wy) hop to drive to, or None when done.

        Robust to wedged hops: a hop the robot can't reach within `max_attempts`
        tries is SKIPPED (move to the next waypoint); after `max_total_fails`
        wedged attempts the whole escape is ABORTED (return None → resume the
        policy). Without this, one DWB wedge on a 0.5 m hop would retry forever.
        """
        while self.escape_idx < len(self.escape_path):
            wx, wy = self.escape_path[self.escape_idx]
            if math.hypot(wx - x, wy - y) <= reach_tol:
                self.escape_idx += 1          # reached this hop
                self._hop_attempts = 0
                continue
            # not yet reached -> still driving this hop (one attempt per control
            # step). Only a hop that stays unreached for max_attempts steps is a
            # real wedge: skip it and count ONE failure THERE. (The old code did
            # self._escape_fails += 1 on EVERY in-progress step, so a normally
            # advancing escape hit max_total_fails after ~4 steps and aborted —
            # which is why multi-hop paths never completed: the robot quit ~4
            # hops in even when nothing was wedged.)
            self._hop_attempts += 1
            if self._hop_attempts > max_attempts:
                self.escape_idx += 1           # this hop is wedged; skip it
                self._hop_attempts = 0
                self._escape_fails += 1
                if self._escape_fails > max_total_fails:
                    break                      # too many wedged hops -> give up
                continue
            return (wx, wy)
        self.escaping = False
        self._hop_attempts = 0
        self._escape_fails = 0
        return None

    def _plan_path_astar(self, sx, sy, gx, gy):
        """A* over SLAM free cells (value 0) from world (sx,sy) to (gx,gy).

        Obstacle-INFLATED so the path keeps clearance from walls. Without this,
        A* cuts corners through doorways with ~0.07 m clearance — well inside the
        robot radius — and Nav2/DWB wedges trying to reach those waypoints (the
        escape then fails with repeated DRIVE TIMEOUTs). We dilate the occupied
        cells by the robot radius (~3 cells @ 0.1 m) and plan on the eroded free
        space; if that leaves no route (genuinely narrow passage) we relax the
        inflation 3->2->1->0 so a doorway is never made unplannable (infl=0
        reproduces the old raw behaviour as a last resort). Diagonal moves may
        not cut through a wall corner. Returns ~0.5 m world waypoints or None.
        """
        self._refresh_live_map()
        og = self.slam_map
        grid = og.grid
        H, W = grid.shape
        res = og.resolution
        ox, oy = og.origin_x, og.origin_y

        occ = (grid == 1)
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        for infl in (3, 2, 1, 0):               # robot radius ~0.25 m ~= 3 cells
            block = occ.copy()
            for _ in range(infl):                # diamond dilation by `infl` cells
                d = block.copy()
                d[1:, :] |= block[:-1, :]; d[:-1, :] |= block[1:, :]
                d[:, 1:] |= block[:, :-1]; d[:, :-1] |= block[:, 1:]
                block = d
            freemask = (grid == 0) & (~block)

            def free(cx, cy):
                return 0 <= cx < W and 0 <= cy < H and freemask[cy, cx]

            start = self._nearest_free((int((sx - ox) / res), int((sy - oy) / res)), free)
            goal = self._nearest_free((int((gx - ox) / res), int((gy - oy) / res)), free)
            if start is None or goal is None:
                continue

            def hcost(a):
                return math.hypot(a[0] - goal[0], a[1] - goal[1])

            openq = [(hcost(start), 0.0, start)]
            gsc = {start: 0.0}
            came = {}
            reached = None
            while openq:
                _, gc, cur = heapq.heappop(openq)
                if cur == goal or hcost(cur) <= 1.5:
                    reached = cur
                    break
                if gc > gsc.get(cur, 1e18):
                    continue
                for dx, dy, cost in nbrs:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if not free(nx, ny):
                        continue
                    if dx and dy and not (free(cur[0] + dx, cur[1]) and
                                          free(cur[0], cur[1] + dy)):
                        continue                 # no diagonal corner-cut through a wall
                    ng = gc + cost
                    if ng < gsc.get((nx, ny), 1e18):
                        gsc[(nx, ny)] = ng
                        came[(nx, ny)] = cur
                        heapq.heappush(openq, (ng + hcost((nx, ny)), ng, (nx, ny)))
            if reached is None:
                continue

            cells = [reached]
            while cells[-1] in came:
                cells.append(came[cells[-1]])
            cells.reverse()
            stepc = max(1, int(round(0.5 / res)))   # ~0.5 m between waypoints
            wps = [og.grid_to_world(*cells[i]) for i in range(0, len(cells), stepc)]
            last = og.grid_to_world(*cells[-1])
            if not wps or wps[-1] != last:
                wps.append(last)
            return wps
        return None

    @staticmethod
    def _nearest_free(cell, free, r=5):
        cx, cy = cell
        if free(cx, cy):
            return (cx, cy)
        for rad in range(1, r + 1):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if free(cx + dx, cy + dy):
                        return (cx + dx, cy + dy)
        return None

    def _dump(self, x, y, target):
        """Debug snapshot of the SLAM grid + chosen target + planned path."""
        self._refresh_live_map()
        if not self.dump_dir:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            fc = (np.array(self.gp.frontier_cells, dtype=np.int32)
                  if self.gp.frontier_cells else np.zeros((0, 2), np.int32))
            wps = (np.array(self.escape_path, dtype=np.float32)
                   if self.escape_path else np.zeros((0, 2), np.float32))
            np.savez(os.path.join(self.dump_dir, f'escape_{self.n_escapes:02d}.npz'),
                     grid=self.slam_map.grid.astype(np.int8),
                     origin_x=self.slam_map.origin_x, origin_y=self.slam_map.origin_y,
                     res=self.slam_map.resolution,
                     robot=np.array([x, y]), target=np.array(target),
                     path=wps, frontier_cells=fc)
        except Exception as exc:
            if self.log is not None:
                self.log.warn(f'[escape] dump failed: {exc}')

    # --------------------------------------------------------------- helpers
    def dump_map(self, step, x, y):
        """Debug: snapshot the current SLAM grid + frontiers + robot pose."""
        self._refresh_live_map()
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
        self._refresh_live_map()
        try:
            g = self.slam_map.grid
            return float((g != -1).sum()) / float(g.size)
        except Exception:
            return 0.0
