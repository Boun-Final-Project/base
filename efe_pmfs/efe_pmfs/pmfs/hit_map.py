"""
PMFS hit-probability map.

Port of PMFSLib::EstimateHitProbabilities + PropagateProbabilities
(src/MAPIR_codes/GSL/gsl_server/.../PMFS/PMFSLib.cpp, lines 19-177).

Maintains per-cell:
  - log_odds        : Bayesian log-odds of p(H_i = hit)
  - hit_prob        : sigmoid(log_odds)
  - omega           : accumulated Gaussian-weighted sample count
  - confidence      : 1 - exp(-omega / σ_ω²)   (= α_i in paper)
  - distance_from_robot : filled during propagation of a single update
  - aux_weight      : propagated log-odds increment from the current update

Update rule (per gas sample, binary hit/miss):
  1. Place a wind-aligned elliptical Gaussian kernel at the robot cell.
     Angle = downwind_angle + π/2  (kernel's long axis is perpendicular to wind
     in the C++ code — see PMFSLib.cpp line 30 for the convention).
     Sigmas: (σ_minor, σ_major) stretched by wind speed.
  2. For every free cell line-of-sight from the robot within the local window,
     set aux_weight from the log-odds falloff.
  3. Propagate across the (8-neighbour) free-cell graph using a BFS that keeps
     the shortest path; aux_weight is evaluated along that propagation path.
  4. log_odds += aux_weight - log_odds_prior (Bayesian binary filter).
  5. omega += gaussian_1d(distance_from_robot, σ_spatial);
     confidence = 1 - exp(-omega / σ_meas_weight²).

With constant wind, the kernel shape is the same on every call; we precompute
a reusable kernel once at construction.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numba import njit


CELL_FREE = 0
CELL_UNKNOWN = -1
CELL_OCCUPIED = 1


@dataclass
class HitMapSettings:
    prior: float = 0.3
    kernel_sigma: float = 1.5              # paper default ~1.5 cells at scale 1
    kernel_stretch_constant: float = 1.0
    local_estimation_window: int = 20      # half-side in cells
    confidence_sigma_spatial: float = 1.0  # cells
    confidence_measurement_weight: float = 1.0


# ---------------------------------------------------------------------------
# Low-level math
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _eval_gaussian2d(dx, dy, sigma_minor, sigma_major, angle):
    """Anisotropic 2D Gaussian, rotated by `angle`. Evaluated without the
    1/(2π σ_x σ_y) normalization — we only use ratios against the peak."""
    c = np.cos(angle)
    s = np.sin(angle)
    # Rotate into the kernel's frame (minor axis = x, major axis = y)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return np.exp(-0.5 * ((xr / sigma_minor) ** 2 + (yr / sigma_major) ** 2))


@njit(cache=True, inline='always')
def _falloff_log_odds(dx, dy, sigma_minor, sigma_major, angle,
                      value_at1, prior):
    """log-odds contribution at offset (dx,dy) from the measurement cell."""
    sample = _eval_gaussian2d(dx, dy, sigma_minor, sigma_major, angle)
    # Peak value is 1 at offset (0,0) with this evaluator.
    t = sample
    prob = prior + (value_at1 - prior) * t
    # Clamp to (0.001, 0.999)
    if prob < 0.001:
        prob = 0.001
    elif prob > 0.999:
        prob = 0.999
    return np.log(prob / (1.0 - prob))


@njit(cache=True)
def _propagate_hit_update(occupancy, robot_gx, robot_gy,
                          sigma_minor, sigma_major, angle, value_at1,
                          prior, local_window,
                          distance_from_robot, aux_weight,
                          prop_dir_x, prop_dir_y, visited):
    """
    Fill `aux_weight` and `distance_from_robot` across the free-cell
    connected component from the robot, within the free-space graph.

    Implements PMFSLib::PropagateProbabilities (BFS-like with shortest-path
    update). 8-connectivity. Only free cells participate.

    Output arrays are in-out parameters; assumed pre-zeroed by caller.
    `visited` is a uint8 array used as a boolean flag (0 = not seen, 1 = in
    frontier, 2 = done).
    """
    h, w = occupancy.shape

    # Seed local window: cells with line-of-sight (approximated by free-cell
    # neighborhood — we let the propagation handle obstacle-aware distance).
    oC = max(0, robot_gx - local_window)
    fC = min(w - 1, robot_gx + local_window)
    oR = max(0, robot_gy - local_window)
    fR = min(h - 1, robot_gy + local_window)

    # Simple frontier queue implemented as a fixed-size array (we don't know
    # the size upfront; worst case w*h). Use two flat arrays alternately.
    queue_gx = np.empty(w * h, dtype=np.int32)
    queue_gy = np.empty(w * h, dtype=np.int32)
    q_head = 0
    q_tail = 0

    # The robot cell is the root.
    if 0 <= robot_gx < w and 0 <= robot_gy < h:
        if occupancy[robot_gy, robot_gx] == CELL_FREE or occupancy[robot_gy, robot_gx] == CELL_UNKNOWN:
            distance_from_robot[robot_gy, robot_gx] = 0.0
            aux_weight[robot_gy, robot_gx] = _falloff_log_odds(
                0.0, 0.0, sigma_minor, sigma_major, angle, value_at1, prior
            )
            prop_dir_x[robot_gy, robot_gx] = 0.0
            prop_dir_y[robot_gy, robot_gx] = 0.0
            visited[robot_gy, robot_gx] = 2

    # Seed: every free cell inside the local window, with straight-line
    # offset from the robot (PMFSLib does this via a visibility check; we
    # approximate by including free cells and letting propagation refine).
    for row in range(oR, fR + 1):
        for col in range(oC, fC + 1):
            if occupancy[row, col] != CELL_FREE:
                continue
            if col == robot_gx and row == robot_gy:
                continue
            dx = float(col - robot_gx)
            dy = float(row - robot_gy)
            dist = np.sqrt(dx * dx + dy * dy)
            distance_from_robot[row, col] = dist
            # Normalized propagation direction (for evaluating falloff along
            # the propagation ray during BFS neighbour updates).
            if dist > 0:
                prop_dir_x[row, col] = dx / dist
                prop_dir_y[row, col] = dy / dist
            # Initial aux_weight from the Gaussian at (dx, dy).
            aux_weight[row, col] = _falloff_log_odds(
                dx, dy, sigma_minor, sigma_major, angle, value_at1, prior
            )
            queue_gx[q_tail] = col
            queue_gy[q_tail] = row
            q_tail += 1
            visited[row, col] = 1

    # BFS: pop from queue, expand 8-neighbours, update if shorter path found.
    while q_head < q_tail:
        gx = queue_gx[q_head]
        gy = queue_gy[q_head]
        q_head += 1
        visited[gy, gx] = 2

        pdx = prop_dir_x[gy, gx]
        pdy = prop_dir_y[gy, gx]
        pdist = distance_from_robot[gy, gx]

        for dy_ in range(-1, 2):
            for dx_ in range(-1, 2):
                if dx_ == 0 and dy_ == 0:
                    continue
                ny = gy + dy_
                nx = gx + dx_
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if occupancy[ny, nx] != CELL_FREE:
                    continue
                # candidate distance: existing + step length
                step = 1.0 if (dx_ == 0 or dy_ == 0) else 1.4142135623730951
                new_dist = pdist + step
                if visited[ny, nx] == 0:
                    # first time reached
                    distance_from_robot[ny, nx] = new_dist
                    prop_dir_x[ny, nx] = pdx
                    prop_dir_y[ny, nx] = pdy
                    aux_weight[ny, nx] = _falloff_log_odds(
                        pdx * new_dist, pdy * new_dist,
                        sigma_minor, sigma_major, angle, value_at1, prior,
                    )
                    queue_gx[q_tail] = nx
                    queue_gy[q_tail] = ny
                    q_tail += 1
                    visited[ny, nx] = 1
                elif visited[ny, nx] == 1 and new_dist < distance_from_robot[ny, nx] - 1e-6:
                    # found a shorter path
                    distance_from_robot[ny, nx] = new_dist
                    prop_dir_x[ny, nx] = pdx
                    prop_dir_y[ny, nx] = pdy
                    aux_weight[ny, nx] = _falloff_log_odds(
                        pdx * new_dist, pdy * new_dist,
                        sigma_minor, sigma_major, angle, value_at1, prior,
                    )


@njit(cache=True)
def _apply_bayesian_update(occupancy, aux_weight, distance_from_robot,
                           visited, log_odds, omega,
                           log_odds_prior, sigma_spatial,
                           confidence_meas_weight):
    """Fold the propagated aux_weight into log_odds, then update omega/confidence."""
    h, w = occupancy.shape
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    cw_sq = confidence_meas_weight * confidence_meas_weight
    for gy in range(h):
        for gx in range(w):
            if occupancy[gy, gx] != CELL_FREE:
                continue
            if visited[gy, gx] == 0:
                continue
            log_odds[gy, gx] += aux_weight[gy, gx] - log_odds_prior
            d = distance_from_robot[gy, gx]
            omega[gy, gx] += np.exp(-d * d * inv_two_sigma_sq)
    # Confidence recomputed below (vectorized, outside numba for clarity)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HitMap:
    def __init__(self, grid_shape, settings: HitMapSettings,
                 wind_speed: float, wind_downwind_angle: float):
        """
        Parameters
        ----------
        grid_shape : (height, width)
        wind_speed, wind_downwind_angle :
            Used to compute the (fixed, for constant wind) kernel parameters.
            `wind_downwind_angle` is the downwind direction in world frame
            (radians). The kernel rotation follows PMFSLib (angle + π/2).
        """
        h, w = grid_shape
        self.height = h
        self.width = w
        self.settings = settings

        self.log_odds = np.zeros((h, w), dtype=np.float64)
        self.omega = np.zeros((h, w), dtype=np.float64)
        self.confidence = np.zeros((h, w), dtype=np.float64)

        # Persistent scratch buffers (one update at a time).
        self._distance_from_robot = np.zeros((h, w), dtype=np.float32)
        self._aux_weight = np.zeros((h, w), dtype=np.float64)
        self._prop_dir_x = np.zeros((h, w), dtype=np.float32)
        self._prop_dir_y = np.zeros((h, w), dtype=np.float32)
        self._visited = np.zeros((h, w), dtype=np.uint8)

        # Kernel params — fixed for constant wind.
        self.update_wind(wind_speed, wind_downwind_angle)
        self._log_odds_prior = np.log(settings.prior / (1.0 - settings.prior))

        self.num_updates = 0

    def update_wind(self, wind_speed, wind_downwind_angle):
        s = self.settings
        self.sigma_minor = s.kernel_sigma / (1.0 + s.kernel_stretch_constant * wind_speed / s.kernel_sigma)
        self.sigma_major = s.kernel_sigma + s.kernel_stretch_constant * wind_speed
        # PMFSLib: kernel_rotation = -normalize_angle(downwind + π/2)
        ang = wind_downwind_angle + np.pi / 2.0
        # normalize to [-π, π]
        ang = (ang + np.pi) % (2.0 * np.pi) - np.pi
        self.kernel_angle = -ang

    def update(self, occupancy, robot_cell, is_hit):
        """
        Fold one gas measurement into the hit probability map.

        Parameters
        ----------
        occupancy : np.int8[H, W]   -1/0/1 grid from SLAM
        robot_cell : (gx, gy)
        is_hit : bool               gas_ppm > threshold
        """
        gx, gy = int(robot_cell[0]), int(robot_cell[1])
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return

        # Reset per-update scratch.
        self._distance_from_robot.fill(0.0)
        self._aux_weight.fill(0.0)
        self._prop_dir_x.fill(0.0)
        self._prop_dir_y.fill(0.0)
        self._visited.fill(0)

        value_at1 = 0.6 if is_hit else 0.1

        _propagate_hit_update(
            occupancy, gx, gy,
            self.sigma_minor, self.sigma_major, self.kernel_angle, value_at1,
            self.settings.prior, self.settings.local_estimation_window,
            self._distance_from_robot, self._aux_weight,
            self._prop_dir_x, self._prop_dir_y, self._visited,
        )
        _apply_bayesian_update(
            occupancy, self._aux_weight, self._distance_from_robot,
            self._visited, self.log_odds, self.omega,
            self._log_odds_prior, self.settings.confidence_sigma_spatial,
            self.settings.confidence_measurement_weight,
        )
        # Vectorized confidence update.
        cw_sq = self.settings.confidence_measurement_weight ** 2
        np.subtract(1.0, np.exp(-self.omega / cw_sq), out=self.confidence)

        self.num_updates += 1

    @property
    def hit_prob(self):
        """p(H_i = hit) = sigmoid(log_odds). Cells never observed → sigmoid(0) = 0.5."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def prior_mask(self, threshold=1e-3):
        """Boolean mask of cells that have been effectively observed (confidence > threshold)."""
        return self.confidence > threshold
