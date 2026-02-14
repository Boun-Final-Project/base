"""
GMRF Wind Mapper - Gaussian Markov Random Field based Wind Estimation.

Estimates the wind field (u, v) on the SLAM grid by solving a sparse linear system.
The system is constructed from:
1.  Smoothness priors (neighboring cells should have similar wind).
2.  Mass conservation (divergence should be zero).
3.  Measurement constraints (wind at measured cells should match anemometer data).
4.  Boundary conditions (zero normal flow at walls/unknown).

Based on: Monroy et al., "Online Estimation of 2D Wind Maps for Olfactory Robots"

Matrix construction is Numba-accelerated for real-time use with growing SLAM maps.
"""

import numpy as np
from numba import njit
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional


# Maximum sparse entries per cell (measurement:2 + obstacle:2 + smooth:16 +
# divergence:16 + boundary:4 = 40). Use 44 for safety margin.
_MAX_ENTRIES_PER_CELL = 44


@njit(cache=True)
def _build_triplets(grid, is_fluid, measurement_mask, measured_u, measured_v,
                    lam_smooth, lam_div, lam_meas, lam_bnd, lam_obs):
    """
    Build sparse matrix triplets (rows, cols, data) and RHS vector b
    for the GMRF wind estimation system Qx = b.

    All arrays are pre-allocated; `count` tracks how many entries are used.

    Parameters
    ----------
    grid : int8 array (H, W)
    is_fluid : bool array (H, W)
    measurement_mask : bool array (H, W)
    measured_u, measured_v : float64 arrays (H, W)
    lam_smooth, lam_div, lam_meas, lam_bnd, lam_obs : float64
        Weights for each constraint type.

    Returns
    -------
    rows, cols, data : int32/float64 arrays (pre-allocated, use [:count])
    b : float64 array (2*N,)
    count : int - number of valid entries
    """
    height, width = grid.shape
    N = height * width
    max_entries = _MAX_ENTRIES_PER_CELL * N

    rows = np.empty(max_entries, dtype=np.int32)
    cols = np.empty(max_entries, dtype=np.int32)
    data = np.empty(max_entries, dtype=np.float64)
    b = np.zeros(2 * N, dtype=np.float64)

    count = 0

    for y in range(height):
        for x in range(width):
            u_idx = y * width + x        # index for u component
            v_idx = u_idx + N             # index for v component

            # --- 1. Measurement Constraint ---
            if measurement_mask[y, x]:
                rows[count] = u_idx; cols[count] = u_idx; data[count] = lam_meas; count += 1
                b[u_idx] += lam_meas * measured_u[y, x]

                rows[count] = v_idx; cols[count] = v_idx; data[count] = lam_meas; count += 1
                b[v_idx] += lam_meas * measured_v[y, x]

            # --- Non-fluid cells: force zero flow ---
            if not is_fluid[y, x]:
                rows[count] = u_idx; cols[count] = u_idx; data[count] = lam_obs; count += 1
                rows[count] = v_idx; cols[count] = v_idx; data[count] = lam_obs; count += 1
                continue

            # ===== From here: cell is fluid =====

            # --- 2. Smoothness (right neighbor) ---
            if x + 1 < width and is_fluid[y, x + 1]:
                u_r = y * width + (x + 1)
                v_r = u_r + N

                # u-component pair
                rows[count] = u_idx; cols[count] = u_idx;  data[count] =  lam_smooth; count += 1
                rows[count] = u_r;   cols[count] = u_r;    data[count] =  lam_smooth; count += 1
                rows[count] = u_idx; cols[count] = u_r;    data[count] = -lam_smooth; count += 1
                rows[count] = u_r;   cols[count] = u_idx;  data[count] = -lam_smooth; count += 1

                # v-component pair
                rows[count] = v_idx; cols[count] = v_idx;  data[count] =  lam_smooth; count += 1
                rows[count] = v_r;   cols[count] = v_r;    data[count] =  lam_smooth; count += 1
                rows[count] = v_idx; cols[count] = v_r;    data[count] = -lam_smooth; count += 1
                rows[count] = v_r;   cols[count] = v_idx;  data[count] = -lam_smooth; count += 1

            # --- Smoothness (bottom neighbor) ---
            if y + 1 < height and is_fluid[y + 1, x]:
                u_d = (y + 1) * width + x
                v_d = u_d + N

                rows[count] = u_idx; cols[count] = u_idx;  data[count] =  lam_smooth; count += 1
                rows[count] = u_d;   cols[count] = u_d;    data[count] =  lam_smooth; count += 1
                rows[count] = u_idx; cols[count] = u_d;    data[count] = -lam_smooth; count += 1
                rows[count] = u_d;   cols[count] = u_idx;  data[count] = -lam_smooth; count += 1

                rows[count] = v_idx; cols[count] = v_idx;  data[count] =  lam_smooth; count += 1
                rows[count] = v_d;   cols[count] = v_d;    data[count] =  lam_smooth; count += 1
                rows[count] = v_idx; cols[count] = v_d;    data[count] = -lam_smooth; count += 1
                rows[count] = v_d;   cols[count] = v_idx;  data[count] = -lam_smooth; count += 1

            # --- 3. Divergence-Free (mass conservation) ---
            # Central difference: (u_right - u_left + v_down - v_up) = 0
            if (x > 0 and x < width - 1 and y > 0 and y < height - 1
                    and is_fluid[y, x-1] and is_fluid[y, x+1]
                    and is_fluid[y-1, x] and is_fluid[y+1, x]):

                idx_ul = y * width + (x - 1)          # u_left
                idx_ur = y * width + (x + 1)          # u_right
                idx_vu = (y - 1) * width + x + N      # v_up
                idx_vd = (y + 1) * width + x + N      # v_down

                # Indices and coefficients for div = c^T x
                vi0 = idx_ur; vi1 = idx_ul; vi2 = idx_vd; vi3 = idx_vu
                c0 = 1.0; c1 = -1.0; c2 = 1.0; c3 = -1.0

                # Outer product: Q += lam_div * c c^T (4x4 = 16 entries)
                # Unrolled for Numba performance
                rows[count] = vi0; cols[count] = vi0; data[count] = lam_div * c0 * c0; count += 1
                rows[count] = vi0; cols[count] = vi1; data[count] = lam_div * c0 * c1; count += 1
                rows[count] = vi0; cols[count] = vi2; data[count] = lam_div * c0 * c2; count += 1
                rows[count] = vi0; cols[count] = vi3; data[count] = lam_div * c0 * c3; count += 1

                rows[count] = vi1; cols[count] = vi0; data[count] = lam_div * c1 * c0; count += 1
                rows[count] = vi1; cols[count] = vi1; data[count] = lam_div * c1 * c1; count += 1
                rows[count] = vi1; cols[count] = vi2; data[count] = lam_div * c1 * c2; count += 1
                rows[count] = vi1; cols[count] = vi3; data[count] = lam_div * c1 * c3; count += 1

                rows[count] = vi2; cols[count] = vi0; data[count] = lam_div * c2 * c0; count += 1
                rows[count] = vi2; cols[count] = vi1; data[count] = lam_div * c2 * c1; count += 1
                rows[count] = vi2; cols[count] = vi2; data[count] = lam_div * c2 * c2; count += 1
                rows[count] = vi2; cols[count] = vi3; data[count] = lam_div * c2 * c3; count += 1

                rows[count] = vi3; cols[count] = vi0; data[count] = lam_div * c3 * c0; count += 1
                rows[count] = vi3; cols[count] = vi1; data[count] = lam_div * c3 * c1; count += 1
                rows[count] = vi3; cols[count] = vi2; data[count] = lam_div * c3 * c2; count += 1
                rows[count] = vi3; cols[count] = vi3; data[count] = lam_div * c3 * c3; count += 1

            # --- 4. Boundary Conditions ---
            # Penalize normal flow component at walls, unknown, and grid edges.
            # Outlets (grid==2) are open boundaries — no penalty.

            # Left
            if x == 0 or (not is_fluid[y, x-1] and grid[y, x-1] != 2):
                rows[count] = u_idx; cols[count] = u_idx; data[count] = lam_bnd; count += 1

            # Right
            if x == width - 1 or (not is_fluid[y, x+1] and grid[y, x+1] != 2):
                rows[count] = u_idx; cols[count] = u_idx; data[count] = lam_bnd; count += 1

            # Top
            if y == 0 or (not is_fluid[y-1, x] and grid[y-1, x] != 2):
                rows[count] = v_idx; cols[count] = v_idx; data[count] = lam_bnd; count += 1

            # Bottom
            if y == height - 1 or (not is_fluid[y+1, x] and grid[y+1, x] != 2):
                rows[count] = v_idx; cols[count] = v_idx; data[count] = lam_bnd; count += 1

    return rows, cols, data, b, count


class GMRFWindMapper:
    """
    GMRF-based wind field estimator.

    Solves for (u, v) wind components by minimizing:
        E(W) = E_z(observations) + E_m(mass conservation)
             + E_o(obstacles) + E_r(regularization)

    Matrix construction is Numba JIT-compiled for performance.
    """

    def __init__(self, lambda_smoothness: float = 5.0,
                 lambda_divergence: float = 50.0,
                 lambda_measurement: float = 100.0,
                 lambda_boundary: float = 10.0,
                 lambda_obstacle: float = 1000.0):
        self.lambda_smoothness = lambda_smoothness
        self.lambda_divergence = lambda_divergence
        self.lambda_measurement = lambda_measurement
        self.lambda_boundary = lambda_boundary
        self.lambda_obstacle = lambda_obstacle

        self.u_field: Optional[np.ndarray] = None
        self.v_field: Optional[np.ndarray] = None

    def solve(self, grid: np.ndarray, resolution: float,
              measured_u: np.ndarray, measured_v: np.ndarray,
              measurement_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the wind field (u, v) on the given grid.

        Args:
            grid: SLAM grid (height, width). 0=free, 1=occupied, 2=outlet, -1=unknown.
            resolution: Cell size in meters.
            measured_u: Measured x-wind component map (height, width).
            measured_v: Measured y-wind component map (height, width).
            measurement_mask: Boolean mask of where measurements exist.

        Returns:
            u_field, v_field: Estimated wind components (height, width).
        """
        height, width = grid.shape
        N = height * width

        is_fluid = ((grid == 0) | (grid == 2)).astype(np.bool_)
        grid_i8 = grid.astype(np.int8)
        mask = measurement_mask.astype(np.bool_)
        mu = measured_u.astype(np.float64)
        mv = measured_v.astype(np.float64)

        # Build triplets via Numba
        rows, cols, data, b, count = _build_triplets(
            grid_i8, is_fluid, mask, mu, mv,
            self.lambda_smoothness, self.lambda_divergence,
            self.lambda_measurement, self.lambda_boundary,
            self.lambda_obstacle
        )

        # Build sparse matrix from valid entries
        Q = sp.coo_matrix(
            (data[:count], (rows[:count], cols[:count])),
            shape=(2 * N, 2 * N)
        ).tocsr()

        # Solve Qx = b
        try:
            x_sol = spsolve(Q, b)
        except Exception as e:
            print(f"GMRF solve failed: {e}")
            return np.zeros((height, width)), np.zeros((height, width))

        self.u_field = x_sol[:N].reshape((height, width))
        self.v_field = x_sol[N:].reshape((height, width))

        # Zero out negligible values
        self.u_field[np.abs(self.u_field) < 1e-4] = 0
        self.v_field[np.abs(self.v_field) < 1e-4] = 0

        return self.u_field, self.v_field
