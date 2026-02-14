"""
Potential Flow Wind Estimator.

Solves Laplace's equation (∇²φ = 0) on the explored SLAM grid to estimate
a physically plausible wind field from known outlet positions.

Boundary conditions:
  - Outlets: fixed potential (source/sink based on clustering or anemometer)
  - Walls/obstacles: zero normal flux (natural Neumann BC via exclusion)
  - Unknown cells: treated as walls (conservative)

Wind = ∇φ (gradient of potential field)
"""

import numpy as np
from numba import njit
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from typing import Optional, Tuple


@njit(cache=True)
def _jacobi_iteration(phi, grid, outlet_labels, outlet_potentials,
                      max_iter, tol):
    """
    Jacobi iteration to solve Laplace's equation on the grid.

    Parameters
    ----------
    phi : ndarray (height, width)
        Potential field (modified in place).
    grid : ndarray (height, width)
        SLAM grid: 0=free, 1=occupied, 2=outlet, -1=unknown.
    outlet_labels : ndarray (height, width)
        Label for each outlet cluster (0 = not outlet).
    outlet_potentials : ndarray (max_label+1,)
        Potential value for each outlet cluster label.
    max_iter : int
        Maximum Jacobi iterations.
    tol : float
        Convergence tolerance (max absolute change).

    Returns
    -------
    iterations : int
        Number of iterations performed.
    max_change : float
        Final max absolute change.
    """
    height, width = phi.shape

    # Enforce outlet boundary conditions
    for y in range(height):
        for x in range(width):
            lbl = outlet_labels[y, x]
            if lbl > 0:
                phi[y, x] = outlet_potentials[lbl]

    for iteration in range(max_iter):
        max_change = 0.0

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Skip non-free cells and outlet cells (fixed BC)
                if grid[y, x] != 0:
                    continue

                # Average of free neighbors
                count = 0
                total = 0.0

                if grid[y - 1, x] == 0 or outlet_labels[y - 1, x] > 0:
                    total += phi[y - 1, x]
                    count += 1
                if grid[y + 1, x] == 0 or outlet_labels[y + 1, x] > 0:
                    total += phi[y + 1, x]
                    count += 1
                if grid[y, x - 1] == 0 or outlet_labels[y, x - 1] > 0:
                    total += phi[y, x - 1]
                    count += 1
                if grid[y, x + 1] == 0 or outlet_labels[y, x + 1] > 0:
                    total += phi[y, x + 1]
                    count += 1

                if count == 0:
                    continue

                new_val = total / count
                change = abs(new_val - phi[y, x])
                if change > max_change:
                    max_change = change
                phi[y, x] = new_val

        # Check convergence
        if max_change < tol:
            return iteration + 1, max_change

    return max_iter, max_change


@njit(cache=True)
def _compute_gradient(phi, grid, resolution):
    """
    Compute wind vectors as gradient of the potential field.

    Uses central differences for interior cells, one-sided at boundaries.
    Only computes for free cells.

    Returns
    -------
    wind_u : ndarray (height, width) - x-component
    wind_v : ndarray (height, width) - y-component
    """
    height, width = phi.shape
    wind_u = np.zeros((height, width), dtype=np.float64)
    wind_v = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            if grid[y, x] != 0:
                continue

            # dφ/dx (central difference where possible)
            if x > 0 and x < width - 1 and (grid[y, x - 1] == 0 or grid[y, x - 1] == 2) and (grid[y, x + 1] == 0 or grid[y, x + 1] == 2):
                wind_u[y, x] = (phi[y, x + 1] - phi[y, x - 1]) / (2.0 * resolution)
            elif x < width - 1 and (grid[y, x + 1] == 0 or grid[y, x + 1] == 2):
                wind_u[y, x] = (phi[y, x + 1] - phi[y, x]) / resolution
            elif x > 0 and (grid[y, x - 1] == 0 or grid[y, x - 1] == 2):
                wind_u[y, x] = (phi[y, x] - phi[y, x - 1]) / resolution

            # dφ/dy
            if y > 0 and y < height - 1 and (grid[y - 1, x] == 0 or grid[y - 1, x] == 2) and (grid[y + 1, x] == 0 or grid[y + 1, x] == 2):
                wind_v[y, x] = (phi[y + 1, x] - phi[y - 1, x]) / (2.0 * resolution)
            elif y < height - 1 and (grid[y + 1, x] == 0 or grid[y + 1, x] == 2):
                wind_v[y, x] = (phi[y + 1, x] - phi[y, x]) / resolution
            elif y > 0 and (grid[y - 1, x] == 0 or grid[y - 1, x] == 2):
                wind_v[y, x] = (phi[y, x] - phi[y - 1, x]) / resolution

    return wind_u, wind_v


class PotentialFlowEstimator:
    """
    Estimates wind field by solving Laplace's equation on the SLAM grid.

    Outlets are clustered by connected components and assigned different
    potential values. The solver uses Jacobi iteration (Numba JIT).
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol

        # Results
        self.phi: Optional[np.ndarray] = None
        self.wind_u: Optional[np.ndarray] = None
        self.wind_v: Optional[np.ndarray] = None
        self.outlet_labels: Optional[np.ndarray] = None
        self.num_clusters = 0

    def _compute_cluster_centroid(self, cluster_label: int) -> Tuple[float, float]:
        """Compute centroid (gy, gx) of a labeled cluster."""
        ys, xs = np.where(self.outlet_labels == cluster_label)
        return float(np.mean(ys)), float(np.mean(xs))

    def _assign_potentials_from_wind(self, measured_vx, measured_vy,
                                     measurement_count, resolution,
                                     num_real_outlets: int):
        """
        Assign potentials to outlet clusters using anemometer measurements.

        For each real outlet cluster, compute the average dot product of nearby
        measured wind vectors with the direction toward the outlet centroid.
        - Positive dot product → wind flows toward outlet → outlet is a sink → HIGH potential
        - Negative dot product → wind flows away from outlet → outlet is a source → LOW potential

        Returns outlet_potentials array indexed by cluster label.
        """
        outlet_potentials = np.zeros(self.num_clusters + 1, dtype=np.float64)

        has_measurements = np.any(measurement_count > 0)
        if not has_measurements:
            # No anemometer data: fall back to evenly spaced potentials
            for i in range(1, self.num_clusters + 1):
                outlet_potentials[i] = (i - 1) / max(self.num_clusters - 1, 1)
            return outlet_potentials

        measured_mask = measurement_count > 0
        m_ys, m_xs = np.where(measured_mask)

        # Compute dot product score for each real outlet cluster
        scores = {}  # cluster_label -> avg dot product
        for c in range(1, num_real_outlets + 1):
            cy, cx = self._compute_cluster_centroid(c)

            # Direction vectors from each measured cell toward this outlet centroid
            dx = cx - m_xs.astype(np.float64)
            dy = cy - m_ys.astype(np.float64)
            dists = np.sqrt(dx**2 + dy**2)
            valid = dists > 0.5  # skip cells too close to centroid
            if not np.any(valid):
                scores[c] = 0.0
                continue

            dx_n = dx[valid] / dists[valid]
            dy_n = dy[valid] / dists[valid]

            # Measured wind at those cells
            wvx = measured_vx[m_ys[valid], m_xs[valid]]
            wvy = measured_vy[m_ys[valid], m_xs[valid]]

            # Dot product: positive if wind flows toward outlet
            dots = wvx * dx_n + wvy * dy_n
            scores[c] = float(np.mean(dots))

        # Assign potentials based on scores
        # Positive score → sink (wind flows toward it) → HIGH potential
        # Negative score → source (wind flows away) → LOW potential
        if num_real_outlets == 1:
            # 1 real outlet + frontier as cluster 2
            if scores.get(1, 0.0) >= 0:
                # Wind toward outlet → outlet is sink → high potential
                outlet_potentials[1] = 1.0  # outlet
                outlet_potentials[2] = 0.0  # frontier
            else:
                # Wind away from outlet → outlet is source → low potential
                outlet_potentials[1] = 0.0  # outlet
                outlet_potentials[2] = 1.0  # frontier
        else:
            # Multiple real outlets: rank by score
            # Highest score (most sink-like) → highest potential
            labels = list(range(1, self.num_clusters + 1))
            label_scores = [(c, scores.get(c, 0.0)) for c in labels]
            label_scores.sort(key=lambda x: x[1])  # ascending by score
            for rank, (c, _) in enumerate(label_scores):
                outlet_potentials[c] = rank / max(len(label_scores) - 1, 1)

        return outlet_potentials

    def solve(self, grid: np.ndarray, resolution: float,
              measured_vx: np.ndarray = None, measured_vy: np.ndarray = None,
              measurement_count: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the wind field on the given SLAM grid.

        Parameters
        ----------
        grid : ndarray (height, width)
            SLAM grid with values: -1=unknown, 0=free, 1=occupied, 2=outlet.
        resolution : float
            Grid cell size in meters.
        measured_vx, measured_vy : ndarray (height, width), optional
            Anemometer wind measurements (NaN where unmeasured).
            Used to determine flow direction at outlets.
        measurement_count : ndarray (height, width), optional
            Number of anemometer measurements per cell.

        Returns
        -------
        wind_u : ndarray (height, width) - wind x-component (m/s normalized)
        wind_v : ndarray (height, width) - wind y-component (m/s normalized)
        """
        height, width = grid.shape

        # 1. Cluster outlet cells using connected components
        outlet_mask = (grid == 2)
        self.outlet_labels, self.num_clusters = label(outlet_mask)
        num_real_outlets = self.num_clusters

        if self.num_clusters == 0:
            self.wind_u = np.zeros((height, width), dtype=np.float64)
            self.wind_v = np.zeros((height, width), dtype=np.float64)
            self.phi = np.zeros((height, width), dtype=np.float64)
            return self.wind_u, self.wind_v

        if self.num_clusters == 1:
            # Single outlet: use frontier (free cells adjacent to unknown)
            # as a virtual second boundary — air must flow to/from unexplored
            is_free = (grid == 0)
            is_unknown = (grid == -1)
            struct = generate_binary_structure(2, 2)  # 8-connectivity
            unknown_dilated = binary_dilation(is_unknown, structure=struct)
            frontier_mask = is_free & unknown_dilated

            if np.any(frontier_mask):
                # Add frontier as cluster 2
                self.outlet_labels[frontier_mask] = 2
                self.num_clusters = 2

        if self.num_clusters < 2:
            self.wind_u = np.zeros((height, width), dtype=np.float64)
            self.wind_v = np.zeros((height, width), dtype=np.float64)
            self.phi = np.zeros((height, width), dtype=np.float64)
            return self.wind_u, self.wind_v

        # 2. Assign potentials using anemometer data for correct flow direction
        if measured_vx is not None and measurement_count is not None:
            outlet_potentials = self._assign_potentials_from_wind(
                measured_vx, measured_vy, measurement_count,
                resolution, num_real_outlets
            )
        else:
            # Fallback: evenly spaced (arbitrary direction)
            outlet_potentials = np.zeros(self.num_clusters + 1, dtype=np.float64)
            for i in range(1, self.num_clusters + 1):
                outlet_potentials[i] = (i - 1) / (self.num_clusters - 1)

        # 3. Initialize potential field
        self.phi = np.full((height, width), 0.5, dtype=np.float64)

        # 4. Solve Laplace via Jacobi
        # Mark all BC cells (outlets + virtual frontier) in grid copy
        # so Jacobi skips them (it only updates cells where grid == 0)
        outlet_labels_i32 = self.outlet_labels.astype(np.int32)
        grid_i8 = grid.astype(np.int8)
        grid_i8[outlet_labels_i32 > 0] = 2

        iterations, max_change = _jacobi_iteration(
            self.phi, grid_i8, outlet_labels_i32, outlet_potentials,
            self.max_iter, self.tol
        )

        # 5. Compute wind as gradient
        self.wind_u, self.wind_v = _compute_gradient(self.phi, grid_i8, resolution)

        return self.wind_u, self.wind_v

    def get_wind_at_grid(self, gx: int, gy: int) -> Tuple[float, float]:
        """Get wind vector at grid coordinates."""
        if self.wind_u is None:
            return (0.0, 0.0)
        if 0 <= gx < self.wind_u.shape[1] and 0 <= gy < self.wind_u.shape[0]:
            return (self.wind_u[gy, gx], self.wind_v[gy, gx])
        return (0.0, 0.0)
