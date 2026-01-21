"""
Indoor Gaussian Dispersion Model (IGDM) for gas source localization.

Based on: "Gas Source Localization in Unknown Indoor Environments Using
Dual-Mode Information-Theoretic Search" by Kim et al., IEEE RA-L 2025

The IGDM models isotropic gas dispersion in indoor environments without airflow,
accounting for obstacles using Dijkstra's algorithm.

STRICT MODE: Numba acceleration is REQUIRED.
"""

import numpy as np
from typing import Tuple, Optional
import heapq
from numba import jit

# ============================================================================
# NUMBA-ACCELERATED DIJKSTRA ALGORITHM
# ============================================================================

@jit(nopython=True, cache=True)
def _dijkstra_numba_core(grid_data, start_gx, start_gy, width, height, resolution):
    """
    Numba-compiled core Dijkstra algorithm using standard heapq.

    Parameters:
    -----------
    grid_data : np.ndarray
        Occupancy grid (0=free, >0=occupied)
    start_gx, start_gy : int
        Start coordinates
    width, height : int
        Grid dimensions
    resolution : float
        Grid resolution

    Returns:
    --------
    distances : np.ndarray
        Distance map (float64)
    """
    # Initialize distances to infinity
    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[start_gy, start_gx] = 0.0

    # Priority queue: list of tuples (distance, gx, gy)
    pq = [(0.0, start_gx, start_gy)]

    # Pre-compute edge costs
    straight_cost = resolution
    diag_cost = resolution * 1.41421356  # sqrt(2)

    # 8-connected neighbors: (dx, dy, cost)
    neighbors = [
        (-1, 0, straight_cost), (1, 0, straight_cost),
        (0, -1, straight_cost), (0, 1, straight_cost),
        (-1, -1, diag_cost), (-1, 1, diag_cost),
        (1, -1, diag_cost), (1, 1, diag_cost)
    ]

    while len(pq) > 0:
        d, cx, cy = heapq.heappop(pq)

        # Optimization: If we found a shorter path already, skip
        if d > distances[cy, cx]:
            continue

        for dx, dy, cost in neighbors:
            nx, ny = cx + dx, cy + dy

            # Bounds check
            if 0 <= nx < width and 0 <= ny < height:
                # Collision check (0 is free)
                if grid_data[ny, nx] == 0:
                    new_dist = d + cost

                    # Relaxation
                    if new_dist < distances[ny, nx]:
                        distances[ny, nx] = new_dist
                        heapq.heappush(pq, (new_dist, nx, ny))

    return distances


@jit(nopython=True, cache=True)
def _dijkstra_coarse_numba(grid_data, start_c, start_r, width, height, resolution):
    """
    Numba-compiled Dijkstra for coarse grid (optimized for larger cells).

    Parameters:
    -----------
    grid_data : np.ndarray
        Coarse occupancy grid (0=free, >0=occupied)
    start_c, start_r : int
        Start coordinates (column, row)
    width, height : int
        Grid dimensions
    resolution : float
        Coarse grid resolution (e.g., 0.5m)

    Returns:
    --------
    distances : np.ndarray
        Distance map in meters (float64)
    """
    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[start_r, start_c] = 0.0

    pq = [(0.0, start_c, start_r)]

    straight_cost = resolution
    diag_cost = resolution * 1.41421356

    neighbors = [
        (-1, 0, straight_cost), (1, 0, straight_cost),
        (0, -1, straight_cost), (0, 1, straight_cost),
        (-1, -1, diag_cost), (-1, 1, diag_cost),
        (1, -1, diag_cost), (1, 1, diag_cost)
    ]

    while len(pq) > 0:
        d, cc, cr = heapq.heappop(pq)

        if d > distances[cr, cc]:
            continue

        for dc, dr, cost in neighbors:
            nc, nr = cc + dc, cr + dr

            if 0 <= nc < width and 0 <= nr < height:
                if grid_data[nr, nc] == 0:
                    new_dist = d + cost

                    if new_dist < distances[nr, nc]:
                        distances[nr, nc] = new_dist
                        heapq.heappush(pq, (new_dist, nc, nr))

    return distances


class IndoorGaussianDispersionModel:
    """
    Indoor Gaussian Dispersion Model (IGDM) for environments without airflow.
    Strictly uses Numba acceleration for Dijkstra calculations.

    NEW: Dual-grid system for performance optimization:
    - Fine grid for compatibility with existing code
    - Coarse grid for faster Dijkstra pathfinding

    Equation 18 from paper:
    R(rk|θ) = Qm · exp(-cobs(rk, r0)² / (2·σm²))
    """

    def __init__(self, sigma_m: float = 1.0, occupancy_grid=None, coarse_resolution: float = 0.5):
        self.sigma_m = sigma_m
        self.occupancy_grid = occupancy_grid
        self.coarse_resolution = coarse_resolution

        # Coarse grid for optimized Dijkstra
        self.coarse_grid = None
        self.coarse_width = 0
        self.coarse_height = 0

        if occupancy_grid is not None:
            self._create_coarse_grid()

        # Cache for Dijkstra distance maps (on coarse grid)
        self._distance_cache = {}
        self._cache_max_size = 20  # Keep small to save RAM
        self._cache_hits = 0
        self._cache_misses = 0

        # Performance tracking
        self._dijkstra_call_count = 0
        self._dijkstra_total_time = 0.0

        print(f"IGDM: Numba-accelerated mode active. Coarse grid: {coarse_resolution}m")

    def set_occupancy_grid(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self._create_coarse_grid()
        self._distance_cache.clear()

    def _create_coarse_grid(self):
        """
        Downsample the fine grid to a coarse grid using max pooling.
        If any fine cell in a coarse block is an obstacle, the coarse block is an obstacle.
        """
        if self.occupancy_grid is None:
            return

        fine_res = self.occupancy_grid.resolution
        fine_grid = self.occupancy_grid.grid
        width_m = self.occupancy_grid.real_world_width
        height_m = self.occupancy_grid.real_world_height

        # Calculate coarse grid dimensions
        self.coarse_width = int(np.ceil(width_m / self.coarse_resolution))
        self.coarse_height = int(np.ceil(height_m / self.coarse_resolution))

        ratio = self.coarse_resolution / fine_res
        self.coarse_grid = np.zeros((self.coarse_height, self.coarse_width), dtype=np.int8)

        for r in range(self.coarse_height):
            for c in range(self.coarse_width):
                # Calculate bounds in fine grid indices
                r_start = int(r * ratio)
                r_end = min(int((r + 1) * ratio), fine_grid.shape[0])
                c_start = int(c * ratio)
                c_end = min(int((c + 1) * ratio), fine_grid.shape[1])

                # Extract chunk and check for ANY obstacle (value > 0)
                chunk = fine_grid[r_start:r_end, c_start:c_end]
                if np.any(chunk > 0):
                    self.coarse_grid[r, c] = 1

        print(f"IGDM: Coarse grid created: {self.coarse_width}x{self.coarse_height} cells")

    def _world_to_coarse_idx(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to coarse grid indices."""
        if self.occupancy_grid is None:
            return (0, 0)

        # Calculate relative position
        rel_x = x - self.occupancy_grid.origin_x
        rel_y = y - self.occupancy_grid.origin_y

        c = int(rel_x / self.coarse_resolution)
        r = int(rel_y / self.coarse_resolution)

        # Clamp to bounds
        c = max(0, min(c, self.coarse_width - 1))
        r = max(0, min(r, self.coarse_height - 1))

        return (c, r)

    def compute_concentration(self, position: Tuple[float, float],
                            source_location: Tuple[float, float],
                            release_rate: float) -> float:
        """Compute gas concentration at a single position."""
        Qm = release_rate
        cobs = self._compute_obstacle_distance(position, source_location)
        concentration = Qm * np.exp(-cobs**2 / (2 * self.sigma_m**2))
        return concentration

    def _compute_obstacle_distance(self, pos1: Tuple[float, float],
                                   pos2: Tuple[float, float]) -> float:
        """Compute obstacle-aware distance between two positions using coarse grid."""
        if self.occupancy_grid is None or self.coarse_grid is None:
            return np.linalg.norm(np.array(pos1) - np.array(pos2))

        # Use coarse grid for distance calculation
        c1, r1 = self._world_to_coarse_idx(pos1[0], pos1[1])
        c2, r2 = self._world_to_coarse_idx(pos2[0], pos2[1])

        # Check if positions are in obstacles
        if self.coarse_grid[r1, c1] > 0 or self.coarse_grid[r2, c2] > 0:
            return 1000.0

        # Get distance from cached map
        distance_map = self._dijkstra_all_distances_coarse((c1, r1))

        return distance_map[r2, c2]

    def compute_distance_map_from_sensor(self, sensor_position: Tuple[float, float]) -> np.ndarray:
        """Compute/Retrieve cached distance map from sensor position (using coarse grid)."""
        if self.occupancy_grid is None or self.coarse_grid is None:
            raise ValueError("Occupancy grid not set")

        # Use coarse grid coordinates for caching
        c, r = self._world_to_coarse_idx(sensor_position[0], sensor_position[1])

        cache_key = (c, r)
        if cache_key in self._distance_cache:
            self._cache_hits += 1
            return self._distance_cache[cache_key]

        self._cache_misses += 1
        distance_map = self._dijkstra_all_distances_coarse((c, r))

        if len(self._distance_cache) >= self._cache_max_size:
            self._distance_cache.pop(next(iter(self._distance_cache)))

        self._distance_cache[cache_key] = distance_map
        return distance_map

    def clear_cache(self):
        self._distance_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        avg_time = (self._dijkstra_total_time / self._dijkstra_call_count
                   if self._dijkstra_call_count > 0 else 0.0)
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._distance_cache),
            'dijkstra_calls': self._dijkstra_call_count,
            'dijkstra_avg_time_ms': avg_time * 1000,
        }

    def _dijkstra_distance(self, start_grid: Tuple[int, int], end_grid: Tuple[int, int]) -> float:
        """Get distance between two points using the cached map approach."""
        distance_map = self._dijkstra_all_distances(start_grid)
        return distance_map[end_grid[1], end_grid[0]]

    def _dijkstra_all_distances(self, start_grid: Tuple[int, int]) -> np.ndarray:
        """Compute distances to ALL reachable cells (Strictly Numba) on fine grid."""
        import time
        width = self.occupancy_grid.width
        height = self.occupancy_grid.height
        resolution = self.occupancy_grid.resolution
        grid_data = self.occupancy_grid.grid
        start_gx, start_gy = start_grid

        start_time = time.perf_counter()

        # Direct call to JIT-compiled function
        result = _dijkstra_numba_core(
            grid_data, start_gx, start_gy, width, height, resolution
        )

        elapsed = time.perf_counter() - start_time
        self._dijkstra_call_count += 1
        self._dijkstra_total_time += elapsed

        return result

    def _dijkstra_all_distances_coarse(self, start_coarse: Tuple[int, int]) -> np.ndarray:
        """Compute distances to ALL reachable cells on coarse grid (faster)."""
        import time

        if self.coarse_grid is None:
            raise ValueError("Coarse grid not initialized")

        start_c, start_r = start_coarse

        start_time = time.perf_counter()

        # Call coarse-grid Dijkstra
        result = _dijkstra_coarse_numba(
            self.coarse_grid, start_c, start_r,
            self.coarse_width, self.coarse_height, self.coarse_resolution
        )

        elapsed = time.perf_counter() - start_time
        self._dijkstra_call_count += 1
        self._dijkstra_total_time += elapsed

        return result

    def _is_valid_grid_position(self, gx: int, gy: int) -> bool:
        if self.occupancy_grid is None:
            return False
        return (0 <= gx < self.occupancy_grid.width and
                0 <= gy < self.occupancy_grid.height)

    def likelihood(self, position: Tuple[float, float],
                   source_location: Tuple[float, float],
                   release_rate: float) -> float:
        return self.compute_concentration(position, source_location, release_rate)

    def compute_concentrations_batch(self,
                                    sensor_position: Tuple[float, float],
                                    particle_locations: np.ndarray,
                                    release_rates: np.ndarray) -> np.ndarray:
        """
        Efficiently compute concentrations for multiple particles (Vectorized with coarse grid).
        """
        if self.occupancy_grid is None or self.coarse_grid is None:
            # Simple fallback for no grid (rare)
            concentrations = np.zeros(len(particle_locations))
            for i, (loc, rate) in enumerate(zip(particle_locations, release_rates)):
                concentrations[i] = self.compute_concentration(
                    sensor_position, tuple(loc), rate
                )
            return concentrations

        # 1. Compute distance map on coarse grid (Cached)
        distance_map = self.compute_distance_map_from_sensor(sensor_position)

        # 2. Vectorized World-to-Coarse-Grid conversion
        ox = self.occupancy_grid.origin_x
        oy = self.occupancy_grid.origin_y

        # Calculate coarse grid indices for all particles at once
        rel_xs = particle_locations[:, 0] - ox
        rel_ys = particle_locations[:, 1] - oy
        cs = np.floor(rel_xs / self.coarse_resolution).astype(np.int32)
        rs = np.floor(rel_ys / self.coarse_resolution).astype(np.int32)

        # 3. Create validity mask (Bounds check)
        valid_mask = (cs >= 0) & (cs < self.coarse_width) & (rs >= 0) & (rs < self.coarse_height)

        # 4. Initialize result array
        concentrations = np.zeros(len(particle_locations))

        # 5. Batch Lookup & Computation
        if np.any(valid_mask):
            # Extract valid indices
            valid_cs = cs[valid_mask]
            valid_rs = rs[valid_mask]
            valid_Qm = release_rates[valid_mask]

            # O(1) array lookup using advanced indexing on coarse grid
            cobs = distance_map[valid_rs, valid_cs]

            # Vectorized concentration calculation
            concentrations[valid_mask] = valid_Qm * np.exp(-cobs**2 / (2 * self.sigma_m**2))

        return concentrations