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


class IndoorGaussianDispersionModel:
    """
    Indoor Gaussian Dispersion Model (IGDM) for environments without airflow.
    Strictly uses Numba acceleration for Dijkstra calculations.

    Equation 18 from paper:
    R(rk|θ) = Qm · exp(-cobs(rk, r0)² / (2·σm²))
    """

    def __init__(self, sigma_m: float = 1.0, occupancy_grid=None):
        self.sigma_m = sigma_m
        self.occupancy_grid = occupancy_grid

        # Cache for Dijkstra distance maps
        self._distance_cache = {}
        self._cache_max_size = 20  # Keep small to save RAM
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._dijkstra_call_count = 0
        self._dijkstra_total_time = 0.0

        print("IGDM: Numba-accelerated mode active.")

    def set_occupancy_grid(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self._distance_cache.clear()

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
        """Compute obstacle-aware distance between two positions."""
        if self.occupancy_grid is None:
            return np.linalg.norm(np.array(pos1) - np.array(pos2))

        gx1, gy1 = self.occupancy_grid.world_to_grid(pos1[0], pos1[1])
        gx2, gy2 = self.occupancy_grid.world_to_grid(pos2[0], pos2[1])

        if not self._is_valid_grid_position(gx1, gy1) or \
           not self._is_valid_grid_position(gx2, gy2):
            return 1000.0

        return self._dijkstra_distance((gx1, gy1), (gx2, gy2))

    def compute_distance_map_from_sensor(self, sensor_position: Tuple[float, float]) -> np.ndarray:
        """Compute/Retrieve cached distance map from sensor position."""
        if self.occupancy_grid is None:
            raise ValueError("Occupancy grid not set")

        gx, gy = self.occupancy_grid.world_to_grid(sensor_position[0], sensor_position[1])

        if not self._is_valid_grid_position(gx, gy):
            raise ValueError(f"Sensor position {sensor_position} invalid")

        cache_key = (gx, gy)
        if cache_key in self._distance_cache:
            self._cache_hits += 1
            return self._distance_cache[cache_key]

        self._cache_misses += 1
        distance_map = self._dijkstra_all_distances((gx, gy))

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
        """Compute distances to ALL reachable cells (Strictly Numba)."""
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
        Efficiently compute concentrations for multiple particles (Vectorized).
        """
        if self.occupancy_grid is None:
            # Simple fallback for no grid (rare)
            concentrations = np.zeros(len(particle_locations))
            for i, (loc, rate) in enumerate(zip(particle_locations, release_rates)):
                concentrations[i] = self.compute_concentration(
                    sensor_position, tuple(loc), rate
                )
            return concentrations

        # 1. Compute distance map (Cached)
        distance_map = self.compute_distance_map_from_sensor(sensor_position)

        # 2. Vectorized World-to-Grid conversion
        ox = self.occupancy_grid.origin_x
        oy = self.occupancy_grid.origin_y
        res = self.occupancy_grid.resolution
        w = self.occupancy_grid.width
        h = self.occupancy_grid.height
        
        # Calculate grid indices for all particles at once
        gxs = np.floor((particle_locations[:, 0] - ox) / res).astype(np.int32)
        gys = np.floor((particle_locations[:, 1] - oy) / res).astype(np.int32)
        
        # 3. Create validity mask (Bounds check)
        valid_mask = (gxs >= 0) & (gxs < w) & (gys >= 0) & (gys < h)
        
        # 4. Initialize result array
        concentrations = np.zeros(len(particle_locations))
        
        # 5. Batch Lookup & Computation
        if np.any(valid_mask):
            # Extract valid indices
            valid_gxs = gxs[valid_mask]
            valid_gys = gys[valid_mask]
            valid_Qm = release_rates[valid_mask]
            
            # O(1) array lookup using advanced indexing
            cobs = distance_map[valid_gys, valid_gxs]
            
            # Vectorized concentration calculation
            concentrations[valid_mask] = valid_Qm * np.exp(-cobs**2 / (2 * self.sigma_m**2))
            
        return concentrations