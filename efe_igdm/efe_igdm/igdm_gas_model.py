"""
Indoor Gaussian Dispersion Model (IGDM) for gas source localization.

Based on: "Gas Source Localization in Unknown Indoor Environments Using
Dual-Mode Information-Theoretic Search" by Kim et al., IEEE RA-L 2025

The IGDM models isotropic gas dispersion in indoor environments without airflow,
accounting for obstacles using Dijkstra's algorithm.
"""

import numpy as np
from typing import Tuple, Optional
import heapq


class IndoorGaussianDispersionModel:
    """
    Indoor Gaussian Dispersion Model (IGDM) for environments without airflow.

    Equation 18 from paper:
    R(rk|θ) = Qm · exp(-cobs(rk, r0)² / (2·σm²))

    where:
    - Qm: Release rate (in g/s or μg/s depending on units)
    - σm: Constant standard deviation of concentration
    - cobs(rk, r0): Distance between sensing point and source location,
                    calculated using Dijkstra's algorithm on occupancy grid

    Key features:
    - Isotropic dispersion (no wind direction)
    - Obstacle-aware distance calculation via Dijkstra
    - Works with incomplete/online maps
    - Constant dispersion parameter (no distance dependency)
    """

    def __init__(self, sigma_m: float = 1.0, occupancy_grid=None):
        """
        Initialize IGDM.

        Parameters:
        -----------
        sigma_m : float
            Constant standard deviation of gas concentration (meters)
            Paper suggests this is tuned experimentally
        occupancy_grid : OccupancyGridMap
            Occupancy grid for Dijkstra distance calculation
        """
        self.sigma_m = sigma_m
        self.occupancy_grid = occupancy_grid

        # Cache for Dijkstra distance maps (to avoid recomputation during RRT)
        # Key: (grid_x, grid_y) of sensor position
        # Value: distance_map numpy array
        self._distance_cache = {}
        self._cache_max_size = 100  # Limit cache size to prevent memory issues
        self._cache_hits = 0
        self._cache_misses = 0

    def set_occupancy_grid(self, occupancy_grid):
        """Update occupancy grid (for online mapping scenarios)."""
        self.occupancy_grid = occupancy_grid
        # Clear cache when map changes
        self._distance_cache.clear()

    def compute_concentration(self,
                            position: Tuple[float, float],
                            source_location: Tuple[float, float],
                            release_rate: float) -> float:
        """
        Compute gas concentration at a given position.

        Implements Equation 18 from the paper.

        Parameters:
        -----------
        position : tuple (x, y)
            Sensing location in meters (world coordinates)
        source_location : tuple (x0, y0)
            Source location in meters (world coordinates)
        release_rate : float
            Gas release rate Qm (units determine output concentration units)

        Returns:
        --------
        concentration : float
            Mean gas concentration (units match release_rate parameter)
        """
        # Convert release rate from g/s to μg/s
        # Qm = release_rate * 1e6
        Qm = release_rate

        # Calculate obstacle-aware distance using Dijkstra
        cobs = self._compute_obstacle_distance(position, source_location)

        # Equation 18: R(rk|θ) = Qm · exp(-cobs² / (2·σm²))
        concentration = Qm * np.exp(-cobs**2 / (2 * self.sigma_m**2))

        return concentration

    def _compute_obstacle_distance(self,
                                   pos1: Tuple[float, float],
                                   pos2: Tuple[float, float]) -> float:
        """
        Compute obstacle-aware distance between two positions using Dijkstra's algorithm.

        This implements the cobs(rk, r0) calculation from the paper.

        Parameters:
        -----------
        pos1 : tuple (x, y)
            First position (typically sensor position)
        pos2 : tuple (x, y)
            Second position (typically source location)

        Returns:
        --------
        distance : float
            Shortest path distance considering obstacles (meters)
        """
        if self.occupancy_grid is None:
            # Fallback to Euclidean distance if no grid available
            return np.linalg.norm(np.array(pos1) - np.array(pos2))

        # Convert world coordinates to grid indices
        gx1, gy1 = self.occupancy_grid.world_to_grid(pos1[0], pos1[1])
        gx2, gy2 = self.occupancy_grid.world_to_grid(pos2[0], pos2[1])

        # Check if positions are valid
        if not self._is_valid_grid_position(gx1, gy1) or \
           not self._is_valid_grid_position(gx2, gy2):
            # Return large distance for invalid positions
            return 1000.0

        # Use Dijkstra to find shortest path distance
        distance = self._dijkstra_distance((gx1, gy1), (gx2, gy2))

        return distance

    def compute_distance_map_from_sensor(self,
                                        sensor_position: Tuple[float, float]) -> np.ndarray:
        """
        Compute distance map from sensor position to all grid cells using Dijkstra.

        This is the optimized version mentioned in the paper:
        "By modifying the termination condition of Dijkstra's algorithm to check
        all grids in the map, the distance between all source location hypotheses
        (i.e., particle locations) and the current mobile sensor can be computed
        with a single execution of Dijkstra's algorithm."

        Uses caching to avoid recomputing for same grid cell during RRT exploration.

        Parameters:
        -----------
        sensor_position : tuple (x, y)
            Current sensor position in world coordinates

        Returns:
        --------
        distance_map : np.ndarray
            2D array where each cell contains the shortest path distance
            from sensor position to that cell
        """
        if self.occupancy_grid is None:
            raise ValueError("Occupancy grid not set")

        # Convert to grid coordinates
        gx, gy = self.occupancy_grid.world_to_grid(sensor_position[0], sensor_position[1])

        if not self._is_valid_grid_position(gx, gy):
            raise ValueError(f"Sensor position {sensor_position} is not in valid grid space")

        # Check cache first
        cache_key = (gx, gy)
        if cache_key in self._distance_cache:
            self._cache_hits += 1
            return self._distance_cache[cache_key]

        # Cache miss - compute distance map
        self._cache_misses += 1
        distance_map = self._dijkstra_all_distances((gx, gy))

        # Add to cache (with size limit)
        if len(self._distance_cache) >= self._cache_max_size:
            # Remove oldest entry (FIFO)
            self._distance_cache.pop(next(iter(self._distance_cache)))

        self._distance_cache[cache_key] = distance_map

        return distance_map

    def clear_cache(self):
        """Clear the distance map cache."""
        self._distance_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self):
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._distance_cache)
        }

    def _dijkstra_distance(self, start_grid: Tuple[int, int],
                          end_grid: Tuple[int, int]) -> float:
        """
        Dijkstra's algorithm to find shortest path distance between two grid cells.

        Parameters:
        -----------
        start_grid : tuple (gx, gy)
            Start grid cell indices
        end_grid : tuple (gx, gy)
            End grid cell indices

        Returns:
        --------
        distance : float
            Shortest path distance in meters (or inf if no path exists)
        """
        # Initialize
        width, height = self.occupancy_grid.width, self.occupancy_grid.height
        distances = np.full((height, width), np.inf)
        distances[start_grid[1], start_grid[0]] = 0.0

        # Priority queue: (distance, x, y)
        pq = [(0.0, start_grid[0], start_grid[1])]
        visited = set()

        # 8-connected neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while pq:
            current_dist, x, y = heapq.heappop(pq)

            # Found target
            if (x, y) == end_grid:
                return current_dist

            # Already visited
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Check all neighbors
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy

                # Check bounds and collision
                if not self._is_valid_grid_position(nx, ny):
                    continue
                if self.occupancy_grid.grid[ny, nx] > 0:  # Occupied
                    continue

                # Calculate edge cost (diagonal vs straight)
                if dx != 0 and dy != 0:
                    edge_cost = self.occupancy_grid.resolution * np.sqrt(2)
                else:
                    edge_cost = self.occupancy_grid.resolution

                new_dist = current_dist + edge_cost

                # Update distance if shorter path found
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

        # No path found
        return np.inf

    def _dijkstra_all_distances(self, start_grid: Tuple[int, int]) -> np.ndarray:
        """
        Modified Dijkstra's algorithm that computes distances to ALL reachable cells.

        This is the optimization mentioned in the paper for efficient particle
        filter updates.

        Parameters:
        -----------
        start_grid : tuple (gx, gy)
            Start grid cell indices (typically current sensor position)

        Returns:
        --------
        distance_map : np.ndarray
            2D array with distances from start to all cells (inf for unreachable)
        """
        width, height = self.occupancy_grid.width, self.occupancy_grid.height
        distances = np.full((height, width), np.inf)
        distances[start_grid[1], start_grid[0]] = 0.0

        # Priority queue: (distance, x, y)
        pq = [(0.0, start_grid[0], start_grid[1])]
        visited = np.zeros((height, width), dtype=bool)

        # 8-connected neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while pq:
            current_dist, x, y = heapq.heappop(pq)

            # Already visited
            if visited[y, x]:
                continue
            visited[y, x] = True

            # Check all neighbors
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy

                # Check bounds
                if not self._is_valid_grid_position(nx, ny):
                    continue

                # Check if already visited
                if visited[ny, nx]:
                    continue

                # Check collision
                if self.occupancy_grid.grid[ny, nx] > 0:  # Occupied
                    continue

                # Calculate edge cost (diagonal vs straight)
                if dx != 0 and dy != 0:
                    edge_cost = self.occupancy_grid.resolution * np.sqrt(2)
                else:
                    edge_cost = self.occupancy_grid.resolution

                new_dist = current_dist + edge_cost

                # Update distance if shorter path found
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

        return distances

    def _is_valid_grid_position(self, gx: int, gy: int) -> bool:
        """Check if grid position is within bounds."""
        if self.occupancy_grid is None:
            return False
        return (0 <= gx < self.occupancy_grid.width and
                0 <= gy < self.occupancy_grid.height)

    def likelihood(self, position: Tuple[float, float],
                   source_location: Tuple[float, float],
                   release_rate: float) -> float:
        """
        Wrapper for compute_concentration for compatibility with particle filter.

        This maintains the same interface as GaussianPlumeModel.
        """
        return self.compute_concentration(position, source_location, release_rate)

    def compute_concentrations_batch(self,
                                    sensor_position: Tuple[float, float],
                                    particle_locations: np.ndarray,
                                    release_rates: np.ndarray) -> np.ndarray:
        """
        Efficiently compute concentrations for multiple particles using distance map.

        This is the optimized version for particle filter updates.

        Parameters:
        -----------
        sensor_position : tuple (x, y)
            Current sensor position
        particle_locations : np.ndarray, shape (N, 2)
            Array of particle source locations [(x0, y0), ...]
        release_rates : np.ndarray, shape (N,)
            Array of release rates [Q0, ...]

        Returns:
        --------
        concentrations : np.ndarray, shape (N,)
            Predicted concentrations for each particle
        """
        if self.occupancy_grid is None:
            # Fallback to individual computation
            concentrations = np.zeros(len(particle_locations))
            for i, (loc, rate) in enumerate(zip(particle_locations, release_rates)):
                concentrations[i] = self.compute_concentration(
                    sensor_position, tuple(loc), rate
                )
            return concentrations

        # Compute distance map once for all particles
        distance_map = self.compute_distance_map_from_sensor(sensor_position)

        # Convert particle locations to grid coordinates
        concentrations = np.zeros(len(particle_locations))
        # Qm = release_rates * 1e6  # Convert to μg/s
        Qm = release_rates

        for i, (x0, y0) in enumerate(particle_locations):
            gx, gy = self.occupancy_grid.world_to_grid(x0, y0)

            if self._is_valid_grid_position(gx, gy):
                cobs = distance_map[gy, gx]
                # Equation 18
                concentrations[i] = Qm[i] * np.exp(-cobs**2 / (2 * self.sigma_m**2))
            else:
                concentrations[i] = 0.0

        return concentrations
