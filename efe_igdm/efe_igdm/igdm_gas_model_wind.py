"""
Indoor Gaussian Dispersion Model (IGDM) with Wind Incorporation.

Based on: "Gas Source Localization in Unknown Indoor Environments Using
Dual-Mode Information-Theoretic Search" by Kim et al., IEEE RA-L 2025

This version extends IGDM to incorporate wind by adjusting distances in x and y
directions. Wind effectively "stretches" or "compresses" the dispersion pattern.
"""

import numpy as np
from typing import Tuple, Optional
import heapq


class IndoorGaussianDispersionModelWind:
    """
    Wind-aware Indoor Gaussian Dispersion Model (IGDM).

    Extends the base IGDM equation:
    R(rk|θ) = Qm · exp(-cobs(rk, r0)² / (2·σm²))

    Wind incorporation:
    - Adjusts the effective distance by wind in x and y directions
    - Positive wind_x means wind blowing in +x direction
    - Positive wind_y means wind blowing in +y direction
    - Wind reduces effective distance downwind, increases it upwind

    Key features:
    - Simple wind model suitable for indoor environments
    - Obstacle-aware distance calculation via Dijkstra
    - Wind effect scales with distance
    - Compatible with particle filter framework
    """

    def __init__(self, sigma_m: float = 1.0, occupancy_grid=None,
                 wind_x: float = 0.0, wind_y: float = 0.0):
        """
        Initialize Wind-aware IGDM.

        Parameters:
        -----------
        sigma_m : float
            Constant standard deviation of gas concentration (meters)
        occupancy_grid : OccupancyGridMap
            Occupancy grid for Dijkstra distance calculation
        wind_x : float
            Wind velocity in x direction (m/s), positive means +x direction
        wind_y : float
            Wind velocity in y direction (m/s), positive means +y direction
        """
        self.sigma_m = sigma_m
        self.occupancy_grid = occupancy_grid
        self.wind_x = wind_x
        self.wind_y = wind_y

        # Cache for Dijkstra distance maps
        self._distance_cache = {}
        self._cache_max_size = 100
        self._cache_hits = 0
        self._cache_misses = 0

    def set_wind(self, wind_x: float, wind_y: float):
        """
        Update wind parameters.

        Parameters:
        -----------
        wind_x : float
            Wind velocity in x direction (m/s)
        wind_y : float
            Wind velocity in y direction (m/s)
        """
        self.wind_x = wind_x
        self.wind_y = wind_y

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
        Compute gas concentration at a given position with wind effects.

        Wind-adjusted model:
        1. Calculate base obstacle-aware distance using Dijkstra
        2. Adjust distance based on wind in x and y directions
        3. Apply Gaussian dispersion model

        Parameters:
        -----------
        position : tuple (x, y)
            Sensing location in meters (world coordinates)
        source_location : tuple (x0, y0)
            Source location in meters (world coordinates)
        release_rate : float
            Gas release rate Qm

        Returns:
        --------
        concentration : float
            Mean gas concentration (units match release_rate parameter)
        """
        Qm = release_rate

        # Calculate obstacle-aware distance using Dijkstra
        cobs = self._compute_obstacle_distance_with_wind(position, source_location)

        # Wind-adjusted equation: R(rk|θ) = Qm · exp(-cobs_wind² / (2·σm²))
        concentration = Qm * np.exp(-cobs**2 / (2 * self.sigma_m**2))

        return concentration

    def _compute_obstacle_distance_with_wind(self,
                                            pos1: Tuple[float, float],
                                            pos2: Tuple[float, float]) -> float:
        """
        Compute obstacle-aware distance with wind adjustment.

        Wind effect:
        - Calculate vector from source (pos2) to sensor (pos1)
        - Reduce distance in direction of wind (downwind)
        - Increase distance against wind (upwind)

        Implementation:
        - dx_wind = dx - wind_x * |dx|/distance_factor
        - dy_wind = dy - wind_y * |dy|/distance_factor

        This effectively "shifts" the source position by wind.

        Parameters:
        -----------
        pos1 : tuple (x, y)
            Sensor position
        pos2 : tuple (x, y)
            Source location

        Returns:
        --------
        distance : float
            Wind-adjusted shortest path distance (meters)
        """
        if self.occupancy_grid is None:
            # Fallback to Euclidean distance with wind adjustment
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]

            # Apply wind: reduce distance downwind, increase upwind
            # Wind vector dot displacement vector gives wind effect
            dx_wind_adjusted = dx - self.wind_x
            dy_wind_adjusted = dy - self.wind_y

            return np.sqrt(dx_wind_adjusted**2 + dy_wind_adjusted**2)

        # Calculate actual distances in x and y
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]

        # Wind-adjusted positions: shift source "with" the wind
        # If wind is positive x, source effectively moves +x, reducing downwind distance
        source_wind_adjusted = (
            pos2[0] + self.wind_x,
            pos2[1] + self.wind_y
        )

        # Convert world coordinates to grid indices
        gx1, gy1 = self.occupancy_grid.world_to_grid(pos1[0], pos1[1])
        gx2, gy2 = self.occupancy_grid.world_to_grid(
            source_wind_adjusted[0],
            source_wind_adjusted[1]
        )

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

        This enables batch computation for particle filter optimization.
        Note: Wind adjustment is applied per-particle in batch computation.

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

        # Check cache first (cache key includes wind for correctness)
        cache_key = (gx, gy, self.wind_x, self.wind_y)
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
        """
        return self.compute_concentration(position, source_location, release_rate)

    def compute_concentrations_batch(self,
                                    sensor_position: Tuple[float, float],
                                    particle_locations: np.ndarray,
                                    release_rates: np.ndarray) -> np.ndarray:
        """
        Efficiently compute concentrations for multiple particles with wind effects.

        Wind adjustment:
        - For each particle, shift its location by wind vector
        - Compute Dijkstra distance map from sensor once
        - Look up distances for all wind-adjusted particle positions

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

        # Wind-adjusted particle locations
        # Shift each source by wind vector
        particle_locations_wind = particle_locations + np.array([self.wind_x, self.wind_y])

        # Convert particle locations to grid coordinates and compute concentrations
        concentrations = np.zeros(len(particle_locations))
        Qm = release_rates

        for i, (x0, y0) in enumerate(particle_locations_wind):
            gx, gy = self.occupancy_grid.world_to_grid(x0, y0)

            if self._is_valid_grid_position(gx, gy):
                cobs = distance_map[gy, gx]
                # Wind-adjusted IGDM equation
                concentrations[i] = Qm[i] * np.exp(-cobs**2 / (2 * self.sigma_m**2))
            else:
                concentrations[i] = 0.0

        return concentrations
