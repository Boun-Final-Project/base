"""
Indoor Gaussian Dispersion Model (IGDM) implementation.

Time-dependent model where sigma_m grows with time to simulate gas dispersion:
sigma_m(t) = sigma_m0 * sqrt(1 + alpha*t)
"""

import numpy as np
import heapq


class IGDMModel:
    """Indoor Gaussian Dispersion Model (Dijkstra distance-based).

    Time-dependent model where sigma_m grows with time to simulate gas dispersion:
    sigma_m(t) = sigma_m0 * sqrt(1 + alpha*t)
    """

    def __init__(self, sigma_m=1.0, occupancy_grid=None, dispersion_rate=0.05):
        """
        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter (at t=0)
        occupancy_grid : OccupancyGrid, optional
            Grid for obstacle-aware Dijkstra distances
        dispersion_rate : float
            Rate of dispersion growth (alpha). Higher = faster spread over time.
        """
        self.sigma_m_base = sigma_m
        self.occupancy_grid = occupancy_grid
        self._dijkstra_cache = {}
        self.dispersion_rate = dispersion_rate

    def _dijkstra_distance(self, start_pos, target_pos):
        if self.occupancy_grid is None:
            return np.linalg.norm(np.array(target_pos) - np.array(start_pos))

        start_gx, start_gy = self.occupancy_grid.world_to_grid(start_pos[0], start_pos[1])
        cache_key = (start_gx, start_gy)

        if cache_key not in self._dijkstra_cache:
            distances = self._dijkstra_full(start_gx, start_gy)
            self._dijkstra_cache[cache_key] = distances

        distances = self._dijkstra_cache[cache_key]
        target_gx, target_gy = self.occupancy_grid.world_to_grid(target_pos[0], target_pos[1])

        # Use keyword arguments to pass grid coordinates to is_valid
        if not self.occupancy_grid.is_valid(gx=target_gx, gy=target_gy):
            return np.inf

        return distances[target_gy, target_gx]

    def get_dijkstra_distances_from(self, position):
        """Get Dijkstra distances from position to all grid cells (cached).

        Optimization: Compute once and reuse for all particles at this sensor position.

        Parameters:
        -----------
        position : tuple
            (x, y) starting position

        Returns:
        --------
        distances : np.ndarray or None
            Distance grid where distances[gy, gx] is distance to that grid cell.
            Returns None if occupancy_grid is not available.
        """
        if self.occupancy_grid is None:
            return None

        gx, gy = self.occupancy_grid.world_to_grid(position[0], position[1])
        cache_key = (gx, gy)

        if cache_key not in self._dijkstra_cache:
            distances = self._dijkstra_full(gx, gy)
            self._dijkstra_cache[cache_key] = distances

        return self._dijkstra_cache[cache_key]

    def _dijkstra_full(self, start_gx, start_gy):
        grid = self.occupancy_grid
        distances = np.full((grid.grid_height, grid.grid_width), np.inf)
        distances[start_gy, start_gx] = 0.0

        pq = [(0.0, start_gx, start_gy)]
        visited = set()

        while pq:
            current_dist, gx, gy = heapq.heappop(pq)

            if (gx, gy) in visited:
                continue
            visited.add((gx, gy))

            for dx, dy, cost in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                                  (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
                                  (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]:
                nx, ny = gx + dx, gy + dy
                if not grid.is_valid(gx=nx, gy=ny) or (nx, ny) in visited:
                    continue

                edge_cost = cost * grid.resolution
                new_dist = current_dist + edge_cost
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

        return distances

    def get_sigma_m(self, time_step=0):
        """Compute time-dependent dispersion parameter.

        sigma_m(t) = sigma_m0 * sqrt(1 + alpha*t)

        Parameters:
        -----------
        time_step : int or float
            Time step or elapsed time. Higher values increase sigma (more dispersion).

        Returns:
        --------
        sigma : float
            Time-dependent dispersion parameter
        """
        return self.sigma_m_base * np.sqrt(1.0 + self.dispersion_rate * time_step)

    def compute_concentration(self, position, source_location, release_rate, time_step=0, debug=False, dijkstra_grid=None):
        """Compute gas concentration at a position.

        Parameters:
        -----------
        position : tuple
            (x, y) coordinates where concentration is computed (sensor position)
        source_location : tuple
            (x, y) coordinates of the gas source
        release_rate : float
            Release rate Q_m
        time_step : int or float, optional
            Time step (default 0). Higher values increase dispersion.
        debug : bool
            Whether to print debug info
        dijkstra_grid : np.ndarray, optional
            Precomputed Dijkstra distance grid from position. If provided, uses this
            instead of computing Dijkstra distance (optimization for batch computations).

        Returns:
        --------
        concentration : float
            Concentration value at the given position
        """
        if dijkstra_grid is not None:
            # Use precomputed Dijkstra distance grid (optimization path)
            if self.occupancy_grid is None:
                c_obs = np.linalg.norm(np.array(source_location) - np.array(position))
            else:
                source_gx, source_gy = self.occupancy_grid.world_to_grid(source_location[0], source_location[1])
                if not self.occupancy_grid.is_valid(gx=source_gx, gy=source_gy):
                    c_obs = np.inf
                else:
                    c_obs = dijkstra_grid[source_gy, source_gx]
        else:
            # Compute Dijkstra on the fly (original behavior, used when no precomputed grid)
            c_obs = self._dijkstra_distance(source_location, position)

        if debug:
            print(f"    [DIJKSTRA] source={source_location}, pos={position}, dist={c_obs}")
        if np.isinf(c_obs):
            return 0.0
        sigma_m = self.get_sigma_m(time_step)
        exponent = -(c_obs ** 2) / (2 * sigma_m ** 2)
        conc = release_rate * np.exp(exponent)
        if debug:
            print(f"    [CONC] sigma_m={sigma_m}, exponent={exponent}, conc={conc}")
        return conc
