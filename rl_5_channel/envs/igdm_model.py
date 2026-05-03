"""
Indoor Gaussian Dispersion Model (IGDM) with wind bias.

Extends the original IGDM from rrt_infotaxis/igdm_improved:
- Dual-grid system (fine 0.1m grid + coarse 0.5m Dijkstra grid)
- Time-dependent dispersion: sigma_m(t) = sigma_m0 * sqrt(1 + alpha*t)
- Wind bias: shifts effective source position downwind so that
  concentration is higher in the downwind direction.
"""

import numpy as np
import heapq


class IGDMModel:
    """Indoor Gaussian Dispersion Model with coarse-grid Dijkstra and wind support."""

    def __init__(self, sigma_m=1.0, occupancy_grid=None, dispersion_rate=0.05,
                 coarse_resolution=0.5):
        """
        Parameters
        ----------
        sigma_m : float
            Base dispersion parameter (at t=0).
        occupancy_grid : OccupancyGrid, optional
            High-res grid (e.g. 0.1 m).
        dispersion_rate : float
            Rate of dispersion growth (alpha).
        coarse_resolution : float
            Resolution for Dijkstra pathfinding grid.
        """
        self.sigma_m_base = sigma_m
        self.occupancy_grid = occupancy_grid
        self.dispersion_rate = dispersion_rate
        self._dijkstra_cache = {}

        if self.occupancy_grid:
            self.high_res = occupancy_grid.resolution
            self.width_m = occupancy_grid.width
            self.height_m = occupancy_grid.height

            self.coarse_res = coarse_resolution
            self.coarse_cols = int(np.ceil(self.width_m / self.coarse_res))
            self.coarse_rows = int(np.ceil(self.height_m / self.coarse_res))

            self.coarse_grid = self._create_coarse_map(occupancy_grid.grid)
        else:
            self.coarse_grid = None

    def _create_coarse_map(self, fine_grid):
        """Downsample fine grid to coarse grid.

        A coarse cell is marked occupied if ANY of its fine cells is a wall.
        This guarantees thin (1-fine-cell) walls still block gas dispersion
        on the coarse grid.
        """
        ratio = self.coarse_res / self.high_res
        coarse = np.zeros((self.coarse_rows, self.coarse_cols), dtype=int)

        for r in range(self.coarse_rows):
            for c in range(self.coarse_cols):
                r_start = int(r * ratio)
                r_end = int((r + 1) * ratio)
                c_start = int(c * ratio)
                c_end = int((c + 1) * ratio)

                chunk = fine_grid[r_start:r_end, c_start:c_end]
                if chunk.size > 0 and np.any(chunk > 0):
                    coarse[r, c] = 1

        return coarse

    def _world_to_coarse_idx(self, x, y):
        """Convert world coordinates (meters) to coarse grid indices."""
        c = int(x / self.coarse_res)
        r = int(y / self.coarse_res)
        c = max(0, min(c, self.coarse_cols - 1))
        r = max(0, min(r, self.coarse_rows - 1))
        return r, c

    def snap_to_free_cell(self, x, y):
        """Snap world position to nearest free cell on the coarse grid.

        If (x, y) already maps to a free cell, returns it unchanged.
        Otherwise searches outward (BFS) for the nearest free cell and
        returns its center in world coordinates.
        """
        if self.coarse_grid is None:
            return x, y

        r, c = self._world_to_coarse_idx(x, y)
        if self.coarse_grid[r, c] == 0:
            return x, y

        # BFS outward from (r, c) to find nearest free cell
        from collections import deque
        visited = {(r, c)}
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in visited:
                    continue
                if 0 <= nr < self.coarse_rows and 0 <= nc < self.coarse_cols:
                    visited.add((nr, nc))
                    if self.coarse_grid[nr, nc] == 0:
                        # Return center of this free cell
                        return (nc + 0.5) * self.coarse_res, (nr + 0.5) * self.coarse_res
                    queue.append((nr, nc))

        # Fallback (shouldn't happen — no free cells at all)
        return x, y

    def _dijkstra_full_coarse(self, start_idx):
        """Full Dijkstra on the coarse grid, returns 2D distance array in meters."""
        start_r, start_c = start_idx
        distances = np.full((self.coarse_rows, self.coarse_cols), np.inf)
        distances[start_r, start_c] = 0.0

        pq = [(0.0, start_r, start_c)]
        visited = set()

        while pq:
            current_dist, r, c = heapq.heappop(pq)

            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc, cost_mult in [
                (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
            ]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.coarse_rows and 0 <= nc < self.coarse_cols:
                    if self.coarse_grid[nr, nc] == 0:
                        edge_cost = cost_mult * self.coarse_res
                        new_dist = current_dist + edge_cost
                        if new_dist < distances[nr, nc]:
                            distances[nr, nc] = new_dist
                            heapq.heappush(pq, (new_dist, nr, nc))

        return distances

    def get_dijkstra_distances_from(self, position):
        """Get cached Dijkstra distances from position to all coarse cells."""
        if self.coarse_grid is None:
            return None

        start_r, start_c = self._world_to_coarse_idx(position[0], position[1])
        cache_key = (start_r, start_c)

        if cache_key not in self._dijkstra_cache:
            self._dijkstra_cache[cache_key] = self._dijkstra_full_coarse((start_r, start_c))

        return self._dijkstra_cache[cache_key]

    def clear_cache(self):
        """Clear the Dijkstra distance cache (call on episode reset)."""
        self._dijkstra_cache.clear()

    def get_sigma_m(self, time_step=0):
        """Compute time-dependent dispersion parameter."""
        return self.sigma_m_base * np.sqrt(1.0 + self.dispersion_rate * time_step)

    def compute_concentration(self, position, source_location, release_rate,
                              time_step=0, wind_offset=None, dijkstra_grid=None):
        """Compute gas concentration at position from a source.

        Parameters
        ----------
        position : tuple
            (x, y) sensor position in meters.
        source_location : tuple
            (x, y) true source position in meters.
        release_rate : float
            Source emission rate Q.
        time_step : int
            Current time step for time-dependent dispersion.
        wind_offset : np.ndarray, optional
            (dx, dy) downwind offset from WindModel.get_dispersion_offset().
            Shifts the effective source position so concentration is
            higher downwind: effective_source = source + wind_offset.
        dijkstra_grid : np.ndarray, optional
            Pre-computed Dijkstra distance grid centered on position.

        Returns
        -------
        concentration : float
            Predicted gas concentration (>= 0).
        """
        # Apply wind bias: shift effective source downwind
        if wind_offset is not None:
            eff_x = max(0, min(source_location[0] + wind_offset[0],
                               self.width_m - self.coarse_res))
            eff_y = max(0, min(source_location[1] + wind_offset[1],
                               self.height_m - self.coarse_res))
            # Snap to nearest free cell if wind pushed source into a wall
            eff_source = self.snap_to_free_cell(eff_x, eff_y)
        else:
            eff_source = source_location

        # Compute obstacle-aware distance
        c_obs = np.inf

        if self.coarse_grid is None:
            c_obs = np.linalg.norm(np.array(eff_source) - np.array(position))
        else:
            if dijkstra_grid is not None:
                src_r, src_c = self._world_to_coarse_idx(eff_source[0], eff_source[1])
                if self.coarse_grid[src_r, src_c] == 1:
                    c_obs = np.inf
                else:
                    c_obs = dijkstra_grid[src_r, src_c]
            else:
                d_grid = self.get_dijkstra_distances_from(position)
                src_r, src_c = self._world_to_coarse_idx(eff_source[0], eff_source[1])
                if self.coarse_grid[src_r, src_c] == 1:
                    c_obs = np.inf
                else:
                    c_obs = d_grid[src_r, src_c]

        if np.isinf(c_obs):
            return 0.0

        if c_obs < 0.1:
            c_obs = 0.1

        sigma_m = self.get_sigma_m(time_step)
        exponent = -(c_obs ** 2) / (2 * sigma_m ** 2)
        return release_rate * np.exp(exponent)
