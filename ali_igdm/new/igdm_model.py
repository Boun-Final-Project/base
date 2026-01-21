"""
Indoor Gaussian Dispersion Model (IGDM) implementation.
UPDATED: Dual-Grid System for Performance Optimization.
- Uses fine grid (0.1m) for compatibility.
- Uses coarse grid (0.5m) for Dijkstra pathfinding (25x faster).

Time-dependent model where sigma_m grows with time to simulate gas dispersion:
sigma_m(t) = sigma_m0 * sqrt(1 + alpha*t)
"""

import numpy as np
import heapq


class IGDMModel:
    """Indoor Gaussian Dispersion Model with Coarse-Grid Pathfinding Optimization."""

    def __init__(self, sigma_m=1.0, occupancy_grid=None, dispersion_rate=0.05, coarse_resolution=0.5):
        """
        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter (at t=0)
        occupancy_grid : OccupancyGrid, optional
            High-res grid (e.g., 0.1m) used for the main map.
        dispersion_rate : float
            Rate of dispersion growth (alpha). Higher = faster spread over time.
        coarse_resolution : float
            Target resolution for physics calculations (e.g., 0.5m).
        """
        self.sigma_m_base = sigma_m
        self.occupancy_grid = occupancy_grid
        self.dispersion_rate = dispersion_rate
        
        # Cache for Dijkstra maps on the coarse grid
        self._dijkstra_cache = {}

        if self.occupancy_grid:
            # 1. Store High-Res metadata for coordinate conversion
            self.high_res = occupancy_grid.resolution
            self.width_m = occupancy_grid.width
            self.height_m = occupancy_grid.height
            
            # 2. Create the Coarse Grid
            self.coarse_res = coarse_resolution
            self.coarse_cols = int(np.ceil(self.width_m / self.coarse_res))
            self.coarse_rows = int(np.ceil(self.height_m / self.coarse_res))
            
            # Downsample the high-res grid to the coarse grid
            self.coarse_grid = self._create_coarse_map(occupancy_grid.grid)
            print(f"[IGDM] Physics Grid Downsampled: {self.coarse_cols}x{self.coarse_rows} cells (Res: {self.coarse_res}m)")
        else:
            self.coarse_grid = None

    def _create_coarse_map(self, fine_grid):
        """
        Downsamples the fine grid to a coarse grid using max pooling.
        If any fine cell in a coarse block is an obstacle, the coarse block is an obstacle.
        """
        ratio = self.coarse_res / self.high_res
        coarse = np.zeros((self.coarse_rows, self.coarse_cols), dtype=int)
        
        for r in range(self.coarse_rows):
            for c in range(self.coarse_cols):
                # Calculate bounds in fine grid (indices)
                r_start = int(r * ratio)
                r_end = int((r + 1) * ratio)
                c_start = int(c * ratio)
                c_end = int((c + 1) * ratio)
                
                # Extract chunk and check for ANY obstacle (value 1)
                # Ensure we don't go out of bounds
                chunk = fine_grid[r_start:r_end, c_start:c_end]
                if np.any(chunk == 1): 
                    coarse[r, c] = 1
                    
        return coarse

    def _world_to_coarse_idx(self, x, y):
        """Convert real-world (meters) to coarse grid indices."""
        c = int(x / self.coarse_res)
        r = int(y / self.coarse_res)
        # Clamp to bounds
        c = max(0, min(c, self.coarse_cols - 1))
        r = max(0, min(r, self.coarse_rows - 1))
        return r, c

    def _dijkstra_full_coarse(self, start_idx):
        """
        Run full Dijkstra on the coarse grid starting from start_idx=(row, col).
        Returns a 2D array of distances (in meters).
        """
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

            # 8-connected neighbors (diagonal movement allowed)
            for dr, dc, cost_mult in [
                (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
            ]:
                nr, nc = r + dr, c + dc
                
                # Boundary check
                if 0 <= nr < self.coarse_rows and 0 <= nc < self.coarse_cols:
                    # Obstacle check (on coarse grid)
                    if self.coarse_grid[nr, nc] == 0:
                        edge_cost = cost_mult * self.coarse_res
                        new_dist = current_dist + edge_cost
                        
                        if new_dist < distances[nr, nc]:
                            distances[nr, nc] = new_dist
                            heapq.heappush(pq, (new_dist, nr, nc))
                            
        return distances

    def get_dijkstra_distances_from(self, position):
        """
        Get Dijkstra distances from position to ALL coarse grid cells.
        Returns the coarse distance grid (meters).
        """
        if self.coarse_grid is None:
            return None

        # Calculate cache key based on coarse indices
        start_r, start_c = self._world_to_coarse_idx(position[0], position[1])
        cache_key = (start_r, start_c)

        if cache_key not in self._dijkstra_cache:
            # Compute full map if not cached
            distances = self._dijkstra_full_coarse((start_r, start_c))
            self._dijkstra_cache[cache_key] = distances

        return self._dijkstra_cache[cache_key]

    def get_sigma_m(self, time_step=0):
        """Compute time-dependent dispersion parameter."""
        return self.sigma_m_base * np.sqrt(1.0 + self.dispersion_rate * time_step)

    def compute_concentration(self, position, source_location, release_rate, time_step=0, debug=False, dijkstra_grid=None):
        """
        Compute gas concentration using Coarse-Grid Dijkstra distance.
        """
        # 1. Calculate Distance
        c_obs = np.inf
        
        if self.coarse_grid is None:
            # Fallback if no grid provided
            c_obs = np.linalg.norm(np.array(source_location) - np.array(position))
        else:
            # Use Coarse Grid Logic
            if dijkstra_grid is not None:
                # Optimized: Use pre-computed grid passed from particle filter
                # We need to look up the SOURCE position in the pre-computed grid (which is centered on 'position')
                # OR if the grid is centered on 'source', look up 'position'.
                # NOTE: Typically dijkstra_grid is centered on the SENSOR (position).
                # So we look up the distance to the SOURCE.
                
                src_r, src_c = self._world_to_coarse_idx(source_location[0], source_location[1])
                
                # Check validity of source pos on coarse grid
                if self.coarse_grid[src_r, src_c] == 1:
                    c_obs = np.inf # Source inside wall
                else:
                    c_obs = dijkstra_grid[src_r, src_c]
            else:
                # Compute on the fly (Slow path)
                d_grid = self.get_dijkstra_distances_from(position)
                src_r, src_c = self._world_to_coarse_idx(source_location[0], source_location[1])
                if self.coarse_grid[src_r, src_c] == 1:
                    c_obs = np.inf
                else:
                    c_obs = d_grid[src_r, src_c]

        # 2. Physics Calculation
        if np.isinf(c_obs):
            return 0.0
            
        # Avoid division by zero
        if c_obs < 0.1: 
            c_obs = 0.1
            
        sigma_m = self.get_sigma_m(time_step)
        exponent = -(c_obs ** 2) / (2 * sigma_m ** 2)
        conc = release_rate * np.exp(exponent)
        
        if debug:
            print(f"    [IGDM] dist={c_obs:.2f}m, sigma={sigma_m:.2f}, conc={conc:.6f}")
            
        return conc