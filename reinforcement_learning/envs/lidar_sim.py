"""
Raycast LiDAR simulation on an OccupancyGrid.
Uses DDA (Digital Differential Analyzer) for fast grid traversal.
"""

import numpy as np


class LidarSim:
    """Simulates a 2D LiDAR sensor by raycasting on an occupancy grid."""

    def __init__(self, num_rays, max_range, occupancy_grid):
        """
        Parameters
        ----------
        num_rays : int
            Number of evenly spaced rays (e.g. 24 -> 15 deg apart).
        max_range : float
            Maximum sensing distance in meters.
        occupancy_grid : OccupancyGrid
            The grid to raycast against.
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.grid = occupancy_grid
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    def scan(self, position):
        """Return normalized LiDAR distances from a world-frame position.

        Parameters
        ----------
        position : tuple
            (x, y) in meters.

        Returns
        -------
        distances : np.ndarray
            Shape (num_rays,), each in [0, 1] (distance / max_range).
        """
        distances = np.empty(self.num_rays, dtype=np.float64)
        for i, angle in enumerate(self.ray_angles):
            distances[i] = self._cast_ray(position, angle)
        # Normalize to [0, 1] so all state features share a similar scale,
        # which stabilizes neural network training (position, wind are also [0, 1]).
        return distances / self.max_range

    def _cast_ray(self, origin, angle):
        """Cast a single ray using DDA and return distance to first obstacle.

        Parameters
        ----------
        origin : tuple
            (x, y) world position of the ray origin.
        angle : float
            Ray direction in radians.

        Returns
        -------
        distance : float
            Distance in meters to the nearest obstacle along this ray,
            capped at max_range.
        """
        res = self.grid.resolution
        ox, oy = origin

        dx = np.cos(angle)
        dy = np.sin(angle)

        # Current grid cell
        gx = int(np.floor(ox / res))
        gy = int(np.floor(oy / res))

        # Step direction (+1 or -1) and distance to next cell boundary per axis
        if abs(dx) < 1e-12:
            step_x = 0
            t_max_x = np.inf
            t_delta_x = np.inf
        else:
            step_x = 1 if dx > 0 else -1
            # Distance along ray to the next vertical grid boundary
            if dx > 0:
                t_max_x = ((gx + 1) * res - ox) / dx
            else:
                t_max_x = (gx * res - ox) / dx
            t_delta_x = abs(res / dx)

        if abs(dy) < 1e-12:
            step_y = 0
            t_max_y = np.inf
            t_delta_y = np.inf
        else:
            step_y = 1 if dy > 0 else -1
            if dy > 0:
                t_max_y = ((gy + 1) * res - oy) / dy
            else:
                t_max_y = (gy * res - oy) / dy
            t_delta_y = abs(res / dy)

        max_range_sq = self.max_range ** 2
        w = self.grid.grid_width
        h = self.grid.grid_height

        while True:
            # Advance to next cell boundary
            if t_max_x < t_max_y:
                t = t_max_x
                gx += step_x
                t_max_x += t_delta_x
            else:
                t = t_max_y
                gy += step_y
                t_max_y += t_delta_y

            # Check range
            if t * t > max_range_sq:
                return self.max_range

            # Check bounds
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                return min(t, self.max_range)

            # Check obstacle
            if self.grid.grid[gy, gx] != 0:
                return t
