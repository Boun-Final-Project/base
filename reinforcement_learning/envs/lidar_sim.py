"""
Raycast LiDAR simulation on an OccupancyGrid.
Uses batch grid sampling — all rays checked at discrete steps simultaneously.
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

        res = occupancy_grid.resolution
        # Sample points along each ray at grid resolution
        n_steps = int(np.ceil(max_range / res))
        # distances from origin: res, 2*res, ..., n_steps*res
        t = np.arange(1, n_steps + 1) * res  # (S,)

        cos_a = np.cos(self.ray_angles)  # (R,)
        sin_a = np.sin(self.ray_angles)  # (R,)

        # Offsets for each (ray, step): shape (R, S)
        self._dx = cos_a[:, None] * t[None, :]  # (R, S)
        self._dy = sin_a[:, None] * t[None, :]  # (R, S)
        self._t = t  # (S,)
        self._n_steps = n_steps

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
        res = self.grid.resolution
        gw = self.grid.grid_width
        gh = self.grid.grid_height
        ox, oy = position

        # World coords of all sample points: (R, S)
        wx = ox + self._dx
        wy = oy + self._dy

        # Convert to grid indices
        gx = np.floor(wx / res).astype(np.int32)
        gy = np.floor(wy / res).astype(np.int32)

        # Out of bounds → treat as hit
        oob = (gx < 0) | (gx >= gw) | (gy < 0) | (gy >= gh)

        # Clamp for safe indexing
        gx_safe = np.clip(gx, 0, gw - 1)
        gy_safe = np.clip(gy, 0, gh - 1)

        # Batch lookup: occupied? (R, S)
        occupied = self.grid.grid[gy_safe, gx_safe] != 0

        # A sample is a "hit" if occupied or out of bounds
        hit = occupied | oob  # (R, S)

        # For each ray, find the first hit step
        # argmax on bool returns index of first True; if no True, returns 0
        any_hit = np.any(hit, axis=1)  # (R,)
        first_hit_idx = np.argmax(hit, axis=1)  # (R,)

        # Distance: t[first_hit_idx] for rays that hit, else max_range
        distances = np.where(any_hit, self._t[first_hit_idx], self.max_range)

        return distances / self.max_range
