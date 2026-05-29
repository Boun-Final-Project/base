"""
Raycast LiDAR simulation on an OccupancyGrid.
Uses batch grid sampling — all rays checked at discrete steps simultaneously.
"""

import numpy as np


class LidarSim:
    """Simulates a 2D LiDAR sensor by raycasting on an occupancy grid."""

    def __init__(self, num_rays, max_range, occupancy_grid, noise_std=0.0):
        """
        Parameters
        ----------
        num_rays : int
            Number of evenly spaced rays (e.g. 24 -> 15 deg apart).
        max_range : float
            Maximum sensing distance in meters.
        occupancy_grid : OccupancyGrid
            The grid to raycast against.
        noise_std : float, optional
            Standard deviation of Gaussian noise applied to hit rays (meters).
            Default 0.0 (no noise).
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.grid = occupancy_grid
        self.noise_std = noise_std
        # Clamp noise-perturbed distances to at least one grid resolution (minimum sensing distance)
        self._min_range = occupancy_grid.resolution
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        res = occupancy_grid.resolution
        # Sample points along each ray at grid resolution
        n_steps = int(np.ceil(max_range / res))
        # distances from origin: res, 2*res, ..., n_steps*res
        t = np.arange(1, n_steps + 1) * res  # (S,)

        self._t = t  # (S,)
        self._n_steps = n_steps

    def scan(self, position, heading=0.0):
        """Return normalized LiDAR distances from a world-frame position.

        Parameters
        ----------
        position : tuple
            (x, y) in meters.
        heading : float, optional
            Robot heading in radians. Ray angles are rotated by this amount.
            Default 0.0.

        Returns
        -------
        distances : np.ndarray
            Shape (num_rays,), each in [0, 1] (distance / max_range).
        """
        res = self.grid.resolution
        gw = self.grid.grid_width
        gh = self.grid.grid_height
        ox, oy = position

        # Compute ray directions rotated by heading
        rotated_angles = self.ray_angles + heading  # (R,)
        cos_a = np.cos(rotated_angles)  # (R,)
        sin_a = np.sin(rotated_angles)  # (R,)

        # Offsets for each (ray, step): shape (R, S)
        dx = cos_a[:, None] * self._t[None, :]  # (R, S)
        dy = sin_a[:, None] * self._t[None, :]  # (R, S)

        # World coords of all sample points: (R, S)
        wx = ox + dx
        wy = oy + dy

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

        # Coarse distance: t[first_hit_idx] for rays that hit, else max_range.
        # This is quantized to the grid step (~resolution), which floors the
        # measured wall distance at ~one cell. Real BasicSim casts against
        # continuous geometry and reads sub-cell clearances (e.g. 0.24 m in
        # tight curves) far more often (45% of steps < 0.4 m vs 11% here).
        # Refine each hit to a continuous surface distance below so the
        # near-wall regime matches deployment — otherwise the policy never
        # experiences sub-resolution clearance in training.
        coarse = np.where(any_hit, self._t[first_hit_idx], self.max_range)

        # Sub-cell refinement: the wall surface lies between the last free
        # sample (t = coarse - res) and the first occupied sample (t = coarse).
        # Bisect within that interval to locate the free→occupied transition at
        # sub-resolution precision. Vectorised over hit rays.
        distances = coarse.copy()
        hit_rays = np.where(any_hit)[0]
        if hit_rays.size:
            lo = np.maximum(coarse[hit_rays] - res, 0.0)   # last known-free t
            hi = coarse[hit_rays].copy()                   # first known-occupied t
            ca = cos_a[hit_rays]; sa = sin_a[hit_rays]
            for _ in range(6):  # 2^-6 * res ≈ 1.5 mm precision at res=0.1
                mid = 0.5 * (lo + hi)
                mgx = np.floor((ox + ca * mid) / res).astype(np.int32)
                mgy = np.floor((oy + sa * mid) / res).astype(np.int32)
                m_oob = (mgx < 0) | (mgx >= gw) | (mgy < 0) | (mgy >= gh)
                m_occ = m_oob | (self.grid.grid[np.clip(mgy, 0, gh - 1),
                                                np.clip(mgx, 0, gw - 1)] != 0)
                hi = np.where(m_occ, mid, hi)   # midpoint occupied → surface closer
                lo = np.where(m_occ, lo, mid)
            distances[hit_rays] = hi           # first-occupied boundary = surface distance

        # Apply Gaussian noise to hit rays only
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=distances.shape)
            distances = np.where(
                any_hit,
                np.clip(distances + noise, self._min_range, self.max_range),
                self.max_range,
            )

        return distances / self.max_range
