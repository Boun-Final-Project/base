"""
Occupancy Grid for obstacle representation and collision detection.
"""

import numpy as np


class OccupancyGrid:
    """Grid-based occupancy representation for collision checking."""

    def __init__(self, width, height, resolution):
        """
        Parameters:
        -----------
        width : float
            Width of the environment (meters)
        height : float
            Height of the environment (meters)
        resolution : float
            Grid cell resolution (meters per cell)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices.

        Parameters:
        -----------
        x : float
            X coordinate in world frame
        y : float
            Y coordinate in world frame

        Returns:
        --------
        gx, gy : int, int
            Grid cell indices
        """
        gx = int(np.floor(x / self.resolution))
        gy = int(np.floor(y / self.resolution))
        return gx, gy

    def is_valid(self, position=None, radius=0.2, gx=None, gy=None):
        """Check if a position is collision-free.

        Parameters:
        -----------
        position : tuple, optional
            (x, y) world coordinates
        radius : float
            Robot radius for collision checking
        gx, gy : int, optional
            Grid coordinates (used by Dijkstra)

        Returns:
        --------
        valid : bool
            True if position is collision-free
        """
        # Handle grid coordinates directly (from Dijkstra)
        if gx is not None and gy is not None:
            pass  # Use provided grid coordinates
        # Handle world coordinates (tuple)
        elif isinstance(position, tuple):
            gx, gy = self.world_to_grid(position[0], position[1])
        else:
            return False

        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False

        radius_cells = int(np.ceil(radius / self.resolution))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if 0 <= gx + dx < self.grid_width and 0 <= gy + dy < self.grid_height:
                    if self.grid[gy + dy, gx + dx] != 0:
                        return False
        return True
