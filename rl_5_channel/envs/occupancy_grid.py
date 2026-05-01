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

    def add_rectangular_obstacle(self, x_min, x_max, y_min, y_max, value=1):
        """Add a rectangular obstacle to the grid.

        Parameters:
        -----------
        x_min : float
            Minimum x coordinate of obstacle (world frame)
        x_max : float
            Maximum x coordinate of obstacle (world frame)
        y_min : float
            Minimum y coordinate of obstacle (world frame)
        y_max : float
            Maximum y coordinate of obstacle (world frame)
        value : int
            Grid value for the obstacle (default 1 = occupied)
        """
        gx_min, gy_min = self.world_to_grid(x_min, y_min)
        gx_max, gy_max = self.world_to_grid(x_max, y_max)

        # Ensure valid ranges
        gx_min = max(0, min(gx_min, self.grid_width - 1))
        gx_max = max(0, min(gx_max, self.grid_width - 1))
        gy_min = max(0, min(gy_min, self.grid_height - 1))
        gy_max = max(0, min(gy_max, self.grid_height - 1))

        # Fill the obstacle region
        for gx in range(min(gx_min, gx_max), max(gx_min, gx_max) + 1):
            for gy in range(min(gy_min, gy_max), max(gy_min, gy_max) + 1):
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    self.grid[gy, gx] = value
