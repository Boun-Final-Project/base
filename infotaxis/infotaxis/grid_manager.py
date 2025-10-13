#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional
import rclpy
from rclpy.node import Node
from gaden_msgs.srv import Occupancy


class GridManager:
    """Manages 2D occupancy grids and coordinate transformations for infotaxis."""

    def __init__(self, node: Node, z_level: int):
        """
        Initialize the grid manager.

        Args:
            node: ROS2 node for logging and service calls
            z_level: Z-level for 2D slice (fixed, sensor height)
        """
        self.node = node
        self.z_level = z_level

        # Grid parameters (uses GADEN's resolution directly)
        self.grid_origin: Optional[np.ndarray] = None  # [x, y, z]
        self.cell_size: Optional[float] = None
        self.grid_shape: Optional[Tuple[int, int]] = None  # (nx, ny)
        self.z_height: Optional[float] = None  # Fixed z height in world coords

        # Grid data
        self.occupancy_grid: Optional[np.ndarray] = None
        self.source_log_probabilities: Optional[np.ndarray] = None  # Log probabilities
        self.num_free_cells: int = 0

    def initialize(self, occupancy_client) -> bool:
        """
        Initialize 2D grid by fetching occupancy data at fixed z-level.

        Args:
            occupancy_client: Service client for occupancy grid

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not self._fetch_2d_occupancy_grid(occupancy_client):
            return False

        self._initialize_uniform_log_distribution()
        self._log_initialization_info()
        return True

    def _fetch_2d_occupancy_grid(self, occupancy_client) -> bool:
        """Fetch 2D occupancy grid at fixed z-level from gaden_environment service."""
        request = Occupancy.Request()
        future = occupancy_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is None:
            self.node.get_logger().error('Failed to get occupancy grid from service')
            return False

        response = future.result()

        # Store grid parameters
        self.grid_origin = np.array([
            response.origin.x,
            response.origin.y,
            response.origin.z
        ])
        self.cell_size = response.resolution
        num_cells_x = response.num_cells_x
        num_cells_y = response.num_cells_y
        num_cells_z = response.num_cells_z

        self.grid_shape = (num_cells_x, num_cells_y)

        # Clamp z_level to valid range
        self.z_level = np.clip(self.z_level, 0, num_cells_z - 1)

        # Calculate fixed z height in world coordinates
        self.z_height = self.grid_origin[2] + (self.z_level + 0.5) * self.cell_size

        # Convert occupancy array to 3D grid temporarily to extract 2D slice
        # Values: 0=free space, 1=obstacle, 2=outlet
        occupancy_array = np.array(response.occupancy, dtype=np.uint8)
        occupancy_grid_3d = occupancy_array.reshape(num_cells_z, num_cells_y, num_cells_x)

        # Extract 2D slice at z-level
        self.occupancy_grid = occupancy_grid_3d[self.z_level, :, :]

        self.node.get_logger().info(
            f'Loaded 2D grid at z-level {self.z_level} (z={self.z_height:.2f}m)'
        )
        return True

    def _initialize_uniform_log_distribution(self):
        """Initialize uniform log-probability distribution over free cells."""
        # Count free cells (value = 0)
        self.num_free_cells = np.sum(self.occupancy_grid == 0)

        if self.num_free_cells == 0:
            self.node.get_logger().error('No free cells in infotaxis grid!')
            return

        # Initialize log probability distribution
        # Use -inf for occupied cells
        self.source_log_probabilities = np.full_like(
            self.occupancy_grid, -np.inf, dtype=np.float64
        )

        # Set uniform log probability for free cells: log(1/N) = -log(N)
        uniform_log_prob = -np.log(self.num_free_cells)
        self.source_log_probabilities[self.occupancy_grid == 0] = uniform_log_prob

    def get_cell_coordinates(self, i: int, j: int) -> np.ndarray:
        """
        Convert grid indices to world coordinates (cell center).

        Args:
            i: Grid x-index
            j: Grid y-index

        Returns:
            World coordinates [x, y, z] with fixed z height
        """
        x = self.grid_origin[0] + (i + 0.5) * self.cell_size
        y = self.grid_origin[1] + (j + 0.5) * self.cell_size
        return np.array([x, y, self.z_height])

    def get_cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices.

        Args:
            x: World x-coordinate
            y: World y-coordinate

        Returns:
            Grid indices (i, j)
        """
        i = int((x - self.grid_origin[0]) / self.cell_size)
        j = int((y - self.grid_origin[1]) / self.cell_size)

        # Clamp to grid bounds
        i = np.clip(i, 0, self.grid_shape[0] - 1)
        j = np.clip(j, 0, self.grid_shape[1] - 1)

        return i, j

    def is_cell_valid(self, i: int, j: int) -> bool:
        """
        Check if a grid cell is within bounds.

        Args:
            i: Grid x-index
            j: Grid y-index

        Returns:
            True if cell is within grid bounds
        """
        return 0 <= i < self.grid_shape[0] and 0 <= j < self.grid_shape[1]

    def is_cell_free(self, i: int, j: int) -> bool:
        """
        Check if a grid cell is free (not occupied).

        Args:
            i: Grid x-index
            j: Grid y-index

        Returns:
            True if cell is free
        """
        if not self.is_cell_valid(i, j):
            return False
        return self.occupancy_grid[j, i] == 0
    
    def get_legal_moves(self, pos : Tuple[int, int]) -> list[Tuple[int,int]]:
        """
        Get list of legal moves (to free neighboring cells) from current position.

        Args:
            pos: Current grid position (ix, iy)
        Returns:
            List of legal neighboring positions [(ix1, iy1), (ix2, iy2), ...]
        """
        legal_moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            if self.is_cell_free(new_x, new_y):
                legal_moves.append((new_x, new_y))

        return legal_moves
    
    def get_log_probabilities(self) -> np.ndarray:
        """Return log probabilities with shape (nx, ny) for infotaxis calculations."""
        # Transpose to match (nx, ny) convention expected by infotaxis functions
        return self.source_log_probabilities.T

    def set_log_probabilities(self, log_probs: np.ndarray):
        """Set log probabilities. Input should be shape (nx, ny), stored as (ny, nx)."""
        # Transpose back to (ny, nx) for storage
        self.source_log_probabilities = log_probs.T

    def get_xs(self) -> np.ndarray:
        """Return array of x-coordinates for each grid cell center."""
        return np.arange(self.grid_shape[0]) * self.cell_size + self.grid_origin[0] + 0.5 * self.cell_size

    def get_ys(self) -> np.ndarray:
        """Return array of y-coordinates for each grid cell center."""
        return np.arange(self.grid_shape[1]) * self.cell_size + self.grid_origin[1] + 0.5 * self.cell_size

    def _log_initialization_info(self):
        """Log grid initialization information."""
        self.node.get_logger().info(f'Z-level: {self.z_level} (height: {self.z_height:.2f}m)')
        self.node.get_logger().info(f'Cell size: {self.cell_size}m')
        self.node.get_logger().info(f'Grid dimensions: {self.grid_shape}')
        self.node.get_logger().info(f'Grid origin: {self.grid_origin}')
        self.node.get_logger().info(f'Number of free cells: {self.num_free_cells}')
        if self.num_free_cells > 0:
            uniform_log_prob = self.source_log_probabilities[self.occupancy_grid == 0][0]
            self.node.get_logger().info(
                f'Uniform log-probability per cell: {uniform_log_prob:.6f}'
            )