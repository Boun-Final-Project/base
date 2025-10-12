#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional
import rclpy
from rclpy.node import Node
from gaden_msgs.srv import Occupancy


class GridManager:
    """Manages occupancy grids and coordinate transformations for infotaxis."""

    def __init__(self, node: Node, z_level: int, step_size: float):
        """
        Initialize the grid manager.

        Args:
            node: ROS2 node for logging and service calls
            z_level: Z-level for 2D slice (sensor height)
            step_size: Step size for movement in meters
        """
        self.node = node
        self.z_level = z_level
        self.step_size = step_size

        # GADEN environment parameters
        self.env_origin: Optional[np.ndarray] = None
        self.gaden_cell_size: Optional[float] = None
        self.num_cells_3d: Optional[np.ndarray] = None
        self.occupancy_grid_3d: Optional[np.ndarray] = None
        self.occupancy_grid_2d: Optional[np.ndarray] = None

        # Infotaxis grid parameters
        self.grid_shape: Optional[Tuple[int, int]] = None
        self.grid_origin: Optional[np.ndarray] = None
        self.occupancy_grid: Optional[np.ndarray] = None
        self.probability_dist: Optional[np.ndarray] = None
        self.num_free_cells: int = 0

    def initialize(self, occupancy_client) -> bool:
        """
        Initialize all grids by fetching occupancy data and processing it.

        Args:
            occupancy_client: Service client for occupancy grid

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not self._fetch_occupancy_grid(occupancy_client):
            return False

        self._create_2d_slice()
        self._create_infotaxis_grid()
        self._initialize_uniform_distribution()

        self._log_initialization_info()
        return True

    def _fetch_occupancy_grid(self, occupancy_client) -> bool:
        """Fetch occupancy grid from gaden_environment service."""
        request = Occupancy.Request()
        future = occupancy_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is None:
            self.node.get_logger().error('Failed to get occupancy grid from service')
            return False

        response = future.result()

        # Store GADEN grid parameters
        self.env_origin = np.array([
            response.origin.x,
            response.origin.y,
            response.origin.z
        ])
        self.gaden_cell_size = response.resolution
        self.num_cells_3d = np.array([
            response.num_cells_x,
            response.num_cells_y,
            response.num_cells_z
        ])

        # Convert occupancy array to 3D grid
        # Values: 0=free space, 1=obstacle, 2=outlet
        occupancy_array = np.array(response.occupancy, dtype=np.uint8)

        # Reshape to 3D grid (z, y, x)
        self.occupancy_grid_3d = occupancy_array.reshape(
            self.num_cells_3d[2],  # z
            self.num_cells_3d[1],  # y
            self.num_cells_3d[0]   # x
        )

        self.node.get_logger().info('Received GADEN occupancy grid from service')
        return True

    def _create_2d_slice(self):
        """Extract 2D slice from 3D occupancy grid at specified z-level."""
        if self.occupancy_grid_3d is None:
            self.node.get_logger().error('Cannot create 2D slice: No 3D occupancy grid available')
            return

        # Clamp z_level to valid range
        self.z_level = np.clip(self.z_level, 0, self.num_cells_3d[2] - 1)

        # Extract 2D slice (y, x)
        self.occupancy_grid_2d = self.occupancy_grid_3d[self.z_level, :, :]

        self.node.get_logger().info(f'Created 2D slice at z-level {self.z_level}')

    def _create_infotaxis_grid(self):
        """Create coarser infotaxis grid based on step_size parameter."""
        # Calculate environment bounds
        env_max_x = self.env_origin[0] + self.gaden_cell_size * self.num_cells_3d[0]
        env_max_y = self.env_origin[1] + self.gaden_cell_size * self.num_cells_3d[1]

        # Calculate grid dimensions based on step_size
        grid_nx = int(np.ceil((env_max_x - self.env_origin[0]) / self.step_size))
        grid_ny = int(np.ceil((env_max_y - self.env_origin[1]) / self.step_size))

        self.grid_shape = (grid_nx, grid_ny)
        self.grid_origin = self.env_origin.copy()

        # Create occupancy grid for infotaxis (using step_size cells)
        self.occupancy_grid = np.zeros((grid_ny, grid_nx), dtype=np.uint8)

        # Mark cells as occupied if any of the underlying GADEN cells are occupied
        for j in range(grid_ny):
            for i in range(grid_nx):
                # Get world coordinates for this infotaxis cell
                x_min = self.grid_origin[0] + i * self.step_size
                y_min = self.grid_origin[1] + j * self.step_size
                x_max = x_min + self.step_size
                y_max = y_min + self.step_size

                # Convert to GADEN grid indices
                gaden_i_min = int((x_min - self.env_origin[0]) / self.gaden_cell_size)
                gaden_j_min = int((y_min - self.env_origin[1]) / self.gaden_cell_size)
                gaden_i_max = int((x_max - self.env_origin[0]) / self.gaden_cell_size)
                gaden_j_max = int((y_max - self.env_origin[1]) / self.gaden_cell_size)

                # Clamp to valid range
                gaden_i_min = np.clip(gaden_i_min, 0, self.num_cells_3d[0] - 1)
                gaden_j_min = np.clip(gaden_j_min, 0, self.num_cells_3d[1] - 1)
                gaden_i_max = np.clip(gaden_i_max, 0, self.num_cells_3d[0])
                gaden_j_max = np.clip(gaden_j_max, 0, self.num_cells_3d[1])

                # Check if any underlying GADEN cell is occupied
                region = self.occupancy_grid_2d[gaden_j_min:gaden_j_max, gaden_i_min:gaden_i_max]
                if region.size > 0 and np.any(region != 0):
                    self.occupancy_grid[j, i] = 1

        self.node.get_logger().info(
            f'Created infotaxis grid: {self.grid_shape} cells with {self.step_size}m resolution'
        )

    def _initialize_uniform_distribution(self):
        """Initialize uniform probability distribution over free cells."""
        # Count free cells (value = 0)
        self.num_free_cells = np.sum(self.occupancy_grid == 0)

        if self.num_free_cells == 0:
            self.node.get_logger().error('No free cells in infotaxis grid!')
            return

        # Initialize probability distribution
        self.probability_dist = np.zeros_like(self.occupancy_grid, dtype=np.float64)

        # Set uniform probability for free cells
        uniform_prob = 1.0 / self.num_free_cells
        self.probability_dist[self.occupancy_grid == 0] = uniform_prob

    def get_cell_coordinates(self, i: int, j: int) -> np.ndarray:
        """
        Convert infotaxis grid indices to world coordinates (cell center).

        Args:
            i: Grid x-index
            j: Grid y-index

        Returns:
            World coordinates [x, y, z]
        """
        x = self.grid_origin[0] + (i + 0.5) * self.step_size
        y = self.grid_origin[1] + (j + 0.5) * self.step_size
        z = self.grid_origin[2] + (self.z_level + 0.5) * self.gaden_cell_size
        return np.array([x, y, z])

    def get_cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to infotaxis grid indices.

        Args:
            x: World x-coordinate
            y: World y-coordinate

        Returns:
            Grid indices (i, j)
        """
        i = int((x - self.grid_origin[0]) / self.step_size)
        j = int((y - self.grid_origin[1]) / self.step_size)

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

    def _log_initialization_info(self):
        """Log grid initialization information."""
        self.node.get_logger().info(f'Z-level: {self.z_level}')
        self.node.get_logger().info(f'Step size: {self.step_size}m')
        self.node.get_logger().info(f'Infotaxis grid dimensions: {self.grid_shape}')
        self.node.get_logger().info(f'Grid origin: {self.grid_origin}')
        self.node.get_logger().info(f'Number of free cells: {self.num_free_cells}')
        if self.num_free_cells > 0:
            uniform_prob = self.probability_dist[self.occupancy_grid == 0][0]
            self.node.get_logger().info(f'Uniform probability per cell: {uniform_prob:.6f}')
