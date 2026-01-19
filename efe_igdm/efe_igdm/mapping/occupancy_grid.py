import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from gaden_msgs.srv import Occupancy


def load_3d_occupancy_grid_from_service(node: Node, service_name='/gaden_environment/occupancyMap3D',
                                        z_level: int = 5, timeout_sec: float = 5.0):
    """
    Load 3D occupancy grid from GADEN ROS2 service and extract 2D slice at z_level.

    Parameters:
    -----------
    node : rclpy.node.Node
        ROS2 node to create service client
    service_name : str
        Name of the occupancy service (default: '/gaden_environment/occupancyMap3D')
    z_level : int
        Z-level index to extract 2D slice (default: 5)
    timeout_sec : float
        Timeout for service call in seconds (default: 5.0)

    Returns:
    --------
    grid_2d : np.ndarray
        2D array (y, x) with occupancy values
    params : dict
        Dictionary with env_min, env_max, num_cells, cell_size
    """
    # Create service client
    client = node.create_client(Occupancy, service_name)

    # Wait for service to be available
    node.get_logger().info(f'Waiting for service {service_name}...')
    if not client.wait_for_service(timeout_sec=timeout_sec):
        raise RuntimeError(f'Service {service_name} not available after {timeout_sec} seconds')

    # Create request (empty for this service)
    request = Occupancy.Request()

    # Call service
    node.get_logger().info(f'Calling service {service_name}...')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    if not future.done():
        raise RuntimeError(f'Service call to {service_name} timed out')

    response = future.result()
    if response is None:
        raise RuntimeError(f'Service call to {service_name} failed')

    # Extract response data
    origin = response.origin
    resolution = response.resolution
    num_cells_x = response.num_cells_x
    num_cells_y = response.num_cells_y
    num_cells_z = response.num_cells_z
    occupancy_data = response.occupancy

    node.get_logger().info(f'Received occupancy grid:')
    node.get_logger().info(f'  Origin: ({origin.x}, {origin.y}, {origin.z})')
    node.get_logger().info(f'  Resolution: {resolution} m')
    node.get_logger().info(f'  Dimensions: {num_cells_x} x {num_cells_y} x {num_cells_z}')
    node.get_logger().info(f'  Total cells: {len(occupancy_data)}')

    # Validate z_level
    if z_level < 0 or z_level >= num_cells_z:
        node.get_logger().warn(
            f'z_level {z_level} out of bounds [0, {num_cells_z-1}]. Using z_level=0'
        )
        z_level = 0

    # Reshape occupancy data to 3D grid
    # GADEN stores data as: index = x + y*nx + z*nx*ny (x varies fastest)
    # So we need to reshape to (z, y, x) and then reorder to (x, y, z)
    grid_3d = np.array(occupancy_data, dtype=np.uint8).reshape(num_cells_z, num_cells_y, num_cells_x)

    # Reorder from (z, y, x) to (x, y, z) for easier indexing
    grid_3d = np.transpose(grid_3d, (2, 1, 0))

    # Extract 2D slice at z_level
    # grid_2d shape: (num_cells_x, num_cells_y)
    grid_2d = grid_3d[:, :, z_level]

    # Transpose to match expected (y, x) format for visualization
    grid_2d = grid_2d.T

    # Convert occupancy values: 0=free, anything else (>0)=occupied
    grid_2d = (grid_2d > 0).astype(np.int8)

    # Create params dictionary
    params = {
        'env_min': [origin.x, origin.y, origin.z],
        'env_max': [
            origin.x + num_cells_x * resolution,
            origin.y + num_cells_y * resolution,
            origin.z + num_cells_z * resolution
        ],
        'num_cells': [num_cells_x, num_cells_y, num_cells_z],
        'cell_size': resolution,
        'z_level': z_level,
        'z_height': origin.z + z_level * resolution
    }

    node.get_logger().info(f'Extracted 2D slice at z_level={z_level} (z={params["z_height"]:.2f}m)')
    node.get_logger().info(f'  2D Grid shape: {grid_2d.shape} (y, x)')
    node.get_logger().info(f'  Occupied cells: {np.sum(grid_2d)} / {grid_2d.size}')

    return grid_2d, params

class OccupancyGridMap:
    """Occupancy grid map for indoor environments."""

    def __init__(self, grid, params):
        """
        Initialize occupancy grid map.

        Parameters:
        -----------
        grid : np.ndarray
            2D occupancy grid (y, x) with 0=free, 1=occupied
        params : dict
            Dictionary with env_min, env_max, num_cells, cell_size
        """
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.origin_x = params['env_min'][0]
        self.origin_y = params['env_min'][1]
        self.resolution = params['cell_size']
        self.grid = grid
        self.real_world_width = self.width * self.resolution
        self.real_world_height = self.height * self.resolution

        # Calculate grid dimensions
        self.grid_width = self.width
        self.grid_height = self.height

        # Store z-level info if available
        self.z_level = params.get('z_level', None)
        self.z_height = params.get('z_height', None)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        gx = int(np.floor((x - self.origin_x) / self.resolution))
        gy = int(np.floor((y - self.origin_y) / self.resolution))
        return gx, gy
    
    def is_cell_free(self, gx, gy):
        """
        Fast check if a specific grid cell is free.
        Useful for pathfinders (A*, Dijkstra) that work on grid indices.
        """
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return self.grid[gy, gx] == 0
        return False

    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def is_valid(self, position: tuple[float, float], radius: float = 0.2) -> bool:
        """
        Check if position is valid (within bounds and collision-free).
        Optimized to use squared distances for speed.
        """
        gx, gy = self.world_to_grid(*position)
        
        # Fast bounds check
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False

        # Calculate radius in cells
        radius_cells = int(np.ceil(radius / self.resolution))
        radius_sq_cells = radius_cells**2  # Optimization: compare squares
        
        # Check surrounding cells
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                # Optimization: Skip corners of the square bounding box
                # to approximate a circle without using sqrt()
                if dx*dx + dy*dy > radius_sq_cells:
                    continue
                
                check_gx = gx + dx
                check_gy = gy + dy

                # Check grid bounds and occupancy
                if 0 <= check_gx < self.grid_width and 0 <= check_gy < self.grid_height:
                    if self.grid[check_gy, check_gx] != 0:
                        return False
        return True

    # def is_valid(self, position: tuple[float, float], radius: float = 0.2) -> bool:
    #     """
    #     Check if position is valid (within bounds and collision-free).

    #     Parameters:
    #     -----------
    #     position : tuple[float, float]
    #         World coordinates (x, y)
    #     radius : float
    #         Safety radius to check around the position (default: 0.1m)

    #     Returns:
    #     --------
    #     valid : bool
    #         True if position is valid and collision-free
    #     """
    #     # gx, gy = self.world_to_grid(*position)
    #     # if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
    #     #     return False

    #     # # Check surrounding cells within the radius
    #     # radius_cells = int(np.ceil(radius / self.resolution))
    #     # for dx in range(-radius_cells, radius_cells + 1):
    #     #     for dy in range(-radius_cells, radius_cells + 1):
    #     #         if 0 <= gx + dx < self.grid_width and 0 <= gy + dy < self.grid_height:
    #     #             if self.grid[gy + dy, gx + dx] != 0:
    #     #                 return False
    #     # return True
    #     gx, gy = self.world_to_grid(*position)
    #     if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
    #         return False

    #     # Check surrounding cells within the radius
    #     radius_cells = int(np.ceil(radius / self.resolution))
        
    #     # --- START OF FIX ---
    #     # Pre-calculate the squared radius in grid cells for comparison
    #     radius_sq_cells = radius_cells**2
    #     # --- END OF FIX ---

    #     for dx in range(-radius_cells, radius_cells + 1):
    #         for dy in range(-radius_cells, radius_cells + 1):
                
    #             # --- START OF FIX ---
    #             # Check if the grid cell (dx, dy) is within the circular radius
    #             # by comparing squared distances.
    #             if dx*dx + dy*dy > radius_sq_cells:
    #                 continue  # Skip this cell, it's in the "corner" of the square
    #             # --- END OF FIX ---
                
    #             check_gx = gx + dx
    #             check_gy = gy + dy

    #             if 0 <= check_gx < self.grid_width and 0 <= check_gy < self.grid_height:
    #                 if self.grid[check_gy, check_gx] != 0:
    #                     return False
    #     return True

    def add_rectangle_obstacle(self, x_min, y_min, x_max, y_max):
        """Add a rectangular obstacle to the map."""
        gx_min, gy_min = self.world_to_grid(x_min, y_min)
        gx_max, gy_max = self.world_to_grid(x_max, y_max)

        # Ensure bounds are within grid
        gx_min = max(0, gx_min)
        gy_min = max(0, gy_min)
        gx_max = min(self.grid_width - 1, gx_max)
        gy_max = min(self.grid_height - 1, gy_max)

        self.grid[gy_min:gy_max+1, gx_min:gx_max+1] = 1

    def visualize(self, ax=None, show_grid=False):
        """Visualize the occupancy grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Create custom colormap with transparency for unknown cells
        # Colors: unknown (-1), free (0), occupied (1)
        # RGBA format: (R, G, B, Alpha) where Alpha=0 is transparent
        cmap = ListedColormap([
            (1.0, 1.0, 1.0, 0.0),  # Unknown (-1): Transparent (was orange)
            (1.0, 1.0, 1.0, 1.0),  # Free (0): White
            (0.5, 0.5, 0.5, 1.0)   # Occupied (1): Gray
        ])

        # Display grid with correct world coordinates
        extent = [self.origin_x, self.origin_x + self.width * self.resolution,
                  self.origin_y, self.origin_y + self.height * self.resolution]
        # Shift grid values: -1→0, 0→1, 1→2 for correct colormap indexing
        display_grid = self.grid + 1
        ax.imshow(display_grid, origin='lower', extent=extent, cmap=cmap, alpha=1.0, vmin=0, vmax=2)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')

        # Add z-level info to title if available
        if self.z_level is not None and self.z_height is not None:
            ax.set_title(f'Occupancy Grid (z_level={self.z_level}, z={self.z_height:.2f}m)')

        if show_grid:
            ax.grid(True, alpha=0.3)

        return ax


# Example usage function for ROS2 integration
def create_occupancy_map_from_service(node: Node, z_level: int = 5,
                                     service_name: str = '/gaden_environment/occupancyMap3D',
                                     timeout_sec: float = 5.0) -> OccupancyGridMap:
    """
    Convenience function to create OccupancyGridMap from ROS2 service.

    Parameters:
    -----------
    node : rclpy.node.Node
        ROS2 node
    z_level : int
        Z-level to extract (default: 5)
    service_name : str
        Service name (default: '/gaden_environment/occupancyMap3D')
    timeout_sec : float
        Service call timeout (default: 5.0)

    Returns:
    --------
    occupancy_map : OccupancyGridMap
        Occupancy grid map object
    """
    grid, params = load_3d_occupancy_grid_from_service(
        node,
        service_name=service_name,
        z_level=z_level,
        timeout_sec=timeout_sec
    )
    return OccupancyGridMap(grid, params)


def create_empty_occupancy_map(reference_map: OccupancyGridMap) -> OccupancyGridMap:
    """
    Create empty occupancy map with same dimensions as reference.

    This is useful for creating a SLAM map that starts with all cells
    as unknown (-1) and shares the same coordinate system and resolution
    as an existing map.

    Parameters:
    -----------
    reference_map : OccupancyGridMap
        Existing map to copy dimensions from

    Returns:
    --------
    empty_map : OccupancyGridMap
        Empty OccupancyGridMap (all cells unknown)
    """
    # Create grid initialized to -1 (unknown)
    # ROS OccupancyGrid convention: -1=unknown, 0=free, 100=occupied
    empty_grid = np.full_like(reference_map.grid, -1, dtype=np.int8)

    # Create params dict matching reference map
    params = {
        'env_min': [reference_map.origin_x, reference_map.origin_y, 0.0],
        'env_max': [
            reference_map.origin_x + reference_map.width * reference_map.resolution,
            reference_map.origin_y + reference_map.height * reference_map.resolution,
            0.0
        ],
        'num_cells': [reference_map.width, reference_map.height, 1],
        'cell_size': reference_map.resolution,
        'z_level': reference_map.z_level,
        'z_height': reference_map.z_height
    }

    return OccupancyGridMap(empty_grid, params)
