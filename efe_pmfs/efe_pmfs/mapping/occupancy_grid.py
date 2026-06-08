"""
Occupancy grid for efe_pmfs.

Based on efe_igdm/mapping/occupancy_grid.py, with an added factory
`create_empty_occupancy_map_from_dims` that builds a blank grid from known
bounding-box dimensions and resolution (no dependency on a reference map or
GADEN service). This is the path used when the map extent is a ROS param.
"""
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure, binary_dilation

CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_OUTLET = 2


class OccupancyGridMap:
    """2D occupancy grid. Cell values: -1 unknown, 0 free, 1 occupied, 2 outlet."""

    def __init__(self, grid, params):
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.origin_x = params['env_min'][0]
        self.origin_y = params['env_min'][1]
        self.resolution = params['cell_size']
        self.grid = grid
        self.real_world_width = self.width * self.resolution
        self.real_world_height = self.height * self.resolution
        self.grid_width = self.width
        self.grid_height = self.height
        self.z_level = params.get('z_level', None)
        self.z_height = params.get('z_height', None)

    def world_to_grid(self, x, y):
        gx = int(np.floor((x - self.origin_x) / self.resolution))
        gy = int(np.floor((y - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def is_cell_free(self, gx, gy):
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return self.grid[gy, gx] == 0
        return False

    def is_valid(self, position, radius=0.2):
        gx, gy = self.world_to_grid(*position)
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        radius_cells = int(np.ceil(radius / self.resolution))
        radius_sq_cells = radius_cells ** 2
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_sq_cells:
                    continue
                check_gx = gx + dx
                check_gy = gy + dy
                if 0 <= check_gx < self.grid_width and 0 <= check_gy < self.grid_height:
                    if self.grid[check_gy, check_gx] != 0:
                        return False
        return True

    def fill_enclosed_unknown(self):
        """Fill unknown connected components that are walled off from free space."""
        is_unknown = (self.grid == CELL_UNKNOWN)
        if not np.any(is_unknown):
            return 0
        struct = generate_binary_structure(2, 2)
        labeled, num_features = label(is_unknown, structure=struct)
        if num_features == 0:
            return 0
        is_free = (self.grid == CELL_FREE)
        filled = 0
        for label_id in range(1, num_features + 1):
            component_mask = (labeled == label_id)
            border = binary_dilation(component_mask, structure=struct) & ~component_mask
            if not np.any(border & is_free):
                self.grid[component_mask] = CELL_OCCUPIED
                filled += int(np.sum(component_mask))
        return filled

    def visualize(self, ax=None, show_grid=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        cmap = ListedColormap([
            (1.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 1.0),
            (0.0, 0.7, 1.0, 1.0),
        ])
        extent = [self.origin_x, self.origin_x + self.width * self.resolution,
                  self.origin_y, self.origin_y + self.height * self.resolution]
        display_grid = self.grid + 1
        ax.imshow(display_grid, origin='lower', extent=extent, cmap=cmap, alpha=1.0, vmin=0, vmax=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        if show_grid:
            ax.grid(True, alpha=0.3)
        return ax


def create_empty_occupancy_map_from_dims(width_m, height_m, cell_size_m,
                                         origin_x=0.0, origin_y=0.0):
    """
    Build a blank (all-unknown) occupancy grid from known dimensions.

    This is how efe_pmfs initializes its map: the bounding box and resolution
    are ROS params (map_width_m, map_height_m, cell_size_m, map_origin_*), and
    obstacle positions are discovered online via LiDAR.
    """
    w = int(np.round(width_m / cell_size_m))
    h = int(np.round(height_m / cell_size_m))
    grid = np.full((h, w), CELL_UNKNOWN, dtype=np.int8)
    params = {
        'env_min': [origin_x, origin_y, 0.0],
        'env_max': [origin_x + w * cell_size_m, origin_y + h * cell_size_m, 0.0],
        'num_cells': [w, h, 1],
        'cell_size': cell_size_m,
    }
    return OccupancyGridMap(grid, params)
