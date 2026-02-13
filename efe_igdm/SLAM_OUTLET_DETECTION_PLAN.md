# Plan: Add Outlet Detection to SLAM

## Context
We're building toward wind-aware gas source localization in unknown environments. The first step is detecting **outlets** (openings like doors, windows, vents) during SLAM exploration, using GADEN's ground truth 3D occupancy grid where outlets are marked as cell value `2`. When a LiDAR ray hits a wall cell that's actually an outlet in ground truth, we mark it as `2` (outlet) instead of `1` (occupied) in the SLAM map.

Later, in real deployment, a VLM (Vision Language Model) can replace the ground truth oracle to detect openings from camera data.

## Why Only Outlets (Not Inlets) Is Fine
- GADEN marks all wall openings (doors, windows, vents) as "outlet" (value 2) — the name refers to gas filament behavior, not airflow direction
- GMRF wind estimation doesn't use inlet/outlet info at all
- For potential flow wind estimation: we need to know **where openings are** (which outlets give us). Flow direction is determined by boundary conditions or anemometer measurements
- So "outlet" effectively means "opening in the wall" — exactly what we need

## Key Insight: Minimal Code Impact
Since outlet cells have value `2` (which satisfies `> 0` and `!= 0`), **all existing collision/validity checks already treat outlets as walls** — no changes needed in RRT, global planner, navigator, or Dijkstra. Only 3 files need changes.

---

## Changes

### 1. `efe_igdm/mapping/occupancy_grid.py` — Preserve outlet info from GADEN + add query methods

**a) Add cell value constants** (after imports, line 7):
```python
# Cell value constants
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_OUTLET = 2
```

**b) Modify `load_3d_occupancy_grid_from_service()`** to also return an outlet mask.
Before line 92 binarizes the data (`grid_2d = (grid_2d > 0).astype(np.int8)`), extract the outlet mask:
```python
# Extract outlet mask BEFORE binarizing
outlet_mask = (grid_2d == 2)  # 2D boolean array

# Also check adjacent z-levels for outlets spanning multiple heights
for dz in [-1, 1]:
    alt_z = z_level + dz
    if 0 <= alt_z < num_cells_z:
        alt_slice = grid_3d[:, :, alt_z].T
        outlet_mask |= (alt_slice == 2)

# Then binarize as before
grid_2d = (grid_2d > 0).astype(np.int8)
```
Update the return to: `return grid_2d, outlet_mask, params`

**c) Add outlet query methods** to `OccupancyGridMap` class (after `is_cell_free`, line 158):
```python
def get_outlet_cells(self):
    """Get grid coordinates of all outlet cells."""
    ys, xs = np.where(self.grid == CELL_OUTLET)
    return list(zip(xs.tolist(), ys.tolist()))

def get_outlet_positions_world(self):
    """Get world coordinates of all outlet cell centers."""
    return [self.grid_to_world(gx, gy) for gx, gy in self.get_outlet_cells()]
```

**d) Update `visualize()` colormap** (line 277) to show outlets in blue:
```python
cmap = ListedColormap([
    (1.0, 1.0, 1.0, 0.0),  # Unknown (-1): Transparent
    (1.0, 1.0, 1.0, 1.0),  # Free (0): White
    (0.5, 0.5, 0.5, 1.0),  # Occupied (1): Gray
    (0.0, 0.7, 1.0, 1.0)   # Outlet (2): Blue
])
# Update vmax from 2 to 3
ax.imshow(display_grid, origin='lower', extent=extent, cmap=cmap, alpha=1.0, vmin=0, vmax=3)
```

---

### 2. `efe_igdm/mapping/lidar_mapper.py` — Check outlet mask when marking obstacles

**a) Accept outlet mask in `__init__`** (line 7):
```python
def __init__(self, occupancy_grid, outlet_mask=None):
    self.slam_map = occupancy_grid
    self.outlet_mask = outlet_mask  # 2D boolean array (y, x), same shape as grid
```

**b) Modify `_mark_obstacle()`** (line 42-47) to check the outlet mask:
```python
def _mark_obstacle(self, world_x, world_y):
    gx, gy = self.slam_map.world_to_grid(world_x, world_y)
    if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
        return False
    if self.outlet_mask is not None and self.outlet_mask[gy, gx]:
        self.slam_map.grid[gy, gx] = 2  # CELL_OUTLET
    else:
        self.slam_map.grid[gy, gx] = 1  # CELL_OCCUPIED
    return True
```

**c) Update `_mark_ray_as_free()`** (line 65) to protect outlet cells from being overwritten:
```python
# Change this:
if self.slam_map.grid[y, x] != 1:
    self.slam_map.grid[y, x] = 0

# To this (also protects outlet value 2):
if self.slam_map.grid[y, x] <= 0:
    self.slam_map.grid[y, x] = 0
```

---

### 3. `efe_igdm/igdm.py` — Wire up outlet mask loading and fix map publishing

**a) Update import** (line 13):
```python
from .mapping.occupancy_grid import (
    create_occupancy_map_from_service,
    create_empty_occupancy_map,
    load_3d_occupancy_grid_from_service  # now returns outlet_mask too
)
```

**b) Modify `_init_models_and_planners()`** (lines 146-165):
```python
def _init_models_and_planners(self):
    try:
        # Load occupancy map WITH outlet information
        grid_2d, self.outlet_mask, params = load_3d_occupancy_grid_from_service(
            self, z_level=5, service_name='/gaden_environment/occupancyMap3D', timeout_sec=10.0
        )
        from .mapping.occupancy_grid import OccupancyGridMap
        self.occupancy_map = OccupancyGridMap(grid_2d, params)

        # FIX: GADEN origin correction
        if self.occupancy_map.origin_x == 0.0 and self.occupancy_map.origin_y == 0.0:
            self.occupancy_map.origin_x = -0.2
            self.occupancy_map.origin_y = -0.2

        self.slam_map = create_empty_occupancy_map(self.occupancy_map)
        self.get_logger().info(f'Outlet mask loaded: {np.sum(self.outlet_mask)} outlet cells')
    except Exception as e:
        self.get_logger().error(f'Failed to load occupancy map: {e}')
        raise

    # Pass outlet_mask to LidarMapper
    self.marker_viz = MarkerVisualizer(self, self.slam_map)
    self.navigator = Navigator(self, on_complete_callback=self._on_navigation_complete)
    self.lidar_mapper = LidarMapper(self.slam_map, outlet_mask=self.outlet_mask)
    self.text_visualizer = TextVisualizer(self.text_info_pub, frame_id="map")
    # ... rest unchanged ...
```

**c) Fix `publish_slam_map()`** (line 598) — `grid * 100` overflows for outlet value 2:
```python
def publish_slam_map(self):
    if not hasattr(self, 'slam_map'): return
    msg = OccupancyGrid()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.info.resolution = self.slam_map.resolution
    msg.info.width = self.slam_map.width
    msg.info.height = self.slam_map.height
    msg.info.origin.position.x = self.slam_map.origin_x
    msg.info.origin.position.y = self.slam_map.origin_y
    msg.info.origin.orientation.w = 1.0

    # Explicit mapping: -1→-1, 0→0, 1→100, 2→50 (outlet)
    grid_data = self.slam_map.grid.flatten()
    ros_grid = np.full_like(grid_data, -1, dtype=np.int8)
    ros_grid[grid_data == 0] = 0
    ros_grid[grid_data == 1] = 100
    ros_grid[grid_data == 2] = 50   # outlets: lighter gray in RViz

    msg.data = ros_grid.tolist()
    self.slam_map_pub.publish(msg)
```

---

## Files NOT Changed (already compatible with outlet value 2)
| File | Check | Why it works |
|------|-------|-------------|
| `rrt.py:126` | `grid_values > 0` | Outlets (2) are collision |
| `global_planner.py:133` | `cell_val > 0` | Outlets blocked |
| `global_planner.py:145` | `grid == 0` | Outlets not free, no false frontiers |
| `navigator.py:211,226` | `grid > 0` | Outlets are walls |
| `igdm_gas_model.py:75` | `grid_data == 0` | Outlets not traversable in Dijkstra |
| `particle_filter.py` | No grid interaction | N/A |
| `sensor_model.py` | No grid interaction | N/A |

## Verification Steps
1. Run the node — check log: `"Outlet mask loaded: X outlet cells"`
2. Drive robot near outlet walls — verify `slam_map.grid` has value `2` at those cells
3. RViz `/rrt_infotaxis/slam_map`: outlets show as lighter gray (50) vs walls (100)
4. Confirm RRT/navigation still works (outlets treated as walls)
5. `slam_map.get_outlet_cells()` returns correct positions after exploration

## Next Steps (Future Work)
1. **Potential flow wind estimation**: Use discovered outlet positions to solve Laplace equation on SLAM grid
2. **Wind-biased dispersion model**: Modify IGDM Dijkstra edge weights using wind field
3. **Anemometer integration**: Subscribe to wind sensor, use readings to determine flow direction through outlets
4. **VLM replacement**: Replace ground truth outlet oracle with camera-based VLM detection
