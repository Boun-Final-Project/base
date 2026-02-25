import numpy as np
from numba import njit

# ──────────────────────────────────────────────────────────────
# Numba-accelerated Bresenham ray tracing for ALL beams at once
# ──────────────────────────────────────────────────────────────
@njit(cache=True)
def _trace_rays_numba(grid, outlet_mask, has_outlet_mask,
                      gx0, gy0,
                      end_gx, end_gy, hit_flags,
                      width, height):
    """
    Bresenham ray trace for every beam.
    Marks free cells as 0 (or 2 for outlet), obstacle endpoints as 1 (or 2).
    Returns number of obstacles marked.
    """
    n_beams = len(end_gx)
    obstacles = 0

    for b in range(n_beams):
        x = gx0
        y = gy0
        x1 = end_gx[b]
        y1 = end_gy[b]

        dx = abs(x1 - x)
        dy = abs(y1 - y)
        sx = 1 if x < x1 else -1
        sy = 1 if y < y1 else -1
        err = dx - dy

        # --- ray trace: mark free cells along the ray ---
        while True:
            if 0 <= x < width and 0 <= y < height:
                if x == x1 and y == y1:
                    break
                if grid[y, x] <= 0:
                    if has_outlet_mask and outlet_mask[y, x]:
                        grid[y, x] = 2
                    else:
                        grid[y, x] = 0

            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # --- mark obstacle endpoint ---
        if hit_flags[b]:
            ex, ey = end_gx[b], end_gy[b]
            if 0 <= ex < width and 0 <= ey < height:
                if has_outlet_mask and outlet_mask[ey, ex]:
                    grid[ey, ex] = 2
                else:
                    grid[ey, ex] = 1
                obstacles += 1

    return obstacles


class LidarMapper:
    """
    Handles Simple SLAM logic: updates an occupancy grid using LaserScan data.
    Beam endpoints are computed in vectorized NumPy; ray tracing uses Numba JIT.
    """
    def __init__(self, occupancy_grid, outlet_mask=None):
        self.slam_map = occupancy_grid
        self.outlet_mask = outlet_mask  # 2D boolean array (y, x), same shape as grid

    def update_from_scan(self, scan_msg, robot_x, robot_y, robot_theta):
        """
        Process a LaserScan and update the map.
        Vectorized endpoint computation + Numba ray tracing.
        """
        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        n = len(ranges)

        # --- 1. Vectorized beam endpoint computation ---
        finite_mask = np.isfinite(ranges)
        raw_ranges = ranges.copy()
        ranges = np.minimum(ranges, scan_msg.range_max)

        angles = scan_msg.angle_min + np.arange(n) * scan_msg.angle_increment
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)

        local_x = ranges * cos_a
        local_y = ranges * sin_a
        end_x = robot_x + local_x * cos_t - local_y * sin_t
        end_y = robot_y + local_x * sin_t + local_y * cos_t

        hit_flags = finite_mask & (raw_ranges >= scan_msg.range_min) & (raw_ranges < scan_msg.range_max)

        # Filter to finite beams only
        valid = finite_mask
        end_x = end_x[valid]
        end_y = end_y[valid]
        hit_flags = hit_flags[valid]

        # --- 2. Convert to grid coordinates ---
        og = self.slam_map
        gx0 = int((robot_x - og.origin_x) / og.resolution)
        gy0 = int((robot_y - og.origin_y) / og.resolution)

        end_gx = np.floor((end_x - og.origin_x) / og.resolution).astype(np.int32)
        end_gy = np.floor((end_y - og.origin_y) / og.resolution).astype(np.int32)

        # --- 3. Numba ray tracing ---
        has_outlet = self.outlet_mask is not None
        outlet = self.outlet_mask if has_outlet else np.empty((0, 0), dtype=np.bool_)

        obstacles = _trace_rays_numba(
            og.grid, outlet, has_outlet,
            gx0, gy0,
            end_gx, end_gy, hit_flags,
            og.width, og.height
        )

        return int(obstacles)
