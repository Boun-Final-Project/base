import numpy as np

class LidarMapper:
    """
    Handles Simple SLAM logic: updates an occupancy grid using LaserScan data.
    """
    def __init__(self, occupancy_grid):
        self.slam_map = occupancy_grid

    def update_from_scan(self, scan_msg, robot_x, robot_y, robot_theta):
        """
        Process a LaserScan and update the map using exact logic from igdm.py.
        """
        obstacles_this_scan = 0
        
        # Iterate ranges exactly as before
        for i, range_val in enumerate(scan_msg.ranges):
            if not np.isfinite(range_val):
                continue
            
            raw_range = range_val
            range_val = min(range_val, scan_msg.range_max)
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            hit_obstacle = (raw_range >= scan_msg.range_min and raw_range < scan_msg.range_max)

            local_x = range_val * np.cos(angle)
            local_y = range_val * np.sin(angle)
            
            end_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
            end_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)

            # 1. Ray tracing (Clear free space)
            self._mark_ray_as_free(robot_x, robot_y, end_x, end_y)

            # 2. Mark obstacle (if hit)
            if hit_obstacle:
                if self._mark_obstacle(end_x, end_y):
                    obstacles_this_scan += 1
                    
        return obstacles_this_scan

    def _mark_obstacle(self, world_x, world_y):
        gx, gy = self.slam_map.world_to_grid(world_x, world_y)
        if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
            return False
        self.slam_map.grid[gy, gx] = 1
        return True

    def _mark_ray_as_free(self, x0, y0, x1, y1):
        gx0, gy0 = self.slam_map.world_to_grid(x0, y0)
        gx1, gy1 = self.slam_map.world_to_grid(x1, y1)
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        x = gx0
        y = gy0
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        while True:
            if 0 <= x < self.slam_map.width and 0 <= y < self.slam_map.height:
                if x == gx1 and y == gy1:
                    break
                # BUGFIX: Don't overwrite previously detected obstacles
                if self.slam_map.grid[y, x] != 1:
                    self.slam_map.grid[y, x] = 0

            if x == gx1 and y == gy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy