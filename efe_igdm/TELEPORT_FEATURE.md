# Teleportation Feature for Wall Avoidance

## Overview
This feature automatically teleports the robot 0.5m away from walls when it gets stuck, instead of switching to the global planner.

## Implementation

### 1. Add Publisher (after line 148 in igdm.py)

```python
# Publisher for teleportation when stuck
self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/PioneerP3DX/initialpose', 10)
```

### 2. Add Three New Methods (after line 282 in igdm.py)

#### Method 1: find_safe_position_away_from_wall()
```python
def find_safe_position_away_from_wall(self, current_pos, distance=0.5):
    """
    Find a safe position 'distance' meters away from the nearest wall.

    Parameters:
    -----------
    current_pos : tuple
        Current (x, y) position
    distance : float
        Distance to move away from wall (default 0.5m)

    Returns:
    --------
    tuple : (x, y) safe position, or current_pos if no safe position found
    """
    if current_pos is None:
        return current_pos

    cx, cy = current_pos

    # Sample directions in a circle around robot
    num_samples = 16  # Check 16 directions (every 22.5 degrees)
    max_search_distance = 2.0  # Search up to 2m away

    best_position = None
    max_clearance = 0

    for i in range(num_samples):
        angle = (2 * np.pi * i) / num_samples

        # Try positions at increasing distances in this direction
        for test_dist in np.linspace(distance, max_search_distance, 5):
            test_x = cx + test_dist * np.cos(angle)
            test_y = cy + test_dist * np.sin(angle)
            test_pos = (test_x, test_y)

            # Check if position is valid (collision-free with robot radius)
            if self._is_valid_optimistic(test_pos):
                # Calculate minimum distance to nearest obstacle
                clearance = self._calculate_clearance(test_pos)

                if clearance > max_clearance:
                    max_clearance = clearance
                    best_position = test_pos

    if best_position is not None and max_clearance >= distance:
        return best_position
    else:
        self.get_logger().warn(f'Could not find safe position {distance}m from walls')
        return current_pos
```

#### Method 2: _calculate_clearance()
```python
def _calculate_clearance(self, position):
    """Calculate minimum distance to nearest obstacle."""
    gx, gy = self.slam_map.world_to_grid(*position)

    if gx < 0 or gx >= self.slam_map.width or gy < 0 or gy >= self.slam_map.height:
        return 0.0

    # Search in expanding radius for nearest obstacle
    max_radius = int(2.0 / self.slam_map.resolution)  # 2m in grid cells

    for radius in range(1, max_radius):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue  # Only check perimeter

                check_gx = gx + dx
                check_gy = gy + dy

                if 0 <= check_gx < self.slam_map.width and 0 <= check_gy < self.slam_map.height:
                    if self.slam_map.grid[check_gy, check_gx] > 0:  # Obstacle found
                        distance = radius * self.slam_map.resolution
                        return distance

    return max_radius * self.slam_map.resolution  # No obstacle found within search radius
```

#### Method 3: teleport_robot()
```python
def teleport_robot(self, target_x, target_y):
    """
    Teleport robot to a new position.

    Parameters:
    -----------
    target_x : float
        Target X coordinate
    target_y : float
        Target Y coordinate
    """
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = 'map'

    msg.pose.pose.position.x = float(target_x)
    msg.pose.pose.position.y = float(target_y)
    msg.pose.pose.position.z = 0.0

    # Keep current orientation if available
    if self.current_theta is not None:
        from math import sin, cos
        qz = sin(self.current_theta / 2.0)
        qw = cos(self.current_theta / 2.0)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
    else:
        msg.pose.pose.orientation.w = 1.0

    # Zero covariance
    msg.pose.covariance = [0.0] * 36

    self.initialpose_pub.publish(msg)
    self.get_logger().info(f'Teleported robot to ({target_x:.2f}, {target_y:.2f})')

    # Update internal position tracking
    self.current_position = (target_x, target_y)
```

### 3. Replace trigger_recovery() Method (line 283)

Replace the entire `trigger_recovery()` method with:

```python
def trigger_recovery(self):
    self.get_logger().warn('!!! STUCK DETECTED - TELEPORTING AWAY FROM WALL !!!')
    self.consecutive_failures = 0
    self.in_recovery = False

    # Find safe position 0.5m away from nearest wall
    safe_pos = self.find_safe_position_away_from_wall(self.current_position, distance=0.5)

    if safe_pos != self.current_position:
        self.get_logger().info(f'Safe position found: ({safe_pos[0]:.2f}, {safe_pos[1]:.2f})')
        self.teleport_robot(safe_pos[0], safe_pos[1])

        # Wait for teleport to take effect
        time.sleep(1.0)

        # Reset dead end detector after teleport
        self.dead_end_detector.reset(initial_threshold=self.get_parameter('dead_end_initial_threshold').value)

        # Stay in LOCAL mode and try planning again
        self.get_logger().info('Teleport complete. Resuming LOCAL mode.')
        self.planning_pending = True
        return

    # If teleport failed, fall back to global planner
    self.get_logger().warn('Teleport failed. Switching to GLOBAL planner as fallback.')

    # Reset dead end detector to allow fresh exploration
    self.dead_end_detector.reset(initial_threshold=self.get_parameter('dead_end_initial_threshold').value)

    # Force switch to Global Mode
    self.planner_mode = 'GLOBAL'

    # Plan using Global Planner
    self.get_logger().info('Planning recovery path...')
    global_plan_result = self.global_planner.plan(self.current_position, self.particle_filter)

    if global_plan_result['success']:
        self.get_logger().info(f'Global recovery plan found with {len(global_plan_result["best_global_path"])} waypoints.')
        self.global_path = global_plan_result['best_global_path']
        self.global_path_index = 1

        self.clear_global_planner_visualizations()
        self.visualize_frontier_cells(global_plan_result['frontier_cells'])
        self.visualize_frontier_centroids(global_plan_result['frontier_clusters'])
        self.visualize_prm_graph(global_plan_result['prm_vertices'])
        self.visualize_global_path(self.global_path)

        self.planning_pending = True
    else:
        self.get_logger().error('Global recovery plan FAILED. Resuming LOCAL mode.')
        self.planner_mode = 'LOCAL'
        self.dead_end_detector.reset(initial_threshold=self.get_parameter('dead_end_initial_threshold').value)
        self.planning_pending = True
```

## How It Works

1. **Detection**: When robot gets stuck (3 consecutive movements < 5cm)
2. **Search**: Samples 16 directions around robot, testing distances 0.5m to 2.0m
3. **Selection**: Picks position with maximum clearance from walls (minimum 0.5m)
4. **Teleport**: Publishes to `/PioneerP3DX/initialpose` topic
5. **Resume**: Stays in LOCAL mode and continues RRT-Infotaxis planning

## Benefits

- **Faster recovery**: No need to plan global path
- **Maintains search continuity**: Stays in LOCAL mode
- **Physical escape**: Actually moves robot away from walls
- **Minimal disruption**: Only moves 0.5-2.0m, stays near search area
- **Fallback safety**: If teleport fails, uses original GLOBAL planner

## Configuration

Adjust teleport distance by changing the `distance` parameter in line:
```python
safe_pos = self.find_safe_position_away_from_wall(self.current_position, distance=0.5)
```

Change to `distance=0.8` for more clearance, or `distance=0.3` for tighter spaces.
