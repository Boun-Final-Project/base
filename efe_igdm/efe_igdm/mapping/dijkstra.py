import numpy as np
import heapq
def dijkstra_distances(occupancy_map, start_x, start_y):
    """
    Compute distances from start position to all cells using Dijkstra.
    
    Parameters:
    -----------
    occupancy_map : OccupancyGridMap
    start_x, start_y : float
    
    Returns:
    --------
    distances : np.ndarray
        2D array of distances from start to all cells (inf for unreachable)
    """
    # Convert start position to grid coordinates
    start_gx, start_gy = occupancy_map.world_to_grid(start_x, start_y)
    
    # Initialize distance array with infinity
    distances = np.full((occupancy_map.grid_height, occupancy_map.grid_width), np.inf)
    distances[start_gy, start_gx] = 0.0
    
    # Priority queue: (distance, gx, gy)
    pq = [(0.0, start_gx, start_gy)]
    visited = set()
    
    # 8-connected neighbors (includes diagonals)
    # Cost: 1.0 for cardinal directions, sqrt(2) for diagonals
    neighbors = [
        (-1, 0, 1.0),   # left
        (1, 0, 1.0),    # right
        (0, -1, 1.0),   # down
        (0, 1, 1.0),    # up
        (-1, -1, np.sqrt(2)),  # diagonal
        (-1, 1, np.sqrt(2)),   # diagonal
        (1, -1, np.sqrt(2)),   # diagonal
        (1, 1, np.sqrt(2))     # diagonal
    ]
    
    while pq:
        current_dist, gx, gy = heapq.heappop(pq)
        
        # Skip if already visited
        if (gx, gy) in visited:
            continue
        visited.add((gx, gy))
        
        # Explore neighbors
        for dx, dy, edge_cost in neighbors:
            nx, ny = gx + dx, gy + dy
            
            # Check if neighbor is valid and not visited
            if not occupancy_map.is_valid(nx, ny) or (nx, ny) in visited:
                continue
            
            # Scale edge cost by resolution (convert to meters)
            edge_cost_meters = edge_cost * occupancy_map.resolution
            new_dist = current_dist + edge_cost_meters
            
            # Update distance if shorter path found
            if new_dist < distances[ny, nx]:
                distances[ny, nx] = new_dist
                heapq.heappush(pq, (new_dist, nx, ny))
    
    return distances