"""
Global Planner for Dual-Mode RRT-Infotaxis GSL

Based on Section IV.B.3 from:
"Gas Source Localization in Unknown Indoor Environments Using Dual-Mode
Information-Theoretic Search" by Kim et al., IEEE RA-L 2025

The global planner guides the robot from dead ends to information-rich areas
using frontier-based exploration combined with mutual information.
"""

import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict, Set
from .occupancy_grid import OccupancyGridMap
from .particle_filter_optimized import ParticleFilterOptimized


class FrontierCluster:
    """Represents a cluster of frontier cells."""
    def __init__(self, cells: List[Tuple[int, int]]):
        self.cells = cells  # List of (gx, gy) grid coordinates
        self.centroid_grid = self._compute_centroid()
        self.centroid_world = None  # Will be set by GlobalPlanner
        self.size = len(cells)

    def _compute_centroid(self) -> Tuple[int, int]:
        """
        Compute 'Safe Centroid' of the cluster.
        
        Instead of the geometric mean (which might fall inside an obstacle for 
        concave shapes), this finds the cell within the cluster that is 
        closest to the geometric mean (Medoid).
        """
        if not self.cells:
            return (0, 0)

        # 1. Compute geometric mean
        mean_x = np.mean([c[0] for c in self.cells])
        mean_y = np.mean([c[1] for c in self.cells])

        # 2. Find the cell in the cluster closest to the mean
        # This ensures the target is actually a valid frontier cell, not a wall
        best_cell = self.cells[0]
        min_dist_sq = float('inf')

        for cell in self.cells:
            dist_sq = (cell[0] - mean_x)**2 + (cell[1] - mean_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_cell = cell

        return best_cell


class PRMVertex:
    """Vertex in the Probabilistic Roadmap (PRM) graph."""
    def __init__(self, position: Tuple[float, float], vertex_id: int):
        self.position = position  # World coordinates (x, y)
        self.id = vertex_id
        self.neighbors = []  # List of (neighbor_id, edge_cost) tuples
        self.is_frontier_vertex = False
        self.frontier_cluster = None  # Reference to FrontierCluster if is_frontier_vertex


class GlobalPlanner:
    """
    Global planner for frontier-based exploration with information gain.
    
    Fixes applied:
    1. Targeted Sampling: Explicitly adds frontier centroids to PRM
    2. Optimistic Validity: Allows planning through Unknown space (-1)
    3. Safe Centroids: Uses medoids to avoid targeting walls
    """

    def __init__(self,
                 occupancy_grid: OccupancyGridMap,
                 robot_radius: float = 0.35,
                 prm_samples: int = 300, # Increased samples for better connectivity
                 prm_connection_radius: float = 2.5, # Increased radius
                 frontier_min_size: int = 3,
                 lambda_p: float = 0.1,
                 lambda_s: float = 0.05):
        
        self.occupancy_grid = occupancy_grid
        self.robot_radius = robot_radius
        self.prm_samples = prm_samples
        self.prm_connection_radius = prm_connection_radius
        self.frontier_min_size = frontier_min_size
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s

        self.frontier_cells = []
        self.frontier_clusters = []
        self.vertices = []
        self.vertex_dict = {}
        
        self.best_frontier_vertex = None
        self.best_global_path = []
        self.best_utility = -np.inf

    def _is_valid_optimistic(self, position: Tuple[float, float]) -> bool:
        """
        Check if position is valid using OPTIMISTIC planning.
        Treats Unknown (-1) as Free (0). Only Occupied (>0) is invalid.
        """
        gx, gy = self.occupancy_grid.world_to_grid(*position)
        
        # Check map bounds
        if gx < 0 or gx >= self.occupancy_grid.width or gy < 0 or gy >= self.occupancy_grid.height:
            return False

        # Check radius
        radius_cells = int(np.ceil(self.robot_radius / self.occupancy_grid.resolution))
        radius_sq = radius_cells ** 2

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy > radius_sq:
                    continue
                
                check_gx = gx + dx
                check_gy = gy + dy

                if 0 <= check_gx < self.occupancy_grid.width and 0 <= check_gy < self.occupancy_grid.height:
                    cell_val = self.occupancy_grid.grid[check_gy, check_gx]
                    # FIX: Fail only if strictly occupied (> 0). 
                    # Allow 0 (Free) and -1 (Unknown).
                    if cell_val > 0: 
                        return False
        return True

    def detect_frontiers(self) -> List[Tuple[int, int]]:
        """Detect frontier cells (Free cells adjacent to Unknowns)."""
        frontier_cells = []
        grid = self.occupancy_grid.grid
        height, width = grid.shape
        
        # Optimization: use numpy for faster detection instead of loop
        # Find all free cells
        free_indices = np.where(grid == 0)
        
        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    
        # Check neighbors of free cells
        for y, x in zip(free_indices[0], free_indices[1]):
            is_frontier = False
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] == -1:  # Unknown
                        is_frontier = True
                        break
            if is_frontier:
                frontier_cells.append((x, y))

        self.frontier_cells = frontier_cells
        return frontier_cells

    def cluster_frontiers(self) -> List[FrontierCluster]:
        """Cluster frontier cells and compute safe centroids."""
        if not self.frontier_cells:
            return []

        frontier_set = set(self.frontier_cells)
        visited = set()
        clusters = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for start_cell in self.frontier_cells:
            if start_cell in visited:
                continue

            cluster_cells = []
            queue = deque([start_cell])
            visited.add(start_cell)

            while queue:
                gx, gy = queue.popleft()
                cluster_cells.append((gx, gy))

                for dx, dy in neighbors:
                    nx, ny = gx + dx, gy + dy
                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            if len(cluster_cells) >= self.frontier_min_size:
                cluster = FrontierCluster(cluster_cells)
                # Ensure centroid is set
                cluster.centroid_world = self.occupancy_grid.grid_to_world(
                    cluster.centroid_grid[0], cluster.centroid_grid[1]
                )
                clusters.append(cluster)

        self.frontier_clusters = clusters
        return clusters

    def build_prm_graph(self, current_position: Tuple[float, float]) -> None:
        """
        Build PRM graph with guaranteed connectivity to frontiers.
        """
        self.vertices = []
        self.vertex_dict = {}
        vertex_id = 0

        # 1. Add Current Position (Vertex 0)
        current_vertex = PRMVertex(current_position, vertex_id)
        self.vertices.append(current_vertex)
        self.vertex_dict[vertex_id] = current_vertex
        vertex_id += 1

        # 2. Explicitly Add Frontier Centroids (Targeted Sampling)
        # This fixes the issue where random samples miss the goal
        for cluster in self.frontier_clusters:
            pos = cluster.centroid_world
            # Use optimistic check for frontiers (they are near unknown space)
            if self._is_valid_optimistic(pos):
                vertex = PRMVertex(pos, vertex_id)
                vertex.is_frontier_vertex = True
                vertex.frontier_cluster = cluster
                
                self.vertices.append(vertex)
                self.vertex_dict[vertex_id] = vertex
                vertex_id += 1

        # 3. Add Random Samples
        attempts = 0
        max_attempts = self.prm_samples * 5
        
        x_min = self.occupancy_grid.origin_x
        y_min = self.occupancy_grid.origin_y
        x_max = x_min + self.occupancy_grid.real_world_width
        y_max = y_min + self.occupancy_grid.real_world_height

        while len(self.vertices) < self.prm_samples + len(self.frontier_clusters) and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            # Use optimistic validity check
            if self._is_valid_optimistic((x, y)):
                vertex = PRMVertex((x, y), vertex_id)
                self.vertices.append(vertex)
                self.vertex_dict[vertex_id] = vertex
                vertex_id += 1

        # 4. Connect Vertices
        for i, vertex_i in enumerate(self.vertices):
            for j in range(i + 1, len(self.vertices)):
                vertex_j = self.vertices[j]
                
                dist = np.linalg.norm(np.array(vertex_i.position) - np.array(vertex_j.position))

                if dist <= self.prm_connection_radius:
                    if self._is_path_collision_free(vertex_i.position, vertex_j.position):
                        vertex_i.neighbors.append((vertex_j.id, dist))
                        vertex_j.neighbors.append((vertex_i.id, dist))

    def _is_path_collision_free(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Check collision using optimistic check."""
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        dist = np.linalg.norm(pos2 - pos1)
        
        if dist < 1e-6:
            return self._is_valid_optimistic(tuple(pos1))

        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        num_samples = max(num_samples, 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            if not self._is_valid_optimistic((sample_pos[0], sample_pos[1])):
                return False
        return True

    def dijkstra_in_prm(self, start_vertex_id: int, goal_vertex_id: int) -> Tuple[Optional[List[int]], float]:
        """Standard Dijkstra implementation."""
        distances = {vid: float('inf') for vid in self.vertex_dict.keys()}
        distances[start_vertex_id] = 0.0
        previous = {vid: None for vid in self.vertex_dict.keys()}
        pq = [(0.0, start_vertex_id)]
        visited = set()

        while pq:
            current_dist, current_id = heapq.heappop(pq)
            if current_id in visited: continue
            visited.add(current_id)

            if current_id == goal_vertex_id: break
            if current_dist > distances[current_id]: continue

            for neighbor_id, edge_cost in self.vertex_dict[current_id].neighbors:
                new_dist = current_dist + edge_cost
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))

        if distances[goal_vertex_id] == float('inf'):
            return None, float('inf')

        path = []
        current = goal_vertex_id
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        return path, distances[goal_vertex_id]

    def _compute_mutual_information(self, position: Tuple[float, float], pf: ParticleFilterOptimized) -> float:
        """Compute MI using particle filter."""
        current_entropy = pf.get_entropy()
        num_measurements = pf.sensor_model.num_levels if hasattr(pf.sensor_model, 'num_levels') else 2
        
        expected_entropy = 0.0
        for m in range(num_measurements):
            prob = pf.predict_measurement_probability(position, m)
            hyp_entropy = pf.compute_hypothetical_entropy(m, position)
            expected_entropy += prob * hyp_entropy
            
        return current_entropy - expected_entropy

    def evaluate_frontier_vertices(self, current_pos: Tuple[float, float], pf: ParticleFilterOptimized) -> Dict:
        """Evaluate frontiers based on Eq. 22."""
        estimate, _ = pf.get_estimate()
        source_estimate = np.array([estimate['x'], estimate['y']])
        
        frontier_vertices = [v for v in self.vertices if v.is_frontier_vertex]
        results = {'utilities': [], 'path_costs': [], 'source_dists': [], 'mis': []}
        candidates = []

        for f_vertex in frontier_vertices:
            # 1. Check path
            path_ids, path_cost = self.dijkstra_in_prm(0, f_vertex.id) # 0 is current_pos
            if path_ids is None: continue

            # 2. Compute metrics
            mi = self._compute_mutual_information(f_vertex.position, pf)
            src_dist = np.linalg.norm(np.array(f_vertex.position) - source_estimate)
            
            # 3. Utility (Eq 22)
            utility = mi * np.exp(-self.lambda_p * path_cost) * np.exp(-self.lambda_s * src_dist)
            
            candidates.append((utility, f_vertex, path_ids, mi, path_cost, src_dist))

        if not candidates:
            return {'best_frontier_vertex': None, 'error': 'No reachable frontiers'}

        # Find best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        
        self.best_frontier_vertex = best[1]
        self.best_global_path = [self.vertex_dict[vid].position for vid in best[2]]
        self.best_utility = best[0]

        return {
            'best_frontier_vertex': best[1],
            'best_global_path': self.best_global_path,
            'best_utility': best[0],
            'best_mutual_info': best[3],
            'best_path_cost': best[4],
            'best_source_dist': best[5],
            'num_frontiers': len(frontier_vertices),
            'num_reachable': len(candidates),
            # Debug info needed for visualization
            'frontier_vertices': [c[1] for c in candidates],
            'utilities': [c[0] for c in candidates]
        }

    def plan(self, current_position: Tuple[float, float], particle_filter: ParticleFilterOptimized) -> Dict:
        """Execute global planning pipeline."""
        frontier_cells = self.detect_frontiers()
        if not frontier_cells:
            return {'success': False, 'error': 'No frontiers detected'}

        clusters = self.cluster_frontiers()
        if not clusters:
            return {'success': False, 'error': 'No valid clusters'}

        self.build_prm_graph(current_position)
        
        eval_results = self.evaluate_frontier_vertices(current_position, particle_filter)
        
        if eval_results.get('best_frontier_vertex') is None:
            return {
                'success': False, 
                'error': eval_results.get('error', 'No reachable frontiers'),
                'frontier_cells': self.frontier_cells,
                'frontier_clusters': self.frontier_clusters,
                'prm_vertices': self.vertices
            }

        # Merge results
        eval_results['success'] = True
        eval_results['frontier_cells'] = self.frontier_cells
        eval_results['frontier_clusters'] = self.frontier_clusters
        eval_results['prm_vertices'] = self.vertices
        
        return eval_results