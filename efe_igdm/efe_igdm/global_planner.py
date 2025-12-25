"""
Global Planner for Dual-Mode RRT-Infotaxis GSL - OPTIMIZED VERSION

Based on Section IV.B.3 from:
"Gas Source Localization in Unknown Indoor Environments Using Dual-Mode
Information-Theoretic Search" by Kim et al., IEEE RA-L 2025

OPTIMIZATIONS:
- Replaced Python loop-based frontier detection with scipy.ndimage (Vectorized).
- Replaced BFS clustering with Connected Components labeling (C-backend).
"""

import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict, Set
from scipy.ndimage import binary_dilation, label, find_objects

# Assuming these are in the same directory, keep your relative imports
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
        Compute 'Safe Centroid' of the cluster (Medoid).
        Finds the cell within the cluster closest to the geometric mean.
        """
        if not self.cells:
            return (0, 0)

        # 1. Compute geometric mean
        mean_x = np.mean([c[0] for c in self.cells])
        mean_y = np.mean([c[1] for c in self.cells])

        # 2. Find the cell in the cluster closest to the mean
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
    
    Optimizations applied:
    1. Vectorized Frontier Detection (scipy.ndimage)
    2. Vectorized Clustering (scipy.ndimage.label)
    3. Targeted Sampling: Explicitly adds frontier centroids to PRM
    4. Optimistic Validity: Allows planning through Unknown space (-1)
    """

    def __init__(self,
                 occupancy_grid: OccupancyGridMap,
                 robot_radius: float = 0.35,
                 prm_samples: int = 300, 
                 prm_connection_radius: float = 2.5,
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

        # Store ranked frontiers for fallback when path becomes blocked
        self.ranked_frontiers = []  # List of (utility, vertex, path_ids) tuples
        self.current_frontier_index = 0

    def _is_valid_optimistic(self, position: Tuple[float, float]) -> bool:
        """
        Check if position is valid using OPTIMISTIC planning.
        Treats Unknown (-1) as Free (0). Only Occupied (>0) is invalid.
        """
        gx, gy = self.occupancy_grid.world_to_grid(*position)
        
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
                    # Fail only if strictly occupied (> 0). Allow 0 and -1.
                    if cell_val > 0: 
                        return False
        return True

    def detect_frontiers(self) -> List[Tuple[int, int]]:
        """
        Vectorized frontier detection.
        A frontier is a FREE cell (0) that is adjacent to an UNKNOWN cell (-1).
        """
        grid = self.occupancy_grid.grid
        
        # 1. Create Boolean Masks (Operations on full arrays are fast in NumPy)
        is_free = (grid == 0)
        is_unknown = (grid == -1)
        
        # 2. Dilate the Unknown regions
        # If we expand "Unknown" by 1 pixel (8-connected), the expansion will cover 
        # the "Free" pixels that are touching the edge.
        structure = np.ones((3, 3), dtype=bool) # 8-connectivity
        unknown_dilated = binary_dilation(is_unknown, structure=structure)
        
        # 3. Intersection: Frontier = Is Free AND Is touching Unknown
        frontier_mask = is_free & unknown_dilated
        
        # 4. Extract coordinates
        # np.where returns (row_indices, col_indices) -> (y, x)
        y_idxs, x_idxs = np.where(frontier_mask)
        
        # 5. Convert to list of (x, y) tuples
        self.frontier_cells = list(zip(x_idxs, y_idxs))
        
        return self.frontier_cells

    def cluster_frontiers(self) -> List[FrontierCluster]:
        """
        Cluster frontier cells using Connected Components (Labeling).
        Replaces slow Python BFS with optimized scipy implementation.
        """
        if not self.frontier_cells:
            self.frontier_clusters = []
            return []

        # 1. Reconstruct the Boolean mask from the list of cells
        grid_shape = self.occupancy_grid.grid.shape
        frontier_mask = np.zeros(grid_shape, dtype=bool)
        
        if len(self.frontier_cells) > 0:
            xs, ys = zip(*self.frontier_cells)
            frontier_mask[ys, xs] = True
            
        # 2. Label connected components (8-connectivity)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(frontier_mask, structure=structure)
        
        clusters = []
        
        # 3. Iterate over found labels to group cells efficiently
        # find_objects returns a list of slices for each label
        slices = find_objects(labeled_array)
        
        for i, sl in enumerate(slices):
            if sl is None: continue
            
            # Label IDs start at 1
            label_id = i + 1
            
            # Mask relative to the slice (much faster than masking full grid)
            local_mask = (labeled_array[sl] == label_id)
            
            # Get local coordinates
            ly, lx = np.where(local_mask)
            
            # Convert to global grid coordinates
            global_y = ly + sl[0].start
            global_x = lx + sl[1].start
            
            # Combine into list of (x, y)
            cluster_cells = list(zip(global_x, global_y))
            
            # Filter small clusters
            if len(cluster_cells) >= self.frontier_min_size:
                cluster = FrontierCluster(cluster_cells)
                
                # Calculate World Centroid using medoid logic
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
        for cluster in self.frontier_clusters:
            pos = cluster.centroid_world
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

            if self._is_valid_optimistic((x, y)):
                vertex = PRMVertex((x, y), vertex_id)
                self.vertices.append(vertex)
                self.vertex_dict[vertex_id] = vertex
                vertex_id += 1

        # 4. Connect Vertices
        # Simple N^2 connection strategy (acceptable for small N ~300)
        for i, vertex_i in enumerate(self.vertices):
            for j in range(i + 1, len(self.vertices)):
                vertex_j = self.vertices[j]
                
                dist = np.linalg.norm(np.array(vertex_i.position) - np.array(vertex_j.position))

                if dist <= self.prm_connection_radius:
                    # Allow invalid start position for Vertex 0 (Current Robot Pos)
                    allow_start_invalid = (vertex_i.id == 0)
                    
                    if self._is_path_collision_free(vertex_i.position, vertex_j.position, allow_start_invalid=allow_start_invalid):
                        vertex_i.neighbors.append((vertex_j.id, dist))
                        vertex_j.neighbors.append((vertex_i.id, dist))

    def _is_path_collision_free(self, pos1: Tuple[float, float], pos2: Tuple[float, float], allow_start_invalid: bool = False) -> bool:
        """Check collision using optimistic check along the segment."""
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        dist = np.linalg.norm(pos2 - pos1)
        
        if dist < 1e-6:
            if allow_start_invalid:
                return True
            return self._is_valid_optimistic(tuple(pos1))

        # Check resolution
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        num_samples = max(num_samples, 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            
            # If start is allowed invalid, skip the very first point (i=0)
            if i == 0 and allow_start_invalid:
                continue

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
        candidates = []

        for f_vertex in frontier_vertices:
            # 1. Check path from Current Position (Vertex 0)
            path_ids, path_cost = self.dijkstra_in_prm(0, f_vertex.id)
            if path_ids is None: continue

            # 2. Compute metrics
            mi = self._compute_mutual_information(f_vertex.position, pf)
            src_dist = np.linalg.norm(np.array(f_vertex.position) - source_estimate)
            
            # 3. Utility (Eq 22)
            utility = mi * np.exp(-self.lambda_p * path_cost) * np.exp(-self.lambda_s * src_dist)
            
            candidates.append((utility, f_vertex, path_ids, mi, path_cost, src_dist))

        if not candidates:
            self.ranked_frontiers = []
            self.current_frontier_index = 0
            return {'best_frontier_vertex': None, 'error': 'No reachable frontiers'}

        # Sort by utility (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Store all ranked candidates for fallback
        self.ranked_frontiers = [(c[0], c[1], c[2]) for c in candidates]  # (utility, vertex, path_ids)
        self.current_frontier_index = 0

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
            'frontier_vertices': [c[1] for c in candidates],
            'utilities': [c[0] for c in candidates]
        }

    def get_next_best_frontier(self) -> Optional[Dict]:
        """
        Get the next best frontier when current path is blocked.
        Returns None if no more frontiers available.
        """
        self.current_frontier_index += 1

        if self.current_frontier_index >= len(self.ranked_frontiers):
            return None

        utility, vertex, path_ids = self.ranked_frontiers[self.current_frontier_index]
        path = [self.vertex_dict[vid].position for vid in path_ids]

        self.best_frontier_vertex = vertex
        self.best_global_path = path
        self.best_utility = utility

        return {
            'success': True,
            'best_frontier_vertex': vertex,
            'best_global_path': path,
            'best_utility': utility,
            'frontier_index': self.current_frontier_index,
            'total_frontiers': len(self.ranked_frontiers)
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

        eval_results['success'] = True
        eval_results['frontier_cells'] = self.frontier_cells
        eval_results['frontier_clusters'] = self.frontier_clusters
        eval_results['prm_vertices'] = self.vertices
        
        return eval_results