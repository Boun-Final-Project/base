"""
Global Planner for Dual-Mode RRT-Infotaxis GSL - HIGH PERFORMANCE VERSION

Optimizations:
1. KD-Tree for PRM connectivity: Reduces graph building from O(N^2) to O(N log N).
2. Single-Source Dijkstra: Computes paths to ALL frontiers in one pass.
3. Vectorized Frontier Detection: Uses scipy.ndimage for fast grid operations.
4. Optimistic Validity Checks: Allows planning through Unknown space (-1).
"""

import numpy as np
import heapq
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, label, find_objects
from typing import List, Tuple, Optional, Dict, Set

# Relative imports (keep these matching your file structure)
from ..mapping.occupancy_grid import OccupancyGridMap
from ..estimation.particle_filter import ParticleFilter


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
        # Neighbors list is now populated via KD-Tree query results
        # Format: List of (neighbor_id, edge_cost)
        self.neighbors = []  
        self.is_frontier_vertex = False
        self.frontier_cluster = None  # Reference to FrontierCluster if is_frontier_vertex


class GlobalPlanner:
    """
    Global planner for frontier-based exploration with information gain.
    """

    def __init__(self,
                 occupancy_grid: OccupancyGridMap,
                 robot_radius: float = 0.35,
                 prm_samples: int = 300,
                 prm_connection_radius: float = 2.5,
                 frontier_min_size: int = 3,
                 lambda_p: float = 0.1,
                 lambda_s: float = 0.05,
                 debug: bool = True):

        self.occupancy_grid = occupancy_grid
        self.robot_radius = robot_radius
        self.prm_samples = prm_samples
        self.prm_connection_radius = prm_connection_radius
        self.frontier_min_size = frontier_min_size
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.debug = debug

        self.frontier_cells = []
        self.frontier_clusters = []
        self.vertices = []
        self.vertex_dict = {}
        
        # Adjacency list for efficient graph algorithms: {id: [(neighbor_id, cost), ...]}
        self.adj_list = {} 

        self.best_frontier_vertex = None
        self.best_global_path = []
        self.best_utility = -np.inf

        # Store ranked frontiers for fallback when path becomes blocked
        # List of dicts containing 'vertex', 'utility', 'path_ids'
        self.ranked_frontiers = []
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
        
        # 1. Create Boolean Masks
        is_free = (grid == 0)
        is_unknown = (grid == -1)
        
        # 2. Dilate the Unknown regions (8-connectivity)
        structure = np.ones((3, 3), dtype=bool)
        unknown_dilated = binary_dilation(is_unknown, structure=structure)
        
        # 3. Intersection: Frontier = Is Free AND Is touching Unknown
        frontier_mask = is_free & unknown_dilated
        
        # 4. Extract coordinates
        y_idxs, x_idxs = np.where(frontier_mask)
        self.frontier_cells = list(zip(x_idxs, y_idxs))
        
        return self.frontier_cells

    def cluster_frontiers(self) -> List[FrontierCluster]:
        """
        Cluster frontier cells using Connected Components (Labeling).
        """
        if not self.frontier_cells:
            self.frontier_clusters = []
            return []

        # 1. Reconstruct the Boolean mask
        grid_shape = self.occupancy_grid.grid.shape
        frontier_mask = np.zeros(grid_shape, dtype=bool)
        
        if len(self.frontier_cells) > 0:
            xs, ys = zip(*self.frontier_cells)
            frontier_mask[ys, xs] = True
            
        # 2. Label connected components (8-connectivity)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(frontier_mask, structure=structure)
        
        clusters = []
        
        # 3. Iterate over found labels using find_objects for speed
        slices = find_objects(labeled_array)
        
        for i, sl in enumerate(slices):
            if sl is None: continue
            
            label_id = i + 1
            # Mask relative to the slice
            local_mask = (labeled_array[sl] == label_id)
            ly, lx = np.where(local_mask)
            
            # Convert to global grid coordinates
            global_y = ly + sl[0].start
            global_x = lx + sl[1].start
            
            cluster_cells = list(zip(global_x, global_y))
            
            if len(cluster_cells) >= self.frontier_min_size:
                cluster = FrontierCluster(cluster_cells)
                # Calculate World Centroid
                cluster.centroid_world = self.occupancy_grid.grid_to_world(
                    cluster.centroid_grid[0], cluster.centroid_grid[1]
                )
                clusters.append(cluster)
                
        self.frontier_clusters = clusters
        return clusters

    def build_prm_graph(self, current_position: Tuple[float, float]) -> None:
        """
        Build PRM graph utilizing KD-Tree for O(N log N) connections.
        """
        self.vertices = []
        self.vertex_dict = {}
        self.adj_list = {}
        
        # --- 1. Collect Sample Points ---
        sample_points = []
        vertex_ids = []
        types = [] # 'start', 'frontier_X', 'sample'

        # 1a. Start Position (ID 0)
        sample_points.append(current_position)
        vertex_ids.append(0)
        types.append('start')

        # 1b. Frontier Centroids
        for i, cluster in enumerate(self.frontier_clusters):
            pos = cluster.centroid_world
            if self._is_valid_optimistic(pos):
                sample_points.append(pos)
                vertex_ids.append(len(sample_points)-1)
                types.append(f'frontier_{i}')

        # 1c. Random Sampling
        attempts = 0
        max_attempts = self.prm_samples * 5
        x_min, y_min = self.occupancy_grid.origin_x, self.occupancy_grid.origin_y
        x_max = x_min + self.occupancy_grid.real_world_width
        y_max = y_min + self.occupancy_grid.real_world_height

        target_count = self.prm_samples + len(self.frontier_clusters)
        
        while len(sample_points) < target_count and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            if self._is_valid_optimistic((x, y)):
                sample_points.append((x, y))
                vertex_ids.append(len(sample_points)-1)
                types.append('sample')

        # --- 2. Build KD-Tree ---
        if not sample_points:
            return

        points_array = np.array(sample_points)
        tree = KDTree(points_array)

        # --- 3. Create Vertex Objects ---
        for idx, pos, v_type in zip(vertex_ids, sample_points, types):
            v = PRMVertex(pos, idx)
            if v_type.startswith('frontier'):
                v.is_frontier_vertex = True
                cluster_idx = int(v_type.split('_')[1])
                v.frontier_cluster = self.frontier_clusters[cluster_idx]
            
            self.vertices.append(v)
            self.vertex_dict[idx] = v
            self.adj_list[idx] = [] # Initialize adjacency

        # --- 4. Connect Vertices Efficiently ---
        # query_pairs returns all pairs (i, j) with dist < r
        pairs = tree.query_pairs(self.prm_connection_radius)

        for i, j in pairs:
            pos_i = sample_points[i]
            pos_j = sample_points[j]
            dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))

            # Allow start node (0) to be slightly invalid to escape walls
            allow_start = (i == 0 or j == 0)
            
            if self._is_path_collision_free(pos_i, pos_j, allow_start_invalid=allow_start):
                # Add to Vertex objects (legacy support)
                self.vertex_dict[i].neighbors.append((j, dist))
                self.vertex_dict[j].neighbors.append((i, dist))
                
                # Add to Adjacency List (for fast Dijkstra)
                self.adj_list[i].append((j, dist))
                self.adj_list[j].append((i, dist))

    def _is_path_collision_free(self, pos1: Tuple[float, float], pos2: Tuple[float, float], allow_start_invalid: bool = False) -> bool:
        """Check collision using optimistic check along the segment."""
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        dist = np.linalg.norm(pos2 - pos1)
        
        if dist < 1e-6:
            if allow_start_invalid: return True
            return self._is_valid_optimistic(tuple(pos1))

        # Relaxed sampling: Check every resolution unit
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution)))
        num_samples = max(num_samples, 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            
            # If start is allowed invalid, skip the very first point (i=0)
            if i == 0 and allow_start_invalid:
                continue

            sample_pos = pos1 + t * (pos2 - pos1)
            if not self._is_valid_optimistic((sample_pos[0], sample_pos[1])):
                return False
        return True

    def compute_all_paths_from_start(self, start_id: int = 0) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        Run One-to-All Dijkstra. 
        Returns distances and predecessor map for ALL reachable nodes from start_id.
        """
        distances = {vid: float('inf') for vid in self.vertex_dict}
        previous = {vid: None for vid in self.vertex_dict}
        distances[start_id] = 0.0
        
        # Priority Queue: (distance, vertex_id)
        pq = [(0.0, start_id)]
        
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            
            # Optimization: If we found a shorter path already, skip
            if current_dist > distances[current_id]:
                continue
            
            # Use adj_list for fast iteration
            if current_id in self.adj_list:
                for neighbor_id, weight in self.adj_list[current_id]:
                    new_dist = current_dist + weight
                    
                    if new_dist < distances[neighbor_id]:
                        distances[neighbor_id] = new_dist
                        previous[neighbor_id] = current_id
                        heapq.heappush(pq, (new_dist, neighbor_id))
                    
        return distances, previous

    def reconstruct_path(self, goal_id: int, previous: Dict[int, int]) -> Optional[List[int]]:
        """Reconstruct path from previous map."""
        if goal_id not in previous or (previous[goal_id] is None and goal_id != 0):
            return None # Unreachable
            
        path = []
        curr = goal_id
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
            # Safety break
            if len(path) > len(previous): break 
            
        return path[::-1] # Reverse to get Start -> Goal

    def _compute_mutual_information(self, position: Tuple[float, float], pf: ParticleFilter) -> float:
        """Compute MI using particle filter."""
        current_entropy = pf.get_entropy()
        num_measurements = pf.sensor_model.num_levels if hasattr(pf.sensor_model, 'num_levels') else 2
        
        expected_entropy = 0.0
        for m in range(num_measurements):
            prob = pf.predict_measurement_probability(position, m)
            if prob > 1e-4: # Skip negligible probabilities
                hyp_entropy = pf.compute_hypothetical_entropy(m, position)
                expected_entropy += prob * hyp_entropy
            
        return current_entropy - expected_entropy

    def evaluate_frontier_vertices(self, current_pos: Tuple[float, float], pf: ParticleFilter) -> Dict:
        """
        Evaluate frontiers based on utility function (Eq. 22).
        Uses Single-Source Dijkstra for efficiency.
        """
        # 1. Run Dijkstra ONCE from robot position (ID 0)
        dists, prevs = self.compute_all_paths_from_start(0)
        
        estimate, _ = pf.get_estimate()
        source_estimate = np.array([estimate['x'], estimate['y']])
        frontier_vertices = [v for v in self.vertices if v.is_frontier_vertex]
        
        candidates = []

        if self.debug:
            print(f"DEBUG: Evaluating {len(frontier_vertices)} frontiers...")

        for f_vertex in frontier_vertices:
            # 2. Check Reachability using pre-computed Dijkstra map
            path_cost = dists[f_vertex.id]
            if path_cost == float('inf'):
                continue # Unreachable

            # 3. Compute Metrics
            mi = self._compute_mutual_information(f_vertex.position, pf)
            
            # Filter low info
            if mi <= 1e-6: continue

            src_dist = np.linalg.norm(np.array(f_vertex.position) - source_estimate)

            # 4. Compute Utility (Eq 22)
            utility = mi * np.exp(-self.lambda_p * path_cost) * np.exp(-self.lambda_s * src_dist)

            # Defer path reconstruction until selection to save time
            candidates.append({
                'utility': utility,
                'vertex': f_vertex,
                'path_cost': path_cost,
                'mi': mi,
                'src_dist': src_dist
            })

        if not candidates:
            self.ranked_frontiers = []
            return {'best_frontier_vertex': None, 'error': 'No reachable high-info frontiers'}

        # 5. Sort Candidates
        candidates.sort(key=lambda x: x['utility'], reverse=True)

        # 6. Reconstruct path for the BEST candidate
        best = candidates[0]
        best_path_ids = self.reconstruct_path(best['vertex'].id, prevs)
        self.best_global_path = [self.vertex_dict[vid].position for vid in best_path_ids]
        
        self.best_frontier_vertex = best['vertex']
        self.best_utility = best['utility']

        # 7. Store ranked list for fallback (lazily storing the predecessors map)
        self.ranked_frontiers = []
        for c in candidates:
            self.ranked_frontiers.append({
                'utility': c['utility'],
                'vertex': c['vertex'],
                'prev_map': prevs # Store map reference to reconstruct path later if needed
            })
        self.current_frontier_index = 0

        if self.debug:
            print(f"DEBUG: Selected Vertex {best['vertex'].id}, Utility: {best['utility']:.4f}")

        return {
            'best_frontier_vertex': best['vertex'],
            'best_global_path': self.best_global_path,
            'best_utility': best['utility'],
            'best_mutual_info': best['mi'],
            'best_path_cost': best['path_cost'],
            'best_source_dist': best['src_dist'],
            'num_frontiers': len(frontier_vertices),
            'num_reachable': len(candidates)
        }

    def get_next_best_frontier(self) -> Optional[Dict]:
        """
        Get the next best frontier when current path is blocked.
        """
        self.current_frontier_index += 1

        if self.current_frontier_index >= len(self.ranked_frontiers):
            return None

        # Retrieve candidate info
        candidate = self.ranked_frontiers[self.current_frontier_index]
        vertex = candidate['vertex']
        prev_map = candidate['prev_map']
        
        # Reconstruct path on demand
        path_ids = self.reconstruct_path(vertex.id, prev_map)
        if not path_ids:
            return self.get_next_best_frontier() # Skip if reconstruction fails (shouldn't happen)

        path = [self.vertex_dict[vid].position for vid in path_ids]

        self.best_frontier_vertex = vertex
        self.best_global_path = path
        self.best_utility = candidate['utility']

        return {
            'success': True,
            'best_frontier_vertex': vertex,
            'best_global_path': path,
            'best_utility': candidate['utility'],
            'frontier_index': self.current_frontier_index,
            'total_frontiers': len(self.ranked_frontiers)
        }

    def plan(self, current_position: Tuple[float, float], particle_filter: ParticleFilter) -> Dict:
        """Execute global planning pipeline."""
        
        # 1. Detect
        frontier_cells = self.detect_frontiers()
        if not frontier_cells:
            return {'success': False, 'error': 'No frontiers detected'}

        # 2. Cluster
        clusters = self.cluster_frontiers()
        if not clusters:
            return {'success': False, 'error': 'No valid clusters'}

        # 3. Build Graph (PRM)
        self.build_prm_graph(current_position)
        
        # 4. Evaluate & Select
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