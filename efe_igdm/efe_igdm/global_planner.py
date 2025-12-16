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
        """Compute centroid of cluster in grid coordinates."""
        if not self.cells:
            return (0, 0)

        gx_mean = int(np.mean([cell[0] for cell in self.cells]))
        gy_mean = int(np.mean([cell[1] for cell in self.cells]))
        return (gx_mean, gy_mean)


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

    Implements Section IV.B.3 of the paper:
    1. Detect frontiers using breadth-first search
    2. Cluster frontiers and compute centroids
    3. Build PRM graph with frontier vertices
    4. Evaluate frontier vertices using global utility function (Eq. 22)
    5. Navigate to highest-utility frontier
    """

    def __init__(self,
                 occupancy_grid: OccupancyGridMap,
                 robot_radius: float = 0.35,
                 prm_samples: int = 200,
                 prm_connection_radius: float = 2.0,
                 frontier_min_size: int = 3,
                 lambda_p: float = 0.1,  # Weight for path cost in utility
                 lambda_s: float = 0.05):  # Weight for source distance in utility
        """
        Initialize global planner.

        Parameters:
        -----------
        occupancy_grid : OccupancyGridMap
            Current SLAM map
        robot_radius : float
            Robot footprint radius for collision checking
        prm_samples : int
            Number of vertices to sample for PRM
        prm_connection_radius : float
            Maximum distance for connecting PRM vertices
        frontier_min_size : int
            Minimum number of cells for a valid frontier cluster
        lambda_p : float
            Weight for path cost in global utility (Eq. 22)
        lambda_s : float
            Weight for source distance in global utility (Eq. 22)
        """
        self.occupancy_grid = occupancy_grid
        self.robot_radius = robot_radius
        self.prm_samples = prm_samples
        self.prm_connection_radius = prm_connection_radius
        self.frontier_min_size = frontier_min_size
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s

        # Frontier detection results
        self.frontier_cells = []  # List of (gx, gy) frontier cells
        self.frontier_clusters = []  # List of FrontierCluster objects

        # PRM graph
        self.vertices = []  # List of PRMVertex objects
        self.vertex_dict = {}  # Dict: vertex_id -> PRMVertex

        # Best frontier selection
        self.best_frontier_vertex = None
        self.best_global_path = []
        self.best_utility = -np.inf

    def detect_frontiers(self) -> List[Tuple[int, int]]:
        """
        Detect frontier cells using breadth-first search.

        A frontier cell is:
        - Free (value = 0)
        - Adjacent to at least one unknown cell (value = -1)

        Returns:
        --------
        frontier_cells : List[Tuple[int, int]]
            List of (gx, gy) grid coordinates of frontier cells
        """
        frontier_cells = []

        grid = self.occupancy_grid.grid
        height, width = grid.shape

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Scan all cells
        for gy in range(height):
            for gx in range(width):
                # Check if cell is free
                if grid[gy, gx] != 0:
                    continue

                # Check if adjacent to unknown cell
                is_frontier = False
                for dx, dy in neighbors:
                    nx, ny = gx + dx, gy + dy

                    # Check bounds
                    if 0 <= nx < width and 0 <= ny < height:
                        if grid[ny, nx] == -1:  # Unknown cell
                            is_frontier = True
                            break

                if is_frontier:
                    frontier_cells.append((gx, gy))

        self.frontier_cells = frontier_cells
        return frontier_cells

    def cluster_frontiers(self) -> List[FrontierCluster]:
        """
        Cluster frontier cells based on spatial proximity.

        Uses connected components labeling (4-connected).

        Returns:
        --------
        clusters : List[FrontierCluster]
            List of frontier clusters, each containing multiple cells
        """
        if not self.frontier_cells:
            return []

        # Create a set for fast lookup
        frontier_set = set(self.frontier_cells)
        visited = set()
        clusters = []

        # 4-connected neighbors for clustering
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # BFS to find connected components
        for start_cell in self.frontier_cells:
            if start_cell in visited:
                continue

            # BFS from this cell
            cluster_cells = []
            queue = deque([start_cell])
            visited.add(start_cell)

            while queue:
                gx, gy = queue.popleft()
                cluster_cells.append((gx, gy))

                # Check neighbors
                for dx, dy in neighbors:
                    nx, ny = gx + dx, gy + dy

                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            # Create cluster if it meets minimum size
            if len(cluster_cells) >= self.frontier_min_size:
                cluster = FrontierCluster(cluster_cells)
                # Convert centroid to world coordinates
                cluster.centroid_world = self.occupancy_grid.grid_to_world(
                    cluster.centroid_grid[0], cluster.centroid_grid[1]
                )
                clusters.append(cluster)

        self.frontier_clusters = clusters
        return clusters

    def build_prm_graph(self, current_position: Tuple[float, float]) -> None:
        """
        Build Probabilistic Roadmap (PRM) graph.

        Steps:
        1. Sample random vertices in free space
        2. Add current position as a vertex
        3. Connect vertices within connection radius
        4. Designate frontier vertices (closest to each frontier centroid)

        Parameters:
        -----------
        current_position : Tuple[float, float]
            Current robot position in world coordinates
        """
        self.vertices = []
        self.vertex_dict = {}
        vertex_id = 0

        # Add current position as vertex 0
        current_vertex = PRMVertex(current_position, vertex_id)
        self.vertices.append(current_vertex)
        self.vertex_dict[vertex_id] = current_vertex
        vertex_id += 1

        # Sample random vertices in free space
        attempts = 0
        max_attempts = self.prm_samples * 10  # Allow more attempts for challenging maps

        x_min = self.occupancy_grid.origin_x
        y_min = self.occupancy_grid.origin_y
        x_max = x_min + self.occupancy_grid.real_world_width
        y_max = y_min + self.occupancy_grid.real_world_height

        while len(self.vertices) < self.prm_samples + 1 and attempts < max_attempts:
            attempts += 1

            # Sample random position
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            # Check if position is valid (free space)
            if self.occupancy_grid.is_valid((x, y), radius=self.robot_radius):
                vertex = PRMVertex((x, y), vertex_id)
                self.vertices.append(vertex)
                self.vertex_dict[vertex_id] = vertex
                vertex_id += 1

        # Connect vertices within connection radius
        for i, vertex_i in enumerate(self.vertices):
            for j in range(i + 1, len(self.vertices)):
                vertex_j = self.vertices[j]

                # Compute distance
                dist = np.linalg.norm(
                    np.array(vertex_i.position) - np.array(vertex_j.position)
                )

                # Check if within connection radius
                if dist <= self.prm_connection_radius:
                    # Check collision-free path
                    if self._is_path_collision_free(vertex_i.position, vertex_j.position):
                        # Add bidirectional edge
                        vertex_i.neighbors.append((vertex_j.id, dist))
                        vertex_j.neighbors.append((vertex_i.id, dist))

        # Designate frontier vertices (closest to each frontier centroid)
        for cluster in self.frontier_clusters:
            centroid = cluster.centroid_world

            # Find closest vertex to this centroid
            min_dist = float('inf')
            closest_vertex = None

            for vertex in self.vertices:
                dist = np.linalg.norm(
                    np.array(vertex.position) - np.array(centroid)
                )

                if dist < min_dist:
                    min_dist = dist
                    closest_vertex = vertex

            # Designate as frontier vertex
            if closest_vertex is not None:
                closest_vertex.is_frontier_vertex = True
                closest_vertex.frontier_cluster = cluster

    def _is_path_collision_free(self, pos1: Tuple[float, float],
                                pos2: Tuple[float, float]) -> bool:
        """
        Check if straight-line path between two positions is collision-free.

        Uses discrete sampling similar to RRT.
        """
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return self.occupancy_grid.is_valid(tuple(pos1), radius=self.robot_radius)

        # Sample at resolution of half the grid cell size
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        num_samples = max(num_samples, 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)

            if not self.occupancy_grid.is_valid(
                (sample_pos[0], sample_pos[1]),
                radius=self.robot_radius
            ):
                return False

        return True

    def dijkstra_in_prm(self, start_vertex_id: int,
                       goal_vertex_id: int) -> Tuple[Optional[List[int]], float]:
        """
        Find shortest path in PRM graph using Dijkstra's algorithm.

        Parameters:
        -----------
        start_vertex_id : int
            Starting vertex ID (typically 0 for current position)
        goal_vertex_id : int
            Goal vertex ID (frontier vertex)

        Returns:
        --------
        path : Optional[List[int]]
            List of vertex IDs from start to goal, or None if no path exists
        cost : float
            Total path cost, or inf if no path exists
        """
        # Initialize distances
        distances = {vid: float('inf') for vid in self.vertex_dict.keys()}
        distances[start_vertex_id] = 0.0

        # Previous vertex for path reconstruction
        previous = {vid: None for vid in self.vertex_dict.keys()}

        # Priority queue: (distance, vertex_id)
        pq = [(0.0, start_vertex_id)]
        visited = set()

        while pq:
            current_dist, current_id = heapq.heappop(pq)

            if current_id in visited:
                continue
            visited.add(current_id)

            # Check if reached goal
            if current_id == goal_vertex_id:
                break

            # Skip if we found a better path already
            if current_dist > distances[current_id]:
                continue

            # Explore neighbors
            current_vertex = self.vertex_dict[current_id]
            for neighbor_id, edge_cost in current_vertex.neighbors:
                new_dist = current_dist + edge_cost

                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))

        # Reconstruct path
        if distances[goal_vertex_id] == float('inf'):
            return None, float('inf')

        path = []
        current = goal_vertex_id
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return path, distances[goal_vertex_id]

    def evaluate_frontier_vertices(self,
                                   current_position: Tuple[float, float],
                                   particle_filter: ParticleFilterOptimized) -> Dict:
        """
        Evaluate all frontier vertices using global utility function (Eq. 22).

        Ug(vf,i) = I(vf,i) * exp(-λp * cost(rk, vf,i)) * exp(-λs * D(vf,i, r̂0))

        where:
        - I(vf,i) = mutual information at frontier vertex
        - cost(rk, vf,i) = path length from current position to frontier
        - D(vf,i, r̂0) = Euclidean distance from frontier to estimated source
        - λp, λs = weights

        Parameters:
        -----------
        current_position : Tuple[float, float]
            Current robot position
        particle_filter : ParticleFilterOptimized
            Current particle filter state for mutual information calculation

        Returns:
        --------
        results : Dict
            Dictionary with evaluation results for all frontier vertices
        """
        # Get estimated source location
        estimate, _ = particle_filter.get_estimate()
        source_estimate = np.array([estimate['x'], estimate['y']])

        # Results storage
        frontier_vertices = [v for v in self.vertices if v.is_frontier_vertex]
        utilities = []
        mutual_informations = []
        path_costs = []
        source_distances = []
        paths = []

        # Evaluate each frontier vertex
        for frontier_vertex in frontier_vertices:
            # 1. Compute path cost using Dijkstra in PRM
            path_ids, path_cost = self.dijkstra_in_prm(0, frontier_vertex.id)

            if path_ids is None:
                # No path to this frontier - skip it
                continue

            # 2. Compute mutual information at frontier position
            mutual_info = self._compute_mutual_information(
                frontier_vertex.position, particle_filter
            )

            # 3. Compute Euclidean distance to estimated source
            source_dist = np.linalg.norm(
                np.array(frontier_vertex.position) - source_estimate
            )

            # 4. Compute global utility (Equation 22)
            utility = (mutual_info *
                      np.exp(-self.lambda_p * path_cost) *
                      np.exp(-self.lambda_s * source_dist))

            # Store results
            utilities.append(utility)
            mutual_informations.append(mutual_info)
            path_costs.append(path_cost)
            source_distances.append(source_dist)
            paths.append((frontier_vertex, path_ids, path_cost))

        # Find best frontier
        if not utilities:
            return {
                'best_frontier_vertex': None,
                'best_global_path': [],
                'best_utility': -np.inf,
                'frontier_vertices': [],
                'utilities': [],
                'error': 'No reachable frontiers'
            }

        best_idx = np.argmax(utilities)
        best_frontier_vertex, best_path_ids, best_path_cost = paths[best_idx]

        # Convert path IDs to world coordinates
        best_global_path = [self.vertex_dict[vid].position for vid in best_path_ids]

        # Store results
        self.best_frontier_vertex = best_frontier_vertex
        self.best_global_path = best_global_path
        self.best_utility = utilities[best_idx]

        return {
            'best_frontier_vertex': best_frontier_vertex,
            'best_global_path': best_global_path,
            'best_utility': utilities[best_idx],
            'best_mutual_info': mutual_informations[best_idx],
            'best_path_cost': path_costs[best_idx],
            'best_source_dist': source_distances[best_idx],
            'frontier_vertices': [fv for fv, _, _ in paths],
            'utilities': utilities,
            'mutual_informations': mutual_informations,
            'path_costs': path_costs,
            'source_distances': source_distances,
            'num_frontiers': len(frontier_vertices),
            'num_reachable': len(utilities)
        }

    def _compute_mutual_information(self,
                                   position: Tuple[float, float],
                                   particle_filter: ParticleFilterOptimized) -> float:
        """
        Compute mutual information I(position) = H - E[H | measurement].

        This is the same calculation as in RRT, but for a single position.
        """
        current_entropy = particle_filter.get_entropy()

        # Determine number of measurement levels
        if hasattr(particle_filter.sensor_model, 'num_levels'):
            num_measurements = particle_filter.sensor_model.num_levels
        else:
            num_measurements = 2

        # Compute expected entropy
        expected_entropy = 0.0
        for measurement in range(num_measurements):
            prob = particle_filter.predict_measurement_probability(position, measurement)
            hyp_entropy = particle_filter.compute_hypothetical_entropy(measurement, position)
            expected_entropy += prob * hyp_entropy

        mutual_information = current_entropy - expected_entropy
        return mutual_information

    def plan(self, current_position: Tuple[float, float],
            particle_filter: ParticleFilterOptimized) -> Dict:
        """
        Execute full global planning pipeline.

        Steps:
        1. Detect frontiers
        2. Cluster frontiers
        3. Build PRM graph
        4. Evaluate frontier vertices
        5. Return best global path

        Parameters:
        -----------
        current_position : Tuple[float, float]
            Current robot position
        particle_filter : ParticleFilterOptimized
            Current particle filter state

        Returns:
        --------
        results : Dict
            Complete planning results including best path and debug info
        """
        # 1. Detect frontiers
        frontier_cells = self.detect_frontiers()

        if not frontier_cells:
            return {
                'success': False,
                'error': 'No frontiers detected - exploration complete?',
                'best_global_path': [],
                'best_utility': -np.inf
            }

        # 2. Cluster frontiers
        clusters = self.cluster_frontiers()

        if not clusters:
            return {
                'success': False,
                'error': 'No valid frontier clusters (all too small)',
                'best_global_path': [],
                'best_utility': -np.inf,
                'num_frontier_cells': len(frontier_cells)
            }

        # 3. Build PRM graph
        self.build_prm_graph(current_position)

        # 4. Evaluate frontier vertices
        eval_results = self.evaluate_frontier_vertices(current_position, particle_filter)

        if eval_results['best_frontier_vertex'] is None:
            return {
                'success': False,
                'error': eval_results.get('error', 'No reachable frontiers'),
                'best_global_path': [],
                'best_utility': -np.inf,
                'num_frontiers': len(clusters),
                'num_frontier_cells': len(frontier_cells)
            }

        # Success!
        return {
            'success': True,
            'best_global_path': eval_results['best_global_path'],
            'best_utility': eval_results['best_utility'],
            'best_mutual_info': eval_results['best_mutual_info'],
            'best_path_cost': eval_results['best_path_cost'],
            'best_source_dist': eval_results['best_source_dist'],
            'num_frontier_cells': len(frontier_cells),
            'num_clusters': len(clusters),
            'num_frontiers': eval_results['num_frontiers'],
            'num_reachable': eval_results['num_reachable'],
            # Debug info for visualization
            'frontier_cells': self.frontier_cells,
            'frontier_clusters': self.frontier_clusters,
            'prm_vertices': self.vertices,
            'frontier_vertices': eval_results['frontier_vertices'],
            'all_utilities': eval_results['utilities']
        }
