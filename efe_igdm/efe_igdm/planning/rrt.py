import numpy as np
from ..estimation.particle_filter_optimized import ParticleFilterOptimized
from ..mapping.occupancy_grid import OccupancyGridMap
from typing import List, Tuple, Optional, Dict

class Node:
    """A node in the RRT tree using slots for memory optimization."""
    __slots__ = ['position', 'parent', 'depth', 'entropy_gain']
    
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.entropy_gain = -np.inf

class RRT:
    """Rapidly-exploring Random Tree - OPTIMIZED"""
    def __init__(self, occupancy_grid: OccupancyGridMap, N_tn: int, R_range: float, delta: float,
                 max_depth: int = 4, discount_factor: float = 0.8, positive_weight: float = 0.5,
                 robot_radius: float = 0.35, max_iterations: int = None):
        self.occupancy_grid = occupancy_grid
        self.N_tn = N_tn
        self.R_range = R_range
        self.nodes: List[Node] = []
        self.delta = delta
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.positive_weight = positive_weight
        self.robot_radius = robot_radius
        self.max_iterations = max_iterations if max_iterations is not None else (N_tn * 100)

    def sprawl(self, start_pos: Tuple[float, float]) -> None:
        """Generates the RRT tree."""
        self.nodes = [Node(start_pos)]
        iteration = 0
        
        # Pre-calculate constants
        grid_width = self.occupancy_grid.width
        grid_height = self.occupancy_grid.height

        while len(self.nodes) < self.N_tn and iteration < self.max_iterations:
            iteration += 1

            # Sample random point within R_range
            r = self.R_range * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x_rand = start_pos[0] + r * np.cos(theta)
            y_rand = start_pos[1] + r * np.sin(theta)
            
            # Find closest existing node
            closest_node = self.get_closest_node((x_rand, y_rand))
            
            # Steer towards random point
            diff = np.array([x_rand, y_rand]) - closest_node.position
            dist = np.linalg.norm(diff)
            
            if dist > self.delta:
                direction = diff / dist
                new_pos = closest_node.position + direction * self.delta
            else:
                new_pos = np.array([x_rand, y_rand])

            # Check validity
            if self.is_collision_free_vectorized(closest_node.position, new_pos):
                new_node = Node(new_pos, closest_node)
                self.nodes.append(new_node)

    def get_closest_node(self, position: Tuple[float, float]) -> Node:
        """
        Find closest node using vectorized numpy operations.
        Much faster than looping for < 1000 nodes.
        """
        target = np.array(position)
        # Gather all positions into a (N, 2) array
        # Note: This list comprehension is fast enough for N < 1000. 
        # For N > 1000, we should maintain a separate self.node_positions array.
        all_positions = np.array([n.position for n in self.nodes])
        
        # Compute squared distances (avoids sqrt for comparison)
        dists_sq = np.sum((all_positions - target)**2, axis=1)
        closest_index = np.argmin(dists_sq)
        return self.nodes[closest_index]

    def is_collision_free_vectorized(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        Check collision using vectorized grid sampling.
        """
        # 1. Check endpoints first (fast fail)
        if not self.occupancy_grid.is_valid(tuple(pos2), radius=self.robot_radius):
            return False

        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return True

        # 2. Determine number of samples based on resolution
        # Sampling slightly denser than resolution guarantees no skipping
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        
        # 3. Generate points (Linear Interpolation)
        # Creates (N, 2) array of points along the line
        ts = np.linspace(0, 1, num_samples + 1)
        # Broadcasting: pos1 + t * (pos2 - pos1)
        line_points = pos1 + np.outer(ts, (pos2 - pos1))

        # 4. Batch convert to grid coordinates
        # We access occupancy grid internals for speed here
        # Assuming occupancy_grid has world_to_grid helper or using resolution math:
        grid_xs = ((line_points[:, 0] - self.occupancy_grid.origin_x) / self.occupancy_grid.resolution).astype(int)
        grid_ys = ((line_points[:, 1] - self.occupancy_grid.origin_y) / self.occupancy_grid.resolution).astype(int)

        # 5. Boundary Checks (Vectorized)
        valid_x = (grid_xs >= 0) & (grid_xs < self.occupancy_grid.width)
        valid_y = (grid_ys >= 0) & (grid_ys < self.occupancy_grid.height)
        valid_indices = valid_x & valid_y
        
        if not np.all(valid_indices):
            return False # Path goes out of bounds

        # 6. Check Occupancy
        # 0 = Free, 100 = Occupied, -1 = Unknown
        # We assume > 0 is collision.
        grid_values = self.occupancy_grid.grid[grid_ys, grid_xs]
        
        # If any cell is occupied (> 0), collision detected
        if np.any(grid_values > 0):
            return False
            
        return True

    def prune(self) -> List[List[Node]]:
        """Return all branches that reached max depth."""
        edge_nodes = [node for node in self.nodes if node.depth == self.max_depth]
        paths = []
        for edge_node in edge_nodes:
            path = []
            current = edge_node
            while current is not None:
                path.append(current)
                current = current.parent
            paths.append(path[::-1])
        return paths

    def calculate_branch_information(self, path: List[Node], initial_particle_filter: ParticleFilterOptimized) -> float:
        """Calculates BI (Eq 19) - Optimized with cached entropy."""
        path = path[1:] # Exclude root
        BI = 0.0

        for i, node in enumerate(path):
            if node.entropy_gain != -np.inf:
                # Use cached value
                BI += (self.discount_factor ** i) * node.entropy_gain
                continue

            # --- Information Gain Calculation ---
            current_entropy = initial_particle_filter.get_entropy()
            
            num_measurements = getattr(initial_particle_filter.sensor_model, 'num_levels', 2)
            
            expected_entropy = 0.0
            for measurement in range(num_measurements):
                prob = initial_particle_filter.predict_measurement_probability(node.position, measurement)
                if prob > 1e-6: # Optimization: Skip negligible probabilities
                    hyp_entropy = initial_particle_filter.compute_hypothetical_entropy(measurement, node.position)
                    expected_entropy += prob * hyp_entropy
                
            mutual_information = current_entropy - expected_entropy
            
            # Cache result
            node.entropy_gain = mutual_information
            
            BI += (self.discount_factor ** i) * mutual_information

        return BI

    def calculate_travel_cost(self, path : List[Node], initial_particle_filter : ParticleFilterOptimized) -> float:
        """Calculate the travel cost along a given path. J2 from the paper. Eq(31)."""
        estimation, _ = initial_particle_filter.get_estimate()  # Returns (mean, std)
        target_pos = np.array([estimation["x"], estimation["y"]])
        path = np.array([node.position for node in path])
        # Total path length
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        # Distance from end of path to estimated source
        end_pos = path[-1]
        distance_to_target = np.linalg.norm(end_pos - target_pos)
        total_cost = path_length + distance_to_target
        return total_cost
    
    def get_next_move(self, start_pos: Tuple[float,float], initial_particle_filter: ParticleFilterOptimized) -> Tuple[float,float]:
        self.nodes = [] # Reset tree
        self.sprawl(start_pos)
        paths = self.prune()
        
        best_path = None
        best_BI = -np.inf

        for path in paths:
            BI = self.calculate_branch_information(path, initial_particle_filter)
            if BI > best_BI:
                best_BI = BI
                best_path = path

        if best_path and len(best_path) > 1:
            return tuple(best_path[1].position)
        return start_pos

    def get_next_move_debug(self, start_pos : tuple[float,float], initial_particle_filter : ParticleFilterOptimized) -> dict:
        """
        Get next move with detailed debug information for visualization.

        Parameters:
        -----------
        start_pos : tuple[float, float]
            Current position of the agent
        initial_particle_filter : ParticleFilterOptimized
            Current state of the particle filter

        Returns:
        --------
        debug_info : dict
            Dictionary containing:
            - 'next_position': tuple[float, float] - Next position to move to
            - 'best_path': List[tuple[float, float]] - The selected best path
            - 'best_utility': float - Utility value of best path
            - 'best_entropy_gain': float - Entropy gain (J1) of best path
            - 'best_travel_cost': float - Travel cost (J2) of best path
            - 'all_paths': List[List[tuple[float, float]]] - All pruned branches
            - 'all_utilities': List[float] - Utility values for all paths
            - 'all_entropy_gains': List[float] - Entropy gains for all paths
            - 'all_travel_costs': List[float] - Travel costs for all paths
            - 'tree_nodes': List[Node] - All nodes in the RRT tree
            - 'num_branches': int - Number of complete branches found
            - 'estimated_source': tuple[float, float] - Current source estimate
            - 'start_position': tuple[float, float] - Starting position
            - 'sampling_radius': float - Sampling radius used
            - 'max_depth': int - Maximum tree depth
            - 'num_tree_nodes': int - Total number of nodes in tree
        """
        # Clear previous tree
        self.nodes = []

        # Build tree
        self.sprawl(start_pos)

        # Get all branches
        paths = self.prune()

        if not paths:
            # Return debug info even if no paths found
            return {
                'next_position': start_pos,
                'best_path': [start_pos],
                'best_utility': -np.inf,
                'best_entropy_gain': 0.0,
                'best_travel_cost': np.inf,
                'all_paths': [],
                'all_utilities': [],
                'all_entropy_gains': [],
                'all_travel_costs': [],
                'tree_nodes': self.nodes.copy(),
                'num_branches': 0,
                'estimated_source': (0.0, 0.0),
                'start_position': start_pos,
                'sampling_radius': self.R_range,
                'max_depth': self.max_depth,
                'num_tree_nodes': len(self.nodes),
                'error': 'No valid paths found'
            }

        # Evaluate all paths (branches)
        all_branch_information = []  # BI for each branch
        all_travel_costs = []

        best_path = None
        best_BI = -np.inf  # BI* (Equation 20)
        best_travel_cost = 0.0

        for path in paths:
            # Compute Branch Information (Equation 19)
            BI = self.calculate_branch_information(path, initial_particle_filter)

            # Compute travel cost (for debugging, not used in local planner)
            travel_cost = self.calculate_travel_cost(path, initial_particle_filter)

            all_branch_information.append(BI)
            all_travel_costs.append(travel_cost)

            # Select branch with highest BI (Equation 20)
            if BI > best_BI:
                best_BI = BI
                best_path = path
                best_travel_cost = travel_cost

        # Get next position
        if best_path is not None and len(best_path) > 1:
            next_node = best_path[1]
            next_position = next_node.position
        else:
            next_position = start_pos

        # Get estimated source location
        estimation, _ = initial_particle_filter.get_estimate()  # Returns (mean, std)
        estimated_source = (estimation['x'], estimation['y'])

        # Convert paths to list of tuples for easier visualization
        all_paths_tuples = []
        for path in paths:
            path_tuples = [tuple(pos) if isinstance(pos, np.ndarray) else pos for pos in path]
            all_paths_tuples.append(path_tuples)

        best_path_tuples = []
        if best_path is not None:
            best_path_tuples = [tuple(pos) if isinstance(pos, np.ndarray) else pos for pos in best_path]

        # Return comprehensive debug information
        return {
            'next_position': next_position,
            'best_path': best_path_tuples,
            'best_BI': best_BI,  # BI* (Equation 20) - Branch Information of best path
            'best_branch_information': best_BI,  # Alias for clarity
            'best_utility': best_BI,  # Backward compatibility (BI is the utility for local planner)
            'best_entropy_gain': best_BI,  # Backward compatibility
            'best_travel_cost': best_travel_cost,
            'all_paths': all_paths_tuples,
            'all_branch_information': all_branch_information,  # BI for all branches
            'all_utilities': all_branch_information,  # Backward compatibility
            'all_entropy_gains': all_branch_information,  # Backward compatibility
            'all_travel_costs': all_travel_costs,
            'tree_nodes': self.nodes.copy(),  # Copy to avoid modification
            'num_branches': len(paths),
            'estimated_source': estimated_source,
            'start_position': start_pos,
            'sampling_radius': self.R_range,
            'max_depth': self.max_depth,
            'num_tree_nodes': len(self.nodes)
        }