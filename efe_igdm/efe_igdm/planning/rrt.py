import numpy as np
from ..estimation.particle_filter_optimized import ParticleFilterOptimized
from ..mapping.occupancy_grid import OccupancyGridMap
from typing import List

class Node:
    """A node in the RRT tree."""
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.entropy_gain = -np.inf

class RRT:
    """Rapidly-exploring Random Tree"""
    def __init__(self, occupancy_grid : OccupancyGridMap, N_tn : int, R_range : float, delta : float,
                 max_depth : int = 4, discount_factor : float = 0.8, positive_weight : float = 0.5,
                 robot_radius : float = 0.35, max_iterations : int = None):
        self.occupancy_grid = occupancy_grid
        self.N_tn = N_tn
        self.R_range = R_range
        self.nodes = []
        self.delta = delta
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.positive_weight = positive_weight
        self.robot_radius = robot_radius
        self.kdtree = None
        # Maximum iterations to prevent infinite loops near walls
        # Default: 100x the target node count
        self.max_iterations = max_iterations if max_iterations is not None else (N_tn * 100)

    def sprawl(self, start_pos: tuple[float,float]) -> None:
        root = Node(start_pos)
        self.nodes.append(root)

        # Track iterations to prevent infinite loops near walls
        iteration = 0

        while len(self.nodes) < self.N_tn and iteration < self.max_iterations:
            iteration += 1

            r = self.R_range * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x = start_pos[0] + r * np.cos(theta)
            y = start_pos[1] + r * np.sin(theta)
            closest_node = self.get_closest_node((x, y))
            dist = np.linalg.norm(np.array((x, y)) - closest_node.position)
            if dist > self.delta:
                direction = (np.array((x, y)) - closest_node.position) / dist
                new_pos = closest_node.position + direction * self.delta
                x, y = new_pos[0], new_pos[1]
                # Check both endpoint validity and collision-free path
                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)
            else:
                # Check both endpoint validity and collision-free path
                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)

        # Log warning if we couldn't generate the requested number of nodes
        if len(self.nodes) < self.N_tn:
            import logging
            logging.warning(f'RRT sprawl: Only generated {len(self.nodes)}/{self.N_tn} nodes after {iteration} iterations (near walls/obstacles)')

    def get_closest_node(self, position : tuple[float,float]) -> Node:
        """Find the closest node in the tree to the given position."""
        position = np.array(position)
        positions = np.array([node.position for node in self.nodes])
        dists = np.linalg.norm(positions - position, axis=1)
        closest_index = np.argmin(dists)
        return self.nodes[closest_index]

    def is_collision_free(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> bool:
        """
        Check if the straight-line path between pos1 and pos2 is collision-free.
        Uses discrete sampling along the path.
        """
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        # Sample points along the line
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return self.occupancy_grid.is_valid(tuple(pos1), radius=self.robot_radius)

        # Sample at resolution of half the grid cell size for safety
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        num_samples = max(num_samples, 2)  # At least check start and end

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            if not self.occupancy_grid.is_valid((sample_pos[0], sample_pos[1]), radius=self.robot_radius):
                return False

        return True

    def prune(self) -> List[List[Node]]:
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

    def calculate_branch_information(self, path : List[Node], initial_particle_filter : ParticleFilterOptimized) -> float:
        """
        Calculate Branch Information (BI) for a path using Equation 19 from paper.

        BI(V_b) = Σ_{i=1}^{m} γ^{i-1} · I(v_{b,i})

        where:
        - V_b is a branch (path) in the RRT tree
        - v_{b,i} is the i-th vertex in the branch
        - I(v_{b,i}) is the mutual information at vertex v_{b,i}
        - γ is the discount factor (assigns higher weight to nearer future)
        - m is the number of vertices in the branch

        Parameters:
        -----------
        path : List[Node]
            Branch from root to leaf node
        initial_particle_filter : ParticleFilterOptimized
            Current particle filter state for entropy calculations

        Returns:
        --------
        BI : float
            Total Branch Information (discounted sum of mutual information)
        """
        path = path[1:] # Exclude starting position (root node)
        BI = 0.0  # Branch Information (Eq. 19)

        # Compute BI = Σ γ^{i-1} · I(v_i) where i starts from 1 in paper
        # After excluding root, enumerate starts from 0, so γ^i = γ^{i-1} in paper notation
        for i, node in enumerate(path):
            # Use cached entropy gain if available
            if node.entropy_gain != -np.inf:
                discounted_gain = (self.discount_factor ** i) * node.entropy_gain
                BI += discounted_gain
                continue

            position = node.position

            # Compute mutual information I(v_i) at this vertex
            # I(v_i) = H_k - E[H_{k+1}] (Equation 9 in paper)
            current_entropy = initial_particle_filter.get_entropy()

            # Determine number of measurement levels based on sensor model
            # Binary: 2 levels (0, 1), Discrete: N levels (0 to N-1)
            if hasattr(initial_particle_filter.sensor_model, 'num_levels'):
                num_measurements = initial_particle_filter.sensor_model.num_levels
            else:
                num_measurements = 2  # Binary sensor model

            # Compute expected entropy E[H_{k+1}] = Σ p(z) · H(z) (Eq. 11-12)
            expected_entropy = 0.0
            for measurement in range(num_measurements):
                # p(z | current state) from Equation 29
                probability_of_measurement = initial_particle_filter.predict_measurement_probability(position, measurement)
                # H_{k+1}(z) - hypothetical entropy after measurement z (Eq. 28)
                hypothetical_entropy = initial_particle_filter.compute_hypothetical_entropy(measurement, position)
                expected_entropy += probability_of_measurement * hypothetical_entropy

            # Mutual information at this vertex
            mutual_information = current_entropy - expected_entropy

            # Cache for potential reuse
            node.entropy_gain = mutual_information

            # Apply discount factor: γ^i · I(v_i)
            discounted_gain = (self.discount_factor ** i) * mutual_information
            BI += discounted_gain

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
    
    def get_next_move(self, start_pos : tuple[float,float], initial_particle_filter : ParticleFilterOptimized) -> tuple[float,float]:
        """
        Get the next move position by maximizing Branch Information (BI).

        Implements local planner from Section IV.B.1:
        - Select branch V_b* = argmax_{V_b} BI(V_b)  (Equation 20)
        - Move to first vertex of selected branch
        """
        # Clear previous tree
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()
        best_path = None
        best_BI = -np.inf  # BI* (Equation 20)

        for path in paths:
            # Compute Branch Information (Equation 19)
            BI = self.calculate_branch_information(path, initial_particle_filter)

            # Select branch with highest BI (Equation 20)
            if BI > best_BI:
                best_BI = BI
                best_path = path

        if best_path is not None and len(best_path) > 1:
            return tuple(best_path[1].position)  # Move to first vertex
        else:
            return start_pos  # No move possible

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