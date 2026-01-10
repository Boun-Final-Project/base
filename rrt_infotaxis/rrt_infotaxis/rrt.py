import numpy as np
from .particle_filter import ParticleFilter
from .occupancy_grid import OccupancyGridMap
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
                 max_depth : int = 3, discount_factor : float = 0.8, positive_weight : float = 0.5,
                 robot_radius : float = 0.35):
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

    def sprawl(self, start_pos: tuple[float,float]) -> None:
        root = Node(start_pos)
        self.nodes.append(root)
        while len(self.nodes) < self.N_tn:
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

    def calculate_entropy_gain(self, path : List[Node], initial_particle_filter : ParticleFilter) -> float:
        """
        Calculate the entropy gain along a given path.
        This is the J1 metric from the paper. Eq(30).
        Parameters:
        -----------
        path : list of tuples
            List of (x, y) positions along the path
        discount_factor : float
            Discount factor for entropy gain
        initial_particle_filter : ParticleFilter
            A particle filter to estimate entropy gain
            
        Returns:
        --------
        total_gain : float
            Total discounted entropy gain along the path
        """
        path = path[1:] # Exclude starting position
        I_total = 0.0
        for i, node in enumerate(path):
            if node.entropy_gain != -np.inf:
                discounted_gain = (self.discount_factor ** i) * node.entropy_gain
                I_total += discounted_gain
                continue
            position = node.position
            start_entropy = initial_particle_filter.get_entropy()
            # Two outcomes: detection (1) and no detection (0)
            expected_entropy = 0.0
            for measurement in [0, 1]:
                # Get probability of this measurement
                probability_of_measurement = initial_particle_filter.predict_measurement_probability(position, measurement)
                # Compute hypothetical entropy WITHOUT modifying filter state (Eq. 28-29 from paper)
                hypothetical_entropy = initial_particle_filter.compute_hypothetical_entropy(measurement, position)
                expected_entropy += probability_of_measurement * hypothetical_entropy
            information_gain = start_entropy - expected_entropy
            node.entropy_gain = information_gain
            discounted_gain = (self.discount_factor ** i) * information_gain
            I_total += discounted_gain
        return I_total

    def calculate_travel_cost(self, path : List[Node], initial_particle_filter : ParticleFilter) -> float:
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
    
    def get_next_move(self, start_pos : tuple[float,float], initial_particle_filter : ParticleFilter) -> tuple[float,float]:
        """Get the next move position based on utility maximization."""
        # Clear previous tree
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()
        best_path = None
        best_utility = -np.inf
        for path in paths:
            I_gain = self.calculate_entropy_gain(path, initial_particle_filter)
            travel_cost = self.calculate_travel_cost(path, initial_particle_filter)
            utility = I_gain * self.positive_weight - travel_cost * (1 - self.positive_weight)
            if utility > best_utility:
                best_utility = utility
                best_path = path
        if best_path is not None and len(best_path) > 1:
            return tuple(best_path[1].position)  # Next move
        else:
            return start_pos  # No move possible

    def get_next_move_debug(self, start_pos : tuple[float,float], initial_particle_filter : ParticleFilter) -> dict:
        """
        Get next move with detailed debug information for visualization.

        Parameters:
        -----------
        start_pos : tuple[float, float]
            Current position of the agent
        initial_particle_filter : ParticleFilter
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
            # FALLBACK BEHAVIOR: No complete paths found
            # First, try to use any node from the tree (even if not at max depth)
            if len(self.nodes) > 1:  # More than just the root node
                # Pick a random non-root node
                available_nodes = [node for node in self.nodes if node.parent is not None]
                if available_nodes:
                    random_node = np.random.choice(available_nodes)
                    next_position = tuple(random_node.position)

                    return {
                        'next_position': next_position,
                        'best_path': [Node(start_pos), random_node],
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
                        'error': 'No complete paths - using random node from tree'
                    }

            # GLOBAL MODE: No nodes found at all, explore randomly
            # Sample a random valid position in exploration radius
            for _ in range(100):  # Try up to 100 times
                r = self.R_range * np.sqrt(np.random.random())
                theta = 2 * np.pi * np.random.random()
                x = start_pos[0] + r * np.cos(theta)
                y = start_pos[1] + r * np.sin(theta)

                if self.is_collision_free(start_pos, (x, y)):
                    next_position = (x, y)
                    return {
                        'next_position': next_position,
                        'best_path': [Node(start_pos), Node(next_position)],
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
                        'error': 'Global exploration mode - random valid position'
                    }

            # If all else fails, stay put
            return {
                'next_position': start_pos,
                'best_path': [Node(start_pos)],
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
                'error': 'Completely stuck - no valid moves found'
            }

        # Evaluate all paths
        all_utilities = []
        all_entropy_gains = []
        all_travel_costs = []

        best_path = None
        best_utility = -np.inf
        best_entropy_gain = 0.0
        best_travel_cost = 0.0

        for path in paths:
            I_gain = self.calculate_entropy_gain(path, initial_particle_filter)
            travel_cost = self.calculate_travel_cost(path, initial_particle_filter)
            utility = I_gain * self.positive_weight - travel_cost * (1 - self.positive_weight)

            all_entropy_gains.append(I_gain)
            all_travel_costs.append(travel_cost)
            all_utilities.append(utility)

            if utility > best_utility:
                best_utility = utility
                best_path = path
                best_entropy_gain = I_gain
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
            'best_utility': best_utility,
            'best_entropy_gain': best_entropy_gain,
            'best_travel_cost': best_travel_cost,
            'all_paths': all_paths_tuples,
            'all_utilities': all_utilities,
            'all_entropy_gains': all_entropy_gains,
            'all_travel_costs': all_travel_costs,
            'tree_nodes': self.nodes.copy(),  # Copy to avoid modification
            'num_branches': len(paths),
            'estimated_source': estimated_source,
            'start_position': start_pos,
            'sampling_radius': self.R_range,
            'max_depth': self.max_depth,
            'num_tree_nodes': len(self.nodes)
        }