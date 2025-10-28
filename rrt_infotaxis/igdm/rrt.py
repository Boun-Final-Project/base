"""
Rapidly-exploring Random Tree with Infotaxis (RRT-Infotaxis) for motion planning with information gain.
"""

import numpy as np


class Node:
    """RRT node representing a position in the tree."""

    def __init__(self, position, parent=None):
        """
        Parameters:
        -----------
        position : tuple or list
            (x, y) position of the node
        parent : Node, optional
            Parent node in the tree
        """
        self.position = np.array(position)
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.information_gain = -np.inf


class RRTInfotaxis:
    """Rapidly-exploring Random Tree with Information-theoretic Path Planning (Infotaxis)."""

    def __init__(self, occupancy_grid, N_tn, R_range, delta, max_depth=3,
                 discount_factor=0.8, positive_weight=0.5, robot_radius=0.35):
        """
        Parameters:
        -----------
        occupancy_grid : OccupancyGrid
            Grid for collision checking
        N_tn : int
            Target number of nodes
        R_range : float
            Sampling range radius
        delta : float
            Maximum edge length
        max_depth : int
            Maximum tree depth (planning horizon)
        discount_factor : float
            Discount factor for future information gain
        positive_weight : float
            Weight for information gain (vs. travel cost)
        robot_radius : float
            Robot radius for collision checking
        """
        self.occupancy_grid = occupancy_grid
        self.N_tn = N_tn
        self.R_range = R_range
        self.nodes = []
        self.delta = delta
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.positive_weight = positive_weight
        self.robot_radius = robot_radius

    def sprawl(self, start_pos):
        """Grow the RRT from start position.

        Parameters:
        -----------
        start_pos : tuple
            (x, y) starting position
        """
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

                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)
            else:
                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)

    def get_closest_node(self, position):
        """Find closest node to position.

        Parameters:
        -----------
        position : tuple
            (x, y) query position

        Returns:
        --------
        node : Node
            Closest node in tree
        """
        position = np.array(position)
        positions = np.array([node.position for node in self.nodes])
        dists = np.linalg.norm(positions - position, axis=1)
        closest_index = np.argmin(dists)
        return self.nodes[closest_index]

    def is_collision_free(self, pos1, pos2):
        """Check if path between two positions is collision-free.

        Parameters:
        -----------
        pos1 : tuple
            Start position (x, y)
        pos2 : tuple
            End position (x, y)

        Returns:
        --------
        collision_free : bool
            True if path is collision-free
        """
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return self.occupancy_grid.is_valid(position=tuple(pos1), radius=self.robot_radius)

        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        num_samples = max(num_samples, 2)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = pos1 + t * (pos2 - pos1)
            if not self.occupancy_grid.is_valid(position=(sample_pos[0], sample_pos[1]), radius=self.robot_radius):
                return False

        return True

    def prune(self):
        """Extract paths from root to all leaf nodes at max depth.

        Returns:
        --------
        paths : list of list
            List of paths (each path is list of nodes)
        """
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

    def calculate_information_gain(self, path, initial_particle_filter):
        """Calculate total discounted information gain along path.

        Parameters:
        -----------
        path : list
            List of nodes in path
        initial_particle_filter : ParticleFilter
            Particle filter for entropy computation

        Returns:
        --------
        information_gain : float
            Total discounted information gain
        """
        path = path[1:]  # Exclude root
        I_total = 0.0

        for i, node in enumerate(path):
            if node.information_gain != -np.inf:
                discounted_gain = (self.discount_factor ** i) * node.information_gain
                I_total += discounted_gain
                continue

            position = node.position
            start_entropy = initial_particle_filter.get_entropy()
            expected_entropy = 0.0

            # Predict measurement at future time step (current + i+1)
            future_time_step = initial_particle_filter.current_step + i + 1

            for measurement in [0, 1]:
                pf_copy = initial_particle_filter.copy()
                # Predict probability using future time step
                probability_of_measurement = pf_copy.predict_measurement_probability(position, measurement,
                                                                                      time_step=future_time_step)
                # Update with future time step, skip MCMC for speed during planning
                pf_copy.update(measurement, position, time_step=future_time_step, skip_mcmc=False)
                new_entropy = pf_copy.get_entropy()
                expected_entropy += probability_of_measurement * new_entropy

            information_gain = start_entropy - expected_entropy
            node.information_gain = information_gain
            discounted_gain = (self.discount_factor ** i) * information_gain
            I_total += discounted_gain

        return I_total

    def calculate_travel_cost(self, path, initial_particle_filter):
        """Calculate travel cost of a path.

        Parameters:
        -----------
        path : list
            List of nodes in path
        initial_particle_filter : ParticleFilter
            Used to get estimated source position

        Returns:
        --------
        cost : float
            Path length + distance to estimated source
        """
        estimation, _ = initial_particle_filter.get_estimate()
        target_pos = np.array([estimation["x"], estimation["y"]])
        path_array = np.array([node.position for node in path])
        path_length = np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
        end_pos = path_array[-1]
        distance_to_target = np.linalg.norm(end_pos - target_pos)
        total_cost = path_length + distance_to_target
        return total_cost

    def get_next_move_debug(self, start_pos, initial_particle_filter):
        """Plan next move and return debug information.

        Parameters:
        -----------
        start_pos : tuple
            Current robot position (x, y)
        initial_particle_filter : ParticleFilter
            Current particle filter state

        Returns:
        --------
        debug_info : dict
            Dictionary with keys:
            - 'next_position': Next position to move to
            - 'best_utility': Best utility value
            - 'best_information_gain': Information gain of best path
            - 'best_travel_cost': Travel cost of best path
            - 'all_paths': All explored paths
        """
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()

        if not paths:
            return {
                'next_position': start_pos,
                'best_utility': -np.inf,
                'best_information_gain': 0.0,
                'best_travel_cost': np.inf,
                'all_paths': [],
            }

        all_utilities = []
        all_information_gains = []
        all_travel_costs = []
        best_path = None
        best_utility = -np.inf
        best_information_gain = 0.0
        best_travel_cost = 0.0

        for path in paths:
            I_gain = self.calculate_information_gain(path, initial_particle_filter)
            travel_cost = self.calculate_travel_cost(path, initial_particle_filter)
            utility = I_gain * self.positive_weight - travel_cost * (1 - self.positive_weight)

            all_utilities.append(utility)
            all_information_gains.append(I_gain)
            all_travel_costs.append(travel_cost)

            if utility > best_utility:
                best_utility = utility
                best_path = path
                best_information_gain = I_gain
                best_travel_cost = travel_cost

        next_pos = tuple(best_path[1].position) if best_path and len(best_path) > 1 else start_pos

        return {
            'next_position': next_pos,
            'best_utility': best_utility,
            'best_information_gain': best_information_gain,
            'best_travel_cost': best_travel_cost,
            'all_paths': paths,
        }
