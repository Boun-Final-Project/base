"""
Rapidly-exploring Random Tree with Infotaxis (RRT-Infotaxis) for motion planning with information gain.

Updated version with:
- 4 guaranteed initial nodes (forward, behind, right, left) for better directional coverage
- Improved pruning that preserves shallow branches and prunes deep branches to max_depth
- Exploration penalty to discourage revisiting recent areas
- Min-max normalization of J1 (information gain) and J2 (travel cost)
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


class RRTInfotaxisUpdated:
    """Rapidly-exploring Random Tree with Information-theoretic Path Planning (Infotaxis) - Updated version."""

    def __init__(self, occupancy_grid, N_tn, R_range, delta, max_depth=3,
                 discount_factor=0.8, positive_weight=0.5, robot_radius=0.35,
                 visited_positions=None, current_step=0, penalty_radius=1.0):
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
        visited_positions : list, optional
            List of (position, step) tuples for exploration penalty
        current_step : int, optional
            Current step number for penalty calculation
        penalty_radius : float, optional
            Radius around visited positions to apply penalty (default 1.0m)
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
        self.visited_positions = visited_positions if visited_positions is not None else []
        self.current_step = current_step
        self.penalty_radius = penalty_radius

        # Penalty parameters
        self.MAX_PENALTY_STEPS = 5
        self.INITIAL_PENALTY = 32
        self.PENALTY_DECAY_RATE = 2.0

    def sprawl(self, start_pos):
        """Grow the RRT from start position with 4 guaranteed initial nodes.

        The 4 initial nodes are placed in cardinal directions (120-degree separation):
        - 0° (positive Y)
        - 120° (negative X, negative Y)
        - 240° (positive X, negative Y)

        Parameters:
        -----------
        start_pos : tuple
            (x, y) starting position
        """
        root = Node(start_pos)
        self.nodes.append(root)

        # Define 3 evenly-spaced directions (120 degrees apart)
        cardinal_directions = [
            (0.0, 1.0),                      # 0° (forward in Y)
            (np.sqrt(3) / 2, -1 / 2),       # 120°
            (-np.sqrt(3) / 2, -1 / 2)       # 240°
        ]

        # Add initial nodes in cardinal directions
        for dx, dy in cardinal_directions:
            if len(self.nodes) >= self.N_tn:
                break

            # Random distance between delta/2 and delta
            random_distance = np.random.uniform(self.delta / 2, self.delta)

            # Create position at random distance in the cardinal direction
            x = start_pos[0] + dx * random_distance
            y = start_pos[1] + dy * random_distance

            # Check collision-free path from root to new position
            if self.is_collision_free(start_pos, (x, y)):
                new_node = Node((x, y), root)
                self.nodes.append(new_node)

        # Continue with random sampling
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
        """Extract paths from root, preserving shallow branches and pruning deep ones to max_depth.

        This improved pruning strategy:
        - Preserves paths with depth < max_depth as-is
        - Prunes paths with depth > max_depth down to max_depth
        - This ensures we don't lose information from branches that haven't reached max_depth

        Returns:
        --------
        paths : list of list
            List of paths (each path is list of nodes)
        """
        paths = []

        # Get all leaf nodes (nodes with no children)
        all_nodes = set(self.nodes)
        non_leaf_nodes = set()
        for node in self.nodes:
            if node.parent is not None:
                non_leaf_nodes.add(node.parent)

        leaf_nodes = all_nodes - non_leaf_nodes

        # For each leaf node, extract path to root and truncate at max_depth if needed
        for leaf_node in leaf_nodes:
            path = []
            current = leaf_node
            while current is not None:
                path.append(current)
                current = current.parent
            path = path[::-1]  # Reverse to get root-to-leaf order

            # If path is deeper than max_depth, truncate it
            if len(path) > self.max_depth + 1:  # +1 because path includes root
                path = path[:self.max_depth + 1]

            # Only add paths that have at least 2 nodes (root + at least one other)
            if len(path) > 1:
                paths.append(path)

        # Also add paths to all nodes at exactly max_depth depth
        # (in case they're not leaf nodes but have children deeper)
        max_depth_nodes = [node for node in self.nodes if node.depth == self.max_depth and node not in leaf_nodes]
        for node in max_depth_nodes:
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = current.parent
            path = path[::-1]
            if len(path) > 1 and path not in paths:
                paths.append(path)

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

        # Get measurement levels dynamically from sensor (handles both binary and discrete sensors)
        measurement_levels = initial_particle_filter.sensor_model.get_measurement_levels()

        for i, node in enumerate(path):
            if node.information_gain != -np.inf:
                discounted_gain = (self.discount_factor ** i) * node.information_gain
                I_total += discounted_gain
                continue

            position = node.position
            start_entropy = initial_particle_filter.get_entropy()

            # OPTIMIZED: Compute expected entropy for ALL measurement levels at once (vectorized)
            # This avoids copying the particle filter 5 times per node
            expected_entropy = self._compute_expected_entropy_vectorized(
                position, initial_particle_filter, measurement_levels
            )

            information_gain = start_entropy - expected_entropy
            node.information_gain = information_gain
            discounted_gain = (self.discount_factor ** i) * information_gain
            I_total += discounted_gain

        return I_total

    def _compute_expected_entropy_vectorized(self, position, particle_filter, measurement_levels):
        """
        Vectorized computation of expected entropy E[H(X|Z)] for all measurement levels.

        This is MUCH faster than the loop version because:
        - No particle filter copying (5x speedup)
        - Single distance map computation (reused for all measurements)
        - Vectorized weight updates for all measurement scenarios

        Parameters:
        -----------
        position : np.ndarray
            Position to evaluate (x, y)
        particle_filter : ParticleFilter
            Current particle filter state
        measurement_levels : list
            List of possible measurements (e.g., [0, 1, 2, 3, 4])

        Returns:
        --------
        expected_entropy : float
            E[H(X|Z)] = sum_z P(z) * H(X|Z=z)
        """
        # 1. Get predicted concentrations for all particles at this position (ONCE)
        predicted_concs = particle_filter._compute_concentrations(
            particle_filter.particles, position
        )

        # 2. Compute measurement probabilities for all levels and all particles
        # Shape: (num_levels, num_particles)
        current_weights = particle_filter.weights  # (N,)
        expected_entropy = 0.0

        for measurement in measurement_levels:
            # Compute likelihoods for this measurement for ALL particles (VECTORIZED)
            if particle_filter.is_discrete:
                # Use vectorized method instead of list comprehension for 100x speedup
                likelihoods = particle_filter.sensor_model.probability_discrete_vec(
                    int(measurement), predicted_concs
                )
            else:
                # For continuous sensor (if needed)
                likelihoods = particle_filter.sensor_model.probability_continuous_vec(
                    measurement, predicted_concs
                )

            # Compute posterior weights: w_new = w_old * P(z|particle)
            posterior_weights = current_weights * likelihoods
            weight_sum = np.sum(posterior_weights)

            if weight_sum > 1e-15:
                # P(z) = sum_i P(z|x_i) * P(x_i) = weight_sum (before normalization)
                prob_measurement = weight_sum

                # Normalize to get P(x|z)
                posterior_weights = posterior_weights / weight_sum

                # Compute H(X|Z=z) - entropy given this measurement
                weights_safe = posterior_weights[posterior_weights > 1e-15]
                entropy_given_z = -np.sum(weights_safe * np.log(weights_safe))

                # Add to expected entropy: P(z) * H(X|Z=z)
                expected_entropy += prob_measurement * entropy_given_z

        return expected_entropy

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

    def is_in_penalty_zone(self, path):
        """Check if the next step (first move) of the path is in exploration penalty zone.

        Checks path[1] which is where the robot will move in the next step, not the endpoint.

        Time-dependent penalty: decreases as steps increase since last visit.
        - 1 step since visit: 2^5 = 32 (multiply by 1/32)
        - 2 steps since visit: 2^4 = 16 (multiply by 1/16)
        - 3 steps since visit: 2^3 = 8 (multiply by 1/8)
        - 4 steps since visit: 2^2 = 4 (multiply by 1/4)
        - 5 steps since visit: 2^1 = 2 (multiply by 1/2)

        Parameters:
        -----------
        path : list
            List of nodes in path (path[0] is current position, path[1] is next step)

        Returns:
        --------
        in_penalty : bool
            True if next step is in penalty zone
        penalty_factor : float
            Factor to multiply information gain by (1.0 if no penalty, otherwise 1/2^k)
        steps_since : int
            Number of steps since the penalized visit (0 if no penalty)
        penalty_info : dict
            Dictionary with penalty details: visited_pos, visited_step, distance
        """
        penalty_factor = 1.0
        steps_since = 0
        penalty_info = {'visited_pos': None, 'visited_step': None, 'distance': 0.0}

        # Only apply penalty if we have visited positions
        if not self.visited_positions:
            return False, penalty_factor, steps_since, penalty_info

        # Check the next step (path[1]) - where robot will move in this step
        if path and len(path) > 1:
            next_step_pos = path[1].position

            # Check penalty from recent visits
            for visited_pos, visited_step in self.visited_positions:
                steps_since_visit = self.current_step - visited_step

                # Time-dependent penalty for visits within MAX_PENALTY_STEPS
                if 1 <= steps_since_visit <= self.MAX_PENALTY_STEPS:
                    dist_to_visited = np.linalg.norm(next_step_pos - np.array(visited_pos))
                    if dist_to_visited < self.penalty_radius:
                        # Calculate penalty using decay rate
                        penalty_factor = 1.0 / (self.INITIAL_PENALTY / (self.PENALTY_DECAY_RATE ** (steps_since_visit - 1)))
                        steps_since = steps_since_visit
                        penalty_info = {
                            'visited_pos': visited_pos,
                            'visited_step': visited_step,
                            'distance': dist_to_visited
                        }
                        return True, penalty_factor, steps_since, penalty_info

        return False, penalty_factor, steps_since, penalty_info

    def _normalize_value(self, value, min_val, max_val):
        """Normalize value to [0, 1] range using min-max normalization.

        Parameters:
        -----------
        value : float
            Value to normalize
        min_val : float
            Minimum value in the range
        max_val : float
            Maximum value in the range

        Returns:
        --------
        normalized : float
            Value normalized to [0, 1]
        """
        if max_val == min_val:
            return 0.5  # If all values are the same, return 0.5
        return (value - min_val) / (max_val - min_val)

    def get_next_move_debug(self, start_pos, initial_particle_filter):
        """Plan next move and return debug information.

        Normalizes both J1 (information gain) and J2 (travel cost) independently
        to [0, 1] range before calculating utility. This ensures equal contribution
        regardless of their absolute scales.

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
            - 'best_utility': Best utility value (using normalized J1 and J2)
            - 'best_information_gain': Normalized information gain of best path
            - 'best_travel_cost': Normalized travel cost of best path
            - 'best_information_gain_original': Original (non-normalized) information gain
            - 'best_travel_cost_original': Original (non-normalized) travel cost
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

        # FIRST PASS: Collect all raw values (information gain and travel cost)
        raw_info_gains = []  # Before normalization
        raw_travel_costs = []
        path_metadata = []  # Store penalty info for each path

        for path in paths:
            I_gain = self.calculate_information_gain(path, initial_particle_filter)
            travel_cost = self.calculate_travel_cost(path, initial_particle_filter)

            # Apply exploration penalty if path is in penalty zone
            penalty_applied, penalty_factor, steps_since, penalty_info = self.is_in_penalty_zone(path)

            # Apply penalty to information gain
            if I_gain >= 0:
                I_gain_penalized = I_gain * penalty_factor
            else:
                I_gain_penalized = I_gain / penalty_factor

            raw_info_gains.append(I_gain_penalized)
            raw_travel_costs.append(travel_cost)
            path_metadata.append({
                'path': path,
                'I_gain_original': I_gain,
                'I_gain_penalized': I_gain_penalized,
                'travel_cost': travel_cost,
                'penalty_applied': penalty_applied,
                'penalty_factor': penalty_factor,
                'penalty_info': penalty_info
            })

        # Find min/max for normalization
        min_info_gain = min(raw_info_gains)
        max_info_gain = max(raw_info_gains)
        min_travel_cost = min(raw_travel_costs)
        max_travel_cost = max(raw_travel_costs)

        # SECOND PASS: Normalize J1 and J2, calculate utilities
        all_utilities = []
        all_information_gains_normalized = []
        all_travel_costs_normalized = []
        best_utility = -np.inf
        best_path = None
        best_idx = -1
        paths_with_penalties = 0

        for idx, metadata in enumerate(path_metadata):
            # Normalize J1 (information gain) to [0, 1]
            J1_normalized = self._normalize_value(
                metadata['I_gain_penalized'],
                min_info_gain,
                max_info_gain
            )

            # Normalize J2 (travel cost) to [0, 1]
            # Note: travel cost should be minimized, so we invert it after normalization
            J2_normalized = self._normalize_value(
                metadata['travel_cost'],
                min_travel_cost,
                max_travel_cost
            )

            # Calculate utility with normalized values
            # utility = J1 * positive_weight - J2 * (1 - positive_weight)
            # (inverted J2 because we want to minimize cost)
            utility = J1_normalized * self.positive_weight - J2_normalized * (1 - self.positive_weight)

            all_utilities.append(utility)
            all_information_gains_normalized.append(J1_normalized)
            all_travel_costs_normalized.append(J2_normalized)

            if metadata['penalty_applied']:
                paths_with_penalties += 1

            if utility > best_utility:
                best_utility = utility
                best_path = metadata['path']
                best_idx = idx

        next_pos = tuple(best_path[1].position) if best_path and len(best_path) > 1 else start_pos

        # Get best path metadata for return value
        best_metadata = path_metadata[best_idx]

        return {
            'next_position': next_pos,
            'best_utility': best_utility,
            'best_information_gain': all_information_gains_normalized[best_idx],  # Normalized
            'best_information_gain_original': best_metadata['I_gain_original'],  # Original (before penalty)
            'best_information_gain_penalized': best_metadata['I_gain_penalized'],  # With penalty but not normalized
            'best_travel_cost': all_travel_costs_normalized[best_idx],  # Normalized
            'best_travel_cost_original': best_metadata['travel_cost'],  # Original
            'best_penalty_applied': best_metadata['penalty_applied'],
            'best_penalty_factor': best_metadata['penalty_factor'],
            'best_penalty_info': best_metadata['penalty_info'],
            'paths_with_penalties': paths_with_penalties,
            'total_paths': len(paths),
            'all_paths': paths,
            'norm_info_gain_range': (min_info_gain, max_info_gain),  # For debugging
            'norm_travel_cost_range': (min_travel_cost, max_travel_cost),  # For debugging
            'rrt_nodes': self.nodes,  # All RRT tree nodes
            'rrt_pruned_paths': paths,  # Only pruned paths reaching max_depth for visualization
            'all_utilities': all_utilities,  # Utilities for all paths
            'all_information_gains_normalized': all_information_gains_normalized,  # Normalized J1 for all paths
            'all_travel_costs_normalized': all_travel_costs_normalized,  # Normalized J2 for all paths
            'path_metadata': path_metadata,  # Metadata for all paths including penalty info
        }
