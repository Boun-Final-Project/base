"""
Directional Planner: Simple 4-directional planning with fixed depth.
Plans 4 linear paths: 3 forward, 3 backward, 3 right, 3 left.
Each step has random distance between 0.5 and 1.0m.
"""

import numpy as np


class Node:
    """Node representing a position in a planned path."""

    def __init__(self, position, parent=None):
        """
        Parameters:
        -----------
        position : tuple or list
            (x, y) position of the node
        parent : Node, optional
            Parent node in the path
        """
        self.position = np.array(position)
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.information_gain = -np.inf


class DirectionalPlanner:
    """Simple directional planner with 4 linear paths (forward, backward, left, right)."""

    def __init__(self, occupancy_grid, depth=3, delta_min=0.5, delta_max=1.0,
                 discount_factor=0.8, positive_weight=0.5, robot_radius=0.35,
                 visited_positions=None, current_step=0, penalty_radius=1.0):
        """
        Parameters:
        -----------
        occupancy_grid : OccupancyGrid
            Grid for collision checking
        depth : int
            Planning depth (number of steps per path)
        delta_min : float
            Minimum step distance (meters)
        delta_max : float
            Maximum step distance (meters)
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
        self.depth = depth
        self.delta_min = delta_min
        self.delta_max = delta_max
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

        self.paths = []
        self.all_nodes = []  # For visualization (all nodes from all paths)

    def plan_paths(self, start_pos):
        """Generate 4 linear paths: forward, backward, right, left.

        Parameters:
        -----------
        start_pos : tuple
            (x, y) starting position
        """
        self.paths = []
        self.all_nodes = []

        # Define 4 directions: Forward, Backward, Right, Left
        directions = [
            (1, 0),   # Forward (positive x)
            (-1, 0),  # Backward (negative x)
            (0, 1),   # Right (positive y)
            (0, -1),  # Left (negative y)
        ]
        direction_names = ['Forward', 'Backward', 'Right', 'Left']

        for dir_idx, (dx, dy) in enumerate(directions):
            path = [Node(start_pos)]
            self.all_nodes.append(path[0])
            current_pos = np.array(start_pos)

            # Generate self.depth steps in this direction
            for step in range(self.depth):
                # Random step size between delta_min and delta_max
                step_size = self.delta_min + np.random.random() * (self.delta_max - self.delta_min)

                # Calculate new position
                new_x = current_pos[0] + dx * step_size
                new_y = current_pos[1] + dy * step_size
                new_pos = (new_x, new_y)

                # Check collision
                if self.is_collision_free(tuple(current_pos), new_pos):
                    new_node = Node(new_pos, path[-1])
                    path.append(new_node)
                    self.all_nodes.append(new_node)
                    current_pos = np.array(new_pos)
                else:
                    # Path blocked, stop extending this path
                    break

            # Only add path if it has at least depth steps (full depth reached)
            if len(path) == self.depth + 1:
                self.paths.append(path)

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
                # Update with future time step in planning mode (no resampling, ensures deterministic entropy)
                pf_copy.update(measurement, position, time_step=future_time_step, is_planning=True)
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

    def is_in_penalty_zone(self, path):
        """Check if the next step (first move) of the path is in exploration penalty zone.

        Time-dependent penalty: decreases as steps increase since last visit.

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
        to [0, 1] range before calculating utility.

        Parameters:
        -----------
        start_pos : tuple
            Current robot position (x, y)
        initial_particle_filter : ParticleFilter
            Current particle filter state

        Returns:
        --------
        debug_info : dict
            Dictionary with planning information and pruned paths for visualization
        """
        self.plan_paths(start_pos)
        paths = self.paths

        if not paths:
            return {
                'next_position': start_pos,
                'best_utility': -np.inf,
                'best_information_gain': 0.0,
                'best_travel_cost': np.inf,
                'all_paths': [],
                'rrt_pruned_paths': [],
            }

        # FIRST PASS: Collect all raw values (information gain and travel cost)
        raw_info_gains = []
        raw_travel_costs = []
        path_metadata = []

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
            J2_normalized = self._normalize_value(
                metadata['travel_cost'],
                min_travel_cost,
                max_travel_cost
            )

            # Calculate utility with normalized values
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
            'norm_info_gain_range': (min_info_gain, max_info_gain),
            'norm_travel_cost_range': (min_travel_cost, max_travel_cost),
            'rrt_nodes': self.all_nodes,  # All nodes from all paths
            'rrt_pruned_paths': paths,  # Pruned paths (all generated paths are valid)
        }
