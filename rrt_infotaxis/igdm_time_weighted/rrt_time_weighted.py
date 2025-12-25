"""
Rapidly-exploring Random Tree with Infotaxis (RRT-Infotaxis) for motion planning with information gain.

Configurable weight modes for exploration-exploitation trade-off:

1. "decay" (default): Linear decay from exploration to exploitation
   - w1(t) = max(0.4, 0.6 - 0.005*t)
   - At t=0: 0.6*J1 - 0.4*J2 (more explorative)
   - At t=40+: 0.4*J1 - 0.6*J2 (more exploitative, stays here)

2. "constant": Fixed weights
   - w1 = initial_weight (default 0.6), w2 = 1 - w1

3. "periodic": Oscillating exploration-exploitation
   - amplitude = (initial_weight - min_weight) / 2
   - center = min_weight + amplitude
   - w1(t) = amplitude * cos((π/30) * t) + center
   - Period = 60 steps, w1 oscillates between min_weight and initial_weight
   - Encourages periodic switching between exploration and exploitation

Updated version with:
- 4 guaranteed initial nodes (forward, behind, right, left)
- Improved pruning that preserves shallow branches and prunes deep branches to max_depth
- Configurable weight modes for exploration-exploitation trade-off
"""

import numpy as np
import heapq


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


class RRTInfotaxisTimeWeighted:
    """Rapidly-exploring Random Tree with Information-theoretic Path Planning (Infotaxis) - Time-weighted version."""

    def __init__(self, occupancy_grid, N_tn, R_range, delta, max_depth=3,
                 discount_factor=0.8, initial_weight=0.6, min_weight=0.4,
                 weight_decay_rate=0.005, weight_mode="decay", robot_radius=0.15,
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
        initial_weight : float
            Initial weight for J1 at t=0 (default 0.6)
        min_weight : float
            Minimum weight for J1, clamps at this value (default 0.4)
        weight_decay_rate : float
            Rate at which w1 decreases per step (default 0.005), used in "decay" mode
        weight_mode : str
            Weight calculation mode. Options:
            - "decay": w1(t) = max(min_weight, initial_weight - weight_decay_rate * t)
                       Transitions from exploration to exploitation over time.
            - "constant": w1 = initial_weight (fixed weight)
            - "periodic": w1(t) = amplitude * cos((π/30) * t) + center
                          where amplitude = (initial_weight - min_weight) / 2
                          Oscillates between min_weight and initial_weight with period 60.
        robot_radius : float
            Robot radius for collision checking
        visited_positions : list, optional
            List of (position, step) tuples for exploration penalty
        current_step : int, optional
            Current step number for penalty and weight calculation
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

        # Time-dependent weight parameters
        self.initial_weight = initial_weight  # w1 at t=0
        self.min_weight = min_weight  # minimum w1 (clamp value)
        self.weight_decay_rate = weight_decay_rate  # how fast w1 decreases
        self.weight_mode = weight_mode  # "decay", "constant", or "periodic"

        self.robot_radius = robot_radius
        self.visited_positions = visited_positions if visited_positions is not None else []
        self.current_step = current_step
        self.penalty_radius = penalty_radius

        # Penalty parameters
        self.MAX_PENALTY_STEPS = 5
        self.INITIAL_PENALTY = 32
        self.PENALTY_DECAY_RATE = 2

        # Dijkstra cache for obstacle-aware distance calculations
        self._dijkstra_cache = {}

    def _dijkstra_full(self, start_gx, start_gy):
        """Compute Dijkstra distances from a grid cell to all other cells.

        Parameters:
        -----------
        start_gx, start_gy : int
            Starting grid cell coordinates

        Returns:
        --------
        distances : np.ndarray
            Distance grid where distances[gy, gx] is distance to that cell
        """
        grid = self.occupancy_grid
        distances = np.full((grid.grid_height, grid.grid_width), np.inf)
        distances[start_gy, start_gx] = 0.0

        pq = [(0.0, start_gx, start_gy)]
        visited = set()

        while pq:
            current_dist, gx, gy = heapq.heappop(pq)

            if (gx, gy) in visited:
                continue
            visited.add((gx, gy))

            # 8-connected neighbors with appropriate costs
            for dx, dy, cost in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                                  (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
                                  (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]:
                nx, ny = gx + dx, gy + dy
                if not grid.is_valid(gx=nx, gy=ny) or (nx, ny) in visited:
                    continue

                edge_cost = cost * grid.resolution
                new_dist = current_dist + edge_cost
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

        return distances

    def _dijkstra_distance(self, start_pos, target_pos):
        """Compute obstacle-aware Dijkstra distance between two positions.

        Parameters:
        -----------
        start_pos : tuple
            (x, y) starting position
        target_pos : tuple
            (x, y) target position

        Returns:
        --------
        distance : float
            Obstacle-aware shortest path distance (inf if unreachable)
        """
        if self.occupancy_grid is None:
            return np.linalg.norm(np.array(target_pos) - np.array(start_pos))

        start_gx, start_gy = self.occupancy_grid.world_to_grid(start_pos[0], start_pos[1])
        cache_key = (start_gx, start_gy)

        if cache_key not in self._dijkstra_cache:
            distances = self._dijkstra_full(start_gx, start_gy)
            self._dijkstra_cache[cache_key] = distances

        distances = self._dijkstra_cache[cache_key]
        target_gx, target_gy = self.occupancy_grid.world_to_grid(target_pos[0], target_pos[1])

        # Check if target is valid
        if not self.occupancy_grid.is_valid(gx=target_gx, gy=target_gy):
            return np.inf

        return distances[target_gy, target_gx]

    def get_current_weight(self):
        """Calculate the current weight for J1 based on current step and weight_mode.

        Weight modes:
        - "decay": w1(t) = max(min_weight, initial_weight - weight_decay_rate * t)
        - "constant": w1 = initial_weight
        - "periodic": w1(t) = amplitude * cos((π/30) * t) + center
                      where amplitude = (initial_weight - min_weight) / 2

        Returns:
        --------
        w1, w2 : tuple of float
            Current weights for J1 and J2
        """
        if self.weight_mode == "constant":
            w1 = self.initial_weight
        elif self.weight_mode == "periodic":
            amplitude = (self.initial_weight - self.min_weight) / 2
            center = self.min_weight + amplitude
            w1 = amplitude * np.cos((np.pi / 30) * self.current_step) + center
        else:  # "decay" (default)
            w1 = max(self.min_weight, self.initial_weight - self.weight_decay_rate * self.current_step)

        w2 = 1 - w1
        return w1, w2

    def sprawl(self, start_pos):
        """Grow the RRT from start position with 4 guaranteed initial nodes.

        The 4 initial nodes are placed in cardinal directions:
        - Forward (positive X direction)
        - Behind (negative X direction)
        - Right (positive Y direction)
        - Left (negative Y direction)

        Parameters:
        -----------
        start_pos : tuple
            (x, y) starting position
        """
        root = Node(start_pos)
        self.nodes.append(root)

        cardinal_directions = [
            (0.0, 1.0),
            ( np.sqrt(3) / 2, -1 / 2),
            (-np.sqrt(3) / 2, -1 / 2)
        ]

        # Add 4 initial nodes in cardinal directions
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
            expected_entropy = 0.0

            # Predict measurement at future time step (current + i+1)
            future_time_step = initial_particle_filter.current_step + i + 1

            for measurement in measurement_levels:
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
        """Calculate travel cost of a path using obstacle-aware Dijkstra distance.

        Parameters:
        -----------
        path : list
            List of nodes in path
        initial_particle_filter : ParticleFilter
            Used to get estimated source position

        Returns:
        --------
        cost : float
            Path length + Dijkstra distance to estimated source (obstacle-aware)
        """
        estimation, _ = initial_particle_filter.get_estimate()
        target_pos = (estimation["x"], estimation["y"])
        path_array = np.array([node.position for node in path])
        path_length = np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
        end_pos = tuple(path_array[-1])

        # Use Dijkstra distance (obstacle-aware) instead of Euclidean distance
        distance_to_target = self._dijkstra_distance(end_pos, target_pos)

        # If target is unreachable (inf), use a large but finite cost
        if np.isinf(distance_to_target):
            distance_to_target = 1000.0  # Large penalty for unreachable targets

        total_cost = path_length + distance_to_target
        return total_cost

    def is_in_penalty_zone(self, path):
        """Check if the next step (first move) of the path is in exploration penalty zone.

        Checks path[1] which is where the robot will move in the next step, not the endpoint.

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

        Uses configurable weight modes for J1 and J2:
        - "decay": w1(t) = max(min_weight, initial_weight - weight_decay_rate * t)
        - "constant": w1 = initial_weight
        - "periodic": w1(t) = amplitude * cos((π/30) * t) + center

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
            Dictionary with keys:
            - 'next_position': Next position to move to
            - 'best_utility': Best utility value (using normalized J1 and J2)
            - 'best_information_gain': Normalized information gain of best path
            - 'best_travel_cost': Normalized travel cost of best path
            - 'best_information_gain_original': Original (non-normalized) information gain
            - 'best_travel_cost_original': Original (non-normalized) travel cost
            - 'current_w1': Current weight for J1 (depends on weight_mode)
            - 'current_w2': Current weight for J2 (depends on weight_mode)
            - 'all_paths': All explored paths
        """
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()

        # Get current time-dependent weights
        w1, w2 = self.get_current_weight()

        if not paths:
            return {
                'next_position': start_pos,
                'best_utility': -np.inf,
                'best_information_gain': 0.0,
                'best_travel_cost': np.inf,
                'current_w1': w1,
                'current_w2': w2,
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

        # SECOND PASS: Normalize J1 and J2, calculate utilities with time-dependent weights
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

            # Calculate utility with TIME-DEPENDENT normalized values
            # utility = J1 * w1 - J2 * w2
            # w1 starts at 0.6 and decreases to 0.4 (favors exploration initially)
            # w2 starts at 0.4 and increases to 0.6 (favors exploitation later)
            utility = J1_normalized * w1 - J2_normalized * w2

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
            'current_w1': w1,  # Current weight for J1 (information gain)
            'current_w2': w2,  # Current weight for J2 (travel cost)
        }
