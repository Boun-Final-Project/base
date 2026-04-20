"""
RRT-Infotaxis local planner for efe_pmfs.

Adapted from efe_igdm/planning/rrt.py. Tree building, pruning, and collision
checks are unchanged; the only semantic change is that mutual information at
each vertex is evaluated via SourceBelief (PMFS p(S|H) + cached predicted hit
maps), not a particle filter.

Equation 19: BI(V_b) = Σ γ^{i-1} · I(v_{b,i}).
"""
from typing import List, Tuple
import numpy as np

from ..mapping.occupancy_grid import OccupancyGridMap
from ..pmfs.source_belief import SourceBelief


class Node:
    __slots__ = ['position', 'parent', 'depth', 'entropy_gain']

    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.entropy_gain = -np.inf


class RRT:
    def __init__(self, occupancy_grid: OccupancyGridMap, N_tn: int,
                 R_range: float, delta: float,
                 max_depth: int = 4, discount_factor: float = 0.8,
                 robot_radius: float = 0.35, max_iterations: int = None):
        self.occupancy_grid = occupancy_grid
        self.N_tn = N_tn
        self.R_range = R_range
        self.nodes: List[Node] = []
        self.delta = delta
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.robot_radius = robot_radius
        self.max_iterations = max_iterations if max_iterations is not None else (N_tn * 100)
        self._node_positions = np.empty((N_tn, 2), dtype=np.float64)
        self._node_count = 0

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------
    def sprawl(self, start_pos: Tuple[float, float]) -> None:
        self.nodes = [Node(start_pos)]
        self._node_positions = np.empty((self.N_tn, 2), dtype=np.float64)
        self._node_positions[0] = start_pos
        self._node_count = 1
        iteration = 0

        while self._node_count < self.N_tn and iteration < self.max_iterations:
            iteration += 1
            r = self.R_range * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x_rand = start_pos[0] + r * np.cos(theta)
            y_rand = start_pos[1] + r * np.sin(theta)

            closest_node = self.get_closest_node((x_rand, y_rand))
            diff = np.array([x_rand, y_rand]) - closest_node.position
            dist = np.linalg.norm(diff)
            if dist > self.delta:
                direction = diff / dist
                new_pos = closest_node.position + direction * self.delta
            else:
                new_pos = np.array([x_rand, y_rand])

            if self.is_collision_free_vectorized(closest_node.position, new_pos):
                new_node = Node(new_pos, closest_node)
                self.nodes.append(new_node)
                self._node_positions[self._node_count] = new_pos
                self._node_count += 1

    def get_closest_node(self, position):
        target = np.array(position)
        active = self._node_positions[:self._node_count]
        dists_sq = np.sum((active - target) ** 2, axis=1)
        return self.nodes[int(np.argmin(dists_sq))]

    def is_collision_free_vectorized(self, pos1, pos2):
        if not self.occupancy_grid.is_valid(tuple(pos2), radius=self.robot_radius):
            return False
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            return True
        num_samples = int(np.ceil(dist / (self.occupancy_grid.resolution * 0.5)))
        ts = np.linspace(0, 1, num_samples + 1)
        line_points = pos1 + np.outer(ts, (pos2 - pos1))
        grid_xs = ((line_points[:, 0] - self.occupancy_grid.origin_x) / self.occupancy_grid.resolution).astype(int)
        grid_ys = ((line_points[:, 1] - self.occupancy_grid.origin_y) / self.occupancy_grid.resolution).astype(int)
        valid_x = (grid_xs >= 0) & (grid_xs < self.occupancy_grid.width)
        valid_y = (grid_ys >= 0) & (grid_ys < self.occupancy_grid.height)
        if not np.all(valid_x & valid_y):
            return False
        grid_values = self.occupancy_grid.grid[grid_ys, grid_xs]
        if np.any(grid_values > 0):
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

    # ------------------------------------------------------------------
    # Branch information — now uses SourceBelief.mutual_information_at
    # ------------------------------------------------------------------
    def _world_to_grid_cell(self, pos):
        og = self.occupancy_grid
        gx = int(np.floor((pos[0] - og.origin_x) / og.resolution))
        gy = int(np.floor((pos[1] - og.origin_y) / og.resolution))
        return gx, gy

    def calculate_branch_information(self, path: List[Node],
                                     source_belief: SourceBelief) -> float:
        path = path[1:]  # exclude root
        BI = 0.0
        for i, node in enumerate(path):
            if node.entropy_gain != -np.inf:
                BI += (self.discount_factor ** i) * node.entropy_gain
                continue
            cell = self._world_to_grid_cell(node.position)
            mi = source_belief.mutual_information_at(cell)
            node.entropy_gain = mi
            BI += (self.discount_factor ** i) * mi
        return BI

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def get_next_move_debug(self, start_pos, source_belief: SourceBelief) -> dict:
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()

        if not paths:
            return {
                'next_position': start_pos,
                'best_path': [start_pos],
                'best_BI': -np.inf,
                'best_utility': -np.inf,
                'all_paths': [],
                'all_branch_information': [],
                'all_utilities': [],
                'tree_nodes': self.nodes.copy(),
                'num_branches': 0,
                'estimated_source': (0.0, 0.0),
                'start_position': start_pos,
                'sampling_radius': self.R_range,
                'max_depth': self.max_depth,
                'num_tree_nodes': len(self.nodes),
                'error': 'No valid paths found',
            }

        all_bi = []
        best_path = None
        best_BI = -np.inf
        for path in paths:
            BI = self.calculate_branch_information(path, source_belief)
            all_bi.append(BI)
            if BI > best_BI:
                best_BI = BI
                best_path = path

        if best_path is not None and len(best_path) > 1:
            next_position = tuple(best_path[1].position)
        else:
            next_position = start_pos

        est_source = source_belief.estimate_world(self.occupancy_grid.grid_to_world)

        all_paths_tuples = []
        for path in paths:
            all_paths_tuples.append([tuple(n.position) for n in path])
        best_path_tuples = [tuple(n.position) for n in best_path] if best_path is not None else []

        return {
            'next_position': next_position,
            'best_path': best_path_tuples,
            'best_BI': best_BI,
            'best_utility': best_BI,
            'all_paths': all_paths_tuples,
            'all_branch_information': all_bi,
            'all_utilities': all_bi,
            'tree_nodes': self.nodes.copy(),
            'num_branches': len(paths),
            'estimated_source': est_source,
            'start_position': start_pos,
            'sampling_radius': self.R_range,
            'max_depth': self.max_depth,
            'num_tree_nodes': len(self.nodes),
        }
