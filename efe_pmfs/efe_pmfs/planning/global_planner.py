"""
Global planner for efe_pmfs: frontier-based PRM with PMFS source belief.

Adapted from efe_igdm/planning/global_planner.py. The only semantic change is
that mutual information and the estimated source position come from a
SourceBelief (discrete p(S|H) over a fixed lattice, with cached predicted hit
maps) instead of a particle filter. KD-Tree PRM, single-source Dijkstra, and
frontier clustering are unchanged.

Utility function (Eq. 22):
    U_g(v_f) = I(v_f) · exp(-λ_p · cost(r, v_f)) · exp(-λ_s · D(v_f, r̂_0))
"""
from typing import List, Tuple, Optional, Dict
import heapq

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, label, find_objects

from ..mapping.occupancy_grid import OccupancyGridMap
from ..pmfs.source_belief import SourceBelief


class FrontierCluster:
    def __init__(self, cells: List[Tuple[int, int]]):
        self.cells = cells
        self.centroid_grid = self._compute_centroid()
        self.centroid_world = None
        self.size = len(cells)

    def _compute_centroid(self):
        if not self.cells:
            return (0, 0)
        mean_x = np.mean([c[0] for c in self.cells])
        mean_y = np.mean([c[1] for c in self.cells])
        best_cell = self.cells[0]
        min_dist_sq = float('inf')
        for cell in self.cells:
            dist_sq = (cell[0] - mean_x) ** 2 + (cell[1] - mean_y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_cell = cell
        return best_cell


class PRMVertex:
    def __init__(self, position, vertex_id):
        self.position = position
        self.id = vertex_id
        self.neighbors = []
        self.is_frontier_vertex = False
        self.frontier_cluster = None


class GlobalPlanner:
    def __init__(self,
                 occupancy_grid: OccupancyGridMap,
                 robot_radius: float = 0.35,
                 prm_samples: int = 300,
                 prm_connection_radius: float = 2.5,
                 frontier_min_size: int = 3,
                 lambda_p: float = 0.1,
                 lambda_s: float = 0.05,
                 max_frontiers_to_evaluate: int = 8,
                 debug: bool = False):
        self.occupancy_grid = occupancy_grid
        self.robot_radius = robot_radius
        self.prm_samples = prm_samples
        self.prm_connection_radius = prm_connection_radius
        self.frontier_min_size = frontier_min_size
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.max_frontiers_to_evaluate = max_frontiers_to_evaluate
        self.debug = debug

        self.frontier_cells = []
        self.frontier_clusters = []
        self.vertices = []
        self.vertex_dict = {}
        self.adj_list = {}

        self.best_frontier_vertex = None
        self.best_global_path = []
        self.best_utility = -np.inf

        self.ranked_frontiers = []
        self.current_frontier_index = 0

        self._radius_cells = int(np.ceil(self.robot_radius / self.occupancy_grid.resolution))
        r = self._radius_cells
        dy_range, dx_range = np.mgrid[-r:r + 1, -r:r + 1]
        mask = (dx_range ** 2 + dy_range ** 2) <= r ** 2
        self._disk_dx = dx_range[mask]
        self._disk_dy = dy_range[mask]

    # ------------------------------------------------------------------
    # Collision / validity
    # ------------------------------------------------------------------
    def _is_valid_optimistic(self, position):
        gx, gy = self.occupancy_grid.world_to_grid(*position)
        if gx < 0 or gx >= self.occupancy_grid.width or gy < 0 or gy >= self.occupancy_grid.height:
            return False
        check_gxs = gx + self._disk_dx
        check_gys = gy + self._disk_dy
        in_bounds = (
            (check_gxs >= 0) & (check_gxs < self.occupancy_grid.width) &
            (check_gys >= 0) & (check_gys < self.occupancy_grid.height)
        )
        valid_gxs = check_gxs[in_bounds]
        valid_gys = check_gys[in_bounds]
        if len(valid_gxs) == 0:
            return False
        cell_vals = self.occupancy_grid.grid[valid_gys, valid_gxs]
        return not np.any(cell_vals > 0)

    def _is_path_collision_free(self, pos1, pos2, allow_start_invalid=False):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 1e-6:
            if allow_start_invalid:
                return True
            return self._is_valid_optimistic(tuple(pos1))
        num_samples = max(int(np.ceil(dist / self.occupancy_grid.resolution)), 2)
        ts = np.linspace(0, 1, num_samples + 1)
        if allow_start_invalid:
            ts = ts[1:]
        line_points = pos1 + np.outer(ts, (pos2 - pos1))
        og = self.occupancy_grid
        center_gxs = ((line_points[:, 0] - og.origin_x) / og.resolution).astype(int)
        center_gys = ((line_points[:, 1] - og.origin_y) / og.resolution).astype(int)
        if np.any((center_gxs < 0) | (center_gxs >= og.width) |
                  (center_gys < 0) | (center_gys >= og.height)):
            return False
        all_gxs = center_gxs[:, None] + self._disk_dx[None, :]
        all_gys = center_gys[:, None] + self._disk_dy[None, :]
        all_gxs_clip = np.clip(all_gxs, 0, og.width - 1)
        all_gys_clip = np.clip(all_gys, 0, og.height - 1)
        cell_vals = og.grid[all_gys_clip.ravel(), all_gxs_clip.ravel()]
        if np.any(cell_vals > 0):
            return False
        return True

    # ------------------------------------------------------------------
    # Frontiers
    # ------------------------------------------------------------------
    def detect_frontiers(self):
        grid = self.occupancy_grid.grid
        is_free = (grid == 0)
        is_unknown = (grid == -1)
        structure = np.ones((3, 3), dtype=bool)
        unknown_dilated = binary_dilation(is_unknown, structure=structure)
        frontier_mask = is_free & unknown_dilated
        y_idxs, x_idxs = np.where(frontier_mask)
        self.frontier_cells = list(zip(x_idxs, y_idxs))
        return self.frontier_cells

    def cluster_frontiers(self):
        if not self.frontier_cells:
            self.frontier_clusters = []
            return []
        grid_shape = self.occupancy_grid.grid.shape
        frontier_mask = np.zeros(grid_shape, dtype=bool)
        if len(self.frontier_cells) > 0:
            xs, ys = zip(*self.frontier_cells)
            frontier_mask[ys, xs] = True
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        labeled_array, num_features = label(frontier_mask, structure=structure)
        clusters = []
        slices = find_objects(labeled_array)
        for i, sl in enumerate(slices):
            if sl is None:
                continue
            label_id = i + 1
            local_mask = (labeled_array[sl] == label_id)
            ly, lx = np.where(local_mask)
            global_y = ly + sl[0].start
            global_x = lx + sl[1].start
            cluster_cells = list(zip(global_x, global_y))
            if len(cluster_cells) >= self.frontier_min_size:
                cluster = FrontierCluster(cluster_cells)
                cluster.centroid_world = self.occupancy_grid.grid_to_world(
                    cluster.centroid_grid[0], cluster.centroid_grid[1]
                )
                clusters.append(cluster)

        max_cluster_span = 10
        final_clusters = []
        for cluster in clusters:
            if cluster.size > max_cluster_span:
                final_clusters.extend(self._split_large_cluster(cluster, max_cluster_span))
            else:
                final_clusters.append(cluster)
        self.frontier_clusters = final_clusters
        return final_clusters

    def _split_large_cluster(self, cluster, max_span):
        cells = np.array(cluster.cells, dtype=float)
        span = np.max(np.ptp(cells, axis=0))
        k = max(2, int(np.ceil(span / max_span)))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
        centroids = cells[indices].copy()
        for _ in range(10):
            dists = np.linalg.norm(cells[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                cells[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ])
            if np.allclose(centroids, new_centroids, atol=0.5):
                break
            centroids = new_centroids
        sub_clusters = []
        for j in range(k):
            mask = labels == j
            if np.sum(mask) >= self.frontier_min_size:
                sub_cells = [tuple(c) for c in cells[mask].astype(int)]
                sc = FrontierCluster(sub_cells)
                sc.centroid_world = self.occupancy_grid.grid_to_world(
                    sc.centroid_grid[0], sc.centroid_grid[1]
                )
                sub_clusters.append(sc)
        return sub_clusters if sub_clusters else [cluster]

    # ------------------------------------------------------------------
    # PRM build + Dijkstra
    # ------------------------------------------------------------------
    def build_prm_graph(self, current_position):
        self.vertices = []
        self.vertex_dict = {}
        self.adj_list = {}
        sample_points = []
        vertex_ids = []
        types = []

        sample_points.append(current_position)
        vertex_ids.append(0)
        types.append('start')

        for i, cluster in enumerate(self.frontier_clusters):
            pos = cluster.centroid_world
            if self._is_valid_optimistic(pos):
                sample_points.append(pos)
                vertex_ids.append(len(sample_points) - 1)
                types.append(f'frontier_{i}')

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
                vertex_ids.append(len(sample_points) - 1)
                types.append('sample')

        if not sample_points:
            return
        points_array = np.array(sample_points)
        tree = KDTree(points_array)

        for idx, pos, v_type in zip(vertex_ids, sample_points, types):
            v = PRMVertex(pos, idx)
            if v_type.startswith('frontier'):
                v.is_frontier_vertex = True
                cluster_idx = int(v_type.split('_')[1])
                v.frontier_cluster = self.frontier_clusters[cluster_idx]
            self.vertices.append(v)
            self.vertex_dict[idx] = v
            self.adj_list[idx] = []

        pairs = tree.query_pairs(self.prm_connection_radius)
        for i, j in pairs:
            pos_i = sample_points[i]
            pos_j = sample_points[j]
            dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            allow_start = (i == 0 or j == 0)
            if self._is_path_collision_free(pos_i, pos_j, allow_start_invalid=allow_start):
                self.vertex_dict[i].neighbors.append((j, dist))
                self.vertex_dict[j].neighbors.append((i, dist))
                self.adj_list[i].append((j, dist))
                self.adj_list[j].append((i, dist))

    def compute_all_paths_from_start(self, start_id=0):
        distances = {vid: float('inf') for vid in self.vertex_dict}
        previous = {vid: None for vid in self.vertex_dict}
        distances[start_id] = 0.0
        pq = [(0.0, start_id)]
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            if current_dist > distances[current_id]:
                continue
            if current_id in self.adj_list:
                for neighbor_id, weight in self.adj_list[current_id]:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor_id]:
                        distances[neighbor_id] = new_dist
                        previous[neighbor_id] = current_id
                        heapq.heappush(pq, (new_dist, neighbor_id))
        return distances, previous

    def reconstruct_path(self, goal_id, previous):
        if goal_id not in previous or (previous[goal_id] is None and goal_id != 0):
            return None
        path = []
        curr = goal_id
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
            if len(path) > len(previous):
                break
        return path[::-1]

    # ------------------------------------------------------------------
    # Utility evaluation (SourceBelief-based)
    # ------------------------------------------------------------------
    def _world_to_cell(self, pos):
        og = self.occupancy_grid
        gx = int(np.floor((pos[0] - og.origin_x) / og.resolution))
        gy = int(np.floor((pos[1] - og.origin_y) / og.resolution))
        return gx, gy

    def evaluate_frontier_vertices(self, current_pos, source_belief: SourceBelief):
        dists, prevs = self.compute_all_paths_from_start(0)
        source_estimate = np.array(source_belief.estimate_world(
            self.occupancy_grid.grid_to_world
        ))
        frontier_vertices = [v for v in self.vertices if v.is_frontier_vertex]

        scored_frontiers = []
        for f_vertex in frontier_vertices:
            path_cost = dists[f_vertex.id]
            if path_cost == float('inf'):
                continue
            src_dist = np.linalg.norm(np.array(f_vertex.position) - source_estimate)
            cheap_score = np.exp(-self.lambda_p * path_cost) * np.exp(-self.lambda_s * src_dist)
            scored_frontiers.append((cheap_score, path_cost, src_dist, f_vertex))
        scored_frontiers.sort(key=lambda x: x[0], reverse=True)
        frontiers_to_evaluate = scored_frontiers[:self.max_frontiers_to_evaluate]

        if not frontiers_to_evaluate:
            self.ranked_frontiers = []
            return {'best_frontier_vertex': None, 'error': 'No reachable frontiers'}

        # Batch MI over the candidate frontier cells.
        cells = np.array([self._world_to_cell(f_vertex.position)
                          for (_, _, _, f_vertex) in frontiers_to_evaluate], dtype=np.int32)
        mi_values = source_belief.mutual_information_batch(cells)

        candidates = []
        for (cheap_score, path_cost, src_dist, f_vertex), mi in zip(frontiers_to_evaluate, mi_values):
            if mi <= 1e-6:
                continue
            utility = mi * np.exp(-self.lambda_p * path_cost) * np.exp(-self.lambda_s * src_dist)
            candidates.append({
                'utility': utility,
                'vertex': f_vertex,
                'path_cost': path_cost,
                'mi': float(mi),
                'src_dist': src_dist,
            })

        if not candidates:
            self.ranked_frontiers = []
            return {'best_frontier_vertex': None, 'error': 'No reachable high-info frontiers'}

        candidates.sort(key=lambda x: x['utility'], reverse=True)
        best = candidates[0]
        best_path_ids = self.reconstruct_path(best['vertex'].id, prevs)
        self.best_global_path = [self.vertex_dict[vid].position for vid in best_path_ids]
        self.best_frontier_vertex = best['vertex']
        self.best_utility = best['utility']

        self.ranked_frontiers = [{
            'utility': c['utility'],
            'vertex': c['vertex'],
            'prev_map': prevs,
        } for c in candidates]
        self.current_frontier_index = 0

        return {
            'best_frontier_vertex': best['vertex'],
            'best_global_path': self.best_global_path,
            'best_utility': best['utility'],
            'best_mutual_info': best['mi'],
            'best_path_cost': best['path_cost'],
            'best_source_dist': best['src_dist'],
            'num_frontiers': len(frontier_vertices),
            'num_reachable': len(candidates),
        }

    def get_next_best_frontier(self):
        self.current_frontier_index += 1
        if self.current_frontier_index >= len(self.ranked_frontiers):
            return None
        candidate = self.ranked_frontiers[self.current_frontier_index]
        vertex = candidate['vertex']
        prev_map = candidate['prev_map']
        path_ids = self.reconstruct_path(vertex.id, prev_map)
        if not path_ids:
            return self.get_next_best_frontier()
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
            'total_frontiers': len(self.ranked_frontiers),
        }

    def plan(self, current_position, source_belief: SourceBelief):
        frontier_cells = self.detect_frontiers()
        if not frontier_cells:
            return {'success': False, 'error': 'No frontiers detected'}
        clusters = self.cluster_frontiers()
        if not clusters:
            return {'success': False, 'error': 'No valid clusters'}
        self.build_prm_graph(current_position)
        eval_results = self.evaluate_frontier_vertices(current_position, source_belief)
        if eval_results.get('best_frontier_vertex') is None:
            return {
                'success': False,
                'error': eval_results.get('error', 'No reachable frontiers'),
                'frontier_cells': self.frontier_cells,
                'frontier_clusters': self.frontier_clusters,
                'prm_vertices': self.vertices,
            }
        eval_results['success'] = True
        eval_results['frontier_cells'] = self.frontier_cells
        eval_results['frontier_clusters'] = self.frontier_clusters
        eval_results['prm_vertices'] = self.vertices
        return eval_results
