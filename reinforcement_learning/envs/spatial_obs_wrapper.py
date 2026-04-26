"""
Ego-centric spatial observation wrapper for GasSourceEnv.

Maintains three world-space grids sized to the actual map (at most 100×75 cells
for the maximum 20×15m room at 0.2m/cell). Each step the world grids are
embedded into a fixed 221×221 ego grid centred on the robot.

The ego half-width is 110 cells = 22m, larger than the worst-case map dimension
(20m), so the entire map always fits inside the ego window regardless of where
the robot is — no large buffer or padding tricks needed.

Coordinate convention (verified against OccupancyGrid.world_to_grid):
    world_col = floor(world_x / CELL_RES)
    world_row = floor(world_y / CELL_RES)
Row and y are co-directional — no axis flip.

Channel encoding (consistent negative-signal convention):
    occupancy     : 0=unknown,     1=free,        -1=wall
    gas           : 0=unvisited,   1=detection,   -1=no detection
    recency       : 0=unvisited/stale … 1=just visited
    wind_gradient : dot product of cell offset from robot with wind direction,
                    scaled by wind speed and normalised to [-1, 1]
                    positive = upwind, negative = downwind
"""

import numpy as np

from .. import config as cfg


class SpatialObsWrapper:
    GRID_SIZE = cfg.SPATIAL_GRID_SIZE        # 221 cells — ego window half-width = 110
    CELL_RES  = cfg.SPATIAL_CELL_RES         # 0.2 m per cell (decoupled from reward tracking)

    def __init__(self, env):
        self._env = env
        self._decay = float(np.exp(-cfg.SPATIAL_LAMBDA))

        # World-space grids — shape set on reset from actual map dimensions
        self._occ_world = None   # 0=unknown, 1=free, -1=wall
        self._gas_world = None   # 0=unvisited, 1=detection, -1=no detection
        self._rec_world = None   # recency ∈ [0, 1]

        # Ego-space wind gradient — precomputed at reset, constant per episode
        self._wind_grad = None   # (GRID_SIZE, GRID_SIZE), ∈ [-1, 1]

        # Map cell dimensions — set on reset
        self._map_h_cells = 0
        self._map_w_cells = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)

        h = int(np.floor(self._env._map_height / self.CELL_RES))
        w = int(np.floor(self._env._map_width  / self.CELL_RES))
        self._map_h_cells = h
        self._map_w_cells = w

        self._occ_world = np.zeros((h, w), dtype=np.float32)
        self._gas_world = np.zeros((h, w), dtype=np.float32)   # 0 = unvisited
        self._rec_world = np.zeros((h, w), dtype=np.float32)

        # Precompute wind gradient for this episode (wind is constant per episode)
        center = self.GRID_SIZE // 2  # 110 for GRID_SIZE=221
        dc, dr = np.meshgrid(
            np.arange(self.GRID_SIZE, dtype=np.float32) - center,
            np.arange(self.GRID_SIZE, dtype=np.float32) - center,
        )
        wind_obs   = self._env._wind.get_observation()  # (speed/max, dir/2π)
        speed_norm = wind_obs[0]
        wind_angle = wind_obs[1] * 2.0 * np.pi
        max_dist   = center * np.sqrt(2.0)
        self._wind_grad = (
            speed_norm * (dc * np.cos(wind_angle) + dr * np.sin(wind_angle)) / max_dist
        ).astype(np.float32)

        robot_pos = self._env._robot_pos
        distances = self._env._lidar.scan(tuple(robot_pos))
        self._update_occupancy(robot_pos, distances)
        self._update_robot_cell(robot_pos, binary=0)

        return self._get_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self._env.step(action)

        robot_pos = self._env._robot_pos
        binary    = int(info["gas_reading"])

        self._rec_world *= self._decay

        distances = self._env._lidar.scan(tuple(robot_pos))
        self._update_occupancy(robot_pos, distances)
        self._update_robot_cell(robot_pos, binary)

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Curriculum forwarding
    # ------------------------------------------------------------------

    def render(self):
        return self._env.render()

    def set_room_size_range(self, width_range, height_range):
        self._env.set_room_size_range(width_range, height_range)

    def set_max_template(self, max_template_id):
        self._env.set_max_template(max_template_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_cell(self, wx, wy):
        """World coordinates (m) → world grid (row, col)."""
        return (int(np.floor(wy / self.CELL_RES)),
                int(np.floor(wx / self.CELL_RES)))

    def _update_occupancy(self, robot_pos, distances):
        """Reveal free cells and wall hits from a LiDAR scan into world grids."""
        rx, ry   = robot_pos
        lidar    = self._env._lidar
        n_steps  = lidar._n_steps
        n_rays   = len(distances)

        # World positions for all (ray, step) samples: (R, S)
        wx = rx + lidar._dx   # (R, S)
        wy = ry + lidar._dy   # (R, S)

        wc = np.floor(wx / self.CELL_RES).astype(np.int32)  # world col (R, S)
        wr = np.floor(wy / self.CELL_RES).astype(np.int32)  # world row (R, S)

        in_map = (wc >= 0) & (wc < self._map_w_cells) & \
                 (wr >= 0) & (wr < self._map_h_cells)

        n_free    = (distances * n_steps).astype(np.int32)                   # (R,)
        free_mask = np.arange(n_steps)[None, :] < n_free[:, None]            # (R, S)

        free_valid = free_mask & in_map
        fr, fc = wr[free_valid], wc[free_valid]
        not_wall = self._occ_world[fr, fc] != -1.0
        self._occ_world[fr[not_wall], fc[not_wall]] = 1.0

        # Wall: first hit step per ray (only for rays that actually hit)
        hit_mask = distances < 1.0                                            # (R,)
        hit_s    = np.clip(n_free, 0, n_steps - 1)                           # (R,)
        ray_idx  = np.arange(n_rays)
        hit_wc   = np.floor((rx + lidar._dx[ray_idx, hit_s]) / self.CELL_RES).astype(np.int32)
        hit_wr   = np.floor((ry + lidar._dy[ray_idx, hit_s]) / self.CELL_RES).astype(np.int32)
        hit_valid = hit_mask & \
                    (hit_wc >= 0) & (hit_wc < self._map_w_cells) & \
                    (hit_wr >= 0) & (hit_wr < self._map_h_cells)
        self._occ_world[hit_wr[hit_valid], hit_wc[hit_valid]] = -1.0

    def _update_robot_cell(self, robot_pos, binary):
        """Update gas, recency, and detection count at the robot's current cell."""
        row, col = self._world_to_cell(robot_pos[0], robot_pos[1])
        if not (0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells):
            return

        # Gas: never overwrite a confirmed detection
        mapped = 1.0 if binary else -1.0
        if self._gas_world[row, col] != 1.0:
            self._gas_world[row, col] = mapped

        # Recency
        self._rec_world[row, col] = 1.0

    def _get_obs(self):
        """Embed world grids into the ego window centred on the robot."""
        rx, ry     = self._env._robot_pos
        robot_col  = int(np.floor(rx / self.CELL_RES))
        robot_row  = int(np.floor(ry / self.CELL_RES))

        # Top-left corner of the world grid in ego coordinates
        center = self.GRID_SIZE // 2
        ego_r0 = center - robot_row
        ego_c0 = center - robot_col

        def _embed(world_buf, fill=0.0):
            out  = np.full((self.GRID_SIZE, self.GRID_SIZE), fill, dtype=np.float32)
            dr0  = max(0, ego_r0)
            dc0  = max(0, ego_c0)
            dr1  = min(self.GRID_SIZE, ego_r0 + self._map_h_cells)
            dc1  = min(self.GRID_SIZE, ego_c0 + self._map_w_cells)
            if dr1 <= dr0 or dc1 <= dc0:
                return out
            sr0  = dr0 - ego_r0
            sc0  = dc0 - ego_c0
            out[dr0:dr1, dc0:dc1] = world_buf[sr0:sr0 + (dr1 - dr0),
                                               sc0:sc0 + (dc1 - dc0)]
            return out

        occ = _embed(self._occ_world, fill=0.0)
        gas = _embed(self._gas_world, fill=0.0)   # out-of-map = unvisited
        rec = _embed(self._rec_world, fill=0.0)

        spatial = np.stack([occ, gas, rec, self._wind_grad], axis=0)  # (4, GRID_SIZE, GRID_SIZE)
        return spatial
