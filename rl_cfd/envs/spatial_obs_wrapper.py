"""
Ego-centric spatial observation wrapper for GasSourceEnv.

Maintains five world-space channels sized to the actual map (at most 40×30
cells for the maximum 20×15 m room at 0.5 m/cell). Each step the world grids
are embedded into a fixed 98×98 ego grid centred on the robot.

Occupancy is built by reading the ground-truth grid within LIDAR_MAX_RANGE of
the robot with a per-cell visibility (occlusion) check — physically equivalent
to what a LiDAR-built map would contain, but computed at wrapper resolution so
there is no quantisation-overwrite artefact. Walls still occlude cells behind
them.

Coordinate convention (matches OccupancyGrid.world_to_grid):
    world_col = floor(world_x / CELL_RES)
    world_row = floor(world_y / CELL_RES)
Row and y are co-directional — no axis flip.

Channels (6 total):
    is_known    0=never observed,           1=observed
    is_wall     0=not-wall or unknown,      1=wall
    gas         0=unvisited,                1=detection,   -1=visited, no detection
    recency     0=unvisited/stale … 1=just visited
    det_count   0=unvisited, normalised by running max ∈ [0,1]
    motion      fast-decay trail (decay≈0.6/step) — last few visited cells
                with order preserved by magnitude. Encodes direction-of-motion
                that the cumulative `recency` channel can't capture.
"""

import numpy as np

from .. import config as cfg


class SpatialObsWrapper:
    GRID_SIZE = cfg.SPATIAL_GRID_SIZE        # 98 cells — ego half-width = 49
    CELL_RES  = cfg.VISITED_CELL_RESOLUTION  # 0.5 m per cell
    MOTION_DECAY = float(getattr(cfg, 'MOTION_TRAIL_DECAY', 0.6))  # per-step decay
                                                                  # for motion channel

    def __init__(self, env):
        self._env = env
        self._decay = float(np.exp(-cfg.SPATIAL_LAMBDA))

        # World-space channels — shape set on reset from actual map dimensions
        self._known_world  = None  # {0, 1}
        self._wall_world   = None  # {0, 1}
        self._gas_world    = None  # {-1, 0, 1}
        self._rec_world    = None  # [0, 1]
        self._det_world    = None  # raw counts (unnormalised)
        self._motion_world = None  # [0, 1] — fast-decay motion trail
        self._max_det      = 0

        # Map cell dimensions — set on reset
        self._map_h_cells = 0
        self._map_w_cells = 0

        # Reveal-box precomputations — built once per reset
        self._reveal_radius_m   = cfg.LIDAR_MAX_RANGE
        self._reveal_r_cells    = int(np.ceil(self._reveal_radius_m / self.CELL_RES))
        self._ray_samples_t     = None   # (S,) sample distances along ray, metres
        self._dense_wall        = None   # (map_h, map_w) any-subcell-is-wall

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)

        h = int(np.floor(self._env._map_height / self.CELL_RES))
        w = int(np.floor(self._env._map_width  / self.CELL_RES))
        self._map_h_cells = h
        self._map_w_cells = w

        self._known_world  = np.zeros((h, w), dtype=np.float32)
        self._wall_world   = np.zeros((h, w), dtype=np.float32)
        self._gas_world    = np.zeros((h, w), dtype=np.float32)
        self._rec_world    = np.zeros((h, w), dtype=np.float32)
        self._det_world    = np.zeros((h, w), dtype=np.float32)
        self._motion_world = np.zeros((h, w), dtype=np.float32)
        self._max_det      = 0

        # Pre-compute ray sample distances at true-grid resolution
        true_res = self._env._grid.resolution
        n_steps  = int(np.ceil(self._reveal_radius_m / true_res))
        self._ray_samples_t = np.arange(1, n_steps + 1) * true_res  # (S,)

        # Precompute binary wall map at wrapper resolution: any sub-cell wall
        # → cell is marked as wall. With 0.5m robot diameter, a wrapper cell
        # containing any wall fragment is physically impassable, so this
        # matches navigability.
        ratio = self.CELL_RES / true_res
        sub = int(round(ratio))
        if abs(ratio - sub) > 1e-6:
            raise ValueError(
                f"CELL_RES ({self.CELL_RES}) must be an integer multiple of "
                f"GRID_RESOLUTION ({true_res}); got ratio {ratio}"
            )
        gt  = self._env._grid.grid
        gh, gw = gt.shape
        gh_pad = ((gh + sub - 1) // sub) * sub
        gw_pad = ((gw + sub - 1) // sub) * sub
        padded = np.zeros((gh_pad, gw_pad), dtype=gt.dtype)
        padded[:gh, :gw] = gt
        pooled = padded.reshape(gh_pad // sub, sub, gw_pad // sub, sub).max(axis=(1, 3))
        self._dense_wall = (pooled[:h, :w] != 0)

        robot_pos = self._env._robot_pos
        self._reveal(robot_pos)
        self._update_robot_cell(robot_pos, binary=0)
        self._update_motion(robot_pos)

        return self._get_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self._env.step(action)

        robot_pos = self._env._robot_pos
        binary    = int(info["gas_reading"])

        self._rec_world *= self._decay
        self._reveal(robot_pos)
        self._update_robot_cell(robot_pos, binary)
        self._update_motion(robot_pos)

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
        return (int(np.floor(wy / self.CELL_RES)),
                int(np.floor(wx / self.CELL_RES)))

    def _reveal(self, robot_pos):
        """Reveal wrapper cells within reveal_radius using GT grid + occlusion.

        For each candidate 0.5 m cell around the robot: cast a ray from the
        robot to the cell centre through the true 0.1 m grid; if any wall sits
        between them, the cell stays unknown. Otherwise stamp is_known=1 and
        is_wall = GT-wall-at-centre.
        """
        rx, ry   = robot_pos
        rc       = self._reveal_r_cells
        cx_cell  = int(np.floor(rx / self.CELL_RES))
        cy_cell  = int(np.floor(ry / self.CELL_RES))

        rows = np.arange(cy_cell - rc, cy_cell + rc + 1)  # (H,)
        cols = np.arange(cx_cell - rc, cx_cell + rc + 1)  # (W,)

        # Cell-centre world coordinates for the reveal box
        tx = (cols + 0.5) * self.CELL_RES                 # (W,)
        ty = (rows + 0.5) * self.CELL_RES                 # (H,)
        WX, WY = np.meshgrid(tx, ty)                      # (H, W)

        dx = WX - rx                                       # (H, W)
        dy = WY - ry
        dist = np.sqrt(dx * dx + dy * dy)

        in_radius = dist <= self._reveal_radius_m
        in_map    = ((rows >= 0) & (rows < self._map_h_cells))[:, None] & \
                    ((cols >= 0) & (cols < self._map_w_cells))[None, :]
        candidate = in_radius & in_map

        # --- Occlusion check: sample each ray at true-grid resolution ---
        true_res = self._env._grid.resolution
        gw       = self._env._grid.grid_width
        gh       = self._env._grid.grid_height
        t        = self._ray_samples_t                     # (S,)

        # Unit direction (guard r=0)
        safe_d = np.where(dist > 0, dist, 1.0)
        udx    = dx / safe_d                              # (H, W)
        udy    = dy / safe_d

        # Sample positions: (H, W, S)
        sx = rx + udx[..., None] * t[None, None, :]
        sy = ry + udy[..., None] * t[None, None, :]

        # Samples before (strictly less than) the target distance are the ones
        # that could occlude. Samples at/past the target don't count.
        before_target = t[None, None, :] < dist[..., None]

        gx = np.floor(sx / true_res).astype(np.int32)
        gy = np.floor(sy / true_res).astype(np.int32)
        in_bounds = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx_safe = np.clip(gx, 0, gw - 1)
        gy_safe = np.clip(gy, 0, gh - 1)

        sample_wall = (self._env._grid.grid[gy_safe, gx_safe] != 0) & in_bounds

        # A sample inside the target wrapper cell doesn't occlude the target
        # (otherwise any wall-centred target occludes itself, leaving gaps).
        sample_cell_r = np.floor(sy / self.CELL_RES).astype(np.int32)
        sample_cell_c = np.floor(sx / self.CELL_RES).astype(np.int32)
        same_as_target = (sample_cell_r == rows[:, None, None]) & \
                         (sample_cell_c == cols[None, :, None])

        occluded = np.any(sample_wall & before_target & ~same_as_target, axis=2)

        visible = candidate & ~occluded                             # (H, W)

        # --- Wall status at each target cell (any-subcell-is-wall, precomputed) ---
        rows_safe = np.clip(rows, 0, self._map_h_cells - 1)
        cols_safe = np.clip(cols, 0, self._map_w_cells - 1)
        rows_valid = (rows >= 0) & (rows < self._map_h_cells)
        cols_valid = (cols >= 0) & (cols < self._map_w_cells)
        cell_is_wall = self._dense_wall[rows_safe[:, None], cols_safe[None, :]] \
                       & (rows_valid[:, None] & cols_valid[None, :])

        # --- Stamp ---
        vr, vc = np.where(visible)
        world_r = rows[vr]
        world_c = cols[vc]
        self._known_world[world_r, world_c] = 1.0
        self._wall_world [world_r, world_c] = cell_is_wall[vr, vc].astype(np.float32)

    def _update_motion(self, robot_pos):
        """Decay the motion-trail channel and stamp the current cell.

        Fast decay (≈0.6/step) means the trail fades over ~5 steps, giving
        the policy short-term direction-of-motion without competing with
        the slower-decay `recency` channel.
        """
        self._motion_world *= self.MOTION_DECAY
        row, col = self._world_to_cell(robot_pos[0], robot_pos[1])
        if 0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells:
            self._motion_world[row, col] = 1.0

    def _update_robot_cell(self, robot_pos, binary):
        row, col = self._world_to_cell(robot_pos[0], robot_pos[1])
        if not (0 <= row < self._map_h_cells and 0 <= col < self._map_w_cells):
            return

        mapped = 1.0 if binary else -1.0
        if self._gas_world[row, col] != 1.0:      # never overwrite a detection
            self._gas_world[row, col] = mapped

        self._rec_world[row, col] = 1.0

        if binary:
            # Splat detection to center + 4-neighbors. Models a gas parcel
            # of ~0.5 m extent around the sensor (1 cell = CELL_RES = 0.5 m).
            # Same splat must be applied at deployment (e.g. GADEN).
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = row + dr, col + dc
                if 0 <= r < self._map_h_cells and 0 <= c < self._map_w_cells:
                    self._gas_world[r, c] = 1.0
                    self._det_world[r, c] += 1.0
                    val = int(self._det_world[r, c])
                    if val > self._max_det:
                        self._max_det = val

    def _get_obs(self):
        """Embed world channels into 98×98 ego window centred on the robot."""
        rx, ry    = self._env._robot_pos
        robot_col = int(np.floor(rx / self.CELL_RES))
        robot_row = int(np.floor(ry / self.CELL_RES))

        ego_r0 = 49 - robot_row
        ego_c0 = 49 - robot_col

        def _embed(world_buf, fill=0.0):
            out = np.full((self.GRID_SIZE, self.GRID_SIZE), fill, dtype=np.float32)
            dr0 = max(0, ego_r0)
            dc0 = max(0, ego_c0)
            dr1 = min(self.GRID_SIZE, ego_r0 + self._map_h_cells)
            dc1 = min(self.GRID_SIZE, ego_c0 + self._map_w_cells)
            if dr1 <= dr0 or dc1 <= dc0:
                return out
            sr0 = dr0 - ego_r0
            sc0 = dc0 - ego_c0
            out[dr0:dr1, dc0:dc1] = world_buf[sr0:sr0 + (dr1 - dr0),
                                              sc0:sc0 + (dc1 - dc0)]
            return out

        known  = _embed(self._known_world,  fill=0.0)
        wall   = _embed(self._wall_world,   fill=0.0)
        gas    = _embed(self._gas_world,    fill=0.0)
        rec    = _embed(self._rec_world,    fill=0.0)
        det    = _embed(self._det_world / self._max_det if self._max_det > 0
                        else self._det_world, fill=0.0)
        motion = _embed(self._motion_world, fill=0.0)

        spatial = np.stack([known, wall, gas, rec, det, motion], axis=0)  # (6, 98, 98)
        time_frac = self._env._current_step / cfg.MAX_STEPS
        # Local-cell ctx: query the wind field at the robot's current cell so
        # training and deployment share the same ctx semantics. Eliminates the
        # ctx-vs-local mismatch on real CFD wind fields where the spatial mean
        # diverges sharply from local wind direction (see
        # docs/notes on the GADEN-eval ctx investigation).
        wind = np.array(
            [*self._env._wind.get_observation_spatial_at(self._env._robot_pos),
             time_frac],
            dtype=np.float32,
        )                                                                  # (4,)
        return spatial, wind
