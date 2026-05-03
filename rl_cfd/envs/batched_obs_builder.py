"""Batched GPU-side equivalent of SpatialObsWrapper across N envs.

Workers only run the simulator and hand back raw state (robot pose, gas bit,
etc.).  This class owns all world-space mapping memory and builds the ego
observation for every env in one batched GPU call per step.

Numerical contract with SpatialObsWrapper (rl_cfd / fast-bundle 6-channel):
    - Six channels in the same order:
      [is_known, is_wall, gas, recency, det_count/max, motion]
    - Same ego embedding with out-of-map cells set to 0
    - Same 4-d ctx: [speed/max_speed, cos(dir), sin(dir), step/MAX_STEPS]
    - Same _reveal semantics: ground-truth inside 3 m box, ray-occlusion test
      excluding samples inside the target wrapper cell
    - motion: fast-decay (≈MOTION_DECAY/step) trail of recent robot cells.
      Mirrors SpatialObsWrapper._update_motion exactly.
"""

import numpy as np
import torch

from .. import config as cfg


class BatchedObsBuilder:

    CELL_RES = cfg.VISITED_CELL_RESOLUTION   # 0.5 m
    GRID_SIZE = cfg.SPATIAL_GRID_SIZE        # 98
    REVEAL_RADIUS = cfg.LIDAR_MAX_RANGE      # 3.0 m
    MOTION_DECAY = float(getattr(cfg, 'MOTION_TRAIL_DECAY', 0.6))

    def __init__(self, num_envs, device, true_res=0.1):
        self.N      = num_envs
        self.device = device
        self.true_res = float(true_res)
        self.decay  = float(np.exp(-cfg.SPATIAL_LAMBDA))

        # True-grid dims padded to accommodate every template (template 5 is
        # a maze that exceeds ROOM_*_RANGE; observed max (234,200) → 256×224).
        self.H_true = 256
        self.W_true = 224
        # Wrapper-cell dims scale with true-grid dims by (true_res / CELL_RES)
        self.H_wrap = int(np.ceil(self.H_true * self.true_res / self.CELL_RES))
        self.W_wrap = int(np.ceil(self.W_true * self.true_res / self.CELL_RES))

        # Reveal box: ±RC wrapper cells around robot's wrapper cell
        self.RC = int(np.ceil(self.REVEAL_RADIUS / self.CELL_RES))            # 6
        self.HC = 2 * self.RC + 1                                              # 13

        # Per-env state on device
        t = torch.zeros
        self.gt_grid      = t((num_envs, self.H_true, self.W_true), device=device)
        self.known_world  = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.wall_world   = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.gas_world    = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.rec_world    = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.det_world    = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.motion_world = t((num_envs, self.H_wrap, self.W_wrap), device=device)
        self.max_det      = t((num_envs,),                           device=device)
        self.map_h_cells = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.map_w_cells = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.map_h_true  = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.map_w_true  = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Pre-compute reveal helpers
        off = torch.arange(-self.RC, self.RC + 1, device=device)               # (HC,)
        self._off_r = off.view(1, -1, 1)                                       # (1, HC, 1)
        self._off_c = off.view(1, 1, -1)                                       # (1, 1, HC)
        n_steps = int(np.ceil(self.REVEAL_RADIUS / self.true_res))             # 30
        self._t_samples = (torch.arange(1, n_steps + 1, device=device) * self.true_res)  # (S,)

        # Ego embedding helpers
        self._ego_idx = torch.arange(self.GRID_SIZE, device=device)            # (G,)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_env(self, env_idx, gt_grid_np, map_h_cells, map_w_cells,
                  robot_pos, wind_raw, current_step, binary_gas):
        """Upload a new episode's GT grid + zero world buffers for this env."""
        h_true, w_true = gt_grid_np.shape
        assert h_true <= self.H_true and w_true <= self.W_true, \
            f"GT grid {gt_grid_np.shape} exceeds pad size ({self.H_true},{self.W_true})"
        # Upload padded GT grid
        self.gt_grid[env_idx].zero_()
        self.gt_grid[env_idx, :h_true, :w_true] = torch.from_numpy(
            (gt_grid_np != 0).astype(np.float32)).to(self.device)
        self.map_h_true[env_idx] = h_true
        self.map_w_true[env_idx] = w_true
        self.map_h_cells[env_idx] = map_h_cells
        self.map_w_cells[env_idx] = map_w_cells

        # Zero per-env world buffers
        self.known_world [env_idx].zero_()
        self.wall_world  [env_idx].zero_()
        self.gas_world   [env_idx].zero_()
        self.rec_world   [env_idx].zero_()
        self.det_world   [env_idx].zero_()
        self.motion_world[env_idx].zero_()
        self.max_det[env_idx] = 0.0

    # ------------------------------------------------------------------
    # Batched reveal + point updates across all envs (one GPU launch)
    # ------------------------------------------------------------------

    def update_batch(self, robot_pos_np, binary_gas_np, decay_mask=None):
        """Apply one timestep of updates for all N envs.

        robot_pos_np : (N, 2) float32  world coords (x, y)
        binary_gas_np: (N,)   bool/int  gas detection at robot this step
        decay_mask   : (N,)   bool      if given, only these envs get rec_world decay
                                        (others are freshly reset; decay already skipped)
        """
        N = self.N
        device = self.device
        robot_pos = torch.as_tensor(robot_pos_np, dtype=torch.float32, device=device)  # (N, 2)
        binary    = torch.as_tensor(binary_gas_np, dtype=torch.bool, device=device)    # (N,)

        # 1. Recency + motion decay (skip freshly-reset envs)
        if decay_mask is None:
            self.rec_world.mul_(self.decay)
            self.motion_world.mul_(self.MOTION_DECAY)
        else:
            dm = torch.as_tensor(decay_mask, dtype=torch.bool, device=device)
            self.rec_world[dm]    *= self.decay
            self.motion_world[dm] *= self.MOTION_DECAY

        # 2. Reveal
        self._reveal_batch(robot_pos)

        # 3. Robot-cell point updates (gas, rec, det, max_det, motion)
        self._update_robot_cell_batch(robot_pos, binary)

    def _reveal_batch(self, robot_pos):
        """GPU-batched equivalent of SpatialObsWrapper._reveal for all envs."""
        N   = self.N
        RC  = self.RC
        HC  = self.HC
        dev = self.device

        rx = robot_pos[:, 0]  # (N,)
        ry = robot_pos[:, 1]

        # Robot wrapper-cell indices
        rob_c = (rx / self.CELL_RES).floor().long()  # (N,)
        rob_r = (ry / self.CELL_RES).floor().long()

        # Target wrapper-cell indices per candidate: (N, HC, HC)
        world_r = (rob_r.view(N, 1, 1) + self._off_r).expand(N, HC, HC).contiguous()
        world_c = (rob_c.view(N, 1, 1) + self._off_c).expand(N, HC, HC).contiguous()

        # Target world coords (cell centres), (N, HC, HC)
        target_x = (world_c.float() + 0.5) * self.CELL_RES
        target_y = (world_r.float() + 0.5) * self.CELL_RES

        dx = target_x - rx.view(N, 1, 1)
        dy = target_y - ry.view(N, 1, 1)
        dist = torch.sqrt(dx * dx + dy * dy)                                   # (N, HC, HC)

        in_radius = dist <= self.REVEAL_RADIUS
        in_map = (world_r >= 0) & (world_r < self.map_h_cells.view(N, 1, 1)) & \
                 (world_c >= 0) & (world_c < self.map_w_cells.view(N, 1, 1))
        candidate = in_radius & in_map

        # Unit ray direction (guard r=0)
        safe_d = torch.where(dist > 0, dist, torch.ones_like(dist))
        udx = dx / safe_d
        udy = dy / safe_d

        # Sample positions along ray: (N, HC, HC, S)
        t = self._t_samples                                                     # (S,)
        sx = rx.view(N, 1, 1, 1) + udx.unsqueeze(-1) * t.view(1, 1, 1, -1)
        sy = ry.view(N, 1, 1, 1) + udy.unsqueeze(-1) * t.view(1, 1, 1, -1)

        # True-grid indices
        gx = (sx / self.true_res).floor().long()
        gy = (sy / self.true_res).floor().long()
        in_bounds = (gx >= 0) & (gx < self.map_w_true.view(N, 1, 1, 1)) & \
                    (gy >= 0) & (gy < self.map_h_true.view(N, 1, 1, 1))
        gx_safe = gx.clamp(0, self.W_true - 1)
        gy_safe = gy.clamp(0, self.H_true - 1)

        # Sample wall lookup (N, HC, HC, S)
        env_idx_s = torch.arange(N, device=dev).view(N, 1, 1, 1).expand_as(gx_safe)
        sample_wall = (self.gt_grid[env_idx_s, gy_safe, gx_safe] > 0.5) & in_bounds

        # Exclude samples inside the target wrapper cell (self-occlusion fix)
        sample_cell_r = (sy / self.CELL_RES).floor().long()
        sample_cell_c = (sx / self.CELL_RES).floor().long()
        same_as_target = (sample_cell_r == world_r.unsqueeze(-1)) & \
                         (sample_cell_c == world_c.unsqueeze(-1))

        before_target = t.view(1, 1, 1, -1) < dist.unsqueeze(-1)
        occluded = (sample_wall & before_target & ~same_as_target).any(dim=-1) # (N, HC, HC)

        visible = candidate & ~occluded

        # Wall status at each target cell centre (true-grid lookup)
        cx_true = (target_x / self.true_res).floor().long()
        cy_true = (target_y / self.true_res).floor().long()
        cx_safe = cx_true.clamp(0, self.W_true - 1)
        cy_safe = cy_true.clamp(0, self.H_true - 1)
        env_idx = torch.arange(N, device=dev).view(N, 1, 1).expand_as(cx_safe)
        cell_is_wall = self.gt_grid[env_idx, cy_safe, cx_safe] > 0.5           # (N, HC, HC)

        # Stamp — advanced-index assignment (NOTE: dup indices resolved by last-write,
        # but visible cells map 1:1 to wrapper cells so no duplicates)
        world_r_safe = world_r.clamp(0, self.H_wrap - 1)
        world_c_safe = world_c.clamp(0, self.W_wrap - 1)
        vis_env = env_idx[visible]
        vis_r   = world_r_safe[visible]
        vis_c   = world_c_safe[visible]
        self.known_world[vis_env, vis_r, vis_c] = 1.0
        self.wall_world [vis_env, vis_r, vis_c] = cell_is_wall[visible].float()

    def _update_robot_cell_batch(self, robot_pos, binary):
        N   = self.N
        dev = self.device

        rx = robot_pos[:, 0]
        ry = robot_pos[:, 1]
        col = (rx / self.CELL_RES).floor().long()
        row = (ry / self.CELL_RES).floor().long()
        in_map = (row >= 0) & (row < self.map_h_cells) & \
                 (col >= 0) & (col < self.map_w_cells)
        row_s = row.clamp(0, self.H_wrap - 1)
        col_s = col.clamp(0, self.W_wrap - 1)
        env_idx = torch.arange(N, device=dev)

        # gas: don't overwrite +1 with -1 (non-detection only stamps centre)
        mapped = torch.where(binary, torch.ones_like(binary, dtype=torch.float32),
                             -torch.ones_like(binary, dtype=torch.float32))
        cur_gas = self.gas_world[env_idx, row_s, col_s]
        can_write = in_map & (cur_gas < 0.9999)
        new_gas = torch.where(can_write, mapped, cur_gas)
        self.gas_world[env_idx, row_s, col_s] = new_gas

        # recency
        self.rec_world[env_idx[in_map], row_s[in_map], col_s[in_map]] = 1.0

        # motion: stamp current cell to 1.0 (decay already applied in update_batch)
        self.motion_world[env_idx[in_map], row_s[in_map], col_s[in_map]] = 1.0

        # Splat detection to centre + 4-neighbours (matches SpatialObsWrapper).
        # Same ~0.5 m gas-parcel model that must be applied at deploy time.
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        det_base = in_map & binary
        for dr, dc in offsets:
            nr = row + dr
            nc = col + dc
            nb_valid = (nr >= 0) & (nr < self.map_h_cells) & \
                       (nc >= 0) & (nc < self.map_w_cells) & det_base
            nr_s = nr.clamp(0, self.H_wrap - 1)
            nc_s = nc.clamp(0, self.W_wrap - 1)
            self.gas_world[env_idx[nb_valid], nr_s[nb_valid], nc_s[nb_valid]] = 1.0
            self.det_world[env_idx[nb_valid], nr_s[nb_valid], nc_s[nb_valid]] += 1.0

        # running max per env
        det_at_centre = self.det_world[env_idx, row_s, col_s]
        self.max_det = torch.maximum(self.max_det, det_at_centre)

    # ------------------------------------------------------------------
    # Build ego observation for all envs
    # ------------------------------------------------------------------

    def build_obs(self, robot_pos_np, wind_raw_np, current_step_np):
        """Return (spatial (N, 6, G, G), wind (N, 3)) on device."""
        N   = self.N
        G   = self.GRID_SIZE
        dev = self.device
        robot_pos = torch.as_tensor(robot_pos_np, dtype=torch.float32, device=dev)
        wind_raw  = torch.as_tensor(wind_raw_np,  dtype=torch.float32, device=dev)  # (N, 2)
        step_t    = torch.as_tensor(current_step_np, dtype=torch.float32, device=dev)  # (N,)

        rx = robot_pos[:, 0]
        ry = robot_pos[:, 1]
        rob_c = (rx / self.CELL_RES).floor().long()  # (N,)
        rob_r = (ry / self.CELL_RES).floor().long()

        # Ego grid world-index for each output cell
        ego = self._ego_idx
        origin_r = rob_r - G // 2
        origin_c = rob_c - G // 2
        wr = origin_r.view(N, 1, 1) + ego.view(1, G, 1)        # (N, G, 1)
        wc = origin_c.view(N, 1, 1) + ego.view(1, 1, G)        # (N, 1, G)
        wr = wr.expand(N, G, G)
        wc = wc.expand(N, G, G)

        valid = (wr >= 0) & (wr < self.map_h_cells.view(N, 1, 1)) & \
                (wc >= 0) & (wc < self.map_w_cells.view(N, 1, 1))
        wr_s = wr.clamp(0, self.H_wrap - 1)
        wc_s = wc.clamp(0, self.W_wrap - 1)
        env_idx = torch.arange(N, device=dev).view(N, 1, 1).expand(N, G, G)

        def embed(world):
            v = world[env_idx, wr_s, wc_s]
            return torch.where(valid, v, torch.zeros_like(v))

        known  = embed(self.known_world)
        wall   = embed(self.wall_world)
        gas    = embed(self.gas_world)
        rec    = embed(self.rec_world)
        det_n  = self.det_world / self.max_det.clamp(min=1.0).view(N, 1, 1)
        det    = embed(det_n)
        motion = embed(self.motion_world)

        spatial = torch.stack([known, wall, gas, rec, det, motion], dim=1)  # (N, 6, G, G)

        time_frac = step_t / cfg.MAX_STEPS
        wind = torch.cat([wind_raw, time_frac.view(N, 1)], dim=-1)  # (N, 3)
        return spatial, wind
