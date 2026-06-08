"""
Fast-bundle filament plume: spatially-varying wind + Poisson emission.

Differences vs the baseline ``test_rl/envs/filament_plume.py``:

1. Constructor takes a ``WindField`` instead of scalar ``(wind_speed, wind_angle)``.
   Filaments advect with the *local* wind at their position each step, so
   the plume bends around walls / through corridors the way GADEN's CFD
   plume does.

2. Emission rate is **Poisson-stochastic**: the actual count released per
   step is sampled from ``Poisson(filaments_per_step)``, giving puffy /
   intermittent emission that better matches GADEN's bursty signal.

3. New filaments are **rejected if they'd land inside a wall** (defensive
   for sources placed near walls).

The existing wall-reflection / tunnelling logic is preserved unchanged —
the `_handle_obstacles` machinery already does proper wall bouncing.
"""

import numpy as np

from .. import config as cfg


class FilamentPlume:
    """Lagrangian filament plume driven by a spatially-varying WindField."""

    def __init__(
        self,
        source_pos,
        wind_field,                  # NEW: WindField instance (replaces wind_speed/angle)
        occupancy_grid,
        dt=None,
        K=None,
        turbulence_scale=None,
        max_age=None,
        filaments_per_step=None,
        initial_sigma=None,
        mass=None,
        min_sigma=None,
        reflection_energy=None,
        poisson_emission=True,       # NEW: stochastic release count
        rng=None,
    ):
        """
        Parameters
        ----------
        source_pos : (x, y) in meters.
        wind_field : WindField
            Spatially-varying wind. Must have been ``randomize()``d already.
        occupancy_grid : OccupancyGrid
            For wall-aware advection.
        poisson_emission : bool
            If True, per-step release count is Poisson(filaments_per_step).
        ... (rest match baseline FilamentPlume)
        """
        self.source_pos = np.asarray(source_pos, dtype=np.float64)
        self.wind = wind_field
        self.grid = occupancy_grid

        self.dt = dt if dt is not None else cfg.FILAMENT_DT
        self.K = K if K is not None else cfg.FILAMENT_K
        self.turbulence_scale = (
            turbulence_scale if turbulence_scale is not None
            else cfg.FILAMENT_TURBULENCE_SCALE
        )
        self.max_age = max_age if max_age is not None else cfg.FILAMENT_MAX_AGE
        self.filaments_per_step = (
            filaments_per_step if filaments_per_step is not None
            else cfg.FILAMENTS_PER_STEP
        )
        self.initial_sigma = (
            initial_sigma if initial_sigma is not None
            else cfg.FILAMENT_INITIAL_SIGMA
        )
        self.mass = mass if mass is not None else cfg.FILAMENT_MASS
        self.min_sigma = (
            min_sigma if min_sigma is not None else cfg.FILAMENT_MIN_SIGMA
        )
        self.reflection_energy = (
            reflection_energy if reflection_energy is not None
            else cfg.FILAMENT_REFLECTION_ENERGY
        )
        self.poisson_emission = bool(poisson_emission)

        self.rng = rng if rng is not None else np.random.default_rng()

        # Tunnelling check: bound max displacement by the highest wind speed
        # we expect across the field plus 3-sigma turbulence.
        max_field_speed = float(np.hypot(self.wind.Ux, self.wind.Uy).max())
        _max_disp = (max_field_speed + 3.0 * self.turbulence_scale * max_field_speed) * self.dt
        self._tunnel_check_steps = max(
            8, int(np.ceil(_max_disp / self.grid.resolution)) + 2
        )

        self._capacity = max(1024, self.max_age * self.filaments_per_step * 2)
        self._reset_arrays()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self):
        """One simulation step: release → advect (local wind) → diffuse → reflect → age."""
        # 1. Release with Poisson rate (or fixed for legacy reproducibility)
        if self.poisson_emission:
            n_new = int(self.rng.poisson(self.filaments_per_step))
        else:
            n_new = int(self.filaments_per_step)
        if n_new > 0:
            self._release(n_new)

        if self._n == 0:
            return

        # 2. Advection: per-filament local wind + scaled turbulence
        local_wind = self.wind.local_batch(self.positions[:self._n])  # (n, 2)
        local_speed = np.hypot(local_wind[:, 0], local_wind[:, 1])    # (n,)
        turb_sigma = self.turbulence_scale * local_speed[:, None]     # (n, 1) broadcast
        turbulence = self.rng.normal(0.0, 1.0, size=(self._n, 2)) * turb_sigma
        velocity = local_wind + turbulence
        displacement = velocity * self.dt
        pre_positions = self.positions[:self._n].copy()
        self.positions[:self._n] += displacement
        self.velocities[:self._n] = velocity

        # 3. Diffusion
        self.sigmas[:self._n] = np.sqrt(
            self.sigmas[:self._n] ** 2 + 2.0 * self.K * self.dt
        )

        # 4. Wall reflection (unchanged from baseline)
        self._handle_obstacles(pre_positions)

        # 5. Age + cull
        self.ages[:self._n] += 1
        alive = self.ages[:self._n] < self.max_age
        self._cull(alive)

    def concentration_at(self, pos):
        if self._n == 0:
            return 0.0
        dx = pos[0] - self.positions[:self._n, 0]
        dy = pos[1] - self.positions[:self._n, 1]
        r2 = dx * dx + dy * dy
        sigma2 = np.maximum(self.sigmas[:self._n] ** 2, self.min_sigma ** 2)

        if cfg.FILAMENT_WALL_OCCLUSION:
            visible = r2 < 9.0 * sigma2
            if np.any(visible):
                qgx, qgy = self.grid.world_to_grid(pos[0], pos[1])
                for i in np.where(visible)[0]:
                    fgx, fgy = self.grid.world_to_grid(
                        self.positions[i, 0], self.positions[i, 1]
                    )
                    if not self._bresenham_clear(fgx, fgy, qgx, qgy):
                        visible[i] = False
            contrib = (
                self.masses[:self._n][visible]
                / (2.0 * np.pi * sigma2[visible])
                * np.exp(-r2[visible] / (2.0 * sigma2[visible]))
            )
        else:
            contrib = (
                self.masses[:self._n]
                / (2.0 * np.pi * sigma2)
                * np.exp(-r2 / (2.0 * sigma2))
            )
        return float(np.sum(contrib))

    def get_all_filaments(self):
        n = self._n
        return {
            "positions": self.positions[:n].copy(),
            "sigmas":    self.sigmas[:n].copy(),
            "masses":    self.masses[:n].copy(),
            "ages":      self.ages[:n].copy(),
            "velocities": self.velocities[:n].copy(),
        }

    @property
    def n_active(self):
        return self._n

    # ------------------------------------------------------------------
    # Internals (mostly unchanged from baseline)
    # ------------------------------------------------------------------

    def _reset_arrays(self):
        self.positions = np.zeros((self._capacity, 2), dtype=np.float64)
        self.sigmas    = np.zeros(self._capacity, dtype=np.float64)
        self.masses    = np.zeros(self._capacity, dtype=np.float64)
        self.ages      = np.zeros(self._capacity, dtype=np.int64)
        self.velocities = np.zeros((self._capacity, 2), dtype=np.float64)
        self._n = 0

    def _release(self, n):
        """Release n filaments at the source. Reject any that land in a wall."""
        # Defensive: skip if source is inside a wall.
        sgx, sgy = self.grid.world_to_grid(self.source_pos[0], self.source_pos[1])
        if not (0 <= sgx < self.grid.grid_width and 0 <= sgy < self.grid.grid_height):
            return
        if self.grid.grid[sgy, sgx] != 0:
            return

        needed = self._n + n
        if needed > self._capacity:
            self._capacity = max(self._capacity * 2, needed)
            self.positions.resize((self._capacity, 2), refcheck=False)
            self.sigmas.resize(self._capacity, refcheck=False)
            self.masses.resize(self._capacity, refcheck=False)
            self.ages.resize(self._capacity, refcheck=False)
            self.velocities.resize((self._capacity, 2), refcheck=False)

        idx = slice(self._n, self._n + n)
        self.positions[idx] = self.source_pos
        self.sigmas[idx] = self.initial_sigma
        self.masses[idx] = self.mass
        self.ages[idx] = 0
        self._n = needed

    # The rest is identical to baseline. Reflection/normal/path logic copied verbatim.
    def _handle_obstacles(self, pre_positions):
        n = self._n
        if n == 0:
            return
        res = self.grid.resolution
        W = self.grid.grid_width
        H = self.grid.grid_height

        post_gx = np.floor(self.positions[:n, 0] / res).astype(np.int64)
        post_gy = np.floor(self.positions[:n, 1] / res).astype(np.int64)
        pre_gx  = np.floor(pre_positions[:n, 0]  / res).astype(np.int64)
        pre_gy  = np.floor(pre_positions[:n, 1]  / res).astype(np.int64)

        cell_changed = (pre_gx != post_gx) | (pre_gy != post_gy)
        out_of_bounds = (post_gx < 0) | (post_gx >= W) | (post_gy < 0) | (post_gy >= H)
        cgx = np.clip(post_gx, 0, W - 1)
        cgy = np.clip(post_gy, 0, H - 1)
        in_wall_post = out_of_bounds | (self.grid.grid[cgy, cgx] != 0)

        candidates = np.where(cell_changed | in_wall_post)[0]
        if candidates.size == 0:
            return

        n_s = self._tunnel_check_steps
        t = np.linspace(0.0, 1.0, n_s + 1)[1:]
        px = pre_positions[candidates, 0]
        py = pre_positions[candidates, 1]
        qx = self.positions[candidates, 0]
        qy = self.positions[candidates, 1]

        sx = px[:, None] + t[None, :] * (qx - px)[:, None]
        sy = py[:, None] + t[None, :] * (qy - py)[:, None]
        sgx = np.floor(sx / res).astype(np.int64)
        sgy = np.floor(sy / res).astype(np.int64)

        s_oob = (sgx < 0) | (sgx >= W) | (sgy < 0) | (sgy >= H)
        sgx_c = np.clip(sgx, 0, W - 1)
        sgy_c = np.clip(sgy, 0, H - 1)
        s_in_wall = s_oob | (self.grid.grid[sgy_c, sgx_c] != 0)

        any_blocked = s_in_wall.any(axis=1)
        if not any_blocked.any():
            return

        first_hit = s_in_wall.argmax(axis=1)
        blocked_ci = np.where(any_blocked)[0]
        blocked_fi = candidates[blocked_ci]
        fh = first_hit[blocked_ci]

        wall_gx = sgx[blocked_ci, fh]
        wall_gy = sgy[blocked_ci, fh]

        normals = self._estimate_normals_batch(wall_gx, wall_gy)
        v = self.velocities[blocked_fi]
        v_dot_n = (v * normals).sum(axis=1, keepdims=True)
        v_ref = v - 2.0 * v_dot_n * normals

        pre_b = pre_positions[blocked_fi]
        ref_pos = pre_b + v_ref * self.reflection_energy * self.dt

        ref_gx = np.floor(ref_pos[:, 0] / res).astype(np.int64)
        ref_gy = np.floor(ref_pos[:, 1] / res).astype(np.int64)
        ref_oob = (ref_gx < 0) | (ref_gx >= W) | (ref_gy < 0) | (ref_gy >= H)
        ref_gx_c = np.clip(ref_gx, 0, W - 1)
        ref_gy_c = np.clip(ref_gy, 0, H - 1)
        ref_in_free = ~ref_oob & (self.grid.grid[ref_gy_c, ref_gx_c] == 0)

        ref_path_clear = self._paths_clear(pre_b, ref_pos)
        use_reflection = ref_in_free & ref_path_clear
        self.positions[blocked_fi] = np.where(
            use_reflection[:, None], ref_pos, pre_b
        )

    def _estimate_normals_batch(self, wall_gx, wall_gy):
        W = self.grid.grid_width
        H = self.grid.grid_height
        offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
        dirs = offsets.astype(np.float64)
        ngx = wall_gx[:, None] + offsets[None, :, 0]
        ngy = wall_gy[:, None] + offsets[None, :, 1]
        valid = (ngx >= 0) & (ngx < W) & (ngy >= 0) & (ngy < H)
        ngx_c = np.clip(ngx, 0, W - 1)
        ngy_c = np.clip(ngy, 0, H - 1)
        is_free = valid & (self.grid.grid[ngy_c, ngx_c] == 0)
        normal = (is_free[:, :, None] * dirs[None, :, :]).sum(axis=1)
        norms = np.linalg.norm(normal, axis=1, keepdims=True)
        has_normal = (norms > 0).ravel()
        normal[has_normal] /= norms[has_normal]
        n_stuck = int((~has_normal).sum())
        if n_stuck:
            thetas = self.rng.uniform(0.0, 2.0 * np.pi, size=n_stuck)
            normal[~has_normal] = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
        return normal

    def _paths_clear(self, starts, ends):
        n_s = self._tunnel_check_steps
        t = np.linspace(0.0, 1.0, n_s + 1)[1:]
        res = self.grid.resolution
        W = self.grid.grid_width
        H = self.grid.grid_height
        sx = starts[:, 0:1] + t * (ends[:, 0:1] - starts[:, 0:1])
        sy = starts[:, 1:2] + t * (ends[:, 1:2] - starts[:, 1:2])
        sgx = np.floor(sx / res).astype(np.int64)
        sgy = np.floor(sy / res).astype(np.int64)
        s_oob = (sgx < 0) | (sgx >= W) | (sgy < 0) | (sgy >= H)
        sgx_c = np.clip(sgx, 0, W - 1)
        sgy_c = np.clip(sgy, 0, H - 1)
        s_in_wall = s_oob | (self.grid.grid[sgy_c, sgx_c] != 0)
        return ~s_in_wall.any(axis=1)

    def _bresenham_first_wall(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if not (0 <= x0 < self.grid.grid_width and 0 <= y0 < self.grid.grid_height):
                return (x0, y0)
            if self.grid.grid[y0, x0] != 0:
                return (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy
        return None

    def _bresenham_clear(self, x0, y0, x1, y1):
        return self._bresenham_first_wall(x0, y0, x1, y1) is None

    def _cull(self, alive_mask):
        if not np.all(alive_mask):
            n_alive = int(np.sum(alive_mask))
            self.positions[:n_alive] = self.positions[:self._n][alive_mask]
            self.sigmas[:n_alive] = self.sigmas[:self._n][alive_mask]
            self.masses[:n_alive] = self.masses[:self._n][alive_mask]
            self.ages[:n_alive] = self.ages[:self._n][alive_mask]
            self.velocities[:n_alive] = self.velocities[:self._n][alive_mask]
            self._n = n_alive
