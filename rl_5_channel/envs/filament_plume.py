"""
Lagrangian filament-based gas dispersion model.

Represents a gas plume as a collection of discrete "blobs" (filaments) that:
  - Advect with the mean wind
  - Meander via turbulent fluctuations (random walk)
  - Diffuse (grow in size) over time
  - Reflect off obstacles

Concentration at any point is the sum of 2D Gaussian contributions from all
active filaments. This produces intermittent, bursty sensor readings that
closely match real turbulent plume behavior.

All filament state is stored in structured NumPy arrays for vectorized
operations — tens of thousands of filaments can be simulated in real-time.
"""

import numpy as np

from .. import config as cfg


class FilamentPlume:
    """Lagrangian filament-based gas dispersion model.

    Manages a population of gas filaments that advect with wind,
    meander via turbulence, diffuse (grow), and reflect off walls.

    Filament state is stored in parallel NumPy arrays for vectorized
    update and concentration queries::

        positions : (N, 2)  x, y in meters
        sigmas    : (N,)    standard deviation in meters
        masses    : (N,)    per-filament mass (arbitrary units)
        ages      : (N,)    age in steps
        velocities: (N, 2)  vx, vy in m/s

    Concentration at a point is the sum of 2D Gaussians::

        C(x) = Σ_i  m_i / (2π σ_i²)  exp(-|x - x_i|² / (2 σ_i²))
    """

    def __init__(
        self,
        source_pos,
        wind_speed,
        wind_angle,
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
        rng=None,
        wind_field=None,
    ):
        """
        Parameters
        ----------
        source_pos : tuple or ndarray
            (x, y) source location in meters.
        wind_speed : float
            Wind speed in m/s.
        wind_angle : float
            Wind direction in radians (0 = +x axis).
        occupancy_grid : OccupancyGrid
            Grid for obstacle collision / reflection checks.
        dt : float, optional
            Timestep in seconds. Defaults to ``cfg.FILAMENT_DT``.
        K : float, optional
            Atmospheric diffusivity in m²/s. Defaults to ``cfg.FILAMENT_K``.
        turbulence_scale : float, optional
            Turbulence as fraction of wind speed. Defaults to
            ``cfg.FILAMENT_TURBULENCE_SCALE``.
        max_age : int, optional
            Max filament lifetime in steps. Defaults to ``cfg.FILAMENT_MAX_AGE``.
        filaments_per_step : int, optional
            New filaments released per update. Defaults to
            ``cfg.FILAMENTS_PER_STEP``.
        initial_sigma : float, optional
            Initial filament std deviation in meters. Defaults to
            ``cfg.FILAMENT_INITIAL_SIGMA``.
        mass : float, optional
            Per-filament mass. Defaults to ``cfg.FILAMENT_MASS``.
        min_sigma : float, optional
            Minimum σ to prevent division by zero. Defaults to
            ``cfg.FILAMENT_MIN_SIGMA``.
        reflection_energy : float, optional
            Velocity retention factor on wall bounce. Defaults to
            ``cfg.FILAMENT_REFLECTION_ENERGY``.
        rng : np.random.Generator, optional
            Random number generator for determinism.
        """
        self.source_pos = np.asarray(source_pos, dtype=np.float64)
        self.wind_velocity = np.array(
            [wind_speed * np.cos(wind_angle), wind_speed * np.sin(wind_angle)],
            dtype=np.float64,
        )
        self.wind_speed = wind_speed
        self.grid = occupancy_grid
        # Optional spatially-varying wind field. When set, advection queries
        # this field per-filament instead of using the uniform wind_velocity.
        # The (wind_speed, wind_angle) args still drive the policy ctx vector
        # via the env's WindModel — they're set to the field's spatial mean.
        self._wind_field = wind_field

        # Physics parameters
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

        self.rng = rng if rng is not None else np.random.default_rng()

        # Precompute the number of parametric samples needed per path so that
        # no grid cell can be skipped.  Worst case: axis-aligned displacement of
        # max_speed * dt; we need step_size < resolution.
        # Using 3-sigma turbulence as a practical upper bound on speed.
        _max_speed = self._wind_field.max_speed() if self._wind_field is not None else self.wind_speed
        _max_disp  = _max_speed * (1.0 + 3.0 * self.turbulence_scale) * self.dt
        self._tunnel_check_steps = max(
            8, int(np.ceil(_max_disp / self.grid.resolution)) + 2
        )

        # Filament state: parallel arrays that grow/shrink via culling
        # Pre-allocate with generous capacity to avoid frequent reallocation.
        # Typical steady state is ~240 filaments (2/step × 120 max age).
        self._capacity = max(1024, self.max_age * self.filaments_per_step * 2)
        self._reset_arrays()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self):
        """Advance all filaments by one timestep.

        Steps:
            1. Release new filaments at source.
            2. Advect with wind + meander via turbulence.
            3. Diffuse (grow σ).
            4. Reflect off obstacles.
            5. Age and cull expired / absorbed filaments.
        """
        # 1. Release
        n_new = self.filaments_per_step
        self._release(n_new)

        if self._n == 0:
            return

        # 2. Advection + meandering
        if self._wind_field is not None:
            wind = self._wind_field.query(self.positions[:self._n])    # (n, 2)
            speed_per = np.linalg.norm(wind, axis=1)                   # (n,)
            turb_sigma = self.turbulence_scale * speed_per[:, None]    # (n, 1)
        else:
            wind = self.wind_velocity                                  # (2,)
            turb_sigma = self.turbulence_scale * self.wind_speed       # scalar
        turbulence = self.rng.normal(0.0, 1.0, size=(self._n, 2)) * turb_sigma
        displacement = (wind + turbulence) * self.dt
        pre_positions = self.positions[:self._n].copy()
        self.positions[:self._n, 0] += displacement[:, 0]
        self.positions[:self._n, 1] += displacement[:, 1]

        # Store velocities for reflection computation
        self.velocities[:self._n] = wind + turbulence

        # 3. Diffusion: σ²_new = σ²_old + 2K·dt
        self.sigmas[:self._n] = np.sqrt(
            self.sigmas[:self._n] ** 2 + 2.0 * self.K * self.dt
        )

        # 4. Obstacle reflection
        self._handle_obstacles(pre_positions)

        # 5. Age and cull
        self.ages[:self._n] += 1
        alive = self.ages[:self._n] < self.max_age
        self._cull(alive)

    def concentration_at(self, pos):
        """Compute total concentration at a point.

        Sums 2D Gaussian contributions from all active filaments::

            C = Σ  m / (2πσ²) · exp(-r² / (2σ²))

        When ``cfg.FILAMENT_WALL_OCCLUSION`` is True, filaments with no clear
        line of sight to *pos* (i.e. a wall cell lies on the Bresenham line
        between them) are excluded.  Filaments beyond 3σ are pre-filtered
        before ray-casting since their contribution is negligible
        (< 1.1 % of peak).

        Parameters
        ----------
        pos : tuple or ndarray
            (x, y) query position in meters.

        Returns
        -------
        float
            Total concentration (arbitrary units).
        """
        if self._n == 0:
            return 0.0

        dx = pos[0] - self.positions[:self._n, 0]
        dy = pos[1] - self.positions[:self._n, 1]
        r2 = dx * dx + dy * dy
        sigma2 = np.maximum(self.sigmas[:self._n] ** 2, self.min_sigma ** 2)

        if cfg.FILAMENT_WALL_OCCLUSION:
            # Pre-filter to 3-sigma radius before running ray-casts.
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
            # 2D Gaussian kernel — no occlusion check
            contrib = (
                self.masses[:self._n]
                / (2.0 * np.pi * sigma2)
                * np.exp(-r2 / (2.0 * sigma2))
            )
        return float(np.sum(contrib))

    def get_all_filaments(self):
        """Return a copy of all active filament data for visualization.

        Returns
        -------
        dict
            Keys: ``positions`` (N, 2), ``sigmas`` (N,), ``masses`` (N,),
            ``ages`` (N,), ``velocities`` (N, 2).
        """
        n = self._n
        return {
            "positions": self.positions[:n].copy(),
            "sigmas": self.sigmas[:n].copy(),
            "masses": self.masses[:n].copy(),
            "ages": self.ages[:n].copy(),
            "velocities": self.velocities[:n].copy(),
        }

    @property
    def n_active(self):
        """Number of currently active filaments."""
        return self._n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_arrays(self):
        """Allocate / reset parallel filament arrays."""
        self.positions = np.zeros((self._capacity, 2), dtype=np.float64)
        self.sigmas = np.zeros(self._capacity, dtype=np.float64)
        self.masses = np.zeros(self._capacity, dtype=np.float64)
        self.ages = np.zeros(self._capacity, dtype=np.int64)
        self.velocities = np.zeros((self._capacity, 2), dtype=np.float64)
        self._n = 0

    def _release(self, n):
        """Release *n* new filaments at the source position."""
        needed = self._n + n
        if needed > self._capacity:
            # Grow capacity (rare after initial warmup)
            self._capacity = max(self._capacity * 2, needed)
            self.positions.resize((self._capacity, 2), refcheck=False)
            self.sigmas.resize(self._capacity, refcheck=False)
            self.masses.resize(self._capacity, refcheck=False)
            self.ages.resize(self._capacity, refcheck=False)
            self.velocities.resize((self._capacity, 2), refcheck=False)

        idx = slice(self._n, self._n + n)
        self.positions[idx] = self.source_pos  # (n, 2) via broadcast
        self.sigmas[idx] = self.initial_sigma
        self.masses[idx] = self.mass
        self.ages[idx] = 0
        self._n = needed

    def _handle_obstacles(self, pre_positions):
        """Reflect filaments that crossed or landed inside obstacles.

        Both the tunnelling check (filament jumped through a thin wall) and the
        overlap check (filament landed inside a wall) are performed in a single
        vectorized pass: the path from pre- to post-position is sampled at
        ``_tunnel_check_steps`` equally-spaced points and all sample grid lookups
        are executed as one NumPy array operation over all candidates.

        For each blocked filament the surface normal is estimated at the first
        wall cell hit.  Reflection and reflected-path validation are also
        vectorized.  Filaments whose reflected path is also blocked are snapped
        back to their pre-move position.
        """
        n = self._n
        if n == 0:
            return

        res = self.grid.resolution
        W = self.grid.grid_width
        H = self.grid.grid_height

        # --- Grid coordinates (vectorized) ---
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

        # --- Vectorized path sampling ---
        # Sample the path pre→post at n_s equally-spaced t values, excluding
        # t=0 (start is known free) and including t=1 (catches overlap too).
        n_s = self._tunnel_check_steps
        t   = np.linspace(0.0, 1.0, n_s + 1)[1:]          # (n_s,)

        px = pre_positions[candidates, 0]                   # (n_cand,)
        py = pre_positions[candidates, 1]
        qx = self.positions[candidates, 0]
        qy = self.positions[candidates, 1]

        # Sample world coords and convert to grid indices: (n_cand, n_s)
        sx  = px[:, None] + t[None, :] * (qx - px)[:, None]
        sy  = py[:, None] + t[None, :] * (qy - py)[:, None]
        sgx = np.floor(sx / res).astype(np.int64)
        sgy = np.floor(sy / res).astype(np.int64)

        s_oob    = (sgx < 0) | (sgx >= W) | (sgy < 0) | (sgy >= H)
        sgx_c    = np.clip(sgx, 0, W - 1)
        sgy_c    = np.clip(sgy, 0, H - 1)
        s_in_wall = s_oob | (self.grid.grid[sgy_c, sgx_c] != 0)  # (n_cand, n_s)

        any_blocked = s_in_wall.any(axis=1)                 # (n_cand,)
        if not any_blocked.any():
            return

        first_hit  = s_in_wall.argmax(axis=1)               # (n_cand,) index of first True

        blocked_ci = np.where(any_blocked)[0]               # indices into candidates
        blocked_fi = candidates[blocked_ci]                  # actual filament indices
        fh         = first_hit[blocked_ci]                   # first-hit step per blocked fil

        wall_gx = sgx[blocked_ci, fh]                       # (n_blocked,)
        wall_gy = sgy[blocked_ci, fh]

        # --- Vectorized reflection ---
        normals    = self._estimate_normals_batch(wall_gx, wall_gy)  # (n_blocked, 2)
        v          = self.velocities[blocked_fi]                     # (n_blocked, 2)
        v_dot_n    = (v * normals).sum(axis=1, keepdims=True)        # (n_blocked, 1)
        v_ref      = v - 2.0 * v_dot_n * normals                    # (n_blocked, 2)

        pre_b      = pre_positions[blocked_fi]                       # (n_blocked, 2)
        ref_pos    = pre_b + v_ref * self.reflection_energy * self.dt

        # --- Validate reflected positions ---
        ref_gx = np.floor(ref_pos[:, 0] / res).astype(np.int64)
        ref_gy = np.floor(ref_pos[:, 1] / res).astype(np.int64)

        ref_oob    = (ref_gx < 0) | (ref_gx >= W) | (ref_gy < 0) | (ref_gy >= H)
        ref_gx_c   = np.clip(ref_gx, 0, W - 1)
        ref_gy_c   = np.clip(ref_gy, 0, H - 1)
        ref_in_free = ~ref_oob & (self.grid.grid[ref_gy_c, ref_gx_c] == 0)

        ref_path_clear = self._paths_clear(pre_b, ref_pos)          # (n_blocked,)

        use_reflection = ref_in_free & ref_path_clear
        self.positions[blocked_fi] = np.where(
            use_reflection[:, None], ref_pos, pre_b
        )

    def _estimate_normals_batch(self, wall_gx, wall_gy):
        """Vectorized outward surface normal estimation for N wall cells.

        Checks the 4 cardinal neighbors of each cell. The normal is the mean
        direction toward free neighbors, normalized to unit length.

        Parameters
        ----------
        wall_gx, wall_gy : ndarray of int, shape (N,)
            Grid coordinates of the wall cells.

        Returns
        -------
        ndarray, shape (N, 2)
            Unit normals pointing toward free space.
        """
        W = self.grid.grid_width
        H = self.grid.grid_height

        # Cardinal offsets and matching float direction vectors
        offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
        dirs    = offsets.astype(np.float64)                        # (4, 2)

        # Neighbor grid coords: (N, 4)
        ngx = wall_gx[:, None] + offsets[None, :, 0]
        ngy = wall_gy[:, None] + offsets[None, :, 1]

        valid   = (ngx >= 0) & (ngx < W) & (ngy >= 0) & (ngy < H)
        ngx_c   = np.clip(ngx, 0, W - 1)
        ngy_c   = np.clip(ngy, 0, H - 1)
        is_free = valid & (self.grid.grid[ngy_c, ngx_c] == 0)       # (N, 4)

        # Sum free-neighbor direction vectors: (N, 4, 1) * (1, 4, 2) → (N, 2)
        normal = (is_free[:, :, None] * dirs[None, :, :]).sum(axis=1)

        norms      = np.linalg.norm(normal, axis=1, keepdims=True)  # (N, 1)
        has_normal = (norms > 0).ravel()

        normal[has_normal] /= norms[has_normal]

        # Completely surrounded (rare): assign a random escape direction
        n_stuck = int((~has_normal).sum())
        if n_stuck:
            thetas               = self.rng.uniform(0.0, 2.0 * np.pi, size=n_stuck)
            normal[~has_normal]  = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

        return normal

    def _paths_clear(self, starts, ends):
        """Vectorized path-clear check for N paths using parametric sampling.

        Reuses ``_tunnel_check_steps`` samples.  The start point (t=0) is
        excluded since it is known to be free (pre-move position).

        Parameters
        ----------
        starts, ends : ndarray, shape (N, 2)
            World coordinates of path start and end points.

        Returns
        -------
        ndarray of bool, shape (N,)
            True if the path contains no wall or out-of-bounds cell.
        """
        n_s = self._tunnel_check_steps
        t   = np.linspace(0.0, 1.0, n_s + 1)[1:]                   # (n_s,)

        res = self.grid.resolution
        W   = self.grid.grid_width
        H   = self.grid.grid_height

        # Sample world coords: (N, n_s)
        sx  = starts[:, 0:1] + t * (ends[:, 0:1] - starts[:, 0:1])
        sy  = starts[:, 1:2] + t * (ends[:, 1:2] - starts[:, 1:2])

        sgx  = np.floor(sx / res).astype(np.int64)
        sgy  = np.floor(sy / res).astype(np.int64)
        s_oob = (sgx < 0) | (sgx >= W) | (sgy < 0) | (sgy >= H)

        sgx_c = np.clip(sgx, 0, W - 1)
        sgy_c = np.clip(sgy, 0, H - 1)
        s_in_wall = s_oob | (self.grid.grid[sgy_c, sgx_c] != 0)

        return ~s_in_wall.any(axis=1)

    def _bresenham_first_wall(self, x0, y0, x1, y1):
        """Walk a Bresenham line and return the first wall cell hit, or None.

        Parameters
        ----------
        x0, y0 : int
            Grid coordinates of the start point.
        x1, y1 : int
            Grid coordinates of the end point.

        Returns
        -------
        tuple[int, int] or None
            ``(gx, gy)`` of the first occupied cell, or ``None`` if clear.
        """
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
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return None

    def _bresenham_clear(self, x0, y0, x1, y1):
        """Return True if no obstacle lies on the Bresenham line.

        Convenience wrapper around :meth:`_bresenham_first_wall`.
        """
        return self._bresenham_first_wall(x0, y0, x1, y1) is None

    def _cull(self, alive_mask):
        """Compact the arrays to keep only alive filaments."""
        if not np.all(alive_mask):
            n_alive = int(np.sum(alive_mask))
            self.positions[:n_alive] = self.positions[:self._n][alive_mask]
            self.sigmas[:n_alive] = self.sigmas[:self._n][alive_mask]
            self.masses[:n_alive] = self.masses[:self._n][alive_mask]
            self.ages[:n_alive] = self.ages[:self._n][alive_mask]
            self.velocities[:n_alive] = self.velocities[:self._n][alive_mask]
            self._n = n_alive
