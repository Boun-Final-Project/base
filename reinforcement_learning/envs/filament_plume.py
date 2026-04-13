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
        wind = self.wind_velocity
        turb_sigma = self.turbulence_scale * self.wind_speed
        turbulence = self.rng.normal(0.0, turb_sigma, size=(self._n, 2))
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

        Two checks are performed for each filament:
            A. **Tunnelling check** — a Bresenham ray from pre- to post-position
               detects filaments that jumped completely through a thin wall
               without their centre ever landing in a wall cell.
            B. **Overlap check** — the post-position itself is inside a wall.

        For each blocked filament the normal is estimated at the *wall cell*
        that caused the block (not the post-position, which may be in free
        space on the far side of the wall).  The filament is then reflected
        from its pre-move position.

        Absorbed filaments are marked with ``age = max_age + 1`` so that
        the culling step removes them.
        """
        n = self._n
        if n == 0:
            return

        res = self.grid.resolution

        # Vectorized world→grid for all filaments (replaces 2n scalar calls)
        pre_gx = np.floor(pre_positions[:n, 0] / res).astype(np.int64)
        pre_gy = np.floor(pre_positions[:n, 1] / res).astype(np.int64)
        post_gx = np.floor(self.positions[:n, 0] / res).astype(np.int64)
        post_gy = np.floor(self.positions[:n, 1] / res).astype(np.int64)

        # Filaments that didn't change grid cell cannot have tunnelled and are
        # unlikely to be inside a wall — skip them early.
        cell_changed = (pre_gx != post_gx) | (pre_gy != post_gy)

        # Vectorized in-wall check for post-positions.
        clamped_gx = np.clip(post_gx, 0, self.grid.grid_width - 1)
        clamped_gy = np.clip(post_gy, 0, self.grid.grid_height - 1)
        out_of_bounds = (
            (post_gx < 0) | (post_gx >= self.grid.grid_width) |
            (post_gy < 0) | (post_gy >= self.grid.grid_height)
        )
        in_wall = out_of_bounds | (self.grid.grid[clamped_gy, clamped_gx] != 0)

        candidates = np.where(cell_changed | in_wall)[0]
        if candidates.size == 0:
            return

        blocked = np.zeros(n, dtype=bool)
        # Grid coords of the wall cell that caused the block (for normal
        # estimation).  Only meaningful when blocked[i] is True.
        wall_gx = np.zeros(n, dtype=np.int64)
        wall_gy = np.zeros(n, dtype=np.int64)

        for i in candidates:
            # Check A: did the filament tunnel through a wall?
            hit = self._bresenham_first_wall(
                pre_gx[i], pre_gy[i], post_gx[i], post_gy[i]
            )
            if hit is not None:
                blocked[i] = True
                wall_gx[i], wall_gy[i] = hit
                continue

            # Check B: did the filament land inside a wall?
            if in_wall[i]:
                blocked[i] = True
                wall_gx[i] = post_gx[i]
                wall_gy[i] = post_gy[i]

        if not np.any(blocked):
            return

        # Handle each blocked filament
        for i in np.where(blocked)[0]:
            vx, vy = self.velocities[i]
            v = np.array([vx, vy], dtype=np.float64)

            # Estimate surface normal at the wall cell (not the post-position)
            wall_x = (wall_gx[i] + 0.5) * res
            wall_y = (wall_gy[i] + 0.5) * res
            normal = self._estimate_normal(wall_x, wall_y)

            # Reflect: v' = v - 2(v·n)n
            v_dot_n = np.dot(v, normal)
            v_reflected = v - 2.0 * v_dot_n * normal

            # Apply reflected displacement from the pre-move position, so
            # the filament starts outside the wall before bouncing.
            # Scale by reflection_energy so energy loss actually shortens travel.
            reflected_pos = pre_positions[i] + v_reflected * self.reflection_energy * self.dt
            if self.grid.is_valid(
                position=(reflected_pos[0], reflected_pos[1]),
                radius=0.0,
            ):
                # Verify the reflected path doesn't also cross a wall
                ref_gx, ref_gy = self.grid.world_to_grid(
                    reflected_pos[0], reflected_pos[1]
                )
                if self._bresenham_first_wall(pre_gx[i], pre_gy[i], ref_gx, ref_gy) is None:
                    self.positions[i] = reflected_pos
                else:
                    # Reflection also crosses a wall — snap back
                    self.positions[i] = pre_positions[i]
            else:
                # Reflected position is inside a wall — snap back
                self.positions[i] = pre_positions[i]

    def _estimate_normal(self, x, y):
        """Estimate the outward surface normal at a point inside an obstacle.

        Checks the 4 cardinal neighbors in the occupancy grid. The normal
        is the average direction toward free cells, normalized.

        Parameters
        ----------
        x, y : float
            World coordinates of the point (inside an obstacle).

        Returns
        -------
        ndarray
            Unit vector pointing toward free space, shape (2,).
        """
        gx, gy = self.grid.world_to_grid(x, y)

        free_dirs = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ng = gx + dx
            ngy = gy + dy
            if 0 <= ng < self.grid.grid_width and 0 <= ngy < self.grid.grid_height:
                if self.grid.grid[ngy, ng] == 0:
                    free_dirs.append(np.array([float(dx), float(dy)]))

        if free_dirs:
            n = np.mean(free_dirs, axis=0)
            norm = np.linalg.norm(n)
            if norm > 0:
                return n / norm
            return np.array([1.0, 0.0], dtype=np.float64)

        # Completely surrounded — no valid reflection direction.
        # This is rare (filament trapped in a 3×3 wall pocket).
        # Return a random direction so the filament tries to escape.
        theta = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

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
