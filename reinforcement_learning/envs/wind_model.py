"""
Simple uniform wind field for the Python pretraining environment.
Each episode samples a constant wind direction and speed.

Optionally, a pre-computed spatial wind field (H x W x 2 array) can be
provided so that WindModel also serves as the ``wind_field`` argument to
FilamentPlume during training (bilinear-interpolated queries).
"""

import numpy as np


class WindModel:
    """Per-episode constant uniform wind field, with optional spatial support."""

    def __init__(self, speed_range=(0.1, 1.5), max_speed=2.0,
                 field=None, resolution=None, occupancy=None):
        """
        Parameters
        ----------
        speed_range : tuple
            (min, max) wind speed in m/s, sampled uniformly per episode.
        max_speed : float
            Maximum possible speed, used for observation normalization.
        field : array-like, optional
            Spatial wind field of shape (H, W, 2) [u, v] in m/s.
            When provided the object also exposes ``query()`` and
            ``max_speed()`` so it can be passed as ``wind_field`` to
            FilamentPlume.
        resolution : float, optional
            Cell size in metres (required when field is not None).
        occupancy : array-like of bool, optional
            Boolean occupancy map of shape (H, W); True = wall cell
            (required when field is not None).
        """
        self.speed_range = speed_range
        self._max_speed_uniform = max_speed
        self.speed = 0.0
        self.direction = 0.0  # radians, 0 = +x

        if field is not None:
            self._field = np.asarray(field, dtype=np.float64)
            self._resolution = float(resolution)
            self._occupancy = np.asarray(occupancy, dtype=bool)
            self.H, self.W, _ = self._field.shape
            speeds = np.linalg.norm(self._field, axis=2)
            free = ~self._occupancy
            self._max_speed_spatial = float(speeds[free].max()) if free.any() else 0.0
        else:
            self._field = None
            self._max_speed_spatial = 0.0

    def randomize(self, rng=None):
        """Sample a new wind configuration for an episode.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.speed = rng.uniform(*self.speed_range)
        self.direction = rng.uniform(0, 2 * np.pi)

    def set_uniform(self, speed, direction):
        """Pin wind to a deterministic (speed, direction). Used by GADEN eval
        to drive the policy ctx vector from the spatial mean of a real wind
        field while the plume itself uses the spatial field for advection.
        """
        self.speed = float(speed)
        self.direction = float(direction)

    def get_observation(self):
        """Return normalized (speed, direction) for the state vector.

        Returns
        -------
        obs : tuple
            (speed / max_speed, direction / 2pi), both in [0, 1].
        """
        return (self.speed / self._max_speed_uniform, self.direction / (2 * np.pi))

    def get_dispersion_offset(self, dispersion_factor):
        """Compute the downwind offset applied to the IGDM source position.

        The wind pushes the effective concentration peak downwind.
        A sensor at position P measuring source at S will use an
        effective source position S' = S + offset, so that locations
        downwind of S receive higher concentration.

        Parameters
        ----------
        dispersion_factor : float
            Scaling factor controlling how strongly wind shifts the peak.

        Returns
        -------
        offset : np.ndarray
            (dx, dy) offset in meters.
        """
        dx = self.speed * dispersion_factor * np.cos(self.direction)
        dy = self.speed * dispersion_factor * np.sin(self.direction)
        return np.array([dx, dy])

    # ------------------------------------------------------------------
    # Spatial field interface (used by FilamentPlume when wind_field=self)
    # ------------------------------------------------------------------

    def query(self, positions: np.ndarray) -> np.ndarray:
        """Bilinear interpolation over the spatial wind field.

        Parameters
        ----------
        positions : (N, 2) array
            World-frame positions [x, y] in metres.

        Returns
        -------
        wind : (N, 2) array
            Interpolated [u, v] wind vectors in m/s.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]   # (2,) → (1, 2)
        if positions.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

        x = positions[:, 0]
        y = positions[:, 1]
        res = self._resolution

        c_frac = x / res - 0.5
        r_frac = y / res - 0.5

        c0 = np.floor(c_frac).astype(np.int64)
        r0 = np.floor(r_frac).astype(np.int64)
        c1 = c0 + 1
        r1 = r0 + 1
        tx = c_frac - c0
        ty = r_frac - r0

        # Bounds check on pre-clamp indices — clamped indices outside these
        # ranges are meaningless.
        valid_c0 = (c0 >= 0) & (c0 <= self.W - 1)
        valid_c1 = (c1 >= 0) & (c1 <= self.W - 1)
        valid_r0 = (r0 >= 0) & (r0 <= self.H - 1)
        valid_r1 = (r1 >= 0) & (r1 <= self.H - 1)

        c0c = np.clip(c0, 0, self.W - 1)
        c1c = np.clip(c1, 0, self.W - 1)
        r0c = np.clip(r0, 0, self.H - 1)
        r1c = np.clip(r1, 0, self.H - 1)

        # Gate wall lookup with bounds flags so out-of-bounds clamped indices
        # are never trusted.
        wall_00 = self._occupancy[r0c, c0c] & valid_r0 & valid_c0
        wall_01 = self._occupancy[r0c, c1c] & valid_r0 & valid_c1
        wall_10 = self._occupancy[r1c, c0c] & valid_r1 & valid_c0
        wall_11 = self._occupancy[r1c, c1c] & valid_r1 & valid_c1

        valid_00 = valid_r0 & valid_c0 & ~wall_00
        valid_01 = valid_r0 & valid_c1 & ~wall_01
        valid_10 = valid_r1 & valid_c0 & ~wall_10
        valid_11 = valid_r1 & valid_c1 & ~wall_11

        w00 = np.where(valid_00, (1 - tx) * (1 - ty), 0.0)
        w01 = np.where(valid_01,       tx * (1 - ty), 0.0)
        w10 = np.where(valid_10, (1 - tx) *       ty, 0.0)
        w11 = np.where(valid_11,       tx *       ty, 0.0)

        w_sum = w00 + w01 + w10 + w11
        safe  = w_sum > 0
        inv_sum = 1.0 / np.where(safe, w_sum, 1.0)

        w00 = w00 * inv_sum
        w01 = w01 * inv_sum
        w10 = w10 * inv_sum
        w11 = w11 * inv_sum

        rows_corners = np.stack([r0c, r0c, r1c, r1c], axis=1)   # (N, 4)
        cols_corners = np.stack([c0c, c1c, c0c, c1c], axis=1)   # (N, 4)
        wind_corners = self._field[rows_corners, cols_corners]   # (N, 4, 2)

        weights = np.stack([w00, w01, w10, w11], axis=1)[:, :, None]  # (N, 4, 1)
        return (wind_corners * weights).sum(axis=1)                    # (N, 2)

    def max_speed(self) -> float:
        """Max wind speed across free cells (spatial) or uniform max (no field)."""
        if self._field is None:
            return self._max_speed_uniform
        return self._max_speed_spatial

    def get_local_wind(self, position: np.ndarray) -> np.ndarray:
        """Normalized local wind (Ux, Uy) at position, mapped to [0, 1].

        Encoding: (component / max_speed_uniform + 1) / 2
        so 0.5 = no wind, 1.0 = full positive, 0.0 = full negative.
        Uses _max_speed_uniform as the scale in both spatial and uniform modes
        for consistent encoding across training and eval.
        """
        if self._field is not None:
            uv = self.query(np.atleast_2d(position))[0]
        else:
            uv = np.array([self.speed * np.cos(self.direction),
                           self.speed * np.sin(self.direction)])
        ms = self._max_speed_uniform if self._max_speed_uniform > 0 else 1.0
        return np.clip((uv / ms + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def make_training_wind_field(grid, rng, speed_range=(0.1, 1.5),
                              max_speed=2.0) -> "WindModel":
    """Build a wall-aware spatial WindModel for one training episode.

    Fills every free cell with a constant wind vector sampled from
    (speed, direction), zeros wall cells, and wraps the result in a
    WindModel so it can serve as wind_field for FilamentPlume.
    """
    H, W = grid.grid.shape
    res = grid.resolution
    occupancy = (grid.grid != 0)

    speed = float(rng.uniform(*speed_range))
    direction = float(rng.uniform(0, 2 * np.pi))
    ux = speed * np.cos(direction)
    uy = speed * np.sin(direction)

    field = np.zeros((H, W, 2), dtype=np.float64)
    field[~occupancy, 0] = ux
    field[~occupancy, 1] = uy

    model = WindModel(field=field, resolution=res, occupancy=occupancy,
                      max_speed=max_speed)
    # Populate .speed / .direction so FilamentPlume's turbulence args and
    # get_local_wind's uniform fallback have the sampled episode values.
    model.set_uniform(speed, direction)
    return model
