"""
Simple uniform wind field for the Python pretraining environment.
Each episode samples a constant wind direction and speed.
"""

import numpy as np


class WindModel:
    """Per-episode constant uniform wind field."""

    def __init__(self, speed_range=(0.1, 1.5), max_speed=2.0):
        """
        Parameters
        ----------
        speed_range : tuple
            (min, max) wind speed in m/s, sampled uniformly per episode.
        max_speed : float
            Maximum possible speed, used for observation normalization.
        """
        self.speed_range = speed_range
        self.max_speed = max_speed
        self.speed = 0.0
        self.direction = 0.0  # radians, 0 = +x

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
        """Return normalized (speed, direction) for the legacy state vector."""
        return (self.speed / self.max_speed, self.direction / (2 * np.pi))

    def get_observation_spatial(self):
        """Direction-continuous form for the spatial arch.

        Returns (speed/max_speed, cos(direction), sin(direction)) — avoids
        the wrap-around discontinuity of raw angle / 2pi.
        """
        return (
            self.speed / self.max_speed,
            float(np.cos(self.direction)),
            float(np.sin(self.direction)),
        )

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
