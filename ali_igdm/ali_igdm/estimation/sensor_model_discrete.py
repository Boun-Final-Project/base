"""
Discrete 5-level Sensor Model for gas concentration measurements.
Provides better concentration information than binary sensor (~2.3 bits/measurement vs 1 bit).
"""

import numpy as np
from scipy.stats import norm as scipy_norm


class DiscreteSensorModel:
    """5-level discrete sensor model for detecting gas concentration levels."""

    def __init__(self, alpha=0.1, sigma_env=0.1, threshold_weight=0.5):
        """
        Parameters:
        -----------
        alpha : float
            Proportional noise coefficient
        sigma_env : float
            Environmental noise standard deviation
        threshold_weight : float
            Weight for threshold update (0-1)
        """
        self.alpha = alpha
        self.sigma_env = sigma_env
        self.threshold_weight = threshold_weight
        self.threshold = None
        self.level_thresholds = None  # Will store 4 thresholds for 5 levels

    def get_std(self, true_concentration):
        """Get measurement standard deviation for given concentration."""
        return self.alpha * true_concentration + self.sigma_env

    def initialize_threshold(self, initial_measurement):
        """Initialize sensor threshold with first measurement.

        Sets up 4 threshold levels that divide the concentration space into 5 regions.
        Initial spacing is linear: 0.25x, 0.50x, 0.75x, 1.00x of first measurement.
        Uses minimum threshold of 0.1 to avoid degeneracy when first measurement is near zero.
        """
        # Ensure minimum threshold of 0.1 to avoid degenerate case when first measurement is very low
        self.threshold = max(initial_measurement, 0.1)
        # Initialize level thresholds equally spaced
        # Levels: 0 (< t1), 1 (t1-t2), 2 (t2-t3), 3 (t3-t4), 4 (> t4)
        self.level_thresholds = [
            self.threshold * 0.25,
            self.threshold * 0.50,
            self.threshold * 0.75,
            self.threshold * 1.00
        ]

    def update_threshold(self, current_measurement):
        """Update threshold and level thresholds (only increases if measurement > current threshold).

        This ensures that thresholds monotonically increase as the robot encounters
        higher concentrations, similar to the adaptive threshold in BinarySensorModel.
        """
        if self.threshold is None:
            self.initialize_threshold(current_measurement)
        elif current_measurement > self.threshold:
            old_threshold = self.threshold
            self.threshold = (self.threshold_weight * current_measurement +
                            (1 - self.threshold_weight) * self.threshold)

            # Update level thresholds proportionally to maintain spacing
            scale_factor = self.threshold / old_threshold if old_threshold > 0 else 1.0
            self.level_thresholds = [t * scale_factor for t in self.level_thresholds]

    def get_measurement_levels(self):
        """Get list of possible measurement values for this sensor.

        Returns:
        --------
        levels : list
            List of possible measurement values (e.g., [0, 1, 2, 3, 4] for discrete)
        """
        return [0, 1, 2, 3, 4]

    def get_discrete_measurement(self, actual_measurement):
        """Convert continuous measurement to 5 discrete levels (0-4).

        Parameters:
        -----------
        actual_measurement : float
            Raw continuous concentration measurement

        Returns:
        --------
        level : int
            Discretized level (0-4)

        Levels:
        - 0: Concentration < level_thresholds[0] (very low)
        - 1: level_thresholds[0] <= C < level_thresholds[1] (low)
        - 2: level_thresholds[1] <= C < level_thresholds[2] (medium)
        - 3: level_thresholds[2] <= C < level_thresholds[3] (high)
        - 4: C >= level_thresholds[3] (very high)
        """
        if self.level_thresholds is None:
            raise ValueError("Thresholds not initialized.")

        for i, threshold in enumerate(self.level_thresholds):
            if actual_measurement < threshold:
                return i
        return 4  # Maximum level

    def probability_discrete(self, discrete_level, particle_concentration):
        """Compute probability of discrete measurement given particle state.

        Uses the normal CDF to compute the probability that a particle at
        `particle_concentration` would produce a measurement in the given level.

        Parameters:
        -----------
        discrete_level : int
            Discrete measurement level (0-4)
        particle_concentration : float
            Predicted concentration from particle

        Returns:
        --------
        probability : float
            Likelihood of measurement given particle state (clamped to [1e-10, 1-1e-10])
        """
        if self.level_thresholds is None:
            raise ValueError("Thresholds not initialized.")

        sigma_g = self.get_std(particle_concentration)
        MIN_LIKELIHOOD = 1e-10

        # Compute probability for each level as area under normal distribution
        if discrete_level == 0:
            # P(C < level_thresholds[0])
            prob = scipy_norm.cdf((self.level_thresholds[0] - particle_concentration) / sigma_g)
        elif discrete_level == 4:
            # P(C >= level_thresholds[3])
            prob = 1.0 - scipy_norm.cdf((self.level_thresholds[3] - particle_concentration) / sigma_g)
        else:
            # P(level_thresholds[i] <= C < level_thresholds[i+1])
            lower_threshold = self.level_thresholds[discrete_level - 1]
            upper_threshold = self.level_thresholds[discrete_level]

            prob_lower = scipy_norm.cdf((lower_threshold - particle_concentration) / sigma_g)
            prob_upper = scipy_norm.cdf((upper_threshold - particle_concentration) / sigma_g)
            prob = prob_upper - prob_lower

        return max(MIN_LIKELIHOOD, min(1.0 - MIN_LIKELIHOOD, prob))
