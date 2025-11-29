"""
Binary Sensor Model for gas concentration measurements.
"""

from scipy.stats import norm as scipy_norm


class BinarySensorModel:
    """Binary sensor model for detecting gas concentration above/below threshold."""

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

    def get_std(self, true_concentration):
        """Get measurement standard deviation for given concentration."""
        return self.alpha * true_concentration + self.sigma_env

    def initialize_threshold(self, initial_measurement):
        """Initialize sensor threshold with first measurement."""
        self.threshold = initial_measurement

    def update_threshold(self, current_measurement):
        """Update threshold (Eq. 27): only increases if measurement > current threshold."""
        if self.threshold is None:
            self.initialize_threshold(current_measurement)
        elif current_measurement > self.threshold:
            self.threshold = (self.threshold_weight * current_measurement +
                            (1 - self.threshold_weight) * self.threshold)

    def get_binary_measurement(self, actual_measurement):
        """Convert continuous measurement to binary (0 or 1)."""
        if self.threshold is None:
            raise ValueError("Threshold not initialized.")
        return 1 if actual_measurement > self.threshold else 0

    def get_measurement_levels(self):
        """Get list of possible measurement values for this sensor.

        Returns:
        --------
        levels : list
            List of possible measurement values (e.g., [0, 1] for binary)
        """
        return [0, 1]

    def probability_binary(self, binary_value, particle_concentration):
        """Compute probability of binary measurement given particle state.

        Parameters:
        -----------
        binary_value : int
            Binary measurement (0 or 1)
        particle_concentration : float
            Predicted concentration from particle

        Returns:
        --------
        probability : float
            Likelihood of measurement given particle state
        """
        if self.threshold is None:
            raise ValueError("Threshold not initialized.")

        delta_c = self.threshold - particle_concentration
        sigma_g = self.get_std(particle_concentration)
        beta = scipy_norm.cdf(delta_c / sigma_g)

        MIN_LIKELIHOOD = 1e-10
        prob = beta if binary_value == 0 else (1 - beta)
        return max(MIN_LIKELIHOOD, min(1.0 - MIN_LIKELIHOOD, prob))
