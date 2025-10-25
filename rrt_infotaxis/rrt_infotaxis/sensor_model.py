from scipy.stats import norm

class BinarySensorModel:
    """
    Binary sensor model from the RRT-Infotaxis paper (Equations 24-27, page 6).
    Gives 0 or 1 measurement based on an adaptive threshold.
    """

    def __init__(self, alpha=0.1, sigma_env=1.5, threshold_weight=0.5):
        self.alpha = alpha
        self.sigma_env = sigma_env
        self.threshold_weight = threshold_weight
        self.threshold = None  # Adaptive threshold c̄_k

    def get_std(self, true_concentration):
        return self.alpha * true_concentration + self.sigma_env

    def initialize_threshold(self, initial_measurement):
        """Initialize the adaptive threshold with the first measurement."""
        self.threshold = initial_measurement

    def update_threshold(self, current_measurement):
        """
        Update adaptive threshold based on Eq. 27 from the paper.

        The threshold c̄_k is updated as:
        - c̄_k = a·c_k + (1-a)·c̄_{k-1}  if k > 1 and c_k > c̄_{k-1}
        - c̄_k = c̄_{k-1}                 if k > 1 and c_k ≤ c̄_{k-1}
        - c̄_k = c_k                      if k = 1

        This makes the threshold increase only when new measurements exceed
        the current threshold, encouraging the agent to move toward higher
        concentrations (exploitation).

        Parameters:
        -----------
        current_measurement : float
            Current sensor measurement c_k
        """
        if self.threshold is None:
            self.initialize_threshold(current_measurement)
        elif current_measurement > self.threshold:
            self.threshold = (self.threshold_weight * current_measurement +
                            (1 - self.threshold_weight) * self.threshold)
        # else: threshold remains unchanged

    def get_binary_measurement(self, actual_measurement):
        """Convert to binary measurement based on the adaptive threshold."""
        if self.threshold is None:
            raise ValueError("Threshold not initialized. Call update_threshold() first.")
        return 1 if actual_measurement > self.threshold else 0

    def probability_binary(self, binary_value, particle_concentration):
        """
        Gives likelihood of binary measurement given particle concentration.
        Particle concentration is found by using gas model with specific particle state.

        From Eq. 24-26 in the paper:
        p(b̂_{k+n}|θ_k) = β     if b̂_{k+n} = 0
                       = 1-β   if b̂_{k+n} = 1

        where β = Φ(Δc̄_k^i / σ_g,k^i) from Eq. 25
        Δc̄_k^i = c̄_k - R(r_k|θ_k^i)  from Eq. 26
        σ_g,k^i = α·R(r_k|θ_k^i) + σ_env

        Φ(·) is the CDF of the standard normal distribution.
        """
        if self.threshold is None:
            raise ValueError("Threshold not initialized. Call update_threshold() first.")

        # Eq. 26: Δc̄^i = c̄ - R(r|θ^i)
        delta_c = self.threshold - particle_concentration

        # Eq. 26: σ_g^i = α·R(r|θ^i) + σ_env
        sigma_g = self.get_std(particle_concentration)

        # Eq. 25: β = Φ(Δc̄^i / σ_g^i)
        # Use CDF of standard normal distribution
        beta = norm.cdf(delta_c / sigma_g)

        # Eq. 24
        if binary_value == 0:
            return beta
        else:
            return 1 - beta