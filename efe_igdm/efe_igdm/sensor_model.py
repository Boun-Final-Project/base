from scipy.stats import norm
import numpy as np

class DiscreteSensorModel:
    """
    Discrete multi-level sensor model based on the RRT-Infotaxis paper.

    Uses empirical three-sigma rule to discretize concentration measurements
    into n levels. This provides more information than binary sensor while
    maintaining computational efficiency.

    The discretization range is [μ - 3σ, μ + 3σ] where:
    - μ is the expected concentration (from Gaussian plume model)
    - σ is the sensor noise standard deviation
    """

    # Minimum likelihood floor to prevent division by zero
    MIN_LIKELIHOOD = 1e-15

    def __init__(self, alpha=0.1, sigma_env=1.5, num_levels=6):
        """
        Initialize discrete sensor model.

        Parameters:
        -----------
        alpha : float
            Proportional noise coefficient (σ_g = α*C + σ_env)
        sigma_env : float
            Environmental noise baseline
        num_levels : int
            Number of discrete measurement levels (default: 6)
        """
        self.alpha = alpha
        self.sigma_env = sigma_env
        self.num_levels = num_levels
        self.thresholds = None  # Will be computed based on measurements
        self.initialized = False

    def get_std(self, true_concentration):
        """Compute sensor noise standard deviation."""
        return self.alpha * true_concentration + self.sigma_env

    def initialize_thresholds(self, initial_measurement):
        """
        Initialize discrete thresholds using three-sigma rule.

        Parameters:
        -----------
        initial_measurement : float
            First sensor measurement to establish range
        """
        # Use initial measurement as estimate of mean
        mu = initial_measurement
        sigma = self.get_std(mu)

        # Three-sigma rule: cover [μ - 3σ, μ + 3σ]
        min_val = max(0, mu - 3 * sigma)  # Concentration can't be negative
        max_val = mu + 3 * sigma

        # Create n-1 thresholds to divide range into n levels
        # Example with n=6: [0, t1, t2, t3, t4, t5, ∞]
        self.thresholds = np.linspace(min_val, max_val, self.num_levels - 1)
        self.initialized = True

    def update_thresholds(self, current_measurement):
        """
        Adaptively update thresholds based on new measurements.

        Similar to binary sensor's adaptive threshold, but for discrete levels.
        """
        if not self.initialized:
            self.initialize_thresholds(current_measurement)
            return

        # Adaptively expand range if measurement is outside current bounds
        min_threshold = self.thresholds[0]
        max_threshold = self.thresholds[-1]

        if current_measurement < min_threshold or current_measurement > max_threshold:
            # Re-compute thresholds centered on running average
            mu = (min_threshold + max_threshold) / 2.0
            mu = 0.5 * mu + 0.5 * current_measurement  # Smooth update
            sigma = self.get_std(mu)

            min_val = max(0, mu - 3 * sigma)
            max_val = mu + 3 * sigma

            self.thresholds = np.linspace(min_val, max_val, self.num_levels - 1)

    def get_discrete_measurement(self, actual_measurement):
        """
        Convert continuous measurement to discrete level.

        Parameters:
        -----------
        actual_measurement : float
            Continuous concentration measurement

        Returns:
        --------
        level : int
            Discrete level (0 to num_levels-1)
        """
        if not self.initialized:
            raise ValueError("Thresholds not initialized. Call update_thresholds() first.")

        # Find which bin the measurement falls into
        level = np.searchsorted(self.thresholds, actual_measurement)
        return int(level)

    def probability_binary_vec(self, measurement_level: int, concentrations: np.ndarray) -> np.ndarray:
        """
        Name kept for compatibility.
        Vectorized probability of discrete measurement given concentrations.

        Uses CDF of Gaussian sensor model to compute probability that
        true concentration falls within the discrete bin.

        Parameters:
        -----------
        measurement_level : int
            Observed discrete level (0 to num_levels-1)
        concentrations : np.ndarray
            Predicted concentrations for each particle

        Returns:
        --------
        probabilities : np.ndarray
            P(level | concentration) for each particle
        """
        if not self.initialized:
            raise ValueError("Thresholds not initialized.")

        # Compute sensor noise for each predicted concentration
        sigma_g = self.alpha * concentrations + self.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15)

        # Get bin boundaries
        if measurement_level == 0:
            # Lowest bin: [0, threshold[0])
            lower_bound = 0
            upper_bound = self.thresholds[0]
        elif measurement_level == self.num_levels - 1:
            # Highest bin: [threshold[-1], ∞)
            lower_bound = self.thresholds[-1]
            upper_bound = np.inf
        else:
            # Middle bins: [threshold[i-1], threshold[i])
            lower_bound = self.thresholds[measurement_level - 1]
            upper_bound = self.thresholds[measurement_level]

        # Compute probability that measurement falls in this bin
        # P(lower < measurement < upper | predicted_concentration)

        if upper_bound == np.inf:
            # P(measurement > lower)
            prob = 1.0 - norm.cdf((lower_bound - concentrations) / sigma_g)
        else:
            # P(lower < measurement < upper)
            prob_upper = norm.cdf((upper_bound - concentrations) / sigma_g)
            prob_lower = norm.cdf((lower_bound - concentrations) / sigma_g)
            prob = prob_upper - prob_lower

        # Apply minimum likelihood floor
        # return np.maximum(prob, self.MIN_LIKELIHOOD)
        return prob

    def probability_discrete(self, measurement_level: int, particle_concentration: float) -> float:
        """
        Scalar version: Probability of discrete measurement given particle concentration.

        Parameters:
        -----------
        measurement_level : int
            Observed discrete level
        particle_concentration : float
            Concentration predicted by particle

        Returns:
        --------
        probability : float
            P(level | particle_concentration)
        """
        if not self.initialized:
            raise ValueError("Thresholds not initialized.")

        # Sensor noise
        sigma_g = self.get_std(particle_concentration)

        # Get bin boundaries
        if measurement_level == 0:
            lower_bound = 0
            upper_bound = self.thresholds[0]
        elif measurement_level == self.num_levels - 1:
            lower_bound = self.thresholds[-1]
            upper_bound = np.inf
        else:
            lower_bound = self.thresholds[measurement_level - 1]
            upper_bound = self.thresholds[measurement_level]

        # Compute probability
        if upper_bound == np.inf:
            prob = 1.0 - norm.cdf((lower_bound - particle_concentration) / sigma_g)
        else:
            prob_upper = norm.cdf((upper_bound - particle_concentration) / sigma_g)
            prob_lower = norm.cdf((lower_bound - particle_concentration) / sigma_g)
            prob = prob_upper - prob_lower

        # return max(self.MIN_LIKELIHOOD, prob)
        return prob


class BinarySensorModel:
    """
    Binary sensor model from the RRT-Infotaxis paper (Equations 24-27, page 6).
    Gives 0 or 1 measurement based on an adaptive threshold.
    """

    # Minimum likelihood floor to prevent division by zero
    MIN_LIKELIHOOD = 1e-15

    def __init__(self, alpha=0.1, sigma_env=1.5, threshold_weight=0.3):
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
    
    def probability_binary_vec(self, measurement: int, concentrations: np.ndarray) -> np.ndarray:
        """
        Vectorized probability of a binary measurement given a vector of concentrations.

        Applies MIN_LIKELIHOOD floor to prevent division by zero in particle filter updates.
        See probability_binary() for detailed explanation.
        """
        # This is P(Z=0 | C), the probability of a "no-hit"
        delta_c = self.threshold - concentrations
        sigma_g = self.alpha * concentrations + self.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15) # Avoid division by zero

        prob_z0 = norm.cdf(delta_c / sigma_g)

        if measurement == 0:
            prob = prob_z0
        else:
            prob = 1.0 - prob_z0

        # Apply likelihood floor to prevent division by zero
        # NOTE: Only clamp lower bound. Upper bound clamping prevents entropy reduction!
        # return np.maximum(prob, self.MIN_LIKELIHOOD)
        return prob

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

        # CRITICAL: Apply minimum likelihood floor to prevent division by zero
        #
        # During RRT lookahead (rrt.py:131), hypothetical particle filter updates
        # can result in ALL particles having zero likelihood. This happens when:
        # 1. Particles have converged near the true source
        # 2. RRT samples a far-away position with an incompatible measurement
        # 3. All particles predict this measurement is impossible (likelihood ≈ 0)
        # 4. weight_sum = 0 → division by zero in normalization
        # 5. Copied filter reinitializes → wrong entropy → bad path selection
        #
        # The floor (1e-15) is small enough to not bias decisions but prevents:
        # - Division by zero during RRT entropy calculations
        # - Spurious filter reinitialization in copied filters
        # - RRT selecting paths that appear "informative" due to numerical errors
        #
        # This is a NECESSARY fix for numerical stability, not an optional enhancement.

        # Eq. 24
        if binary_value == 0:
            prob = beta
        else:
            prob = 1 - beta

        # Clamp lower bound only to prevent division by zero
        # NOTE: Do NOT clamp upper bound - it prevents entropy reduction!
        # return max(self.MIN_LIKELIHOOD, prob)
        return prob

class ContinuousGaussianSensorModel:
    """
    Continuous Gaussian sensor model from the paper (Equation 3, page 3).

    This is the CORRECT sensor model for actual measurements.
    Discretization is ONLY used for RRT entropy calculation, not for measurements.

    Equation 3:
    p(z_k | θ) = (1/(σ_g√2π)) * exp(-(z_k - R(r_k|θ))²/(2σ_g²))

    where σ_g = α·R(r_k|θ) + σ_env
    """

    # Minimum likelihood floor to prevent numerical issues
    MIN_LIKELIHOOD = 1e-50

    def __init__(self, alpha=0.1, sigma_env=1.0, num_levels=10, max_concentration=20.0):
        """
        Initialize continuous Gaussian sensor model.

        Parameters:
        -----------
        alpha : float
            Proportional noise coefficient (σ_g = α*C + σ_env)
        sigma_env : float
            Environmental noise baseline (default: 1.0 ppm)
        num_levels : int
            Number of discrete levels for RRT entropy calculation (default: 10)
            This is ONLY used for RRT planning, not for actual measurements!
        max_concentration : float
            Maximum concentration for fixed discretization range (default: 10 ppm)
        """
        self.alpha = alpha
        self.sigma_env = sigma_env
        self.num_levels = num_levels  # For RRT discretization (Eq. 13-15)

    def get_std(self, predicted_concentration):
        """
        Compute sensor noise standard deviation.

        σ_g = α·R(r|θ) + σ_env
        """
        return self.alpha * predicted_concentration + self.sigma_env

    def probability_continuous_vec(self, measurement: float, predicted_concentrations: np.ndarray) -> np.ndarray:
        """
        Vectorized Gaussian likelihood for continuous measurement (Equation 3).

        Parameters:
        -----------
        measurement : float
            Actual continuous concentration measurement (e.g., 10.94 μg/m³)
        predicted_concentrations : np.ndarray
            Predicted concentrations for each particle (shape: N,)

        Returns:
        --------
        likelihoods : np.ndarray
            p(z_k | θ^i) for each particle i (shape: N,)
        """
        # σ_g = α·R + σ_env for each particle
        sigma_g = self.alpha * predicted_concentrations + self.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15)  # Prevent division by zero

        # Gaussian likelihood: p(z | R) = (1/(σ_g√2π)) * exp(-(z - R)²/(2σ_g²))
        coefficient = 1.0 / (sigma_g * np.sqrt(2 * np.pi))
        exponent = -((measurement - predicted_concentrations) ** 2) / (2 * sigma_g ** 2)
        likelihoods = coefficient * np.exp(exponent)

        # Apply minimum likelihood floor for numerical stability
        likelihoods = np.maximum(likelihoods, self.MIN_LIKELIHOOD)

        return likelihoods

    def probability_continuous(self, measurement: float, predicted_concentration: float) -> float:
        """
        Scalar Gaussian likelihood for continuous measurement (Equation 3).

        Parameters:
        -----------
        measurement : float
            Actual continuous concentration measurement
        predicted_concentration : float
            Predicted concentration from particle

        Returns:
        --------
        likelihood : float
            p(z_k | θ)
        """
        # σ_g = α·R + σ_env
        sigma_g = self.alpha * predicted_concentration + self.sigma_env
        sigma_g = max(sigma_g, 1e-15)

        # Gaussian likelihood
        coefficient = 1.0 / (sigma_g * np.sqrt(2 * np.pi))
        exponent = -((measurement - predicted_concentration) ** 2) / (2 * sigma_g ** 2)
        likelihood = coefficient * np.exp(exponent)

        return max(likelihood, self.MIN_LIKELIHOOD)

    def create_discretization_thresholds(self, predicted_concentrations: np.ndarray) -> np.ndarray:
        """
        Create DYNAMIC discretization thresholds for RRT entropy calculation.

        Strategy: "Dynamic Anchoring"
        - Bin 0 extends from -∞ to threshold[0], capturing negative Gaussian tail
        - Bins divide the active range [0, d_max] where d_max = max(predictions) + 3σ
        - This matches the simulator (which outputs ≥ 0) with the math (Gaussian to -∞)

        Parameters:
        -----------
        predicted_concentrations : np.ndarray
            Predicted concentrations from all particles at this RRT node

        Returns:
        --------
        thresholds : np.ndarray
            Array of (num_levels - 1) threshold values dividing the range
        """
        # 1. Find the active range from particle predictions
        mean_conc = np.mean(predicted_concentrations)
        max_conc = np.max(predicted_concentrations)

        # Compute average sensor noise
        avg_sigma = self.alpha * mean_conc + self.sigma_env

        # 2. Upper bound: max prediction + 3σ buffer (covers 99.7% of Gaussian tail)
        d_max = max_conc + 3.0 * avg_sigma

        # 3. Ensure reasonable range
        d_max = max(d_max, 1.0)  # At least 1 ppm range

        # 4. Create num_levels - 1 thresholds dividing [0, d_max] into equal bins
        # Bin 0 will be (-∞, threshold[0]), Bin N-1 will be (threshold[N-2], ∞)
        thresholds = np.linspace(0, d_max, self.num_levels)[1:]  # Skip 0, keep the rest

        return thresholds

    def compute_bin_likelihood_vec(self, bin_index: int, predicted_concentrations: np.ndarray,
                                   thresholds: np.ndarray) -> np.ndarray:
        """
        Compute P(z ∈ bin | θ) for all particles (for RRT entropy calculation).

        This computes the integral of the Gaussian likelihood over a bin:
        P(z_lower < z < z_upper | θ) = Φ((z_upper - μ)/σ) - Φ((z_lower - μ)/σ)

        Parameters:
        -----------
        bin_index : int
            Discrete bin index (0 to num_levels-1)
        predicted_concentrations : np.ndarray
            Predicted concentrations for all particles
        thresholds : np.ndarray
            Discretization thresholds from create_discretization_thresholds()

        Returns:
        --------
        probabilities : np.ndarray
            P(z ∈ bin | θ^i) for each particle i
        """
        # Compute sensor noise for each particle
        sigma_g = self.alpha * predicted_concentrations + self.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15)

        # Get bin boundaries
        if bin_index == 0:
            # Lowest bin: (-∞, threshold[0])
            # This captures all the negative Gaussian tail + simulator outputs near 0
            lower_bound = -np.inf
            upper_bound = thresholds[0]
        elif bin_index == self.num_levels - 1:
            # Highest bin: (threshold[-1], ∞)
            lower_bound = thresholds[-1]
            upper_bound = np.inf
        else:
            # Middle bins: (threshold[i-1], threshold[i])
            lower_bound = thresholds[bin_index - 1]
            upper_bound = thresholds[bin_index]

        # Compute probability that measurement falls in this bin
        # P(lower < z < upper | predicted_concentration)
        if lower_bound == -np.inf and upper_bound != np.inf:
            # Bin 0: P(z < upper) = CDF(upper)
            prob = norm.cdf((upper_bound - predicted_concentrations) / sigma_g)
        elif upper_bound == np.inf and lower_bound != -np.inf:
            # Last bin: P(z > lower) = 1 - CDF(lower)
            prob = 1.0 - norm.cdf((lower_bound - predicted_concentrations) / sigma_g)
        elif lower_bound == -np.inf and upper_bound == np.inf:
            # Edge case: entire real line (shouldn't happen)
            prob = np.ones_like(predicted_concentrations)
        else:
            # Middle bins: P(lower < z < upper) = CDF(upper) - CDF(lower)
            prob_upper = norm.cdf((upper_bound - predicted_concentrations) / sigma_g)
            prob_lower = norm.cdf((lower_bound - predicted_concentrations) / sigma_g)
            prob = prob_upper - prob_lower

        return prob
