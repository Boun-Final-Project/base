from scipy.stats import norm
from .sensor_interface import SensorModel
import numpy as np

class ContinuousGaussianSensorModel(SensorModel):
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
        self.max_concentration = max_concentration

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
    
    def compute_discretized_distribution(self, predicted_concentrations: np.ndarray) -> np.ndarray:
        """
        Compute probability mass for ALL discretization bins for ALL particles at once.
        
        This replaces the manual CDF math previously done in ParticleFilter.
        
        Returns:
        --------
        bin_probs : np.ndarray
            Shape (num_levels, N). 
            Row i contains the probability that the measurement falls in Bin i 
            for every particle.
        """
        # 1. Generate thresholds based on current predictions
        inner_thresholds = self.create_discretization_thresholds(predicted_concentrations)
        
        # 2. define all bin edges: [-inf, t1, t2, ..., inf]
        # Shape: (num_levels + 1,)
        bin_edges = np.concatenate([[-np.inf], inner_thresholds, [np.inf]])

        # 3. Compute sensor noise for every particle
        # Shape: (N,)
        sigma_g = self.alpha * predicted_concentrations + self.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15)

        # 4. Compute Z-scores matrix
        # Broadcasting: (Num_Edges, 1) - (1, N) -> (Num_Edges, N)
        z_scores = (bin_edges[:, None] - predicted_concentrations[None, :]) / sigma_g[None, :]

        # 5. Compute CDFs
        cdfs = norm.cdf(z_scores)

        # 6. Compute bin probabilities (CDF[i+1] - CDF[i])
        # Shape: (num_levels, N)
        bin_probs_per_particle = cdfs[1:, :] - cdfs[:-1, :]

        return bin_probs_per_particle

    def compute_likelihood(self, measurement: float, predictions: np.ndarray) -> np.ndarray:
        return self.probability_continuous_vec(measurement, predictions)

    def compute_predictive_distribution(self, predictions: np.ndarray) -> np.ndarray:
        return self.compute_discretized_distribution(predictions)

    def compute_likelihood_for_bin(self, bin_index: int, predictions: np.ndarray) -> np.ndarray:
        thresholds = self.create_discretization_thresholds(predictions)
        return self.compute_bin_likelihood_vec(bin_index, predictions, thresholds)