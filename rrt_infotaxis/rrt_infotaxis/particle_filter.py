import numpy as np
from scipy.stats import multivariate_normal
from .gaussian_plume import GaussianPlumeModel
from .sensor_model import BinarySensorModel
from copy import deepcopy

class ParticleFilter:
    """
    Particle filter for gas source term estimation.
    Estimates source parameters θ = [x₀, y₀, Q₀] source location and release rate.
    Uses Sequential Importance Sampling (SIS) with resampling and MCMC move steps.
    """

    def __init__(self, num_particles : int, search_bounds : dict[str, list[float]], binary_sensor_model : BinarySensorModel, 
                 dispersion_model : GaussianPlumeModel, resample_threshold : float = 0.5, mcmc_std=None):
        """
        Parameters:
        -----------
        num_particles : int
            Number of particles N
        search_bounds : dict
            Search space bounds: {'x': [xmin, xmax], 'y': [ymin, ymax], 'Q': [Qmin, Qmax]}
        binary_sensor_model : BinarySensorModel
            Binary sensor model for computing measurement likelihood
        dispersion_model : GaussianPlumeModel
            Gaussian plume model for predicting concentrations
        resample_threshold : float
            Resample when N_eff < resample_threshold * N (default: 0.5)
        mcmc_std : dict or None
            Standard deviations for MCMC move step: {'x': σx, 'y': σy, 'Q': σQ}
            If None, uses 5% of search range
        """
        self.N = num_particles
        self.bounds = search_bounds
        self.sensor_model = binary_sensor_model
        self.dispersion_model = dispersion_model
        self.resample_threshold = resample_threshold

        # Set MCMC standard deviations
        if mcmc_std is None:
            x_range = search_bounds['x'][1] - search_bounds['x'][0]
            y_range = search_bounds['y'][1] - search_bounds['y'][0]
            Q_range = search_bounds['Q'][1] - search_bounds['Q'][0]
            self.mcmc_std = {
                'x': 0.05 * x_range,
                'y': 0.05 * y_range,
                'Q': 0.05 * Q_range
            }
        else:
            self.mcmc_std = mcmc_std

        # Initialize particles uniformly in search space
        self.particles = self._initialize_particles()
        self.weights = np.ones(self.N) / self.N

        # Statistics
        self.iteration = 0

        # Store last measurement for MCMC
        self.last_measurement = None
        self.last_sensor_position = None

    def _initialize_particles(self):
        """Initialize particles uniformly in search space."""
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']
        Q_min, Q_max = self.bounds['Q']

        particles = np.zeros((self.N, 3))
        particles[:, 0] = np.random.uniform(x_min, x_max, self.N)  # x₀
        particles[:, 1] = np.random.uniform(y_min, y_max, self.N)  # y₀
        particles[:, 2] = np.random.uniform(Q_min, Q_max, self.N)  # Q₀

        return particles

    def update(self, measurement : int, sensor_position : tuple[float, float]):
        """
        Update particle filter with new binary measurement (Equations 5-9 from paper).

        Parameters:
        -----------
        measurement : int
            Binary sensor measurement (0 or 1)
        sensor_position : tuple or array
            Sensor position (x, y)
        """
        self.iteration += 1

        # Store measurement for MCMC
        self.last_measurement = measurement
        self.last_sensor_position = sensor_position

        # Step 1: Compute likelihood for each particle (Eq. 8)
        likelihoods = self._compute_likelihoods(measurement, sensor_position)

        # Step 2: Update weights (Eq. 8): w̄ᵢₖ = p(zₖ|θⁱₖ₋₁) · wⁱₖ₋₁
        self.weights *= likelihoods

        # Step 3: Normalize weights (Eq. 9)
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All weights are zero - reinitialize
            print("Warning: All weights are zero. Reinitializing particles.")
            self.particles = self._initialize_particles()
            self.weights = np.ones(self.N) / self.N
            return

        # Step 4: Check effective sample size and resample if needed
        N_eff = self._effective_sample_size()
        if N_eff < self.resample_threshold * self.N:
            self._resample()
            self._mcmc_move()

    def _compute_likelihoods(self, measurement : int, sensor_position : tuple[float, float]):
        """
        Compute binary measurement likelihood for each particle.

        Parameters:
        -----------
        measurement : int
            Binary sensor measurement (0 or 1)
        sensor_position : tuple
            Sensor position (x, y)

        Returns:
        --------
        likelihoods : np.ndarray, shape (N,)
            Likelihood p(z|θⁱ) for each particle
        """
        likelihoods = np.zeros(self.N)

        for i in range(self.N):
            x0, y0, Q0 = self.particles[i]

            # Compute predicted concentration for this particle
            predicted_conc = self.dispersion_model.compute_concentration((sensor_position[0], sensor_position[1]), (x0, y0), Q0)

            # Compute likelihood using sensor model
            likelihood = self.sensor_model.probability_binary(measurement, predicted_conc)

            likelihoods[i] = likelihood

        return likelihoods

    def _effective_sample_size(self):
        """Compute effective sample size N_eff ≈ 1 / Σ(wⁱ)²."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """Resample particles using systematic resampling."""
        # Systematic resampling
        indices = self._systematic_resample()

        # Resample particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def _systematic_resample(self):
        """
        Systematic resampling algorithm.

        Returns:
        --------
        indices : np.ndarray
            Indices of resampled particles
        """
        positions = (np.arange(self.N) + np.random.random()) / self.N
        cumulative_sum = np.cumsum(self.weights)

        indices = np.zeros(self.N, dtype=int)
        i, j = 0, 0

        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def _mcmc_move(self):
        """
        Markov Chain Monte Carlo (MCMC) move step to increase particle diversity.

        Uses Metropolis-Hastings algorithm to perturb particles while respecting
        the posterior distribution. This is crucial after resampling to avoid
        particle depletion.

        The acceptance probability is:
        α = min(1, π(θ')/π(θ)) = min(1, [p(z|θ') · p(θ')] / [p(z|θ) · p(θ)])

        With uniform prior p(θ) = constant, this simplifies to:
        α = min(1, p(z|θ') / p(z|θ)) = min(1, likelihood_ratio)
        """
        if self.last_measurement is None or self.last_sensor_position is None:
            # No measurement yet, skip MCMC
            return

        for i in range(self.N):
            current_particle = self.particles[i].copy()

            # Propose new particle state with Gaussian random walk
            proposal = current_particle.copy()
            proposal[0] += np.random.normal(0, self.mcmc_std['x'])  # x₀
            proposal[1] += np.random.normal(0, self.mcmc_std['y'])  # y₀
            proposal[2] += np.random.normal(0, self.mcmc_std['Q'])  # Q₀

            # Check bounds - reject if out of bounds
            if not self._in_bounds(proposal):
                continue

            # Compute likelihood for current particle
            x0_curr, y0_curr, Q0_curr = current_particle
            conc_curr = self.dispersion_model.compute_concentration(
                self.last_sensor_position, (x0_curr, y0_curr), Q0_curr
            )
            likelihood_curr = self.sensor_model.probability_binary(
                self.last_measurement, conc_curr
            )

            # Compute likelihood for proposal
            x0_prop, y0_prop, Q0_prop = proposal
            conc_prop = self.dispersion_model.compute_concentration(
                self.last_sensor_position, (x0_prop, y0_prop), Q0_prop
            )
            likelihood_prop = self.sensor_model.probability_binary(
                self.last_measurement, conc_prop
            )

            # Metropolis-Hastings acceptance probability
            # α = min(1, likelihood_proposal / likelihood_current)
            if likelihood_curr > 1e-10:  # Avoid division by zero
                acceptance_prob = min(1.0, likelihood_prop / likelihood_curr)
            else:
                # Current likelihood is ~0, accept proposal if it has any likelihood
                acceptance_prob = 1.0 if likelihood_prop > 1e-10 else 0.5

            if np.random.random() < acceptance_prob:
                self.particles[i] = proposal

    def _in_bounds(self, particle : np.ndarray):
        """Check if particle is within bounds."""
        x0, y0, Q0 = particle

        x_valid = self.bounds['x'][0] <= x0 <= self.bounds['x'][1]
        y_valid = self.bounds['y'][0] <= y0 <= self.bounds['y'][1]
        Q_valid = self.bounds['Q'][0] <= Q0 <= self.bounds['Q'][1]

        return x_valid and y_valid and Q_valid

    def get_estimate(self):
        """
        Get current source term estimate (weighted mean).

        Returns:
        --------
        estimate : dict
            Estimated source parameters: {'x': x₀, 'y': y₀, 'Q': Q₀}
        std : dict
            Standard deviations: {'x': σx, 'y': σy, 'Q': σQ}
        """
        # Weighted mean
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        Q_est = np.sum(self.weights * self.particles[:, 2])

        # Weighted standard deviation
        x_var = np.sum(self.weights * (self.particles[:, 0] - x_est) ** 2)
        y_var = np.sum(self.weights * (self.particles[:, 1] - y_est) ** 2)
        Q_var = np.sum(self.weights * (self.particles[:, 2] - Q_est) ** 2)

        estimate = {'x': x_est, 'y': y_est, 'Q': Q_est}
        std = {'x': np.sqrt(x_var), 'y': np.sqrt(y_var), 'Q': np.sqrt(Q_var)}

        return estimate, std

    def get_entropy(self):
        """
        Compute Shannon entropy of particle distribution (Eq. 11-13).
        H_k = -Σ wⁱ log(wⁱ)
        """
        # Avoid log(0)
        weights_safe = self.weights[self.weights > 1e-10]
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return entropy

    def predict_measurement_probability(self, sensor_position, binary_value=None):
        """
        Predict probability of future binary measurement (for RRT-Infotaxis utility).

        For binary sensor: β = Σ[Φ(Δc̄ᵢ/σᵍᵢ) · wⁱ] (Equation 25)

        Parameters:
        -----------
        sensor_position : tuple
            Future sensor position (x, y)
        binary_value : int or None
            Binary value to predict (0 or 1). If None, returns both.

        Returns:
        --------
        probability : float or tuple
            p(b=binary_value) or (p(b=0), p(b=1))
        """
        from scipy.stats import norm as scipy_norm

        # Compute β = Σ[Φ(Δc̄ᵢ/σᵍᵢ) · wⁱ]
        beta = 0.0
        for i in range(self.N):
            x0, y0, Q0 = self.particles[i]
            predicted_conc = self.dispersion_model.compute_concentration(sensor_position, (x0, y0), Q0)

            delta_c = self.sensor_model.threshold - predicted_conc
            sigma_g = self.sensor_model.get_std(predicted_conc)
            phi = scipy_norm.cdf(delta_c / sigma_g)

            beta += phi * self.weights[i]

        if binary_value is None:
            return beta, 1 - beta  # p(b=0), p(b=1)
        elif binary_value == 0:
            return beta
        else:
            return 1 - beta

    def get_particles(self):
        """Get current particles and weights."""
        return self.particles.copy(), self.weights.copy()
    
    def copy(self) -> "ParticleFilter":
        """Create a deep copy of the particle filter."""
        return deepcopy(self)