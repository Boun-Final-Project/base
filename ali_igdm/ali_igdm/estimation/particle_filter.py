import numpy as np
from scipy.stats import multivariate_normal, norm as scipy_norm
from .igdm_gas_model import IndoorGaussianDispersionModel
from .sensor_interface import SensorModel
from .sensor_model import ContinuousGaussianSensorModel
from copy import deepcopy

class ParticleFilter:
    """
    A particle filter for localizing an indoor gas source using continuous measurements.
    Each particle represents a hypothesis of the source location (x₀, y₀) and release rate (Q₀).
    Attributes:
        N: Number of particles
        bounds: Search space bounds for x, y, Q
        sensor_model: Instance of ContinuousGaussianSensorModel
        dispersion_model: Instance of IndoorGaussianDispersionModel
        resample_threshold: Effective sample size threshold for resampling
        mcmc_std: Standard deviations for MCMC proposal distributions
    """
    def __init__(self, num_particles: int, search_bounds: dict[str, list[float]],
                 sensor_model: SensorModel,
                 dispersion_model: IndoorGaussianDispersionModel,
                 resample_threshold: float = 0.5, mcmc_std=None):
        self.N = num_particles
        self.bounds = search_bounds
        self.sensor_model = sensor_model
        self.dispersion_model = dispersion_model
        self.resample_threshold = resample_threshold
        self.mcmc_std = mcmc_std

        # Detect sensor type: check if it has discrete probability method
        self.is_discrete = hasattr(sensor_model, 'probability_discrete')
        # Set MCMC standard deviations
        if mcmc_std is None:
            x_range = search_bounds['x'][1] - search_bounds['x'][0]
            y_range = search_bounds['y'][1] - search_bounds['y'][0]
            Q_range = search_bounds['Q'][1] - search_bounds['Q'][0]
            # Increased from 0.05 to 0.10 for better exploration
            self.mcmc_std = {
                'x': 0.10 * x_range,
                'y': 0.10 * y_range,
                'Q': 0.10 * Q_range
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

        # IGDM: Store precomputed distance map from current sensor position
        self._current_distance_map = None
        self._current_sensor_position = None

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

    def update(self, measurement: float, sensor_position: tuple[float, float], skip_resample: bool = False, is_planning: bool = False):
        """
        Update particle filter with new measurement (supports both continuous and discrete).

        Args:
            measurement: Sensor measurement (continuous float or discrete int 0-4)
            sensor_position: Position where measurement was taken
            skip_resample: If True, skip resampling and MCMC (legacy parameter, use is_planning instead)
            is_planning: If True, ONLY update weights (skip resampling and MCMC)
                        RULE: During planning, we do NOT resample or move particles because:
                        - Resampling introduces randomness into entropy calculations
                        - MCMC is expensive and modifies particle positions
                        - The particle set must remain consistent for deterministic information gain
                        - We only need updated weights to calculate entropy for planning
        """
        self.iteration += 1

        # Store measurement for MCMC
        self.last_measurement = measurement
        self.last_sensor_position = sensor_position

        # Step 1: Compute likelihood for each particle (VECTORIZED)
        likelihoods = self._compute_likelihoods_vectorized(measurement, sensor_position)

        # Step 2: Update weights
        self.weights *= likelihoods

        # Step 3: Normalize weights
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
        # During planning: DON'T resample and DON'T do MCMC (only update weights for entropy calc)
        if is_planning:
            return  # Just return after weight update - no particle movement during planning

        # During actual measurement: resample when needed and do MCMC
        if not skip_resample:
            N_eff = self._effective_sample_size()
            if N_eff < self.resample_threshold * self.N:
                self._resample()
                self._mcmc_move()

    def _compute_concentrations(self, particles: np.ndarray, sensor_position: tuple[float, float]) -> np.ndarray:
        """
        Compute concentrations for a specific array of particle beliefs.
        Args:
            particles: Array of shape (N, 3) containing [x, y, Q]
            sensor_position: Tuple (x, y) of sensor location
        """
        particle_locations = particles[:, :2]  # (N, 2)
        release_rates = particles[:, 2]        # (N,)

        return self.dispersion_model.compute_concentrations_batch(
            sensor_position, particle_locations, release_rates
        )


    def _compute_likelihoods_vectorized(self, measurement: float, sensor_position: tuple[float, float]):
        """
        Compute measurement likelihood for all particle beliefs (supports continuous and discrete).
        """
        # Compute all predicted concentrations at once
        predicted_concs = self._compute_concentrations(self.particles, sensor_position)

        # Use appropriate probability method based on sensor type
        if self.is_discrete:
            # For discrete sensor, compute probabilities for all particles
            likelihoods = np.array([
                self.sensor_model.probability_discrete(int(measurement), conc)
                for conc in predicted_concs
            ])
        else:
            # For continuous sensor (original behavior)
            likelihoods = self.sensor_model.probability_continuous_vec(measurement, predicted_concs)

        return likelihoods

    def _effective_sample_size(self):
        """Compute effective sample size N_eff ≈ 1 / Σ(wⁱ)²."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """
        Resample particle beliefs using systematic resampling.
        """
        # 1. Perform systematic resampling
        indices = self._systematic_resample()
        self.particles = self.particles[indices]

        # 2. Reset weights
        self.weights = np.ones(self.N) / self.N

    def _systematic_resample(self):
        """
        This puts something like a ruler with N evenly spaced marks over the cumulative sum
        Therefore high weight particle beliefs get multiple samples, low weight particle beliefs get none.
        """
        # 1. Generate cumulative weights
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure last is exactly 1.0 to avoid float precision errors

        # 2. Generate systematic positions
        # (np.arange(N) + random) / N
        positions = (np.arange(self.N) + np.random.random()) / self.N

        # 3. Find indices using binary search (vectorized)
        indices = np.searchsorted(cumulative_sum, positions)

        return indices

    def _mcmc_move(self):
        """MCMC move step (FULLY VECTORIZED, supports continuous and discrete sensors)."""
        if self.last_measurement is None or self.last_sensor_position is None:
            return

        # 1. Generate all proposals at once
        proposals = self.particles.copy()
        proposals[:, 0] += np.random.normal(0, self.mcmc_std['x'], self.N)
        proposals[:, 1] += np.random.normal(0, self.mcmc_std['y'], self.N)
        proposals[:, 2] += np.random.normal(0, self.mcmc_std['Q'], self.N)

        # 2. Check bounds
        in_bounds = (
            (proposals[:, 0] >= self.bounds['x'][0]) & (proposals[:, 0] <= self.bounds['x'][1]) &
            (proposals[:, 1] >= self.bounds['y'][0]) & (proposals[:, 1] <= self.bounds['y'][1]) &
            (proposals[:, 2] >= self.bounds['Q'][0]) & (proposals[:, 2] <= self.bounds['Q'][1])
        )

        # 3. Compute likelihoods for ALL CURRENT particles at once
        conc_curr = self._compute_concentrations(
            self.particles, self.last_sensor_position
        )

        # 4. Compute likelihoods for ALL PROPOSED particles at once
        conc_prop = self._compute_concentrations(
            proposals, self.last_sensor_position
        )

        # 5. Calculate likelihoods based on sensor type
        if self.is_discrete:
            # Discrete sensor
            likelihood_curr = np.array([
                self.sensor_model.probability_discrete(int(self.last_measurement), conc)
                for conc in conc_curr
            ])
            likelihood_prop = np.array([
                self.sensor_model.probability_discrete(int(self.last_measurement), conc)
                for conc in conc_prop
            ])
        else:
            # Continuous sensor (original behavior)
            likelihood_curr = self.sensor_model.probability_continuous_vec(
                self.last_measurement, conc_curr
            )
            likelihood_prop = self.sensor_model.probability_continuous_vec(
                self.last_measurement, conc_prop
            )

        # 6. Metropolis-Hastings acceptance (FULLY VECTORIZED)
        likelihood_curr = np.maximum(likelihood_curr, 1e-50)

        acceptance_prob = likelihood_prop / likelihood_curr
        acceptance_prob = np.minimum(1.0, acceptance_prob)

        # 7. Find particles to move
        rand_vals = np.random.random(self.N)
        move_mask = (rand_vals < acceptance_prob) & in_bounds

        # 8. Update particles in one operation
        self.particles[move_mask] = proposals[move_mask]

    def get_estimate(self):
        """Get current source term estimate (weighted mean)."""
        estimate = {
            'x': np.sum(self.weights * self.particles[:, 0]),
            'y': np.sum(self.weights * self.particles[:, 1]),
            'Q': np.sum(self.weights * self.particles[:, 2])
        }

        x_var = np.sum(self.weights * (self.particles[:, 0] - estimate['x']) ** 2)
        y_var = np.sum(self.weights * (self.particles[:, 1] - estimate['y']) ** 2)
        Q_var = np.sum(self.weights * (self.particles[:, 2] - estimate['Q']) ** 2)

        std = {'x': np.sqrt(x_var), 'y': np.sqrt(y_var), 'Q': np.sqrt(Q_var)}

        return estimate, std

    def get_entropy(self):
        """Compute Shannon entropy of particle distribution."""
        weights_safe = self.weights[self.weights > 1e-15]
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return entropy

    def compute_hypothetical_entropy(self, measurement: int, sensor_position: tuple[float, float]):
        """
        Compute hypothetical entropy after a measurement WITHOUT modifying filter state.
        """
        # Compute likelihoods by discretizing hypothetical measurements
        predicted_concs = self._compute_concentrations(self.particles, sensor_position)
        thresholds = self.sensor_model.create_discretization_thresholds(predicted_concs)
        likelihoods = self.sensor_model.compute_bin_likelihood_vec(
            measurement, predicted_concs, thresholds
        )

        hypothetical_weights = self.weights * likelihoods

        weight_sum = np.sum(hypothetical_weights)
        if weight_sum > 0:
            hypothetical_weights /= weight_sum
        else:
            return np.log(self.N)

        weights_safe = hypothetical_weights[hypothetical_weights > 1e-15]
        entropy = -np.sum(weights_safe * np.log(weights_safe))

        return entropy
    
    def predict_measurement_probability(self, sensor_position, measurement_value=None):
        """
        Predict probability of future measurement (supports both continuous and discrete sensors).

        Parameters:
        -----------
        sensor_position : tuple
            (x, y) position where measurement would be made
        measurement_value : {None, int}, optional
            If None, return probabilities for all levels
            If int, return probability of that specific level

        Returns:
        --------
        probability : float or array
            For discrete sensor: (P(0), P(1), P(2), P(3), P(4)) or P(measurement_value)
            For continuous sensor: probabilities per bin
        """
        # 1. Get Concentration Predictions
        predicted_concs = self._compute_concentrations(self.particles, sensor_position)

        if self.is_discrete:
            # For discrete sensor, compute probability for all 5 levels
            level_probs = np.zeros(5)

            for i in range(self.N):
                predicted_conc = predicted_concs[i]

                for level in range(5):
                    prob_level = self.sensor_model.probability_discrete(level, predicted_conc)
                    level_probs[level] += prob_level * self.weights[i]

            if measurement_value is None:
                return level_probs
            else:
                return level_probs[measurement_value]
        else:
            # For continuous sensor (original behavior)
            # 2. Ask Sensor Model for Probabilities (The "How")
            # Returns matrix of shape (Num_Bins, Num_Particles)
            bin_probs_per_particle = self.sensor_model.compute_discretized_distribution(predicted_concs)

            # 3. Marginalize over particle weights
            # (Num_Bins, N) @ (N,) -> (Num_Bins,)
            level_probs = bin_probs_per_particle @ self.weights

            if measurement_value is None:
                return level_probs
            else:
                return level_probs[measurement_value]

    def get_particles(self):
        """Get current particles and weights."""
        return self.particles.copy(), self.weights.copy()

    def copy(self):
        """
        OPTIMIZED: Lightweight copy for prediction-only use cases.
        """
        new_pf = ParticleFilter.__new__(ParticleFilter)

        # Copy mutable state
        new_pf.particles = self.particles.copy()
        new_pf.weights = self.weights.copy()

        # Share immutable/reference state
        new_pf.N = self.N
        new_pf.bounds = self.bounds
        new_pf.sensor_model = self.sensor_model
        new_pf.dispersion_model = self.dispersion_model
        new_pf.resample_threshold = self.resample_threshold
        new_pf.mcmc_std = self.mcmc_std
        new_pf.iteration = self.iteration
        new_pf.is_discrete = self.is_discrete  # Include discrete sensor flag

        new_pf.last_measurement = None
        new_pf.last_sensor_position = None

        new_pf._current_distance_map = None
        new_pf._current_sensor_position = None

        return new_pf

    def deep_copy(self):
        return deepcopy(self)

    def compute_expected_entropy(self, sensor_position: tuple[float, float]) -> float:
        """
        OPTIMIZED: Compute Expected Entropy (Information Gain) in one pass.
        Replaces the slow loop over sensor bins.
        """
        # 1. Compute concentrations ONCE (The expensive part)
        predicted_concs = self._compute_concentrations(self.particles, sensor_position)

        # 2. Get probabilities for ALL bins at once
        # Shape: (Num_Bins, N_Particles)
        bin_probs_matrix = self.sensor_model.compute_predictive_distribution(predicted_concs)

        # 3. Compute P(z) for each bin (marginalized over particles)
        # Shape: (Num_Bins,)
        bin_total_probs = bin_probs_matrix @ self.weights

        # 4. Compute posterior weights for ALL scenarios simultaneously
        # Bayes Rule: P(x|z) = P(z|x) * P(x) / P(z)
        # Avoid division by zero
        safe_denominators = bin_total_probs[:, None] + 1e-15
        posterior_weights = (bin_probs_matrix * self.weights[None, :]) / safe_denominators

        # 5. Compute Entropy for each bin scenario
        # H(X|Z=k) = -Σ w_new * log(w_new)
        posterior_weights_safe = np.maximum(posterior_weights, 1e-15)
        entropies = -np.sum(posterior_weights * np.log(posterior_weights_safe), axis=1)

        # 6. Compute Expected Entropy: E[H] = Σ P(z) * H(X|Z=z)
        expected_entropy = np.sum(bin_total_probs * entropies)

        return expected_entropy