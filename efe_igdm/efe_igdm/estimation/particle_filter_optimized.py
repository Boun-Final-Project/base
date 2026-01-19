import numpy as np
from scipy.stats import multivariate_normal, norm as scipy_norm
from ..models.igdm_gas_model import IndoorGaussianDispersionModel
from .sensor_model import ContinuousGaussianSensorModel
from copy import deepcopy

class ParticleFilterOptimized:
    """
    Optimized particle filter for gas source term estimation.

    Key optimizations:
    1. Vectorized operations for likelihoods and resampling (np.searchsorted).
    2. Lightweight copy for prediction (avoids deepcopy).
    3. Cached computations for repeated queries.
    4. NumPy broadcasting for batch operations.
    """

    def __init__(self, num_particles: int, search_bounds: dict[str, list[float]],
                 binary_sensor_model: ContinuousGaussianSensorModel,
                 dispersion_model: IndoorGaussianDispersionModel,
                 resample_threshold: float = 0.3, mcmc_std=None):
        """
        Parameters are identical to original ParticleFilter for drop-in replacement.
        Note: 'binary_sensor_model' parameter name kept for backwards compatibility,
        but now expects ContinuousGaussianSensorModel.
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

        # Cache for concentration computations
        self._concentration_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

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

    def update(self, measurement: float, sensor_position: tuple[float, float], skip_resample: bool = False):
        """
        Update particle filter with new continuous measurement (vectorized).

        Args:
            measurement: Continuous sensor measurement (e.g., 10.94 ppm)
            sensor_position: Position where measurement was taken
            skip_resample: If True, skip resampling and MCMC (for RRT hypothetical updates)
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
        # SKIP this for RRT hypothetical updates to prevent entropy increase
        if not skip_resample:
            N_eff = self._effective_sample_size()
            if N_eff < self.resample_threshold * self.N:
                self._resample()
                self._mcmc_move()

    def _compute_concentrations_for_particles(self, particles_array: np.ndarray,
                                            sensor_position: tuple[float, float]) -> np.ndarray:
        """
        Compute concentrations for a given array of particles (VECTORIZED).
        Used for MCMC moves where we need to evaluate arbitrary particle arrays.
        """
        # Check if using IGDM
        if isinstance(self.dispersion_model, IndoorGaussianDispersionModel):
            # IGDM path: Use batch computation (will compute distance map)
            particle_locations = particles_array[:, :2]  # (N, 2) - x, y positions
            release_rates = particles_array[:, 2]         # (N,) - Q values

            concentrations = self.dispersion_model.compute_concentrations_batch(
                sensor_position, particle_locations, release_rates
            )
            return concentrations

        # Gaussian Plume model path (fallback if you ever swap models)
        x0 = particles_array[:, 0]
        y0 = particles_array[:, 1]
        Q0 = particles_array[:, 2] * 1e6  # Convert to μg/s

        sx, sy = sensor_position

        dx = sx - x0
        dy = sy - y0

        wind_dir = self.dispersion_model.wind_direction
        downwind = dx * np.cos(wind_dir) + dy * np.sin(wind_dir)
        crosswind = -dx * np.sin(wind_dir) + dy * np.cos(wind_dir)

        concentrations = np.zeros(particles_array.shape[0])

        mask = downwind > 0.1
        if not np.any(mask):
            return concentrations

        dw = downwind[mask]
        sigma_y = self.dispersion_model.zeta1 * dw / np.sqrt(1 + 0.0001 * dw)
        sigma_z = self.dispersion_model.zeta2 * dw / np.sqrt(1 + 0.0001 * dw)

        valid_sigma = (sigma_y >= 0.01) & (sigma_z >= 0.01)
        if not np.any(valid_sigma):
            return concentrations

        cw = crosswind[mask][valid_sigma]
        sy_v = sigma_y[valid_sigma]
        sz_v = sigma_z[valid_sigma]
        Q0_v = Q0[mask][valid_sigma]

        crosswind_term = np.exp(-cw**2 / (2 * sy_v**2))

        z_diff = self.dispersion_model.agent_height - self.dispersion_model.z0
        z_sum = self.dispersion_model.agent_height + self.dispersion_model.z0
        z_term = (np.exp(-z_diff**2 / (2 * sz_v**2)) +
                  np.exp(-z_sum**2 / (2 * sz_v**2)))

        conc_valid = (Q0_v / (2 * np.pi * self.dispersion_model.V * sy_v * sz_v) *
                      crosswind_term * z_term)

        valid_indices = np.where(mask)[0][valid_sigma]
        concentrations[valid_indices] = conc_valid

        return concentrations

    def _compute_concentrations_batch(self, sensor_position: tuple[float, float]) -> np.ndarray:
        """
        Compute concentrations for all SELF.particles at once (VECTORIZED).
        """
        # Check if using IGDM
        if isinstance(self.dispersion_model, IndoorGaussianDispersionModel):
            # IGDM path: Use optimized batch computation with distance map
            particle_locations = self.particles[:, :2]  # (N, 2) - x, y positions
            release_rates = self.particles[:, 2]         # (N,) - Q values

            concentrations = self.dispersion_model.compute_concentrations_batch(
                sensor_position, particle_locations, release_rates
            )
            return concentrations
        else:
            return self._compute_concentrations_for_particles(self.particles, sensor_position)

    def _compute_likelihoods_vectorized(self, measurement: float, sensor_position: tuple[float, float]):
        """
        Compute Gaussian measurement likelihood for all particles (VECTORIZED).
        """
        # Compute all predicted concentrations at once
        predicted_concs = self._compute_concentrations_batch(sensor_position)

        # Check if sensor model has continuous Gaussian likelihood (paper's method)
        if hasattr(self.sensor_model, 'probability_continuous_vec'):
            likelihoods = self.sensor_model.probability_continuous_vec(measurement, predicted_concs)
        else:
            likelihoods = self.sensor_model.probability_binary_vec(measurement, predicted_concs)

        return likelihoods

    def _effective_sample_size(self):
        """Compute effective sample size N_eff ≈ 1 / Σ(wⁱ)²."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """
        Resample particles using systematic resampling (VECTORIZED).
        """
        # 1. Perform systematic resampling
        indices = self._systematic_resample()
        self.particles = self.particles[indices]

        # 2. Reset weights
        self.weights = np.ones(self.N) / self.N

    def _systematic_resample(self):
        """
        Vectorized systematic resampling.
        O(N) in C-speed vs O(N) in Python-speed.
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
        """MCMC move step (FULLY VECTORIZED)."""
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
        conc_curr = self._compute_concentrations_for_particles(
            self.particles, self.last_sensor_position
        )
        if hasattr(self.sensor_model, 'probability_continuous_vec'):
            likelihood_curr = self.sensor_model.probability_continuous_vec(
                self.last_measurement, conc_curr
            )
        else:
            likelihood_curr = self.sensor_model.probability_binary_vec(
                self.last_measurement, conc_curr
            )

        # 4. Compute likelihoods for ALL PROPOSED particles at once
        conc_prop = self._compute_concentrations_for_particles(
            proposals, self.last_sensor_position
        )
        if hasattr(self.sensor_model, 'probability_continuous_vec'):
            likelihood_prop = self.sensor_model.probability_continuous_vec(
                self.last_measurement, conc_prop
            )
        else:
            likelihood_prop = self.sensor_model.probability_binary_vec(
                self.last_measurement, conc_prop
            )

        # 5. Metropolis-Hastings acceptance (FULLY VECTORIZED)
        likelihood_curr = np.maximum(likelihood_curr, 1e-50)
        
        acceptance_prob = likelihood_prop / likelihood_curr
        acceptance_prob = np.minimum(1.0, acceptance_prob)

        # 6. Find particles to move
        rand_vals = np.random.random(self.N)
        move_mask = (rand_vals < acceptance_prob) & in_bounds
        
        # 7. Update particles in one operation
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
        # For ContinuousGaussianSensorModel, compute bin likelihoods using discretization
        if hasattr(self.sensor_model, 'create_discretization_thresholds'):
            predicted_concs = self._compute_concentrations_batch(sensor_position)
            thresholds = self.sensor_model.create_discretization_thresholds(predicted_concs)
            likelihoods = self.sensor_model.compute_bin_likelihood_vec(
                measurement, predicted_concs, thresholds
            )
        else:
            likelihoods = self._compute_likelihoods_vectorized(measurement, sensor_position)

        hypothetical_weights = self.weights * likelihoods

        weight_sum = np.sum(hypothetical_weights)
        if weight_sum > 0:
            hypothetical_weights /= weight_sum
        else:
            return np.log(self.N)

        weights_safe = hypothetical_weights[hypothetical_weights > 1e-15]
        entropy = -np.sum(weights_safe * np.log(weights_safe))

        return entropy
    
    def predict_measurement_probability(self, sensor_position, binary_value=None):
        """
        Predict probability of future measurement using ContinuousGaussianSensorModel.
        """
        predicted_concs = self._compute_concentrations_batch(sensor_position)
        inner_thresholds = self.sensor_model.create_discretization_thresholds(predicted_concs)
        bin_edges = np.concatenate([[-np.inf], inner_thresholds, [np.inf]])

        sigma_g = self.sensor_model.alpha * predicted_concs + self.sensor_model.sigma_env
        sigma_g = np.maximum(sigma_g, 1e-15)

        z_scores = (bin_edges[:, None] - predicted_concs[None, :]) / sigma_g[None, :]
        cdfs = scipy_norm.cdf(z_scores)

        bin_probs_per_particle = cdfs[1:, :] - cdfs[:-1, :]
        level_probs = bin_probs_per_particle @ self.weights

        if binary_value is None:
            return level_probs
        else:
            return level_probs[binary_value]

    def get_particles(self):
        """Get current particles and weights."""
        return self.particles.copy(), self.weights.copy()

    def copy(self):
        """
        OPTIMIZED: Lightweight copy for prediction-only use cases.
        """
        new_pf = ParticleFilterOptimized.__new__(ParticleFilterOptimized)

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

        new_pf.last_measurement = None
        new_pf.last_sensor_position = None

        new_pf._concentration_cache = {}
        new_pf._cache_hits = 0
        new_pf._cache_misses = 0

        new_pf._current_distance_map = None
        new_pf._current_sensor_position = None

        return new_pf

    def deep_copy(self):
        return deepcopy(self)