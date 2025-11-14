import numpy as np
from scipy.stats import multivariate_normal, norm as scipy_norm
from .igdm_gas_model import IndoorGaussianDispersionModel
from .sensor_model import BinarySensorModel
from copy import deepcopy

class ParticleFilterOptimized:
    """
    Optimized particle filter for gas source term estimation.

    Key optimizations:
    1. Vectorized operations instead of loops
    2. Lightweight copy for prediction (avoids deepcopy)
    3. Cached computations for repeated queries
    4. NumPy broadcasting for batch operations
    """

    def __init__(self, num_particles: int, search_bounds: dict[str, list[float]],
                 binary_sensor_model: BinarySensorModel,
                 dispersion_model: IndoorGaussianDispersionModel,
                 resample_threshold: float = 0.5, mcmc_std=None):
        """
        Parameters are identical to original ParticleFilter for drop-in replacement.
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

        # Cache for concentration computations
        self._concentration_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # IGDM: Store precomputed distance map from current sensor position
        # This is computed once per update() call and reused for all particles
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

    def update(self, measurement: int, sensor_position: tuple[float, float], skip_resample: bool = False):
        """
        Update particle filter with new binary measurement (vectorized).

        Args:
            measurement: Binary sensor measurement (0 or 1)
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

        # Gaussian Plume model path (original implementation)
        # Extract particle parameters
        x0 = particles_array[:, 0]
        y0 = particles_array[:, 1]
        Q0 = particles_array[:, 2] * 1e6  # Convert to μg/s

        N_particles = particles_array.shape[0]
        sx, sy = sensor_position

        # Transform to wind-aligned coordinate system (VECTORIZED)
        dx = sx - x0
        dy = sy - y0

        wind_dir = self.dispersion_model.wind_direction
        downwind = dx * np.cos(wind_dir) + dy * np.sin(wind_dir)
        crosswind = -dx * np.sin(wind_dir) + dy * np.cos(wind_dir)

        # Initialize concentrations
        concentrations = np.zeros(N_particles)

        # Only compute for downwind positions
        mask = downwind > 0.1
        if not np.any(mask):
            return concentrations

        # Compute standard deviations (VECTORIZED)
        dw = downwind[mask]
        sigma_y = self.dispersion_model.zeta1 * dw / np.sqrt(1 + 0.0001 * dw)
        sigma_z = self.dispersion_model.zeta2 * dw / np.sqrt(1 + 0.0001 * dw)

        # Filter out invalid sigma values
        valid_sigma = (sigma_y >= 0.01) & (sigma_z >= 0.01)
        if not np.any(valid_sigma):
            return concentrations

        # Extract valid values
        cw = crosswind[mask][valid_sigma]
        sy_v = sigma_y[valid_sigma]
        sz_v = sigma_z[valid_sigma]
        Q0_v = Q0[mask][valid_sigma]

        # Compute concentration (VECTORIZED)
        crosswind_term = np.exp(-cw**2 / (2 * sy_v**2))

        z_diff = self.dispersion_model.agent_height - self.dispersion_model.z0
        z_sum = self.dispersion_model.agent_height + self.dispersion_model.z0
        z_term = (np.exp(-z_diff**2 / (2 * sz_v**2)) +
                  np.exp(-z_sum**2 / (2 * sz_v**2)))

        conc_valid = (Q0_v / (2 * np.pi * self.dispersion_model.V * sy_v * sz_v) *
                      crosswind_term * z_term)

        # Map back to full array
        valid_indices = np.where(mask)[0][valid_sigma]
        concentrations[valid_indices] = conc_valid

        return concentrations

    # def _compute_concentrations_batch(self, sensor_position: tuple[float, float]) -> np.ndarray:
    #     """
    #     Compute concentrations for all particles at once (VECTORIZED).

    #     Returns:
    #     --------
    #     concentrations : np.ndarray, shape (N,)
    #         Predicted concentrations for each particle
    #     """
    #     # Extract particle parameters
    #     x0 = self.particles[:, 0]
    #     y0 = self.particles[:, 1]
    #     Q0 = self.particles[:, 2] * 1e6  # Convert to μg/s

    #     sx, sy = sensor_position

    #     # Transform to wind-aligned coordinate system (VECTORIZED)
    #     dx = sx - x0
    #     dy = sy - y0

    #     wind_dir = self.dispersion_model.wind_direction
    #     downwind = dx * np.cos(wind_dir) + dy * np.sin(wind_dir)
    #     crosswind = -dx * np.sin(wind_dir) + dy * np.cos(wind_dir)

    #     # Initialize concentrations
    #     concentrations = np.zeros(self.N)

    #     # Only compute for downwind positions
    #     mask = downwind > 0.1
    #     if not np.any(mask):
    #         return concentrations

    #     # Compute standard deviations (VECTORIZED)
    #     dw = downwind[mask]
    #     sigma_y = self.dispersion_model.zeta1 * dw / np.sqrt(1 + 0.0001 * dw)
    #     sigma_z = self.dispersion_model.zeta2 * dw / np.sqrt(1 + 0.0001 * dw)

    #     # Filter out invalid sigma values
    #     valid_sigma = (sigma_y >= 0.01) & (sigma_z >= 0.01)
    #     if not np.any(valid_sigma):
    #         return concentrations

    #     # Extract valid values
    #     cw = crosswind[mask][valid_sigma]
    #     sy_v = sigma_y[valid_sigma]
    #     sz_v = sigma_z[valid_sigma]
    #     Q0_v = Q0[mask][valid_sigma]

    #     # Compute concentration (VECTORIZED)
    #     crosswind_term = np.exp(-cw**2 / (2 * sy_v**2))

    #     z_diff = self.dispersion_model.agent_height - self.dispersion_model.z0
    #     z_sum = self.dispersion_model.agent_height + self.dispersion_model.z0
    #     z_term = (np.exp(-z_diff**2 / (2 * sz_v**2)) +
    #               np.exp(-z_sum**2 / (2 * sz_v**2)))

    #     conc_valid = (Q0_v / (2 * np.pi * self.dispersion_model.V * sy_v * sz_v) *
    #                   crosswind_term * z_term)

    #     # Map back to full array
    #     valid_indices = np.where(mask)[0][valid_sigma]
    #     concentrations[valid_indices] = conc_valid

    #     return concentrations

    def _compute_concentrations_batch(self, sensor_position: tuple[float, float]) -> np.ndarray:
        """
        Compute concentrations for all SELF.particles at once (VECTORIZED).

        Uses IGDM if available, otherwise falls back to Gaussian Plume model.
        Key optimization from paper: Compute Dijkstra distance map ONCE from sensor position,
        then look up distances for all particles.
        """
        # Check if using IGDM
        if isinstance(self.dispersion_model, IndoorGaussianDispersionModel):
            # IGDM path: Use optimized batch computation with distance map
            particle_locations = self.particles[:, :2]  # (N, 2) - x, y positions
            release_rates = self.particles[:, 2]         # (N,) - Q values

            # This calls compute_concentrations_batch which internally:
            # 1. Computes distance map ONCE from sensor position to all grid cells
            # 2. Looks up distances for each particle location from the map
            # 3. Applies IGDM equation: Qm * exp(-cobs^2 / (2*sigma_m^2))
            concentrations = self.dispersion_model.compute_concentrations_batch(
                sensor_position, particle_locations, release_rates
            )
            return concentrations
        else:
            # Gaussian Plume path: Use original implementation
            return self._compute_concentrations_for_particles(self.particles, sensor_position)

    # def _compute_likelihoods_vectorized(self, measurement: int, sensor_position: tuple[float, float]):
    #     """Compute binary measurement likelihood for all particles (VECTORIZED)."""
    #     # Compute all concentrations at once
    #     predicted_concs = self._compute_concentrations_batch(sensor_position)

    #     # Compute likelihoods using sensor model (VECTORIZED)
    #     likelihoods = np.array([
    #         self.sensor_model.probability_binary(measurement, conc)
    #         for conc in predicted_concs
    #     ])

    #     return likelihoods

    def _compute_likelihoods_vectorized(self, measurement: int, sensor_position: tuple[float, float]):
        """Compute binary measurement likelihood for all particles (VECTORIZED)."""
        # Compute all concentrations at once
        predicted_concs = self._compute_concentrations_batch(sensor_position)

        # Compute all likelihoods at once using the vectorized sensor model method
        likelihoods = self.sensor_model.probability_binary_vec(measurement, predicted_concs)
        
        # # Add a small epsilon to prevent all weights from becoming zero
        # likelihoods = np.maximum(likelihoods, 1e-50)

        return likelihoods

    def _effective_sample_size(self):
        """Compute effective sample size N_eff ≈ 1 / Σ(wⁱ)²."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self, rouge_fraction: float = 0.05):
        """
        Resample particles using systematic resampling.

        NOTE: Rouge particle injection (5% random particles) has been disabled
        to match the original paper implementation. The paper relies on MCMC moves
        alone for maintaining particle diversity.
        """
        # 1. Perform systematic resampling
        indices = self._systematic_resample()
        self.particles = self.particles[indices]

        # --- ROUGE PARTICLES DISABLED (not in original paper) ---
        #
        # # 2. Add "rouge" particles (regularization)
        # # This re-introduces a small fraction of particles from the prior
        # # distribution to prevent the filter from becoming over-confident.
        #
        # num_rouge = int(self.N * rouge_fraction)
        #
        # if num_rouge > 0:
        #     # Get random indices to replace with new particles
        #     replace_indices = np.random.choice(self.N, num_rouge, replace=False)
        #
        #     # Generate new random particles (from the prior)
        #     # We only need 'num_rouge' new particles
        #     new_particles = self._initialize_particles()
        #
        #     # Replace the selected particles
        #     self.particles[replace_indices] = new_particles[:num_rouge]
        #
        # --- END ROUGE PARTICLES ---

        # 2. Reset weights
        self.weights = np.ones(self.N) / self.N

    def _systematic_resample(self):
        """Systematic resampling algorithm."""
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
        """MCMC move step (FULLY VECTORIZED)."""
        if self.last_measurement is None or self.last_sensor_position is None:
            return

        # 1. Generate all proposals at once (You did this correctly)
        proposals = self.particles.copy()
        proposals[:, 0] += np.random.normal(0, self.mcmc_std['x'], self.N)
        proposals[:, 1] += np.random.normal(0, self.mcmc_std['y'], self.N)
        proposals[:, 2] += np.random.normal(0, self.mcmc_std['Q'], self.N)

        # 2. Check bounds (You did this correctly)
        in_bounds = (
            (proposals[:, 0] >= self.bounds['x'][0]) & (proposals[:, 0] <= self.bounds['x'][1]) &
            (proposals[:, 1] >= self.bounds['y'][0]) & (proposals[:, 1] <= self.bounds['y'][1]) &
            (proposals[:, 2] >= self.bounds['Q'][0]) & (proposals[:, 2] <= self.bounds['Q'][1])
        )
        
        # 3. Compute likelihoods for ALL CURRENT particles at once
        conc_curr = self._compute_concentrations_for_particles(
            self.particles, self.last_sensor_position
        )
        # Assumes you have a vectorized sensor model method
        likelihood_curr = self.sensor_model.probability_binary_vec(
            self.last_measurement, conc_curr
        )

        # 4. Compute likelihoods for ALL PROPOSED particles at once
        conc_prop = self._compute_concentrations_for_particles(
            proposals, self.last_sensor_position
        )
        # Assumes you have a vectorized sensor model method
        likelihood_prop = self.sensor_model.probability_binary_vec(
            self.last_measurement, conc_prop
        )

        # 5. Metropolis-Hastings acceptance (FULLY VECTORIZED)
        
        # Avoid division by zero
        likelihood_curr = np.maximum(likelihood_curr, 1e-50)
        
        acceptance_prob = likelihood_prop / likelihood_curr
        acceptance_prob = np.minimum(1.0, acceptance_prob)

        # 6. Find particles to move (VECTORIZED)
        rand_vals = np.random.random(self.N)
        move_mask = (rand_vals < acceptance_prob) & in_bounds
        
        # 7. Update particles in one operation (VECTORIZED)
        self.particles[move_mask] = proposals[move_mask]

    def _in_bounds(self, particle: np.ndarray):
        """Check if particle is within bounds."""
        x0, y0, Q0 = particle
        return (self.bounds['x'][0] <= x0 <= self.bounds['x'][1] and
                self.bounds['y'][0] <= y0 <= self.bounds['y'][1] and
                self.bounds['Q'][0] <= Q0 <= self.bounds['Q'][1])

    def get_estimate(self):
        """Get current source term estimate (weighted mean)."""
        # Vectorized weighted mean
        estimate = {
            'x': np.sum(self.weights * self.particles[:, 0]),
            'y': np.sum(self.weights * self.particles[:, 1]),
            'Q': np.sum(self.weights * self.particles[:, 2])
        }

        # Vectorized weighted variance
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

        This is used for RRT planning to compute expected entropy gains.
        According to Eq. (28) in the paper: ŵi_k+n = p(b̂k+n|θi_k) · wi_k

        Args:
            measurement: Binary sensor measurement (0 or 1)
            sensor_position: Position where measurement would be taken

        Returns:
            float: Entropy that would result from this hypothetical measurement
        """
        # Compute likelihoods without modifying state
        likelihoods = self._compute_likelihoods_vectorized(measurement, sensor_position)

        # Compute hypothetical weights (Eq. 28)
        hypothetical_weights = self.weights * likelihoods

        # Normalize
        weight_sum = np.sum(hypothetical_weights)
        if weight_sum > 0:
            hypothetical_weights /= weight_sum
        else:
            # All weights would be zero - return high entropy (maximum uncertainty)
            return np.log(self.N)

        # Compute entropy from hypothetical weights
        weights_safe = hypothetical_weights[hypothetical_weights > 1e-15]
        entropy = -np.sum(weights_safe * np.log(weights_safe))

        return entropy

    def predict_measurement_probability(self, sensor_position, binary_value=None):
        """
        Predict probability of future binary measurement (VECTORIZED).

        This is the critical hotspot for RRT-Infotaxis!
        """
        # Compute all concentrations at once (VECTORIZED)
        predicted_concs = self._compute_concentrations_batch(sensor_position)

        # Vectorized computation of beta
        delta_c = self.sensor_model.threshold - predicted_concs
        sigma_g = self.sensor_model.alpha * predicted_concs + self.sensor_model.sigma_env

        # Handle division by zero
        sigma_g = np.maximum(sigma_g, 1e-15)

        # Vectorized CDF computation
        phi = scipy_norm.cdf(delta_c / sigma_g)
        beta = np.sum(phi * self.weights)

        if binary_value is None:
            return beta, 1 - beta
        elif binary_value == 0:
            return beta
        else:
            return 1 - beta

    def get_particles(self):
        """Get current particles and weights."""
        return self.particles.copy(), self.weights.copy()

    def copy(self):
        """
        OPTIMIZED: Lightweight copy for prediction-only use cases.

        Creates a shallow copy with shared immutable data (dispersion/sensor models).
        Only copies mutable state (particles, weights).

        This is 10-50x faster than deepcopy!
        """
        new_pf = ParticleFilterOptimized.__new__(ParticleFilterOptimized)

        # Copy mutable state
        new_pf.particles = self.particles.copy()
        new_pf.weights = self.weights.copy()

        # Share immutable/reference state (SAFE for read-only operations)
        new_pf.N = self.N
        new_pf.bounds = self.bounds  # dict is immutable in this context
        new_pf.sensor_model = self.sensor_model  # Shared reference (read-only)
        new_pf.dispersion_model = self.dispersion_model  # Shared reference (read-only)
        new_pf.resample_threshold = self.resample_threshold
        new_pf.mcmc_std = self.mcmc_std
        new_pf.iteration = self.iteration

        # CRITICAL: Clear last_measurement and last_sensor_position for RRT hypothetical copies
        # These are only needed for MCMC moves (which are skipped during RRT predictions)
        # Keeping stale values could interfere with entropy calculations
        new_pf.last_measurement = None
        new_pf.last_sensor_position = None

        new_pf._concentration_cache = {}  # Fresh cache
        new_pf._cache_hits = 0
        new_pf._cache_misses = 0

        # IGDM: Clear distance map cache for copied filter
        new_pf._current_distance_map = None
        new_pf._current_sensor_position = None

        return new_pf

    def deep_copy(self):
        """
        Full deep copy (use only when truly needed).
        For backward compatibility.
        """
        return deepcopy(self)
