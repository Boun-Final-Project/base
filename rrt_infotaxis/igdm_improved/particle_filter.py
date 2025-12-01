"""
Particle Filter for source localization estimation.
"""

import numpy as np
from scipy.stats import norm as scipy_norm
from copy import deepcopy


class ParticleFilter:
    """Particle filter for estimating source location and release rate."""

    def __init__(self, num_particles, search_bounds, binary_sensor_model, dispersion_model,
                 resample_threshold=0.5, mcmc_std=None):
        """
        Parameters:
        -----------
        num_particles : int
            Number of particles
        search_bounds : dict
            Bounds for x, y, Q: {'x': (min, max), 'y': (min, max), 'Q': (min, max)}
        binary_sensor_model : BinarySensorModel
            Sensor model for likelihood computation
        dispersion_model : IGDMModel
            Gas dispersion model
        resample_threshold : float
            Resample when N_eff < threshold * N
        mcmc_std : dict, optional
            Standard deviations for MCMC proposal distribution
        """
        self.N = num_particles
        self.bounds = search_bounds
        self.sensor_model = binary_sensor_model
        self.dispersion_model = dispersion_model
        self.resample_threshold = resample_threshold

        # Detect sensor type: check if it has discrete probability method
        self.is_discrete = hasattr(binary_sensor_model, 'probability_discrete')

        if mcmc_std is None:
            x_range = search_bounds['x'][1] - search_bounds['x'][0]
            y_range = search_bounds['y'][1] - search_bounds['y'][0]
            Q_range = search_bounds['Q'][1] - search_bounds['Q'][0]
            self.mcmc_std = {'x': 0.05 * x_range, 'y': 0.05 * y_range, 'Q': 0.05 * Q_range}
        else:
            self.mcmc_std = mcmc_std

        self.particles = self._initialize_particles()
        self.weights = np.ones(self.N) / self.N
        self.iteration = 0
        self.last_measurement = None
        self.last_sensor_position = None
        self.current_step = 0  # Track current time step for time-dependent dispersion

    def _initialize_particles(self):
        """Initialize particles uniformly within search bounds."""
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']
        Q_min, Q_max = self.bounds['Q']

        particles = np.zeros((self.N, 3))
        particles[:, 0] = np.random.uniform(x_min, x_max, self.N)
        particles[:, 1] = np.random.uniform(y_min, y_max, self.N)
        particles[:, 2] = np.random.uniform(Q_min, Q_max, self.N)

        return particles

    def update(self, measurement, sensor_position, time_step=None, is_planning=False):
        """Update particle filter with binary measurement.

        Parameters:
        -----------
        measurement : int
            Binary measurement (0 or 1)
        sensor_position : tuple
            (x, y) sensor position
        time_step : int, optional
            Current time step for time-dependent dispersion
        is_planning : bool, optional
            If True, skip resampling (used during RRT planning entropy calculations).
            RULE: During planning, we do NOT resample because:
            - Resampling introduces randomness into entropy calculations
            - The particle set must remain consistent for deterministic information gain
            - Entropy predictions should not change the particle distribution
            MCMC is always performed to refine particle positions based on updated weights.
        """
        if time_step is not None:
            self.current_step = time_step

        self.iteration += 1
        self.last_measurement = measurement
        self.last_sensor_position = sensor_position

        likelihoods = self._compute_likelihoods(measurement, sensor_position)
        self.weights *= likelihoods

        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.particles = self._initialize_particles()
            self.weights = np.ones(self.N) / self.N
            return

        # During planning: never resample, but always do MCMC
        if is_planning:
            self._mcmc_move()
            return

        # During actual measurement: resample when needed and do MCMC
        N_eff = self._effective_sample_size()
        if N_eff < self.resample_threshold * self.N:
            self._resample()
            self._mcmc_move()

    def _compute_likelihoods(self, measurement, sensor_position):
        """Compute likelihood for each particle given measurement (binary or discrete).

        Optimization: Compute Dijkstra distance grid once from sensor position,
        then reuse for all particles instead of computing 200 separate Dijkstras.

        Parameters:
        -----------
        measurement : int
            Measurement value (0-1 for binary, 0-4 for discrete)
        sensor_position : tuple
            (x, y) position where measurement was taken

        Returns:
        --------
        likelihoods : np.ndarray
            Likelihood for each particle
        """
        # Compute Dijkstra distance grid once from sensor position (optimization)
        dijkstra_grid = self.dispersion_model.get_dijkstra_distances_from(sensor_position)

        likelihoods = []
        for i in range(self.N):
            x0, y0, Q0 = self.particles[i]
            predicted_conc = self.dispersion_model.compute_concentration(
                sensor_position, (x0, y0), Q0,
                time_step=self.current_step, debug=False,
                dijkstra_grid=dijkstra_grid
            )

            # Use appropriate probability method based on sensor type
            if self.is_discrete:
                likelihood = self.sensor_model.probability_discrete(measurement, predicted_conc)
            else:
                likelihood = self.sensor_model.probability_binary(measurement, predicted_conc)

            likelihoods.append(likelihood)
        return np.array(likelihoods)

    def _effective_sample_size(self):
        """Compute effective sample size (N_eff)."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self):
        """Resample particles using systematic resampling."""
        indices = self._systematic_resample()
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def _systematic_resample(self):
        """Systematic resampling algorithm.

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
        """MCMC move step for particle refinement (Metropolis-Hastings).

        Optimization: Compute Dijkstra distance grid once from sensor position,
        then reuse for all particles.
        """
        if self.last_measurement is None or self.last_sensor_position is None:
            return

        # Compute Dijkstra distance grid once from sensor position (optimization)
        dijkstra_grid = self.dispersion_model.get_dijkstra_distances_from(self.last_sensor_position)

        for i in range(self.N):
            current_particle = self.particles[i].copy()
            proposal = current_particle.copy()
            proposal[0] += np.random.normal(0, self.mcmc_std['x'])
            proposal[1] += np.random.normal(0, self.mcmc_std['y'])
            proposal[2] += np.random.normal(0, self.mcmc_std['Q'])

            if not self._in_bounds(proposal):
                continue

            x0_curr, y0_curr, Q0_curr = current_particle
            conc_curr = self.dispersion_model.compute_concentration(
                self.last_sensor_position, (x0_curr, y0_curr), Q0_curr,
                time_step=self.current_step, debug=False,
                dijkstra_grid=dijkstra_grid
            )

            x0_prop, y0_prop, Q0_prop = proposal
            conc_prop = self.dispersion_model.compute_concentration(
                self.last_sensor_position, (x0_prop, y0_prop), Q0_prop,
                time_step=self.current_step, debug=False,
                dijkstra_grid=dijkstra_grid
            )

            # Use appropriate probability method based on sensor type
            if self.is_discrete:
                likelihood_curr = self.sensor_model.probability_discrete(self.last_measurement, conc_curr)
                likelihood_prop = self.sensor_model.probability_discrete(self.last_measurement, conc_prop)
            else:
                likelihood_curr = self.sensor_model.probability_binary(self.last_measurement, conc_curr)
                likelihood_prop = self.sensor_model.probability_binary(self.last_measurement, conc_prop)

            if likelihood_curr > 1e-10:
                acceptance_prob = min(1.0, likelihood_prop / likelihood_curr)
            else:
                acceptance_prob = 1.0 if likelihood_prop > 1e-10 else 0.5

            if np.random.random() < acceptance_prob:
                self.particles[i] = proposal

    def _in_bounds(self, particle):
        """Check if particle is within search bounds.

        Parameters:
        -----------
        particle : np.ndarray
            Particle state [x, y, Q]

        Returns:
        --------
        valid : bool
            True if particle is within bounds
        """
        x0, y0, Q0 = particle
        x_valid = self.bounds['x'][0] <= x0 <= self.bounds['x'][1]
        y_valid = self.bounds['y'][0] <= y0 <= self.bounds['y'][1]
        Q_valid = self.bounds['Q'][0] <= Q0 <= self.bounds['Q'][1]
        return x_valid and y_valid and Q_valid

    def get_estimate(self):
        """Get weighted mean estimate and standard deviation.

        Returns:
        --------
        estimate : dict
            Mean estimate {'x': x_est, 'y': y_est, 'Q': Q_est}
        std : dict
            Standard deviation {'x': sigma_x, 'y': sigma_y, 'Q': sigma_Q}
        """
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        Q_est = np.sum(self.weights * self.particles[:, 2])

        x_var = np.sum(self.weights * (self.particles[:, 0] - x_est) ** 2)
        y_var = np.sum(self.weights * (self.particles[:, 1] - y_est) ** 2)
        Q_var = np.sum(self.weights * (self.particles[:, 2] - Q_est) ** 2)

        estimate = {'x': x_est, 'y': y_est, 'Q': Q_est}
        std = {'x': np.sqrt(x_var), 'y': np.sqrt(y_var), 'Q': np.sqrt(Q_var)}

        return estimate, std

    def get_entropy(self):
        """Compute Shannon entropy of particle weights.

        Returns:
        --------
        entropy : float
            Entropy of weight distribution
        """
        weights_safe = self.weights[self.weights > 1e-10]
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return entropy

    def predict_measurement_probability(self, sensor_position, measurement_value=None, time_step=None):
        """Predict probability of measurement at a position (binary or discrete).

        Optimization: Compute Dijkstra distance grid once from sensor position,
        then reuse for all particles.

        Parameters:
        -----------
        sensor_position : tuple
            (x, y) position where measurement would be made
        measurement_value : {None, int}, optional
            If None, return tuple with probabilities for all levels
            If int, return probability of that specific level
        time_step : int or float, optional
            Time step for time-dependent gas dispersion. If None, uses current_step.

        Returns:
        --------
        probability : float or tuple
            For binary sensor: (P(0), P(1))
            For discrete sensor:
            - If measurement_value is None: tuple of 5 probabilities P(0)-P(4)
            - If measurement_value in [0-4]: probability of that level
        """
        if time_step is None:
            time_step = self.current_step

        # Compute Dijkstra distance grid once from sensor position (optimization)
        dijkstra_grid = self.dispersion_model.get_dijkstra_distances_from(sensor_position)

        if self.is_discrete:
            # For discrete sensor, compute probability for all 5 levels
            level_probs = [0.0] * 5

            for i in range(self.N):
                x0, y0, Q0 = self.particles[i]
                predicted_conc = self.dispersion_model.compute_concentration(
                    sensor_position, (x0, y0), Q0,
                    time_step=time_step,
                    dijkstra_grid=dijkstra_grid
                )

                for level in range(5):
                    prob_level = self.sensor_model.probability_discrete(level, predicted_conc)
                    level_probs[level] += prob_level * self.weights[i]

            if measurement_value is None:
                return tuple(level_probs)
            else:
                return level_probs[measurement_value]
        else:
            # Original binary sensor code
            beta = 0.0
            for i in range(self.N):
                x0, y0, Q0 = self.particles[i]
                predicted_conc = self.dispersion_model.compute_concentration(
                    sensor_position, (x0, y0), Q0,
                    time_step=time_step,
                    dijkstra_grid=dijkstra_grid
                )

                delta_c = self.sensor_model.threshold - predicted_conc
                sigma_g = self.sensor_model.get_std(predicted_conc)
                phi = scipy_norm.cdf(delta_c / sigma_g)

                beta += phi * self.weights[i]

            if measurement_value is None:
                return beta, 1 - beta
            elif measurement_value == 0:
                return beta
            else:
                return 1 - beta

    def get_particles(self):
        """Get copy of particles and weights.

        Returns:
        --------
        particles : np.ndarray
            Particle states (N x 3)
        weights : np.ndarray
            Particle weights (N,)
        """
        return self.particles.copy(), self.weights.copy()

    def copy(self):
        """Create deep copy of particle filter.

        Returns:
        --------
        pf : ParticleFilter
            Deep copy of this particle filter
        """
        return deepcopy(self)
