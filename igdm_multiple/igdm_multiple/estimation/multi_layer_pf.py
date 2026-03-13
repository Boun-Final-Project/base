"""
Multi-Layer Particle Filter for multiple gas source localization.

Each layer is an independent ParticleFilter tracking one potential source.
Peak suppression weights prevent all layers from collapsing onto the same source.

Based on:
- Gao et al., "Robust Radiation Sources Localization Based on the Peak Suppressed
  Particle Filter", Sensors, 2018 (PSPF)
- Bai et al., "Autonomous radiation source searching using ADE-PSPF",
  Robotics & Autonomous Systems, 2023 (OIC)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .particle_filter import ParticleFilter, _log_sum_exp, _log_normalize, _get_weights
from .igdm_gas_model import IndoorGaussianDispersionModel
from .sensor_model import ContinuousGaussianSensorModel


class LayerState:
    """Tracks the state of a single particle filter layer."""
    def __init__(self, layer_id: int, pf: ParticleFilter):
        self.id = layer_id
        self.pf = pf
        self.confirmed = False
        self.converged = False
        self.estimate = None  # {'x', 'y', 'Q'} when converged
        self.std = None


class MultiLayerParticleFilter:
    """
    Multi-layer particle filter for multi-source gas source localization.

    L independent ParticleFilter layers, each estimating one source.
    Peak suppression prevents multiple layers from converging to the same source.
    OIC subtracts confirmed source contributions from measurements.
    """

    def __init__(self,
                 num_layers: int,
                 num_particles_per_layer: int,
                 search_bounds: dict,
                 sensor_model: ContinuousGaussianSensorModel,
                 dispersion_model: IndoorGaussianDispersionModel,
                 peak_suppression_radius: float = 1.0,
                 convergence_sigma: float = 0.5,
                 verification_threshold: float = 5.0):
        self.num_layers = num_layers
        self.ps_radius = peak_suppression_radius
        self.convergence_sigma = convergence_sigma
        self.verification_threshold = verification_threshold
        self.sensor_model = sensor_model
        self.dispersion_model = dispersion_model
        self.search_bounds = search_bounds

        # Create L independent particle filter layers
        self.layers: List[LayerState] = []
        for i in range(num_layers):
            pf = ParticleFilter(
                num_particles=num_particles_per_layer,
                search_bounds=search_bounds,
                sensor_model=sensor_model,
                dispersion_model=dispersion_model,
                resample_threshold=0.5
            )
            self.layers.append(LayerState(i, pf))

        # Confirmed sources (passed verification)
        self.confirmed_sources: List[Dict] = []

        # Measurement history for verification
        self.measurement_history: List[Tuple[Tuple[float, float], float]] = []
        self.max_history = 100

        self.iteration = 0

    def update(self, measurement: float, sensor_position: Tuple[float, float]):
        """
        Update all layers with the new measurement.

        For each layer:
        1. Compute corrected measurement (subtract confirmed sources via OIC)
        2. Compute standard observation likelihood
        3. Apply peak suppression weight
        4. Update layer
        """
        self.iteration += 1

        # Store measurement
        self.measurement_history.append((sensor_position, measurement))
        if len(self.measurement_history) > self.max_history:
            self.measurement_history.pop(0)

        # Get current estimates from all layers (for peak suppression)
        layer_estimates = []
        for layer in self.layers:
            est, std = layer.pf.get_estimate()
            layer_estimates.append(est)

        for layer in self.layers:
            if layer.confirmed:
                # Confirmed layers still update but don't need peak suppression
                layer.pf.update(measurement, sensor_position)
                continue

            # 1. Compute corrected measurement (OIC)
            corrected_measurement = self._compute_corrected_measurement(
                measurement, sensor_position, exclude_layer=layer.id
            )
            corrected_measurement = max(corrected_measurement, 0.0)

            # 2. Compute standard observation likelihood
            predicted_concs = layer.pf._compute_concentrations(
                layer.pf.particles, sensor_position
            )
            likelihoods = self.sensor_model.probability_continuous_vec(
                corrected_measurement, predicted_concs
            )

            # 3. Compute peak suppression weights
            ps_weights = self._compute_peak_suppression(
                layer, layer_estimates
            )

            # 4. Combine: w = w_obs * w_ps
            combined_likelihoods = likelihoods * ps_weights

            # 5. Update log-weights manually (bypass pf.update to inject ps)
            log_lik = np.log(np.maximum(combined_likelihoods, 1e-300))
            layer.pf.log_weights += log_lik

            # Normalize
            lse = _log_sum_exp(layer.pf.log_weights)
            if np.isfinite(lse):
                layer.pf.log_weights -= lse
            else:
                # Collapsed — reinitialize this layer
                layer.pf.particles = layer.pf._initialize_particles()
                layer.pf.log_weights = np.full(layer.pf.N, -np.log(layer.pf.N))
                continue

            # Store for MCMC
            layer.pf.last_measurement = corrected_measurement
            layer.pf.last_sensor_position = sensor_position
            layer.pf.iteration += 1

            # Resample if needed
            w = layer.pf.weights
            n_eff = 1.0 / np.sum(w ** 2)
            if n_eff < layer.pf.resample_threshold * layer.pf.N:
                layer.pf._regularized_resample()

        # Update convergence status for all layers
        self._update_convergence_status()

    def _compute_corrected_measurement(self, measurement: float,
                                        sensor_position: Tuple[float, float],
                                        exclude_layer: int) -> float:
        """
        OIC: Subtract contributions from confirmed sources.

        z_corrected = z_observed - sum(R(rk|theta_j)) for confirmed j != exclude_layer
        """
        correction = 0.0
        for source in self.confirmed_sources:
            source_loc = (source['x'], source['y'])
            correction += self.dispersion_model.compute_concentration(
                sensor_position, source_loc, source['Q']
            )
        return measurement - correction

    def _compute_peak_suppression(self, current_layer: LayerState,
                                   layer_estimates: List[Dict]) -> np.ndarray:
        """
        Peak suppression weight for each particle in current_layer.

        w_ps^l(theta) = prod_{j != l} [1 - exp(-||theta_xy - mu_j||^2 / (2 * r_ps^2))]

        This makes particles near other layers' estimates have low weight,
        forcing each layer to find a different source.
        """
        particles_xy = current_layer.pf.particles[:, :2]  # (N, 2)
        ps_weight = np.ones(len(particles_xy))

        for j, est in enumerate(layer_estimates):
            if j == current_layer.id:
                continue

            other_layer = self.layers[j]
            # Don't suppress against non-converged, dispersed layers
            other_est, other_std = other_layer.pf.get_estimate()
            if max(other_std['x'], other_std['y']) > self.convergence_sigma * 3:
                # Layer j is too dispersed to suppress against
                continue

            mu_j = np.array([est['x'], est['y']])
            dist_sq = np.sum((particles_xy - mu_j) ** 2, axis=1)
            suppression = 1.0 - np.exp(-dist_sq / (2.0 * self.ps_radius ** 2))
            ps_weight *= suppression

        return ps_weight

    def _update_convergence_status(self):
        """Check each layer for convergence."""
        for layer in self.layers:
            if layer.confirmed:
                continue
            est, std = layer.pf.get_estimate()
            layer.estimate = est
            layer.std = std
            sigma_p = max(std['x'], std['y'])
            layer.converged = sigma_p < self.convergence_sigma

    def verify_and_confirm(self, layer_id: int) -> bool:
        """
        Verify a converged layer represents a real source.

        Checks if the estimated source is not a duplicate of an already
        confirmed source, and if the predicted concentrations from this
        source + confirmed sources explain the measurement history.

        Returns True if confirmed, False if rejected (pseudo-source).
        """
        layer = self.layers[layer_id]
        if not layer.converged:
            return False

        est = layer.estimate

        # 1. Check not duplicate of confirmed source
        for cs in self.confirmed_sources:
            dist = np.hypot(est['x'] - cs['x'], est['y'] - cs['y'])
            if dist < self.ps_radius:
                return False  # Too close to existing confirmed source

        # 2. RMSE verification against recent measurement history
        if len(self.measurement_history) < 5:
            # Not enough data, confirm optimistically
            layer.confirmed = True
            self.confirmed_sources.append(dict(est))
            return True

        recent = self.measurement_history[-min(20, len(self.measurement_history)):]
        errors = []
        for pos, z_obs in recent:
            z_pred = self.dispersion_model.compute_concentration(
                pos, (est['x'], est['y']), est['Q']
            )
            for cs in self.confirmed_sources:
                z_pred += self.dispersion_model.compute_concentration(
                    pos, (cs['x'], cs['y']), cs['Q']
                )
            errors.append((z_obs - z_pred) ** 2)

        rmse = np.sqrt(np.mean(errors))

        if rmse < self.verification_threshold:
            layer.confirmed = True
            self.confirmed_sources.append(dict(est))
            return True

        return False

    def get_all_estimates(self) -> List[Dict]:
        """Return estimates from all layers with metadata."""
        results = []
        for layer in self.layers:
            est, std = layer.pf.get_estimate()
            results.append({
                'layer_id': layer.id,
                'estimate': est,
                'std': std,
                'converged': layer.converged,
                'confirmed': layer.confirmed,
                'sigma_p': max(std['x'], std['y'])
            })
        return results

    def get_confirmed_sources(self) -> List[Dict]:
        """Return list of confirmed source parameters."""
        return list(self.confirmed_sources)

    def get_active_estimates(self) -> List[Dict]:
        """Return estimates from converged (including confirmed) layers."""
        results = []
        for layer in self.layers:
            if layer.converged or layer.confirmed:
                est, std = layer.pf.get_estimate()
                results.append({
                    'layer_id': layer.id,
                    'x': est['x'], 'y': est['y'], 'Q': est['Q'],
                    'confirmed': layer.confirmed
                })
        return results

    def get_source_count_estimate(self) -> int:
        """Estimate number of sources from converged/confirmed layers."""
        count = 0
        for layer in self.layers:
            if layer.converged or layer.confirmed:
                count += 1
        return count

    def all_sources_resolved(self) -> bool:
        """
        Check if search is complete.
        All layers are either confirmed or dispersed (sigma > 3x threshold).
        """
        for layer in self.layers:
            if layer.confirmed:
                continue
            est, std = layer.pf.get_estimate()
            sigma_p = max(std['x'], std['y'])
            # Layer is neither confirmed nor dispersed
            if sigma_p < self.convergence_sigma * 3:
                if layer.converged:
                    # Converged but not confirmed — needs verification
                    return False
        # At least one source must be confirmed
        return len(self.confirmed_sources) > 0

    def get_best_layer_for_planning(self) -> Optional[LayerState]:
        """
        Return the most uncertain (highest entropy) non-confirmed layer
        for planning purposes.
        """
        best = None
        best_entropy = -np.inf
        for layer in self.layers:
            if layer.confirmed:
                continue
            entropy = layer.pf.get_entropy()
            if entropy > best_entropy:
                best_entropy = entropy
                best = layer
        return best

    def get_all_particles_and_weights(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Return particles, weights, and layer IDs for visualization."""
        all_particles = []
        all_weights = []
        all_layer_ids = []
        for layer in self.layers:
            p, w = layer.pf.get_particles()
            all_particles.append(p)
            all_weights.append(w)
            all_layer_ids.append(layer.id)
        return all_particles, all_weights, all_layer_ids
