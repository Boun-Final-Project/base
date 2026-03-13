"""
Source Verifier for multi-source gas source localization.

Validates whether a converged particle filter layer represents a real source
or a pseudo-source (artifact of concentration superposition).

Simplified from PID-STE (Bai et al., Building & Environment 2023):
Uses RMSE comparison instead of full Poisson kriging interpolation.
"""

import numpy as np
from typing import List, Dict, Tuple
from .igdm_gas_model import IndoorGaussianDispersionModel


class SourceVerifier:
    """
    Verifies estimated sources against measurement history.
    """

    def __init__(self,
                 dispersion_model: IndoorGaussianDispersionModel,
                 rmse_threshold: float = 5.0,
                 min_measurements: int = 5,
                 history_window: int = 20):
        self.dispersion_model = dispersion_model
        self.rmse_threshold = rmse_threshold
        self.min_measurements = min_measurements
        self.history_window = history_window

    def verify(self,
               candidate: Dict,
               confirmed_sources: List[Dict],
               measurement_history: List[Tuple[Tuple[float, float], float]],
               min_distance: float = 1.0) -> bool:
        """
        Check if a candidate source is real.

        Args:
            candidate: {'x', 'y', 'Q'} estimated source parameters
            confirmed_sources: List of already confirmed source dicts
            measurement_history: List of (position, measurement) tuples
            min_distance: Minimum distance from confirmed sources

        Returns:
            True if source passes verification
        """
        # 1. Duplicate check
        for cs in confirmed_sources:
            dist = np.hypot(candidate['x'] - cs['x'], candidate['y'] - cs['y'])
            if dist < min_distance:
                return False

        # 2. Not enough data — accept optimistically
        if len(measurement_history) < self.min_measurements:
            return True

        # 3. RMSE check
        recent = measurement_history[-self.history_window:]
        errors = []

        for pos, z_obs in recent:
            # Predict total concentration from candidate + confirmed
            z_pred = self.dispersion_model.compute_concentration(
                pos, (candidate['x'], candidate['y']), candidate['Q']
            )
            for cs in confirmed_sources:
                z_pred += self.dispersion_model.compute_concentration(
                    pos, (cs['x'], cs['y']), cs['Q']
                )
            errors.append((z_obs - z_pred) ** 2)

        rmse = np.sqrt(np.mean(errors))
        return rmse < self.rmse_threshold

    def cross_validate(self,
                       all_estimates: List[Dict],
                       measurement_history: List[Tuple[Tuple[float, float], float]]) -> float:
        """
        Compute RMSE for the complete set of estimated sources.

        Args:
            all_estimates: List of {'x', 'y', 'Q'} dicts
            measurement_history: List of (position, measurement) tuples

        Returns:
            RMSE value
        """
        if not measurement_history:
            return float('inf')

        recent = measurement_history[-self.history_window:]
        errors = []

        for pos, z_obs in recent:
            z_pred = 0.0
            for est in all_estimates:
                z_pred += self.dispersion_model.compute_concentration(
                    pos, (est['x'], est['y']), est['Q']
                )
            errors.append((z_obs - z_pred) ** 2)

        return np.sqrt(np.mean(errors))
