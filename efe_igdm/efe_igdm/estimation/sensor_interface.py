from abc import ABC, abstractmethod
import numpy as np

class SensorModel(ABC):
    """
    Abstract Base Class for all sensor models.
    Enforces the interface required by the Particle Filter.
    """

    @abstractmethod
    def compute_likelihood(self, measurement: float, predictions: np.ndarray) -> np.ndarray:
        """
        P(z | x): Calculate the likelihood of a real measurement given state predictions.
        
        Args:
            measurement: The actual sensor reading.
            predictions: Predicted concentrations for all particles (N,).
            
        Returns:
            likelihoods: Probability of the measurement for each particle (N,).
        """
        pass

    @abstractmethod
    def compute_predictive_distribution(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate the probability of ALL discrete outcomes (bins) for planning.
        
        Args:
            predictions: Predicted concentrations for all particles (N,).
            
        Returns:
            bin_probs: Shape (Num_Bins, N_Particles).
        """
        pass

    @abstractmethod
    def compute_likelihood_for_bin(self, bin_index: int, predictions: np.ndarray) -> np.ndarray:
        """
        P(z in Bin_i | x): Calculate likelihood that a measurement falls into a specific bin.
        Used for hypothetical entropy calculation.
        """
        pass