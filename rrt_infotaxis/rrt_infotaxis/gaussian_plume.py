import numpy as np
class GaussianPlumeModel:
    """Gaussian plume dispersion model for gas concentration."""
    
    def __init__(self, wind_velocity=2.0, wind_direction=90, 
                 zeta1=0.2, zeta2=0.1, source_height=11, agent_height=11):
        """
        Initialize the Gaussian plume model.

        Parameters:
        -----------
        wind_velocity : float
            Mean wind velocity V in m/s
        wind_direction : float
            Wind direction in degrees (0° = +X, 90° = +Y)
            Direction the wind blows TO
        zeta1, zeta2 : float
            Stochastic diffusion terms (crosswind, vertical)
        source_height : float
            Height of gas source in meters
        agent_height : float
            Height of mobile agent in meters
        """
        self.V = wind_velocity
        self.wind_direction = np.radians(wind_direction)  # Direction wind blows TO
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.z0 = source_height
        self.agent_height = agent_height

    def compute_concentration(self, position : tuple[float, float], source_location : tuple[float, float], release_rate : float) -> float:
        """
        Compute gas concentration at a given position.
        
        Parameters:
        -----------
        position : tuple (x, y)
            Position to compute concentration at
        source_location : tuple (x0, y0)
            Source location
        release_rate : float
            Gas release rate Q0 in g/s
            
        Returns:
        --------
        concentration : float
            Mean gas concentration in μg/m³
        """
        x, y = position
        x0, y0 = source_location
        Q0 = release_rate * 1e6 # Convert g/s to μg/s
        
        # Transform to wind-aligned coordinate system
        dx = x - x0
        dy = y - y0
        
        # Rotate to align with wind direction
        downwind = dx * np.cos(self.wind_direction) + dy * np.sin(self.wind_direction)
        crosswind = -dx * np.sin(self.wind_direction) + dy * np.cos(self.wind_direction)
        
        # Only compute concentration downwind of source
        if downwind <= 0.1:
            return 0.0
        
        # Compute standard deviations (Equation 2 from paper)
        sigma_y = self.zeta1 * downwind / np.sqrt(1 + 0.0001 * downwind)
        sigma_z = self.zeta2 * downwind / np.sqrt(1 + 0.0001 * downwind)
        
        if sigma_y < 0.01 or sigma_z < 0.01:
            return 0.0
        
        # Compute concentration (Equation 1)
        crosswind_term = np.exp(-crosswind**2 / (2 * sigma_y**2))
        z_term = (np.exp(-(self.agent_height - self.z0)**2 / (2 * sigma_z**2)) + 
                  np.exp(-(self.agent_height + self.z0)**2 / (2 * sigma_z**2)))
        
        concentration = (Q0 / (2 * np.pi * self.V * sigma_y * sigma_z) * 
                        crosswind_term * z_term)
        
        return concentration
    def likelihood(self, position, source_location, release_rate):
        """Wrapper for compute_concentration for likelihood calculations."""
        return self.compute_concentration(position, source_location, release_rate)