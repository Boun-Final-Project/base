"""
Unit conversion utilities for gas concentration measurements.
"""

import numpy as np


class GasUnitConverter:
    """Convert between PPM and mass concentration for different gases."""

    # Molecular weights in g/mol
    MOLECULAR_WEIGHTS = {
        'ethanol': 46.07,    # C2H6O
        'acetone': 58.08,    # C3H6O
        'methane': 16.04,    # CH4
        'propane': 44.10,    # C3H8
        'air': 28.97         # Average for air
    }

    def __init__(self, gas_type='ethanol', temperature=25.0, pressure=101.325):
        """
        Initialize unit converter.

        Parameters:
        -----------
        gas_type : str
            Type of gas ('ethanol', 'acetone', 'methane', 'propane')
        temperature : float
            Temperature in Celsius (default: 25°C)
        pressure : float
            Pressure in kPa (default: 101.325 kPa = 1 atm)
        """
        self.gas_type = gas_type.lower()
        if self.gas_type not in self.MOLECULAR_WEIGHTS:
            raise ValueError(f"Unknown gas type: {gas_type}. Available: {list(self.MOLECULAR_WEIGHTS.keys())}")

        self.MW = self.MOLECULAR_WEIGHTS[self.gas_type]
        self.temperature = temperature  # Celsius
        self.pressure = pressure  # kPa

    def ppm_to_ug_m3(self, ppm):
        """
        Convert PPM (parts per million by volume) to μg/m³.

        Uses ideal gas law at given temperature and pressure:
        C [μg/m³] = PPM × (MW / 24.45) × (273.15 / (273.15 + T)) × (P / 101.325) × 1000

        At standard conditions (25°C, 1 atm):
        C [μg/m³] = PPM × (MW / 24.45) × 1000

        Parameters:
        -----------
        ppm : float or np.ndarray
            Concentration in parts per million

        Returns:
        --------
        ug_m3 : float or np.ndarray
            Concentration in μg/m³
        """
        # Temperature correction factor
        T_kelvin = 273.15 + self.temperature
        temp_factor = 273.15 / T_kelvin

        # Pressure correction factor
        pressure_factor = self.pressure / 101.325

        # Conversion factor (at STP, molar volume is 24.45 L/mol)
        # PPM to mg/m³: multiply by (MW / 24.45)
        # mg/m³ to μg/m³: multiply by 1000
        ug_m3 = ppm * (self.MW / 24.45) * temp_factor * pressure_factor * 1000

        return ug_m3

    def ug_m3_to_ppm(self, ug_m3):
        """
        Convert μg/m³ to PPM (parts per million by volume).

        Parameters:
        -----------
        ug_m3 : float or np.ndarray
            Concentration in μg/m³

        Returns:
        --------
        ppm : float or np.ndarray
            Concentration in parts per million
        """
        # Temperature correction factor
        T_kelvin = 273.15 + self.temperature
        temp_factor = T_kelvin / 273.15

        # Pressure correction factor
        pressure_factor = 101.325 / self.pressure

        # Reverse conversion
        ppm = ug_m3 / 1000 * (24.45 / self.MW) * temp_factor * pressure_factor

        return ppm
