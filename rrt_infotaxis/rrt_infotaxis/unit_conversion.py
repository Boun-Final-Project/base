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


def calibrate_release_rate_from_measurements(measurements, positions, source_location,
                                             plume_model, converter, percentile=90):
    """
    Calibrate the release rate Q from real measurements.

    Strategy: Use high measurements (e.g., 90th percentile) near the source
    to estimate Q, since the plume model should match peak concentrations.

    Parameters:
    -----------
    measurements : list of float
        Sensor measurements in PPM
    positions : list of tuple
        Corresponding (x, y) positions where measurements were taken
    source_location : tuple
        Estimated source location (x0, y0)
    plume_model : GaussianPlumeModel
        Gaussian plume model instance
    converter : GasUnitConverter
        Unit converter instance
    percentile : float
        Percentile of measurements to use for calibration (default: 90)

    Returns:
    --------
    Q_estimated : float
        Estimated release rate in g/s
    """
    # Convert PPM measurements to μg/m³
    measurements_ug_m3 = [converter.ppm_to_ug_m3(ppm) for ppm in measurements]

    # Find high measurements for calibration
    threshold = np.percentile(measurements_ug_m3, percentile)

    # Collect (position, measurement) pairs above threshold
    calibration_data = []
    for pos, meas in zip(positions, measurements_ug_m3):
        if meas >= threshold:
            calibration_data.append((pos, meas))

    if len(calibration_data) == 0:
        print(f"Warning: No measurements above {percentile}th percentile. Using all data.")
        calibration_data = list(zip(positions, measurements_ug_m3))

    # Estimate Q for each calibration point and take median
    Q_estimates = []
    for pos, meas in calibration_data:
        # Compute what Q would be needed to produce this measurement
        # C = Q × f(x, y)  =>  Q = C / f(x, y)
        # Use Q=1.0 to get the spatial distribution factor
        spatial_factor = plume_model.compute_concentration(pos, source_location, Q0=1.0)

        if spatial_factor > 1e-6:  # Avoid division by zero
            Q_estimate = meas / spatial_factor
            Q_estimates.append(Q_estimate)

    if len(Q_estimates) == 0:
        print("Warning: Could not estimate Q. Using default Q=1.0 g/s")
        return 1.0

    # Take median to be robust to outliers
    Q_estimated = np.median(Q_estimates) / 1e6  # Convert μg/s back to g/s

    return Q_estimated


def estimate_source_strength_simple(max_ppm, distance_to_source, wind_velocity, converter):
    """
    Simple rule-of-thumb estimation of source strength from maximum PPM reading.

    This is a rough approximation: Q ≈ C_max × V × distance²

    Parameters:
    -----------
    max_ppm : float
        Maximum sensor reading in PPM
    distance_to_source : float
        Approximate distance to source in meters
    wind_velocity : float
        Wind velocity in m/s
    converter : GasUnitConverter
        Unit converter

    Returns:
    --------
    Q_estimated : float
        Rough estimate of release rate in g/s
    """
    # Convert to μg/m³
    max_ug_m3 = converter.ppm_to_ug_m3(max_ppm)

    # Very rough approximation based on inverse square law
    # Q ≈ C × V × distance² / (some constant)
    # The constant depends on atmospheric stability, typically ~10-100
    Q_estimated = max_ug_m3 * wind_velocity * (distance_to_source ** 2) / 50.0

    # Convert to g/s
    Q_estimated = Q_estimated / 1e6

    return Q_estimated


# Example usage
if __name__ == '__main__':
    # Create converter for ethanol
    converter = GasUnitConverter('ethanol', temperature=25.0)

    # Test conversion
    test_ppm = 100.0
    ug_m3 = converter.ppm_to_ug_m3(test_ppm)
    back_to_ppm = converter.ug_m3_to_ppm(ug_m3)

    print(f"Ethanol conversion:")
    print(f"  {test_ppm} PPM = {ug_m3:.2f} μg/m³")
    print(f"  Back conversion: {back_to_ppm:.2f} PPM")

    # Test for different gases
    print("\nComparison for 100 PPM:")
    for gas in ['ethanol', 'acetone', 'methane']:
        conv = GasUnitConverter(gas)
        result = conv.ppm_to_ug_m3(100.0)
        print(f"  {gas}: {result:.2f} μg/m³")
