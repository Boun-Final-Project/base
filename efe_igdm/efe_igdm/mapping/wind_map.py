"""
Wind Map - Spatial record of wind measurements on an occupancy grid.

Stores wind vector (vx, vy) at each grid cell as the robot explores,
enabling later extrapolation/interpolation of the wind field.
"""

import numpy as np
from typing import Optional, Tuple, List
from .potential_flow import PotentialFlowEstimator
from .gmrf_wind_mapper import GMRFWindMapper


class WindMap:
    """
    Spatial wind measurement map aligned with the occupancy grid.
    
    Stores running averages of wind vectors (vx, vy) at each grid cell.
    Cells without measurements are marked with NaN.
    """

    def __init__(self, width: int, height: int, resolution: float,
                 origin_x: float, origin_y: float):
        """
        Initialize wind map with same dimensions as occupancy grid.

        Parameters
        ----------
        width : int
            Grid width in cells
        height : int
            Grid height in cells
        resolution : float
            Cell size in meters
        origin_x, origin_y : float
            World coordinates of grid origin
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Wind vector components (y, x) indexing to match occupancy grid
        self.wind_vx = np.full((height, width), np.nan, dtype=np.float64)
        self.wind_vy = np.full((height, width), np.nan, dtype=np.float64)

        # Measurement count per cell (for running average)
        self.measurement_count = np.zeros((height, width), dtype=np.int32)

        # History: list of (x, y, vx, vy, timestamp) for all raw measurements
        self.history: List[Tuple[float, float, float, float, float]] = []

        # Potential flow estimator and its output
        self.pf_estimator = PotentialFlowEstimator(max_iter=500, tol=1e-4)
        self.gmrf_mapper = GMRFWindMapper()

        self.estimated_vx = np.zeros((height, width), dtype=np.float64)
        self.estimated_vy = np.zeros((height, width), dtype=np.float64)
        self.pf_solved = False

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(np.floor((x - self.origin_x) / self.resolution))
        gy = int(np.floor((y - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def add_measurement(self, world_x: float, world_y: float,
                        wind_speed: float, wind_direction: float,
                        timestamp: Optional[float] = None):
        """
        Record a wind measurement at the given world position.

        Uses incremental running average so each new measurement
        at a cell updates the stored value smoothly.

        Parameters
        ----------
        world_x, world_y : float
            Robot position in world coordinates
        wind_speed : float
            Wind speed in m/s
        wind_direction : float
            Wind direction in radians
        timestamp : float, optional
            Time of measurement (seconds)
        """
        gx, gy = self.world_to_grid(world_x, world_y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return

        vx = wind_speed * np.cos(wind_direction)
        vy = wind_speed * np.sin(wind_direction)

        # Store raw measurement in history
        self.history.append((world_x, world_y, vx, vy, timestamp or 0.0))

        # Incremental running average
        n = self.measurement_count[gy, gx]
        if n == 0:
            self.wind_vx[gy, gx] = vx
            self.wind_vy[gy, gx] = vy
        else:
            self.wind_vx[gy, gx] = (self.wind_vx[gy, gx] * n + vx) / (n + 1)
            self.wind_vy[gy, gx] = (self.wind_vy[gy, gx] * n + vy) / (n + 1)
        self.measurement_count[gy, gx] = n + 1

    def get_wind_at(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        """
        Get the recorded wind vector at a world position.

        Returns
        -------
        (vx, vy) or None if no measurement at this cell
        """
        gx, gy = self.world_to_grid(world_x, world_y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return None
        if self.measurement_count[gy, gx] == 0:
            return None
        return (self.wind_vx[gy, gx], self.wind_vy[gy, gx])

    def get_measured_cells(self) -> np.ndarray:
        """Return boolean mask of cells that have been measured."""
        return self.measurement_count > 0

    def get_speed_map(self) -> np.ndarray:
        """Return 2D array of wind speeds (NaN where unmeasured)."""
        return np.sqrt(self.wind_vx**2 + self.wind_vy**2)

    def get_direction_map(self) -> np.ndarray:
        """Return 2D array of wind directions in radians (NaN where unmeasured)."""
        return np.arctan2(self.wind_vy, self.wind_vx)

    @property
    def num_measured_cells(self) -> int:
        """Number of cells with at least one measurement."""
        return int(np.sum(self.measurement_count > 0))

    @property
    def total_measurements(self) -> int:
        """Total number of measurements recorded."""
        return len(self.history)

    def solve_potential_flow(self, slam_grid: np.ndarray) -> bool:
        """
        Estimate wind field using potential flow on the SLAM grid.

        Uses anemometer measurements to determine correct flow direction
        at outlets (source vs sink). Requires at least 1 outlet cluster
        and either a second outlet or frontier cells.

        Parameters
        ----------
        slam_grid : ndarray (height, width)
            Current SLAM grid with values: -1=unknown, 0=free, 1=occupied, 2=outlet.

        Returns
        -------
        success : bool
            True if solve succeeded.
        """
        self.estimated_vx, self.estimated_vy = self.pf_estimator.solve(
            slam_grid, self.resolution,
            measured_vx=self.wind_vx,
            measured_vy=self.wind_vy,
            measurement_count=self.measurement_count
        )
        self.pf_solved = self.pf_estimator.num_clusters >= 2
        return self.pf_solved

    def get_estimated_wind_at(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Get the estimated wind vector (from potential flow) at a world position.

        Falls back to (0, 0) if not solved or out of bounds.
        """
        if not self.pf_solved:
            return (0.0, 0.0)
        gx, gy = self.world_to_grid(world_x, world_y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return (self.estimated_vx[gy, gx], self.estimated_vy[gy, gx])
        return (0.0, 0.0)

    def solve_gmrf(self, slam_grid: np.ndarray) -> bool:
        """
        Estimate wind field using GMRF on the SLAM grid.
        
        Uses all sparse measurements to solve for the field globally.
        """
        # Prepare measurement inputs (replace NaNs with 0, pass mask)
        mask = self.measurement_count > 0
        m_u = np.nan_to_num(self.wind_vx)
        m_v = np.nan_to_num(self.wind_vy)
        
        self.estimated_vx, self.estimated_vy = self.gmrf_mapper.solve(
            slam_grid, self.resolution,
            measured_u=m_u, measured_v=m_v,
            measurement_mask=mask
        )
        self.pf_solved = True # Reuse flag for simplicity
        return True
