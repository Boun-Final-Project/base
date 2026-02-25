import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import efe_igdm
    print(f"DEBUG: efe_igdm location: {efe_igdm.__file__}")
    import efe_igdm.mapping
    print(f"DEBUG: efe_igdm.mapping location: {efe_igdm.mapping.__file__}")
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")

from efe_igdm.mapping.gmrf_wind_mapper import GMRFWindMapper

class TestGMRFWindMapper(unittest.TestCase):
    def test_simple_flow(self):
        """Test simple flow from left to right in an empty corridor."""
        width = 10
        height = 5
        grid = np.zeros((height, width), dtype=np.int8)
        
        # Measurement at left (inlet) and right (outlet)
        measured_u = np.zeros((height, width))
        measured_v = np.zeros((height, width))
        mask = np.zeros((height, width), dtype=bool)
        
        # Inlet at (2, 0): Wind (1, 0)
        measured_u[2, 0] = 1.0
        mask[2, 0] = True
        
        # Outlet at (2, 9): Wind (1, 0)
        measured_u[2, 9] = 1.0
        mask[2, 9] = True
        
        mapper = GMRFWindMapper(lambda_smoothness=1.0, lambda_divergence=10.0)
        u, v = mapper.solve(grid, 1.0, measured_u, measured_v, mask)
        
        # Check center flow
        print("Center U:", u[2, 5])
        self.assertGreater(u[2, 5], 0.1, "Flow should propagate to center")
        self.assertAlmostEqual(v[2, 5], 0.0, delta=0.1, msg="Vertical flow should be minimal")

    def test_obstacle_avoidance(self):
        """Test flow around a central obstacle."""
        width = 10
        height = 5
        grid = np.zeros((height, width), dtype=np.int8)
        
        # Obstacle in center
        grid[2, 5] = 1 # Occupied
        
        measured_u = np.zeros((height, width))
        measured_v = np.zeros((height, width))
        mask = np.zeros((height, width), dtype=bool)
        
        # Inlet (2, 0) -> (1, 0)
        measured_u[2, 0] = 1.0
        mask[2, 0] = True
        
        # Outlet (2, 9) -> (1, 0)
        measured_u[2, 9] = 1.0
        mask[2, 9] = True
        
        mapper = GMRFWindMapper(lambda_smoothness=1.0, lambda_divergence=10.0)
        u, v = mapper.solve(grid, 1.0, measured_u, measured_v, mask)
        
        # Flow should be zero AT the obstacle
        self.assertAlmostEqual(u[2, 5], 0.0, delta=1e-3)
        self.assertAlmostEqual(v[2, 5], 0.0, delta=1e-3)
        
        # Flow should divert around (e.g. at 1,5 and 3,5)
        print("Flow above obstacle (1,5):", u[1, 5], v[1, 5])
        print("Flow below obstacle (3,5):", u[3, 5], v[3, 5])
        
        self.assertGreater(u[1, 5], 0.0)
        self.assertGreater(u[3, 5], 0.0)

if __name__ == '__main__':
    unittest.main()
