"""
Test script for RRT-Infotaxis with IGDM.

This script demonstrates RRT-Infotaxis implementation with the 
Indoor Gaussian Dispersion Model (IGDM).

Algorithm:
1. MEASURE: Take sensor measurement, update threshold, update particle filter
2. PLAN: Build RRT, evaluate paths with entropy gain vs travel cost
3. MOVE: Navigate to next position
4. Check: If estimation converged, stop

- 10x6 meter empty room
- Source at (2, 3) with Q=1.0
- Robot starts at (9, 3)
- IGDM uses Dijkstra distance (no wind)
"""

from rrt_infotaxis_igdm import RRTInfotaxisIGDM

def main():
    """Run the RRT-Infotaxis with IGDM test."""
    test = RRTInfotaxisIGDM(sigma_m=1.0)
    test.run()
    test.visualize_final(filename='rrt_infotaxis_igdm_result.png')

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("Outputs created:")
    print(f"  - Final plot: rrt_infotaxis_igdm_result.png")
    print(f"  - Step frames: igdm_steps/*.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
