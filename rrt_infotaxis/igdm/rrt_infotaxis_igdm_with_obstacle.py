"""
RRT-Infotaxis with IGDM and Obstacle Configuration.

This is a variant of the standard RRT-Infotaxis algorithm that includes an obstacle
in the environment. The obstacle tests whether the gas dispersion model correctly
respects the occupancy grid and computes Dijkstra distances around obstacles.

Configuration:
- 10x6 meter room with one rectangular obstacle
- Obstacle at x=4.0, spanning y=2.0 to y=4.0 (height=2.0m)
- Source at (2, 3) with Q=1.0 (source is in free space, not blocked by obstacle)
- Robot starts at (9, 3)
- IGDM uses Dijkstra distance (obstacle-aware)
"""

from rrt_infotaxis_igdm import RRTInfotaxisIGDM


class RRTInfotaxisIGDMWithObstacle(RRTInfotaxisIGDM):
    """RRT-Infotaxis with IGDM and obstacle in the environment."""

    def __init__(self, sigma_m=1.0):
        """
        Initialize RRT-Infotaxis with IGDM and add an obstacle.

        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter for IGDM model
        """
        # Initialize parent class
        super().__init__(sigma_m=sigma_m)

        # Add rectangular obstacle at x=4, spanning y=2 to y=4
        # This creates a vertical wall that blocks gas from spreading straight through
        self.grid.add_rectangular_obstacle(
            x_min=3.9,
            x_max=4.1,
            y_min=2.0,
            y_max=4.0,
            value=1
        )

        print("\n[OBSTACLE] Added rectangular obstacle:")
        print(f"  Position: x=4.0 (width=0.2m)")
        print(f"  Height: y=2.0 to y=4.0 (height=2.0m)")
        print(f"  Gas dispersion will compute Dijkstra distances around this obstacle")


def main():
    """Run the RRT-Infotaxis with IGDM and obstacle test."""
    test = RRTInfotaxisIGDMWithObstacle(sigma_m=1.0)
    test.run()
    test.visualize_final(filename='rrt_infotaxis_igdm_obstacle_result.png')

    print("\n" + "=" * 70)
    print("Test with obstacle completed!")
    print("=" * 70)
    print("Outputs created:")
    print(f"  - Final plot: rrt_infotaxis_igdm_obstacle_result.png")
    print(f"  - Step frames: igdm_steps/*.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
