"""
RRT-Infotaxis with IGDM - Large Map with Room Configuration.

This test uses a larger 25x25 meter environment with a room in the top-left corner.
The source is located inside the room, and the robot must navigate from the
open area to find it. The room has an open door on its bottom wall.

Configuration:
- Map size: 25x25 meters
- Room: Located in top-left corner
  - Bounds: x ∈ [0, 5], y ∈ [20, 25]
  - Room size: 5m x 5m
- Walls:
  - Right wall at x=5 (closed)
  - Bottom wall at y=20 with 2m door at x ∈ [0, 2]
  - Top wall at y=25 (boundary)
  - Left wall at x=0 (boundary)
- Source location: (2.5, 22.5) inside the room (center)
- Robot start: (12, 13) in the open area
- IGDM uses Dijkstra distance (obstacle-aware)
"""

from rrt_infotaxis_igdm import RRTInfotaxisIGDM


class RRTInfotaxisIGDMLargeMap(RRTInfotaxisIGDM):
    """RRT-Infotaxis on a large 25x25 map with a room in top-left corner."""

    def __init__(self, sigma_m=1.0):
        """
        Initialize RRT-Infotaxis with IGDM on a large map with room.

        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter for IGDM model
        """
        # Initialize parent class (will create 10x6 grid first)
        super().__init__(sigma_m=sigma_m)

        # Replace room dimensions and grid
        self.room_width = 25.0
        self.room_height = 25.0
        self.resolution = 0.25
        self.true_Q = 5.0


        # Recreate the grid with new dimensions
        from occupancy_grid import OccupancyGrid
        from igdm_model import IGDMModel
        from particle_filter import ParticleFilter
        from rrt import RRTInfotaxis

        self.grid = OccupancyGrid(self.room_width, self.room_height, self.resolution)

        # Add walls to create the room (5x5 at top-left corner)
        # Right wall at x=5 (from y=20 to y=25)
        self.grid.add_rectangular_obstacle(
            x_min=4.9,
            x_max=5.1,
            y_min=20.0,
            y_max=25.0,
            value=1
        )

        # Bottom wall at y=20 with 2m door at x=[0,2]
        # Wall spans from x=2 to x=5 (door at x=[0,2])
        self.grid.add_rectangular_obstacle(
            x_min=2.0,
            x_max=5.0,
            y_min=19.9,
            y_max=20.1,
            value=1
        )

        # Update IGDM with new grid
        self.igdm = IGDMModel(sigma_m=sigma_m, occupancy_grid=self.grid, dispersion_rate=1.80)

        # Update visualizer with new IGDM model
        self.visualizer.igdm_model = self.igdm

        # Update particle filter search bounds for new map
        self.particle_filter = ParticleFilter(
            num_particles=400,
            search_bounds={'x': (0, self.room_width), 'y': (0, self.room_height), 'Q': (0, 2.0)},
            binary_sensor_model=self.sensor,
            dispersion_model=self.igdm
        )

        # Update RRT with new grid
        self.rrt = RRTInfotaxis(self.grid, N_tn=50, R_range=10, delta=1.0, max_depth=2,
                               discount_factor=0.8, positive_weight=0.5)

        # Set source location inside the room (center)
        self.true_source = (2.5, 22.5)

        # Set robot start position outside the room
        self.robot_start = (12.0, 13.0)
        self.robot_pos = self.robot_start
        self.trajectory = [self.robot_pos]

        print("\n[LARGE MAP] Configuration:")
        print(f"  Map size: {self.room_width}m × {self.room_height}m")
        print(f"  Room location: Top-left corner")
        print(f"  Room bounds: x ∈ [0, 5], y ∈ [20, 25]")
        print(f"  Room walls:")
        print(f"    - Right wall at x=5.0 (y: 20→25)")
        print(f"    - Bottom wall at y=20.0 (x: 2→5, door at x: 0→2)")
        print(f"  Source location: {self.true_source} (inside room, center)")
        print(f"  Robot start: {self.robot_start} (open area)")


def main():
    """Run the RRT-Infotaxis with IGDM on large map test."""
    test = RRTInfotaxisIGDMLargeMap(sigma_m=1.0)
    test.run()
    test.visualize_final(filename='rrt_infotaxis_igdm_large_map_result.png')

    print("\n" + "=" * 70)
    print("Large map test completed!")
    print("=" * 70)
    print("Outputs created:")
    print(f"  - Final plot: rrt_infotaxis_igdm_large_map_result.png")
    print(f"  - Step frames: igdm_steps/*.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
