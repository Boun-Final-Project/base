#!/usr/bin/env python3
"""
Standalone ideal test scenario for RRT-Infotaxis (no ROS2 dependencies).
- 10x6 meter empty room
- Source at (2, 3) with Q=1.0
- Robot starts at (9, 3)
- Wind blowing at 0.1 m/s in +X direction (0 degrees)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from gaussian_plume import GaussianPlumeModel
from sensor_model import BinarySensorModel
from particle_filter import ParticleFilter
from occupancy_grid import OccupancyGridMap
from rrt import RRT


class IdealScenario:
    """Ideal test scenario for RRT-Infotaxis."""

    def __init__(self):
        # Environment parameters
        self.room_width = 10.0  # meters
        self.room_height = 6.0  # meters
        self.resolution = 0.1   # 10cm grid cells

        # True source parameters
        self.true_source_x = 2.0
        self.true_source_y = 3.0
        self.true_source_Q = 1.0

        # Robot start position
        self.robot_x = 9.0
        self.robot_y = 3.0

        # Wind parameters (0 degrees = +X direction)
        self.wind_velocity = 0.1  # m/s
        self.wind_direction = 0.0  # degrees (0=+X, 90=+Y)

        # RRT parameters
        self.n_tn = 50
        self.delta = 1.0
        self.robot_radius = 0.35

        # Simulation parameters
        self.max_steps = 30

        # Create empty occupancy grid
        self.occupancy_grid = self._create_empty_grid()

        # Initialize models
        self.plume_model = GaussianPlumeModel(
            wind_velocity=self.wind_velocity,
            wind_direction=self.wind_direction
        )

        self.sensor_model = BinarySensorModel(
            alpha=0.1,
            sigma_env=1.5,
            threshold_weight=0.5
        )

        self.particle_filter = ParticleFilter(
            num_particles=1000,
            search_bounds={
                "x": (0, self.room_width),
                "y": (0, self.room_height),
                "Q": (0, 2.0)
            },
            binary_sensor_model=self.sensor_model,
            dispersion_model=self.plume_model
        )

        self.rrt = RRT(
            occupancy_grid=self.occupancy_grid,
            N_tn=self.n_tn,
            R_range=self.n_tn * self.delta,
            delta=self.delta,
            robot_radius=self.robot_radius
        )

        # Trajectory storage
        self.robot_trajectory = [(self.robot_x, self.robot_y)]
        self.measurements = []
        self.estimates = []

    def _create_empty_grid(self):
        """Create an empty occupancy grid (no obstacles)."""
        # Calculate grid dimensions
        grid_width = int(self.room_width / self.resolution)
        grid_height = int(self.room_height / self.resolution)

        # Create empty grid (all zeros = free space)
        grid = np.zeros((grid_height, grid_width), dtype=np.int8)

        # Create params dictionary
        params = {
            'env_min': [0.0, 0.0, 0.0],
            'env_max': [self.room_width, self.room_height, 1.0],
            'num_cells': [grid_width, grid_height, 1],
            'cell_size': self.resolution
        }

        return OccupancyGridMap(grid, params)

    def get_measurement(self, position):
        """Get sensor measurement at given position."""
        # Compute true concentration
        true_conc = self.plume_model.compute_concentration(
            position,
            (self.true_source_x, self.true_source_y),
            self.true_source_Q
        )

        # Add Gaussian noise
        sigma = self.sensor_model.get_std(true_conc)
        noisy_measurement = true_conc + np.random.normal(0, sigma)
        noisy_measurement = max(0, noisy_measurement)  # Concentrations can't be negative

        return noisy_measurement

    def run_step(self):
        """Run one iteration of the measure-plan-move loop."""
        current_pos = (self.robot_x, self.robot_y)

        # MEASURE
        measurement = self.get_measurement(current_pos)

        # Initialize or update threshold
        if self.sensor_model.threshold is None:
            self.sensor_model.initialize_threshold(measurement)
            print(f"Initialized threshold: {self.sensor_model.threshold:.4f}")
            return True  # Continue to next step

        self.sensor_model.update_threshold(measurement)
        binary_measurement = self.sensor_model.get_binary_measurement(measurement)

        print(f"\n--- Step {len(self.measurements) + 1} ---")
        print(f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f})")
        print(f"Measurement: {measurement:.4f}, Binary: {binary_measurement}, Threshold: {self.sensor_model.threshold:.4f}")

        self.measurements.append({
            'position': current_pos,
            'raw': measurement,
            'binary': binary_measurement
        })

        # Update particle filter
        self.particle_filter.update(binary_measurement, current_pos)

        # Get estimate
        means, stds = self.particle_filter.get_estimate()
        est_x, est_y, est_Q = means['x'], means['y'], means['Q']
        print(f"Estimate: x={est_x:.2f}±{stds['x']:.2f}, y={est_y:.2f}±{stds['y']:.2f}, Q={est_Q:.2f}±{stds['Q']:.2f}")
        print(f"True source: ({self.true_source_x}, {self.true_source_y}, {self.true_source_Q})")

        self.estimates.append({
            'mean': means,
            'std': stds
        })

        # Check if we found the source
        distance_to_source = np.sqrt((est_x - self.true_source_x)**2 + (est_y - self.true_source_y)**2)
        if distance_to_source < 0.5 and len(self.measurements) > 5:
            print(f"\n✓ Source found! Distance to true source: {distance_to_source:.2f}m")
            return False

        # PLAN
        print(f"Planning RRT paths...")
        debug_info = self.rrt.get_next_move_debug(current_pos, self.particle_filter)
        next_pos = debug_info['next_position']

        print(f"Next position: ({next_pos[0]:.2f}, {next_pos[1]:.2f})")

        # MOVE
        self.robot_x = next_pos[0]
        self.robot_y = next_pos[1]
        self.robot_trajectory.append((self.robot_x, self.robot_y))

        return True

    def run(self):
        """Run the full simulation."""
        print("=" * 60)
        print("RRT-INFOTAXIS IDEAL SCENARIO")
        print("=" * 60)
        print(f"Room: {self.room_width}m x {self.room_height}m")
        print(f"Source: ({self.true_source_x}, {self.true_source_y}) with Q={self.true_source_Q}")
        print(f"Robot start: ({self.robot_x}, {self.robot_y})")
        print(f"Wind: {self.wind_velocity} m/s at {self.wind_direction}° (+X direction)")
        print("=" * 60)

        for step in range(self.max_steps):
            should_continue = self.run_step()
            if not should_continue:
                break

        print(f"\nSimulation complete after {len(self.measurements)} steps")

    def visualize(self, save_path='ideal_scenario_result.png'):
        """Visualize the results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Trajectory and particles
        ax1 = axes[0, 0]
        ax1.set_xlim(0, self.room_width)
        ax1.set_ylim(0, self.room_height)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Robot Trajectory and Particle Distribution')
        ax1.grid(True, alpha=0.3)

        # Plot particles
        particles, weights = self.particle_filter.get_particles()
        normalized_weights = weights / weights.max() if weights.max() > 0 else weights
        scatter = ax1.scatter(particles[:, 0], particles[:, 1], c=normalized_weights,
                            s=20, alpha=0.5, cmap='viridis')
        plt.colorbar(scatter, ax=ax1, label='Normalized Weight')

        # Plot trajectory
        traj_x = [p[0] for p in self.robot_trajectory]
        traj_y = [p[1] for p in self.robot_trajectory]
        ax1.plot(traj_x, traj_y, 'b-o', linewidth=2, markersize=4, label='Robot path')

        # Plot start and true source
        ax1.plot(self.robot_trajectory[0][0], self.robot_trajectory[0][1],
                'go', markersize=15, label='Start')
        ax1.plot(self.true_source_x, self.true_source_y,
                'r*', markersize=20, label='True source')

        # Plot estimated source
        if self.estimates:
            est = self.estimates[-1]['mean']
            ax1.plot(est['x'], est['y'], 'orange', marker='X', markersize=15,
                    label='Estimated source')

        # Add wind arrow
        ax1.arrow(0.5, self.room_height - 0.5, 1.0, 0, head_width=0.3,
                 head_length=0.2, fc='cyan', ec='cyan', linewidth=2)
        ax1.text(1.8, self.room_height - 0.5, 'Wind', fontsize=12, color='cyan')

        ax1.legend()

        # 2. Concentration field (ground truth)
        ax2 = axes[0, 1]
        ax2.set_xlim(0, self.room_width)
        ax2.set_ylim(0, self.room_height)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('True Concentration Field')

        # Create concentration field
        x_grid = np.linspace(0, self.room_width, 100)
        y_grid = np.linspace(0, self.room_height, 60)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.plume_model.compute_concentration(
                    (X[i, j], Y[i, j]),
                    (self.true_source_x, self.true_source_y),
                    self.true_source_Q
                )

        contour = ax2.contourf(X, Y, Z, levels=20, cmap='hot')
        plt.colorbar(contour, ax=ax2, label='Concentration')
        ax2.plot(self.true_source_x, self.true_source_y, 'g*', markersize=20)

        # 3. Estimation error over time
        ax3 = axes[1, 0]
        if self.estimates:
            steps = range(1, len(self.estimates) + 1)
            errors_x = [abs(e['mean']['x'] - self.true_source_x) for e in self.estimates]
            errors_y = [abs(e['mean']['y'] - self.true_source_y) for e in self.estimates]
            errors_total = [np.sqrt((e['mean']['x'] - self.true_source_x)**2 +
                                   (e['mean']['y'] - self.true_source_y)**2)
                          for e in self.estimates]

            ax3.plot(steps, errors_x, 'r-', label='X error')
            ax3.plot(steps, errors_y, 'b-', label='Y error')
            ax3.plot(steps, errors_total, 'k-', linewidth=2, label='Total error')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Error (m)')
            ax3.set_title('Source Localization Error')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Measurements over time
        ax4 = axes[1, 1]
        if self.measurements:
            steps = range(1, len(self.measurements) + 1)
            raw_measurements = [m['raw'] for m in self.measurements]
            binary_measurements = [m['binary'] for m in self.measurements]

            ax4_twin = ax4.twinx()
            ax4.plot(steps, raw_measurements, 'g-o', label='Raw measurement')
            ax4.axhline(self.sensor_model.threshold, color='orange',
                       linestyle='--', label='Final threshold')
            ax4_twin.plot(steps, binary_measurements, 'b-s', label='Binary measurement', alpha=0.7)

            ax4.set_xlabel('Step')
            ax4.set_ylabel('Raw Measurement', color='g')
            ax4_twin.set_ylabel('Binary (0/1)', color='b')
            ax4.set_title('Sensor Measurements')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nVisualization saved to {save_path}")


def main():
    """Run the ideal scenario test."""
    scenario = IdealScenario()
    scenario.run()
    scenario.visualize()


if __name__ == '__main__':
    main()
