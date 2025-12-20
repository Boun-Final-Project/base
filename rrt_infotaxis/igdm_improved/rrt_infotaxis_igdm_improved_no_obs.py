"""
Standalone RRT-Infotaxis with IGDM (Indoor Gaussian Dispersion Model) with Exploration Penalty (No Obstacles).

Implements the complete measure-plan-move loop from the paper with an exploration penalty
to discourage revisiting recent areas.

Algorithm:
1. MEASURE: Take sensor measurement, update threshold, update particle filter
2. PLAN: Build RRT, evaluate paths with entropy gain vs travel cost, apply exploration penalty
3. MOVE: Navigate to next position
4. Check: If estimation converged, stop

Exploration Penalty:
- Divides information gain by 4 for nodes within 1m radius of positions visited 1-2 steps ago
- Encourages exploration of new areas rather than revisiting recent locations

- 10x6 meter empty room (no obstacles)
- Source at (2, 3) with Q=1.0
- Robot starts at (9, 3)
- IGDM uses Euclidean distance (no obstacle awareness)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from igdm_model import IGDMModel
from sensor_model import BinarySensorModel
from particle_filter import ParticleFilter
from occupancy_grid import OccupancyGrid
from rrt import RRTInfotaxis
from visualizer import StepVisualizer


class RRTInfotaxisIGDM:
    """Standalone RRT-Infotaxis with IGDM."""

    def __init__(self, sigma_m=1.0):
        """
        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter for IGDM model
        """
        self.room_width = 10.0
        self.room_height = 6.0
        self.resolution = 0.25

        self.true_source = (2.0, 3.0)
        self.true_Q = 1.0
        self.robot_start = (9.0, 3.0)

        self.sigma_m = sigma_m
        self.max_steps = 50

        # Algorithm parameters
        self.sigma_threshold = 0.3  # Standard deviation threshold for particles (kept for calculation)
        self.d_success_thr = 0.5   # Success distance from true source location (meters)

        self.grid = OccupancyGrid(self.room_width, self.room_height, self.resolution)

        # Use faster dispersion rate so gas spreads noticeably over the time horizon
        # This ensures measurements change significantly at same locations, providing
        # information for the particle filter to converge
        # No obstacles: occupancy_grid is passed but empty
        self.igdm = IGDMModel(sigma_m=sigma_m, occupancy_grid=self.grid, dispersion_rate=0.05)
        self.sensor = BinarySensorModel()

        self.particle_filter = ParticleFilter(
            num_particles=200,  # Reduced for faster computation
            search_bounds={'x': (0, self.room_width), 'y': (0, self.room_height), 'Q': (0, 2.0)},
            binary_sensor_model=self.sensor,
            dispersion_model=self.igdm
        )

        self.rrt = RRTInfotaxis(self.grid, N_tn=30, R_range=8, delta=1.0, max_depth=2,
                      discount_factor=0.8, positive_weight=0.5)

        self.robot_pos = self.robot_start
        self.trajectory = [self.robot_pos]
        self.trajectory_with_steps = [(self.robot_start, 0)]  # Track (position, step) pairs for penalty
        self.measurements = []
        self.estimates = []
        self.sensor_initialized = False
        self.search_complete = False
        self.current_step = 0  # Track current time step for time-dependent gas model

        # Visualization
        self.visualizer = StepVisualizer(output_dir="igdm_improved_no_obs_steps", igdm_model=self.igdm)

    def get_measurement(self, position):
        """Get sensor measurement at position, accounting for time-dependent gas dispersion.

        Parameters:
        -----------
        position : tuple
            (x, y) measurement position

        Returns:
        --------
        measurement : float
            Noisy concentration measurement
        """
        true_conc = self.igdm.compute_concentration(position, self.true_source, self.true_Q,
                                                     time_step=self.current_step)
        sigma = self.sensor.get_std(true_conc)
        noisy = true_conc + np.random.normal(0, sigma)
        return max(0, noisy)

    def is_estimation_converged(self):
        """Check if estimation has converged (sigma < threshold).

        Returns:
        --------
        converged : bool
            True if estimation has converged
        """
        _, stds = self.particle_filter.get_estimate()
        sigma_p = max(stds['x'], stds['y'])
        return sigma_p < self.sigma_threshold

    def take_step(self, step_num):
        """Execute one measure-plan-move cycle (Algorithm 1 from paper).

        Parameters:
        -----------
        step_num : int
            Current step number

        Returns:
        --------
        should_continue : bool
            False if search is complete, True otherwise
        """
        self.current_step = step_num  # Update current time step for time-dependent gas model

        print(f"\n--- Step {step_num} ---", flush=True)
        print(f"Robot at: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})", flush=True)

        # ==== MEASURE PHASE ====
        measurement = self.get_measurement(self.robot_pos)

        # Initialize on first measurement (skip initialization step)
        if not self.sensor_initialized:
            self.sensor.initialize_threshold(measurement)
            self.sensor_initialized = True
            print(f"[MEASURE] Initialized (measurement: {measurement:.6f})")
            return True

        print(f"[MEASURE] Continuous concentration: {measurement:.6f}")

        # Update threshold (Eq. 27): only increases if measurement > current threshold
        self.sensor.update_threshold(measurement)

        # Convert to binary measurement
        binary_measurement = self.sensor.get_binary_measurement(measurement)
        print(f"[MEASURE] Binary measurement: {binary_measurement}")

        # Update particle filter with BINARY measurement
        self.particle_filter.update(binary_measurement, self.robot_pos, time_step=step_num)

        # Debug: Check effective sample size and weight variance
        N_eff = self.particle_filter._effective_sample_size()
        weight_var = np.var(self.particle_filter.weights)
        print(f"[UPDATE] N_eff={N_eff:.1f}/{self.particle_filter.N}, weight_var={weight_var:.6f}")

        # Get estimate
        mean, std = self.particle_filter.get_estimate()
        print(f"[ESTIMATE] x={mean['x']:.2f}±{std['x']:.2f}, y={mean['y']:.2f}±{std['y']:.2f}, Q={mean['Q']:.2f}±{std['Q']:.2f}")

        self.measurements.append({'pos': self.robot_pos, 'raw': measurement})
        self.estimates.append((mean, std))

        # ==== CHECK CONVERGENCE (early check before planning) ====
        sigma_p = max(std['x'], std['y'])
        print(f"[CONVERGE] sigma_p = {sigma_p:.3f}, sigma_t = {self.sigma_threshold:.3f}")

        # Check if robot has reached true source
        dist_to_true = np.sqrt((self.robot_pos[0] - self.true_source[0])**2 + (self.robot_pos[1] - self.true_source[1])**2)
        print(f"[DISTANCE] Robot to true source: {dist_to_true:.3f}m (threshold: {self.d_success_thr:.3f}m)")

        if dist_to_true < self.d_success_thr:
            print(f"\n✓✓✓ ROBOT REACHED TRUE SOURCE! ✓✓✓")
            print(f"  True source: {self.true_source}")
            print(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            print(f"  Robot position: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})")
            print(f"  Localization error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

            # Save final visualization
            self.visualizer.save_step(
                robot_pos=self.robot_pos,
                trajectory=self.trajectory,
                est_source=(mean['x'], mean['y']),
                est_std=(std['x'], std['y']),
                true_source=self.true_source,
                step_num=step_num,
                sigma_p=sigma_p,
                current_step=step_num,
                particle_filter=self.particle_filter,
                distance_to_true=dist_to_true,
                d_success_thr=self.d_success_thr,
                occupancy_grid=self.grid
            )

            self.search_complete = True
            return False

        if self.is_estimation_converged():
            print(f"\n✓✓✓ ESTIMATION CONVERGED! ✓✓✓")
            print(f"  True source: {self.true_source}")
            print(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            print(f"  Error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

            # Save final converged visualization
            self.visualizer.save_step(
                robot_pos=self.robot_pos,
                trajectory=self.trajectory,
                est_source=(mean['x'], mean['y']),
                est_std=(std['x'], std['y']),
                true_source=self.true_source,
                step_num=step_num,
                sigma_p=sigma_p,
                current_step=step_num,
                particle_filter=self.particle_filter,
                distance_to_true=dist_to_true,
                d_success_thr=self.d_success_thr,
                occupancy_grid=self.grid
            )

            self.search_complete = True
            return False

        # ==== SAVE STEP VISUALIZATION ====
        self.visualizer.save_step(
            robot_pos=self.robot_pos,
            trajectory=self.trajectory,
            est_source=(mean['x'], mean['y']),
            est_std=(std['x'], std['y']),
            true_source=self.true_source,
            step_num=step_num,
            sigma_p=sigma_p,
            current_step=step_num,
            particle_filter=self.particle_filter,
            distance_to_true=dist_to_true,
            d_success_thr=self.d_success_thr,
            occupancy_grid=self.grid
        )

        # ==== PLAN PHASE ====
        print(f"[PLAN] Building RRT with exploration penalty...")
        # Update RRT with visited positions and current step for exploration penalty
        self.rrt.visited_positions = self.trajectory_with_steps
        self.rrt.current_step = step_num
        debug_info = self.rrt.get_next_move_debug(self.robot_pos, self.particle_filter)
        next_pos = debug_info['next_position']

        print(f"[PLAN] Best utility: {debug_info['best_utility']:.4f}")
        print(f"[PLAN] Information gain: {debug_info['best_information_gain']:.4f}")
        print(f"[PLAN] Travel cost: {debug_info['best_travel_cost']:.4f}")

        # ==== MOVE PHASE ====
        print(f"[MOVE] Moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
        self.robot_pos = next_pos
        self.trajectory.append(self.robot_pos)
        self.trajectory_with_steps.append((self.robot_pos, step_num))  # Track step for penalty

        return True

    def run(self):
        """Run the full RRT-Infotaxis algorithm."""
        print("=" * 70)
        print("RRT-INFOTAXIS WITH IGDM + EXPLORATION PENALTY (No Obstacles)")
        print("=" * 70)
        print(f"Environment: {self.room_width}m × {self.room_height}m (empty room)")
        print(f"True source: {self.true_source} (Q={self.true_Q})")
        print(f"Robot start: {self.robot_start}")
        print(f"Algorithm parameters:")
        print(f"  - IGDM sigma_m: {self.sigma_m}m (Euclidean distance, no obstacle awareness)")
        print(f"  - Exploration penalty: Divide info gain by 4 for nodes within 1m of positions visited 1-2 steps ago")
        print(f"  - Success distance threshold: {self.d_success_thr}m")
        print(f"  - Max steps: {self.max_steps}")
        print("=" * 70)

        for step in range(1, self.max_steps + 1):
            should_continue = self.take_step(step)
            if not should_continue:
                break

        print(f"\n{'='*70}")
        print(f"Test completed after {len(self.trajectory)-1} steps")
        print(f"{'='*70}")

    def visualize_final(self, filename='rrt_infotaxis_igdm_improved_no_obs_result.png'):
        """Create final summary plot.

        Parameters:
        -----------
        filename : str
            Output filename for final visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Trajectory
        ax1 = axes[0, 0]
        ax1.set_xlim(0, self.room_width)
        ax1.set_ylim(0, self.room_height)
        ax1.set_aspect('equal')
        ax1.set_title('RRT-Infotaxis Trajectory (IGDM + Exploration Penalty, No Obstacles)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        traj = np.array(self.trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=4, label='Path')
        ax1.plot(self.robot_start[0], self.robot_start[1], 'go', markersize=10, label='Start')
        ax1.plot(self.true_source[0], self.true_source[1], 'r*', markersize=10, label='True source')

        if self.estimates:
            final_mean, final_std = self.estimates[-1]
            ax1.plot(final_mean['x'], final_mean['y'], 'X', color='orange', markersize=15, label='Estimated')

        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: IGDM Concentration field
        ax2 = axes[0, 1]
        ax2.set_xlim(0, self.room_width)
        ax2.set_ylim(0, self.room_height)
        ax2.set_aspect('equal')
        ax2.set_title('IGDM Concentration Field', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        x_grid = np.linspace(0, self.room_width, 100)
        y_grid = np.linspace(0, self.room_height, 60)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Use the final step number for time-dependent dispersion visualization
        final_step = len(self.trajectory) - 1 if self.trajectory else 0

        # Compute concentration field
        Z = np.zeros_like(X)
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                Z[i, j] = self.igdm.compute_concentration(
                    (X[i, j], Y[i, j]), self.true_source, self.true_Q, time_step=final_step
                )

        im = ax2.contourf(X, Y, Z, levels=20, cmap='hot_r')
        cbar = plt.colorbar(im, ax=ax2, label='Concentration')
        ax2.plot(self.true_source[0], self.true_source[1], 'r*', markersize=10)

        # Plot 3: Estimation error over time
        ax3 = axes[1, 0]
        if self.estimates:
            steps = range(1, len(self.estimates) + 1)
            errors = [np.sqrt((e[0]['x']-self.true_source[0])**2 + (e[0]['y']-self.true_source[1])**2)
                     for e in self.estimates]
            ax3.plot(steps, errors, 'k-o', linewidth=2, markersize=6)
            ax3.axhline(self.d_success_thr, color='r', linestyle='--', label=f'd_success = {self.d_success_thr}')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Localization Error (m)')
            ax3.set_title('Source Localization Error (Convergence)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        if self.estimates:
            final_mean, final_std = self.estimates[-1]
            final_error = np.sqrt((final_mean['x']-self.true_source[0])**2 + (final_mean['y']-self.true_source[1])**2)
            final_sigma = max(final_std['x'], final_std['y'])

            info_text = "RRT-INFOTAXIS RESULTS\n"
            info_text += "="*40 + "\n\n"
            info_text += f"Steps taken: {len(self.trajectory)-1}\n"
            info_text += f"Converged: {self.search_complete}\n\n"
            info_text += f"True source: ({self.true_source[0]:.2f}, {self.true_source[1]:.2f})\n"
            info_text += f"Estimated:  ({final_mean['x']:.2f}, {final_mean['y']:.2f})\n"
            info_text += f"Error: {final_error:.3f} m\n\n"
            info_text += f"Success distance threshold: {self.d_success_thr:.3f} m\n"

            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nFinal visualization saved to {filename}")


if __name__ == "__main__":
    # Run with default sigma_m=1.0
    infotaxis = RRTInfotaxisIGDM(sigma_m=1.0)
    infotaxis.run()
    infotaxis.visualize_final()
