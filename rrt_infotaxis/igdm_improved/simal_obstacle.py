"""
Standalone RRT-Infotaxis with IGDM - Three Obstacle Scenario (25x25m).
ALGORITHM: RRT with 4-Way Initial Pruning + Geometric Penalty (FIXED PARAMETERS).

Algorithm:
1. MEASURE: Take sensor measurement, update threshold, update particle filter
2. PLAN: Build RRT (Force 4 directional nodes first), apply penalty
   * Uses FIXED N_tn=30 and max_depth=3 for consistent planning.
3. VISUALIZE: Save step with RRT tree overlay
4. MOVE: Navigate to next position
5. Check: If estimation converged, stop
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from igdm_model import IGDMModel
from sensor_model_discrete import DiscreteSensorModel
from particle_filter import ParticleFilter
from occupancy_grid import OccupancyGrid
from rrt import RRTInfotaxis
from visualizer import StepVisualizer

# --- CUSTOM WRAPPER TO FORCE 4-WAY INITIALIZATION ---
class RRTInfotaxis4Way(RRTInfotaxis):
    """
    Extends the standard RRT class to force the first 4 nodes
    to be in the cardinal directions (Up, Down, Left, Right).
    """
    def sprawl(self, start_pos):
        """Grow the RRT from start position with forced 4-way start."""
        from rrt import Node 

        # 1. Clear nodes and create Root
        self.nodes = []
        root = Node(start_pos)
        self.nodes.append(root)

        # 2. Force 4 Cardinal Directions
        directions = [
            (self.delta, 0),   # Right
            (-self.delta, 0),  # Left
            (0, self.delta),   # Up
            (0, -self.delta)   # Down
        ]

        for dx, dy in directions:
            x = start_pos[0] + dx
            y = start_pos[1] + dy
            
            if self.is_collision_free(start_pos, (x, y)):
                new_node = Node((x, y), root)
                self.nodes.append(new_node)
        
        # 3. Fill the rest with Random Sampling
        while len(self.nodes) < self.N_tn:
            r = self.R_range * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x = start_pos[0] + r * np.cos(theta)
            y = start_pos[1] + r * np.sin(theta)

            closest_node = self.get_closest_node((x, y))
            dist = np.linalg.norm(np.array((x, y)) - closest_node.position)

            if dist > self.delta:
                direction = (np.array((x, y)) - closest_node.position) / dist
                new_pos = closest_node.position + direction * self.delta
                x, y = new_pos[0], new_pos[1]

                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)
            else:
                if self.is_collision_free((closest_node.position[0], closest_node.position[1]), (x, y)):
                    new_node = Node((x, y), closest_node)
                    self.nodes.append(new_node)


class RRTInfotaxisIGDM:
    """Standalone RRT-Infotaxis with IGDM and Discrete Sensor."""

    def __init__(self, sigma_m=1.0, robot_start=None):
        # 25x25m Dimensions
        self.room_width = 25.0
        self.room_height = 25.0
        self.resolution = 0.1

        # Source (Left) and Robot (Right)
        self.true_source = (2.0, 6.0)
        self.true_Q = 1.0
        if robot_start:
            self.robot_start = robot_start
        else:
            self.robot_start = (20.0, 8.0)

        self.sigma_m = sigma_m
        self.max_steps = 200

        # Algorithm parameters
        self.sigma_threshold = 0.3
        self.d_success_thr = 1.0
        
        self.grid = OccupancyGrid(self.room_width, self.room_height, self.resolution)

        # --- Three Rectangular Obstacles ---
        # Obstacle 1: x=5.0, y=4.0 to 10.0
        self.grid.add_rectangular_obstacle(
            x_min=4.9, x_max=5.1, 
            y_min=4.0, y_max=10.0, 
            value=1
        )
        
        # Obstacle 2: x=15.0, y=4.0 to 10.0
        self.grid.add_rectangular_obstacle(
            x_min=14.9, x_max=15.1, 
            y_min=4.0, y_max=10.0, 
            value=1
        )

        # Obstacle 3: x=10.0, y=10.0 to 18.0 (Central, Higher)
        self.grid.add_rectangular_obstacle(
            x_min=9.9, x_max=10.1, 
            y_min=10.0, y_max=18.0, 
            value=1
        )
        # --------------------------------------------

        self.igdm = IGDMModel(sigma_m=sigma_m, occupancy_grid=self.grid, dispersion_rate=3.00)
        self.sensor = DiscreteSensorModel()

        self.particle_filter = ParticleFilter(
            num_particles=400, 
            search_bounds={'x': (0, self.room_width), 'y': (0, self.room_height), 'Q': (0, 2.0)},
            binary_sensor_model=self.sensor,
            dispersion_model=self.igdm
        )

        # FIXED PARAMETERS: N_tn=30, Depth=3 (Robust for complex map)
        self.rrt = RRTInfotaxis4Way(self.grid, N_tn=20, R_range=8, delta=1.0, max_depth=2,
                      discount_factor=0.8, positive_weight=0.5)

        self.robot_pos = self.robot_start
        self.trajectory = [self.robot_pos]
        self.trajectory_with_steps = [(self.robot_start, 0)]
        self.measurements = []
        self.estimates = []
        self.sensor_initialized = False
        self.search_complete = False
        self.current_step = 0

        self.visualizer = None

    def log_path_evaluations(self, debug_info, best_idx):
        """Log detailed evaluation results."""
        all_utilities = debug_info.get('all_utilities', [])
        path_metadata = debug_info.get('path_metadata', [])

        print(f"[PLAN] Path Evaluation Details ({len(all_utilities)} paths):")
        
        if 0 <= best_idx < len(all_utilities):
             metadata = path_metadata[best_idx] if best_idx < len(path_metadata) else {}
             
             if metadata.get('penalty_applied'):
                penalty_info = metadata.get('penalty_info', {})
                steps_since = self.current_step - penalty_info.get('visited_step', 0)
                penalty_divisor = 32.0 / (1.1 ** (steps_since - 1))
                print(f"[PLAN] ⚠️  PENALTY APPLIED TO CHOSEN PATH:")
                print(f"[PLAN]    Steps since visit: {steps_since}")
                print(f"[PLAN]    Geometric Divisor: {penalty_divisor:.2f}x")

    def get_measurement(self, position):
        true_conc = self.igdm.compute_concentration(position, self.true_source, self.true_Q,
                                                     time_step=self.current_step)
        sigma = self.sensor.get_std(true_conc)
        noisy = true_conc + np.random.normal(0, sigma)
        return max(0, noisy)

    def is_estimation_converged(self):
        _, stds = self.particle_filter.get_estimate()
        sigma_p = max(stds['x'], stds['y'])
        return sigma_p < self.sigma_threshold

    def take_step(self, step_num):
        """Execute one measure-plan-move cycle."""
        self.current_step = step_num

        print(f"\n--- Step {step_num} ---", flush=True)
        print(f"Robot at: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})", flush=True)

        # ==== MEASURE PHASE ====
        measurement = self.get_measurement(self.robot_pos)

        if not self.sensor_initialized:
            self.sensor.initialize_threshold(measurement)
            self.sensor_initialized = True
            print(f"[MEASURE] Initialized (measurement: {measurement:.6f})")
            return True

        print(f"[MEASURE] Continuous concentration: {measurement:.6f}")
        self.sensor.update_threshold(measurement)

        discrete_measurement = self.sensor.get_discrete_measurement(measurement)
        level_names = ["Very Low (0)", "Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
        print(f"[MEASURE] Discrete measurement: {discrete_measurement} ({level_names[discrete_measurement]})")
        
        self.particle_filter.update(discrete_measurement, self.robot_pos, time_step=step_num)

        mean, std = self.particle_filter.get_estimate()
        print(f"[ESTIMATE] x={mean['x']:.2f}±{std['x']:.2f}, y={mean['y']:.2f}±{std['y']:.2f}")

        self.measurements.append({'pos': self.robot_pos, 'raw': measurement})
        self.estimates.append((mean, std))

        # ==== CHECK CONVERGENCE ====
        sigma_p = max(std['x'], std['y'])
        dist_to_true = np.sqrt((self.robot_pos[0] - self.true_source[0])**2 + (self.robot_pos[1] - self.true_source[1])**2)
        print(f"[DISTANCE] Robot to true source: {dist_to_true:.3f}m")

        if dist_to_true < self.d_success_thr:
            print(f"\n✓✓✓ ROBOT REACHED TRUE SOURCE! ✓✓✓")
            if self.visualizer:
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
                occupancy_grid=self.grid,
                sensor_reading=measurement,
                threshold_bins=self.sensor.level_thresholds,
                digital_value=discrete_measurement,
                rrt_nodes=None,
                rrt_pruned_paths=None
            )
            self.search_complete = True
            return False

        if self.is_estimation_converged():
            print(f"\n✓✓✓ ESTIMATION CONVERGED! ✓✓✓")
            if self.visualizer:
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
                occupancy_grid=self.grid,
                sensor_reading=measurement,
                threshold_bins=self.sensor.level_thresholds,
                digital_value=discrete_measurement,
                rrt_nodes=None,
                rrt_pruned_paths=None
            )
            self.search_complete = True
            return False

        # ==== PLAN PHASE ====
        print(f"[PLAN] Building RRT (Fixed N_tn=30, Depth=3) + Geometric Penalty...")
        
        self.rrt.visited_positions = self.trajectory_with_steps
        self.rrt.current_step = step_num
        
        debug_info = self.rrt.get_next_move_debug(self.robot_pos, self.particle_filter)
        next_pos = debug_info['next_position']
        
        # EXTRACT VISUALIZATION DATA
        rrt_nodes = debug_info.get('rrt_nodes', None)
        rrt_pruned_paths = debug_info.get('rrt_pruned_paths', None)
        
        best_idx = len(debug_info.get('all_utilities', [])) - 1
        if 'all_utilities' in debug_info:
             best_utility = debug_info['best_utility']
             for idx, util in enumerate(debug_info['all_utilities']):
                  if abs(util - best_utility) < 1e-6:
                       best_idx = idx
                       break
        self.log_path_evaluations(debug_info, best_idx)

        # ==== SAVE STEP VISUALIZATION ====
        if self.visualizer:
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
            occupancy_grid=self.grid,
            sensor_reading=measurement,
            threshold_bins=self.sensor.level_thresholds,
            digital_value=discrete_measurement,
            rrt_nodes=rrt_nodes,
            rrt_pruned_paths=rrt_pruned_paths
        )

        # ==== MOVE PHASE ====
        print(f"[MOVE] Moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
        self.robot_pos = next_pos
        self.trajectory.append(self.robot_pos)
        self.trajectory_with_steps.append((self.robot_pos, step_num))

        return True

    def run(self):
        """Run the full RRT-Infotaxis algorithm."""
        print("=" * 70)
        print("RRT-INFOTAXIS (3 OBSTACLES) + FIXED PARAMETERS (NO ADAPTIVE)")
        print("=" * 70)
        print(f"Environment: 25m x 25m")
        print(f"Obstacle 1: x=5.0, y=4-10")
        print(f"Obstacle 2: x=15.0, y=4-10")
        print(f"Obstacle 3: x=10.0, y=10-18")
        print(f"Source: (2,6) | Robot: (20,8)")
        print(f"Planning: RRT 4-way Initial | N_tn=30 | Depth=3")
        print("=" * 70)

        for step in range(1, self.max_steps + 1):
            should_continue = self.take_step(step)
            if not should_continue:
                break

        print(f"\n{'='*70}")
        print(f"Test completed after {len(self.trajectory)-1} steps")
        print(f"{'='*70}")
        return len(self.trajectory), self.search_complete

    def visualize_final(self, filename='rrt_infotaxis_igdm_discrete_three_obstacles_fixed_result.png'):
        """Create final summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Trajectory
        ax1 = axes[0, 0]
        ax1.set_xlim(0, self.room_width)
        ax1.set_ylim(0, self.room_height)
        ax1.set_aspect('equal')
        ax1.set_title('RRT-Infotaxis Trajectory (Three Obstacles + Fixed Params)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        
        # Plot Three Obstacles
        ax1.add_patch(plt.Rectangle((4.9, 4.0), 0.2, 6.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((14.9, 4.0), 0.2, 6.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((9.9, 10.0), 0.2, 8.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))

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
        y_grid = np.linspace(0, self.room_height, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        final_step = len(self.trajectory) - 1 if self.trajectory else 0

        Z = np.zeros_like(X)
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                Z[i, j] = self.igdm.compute_concentration(
                    (X[i, j], Y[i, j]), self.true_source, self.true_Q, time_step=final_step
                )

        im = ax2.contourf(X, Y, Z, levels=20, cmap='hot_r')
        cbar = plt.colorbar(im, ax=ax2, label='Concentration')
        # Plot Three Obstacles
        ax2.add_patch(plt.Rectangle((4.9, 4.0), 0.2, 6.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax2.add_patch(plt.Rectangle((14.9, 4.0), 0.2, 6.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax2.add_patch(plt.Rectangle((9.9, 10.0), 0.2, 8.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
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
            
            info_text = "RRT-INFOTAXIS RESULTS\n"
            info_text += "="*40 + "\n\n"
            info_text += f"Steps taken: {len(self.trajectory)-1}\n"
            info_text += f"Converged: {self.search_complete}\n\n"
            info_text += f"True source: ({self.true_source[0]:.2f}, {self.true_source[1]:.2f})\n"
            info_text += f"Estimated:  ({final_mean['x']:.2f}, {final_mean['y']:.2f})\n"
            info_text += f"Error: {final_error:.3f} m\n\n"
            info_text += f"Parameters:\n"
            info_text += f"N_tn=30, Depth=3\n"

            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nFinal visualization saved to {filename}")


if __name__ == "__main__":
    infotaxis = RRTInfotaxisIGDM(sigma_m=1.0)
    infotaxis.run()
    infotaxis.visualize_final()