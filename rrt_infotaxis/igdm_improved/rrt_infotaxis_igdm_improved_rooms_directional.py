"""
Directional Planner with IGDM (Indoor Gaussian Dispersion Model) with Multiple Rooms.
Uses 5-level DISCRETE SENSOR and simple 4-directional planning (forward, backward, left, right).

Implements the complete measure-plan-move loop with exploration penalty
to discourage revisiting recent areas.

Algorithm:
1. MEASURE: Take sensor measurement, update threshold, update particle filter
2. PLAN: Evaluate 4 linear paths (3 steps each in cardinal direction), apply exploration penalty
3. MOVE: Navigate to next position
4. Check: If estimation converged, stop

Planning Strategy:
- 4 paths: Forward (3 steps), Backward (3 steps), Left (3 steps), Right (3 steps)
- Each step has random distance between 0.5 and 1.0 meters
- No branching: each path is a straight line in one direction
- Total 4 paths evaluated per step

Exploration Penalty (time-dependent):
- 1 step since visit: Divide info gain by 2^5 = 32
- 2 steps since visit: Divide info gain by 2^4 = 16
- 3 steps since visit: Divide info gain by 2^3 = 8
- 4 steps since visit: Divide info gain by 2^2 = 4
- 5 steps since visit: Divide info gain by 2^1 = 2
- For nodes within 1m radius of visited positions
- Encourages exploration of new areas rather than revisiting recent locations

Building Layout:
- 25x25 meter map with 3 fully enclosed rooms connected by a central hallway
- Room 1 (top-left): x: 0-10, y: 15-25 (right wall at x=10 with door at y: 19-21)
- Room 2 (bottom-left): x: 0-10, y: 0-10 (right wall at x=10 with door at y: 4-6)
- Room 3 (top-right): x: 15-25, y: 15-25 (left wall at x=15 with door at y: 19-21)
- Central hallway: x: 10-15, y: 0-25 connecting the rooms
- Horizontal walls: (0,15)-(10,15), (0,10)-(10,10), (15,15)-(25,15) close the rooms
- Gas source at (5.0, 20.0) inside Room 1 with Q=1.0
- Robot starts at (12.5, 12.5) in the central hallway
- IGDM uses Dijkstra distance (obstacle-aware)

Discrete Sensor Levels:
- Level 0: Very Low (C < threshold[0])
- Level 1: Low (threshold[0] <= C < threshold[1])
- Level 2: Medium (threshold[1] <= C < threshold[2])
- Level 3: High (threshold[2] <= C < threshold[3])
- Level 4: Very High (C >= threshold[3])

This provides ~2.3 bits of information per measurement vs 1 bit for binary.

Utility Function:
- utility = J1 * 0.5 - J2 * 0.5
- J1 (information gain) and J2 (travel cost) contribute equally
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Suppress matplotlib font warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from igdm_model import IGDMModel
from sensor_model_discrete import DiscreteSensorModel
from particle_filter import ParticleFilter
from occupancy_grid import OccupancyGrid
from directional_planner import DirectionalPlanner
from visualizer import StepVisualizer


# Setup logging
def setup_logging(log_file):
    """Setup logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class DirectionalPlannerIGDMRoomsDiscrete:
    """Directional Planner with IGDM and 5-level discrete sensor for multi-room building."""

    def __init__(self, sigma_m=1.0, logger=None):
        """
        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter for IGDM model
        """
        self.logger = logger or logging.getLogger()
        self.room_width = 25.0
        self.room_height = 25.0
        self.resolution = 0.1

        self.true_source = (5.0, 20.0)
        self.true_Q = 1.0
        self.robot_start = (12.5, 12.5)

        self.sigma_m = sigma_m
        self.max_steps = 150

        # Algorithm parameters
        self.sigma_threshold = 0.3  # Standard deviation threshold for particles (kept for calculation)
        self.d_success_thr = 0.5   # Success distance from true source location (meters)

        self.logger = logger or logging.getLogger()
        self.grid = OccupancyGrid(self.room_width, self.room_height, self.resolution)

        # Build room layout with walls and doors
        # Vertical walls
        # Room 1 (top-left): x: 0-10, y: 15-25
        # Right wall at x=10 with door opening at y: 19-21
        self.grid.add_rectangular_obstacle(x_min=9.9, x_max=10.1, y_min=15.0, y_max=19.0, value=1)
        self.grid.add_rectangular_obstacle(x_min=9.9, x_max=10.1, y_min=21.0, y_max=25.0, value=1)

        # Room 2 (bottom-left): x: 0-10, y: 0-10
        # Right wall at x=10 with door opening at y: 4-6
        self.grid.add_rectangular_obstacle(x_min=9.9, x_max=10.1, y_min=0.0, y_max=4.0, value=1)
        self.grid.add_rectangular_obstacle(x_min=9.9, x_max=10.1, y_min=6.0, y_max=10.0, value=1)

        # Room 3 (top-right): x: 15-25, y: 15-25
        # Left wall at x=15 with door opening at y: 19-21
        self.grid.add_rectangular_obstacle(x_min=14.9, x_max=15.1, y_min=15.0, y_max=19.0, value=1)
        self.grid.add_rectangular_obstacle(x_min=14.9, x_max=15.1, y_min=21.0, y_max=25.0, value=1)

        # Horizontal walls to close rooms
        # Wall from (0, 15) to (10, 15) - separates Room 1 and Room 2
        self.grid.add_rectangular_obstacle(x_min=0.0, x_max=10.0, y_min=14.9, y_max=15.1, value=1)

        # Wall from (0, 10) to (10, 10) - closes bottom of Room 2
        self.grid.add_rectangular_obstacle(x_min=0.0, x_max=10.0, y_min=9.9, y_max=10.1, value=1)

        # Wall from (15, 15) to (25, 15) - closes bottom of Room 3
        self.grid.add_rectangular_obstacle(x_min=15.0, x_max=25.0, y_min=14.9, y_max=15.1, value=1)

        # Use faster dispersion rate so gas spreads noticeably over the time horizon
        self.igdm = IGDMModel(sigma_m=sigma_m, occupancy_grid=self.grid, dispersion_rate=3.00)
        self.sensor = DiscreteSensorModel()

        self.particle_filter = ParticleFilter(
            num_particles=400,
            search_bounds={'x': (0, self.room_width), 'y': (0, self.room_height), 'Q': (0, 2.0)},
            binary_sensor_model=self.sensor,
            dispersion_model=self.igdm
        )

        self.planner = DirectionalPlanner(
            self.grid, depth=3, delta_min=0.5, delta_max=1.0,
            discount_factor=0.8, positive_weight=0.60, penalty_radius=0.48
        )

        self.robot_pos = self.robot_start
        self.trajectory = [self.robot_pos]
        self.trajectory_with_steps = [(self.robot_start, 0)]  # Track (position, step) pairs for penalty
        self.measurements = []
        self.estimates = []
        self.sensor_initialized = False
        self.search_complete = False
        self.current_step = 0  # Track current time step for time-dependent gas model

        # Visualization
        # Visualization - save to week-9
        viz_dir = Path("/home/hdd/akademia/cmpe/final-project/week-9/igdm_improved_rooms_directional_steps")
        self.visualizer = StepVisualizer(output_dir=str(viz_dir), igdm_model=self.igdm)

    def log(self, message, flush=True):
        """Log message to both file and console."""
        self.logger.info(message)
        if flush:
            for handler in self.logger.handlers:
                handler.flush()


    def log_all_path_evaluations(self, debug_info, best_idx):
        """Log detailed evaluation results for all paths."""
        all_utilities = debug_info.get("all_utilities", [])
        all_J1 = debug_info.get("all_information_gains_normalized", [])
        all_J2 = debug_info.get("all_travel_costs_normalized", [])
        path_metadata = debug_info.get("path_metadata", [])

        self.log(f"[PLAN] ═══════════════════════════════════════════════════════════════" )
        self.log(f"[PLAN] PATH EVALUATION DETAILS (Total: {len(all_utilities)} paths)")
        self.log(f"[PLAN] ═══════════════════════════════════════════════════════════════")

        direction_names = {0: "Forward", 1: "Backward", 2: "Left", 3: "Right"}
        for idx in range(len(all_utilities)):
            utility = all_utilities[idx]
            J1_norm = all_J1[idx]
            J2_norm = all_J2[idx]
            direction = direction_names.get(idx, f"Path {idx+1}")
            is_best = " ◄── SELECTED" if idx == best_idx else ""
            penalty_mark = " [PENALTY]" if path_metadata[idx].get("penalty_applied") else ""
            self.log(f"[PLAN] {direction:10s}: Utility={utility:8.4f} | J1(IG)={J1_norm:.4f} | J2(Cost)={J2_norm:.4f}{penalty_mark}{is_best}")
        self.log(f"[PLAN] ═══════════════════════════════════════════════════════════════")

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
        """Execute one measure-plan-move cycle.

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

        self.log(f"\n--- Step {step_num} ---")
        self.log(f"Robot at: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})")

        # ==== MEASURE PHASE ====
        measurement = self.get_measurement(self.robot_pos)

        # Initialize on first measurement
        if not self.sensor_initialized:
            self.sensor.initialize_threshold(measurement)
            self.sensor_initialized = True
            self.log(f"[MEASURE] Initialized (measurement: {measurement:.6f})")
            return True

        self.log(f"[MEASURE] Continuous concentration: {measurement:.6f}")

        # Update threshold
        self.sensor.update_threshold(measurement)

        # Convert to discrete measurement
        discrete_measurement = self.sensor.get_discrete_measurement(measurement)
        level_names = ["Very Low (0)", "Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
        self.log(f"[MEASURE] Discrete measurement: {discrete_measurement} ({level_names[discrete_measurement]})")
        self.log(f"[MEASURE] Level thresholds: {[f'{t:.4f}' for t in self.sensor.level_thresholds]}")

        # Update particle filter
        self.particle_filter.update(discrete_measurement, self.robot_pos, time_step=step_num)

        # Debug info
        N_eff = self.particle_filter._effective_sample_size()
        weight_var = np.var(self.particle_filter.weights)
        self.log(f"[UPDATE] N_eff={N_eff:.1f}/{self.particle_filter.N}, weight_var={weight_var:.6f}")

        # Get estimate
        mean, std = self.particle_filter.get_estimate()
        self.log(f"[ESTIMATE] x={mean['x']:.2f}±{std['x']:.2f}, y={mean['y']:.2f}±{std['y']:.2f}, Q={mean['Q']:.2f}±{std['Q']:.2f}")

        self.measurements.append({'pos': self.robot_pos, 'raw': measurement})
        self.estimates.append((mean, std))

        # ==== CHECK CONVERGENCE ====
        sigma_p = max(std['x'], std['y'])
        self.log(f"[CONVERGE] sigma_p = {sigma_p:.3f}, sigma_t = {self.sigma_threshold:.3f}")

        dist_to_true = np.sqrt((self.robot_pos[0] - self.true_source[0])**2 + (self.robot_pos[1] - self.true_source[1])**2)
        self.log(f"[DISTANCE] Robot to true source: {dist_to_true:.3f}m (threshold: {self.d_success_thr:.3f}m)")

        if dist_to_true < self.d_success_thr:
            self.log(f"\n✓✓✓ ROBOT REACHED TRUE SOURCE! ✓✓✓")
            self.log(f"  True source: {self.true_source}")
            self.log(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            self.log(f"  Robot position: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})")
            self.log(f"  Localization error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

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
                rrt_nodes=None,
                rrt_pruned_paths=None
            )

            self.search_complete = True
            return False

        if self.is_estimation_converged():
            self.log(f"\n✓✓✓ ESTIMATION CONVERGED! ✓✓✓")
            self.log(f"  True source: {self.true_source}")
            self.log(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            self.log(f"  Error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

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
                rrt_nodes=None,
                rrt_pruned_paths=None
            )

            self.search_complete = True
            return False

        # ==== PLAN PHASE ====
        self.log(f"[PLAN] Evaluating 4 directional paths with exploration penalty...")
        # Update planner with visited positions
        self.planner.visited_positions = self.trajectory_with_steps
        self.planner.current_step = step_num

        # Show recent visited positions
        self.log(f"[PLAN] Recent visited positions (for penalty window):")
        for pos, step in self.trajectory_with_steps[-5:]:
            steps_ago = step_num - step
            if steps_ago >= 0 and steps_ago <= 5:
                self.log(f"[PLAN]   Step {step}: ({pos[0]:.2f}, {pos[1]:.2f}) - {steps_ago} step{'s' if steps_ago != 1 else ''} ago")

        debug_info = self.planner.get_next_move_debug(self.robot_pos, self.particle_filter)
        next_pos = debug_info['next_position']
        rrt_nodes = debug_info.get('rrt_nodes', None)
        rrt_pruned_paths = debug_info.get('rrt_pruned_paths', None)
        best_idx = len(debug_info.get('all_utilities', [])) - 1

        # Find the best index
        if 'all_utilities' in debug_info:
            all_utilities = debug_info['all_utilities']
            best_utility = debug_info['best_utility']
            for idx, util in enumerate(all_utilities):
                if abs(util - best_utility) < 1e-6:
                    best_idx = idx
                    break

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
            occupancy_grid=self.grid,
            rrt_nodes=rrt_nodes,
            rrt_pruned_paths=rrt_pruned_paths
        )

        # Log all path evaluations with details
        self.log_all_path_evaluations(debug_info, best_idx)

        self.log(f"[PLAN] Best utility: {debug_info['best_utility']:.4f}")
        self.log(f"[PLAN] J1 (Information gain):")
        self.log(f"[PLAN]   - Original (before penalty): {debug_info['best_information_gain_original']:.4f}")
        self.log(f"[PLAN]   - With penalty applied: {debug_info['best_information_gain_penalized']:.4f}")
        self.log(f"[PLAN]   - Normalized [0-1]: {debug_info['best_information_gain']:.4f}")
        self.log(f"[PLAN]   - Range in all paths: [{debug_info['norm_info_gain_range'][0]:.4f}, {debug_info['norm_info_gain_range'][1]:.4f}]")
        self.log(f"[PLAN] J2 (Travel cost):")
        self.log(f"[PLAN]   - Original: {debug_info['best_travel_cost_original']:.4f}")
        self.log(f"[PLAN]   - Normalized [0-1]: {debug_info['best_travel_cost']:.4f}")
        self.log(f"[PLAN]   - Range in all paths: [{debug_info['norm_travel_cost_range'][0]:.4f}, {debug_info['norm_travel_cost_range'][1]:.4f}]")
        self.log(f"[PLAN] Paths analyzed: {debug_info['total_paths']} total | {debug_info['paths_with_penalties']} with penalties")

        # Show penalty information
        if debug_info['best_penalty_applied']:
            original = debug_info['best_information_gain_original']
            penalized = debug_info['best_information_gain_penalized']
            normalized = debug_info['best_information_gain']
            penalty_info = debug_info['best_penalty_info']
            steps_since = self.current_step - penalty_info['visited_step']
            penalty_exponent = 6 - steps_since
            penalty_divisor = 2 ** penalty_exponent
            visited_pos = penalty_info['visited_pos']
            visited_step = penalty_info['visited_step']
            distance = penalty_info['distance']

            if original >= 0:
                penalty_reduction = (1 - debug_info['best_penalty_factor']) * 100
                self.log(f"[PLAN] ⚠️  PENALTY APPLIED (visited {steps_since} step{'s' if steps_since != 1 else ''} ago):")
                self.log(f"[PLAN]    Penalized node: ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
                self.log(f"[PLAN]    Visited position: ({visited_pos[0]:.2f}, {visited_pos[1]:.2f}) at step {visited_step}")
                self.log(f"[PLAN]    Distance to visited node: {distance:.3f}m (threshold: 1.0m)")
                self.log(f"[PLAN]    Original J1: {original:.4f} → Penalized: {penalized:.4f} → Normalized: {normalized:.4f}")
                self.log(f"[PLAN]    Information gain reduced by {penalty_reduction:.1f}% (2^{penalty_exponent} = {penalty_divisor}x divisor)")
            else:
                penalty_factor_inverse = 1 / debug_info['best_penalty_factor']
                penalty_increase = (penalty_factor_inverse - 1) * 100
                self.log(f"[PLAN] ⚠️  PENALTY APPLIED (visited {steps_since} step{'s' if steps_since != 1 else ''} ago):")
                self.log(f"[PLAN]    Penalized node: ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
                self.log(f"[PLAN]    Visited position: ({visited_pos[0]:.2f}, {visited_pos[1]:.2f}) at step {visited_step}")
                self.log(f"[PLAN]    Distance to visited node: {distance:.3f}m (threshold: 1.0m)")
                self.log(f"[PLAN]    Original J1: {original:.4f} → Penalized: {penalized:.4f} → Normalized: {normalized:.4f}")
                self.log(f"[PLAN]    Information gain made worse by {penalty_increase:.1f}% (2^{penalty_exponent} = {penalty_divisor}x amplification)")
        else:
            self.log(f"[PLAN] No penalty applied (exploring new areas)")

        # ==== MOVE PHASE ====
        self.log(f"[MOVE] Moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
        self.robot_pos = next_pos
        self.trajectory.append(self.robot_pos)
        self.trajectory_with_steps.append((self.robot_pos, step_num))

        return True

    def run(self):
        """Run the full Directional Planner with discrete sensor."""
        self.log("=" * 70)
        self.log("DIRECTIONAL PLANNER WITH IGDM + DISCRETE SENSOR + EXPLORATION PENALTY")
        self.log("(Multi-Room Building)")
        self.log("=" * 70)
        self.log(f"Environment: {self.room_width}m × {self.room_height}m (with multiple fully enclosed rooms)")
        self.log(f"Room 1 (top-left): x∈[0,10], y∈[15,25] with door at y∈[19,21]")
        self.log(f"Room 2 (bottom-left): x∈[0,10], y∈[0,10] with door at y∈[4,6]")
        self.log(f"Room 3 (top-right): x∈[15,25], y∈[15,25] with door at y∈[19,21]")
        self.log(f"Central hallway: x∈[10,15], y∈[0,25]")
        self.log(f"Horizontal walls: (0,15)-(10,15), (0,10)-(10,10), (15,15)-(25,15)")
        self.log(f"True source: {self.true_source} (Q={self.true_Q}) in Room 1")
        self.log(f"Robot start: {self.robot_start} in central hallway")
        self.log(f"Algorithm parameters:")
        self.log(f"  - IGDM sigma_m: {self.sigma_m}m (Dijkstra distance, obstacle-aware)")
        self.log(f"  - Sensor: 5-level DISCRETE (provides ~2.3 bits/measurement vs 1 bit for binary)")
        self.log(f"  - Planning: 4 linear paths (Forward, Backward, Left, Right) × 3 steps each")
        self.log(f"  - Step size: Random between 0.5-1.0m per step")
        self.log(f"  - Utility weights: J1 (information gain) = 0.5, J2 (travel cost) = 0.5 (normalized, equal contribution)")
        self.log(f"  - Exploration penalty (time-dependent): 2^(6-steps_since_visit) divisor for visited positions")
        self.log(f"    * 1 step ago: ÷32  | 2 steps: ÷16  | 3 steps: ÷8  | 4 steps: ÷4  | 5 steps: ÷2")
        self.log(f"  - Success distance threshold: {self.d_success_thr}m")
        self.log(f"  - Max steps: {self.max_steps}")
        self.log("=" * 70)

        for step in range(1, self.max_steps + 1):
            should_continue = self.take_step(step)
            if not should_continue:
                break

        self.log(f"\n{'='*70}")
        self.log(f"Test completed after {len(self.trajectory)-1} steps")
        self.log(f"{'='*70}")

    def visualize_final(self, filename='directional_planner_igdm_improved_rooms_discrete_result.png'):
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
        ax1.set_title('Directional Planner Trajectory (IGDM + Discrete Sensor + Exploration Penalty)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        # Plot room walls
        ax1.add_patch(plt.Rectangle((9.9, 15.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((9.9, 21.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((9.9, 0.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((9.9, 6.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((14.9, 15.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((14.9, 21.0), 0.2, 4.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((0.0, 14.9), 10.0, 0.2, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((0.0, 9.9), 10.0, 0.2, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax1.add_patch(plt.Rectangle((15.0, 14.9), 10.0, 0.2, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))

        traj = np.array(self.trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=4, label='Path')
        ax1.plot(self.robot_start[0], self.robot_start[1], 'go', markersize=10, label='Start')
        ax1.plot(self.true_source[0], self.true_source[1], 'r*', markersize=10, label='True source')

        if len(self.estimates) > 0:
            mean, std = self.estimates[-1]
            ax1.plot(mean['x'], mean['y'], 'o', color='orange', markersize=10, label='Est. source')

        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)

        self.log(f"Saved final visualization to {filename}")
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    # Setup logging
    log_dir = Path("/home/hdd/akademia/cmpe/final-project/week-9")
    log_file = log_dir / "rrt_infotaxis_igdm_improved_rooms_directional.log"
    logger = setup_logging(str(log_file))

    planner = DirectionalPlannerIGDMRoomsDiscrete(sigma_m=1.0, logger=logger)
    planner.run()
    planner.visualize_final()
