"""
Standalone RRT-Infotaxis with IGDM (Indoor Gaussian Dispersion Model) on Large Map with Discrete Sensor.
EXTENDED PENALTY VERSION: 20-step penalty window with geometric decay (divide by 1.1 each step).

Implements the complete measure-plan-move loop from the paper with an exploration penalty
to discourage revisiting recent areas. Uses 5-level DISCRETE SENSOR instead of binary sensor.

Algorithm:
1. MEASURE: Take sensor measurement, update threshold, update particle filter
2. PLAN: Build RRT, evaluate paths with entropy gain vs travel cost, apply exploration penalty
3. MOVE: Navigate to next position
4. Check: If estimation converged, stop

Exploration Penalty (EXTENDED time-dependent):
- 1-20 steps since visit: Penalty divisor = 32 / (1.1 ^ (steps_since_visit - 1))
- 1 step: Divide info gain by 32.00
- 2 steps: Divide info gain by 29.09
- 3 steps: Divide info gain by 26.45
- ...
- 20 steps: Divide info gain by 5.10
- 21+ steps: No penalty
- For nodes within 1m radius of visited positions
- Encourages exploration of new areas rather than revisiting recent locations

Large Map Layout:
- 25x25 meter map with a room in top-left corner (x: 0-5, y: 20-25)
- Room walls: Right wall at x=5, Bottom wall at y=20 with 2m door at x: 0-2
- Source at (2.5, 22.5) inside the room with Q=1.0
- Robot starts at (12.0, 13.0) in open area
- IGDM uses Dijkstra distance (obstacle-aware)

Discrete Sensor Levels:
- Level 0: Very Low (C < threshold[0])
- Level 1: Low (threshold[0] <= C < threshold[1])
- Level 2: Medium (threshold[1] <= C < threshold[2])
- Level 3: High (threshold[2] <= C < threshold[3])
- Level 4: Very High (C >= threshold[3])

This provides ~2.3 bits of information per measurement vs 1 bit for binary.
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
from rrt_extended_penalty import RRTInfotaxis
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


class RRTInfotaxisIGDMDiscreteLargeMapExtendedPenalty:
    """Standalone RRT-Infotaxis with IGDM and discrete sensor on large map (Extended Penalty)."""

    def __init__(self, sigma_m=1.0, logger=None):
        """
        Parameters:
        -----------
        sigma_m : float
            Base dispersion parameter for IGDM model
        logger : logging.Logger
            Logger instance for output
        """
        self.logger = logger or logging.getLogger()
        self.room_width = 25.0
        self.room_height = 25.0
        self.resolution = 0.25

        self.true_source = (2.5, 22.5)
        self.true_Q = 1.0
        self.robot_start = (12.0, 13.0)

        self.sigma_m = sigma_m
        self.max_steps = 150

        # Algorithm parameters
        self.sigma_threshold = 0.3  # Standard deviation threshold for particles (kept for calculation)
        self.d_success_thr = 0.5   # Success distance from true source location (meters)

        self.grid = OccupancyGrid(self.room_width, self.room_height, self.resolution)

        # Add room obstacles (walls with door opening)
        # Right wall at x=5 (vertical wall spanning y: 20 to 25)
        self.grid.add_rectangular_obstacle(x_min=4.9, x_max=5.1, y_min=20.0, y_max=25.0, value=1)

        # Bottom wall at y=20 with door opening at x: 0-2 (wall spans x: 2-5)
        self.grid.add_rectangular_obstacle(x_min=2.0, x_max=5.0, y_min=19.9, y_max=20.1, value=1)

        # Use faster dispersion rate so gas spreads noticeably over the time horizon
        # This ensures measurements change significantly at same locations, providing
        # information for the particle filter to converge
        self.igdm = IGDMModel(sigma_m=sigma_m, occupancy_grid=self.grid, dispersion_rate=3.00)
        self.sensor = DiscreteSensorModel()

        self.particle_filter = ParticleFilter(
            num_particles=400,
            search_bounds={'x': (0, self.room_width), 'y': (0, self.room_height), 'Q': (0, 2.0)},
            binary_sensor_model=self.sensor,
            dispersion_model=self.igdm,
            resample_threshold=0.42
        )

        self.rrt = RRTInfotaxis(self.grid, N_tn=20, R_range=8, delta=1.0, max_depth=2,
                      discount_factor=0.8, positive_weight=0.60, penalty_radius=0.50)

        self.robot_pos = self.robot_start
        self.trajectory = [self.robot_pos]
        self.trajectory_with_steps = [(self.robot_start, 0)]  # Track (position, step) pairs for penalty
        self.measurements = []
        self.estimates = []
        self.sensor_initialized = False
        self.search_complete = False
        self.current_step = 0  # Track current time step for time-dependent gas model

        # Visualization - save to results folder
        viz_dir = Path(__file__).parent / "results" / "igdm_improved_large_map_discrete_extended_penalty_steps"
        self.visualizer = StepVisualizer(output_dir=str(viz_dir), igdm_model=self.igdm)

    def log(self, message, flush=True):
        """Log message to both file and console."""
        self.logger.info(message)
        if flush:
            for handler in self.logger.handlers:
                handler.flush()

    def log_all_path_evaluations(self, debug_info, best_idx):
        """Log detailed evaluation results for all paths.

        Parameters:
        -----------
        debug_info : dict
            Debug information from RRT planner
        best_idx : int
            Index of the best path
        """
        all_utilities = debug_info.get('all_utilities', [])
        all_J1 = debug_info.get('all_information_gains_normalized', [])
        all_J2 = debug_info.get('all_travel_costs_normalized', [])
        path_metadata = debug_info.get('path_metadata', [])

        self.log(f"[PLAN] ═══════════════════════════════════════════════════════════════")
        self.log(f"[PLAN] PATH EVALUATION DETAILS (Total: {len(all_utilities)} paths)")
        self.log(f"[PLAN] ═══════════════════════════════════════════════════════════════")

        for idx in range(len(all_utilities)):
            utility = all_utilities[idx]
            J1_norm = all_J1[idx]
            J2_norm = all_J2[idx]
            metadata = path_metadata[idx] if idx < len(path_metadata) else {}

            is_best = " ◄── SELECTED" if idx == best_idx else ""
            penalty_mark = " [PENALTY]" if metadata.get('penalty_applied') else ""

            self.log(f"[PLAN] Path {idx+1:3d}: Utility={utility:8.4f} | J1(IG)={J1_norm:.4f} | J2(Cost)={J2_norm:.4f}{penalty_mark}{is_best}")

            # Log penalty details if applied
            if metadata.get('penalty_applied'):
                penalty_factor = metadata.get('penalty_factor', 1.0)
                I_gain_original = metadata.get('I_gain_original', 0)
                I_gain_penalized = metadata.get('I_gain_penalized', 0)
                penalty_info = metadata.get('penalty_info', {})

                if penalty_info:
                    steps_since = self.current_step - penalty_info.get('visited_step', 0)
                    penalty_divisor = 32.0 / (1.1 ** (steps_since - 1))
                    distance = penalty_info.get('distance', 0)

                    self.log(f"[PLAN]         → Original J1={I_gain_original:.4f} → Penalized={I_gain_penalized:.4f} (divisor: 32/(1.1^{steps_since-1}) = {penalty_divisor:.2f}x)")
                    self.log(f"[PLAN]         → Distance to visited: {distance:.3f}m, {steps_since} steps since visit")

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

        self.log(f"\n--- Step {step_num} ---")
        self.log(f"Robot at: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})")

        # ==== MEASURE PHASE ====
        measurement = self.get_measurement(self.robot_pos)

        # Initialize on first measurement (skip initialization step)
        if not self.sensor_initialized:
            self.sensor.initialize_threshold(measurement)
            self.sensor_initialized = True
            self.log(f"[MEASURE] Initialized (measurement: {measurement:.6f})")
            return True

        self.log(f"[MEASURE] Continuous concentration: {measurement:.6f}")

        # Update threshold (Eq. 27): only increases if measurement > current threshold
        self.sensor.update_threshold(measurement)

        # Convert to discrete measurement (5 levels: 0-4)
        discrete_measurement = self.sensor.get_discrete_measurement(measurement)
        level_names = ["Very Low (0)", "Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
        self.log(f"[MEASURE] Discrete measurement: {discrete_measurement} ({level_names[discrete_measurement]})")
        self.log(f"[MEASURE] Level thresholds: {[f'{t:.4f}' for t in self.sensor.level_thresholds]}")

        # Update particle filter with DISCRETE measurement
        self.particle_filter.update(discrete_measurement, self.robot_pos, time_step=step_num)

        # Debug: Check effective sample size and weight variance
        N_eff = self.particle_filter._effective_sample_size()
        weight_var = np.var(self.particle_filter.weights)
        self.log(f"[UPDATE] N_eff={N_eff:.1f}/{self.particle_filter.N}, weight_var={weight_var:.6f}")

        # Get estimate
        mean, std = self.particle_filter.get_estimate()
        self.log(f"[ESTIMATE] x={mean['x']:.2f}±{std['x']:.2f}, y={mean['y']:.2f}±{std['y']:.2f}, Q={mean['Q']:.2f}±{std['Q']:.2f}")

        self.measurements.append({'pos': self.robot_pos, 'raw': measurement})
        self.estimates.append((mean, std))

        # ==== CHECK CONVERGENCE (early check before planning) ====
        sigma_p = max(std['x'], std['y'])
        self.log(f"[CONVERGE] sigma_p = {sigma_p:.3f}, sigma_t = {self.sigma_threshold:.3f}")

        # Check if robot has reached true source
        dist_to_true = np.sqrt((self.robot_pos[0] - self.true_source[0])**2 + (self.robot_pos[1] - self.true_source[1])**2)
        self.log(f"[DISTANCE] Robot to true source: {dist_to_true:.3f}m (threshold: {self.d_success_thr:.3f}m)")

        if dist_to_true < self.d_success_thr:
            self.log(f"\n✓✓✓ ROBOT REACHED TRUE SOURCE! ✓✓✓")
            self.log(f"  True source: {self.true_source}")
            self.log(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            self.log(f"  Robot position: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f})")
            self.log(f"  Localization error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

            # Save final visualization (no RRT since we return early)
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
                sensor_reading=measurement,
                threshold_bins=self.sensor.level_thresholds,
                digital_value=discrete_measurement,
            penalty_step_count=self.rrt.MAX_PENALTY_STEPS
            )

            self.search_complete = True
            return False

        if self.is_estimation_converged():
            self.log(f"\n✓✓✓ ESTIMATION CONVERGED! ✓✓✓")
            self.log(f"  True source: {self.true_source}")
            self.log(f"  Estimated: ({mean['x']:.2f}, {mean['y']:.2f})")
            self.log(f"  Error: {np.sqrt((mean['x']-self.true_source[0])**2 + (mean['y']-self.true_source[1])**2):.3f}m")

            # Save final converged visualization (no RRT since we return early)
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
                sensor_reading=measurement,
                threshold_bins=self.sensor.level_thresholds,
                digital_value=discrete_measurement,
            penalty_step_count=self.rrt.MAX_PENALTY_STEPS
            )

            self.search_complete = True
            return False

        # ==== PLAN PHASE (before visualization for RRT drawing) ====
        self.log(f"[PLAN] Building RRT with EXTENDED exploration penalty (20-step window, 1.1x decay)...")
        # Update RRT with visited positions and current step for exploration penalty
        self.rrt.visited_positions = self.trajectory_with_steps
        self.rrt.current_step = step_num

        # Debug: Show recent visited positions
        self.log(f"[PLAN] Recent visited positions (for extended 20-step penalty window):")
        for pos, step in self.trajectory_with_steps[-10:]:
            steps_ago = step_num - step
            if steps_ago >= 0:
                self.log(f"[PLAN]   Step {step}: ({pos[0]:.2f}, {pos[1]:.2f}) - {steps_ago} step{'s' if steps_ago != 1 else ''} ago")

        debug_info = self.rrt.get_next_move_debug(self.robot_pos, self.particle_filter)
        next_pos = debug_info['next_position']
        rrt_nodes = debug_info.get('rrt_nodes', None)
        rrt_pruned_paths = debug_info.get('rrt_pruned_paths', None)
        best_idx = len(debug_info.get('all_utilities', [])) - 1  # Default to last if not found

        # Find the best index
        if 'all_utilities' in debug_info:
            all_utilities = debug_info['all_utilities']
            best_utility = debug_info['best_utility']
            for idx, util in enumerate(all_utilities):
                if abs(util - best_utility) < 1e-6:
                    best_idx = idx
                    break

        # ==== SAVE STEP VISUALIZATION (with RRT tree) ====
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
            rrt_pruned_paths=rrt_pruned_paths,
            sensor_reading=measurement,
            threshold_bins=self.sensor.level_thresholds,
            digital_value=discrete_measurement,
            penalty_step_count=self.rrt.MAX_PENALTY_STEPS
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
            penalty_divisor = 32.0 / (1.1 ** (steps_since - 1))
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
                self.log(f"[PLAN]    Information gain reduced by {penalty_reduction:.1f}% (divisor: 32/(1.1^{steps_since-1}) = {penalty_divisor:.2f}x)")
            else:
                penalty_factor_inverse = 1 / debug_info['best_penalty_factor']
                penalty_increase = (penalty_factor_inverse - 1) * 100
                self.log(f"[PLAN] ⚠️  PENALTY APPLIED (visited {steps_since} step{'s' if steps_since != 1 else ''} ago):")
                self.log(f"[PLAN]    Penalized node: ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
                self.log(f"[PLAN]    Visited position: ({visited_pos[0]:.2f}, {visited_pos[1]:.2f}) at step {visited_step}")
                self.log(f"[PLAN]    Distance to visited node: {distance:.3f}m (threshold: 1.0m)")
                self.log(f"[PLAN]    Original J1: {original:.4f} → Penalized: {penalized:.4f} → Normalized: {normalized:.4f}")
                self.log(f"[PLAN]    Information gain made worse by {penalty_increase:.1f}% (divisor: 32/(1.1^{steps_since-1}) = {penalty_divisor:.2f}x)")
        else:
            self.log(f"[PLAN] No penalty applied (exploring new areas)")

        # ==== MOVE PHASE ====
        self.log(f"[MOVE] Moving to ({next_pos[0]:.2f}, {next_pos[1]:.2f})")
        self.robot_pos = next_pos
        self.trajectory.append(self.robot_pos)
        self.trajectory_with_steps.append((self.robot_pos, step_num))  # Track step for penalty

        return True

    def run(self):
        """Run the full RRT-Infotaxis algorithm with discrete sensor on large map."""
        self.log("=" * 70)
        self.log("RRT-INFOTAXIS WITH IGDM + DISCRETE SENSOR + EXTENDED PENALTY")
        self.log("(Large Map - Single Room - 20-step penalty window with 1.1x decay)")
        self.log("=" * 70)
        self.log(f"Environment: {self.room_width}m × {self.room_height}m (large open area with one room)")
        self.log(f"Room (top-left): x∈[0,5], y∈[20,25] with door at x∈[0,2]")
        self.log(f"Room walls: Right wall at x=5, Bottom wall at y=20 with door opening")
        self.log(f"True source: {self.true_source} (Q={self.true_Q}) inside room")
        self.log(f"Robot start: {self.robot_start} in open area")
        self.log(f"Algorithm parameters:")
        self.log(f"  - IGDM sigma_m: {self.sigma_m}m (Dijkstra distance, obstacle-aware)")
        self.log(f"  - Sensor: 5-level DISCRETE (provides ~2.3 bits/measurement vs 1 bit for binary)")
        self.log(f"  - Utility weights: J1 (information gain) = 0.5, J2 (travel cost) = 0.5 (normalized, equal contribution)")
        self.log(f"  - EXTENDED Exploration penalty (time-dependent): 32 / (1.1^(steps_since_visit-1)) divisor for visited positions")
        self.log(f"    * 1 step ago: ÷32.00  | 2 steps: ÷29.09  | 3 steps: ÷26.45  | ... | 20 steps: ÷5.10  | 21+ steps: No penalty")
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

    def visualize_final(self, filename='rrt_infotaxis_igdm_improved_large_map_discrete_extended_penalty_result.png'):
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
        ax1.set_title('RRT-Infotaxis Trajectory (IGDM + Discrete Sensor + Extended Penalty)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        # Plot room walls
        # Right wall at x=5
        ax1.add_patch(plt.Rectangle((4.9, 20.0), 0.2, 5.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        # Bottom wall at y=20 (with door opening at x: 0-2)
        ax1.add_patch(plt.Rectangle((2.0, 19.9), 3.0, 0.2, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))

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
        # Plot room walls on concentration field
        ax2.add_patch(plt.Rectangle((4.9, 20.0), 0.2, 5.0, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
        ax2.add_patch(plt.Rectangle((2.0, 19.9), 3.0, 0.2, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7))
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

            info_text = "RRT-INFOTAXIS RESULTS (LARGE MAP + DISCRETE SENSOR + EXTENDED PENALTY)\n"
            info_text += "="*40 + "\n\n"
            info_text += f"Steps taken: {len(self.trajectory)-1}\n"
            info_text += f"Converged: {self.search_complete}\n\n"
            info_text += f"True source: ({self.true_source[0]:.2f}, {self.true_source[1]:.2f})\n"
            info_text += f"Estimated:  ({final_mean['x']:.2f}, {final_mean['y']:.2f})\n"
            info_text += f"Error: {final_error:.3f} m\n\n"
            info_text += f"Success distance threshold: {self.d_success_thr:.3f} m\n\n"
            info_text += f"Penalty Window: 20 steps\n"
            info_text += f"Decay Rate: 1.1x per step\n"

            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        self.log(f"\nFinal visualization saved to {filename}")


if __name__ == "__main__":
    # Setup logging
    log_dir = Path(__file__).parent / "results"
    log_file = log_dir / "rrt_infotaxis_igdm_improved_large_map_discrete_extended_penalty.log"
    logger = setup_logging(str(log_file))

    # Run with default sigma_m=1.0
    infotaxis = RRTInfotaxisIGDMDiscreteLargeMapExtendedPenalty(sigma_m=1.0, logger=logger)
    infotaxis.run()
    infotaxis.visualize_final()
