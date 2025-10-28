"""
Step-by-step visualization for RRT-Infotaxis algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class StepVisualizer:
    """Save step visualizations as PNG frames."""

    def __init__(self, output_dir="igdm_steps", igdm_model=None):
        """
        Parameters:
        -----------
        output_dir : str
            Directory to save step visualizations
        igdm_model : IGDMModel, optional
            IGDM model for concentration field visualization
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self.igdm_model = igdm_model

    def _plot_obstacles(self, ax, occupancy_grid):
        """Helper method to plot obstacles on a subplot.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to plot obstacles on
        occupancy_grid : OccupancyGrid
            The occupancy grid containing obstacle information
        """
        if occupancy_grid is None:
            return

        # Find all occupied cells and draw them
        for gy in range(occupancy_grid.grid_height):
            for gx in range(occupancy_grid.grid_width):
                if occupancy_grid.grid[gy, gx] != 0:
                    # Convert grid coordinates to world coordinates
                    x = gx * occupancy_grid.resolution
                    y = gy * occupancy_grid.resolution
                    # Draw a filled rectangle for the cell
                    rect = plt.Rectangle((x, y), occupancy_grid.resolution, occupancy_grid.resolution,
                                       facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
                    ax.add_patch(rect)

    def save_step(self, robot_pos, trajectory, est_source, est_std, true_source,
                  step_num, sigma_p, current_step=None, particle_filter=None,
                  distance_to_true=None, d_success_thr=None, occupancy_grid=None):
        """Save a step visualization with particle filter visualization.

        Parameters:
        -----------
        robot_pos : tuple
            Current robot position (x, y)
        trajectory : list
            List of previous positions visited
        est_source : tuple
            Estimated source location (x, y)
        est_std : tuple
            Standard deviation of estimate (sigma_x, sigma_y)
        true_source : tuple
            True source location (x, y)
        step_num : int
            Current step number
        sigma_p : float
            Current position estimation uncertainty
        current_step : int, optional
            Current time step for gas dispersion
        particle_filter : ParticleFilter, optional
            Particle filter for visualization
        distance_to_true : float, optional
            Current distance to true source location
        d_success_thr : float, optional
            Success distance threshold
        occupancy_grid : OccupancyGrid, optional
            Occupancy grid for obstacle visualization
        """
        # Create figure with 3 subplots: trajectory, concentration field, and particles
        fig = plt.figure(figsize=(16, 5))

        # Plot 1: Trajectory and estimation
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 6)
        ax1.set_aspect('equal')
        ax1.set_title(f'Step {step_num}: RRT-Infotaxis Trajectory', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        # Plot obstacles first (so they appear behind other elements)
        self._plot_obstacles(ax1, occupancy_grid)

        traj = np.array(trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Robot path')
        ax1.plot(robot_pos[0], robot_pos[1], 'ko', markersize=10, label='Current position')
        ax1.plot(est_source[0], est_source[1], 'o', color='orange', markersize=10, label='Estimated source')
        ax1.plot(true_source[0], true_source[1], 'r*', markersize=10, label='True source')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)

        # Plot 2: IGDM Concentration field (time-dependent)
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 6)
        ax2.set_aspect('equal')

        # Show time-dependent dispersion in title
        if current_step is not None:
            ax2.set_title(f'IGDM Concentration Field (Step {current_step})', fontsize=12, fontweight='bold')
        else:
            ax2.set_title('IGDM Concentration Field', fontsize=12, fontweight='bold')

        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        x_grid = np.linspace(0, 10, 50)
        y_grid = np.linspace(0, 6, 30)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Compute concentration field using actual IGDM model with time-dependent sigma
        if self.igdm_model is not None and current_step is not None:
            Z = np.zeros_like(X)
            for i in range(len(y_grid)):
                for j in range(len(x_grid)):
                    Z[i, j] = self.igdm_model.compute_concentration(
                        (X[i, j], Y[i, j]), true_source, 1.0, time_step=current_step
                    )
        else:
            # Fallback to simplified visualization if IGDM model not available
            Z = np.exp(-(np.sqrt((X-true_source[0])**2 + (Y-true_source[1])**2)**2 / 2.0))

        im = ax2.contourf(X, Y, Z, levels=15, cmap='hot_r')
        cbar = plt.colorbar(im, ax=ax2, label='Concentration')

        # Plot obstacles on concentration field
        self._plot_obstacles(ax2, occupancy_grid)

        ax2.plot(true_source[0], true_source[1], 'r*', markersize=10)

        # Plot 3: Particle filter visualization
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 6)
        ax3.set_aspect('equal')
        ax3.set_title(f'Particle Positions (N={particle_filter.N if particle_filter else 0})',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')

        # Plot obstacles first (so they appear behind particles)
        self._plot_obstacles(ax3, occupancy_grid)

        if particle_filter is not None:
            # Extract particle positions (x, y) and weights
            particles = particle_filter.particles
            weights = particle_filter.weights
            N = particle_filter.N

            # Use fixed colorbar scale: vmin=0, vmax=uniform_weight * scale_factor
            # This makes weight colors consistent across steps
            uniform_weight = 1.0 / N
            vmax = uniform_weight * 5.0  # Scale up uniform weight by 5x for better visibility

            # Plot particles colored by weight (darker = higher weight)
            scatter = ax3.scatter(particles[:, 0], particles[:, 1],
                                s=50, c=weights, cmap='viridis_r',
                                alpha=0.6, edgecolors='black', linewidth=0.5,
                                vmin=0, vmax=vmax)
            cbar = plt.colorbar(scatter, ax=ax3, label='Weight')

            # Overlay true and estimated sources
            ax3.plot(true_source[0], true_source[1], 'r*', markersize=10, label='True source')
            ax3.plot(est_source[0], est_source[1], 'o', color='orange', markersize=10, label='Est. source')
            ax3.legend(loc='upper right', fontsize=9)

        ax3.grid(True, alpha=0.3)

        # Display distance to true source and success threshold
        if distance_to_true is not None and d_success_thr is not None:
            plt.suptitle(f'Distance to true source = {distance_to_true:.3f}m (threshold: {d_success_thr:.3f}m)', fontsize=11, y=1.02)
        else:
            plt.suptitle(f'sigma_p = {sigma_p:.3f}m', fontsize=11, y=1.02)
        plt.tight_layout()

        filename = self.output_dir / f"step_{self.step_count:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        self.step_count += 1
