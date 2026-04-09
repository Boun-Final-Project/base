"""
Step-by-step visualization for the RL gas source environment.

Mirrors rrt_infotaxis/igdm_improved/visualizer.py, but trimmed to a 2-panel
layout (trajectory + IGDM concentration field) since the RL setting has no
RRT tree or particle filter to display.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class StepVisualizer:
    """Save step visualizations as PNG frames."""

    def __init__(self, output_dir="rl_steps", igdm_model=None):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save step visualizations.
        igdm_model : IGDMModel, optional
            IGDM model for concentration field visualization.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self.igdm_model = igdm_model

    def _plot_obstacles(self, ax, occupancy_grid):
        """Draw occupied cells from an OccupancyGrid onto an axis."""
        if occupancy_grid is None:
            return

        for gy in range(occupancy_grid.grid_height):
            for gx in range(occupancy_grid.grid_width):
                if occupancy_grid.grid[gy, gx] != 0:
                    x = gx * occupancy_grid.resolution
                    y = gy * occupancy_grid.resolution
                    rect = plt.Rectangle(
                        (x, y), occupancy_grid.resolution, occupancy_grid.resolution,
                        facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7,
                    )
                    ax.add_patch(rect)

    def save_step(self, robot_pos, trajectory, true_source, step_num,
                  current_step=None, occupancy_grid=None,
                  distance_to_true=None, d_success_thr=None,
                  sensor_reading=None, sensor_threshold=None, digital_value=None,
                  wind_offset=None, eff_source=None):
        """Save a 2-panel step visualization (trajectory + IGDM concentration).

        Parameters
        ----------
        robot_pos : tuple
            Current robot position (x, y).
        trajectory : list
            List of previous positions visited.
        true_source : tuple
            True source location (x, y).
        step_num : int
            Step number used in the title.
        current_step : int, optional
            Current time step for time-dependent dispersion.
        occupancy_grid : OccupancyGrid, optional
            Occupancy grid for obstacle visualization.
        distance_to_true : float, optional
            Distance from robot to true source.
        d_success_thr : float, optional
            Success distance threshold.
        sensor_reading : float, optional
            Raw (continuous) sensor reading.
        sensor_threshold : float, optional
            Current sensor detection threshold.
        digital_value : int, optional
            Discretized/binary sensor reading.
        wind_offset : np.ndarray, optional
            Downwind offset passed through to IGDM.compute_concentration.
        """
        if occupancy_grid is not None:
            map_width = occupancy_grid.width
            map_height = occupancy_grid.height
        else:
            map_width = 10.0
            map_height = 6.0

        fig = plt.figure(figsize=(11, 5))

        # Plot 1: Trajectory
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_xlim(0, map_width)
        ax1.set_ylim(0, map_height)
        ax1.set_aspect('equal')
        ax1.set_title(f'Step {step_num}: Trajectory', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        self._plot_obstacles(ax1, occupancy_grid)

        traj = np.array(trajectory)
        for i in range(len(traj) - 1):
            ax1.plot([traj[i, 0], traj[i + 1, 0]],
                     [traj[i, 1], traj[i + 1, 1]],
                     color='blue', linewidth=2, alpha=0.7)
        for i in range(len(traj)):
            ax1.plot(traj[i, 0], traj[i, 1], 'o', color='blue',
                     markersize=4, alpha=0.7)

        ax1.plot(robot_pos[0], robot_pos[1], 'ko', markersize=10)
        ax1.plot(true_source[0], true_source[1], 'r*', markersize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: IGDM concentration field
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_xlim(0, map_width)
        ax2.set_ylim(0, map_height)
        ax2.set_aspect('equal')

        if current_step is not None:
            ax2.set_title(f'IGDM Concentration Field (Step {current_step})',
                          fontsize=12, fontweight='bold')
        else:
            ax2.set_title('IGDM Concentration Field', fontsize=12, fontweight='bold')

        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        x_resolution = max(100, int(map_width * 10))
        y_resolution = max(60, int(map_height * 10))
        x_grid = np.linspace(0, map_width, x_resolution)
        y_grid = np.linspace(0, map_height, y_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        if self.igdm_model is not None and current_step is not None:
            Z = np.zeros_like(X)
            for i in range(len(y_grid)):
                for j in range(len(x_grid)):
                    Z[i, j] = self.igdm_model.compute_concentration(
                        (X[i, j], Y[i, j]), true_source, 1.0,
                        time_step=current_step, wind_offset=wind_offset,
                    )
        else:
            Z = np.exp(-(np.sqrt((X - true_source[0]) ** 2 +
                                 (Y - true_source[1]) ** 2) ** 2 / 2.0))

        im = ax2.contourf(X, Y, Z, levels=25, cmap='hot_r')
        plt.colorbar(im, ax=ax2, label='Concentration')

        self._plot_obstacles(ax2, occupancy_grid)
        ax2.plot(true_source[0], true_source[1], 'r*', markersize=12,
                 label='True source')
        if eff_source is not None:
            ax2.plot(eff_source[0], eff_source[1], '^', color='magenta',
                     markersize=10, markeredgecolor='black',
                     label='Effective source')
            ax2.legend(loc='upper right', fontsize=8)

        suptitle_parts = []
        if distance_to_true is not None and d_success_thr is not None:
            suptitle_parts.append(
                f'Distance to true source = {distance_to_true:.3f}m '
                f'(threshold: {d_success_thr:.3f}m)'
            )
        if sensor_reading is not None or sensor_threshold is not None or digital_value is not None:
            gas_parts = []
            if sensor_reading is not None:
                gas_parts.append(f'Reading: {sensor_reading:.4f}')
            if sensor_threshold is not None:
                gas_parts.append(f'Threshold: {sensor_threshold:.4f}')
            if digital_value is not None:
                gas_parts.append(f'Digital: {digital_value}')
            suptitle_parts.append('\n' + '   '.join(gas_parts))

        if suptitle_parts:
            plt.suptitle(''.join(suptitle_parts), fontsize=11, y=1.02)
        plt.tight_layout()

        filename = self.output_dir / f"step_{self.step_count:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        self.step_count += 1
