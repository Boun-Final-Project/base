#!/usr/bin/env python3
"""
Plot gas source localization search trajectory from logged data.
Generates visualizations similar to Figure 5 in the paper.

Usage:
    python3 plot_search_trajectory.py <log_file.csv>
    python3 plot_search_trajectory.py <log_file.csv> --map <map_image.png>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import pandas as pd
import argparse
import os
from PIL import Image


class SearchTrajectoryPlotter:
    """Visualize gas source search trajectory from CSV log."""

    def __init__(self, csv_file, map_image=None):
        """
        Initialize plotter.

        Args:
            csv_file: Path to CSV log file
            map_image: Optional path to map image
        """
        self.csv_file = csv_file
        self.map_image = map_image
        self.data = None
        self.load_data()

    def load_data(self):
        """Load CSV data."""
        self.data = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.data)} steps from {self.csv_file}")
        print(f"Columns: {list(self.data.columns)}")

    def plot_trajectory(self, save_path=None, show_particles=False, title=None):
        """
        Plot search trajectory with estimated source evolution.

        Args:
            save_path: Path to save figure (if None, displays interactively)
            show_particles: Whether to show particle filter estimate evolution
            title: Custom title for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Load map image if provided
        if self.map_image and os.path.exists(self.map_image):
            img = Image.open(self.map_image)
            # You may need to adjust extent based on your map scale
            # extent = [x_min, x_max, y_min, y_max]
            ax.imshow(img, alpha=0.5, extent=self._get_map_extent())

        # Extract trajectory data
        robot_x = self.data['robot_x'].values
        robot_y = self.data['robot_y'].values
        est_x = self.data['est_x'].values
        est_y = self.data['est_y'].values

        # Get planner mode for coloring
        planner_mode = self.data['planner_mode'].values

        # Plot search trajectory with color-coded segments
        self._plot_colored_trajectory(ax, robot_x, robot_y, planner_mode)

        # Plot initial position
        ax.plot(robot_x[0], robot_y[0], 'go', markersize=15,
                label='Initial position', zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)

        # Plot final position
        ax.plot(robot_x[-1], robot_y[-1], 'bs', markersize=15,
                label='Final position', zorder=10, markeredgecolor='darkblue', markeredgewidth=2)

        # Plot estimated source evolution
        if show_particles:
            # Show every Nth estimate to avoid clutter
            step_size = max(1, len(est_x) // 20)
            ax.scatter(est_x[::step_size], est_y[::step_size],
                      c=np.arange(0, len(est_x), step_size),
                      cmap='YlOrRd', s=50, alpha=0.6,
                      label='Estimated source (evolution)', zorder=5)

        # Plot final estimated source location
        ax.plot(est_x[-1], est_y[-1], 'r*', markersize=25,
                label='Estimated source location', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)

        # Plot true source if available (from parameters)
        # You can manually set this or read from a config file
        true_x = self.data.get('true_source_x', [None])[0] if 'true_source_x' in self.data.columns else None
        true_y = self.data.get('true_source_y', [None])[0] if 'true_source_y' in self.data.columns else None

        # Alternatively, try to read from summary file or set manually
        if true_x is None or true_y is None or np.isnan(true_x) or np.isnan(true_y):
            # Try to read from parameters or summary
            summary_file = self.csv_file.replace('.csv', '_summary.txt')
            if os.path.exists(summary_file):
                true_x, true_y = self._read_true_source_from_summary(summary_file)

        if true_x is not None and true_y is not None and not np.isnan(true_x) and not np.isnan(true_y):
            ax.plot(true_x, true_y, 'y*', markersize=30,
                    label='True source location', zorder=12,
                    markeredgecolor='orange', markeredgewidth=3)

            # Draw error line
            ax.plot([est_x[-1], true_x], [est_y[-1], true_y],
                   'r--', linewidth=2, alpha=0.5, label='Localization error')

            # Calculate final error
            final_error = np.sqrt((est_x[-1] - true_x)**2 + (est_y[-1] - true_y)**2)
            ax.text(0.02, 0.98, f'Final error: {final_error:.3f} m',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Formatting
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_ylabel('Y (m)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        else:
            filename = os.path.basename(self.csv_file)
            ax.set_title(f'Gas Source Search Trajectory\n{filename}', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot to {save_path}")
        else:
            plt.show()

    def plot_trajectory_with_concentration(self, save_path=None, title=None,
                                           map_extent=None, show_colorbar=False):
        """
        Plot trajectory on map with gas concentration visualization.

        Args:
            save_path: Path to save figure
            title: Custom title
            map_extent: [x_min, x_max, y_min, y_max] for map, auto-detect if None
            show_colorbar: Show concentration colorbar (deprecated, uses bubbles now)
        """
        fig, ax = plt.subplots(figsize=(14, 12))

        # Load and display map image
        if self.map_image and os.path.exists(self.map_image):
            img = Image.open(self.map_image)

            # Determine map extent
            if map_extent is None:
                map_extent = self._auto_detect_map_extent()

            ax.imshow(img, extent=map_extent, alpha=0.9, zorder=1)
            print(f"Map extent: {map_extent}")

        # Extract data
        robot_x = self.data['robot_x'].values
        robot_y = self.data['robot_y'].values
        sensor_values = self.data['sensor_value'].values
        est_x = self.data['est_x'].values
        est_y = self.data['est_y'].values

        # Plot trajectory path (simple line without color coding)
        ax.plot(robot_x, robot_y, 'b-', linewidth=3, alpha=0.7, zorder=4,
                label='Search path')

        # Plot concentration as bubble sizes
        # Normalize sizes: min concentration -> small bubble, max -> large bubble
        max_bubble_size = 800
        min_bubble_size = 50

        if sensor_values.max() > 0:
            # Scale bubble sizes proportionally to concentration
            sizes = min_bubble_size + (sensor_values / sensor_values.max()) * (max_bubble_size - min_bubble_size)
        else:
            sizes = np.full_like(sensor_values, min_bubble_size)

        # Plot bubbles with opaque fill
        ax.scatter(robot_x, robot_y, s=sizes, c='yellow', alpha=1.0,
                  edgecolors='orange', linewidths=1.5, zorder=5,
                  label='Concentration')

        # Plot initial position
        ax.plot(robot_x[0], robot_y[0], 'go', markersize=20,
                label='Start', zorder=10, markeredgecolor='darkgreen',
                markeredgewidth=3)

        # Plot final position
        ax.plot(robot_x[-1], robot_y[-1], 'bs', markersize=20,
                label='End', zorder=10, markeredgecolor='darkblue',
                markeredgewidth=3)

        # Plot final estimated source
        ax.plot(est_x[-1], est_y[-1], 'r*', markersize=25,
                label='Estimated source', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)

        # Plot true source - always show at (2.5, 4.5) for env_a
        # Or get from data if available
        true_x, true_y = self._get_true_source()
        if true_x is None or true_y is None or np.isnan(true_x) or np.isnan(true_y):
            # Default to env_a source location (from sim1.yaml)
            true_x, true_y = 2.5, 4.5

        ax.plot(true_x, true_y, 'y*', markersize=25,
                label='True source', zorder=12,
                markeredgecolor='orange', markeredgewidth=2)

        # Draw error line
        ax.plot([est_x[-1], true_x], [est_y[-1], true_y],
               'r--', linewidth=2, alpha=0.6, zorder=4)

        # Calculate and display error
        final_error = np.sqrt((est_x[-1] - true_x)**2 + (est_y[-1] - true_y)**2)
        info_text = f'Error: {final_error:.2f} m | Steps: {len(robot_x)}'
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=1.5))

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Formatting
        ax.set_aspect('equal')
        ax.grid(False)

        # Legend at bottom center with proper marker sizing
        legend = ax.legend(loc='lower center', fontsize=14, framealpha=0.95,
                 edgecolor='black', ncol=3, bbox_to_anchor=(0.5, -0.01),
                 markerscale=0.8, handlelength=1.5, handletextpad=0.5,
                 columnspacing=1.0, borderpad=0.5)

        # Title - only add if explicitly provided
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved concentration trajectory plot to {save_path}")
        else:
            plt.show()

    def _add_scale_arrows(self, ax, map_extent):
        """Add scale arrows showing map dimensions."""
        from matplotlib.patches import FancyArrowPatch

        x_min, x_max, y_min, y_max = map_extent

        # Use actual environment dimensions (20m x 16m for env_a)
        # These are cleaner numbers than the extent values which include margins
        x_arrow_length = 20  # meters
        y_arrow_length = 16  # meters

        # Position scale arrows along the edges
        arrow_margin = 0.5  # meters from edge

        # X-axis scale arrow (horizontal, at bottom)
        arrow_x_y_pos = y_min + arrow_margin
        arrow_x = FancyArrowPatch(
            (x_min + arrow_margin, arrow_x_y_pos),
            (x_max - arrow_margin, arrow_x_y_pos),
            arrowstyle='<->', mutation_scale=20, linewidth=3,
            color='black', zorder=15
        )
        ax.add_patch(arrow_x)

        # X-axis label
        ax.text((x_min + x_max) / 2, arrow_x_y_pos - 0.8,
               f'{x_arrow_length}m', fontsize=14, fontweight='bold',
               ha='center', va='top', color='black',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='black', linewidth=2))

        # Y-axis scale arrow (vertical, at left)
        arrow_y_x_pos = x_min + arrow_margin
        arrow_y = FancyArrowPatch(
            (arrow_y_x_pos, y_min + arrow_margin),
            (arrow_y_x_pos, y_max - arrow_margin),
            arrowstyle='<->', mutation_scale=20, linewidth=3,
            color='black', zorder=15
        )
        ax.add_patch(arrow_y)

        # Y-axis label
        ax.text(arrow_y_x_pos - 0.8, (y_min + y_max) / 2,
               f'{y_arrow_length}m', fontsize=14, fontweight='bold',
               ha='right', va='center', color='black', rotation=90,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='black', linewidth=2))

    def _auto_detect_map_extent(self):
        """
        Auto-detect map extent from trajectory data.

        For env_a, uses values from OccupancyGrid3D.csv:
        Bottom-left: (-0.3, -0.3)
        Top-right: (20.3, 16.3)
        """
        # Default extent for env_a from GADEN OccupancyGrid3D.csv
        # If using different environment, specify with --map-extent
        return [-0.3, 20.3, -0.3, 16.3]

    def _get_true_source(self):
        """Get true source location from data."""
        true_x = self.data.get('true_source_x', [None])[0] if 'true_source_x' in self.data.columns else None
        true_y = self.data.get('true_source_y', [None])[0] if 'true_source_y' in self.data.columns else None

        # Try from summary file
        if true_x is None or true_y is None or np.isnan(true_x) or np.isnan(true_y):
            summary_file = self.csv_file.replace('.csv', '_summary.txt')
            if os.path.exists(summary_file):
                true_x, true_y = self._read_true_source_from_summary(summary_file)

        if true_x is not None and true_y is not None and not np.isnan(true_x) and not np.isnan(true_y):
            return true_x, true_y

        return None, None

    def _plot_colored_trajectory(self, ax, x, y, modes):
        """Plot trajectory with color-coded segments based on planner mode."""
        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color map: LOCAL = blue, GLOBAL = red
        colors = ['blue' if mode == 'LOCAL' else 'red' for mode in modes[:-1]]

        # Create line collection
        lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.7, zorder=3)
        ax.add_collection(lc)

        # Add legend entries for planner modes
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='Local planner (RRT-Infotaxis)'),
            Line2D([0], [0], color='red', linewidth=2, label='Global planner (Frontier exploration)')
        ]
        # Add to existing legend
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(legend_elements)

    def _get_map_extent(self):
        """Get map extent from trajectory bounds."""
        robot_x = self.data['robot_x'].values
        robot_y = self.data['robot_y'].values

        x_min, x_max = robot_x.min() - 2, robot_x.max() + 2
        y_min, y_max = robot_y.min() - 2, robot_y.max() + 2

        return [x_min, x_max, y_min, y_max]

    def _read_true_source_from_summary(self, summary_file):
        """Read true source location from summary file."""
        try:
            with open(summary_file, 'r') as f:
                for line in f:
                    if 'True source location' in line:
                        # Extract coordinates from line like "(2.00, 4.50)"
                        coords = line.split('(')[1].split(')')[0]
                        x, y = map(float, coords.split(','))
                        return x, y
        except Exception as e:
            print(f"Could not read true source from {summary_file}: {e}")
        return None, None

    def plot_metrics_over_time(self, save_path=None):
        """Plot key metrics over time (entropy, std dev, sensor readings)."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        steps = self.data['step'].values

        # Plot 1: Entropy over time
        axes[0].plot(steps, self.data['entropy'].values, 'b-', linewidth=2)
        axes[0].set_ylabel('Entropy', fontsize=12)
        axes[0].set_title('Search Progress Metrics', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Highlight mode transitions
        self._highlight_mode_transitions(axes[0], steps)

        # Plot 2: Standard deviations
        axes[1].plot(steps, self.data['std_dev_x'].values, 'r-', label='σ_x', linewidth=2)
        axes[1].plot(steps, self.data['std_dev_y'].values, 'g-', label='σ_y', linewidth=2)
        axes[1].plot(steps, self.data['std_dev_Q'].values, 'b-', label='σ_Q', linewidth=2, alpha=0.6)
        axes[1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Convergence threshold')
        axes[1].set_ylabel('Standard Deviation', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        self._highlight_mode_transitions(axes[1], steps)

        # Plot 3: Sensor readings
        axes[2].plot(steps, self.data['sensor_value'].values, 'purple', linewidth=2, label='Sensor reading')
        if 'threshold' in self.data.columns:
            axes[2].plot(steps, self.data['threshold'].values, 'orange', linewidth=2,
                        linestyle='--', label='Adaptive threshold', alpha=0.7)
        axes[2].set_ylabel('Gas Concentration', fontsize=12)
        axes[2].set_xlabel('Step', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        self._highlight_mode_transitions(axes[2], steps)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot to {save_path}")
        else:
            plt.show()

    def _highlight_mode_transitions(self, ax, steps):
        """Highlight global planner mode regions."""
        modes = self.data['planner_mode'].values

        # Find contiguous GLOBAL mode regions
        in_global = False
        start_step = None

        for i, mode in enumerate(modes):
            if mode == 'GLOBAL' and not in_global:
                start_step = steps[i]
                in_global = True
            elif mode != 'GLOBAL' and in_global:
                ax.axvspan(start_step, steps[i-1], alpha=0.2, color='red')
                in_global = False

        # Handle case where it ends in GLOBAL mode
        if in_global:
            ax.axvspan(start_step, steps[-1], alpha=0.2, color='red')

    def plot_dead_end_analysis(self, save_path=None):
        """Plot dead end detection analysis."""
        fig, ax = plt.subplots(figsize=(12, 6))

        steps = self.data['step'].values
        bi_optimal = self.data['bi_optimal'].values
        bi_threshold = self.data['bi_threshold'].values
        dead_end = self.data['dead_end_detected'].values

        # Plot BI optimal and threshold
        ax.plot(steps, bi_optimal, 'b-', linewidth=2, label='BI* (Optimal Branch Information)')
        ax.plot(steps, bi_threshold, 'r--', linewidth=2, label='BI_threshold (Dead-end threshold)')

        # Mark dead end detection points
        dead_end_steps = steps[dead_end == 1]
        dead_end_bi = bi_optimal[dead_end == 1]
        ax.scatter(dead_end_steps, dead_end_bi, c='red', s=100, marker='x',
                  linewidths=3, label='Dead-end detected', zorder=10)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Branch Information', fontsize=12)
        ax.set_title('Dead-End Detection Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dead-end analysis plot to {save_path}")
        else:
            plt.show()

    def generate_all_plots(self, output_dir=None, include_concentration=True):
        """Generate all visualization plots."""
        if output_dir is None:
            output_dir = os.path.dirname(self.csv_file)

        base_name = os.path.basename(self.csv_file).replace('.csv', '')

        print("Generating plots...")

        # Concentration trajectory plot (if map available)
        if include_concentration and self.map_image and os.path.exists(self.map_image):
            conc_path = os.path.join(output_dir, f'{base_name}_concentration.png')
            self.plot_trajectory_with_concentration(save_path=conc_path)

        # Standard trajectory plot
        traj_path = os.path.join(output_dir, f'{base_name}_trajectory.png')
        self.plot_trajectory(save_path=traj_path, show_particles=True)

        # Metrics plot
        metrics_path = os.path.join(output_dir, f'{base_name}_metrics.png')
        self.plot_metrics_over_time(save_path=metrics_path)

        # Dead-end analysis plot
        deadend_path = os.path.join(output_dir, f'{base_name}_deadend.png')
        self.plot_dead_end_analysis(save_path=deadend_path)

        print(f"\nAll plots saved to {output_dir}/")
        if include_concentration and self.map_image:
            print(f"  - {base_name}_concentration.png")
        print(f"  - {base_name}_trajectory.png")
        print(f"  - {base_name}_metrics.png")
        print(f"  - {base_name}_deadend.png")


def main():
    parser = argparse.ArgumentParser(
        description='Plot gas source localization search trajectory from CSV log'
    )
    parser.add_argument('csv_file', help='Path to CSV log file')
    parser.add_argument('--map', help='Path to map image (optional)', default=None)
    parser.add_argument('--output-dir', help='Output directory for plots', default=None)
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--trajectory', action='store_true', help='Plot trajectory only')
    parser.add_argument('--concentration', action='store_true',
                       help='Plot trajectory with gas concentration visualization (requires --map)')
    parser.add_argument('--metrics', action='store_true', help='Plot metrics only')
    parser.add_argument('--deadend', action='store_true', help='Plot dead-end analysis only')
    parser.add_argument('--title', help='Custom title for trajectory plot', default=None)
    parser.add_argument('--map-extent', nargs=4, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                       help='Map extent [x_min, x_max, y_min, y_max] (auto-detect if not specified)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return

    # Create plotter
    plotter = SearchTrajectoryPlotter(args.csv_file, args.map)

    # Determine what to plot
    if args.all:
        plotter.generate_all_plots(args.output_dir)
    elif args.concentration:
        if not args.map:
            print("Error: --concentration requires --map to be specified")
            return
        plotter.plot_trajectory_with_concentration(title=args.title, map_extent=args.map_extent)
    elif args.trajectory:
        plotter.plot_trajectory(title=args.title)
    elif args.metrics:
        plotter.plot_metrics_over_time()
    elif args.deadend:
        plotter.plot_dead_end_analysis()
    else:
        # Default: show concentration plot if map available, otherwise standard plots
        if args.map and os.path.exists(args.map):
            print("Showing concentration trajectory plot with map")
            plotter.plot_trajectory_with_concentration(title=args.title, map_extent=args.map_extent)
        else:
            print("Showing all plots interactively (use --all to save to files)")
            plotter.plot_trajectory(show_particles=True, title=args.title)
            plotter.plot_metrics_over_time()
            plotter.plot_dead_end_analysis()


if __name__ == '__main__':
    main()
