#!/usr/bin/env python3
"""
Plot entropy and other metrics from IGDM RRT-Infotaxis CSV log files.

Usage:
    python3 plot_entropy.py ~/igdm_logs/igdm_log_YYYYMMDD_HHMMSS.csv

Or to plot the most recent log:
    python3 plot_entropy.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def find_latest_log():
    """Find the most recent IGDM log file."""
    log_dir = Path.home() / 'igdm_logs'
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        sys.exit(1)

    log_files = list(log_dir.glob('igdm_log_*.csv'))
    if not log_files:
        print(f"Error: No log files found in {log_dir}")
        sys.exit(1)

    # Sort by modification time, most recent first
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    return latest_log

def plot_entropy_data(csv_file):
    """Plot entropy and related metrics from CSV file."""
    # Read CSV
    df = pd.read_csv(csv_file)

    print(f"Loaded {len(df)} data points from {csv_file}")
    print(f"Columns: {list(df.columns)}")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'IGDM RRT-Infotaxis Metrics\n{os.path.basename(csv_file)}', fontsize=14)

    # Convert to numpy arrays for pandas compatibility
    time = df['elapsed_time'].values

    # Plot 1: Entropy over time
    ax1 = axes[0, 0]
    ax1.plot(time, df['entropy'].values, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Entropy (nats)')
    ax1.set_title('Shannon Entropy over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Standard deviations
    ax2 = axes[0, 1]
    ax2.plot(time, df['std_dev_x'].values, 'r-', label='σ_x', linewidth=2)
    ax2.plot(time, df['std_dev_y'].values, 'g-', label='σ_y', linewidth=2)
    ax2.plot(time, df['std_dev_Q'].values, 'b-', label='σ_Q', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Estimation Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: J1 (Entropy Gain)
    ax3 = axes[1, 0]
    ax3.plot(time, df['J1_entropy_gain'].values, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('J1 (Information Gain)')
    ax3.set_title('Entropy Gain (Exploration)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: J2 (Travel Cost)
    ax4 = axes[1, 1]
    ax4.plot(time, df['J2_travel_cost'].values, 'orange', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('J2 (Path Cost)')
    ax4.set_title('Travel Cost (Exploitation)')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Sensor readings and threshold
    ax5 = axes[2, 0]
    ax5.plot(time, df['sensor_value'].values, 'b-', label='Raw Sensor', alpha=0.6, linewidth=1.5)
    ax5.plot(time, df['threshold'].values, 'r--', label='Threshold', linewidth=2)

    # Add binary value as scatter plot
    if 'binary_value' in df.columns:
        binary_times = time[df['binary_value'] == 1]
        binary_values = df['sensor_value'].values[df['binary_value'] == 1]
        ax5.scatter(binary_times, binary_values, c='green', s=30,
                   label='Binary=1 (Detection)', zorder=5, alpha=0.7)

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Concentration (μg/m³)')
    ax5.set_title('Sensor Readings & Adaptive Threshold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Number of branches
    ax6 = axes[2, 1]
    ax6.plot(time, df['num_branches'].values, 'g-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Number of Branches')
    ax6.set_title('RRT Branch Count')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total duration: {time[-1]:.2f} seconds")
    print(f"Number of steps: {len(df)}")
    print(f"Final entropy: {df['entropy'].values[-1]:.4f} nats")
    print(f"Entropy reduction: {df['entropy'].values[0] - df['entropy'].values[-1]:.4f} nats")
    print(f"\nFinal estimation:")
    print(f"  Source X: {df['est_x'].values[-1]:.3f} ± {df['std_dev_x'].values[-1]:.3f} m")
    print(f"  Source Y: {df['est_y'].values[-1]:.3f} ± {df['std_dev_y'].values[-1]:.3f} m")
    print(f"  Release rate Q: {df['est_Q'].values[-1]:.6f} ± {df['std_dev_Q'].values[-1]:.6f}")
    if 'binary_value' in df.columns:
        detections = df['binary_value'].sum()
        print(f"\nTotal detections (binary=1): {int(detections)} / {len(df)} ({100*detections/len(df):.1f}%)")
    if 'sigma_m' in df.columns:
        print(f"IGDM σ_m: {df['sigma_m'].values[0]:.3f} m")
    print("="*60)

    # Save figure
    output_file = str(csv_file).replace('.csv', '_plots.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()

def main():
    """Main function."""
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}")
            sys.exit(1)
    else:
        print("No file specified, using most recent log...")
        csv_file = find_latest_log()
        print(f"Using: {csv_file}")

    plot_entropy_data(csv_file)

if __name__ == '__main__':
    main()
