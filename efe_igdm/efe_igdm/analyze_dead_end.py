#!/usr/bin/env python3
"""
Analyze dead end detection from IGDM log files.

This script reads the CSV log files and visualizes the dead end detection
behavior, showing when the detector triggers and how the adaptive threshold
evolves over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def plot_dead_end_analysis(csv_file: str, output_dir: str = None):
    """
    Plot dead end detection analysis from a CSV log file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV log file.
    output_dir : str, optional
        Directory to save plots. If None, displays plots instead.
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Check if dead end columns exist
    required_cols = ['bi_optimal', 'bi_threshold', 'dead_end_detected']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file missing required columns: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Get dead end points
    dead_end_steps = df[df['dead_end_detected'] == 1]['step'].values
    dead_end_bi = df[df['dead_end_detected'] == 1]['bi_optimal'].values

    # Plot 1: Branch Information and Threshold
    ax1 = axes[0]
    ax1.plot(df['step'], df['bi_optimal'], 'b-', label='BI* (Optimal Branch Info)', linewidth=2)
    ax1.plot(df['step'], df['bi_threshold'], 'r--', label='BI_threshold', linewidth=2)

    if len(dead_end_steps) > 0:
        ax1.scatter(dead_end_steps, dead_end_bi, color='red', s=150,
                   marker='X', label='Dead End Detected', zorder=5)

    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Branch Information', fontsize=11)
    ax1.set_title('Dead End Detection (Equations 20-21)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy over time
    ax2 = axes[1]
    ax2.plot(df['step'], df['entropy'], 'g-', linewidth=2)
    if len(dead_end_steps) > 0:
        dead_end_entropy = df[df['dead_end_detected'] == 1]['entropy'].values
        ax2.scatter(dead_end_steps, dead_end_entropy, color='red', s=150,
                   marker='X', zorder=5)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Entropy (bits)', fontsize=11)
    ax2.set_title('Particle Filter Entropy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Standard deviations
    ax3 = axes[2]
    ax3.plot(df['step'], df['std_dev_x'], 'b-', label='σ_x', linewidth=2)
    ax3.plot(df['step'], df['std_dev_y'], 'r-', label='σ_y', linewidth=2)

    # Add sigma_threshold line if available
    if 'sigma_threshold' in df.columns:
        sigma_t = df['sigma_threshold'].iloc[0]
    else:
        sigma_t = 0.6  # Default value
    ax3.axhline(y=sigma_t, color='k', linestyle='--', label=f'σ_threshold = {sigma_t:.2f}')

    if len(dead_end_steps) > 0:
        dead_end_sigma = df[df['dead_end_detected'] == 1]['std_dev_x'].values
        ax3.scatter(dead_end_steps, dead_end_sigma, color='red', s=150,
                   marker='X', zorder=5)

    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Standard Deviation (m)', fontsize=11)
    ax3.set_title('Estimation Uncertainty', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Dead end detection events
    ax4 = axes[3]
    ax4.scatter(df['step'], df['dead_end_detected'], color='red', s=50, alpha=0.6)
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Dead End', fontsize=11)
    ax4.set_title('Dead End Detection Events', fontsize=13, fontweight='bold')
    ax4.set_ylim([-0.1, 1.1])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        output_file = os.path.join(output_dir, f'{base_name}_dead_end_analysis.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("DEAD END DETECTION STATISTICS")
    print("="*60)
    print(f"Total steps: {len(df)}")
    print(f"Dead ends detected: {df['dead_end_detected'].sum()}")
    print(f"Dead end rate: {df['dead_end_detected'].mean()*100:.2f}%")
    print(f"\nBranch Information (BI*):")
    print(f"  Mean: {df['bi_optimal'].mean():.4f}")
    print(f"  Std: {df['bi_optimal'].std():.4f}")
    print(f"  Min: {df['bi_optimal'].min():.4f}")
    print(f"  Max: {df['bi_optimal'].max():.4f}")
    print(f"\nThreshold (BI_thresh):")
    print(f"  Mean: {df['bi_threshold'].mean():.4f}")
    print(f"  Std: {df['bi_threshold'].std():.4f}")
    print(f"  Min: {df['bi_threshold'].min():.4f}")
    print(f"  Max: {df['bi_threshold'].max():.4f}")

    if len(dead_end_steps) > 0:
        print(f"\nDead end steps: {dead_end_steps.tolist()}")
    print("="*60 + "\n")


def plot_trajectory_with_dead_ends(csv_file: str, output_dir: str = None):
    """
    Plot robot trajectory with dead end locations marked.

    Parameters
    ----------
    csv_file : str
        Path to the CSV log file.
    output_dir : str, optional
        Directory to save plots. If None, displays plots instead.
    """
    df = pd.read_csv(csv_file)

    # Check required columns
    if not all(col in df.columns for col in ['robot_x', 'robot_y', 'dead_end_detected']):
        print("Error: Missing robot position or dead end columns")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectory
    ax.plot(df['robot_x'], df['robot_y'], 'b-', linewidth=2, alpha=0.6, label='Robot trajectory')

    # Mark start and end
    ax.scatter(df['robot_x'].iloc[0], df['robot_y'].iloc[0],
              color='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter(df['robot_x'].iloc[-1], df['robot_y'].iloc[-1],
              color='blue', s=200, marker='s', label='End', zorder=5)

    # Mark dead ends
    dead_end_mask = df['dead_end_detected'] == 1
    if dead_end_mask.any():
        ax.scatter(df[dead_end_mask]['robot_x'], df[dead_end_mask]['robot_y'],
                  color='red', s=150, marker='X', label='Dead End', zorder=5)

    # Mark estimated source location (final)
    if all(col in df.columns for col in ['est_x', 'est_y']):
        ax.scatter(df['est_x'].iloc[-1], df['est_y'].iloc[-1],
                  color='orange', s=300, marker='*', label='Estimated Source', zorder=5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Robot Trajectory with Dead End Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        output_file = os.path.join(output_dir, f'{base_name}_trajectory.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze dead end detection from IGDM log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single log file
  python3 analyze_dead_end.py ~/igdm_logs/igdm_log_20250102_120000.csv

  # Save plots to a directory
  python3 analyze_dead_end.py ~/igdm_logs/igdm_log_20250102_120000.csv -o ./plots

  # Analyze the latest log file
  python3 analyze_dead_end.py $(ls -t ~/igdm_logs/*.csv | head -1)
        """
    )
    parser.add_argument('csv_file', type=str, help='Path to CSV log file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory for plots (default: display only)')
    parser.add_argument('-t', '--trajectory', action='store_true',
                       help='Also plot trajectory with dead end locations')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return

    print(f"Analyzing: {args.csv_file}\n")

    # Main analysis plot
    plot_dead_end_analysis(args.csv_file, args.output)

    # Optional trajectory plot
    if args.trajectory:
        plot_trajectory_with_dead_ends(args.csv_file, args.output)


if __name__ == '__main__':
    main()
