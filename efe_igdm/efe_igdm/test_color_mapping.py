#!/usr/bin/env python3
"""
Test color mapping visualization for RRT branches.

This script helps you understand how utilities are mapped to colors
and what the branches should look like in RViz.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def utility_to_color(norm_util):
    """
    Convert normalized utility [0, 1] to RGB color.

    Color scheme:
    - 0.0: Pure Red (low utility)
    - 0.5: Yellow (medium)
    - 1.0: Pure Green (high utility)
    """
    # Apply enhancement for better contrast
    norm_util_enhanced = norm_util ** 0.5

    if norm_util_enhanced < 0.5:
        # Red to Yellow transition
        r = 1.0
        g = 2.0 * norm_util_enhanced
        b = 0.0
    else:
        # Yellow to Green transition
        r = 2.0 * (1.0 - norm_util_enhanced)
        g = 1.0
        b = 0.0

    return (r, g, b)


def visualize_color_gradient():
    """Show the color gradient for different utility values."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Continuous color bar
    utilities = np.linspace(0, 1, 256)
    colors = [utility_to_color(u) for u in utilities]

    ax1.imshow([colors], aspect='auto', extent=[0, 1, 0, 1])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Normalized Utility', fontsize=12)
    ax1.set_title('Color Gradient: Red (Low) → Yellow (Med) → Green (High)', fontsize=14, fontweight='bold')
    ax1.set_yticks([])

    # Add markers at key points
    key_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for u in key_points:
        ax1.axvline(u, color='black', linestyle='--', alpha=0.5, linewidth=1)
        color = utility_to_color(u)
        ax1.text(u, 0.5, f'{u:.2f}\n{color}', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Sample utilities and their colors
    ax2.set_title('Example: 10 Paths with Different Utilities', fontsize=14, fontweight='bold')

    # Simulate utilities with varying spread
    sample_utils = np.random.beta(2, 2, 10)  # Beta distribution for realistic spread
    sample_utils_norm = (sample_utils - sample_utils.min()) / (sample_utils.max() - sample_utils.min())

    x_positions = np.arange(10)
    colors_sample = [utility_to_color(u) for u in sample_utils_norm]

    bars = ax2.bar(x_positions, sample_utils, color=colors_sample, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Path Index', fontsize=12)
    ax2.set_ylabel('Utility Value', fontsize=12)
    ax2.set_xticks(x_positions)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate with normalized values
    for i, (u_raw, u_norm) in enumerate(zip(sample_utils, sample_utils_norm)):
        ax2.text(i, u_raw + 0.05, f'{u_norm:.2f}', ha='center', fontsize=9)

    # Plot 3: Effect of utility range on colors
    ax3.set_title('Impact of Utility Range on Color Variation', fontsize=14, fontweight='bold')

    scenarios = [
        ('Wide Range', [0.1, 0.3, 0.5, 0.7, 0.9]),
        ('Narrow Range', [0.45, 0.47, 0.50, 0.52, 0.55]),
        ('All Similar', [0.50, 0.50, 0.50, 0.50, 0.50])
    ]

    y_offset = 0
    for scenario_name, raw_utils in scenarios:
        # Normalize
        raw_utils = np.array(raw_utils)
        if raw_utils.max() - raw_utils.min() > 1e-6:
            norm_utils = (raw_utils - raw_utils.min()) / (raw_utils.max() - raw_utils.min())
        else:
            norm_utils = np.ones_like(raw_utils) * 0.5

        colors_scenario = [utility_to_color(u) for u in norm_utils]

        # Draw colored bars
        for i, color in enumerate(colors_scenario):
            rect = mpatches.Rectangle((i * 0.18, y_offset), 0.15, 0.25,
                                     facecolor=color, edgecolor='black', linewidth=1.5)
            ax3.add_patch(rect)

        # Label
        ax3.text(-0.3, y_offset + 0.125, scenario_name, ha='right', va='center', fontweight='bold')

        # Show raw values
        for i, (raw, norm) in enumerate(zip(raw_utils, norm_utils)):
            ax3.text(i * 0.18 + 0.075, y_offset - 0.05, f'{raw:.2f}',
                    ha='center', fontsize=8)

        y_offset += 0.35

    ax3.set_xlim(-0.5, 1.0)
    ax3.set_ylim(-0.15, y_offset)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('color_mapping_test.png', dpi=150, bbox_inches='tight')
    print("✓ Color mapping visualization saved to: color_mapping_test.png")
    plt.show()


def simulate_rrt_scenario():
    """Simulate what you should see with typical RRT utilities."""
    print("\n" + "="*60)
    print("SIMULATED RRT SCENARIO")
    print("="*60)

    # Typical entropy gains might look like this
    entropy_gains = np.array([0.45, 0.52, 0.38, 0.61, 0.49, 0.55, 0.42, 0.58, 0.47, 0.51])

    print(f"\nRaw utilities (entropy gains): {entropy_gains}")
    print(f"Min: {entropy_gains.min():.4f}, Max: {entropy_gains.max():.4f}")
    print(f"Range: {entropy_gains.max() - entropy_gains.min():.4f}")

    # Normalize
    norm_utils = (entropy_gains - entropy_gains.min()) / (entropy_gains.max() - entropy_gains.min())

    print(f"\nNormalized utilities: {norm_utils}")
    print(f"\nPath Index | Raw Utility | Normalized | Color Description")
    print("-" * 70)

    for i, (raw, norm) in enumerate(zip(entropy_gains, norm_utils)):
        color = utility_to_color(norm)

        if norm < 0.33:
            color_desc = "🔴 RED (low)"
        elif norm < 0.66:
            color_desc = "🟡 YELLOW (med)"
        else:
            color_desc = "🟢 GREEN (high)"

        print(f"   {i:2d}      |    {raw:.4f}    |   {norm:.4f}   | {color_desc}")

    print("\n" + "="*60)
    print("WHAT YOU SHOULD SEE IN RVIZ:")
    print("="*60)
    print("- If utilities vary (like above): Mix of red, yellow, green paths")
    print("- If all utilities are similar: All paths will be same color (yellow)")
    print("- If no variation: Check that entropy gains are being calculated")
    print("="*60 + "\n")


if __name__ == '__main__':
    print("Testing Color Mapping for RRT Branch Visualization\n")

    simulate_rrt_scenario()
    visualize_color_gradient()
