"""
Standalone test for the map generator.

Tests:
1. All 6 templates generate valid grids
2. Source and robot are placed in valid, connected positions
3. Connectivity between source and robot
4. Determinism (same seed → same map)
5. Visualization of all templates
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure 'reinforcement_learning' is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reinforcement_learning import config as cfg
from reinforcement_learning.envs.map_generator import MapGenerator


TEMPLATE_NAMES = [
    "Empty Room",
    "Single Vertical Wall",
    "U-Shaped Obstacle",
    "Three Walls (Staggered)",
    "Complex Maze",
    "Multi-Room (Doorways)",
]


def test_all_templates():
    """Generate each template and check basic validity."""
    print("=== Test: All 6 templates generate valid grids ===")
    rng = np.random.default_rng(42)
    gen = MapGenerator(rng=rng)

    for tid in range(6):
        result = gen.generate(template_id=tid)
        grid = result["grid"]
        src = result["source_pos"]
        rob = result["robot_pos"]

        assert grid.grid.shape[0] > 0 and grid.grid.shape[1] > 0, \
            f"Template {tid}: empty grid"
        assert grid.is_valid(position=src, radius=cfg.ROBOT_RADIUS), \
            f"Template {tid}: source in obstacle"
        assert grid.is_valid(position=rob, radius=cfg.ROBOT_RADIUS), \
            f"Template {tid}: robot in obstacle"

        dist = np.linalg.norm(np.array(src) - np.array(rob))
        assert dist >= cfg.MIN_SOURCE_ROBOT_DIST * 0.9, \
            f"Template {tid}: source-robot too close ({dist:.2f} m)"

        print(f"  Template {tid} ({TEMPLATE_NAMES[tid]}): "
              f"{grid.width:.1f}x{grid.height:.1f} m, "
              f"grid {grid.grid_width}x{grid.grid_height}, "
              f"src={src[0]:.1f},{src[1]:.1f} rob={rob[0]:.1f},{rob[1]:.1f} "
              f"dist={dist:.1f} m  ✓")

    print("PASSED\n")


def test_connectivity():
    """Verify source and robot are always reachable from each other."""
    print("=== Test: Connectivity (100 random maps) ===")
    rng = np.random.default_rng(123)
    gen = MapGenerator(rng=rng)
    failures = 0

    for i in range(100):
        result = gen.generate()
        grid = result["grid"]
        src = result["source_pos"]
        rob = result["robot_pos"]

        connected = gen._are_connected(grid, src, rob)
        if not connected:
            failures += 1
            print(f"  FAIL: map {i}, template {result['template_id']}")

    print(f"  {100 - failures}/100 maps connected")
    assert failures == 0, f"{failures} maps had disconnected positions"
    print("PASSED\n")


def test_determinism():
    """Same seed should produce the same map."""
    print("=== Test: Determinism ===")
    for tid in range(6):
        gen1 = MapGenerator(rng=np.random.default_rng(99))
        gen2 = MapGenerator(rng=np.random.default_rng(99))
        r1 = gen1.generate(template_id=tid)
        r2 = gen2.generate(template_id=tid)

        assert np.array_equal(r1["grid"].grid, r2["grid"].grid), \
            f"Template {tid}: grids differ"
        assert r1["source_pos"] == r2["source_pos"], \
            f"Template {tid}: source positions differ"
        assert r1["robot_pos"] == r2["robot_pos"], \
            f"Template {tid}: robot positions differ"

    print("  All 6 templates are deterministic  ✓")
    print("PASSED\n")


def test_uniform_sampling():
    """When template_id is None, all 6 templates should be sampled."""
    print("=== Test: Uniform template sampling (300 maps) ===")
    rng = np.random.default_rng(0)
    gen = MapGenerator(rng=rng)
    counts = [0] * 6

    for _ in range(300):
        result = gen.generate()
        counts[result["template_id"]] += 1

    for tid, c in enumerate(counts):
        print(f"  Template {tid} ({TEMPLATE_NAMES[tid]}): {c}")
        assert c > 10, f"Template {tid} sampled only {c} times in 300 tries"

    print("PASSED\n")


def test_generation_speed():
    """Benchmark map generation speed."""
    print("=== Test: Generation speed ===")
    rng = np.random.default_rng(7)
    gen = MapGenerator(rng=rng)

    t0 = time.time()
    n = 200
    for _ in range(n):
        gen.generate()
    elapsed = time.time() - t0
    rate = n / elapsed

    print(f"  {n} maps in {elapsed:.2f} s  ({rate:.0f} maps/sec)")
    print("PASSED\n")


def visualize_all_templates(save_path=None):
    """Generate and plot one instance of each template."""
    gen = MapGenerator()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for tid in range(6):
        result = gen.generate(template_id=tid)
        grid = result["grid"]
        src = result["source_pos"]
        rob = result["robot_pos"]
        ax = axes[tid]

        # Plot occupancy grid (0=free white, 1=obstacle gray)
        ax.imshow(
            grid.grid,
            origin="lower",
            extent=[0, grid.width, 0, grid.height],
            cmap="Greys",
            vmin=0, vmax=1,
            aspect="equal",
        )

        # Source and robot
        ax.plot(*src, "r*", markersize=15, label="Source")
        ax.plot(*rob, "bo", markersize=10, label="Robot")

        ax.set_title(f"Template {tid}: {TEMPLATE_NAMES[tid]}\n"
                     f"{grid.width:.1f}×{grid.height:.1f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if tid == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    test_all_templates()
    test_connectivity()
    test_determinism()
    test_uniform_sampling()
    test_generation_speed()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualize_all_templates(
        save_path=os.path.join(script_dir, "map_templates_preview.png")
    )

    print("All tests passed!")
