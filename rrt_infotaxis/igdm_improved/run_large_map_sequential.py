"""
Sequential Execution Script for Large Map Implementations.

Runs all three large map versions sequentially:
1. Binary RRT Planner (original implementation)
2. Discrete RRT Planner
3. Directional Planner with Discrete Sensor

Allows for comparison with parallel execution.

Use this to compare execution times:
- Sequential: Total time = time(binary) + time(discrete) + time(directional)
- Parallel: Total time ≈ max(time(binary), time(discrete), time(directional))
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_binary_rrt():
    """Run the binary sensor RRT version in a subprocess."""
    print("\n" + "="*70)
    print("EXECUTING: RRT-Infotaxis with Binary Sensor (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map.py'],
        cwd=Path(__file__).parent
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Binary RRT Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def run_discrete_planner():
    """Run the discrete sensor version in a subprocess."""
    print("\n" + "="*70)
    print("EXECUTING: RRT-Infotaxis with Discrete Sensor (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map_discrete.py'],
        cwd=Path(__file__).parent
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Discrete RRT Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def run_directional_planner():
    """Run the directional planner version in a subprocess."""
    print("\n" + "="*70)
    print("EXECUTING: Directional Planner with Discrete Sensor (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map_directional.py'],
        cwd=Path(__file__).parent
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Directional Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def main():
    """Run all planners sequentially."""
    print("\n" + "="*80)
    print("SEQUENTIAL EXECUTION: All Large Map Implementations")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    overall_start = time.time()

    # Run each planner in sequence
    result1 = run_binary_rrt()
    result2 = run_discrete_planner()
    result3 = run_directional_planner()

    overall_elapsed = time.time() - overall_start

    # Print results
    print("\n" + "="*80)
    print("EXECUTION SUMMARY (SEQUENTIAL)")
    print("="*80)
    print(f"\n{result1['name']}:")
    print(f"  Return code: {result1['returncode']}")
    print(f"  Execution time: {result1['elapsed_time']:.2f} seconds")

    print(f"\n{result2['name']}:")
    print(f"  Return code: {result2['returncode']}")
    print(f"  Execution time: {result2['elapsed_time']:.2f} seconds")

    print(f"\n{result3['name']}:")
    print(f"  Return code: {result3['returncode']}")
    print(f"  Execution time: {result3['elapsed_time']:.2f} seconds")

    total_individual = result1['elapsed_time'] + result2['elapsed_time'] + result3['elapsed_time']

    print(f"\nTotal sequential time: {overall_elapsed:.2f} seconds")
    print(f"Sum of individual times: {total_individual:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON:")
    print("="*80)
    print("\nSequential execution:")
    print(f"  - Binary RRT runs first ({result1['elapsed_time']:.2f}s)")
    print(f"  - Discrete RRT runs next ({result2['elapsed_time']:.2f}s)")
    print(f"  - Directional Planner runs last ({result3['elapsed_time']:.2f}s)")
    print(f"  - Total time: {overall_elapsed:.2f}s")
    print("\nFor parallel execution comparison:")
    longest_time = max(result1['elapsed_time'], result2['elapsed_time'], result3['elapsed_time'])
    print(f"  - Expected time (parallel): ~{longest_time:.2f}s")
    print(f"  - Time savings vs sequential: ~{overall_elapsed - longest_time:.2f}s")
    print(f"  - Speedup factor: ~{total_individual / longest_time:.2f}x")
    print("\n✓ Log files are saved in '/home/hdd/akademia/cmpe/final-project/week-9/' for detailed analysis.")
    print("="*80)


if __name__ == "__main__":
    main()
