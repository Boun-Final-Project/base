"""
Parallel Execution Script for Large Map Implementations.

Runs five large map versions in parallel:
1. Discrete RRT Planner
2. Directional Planner
3. Updated RRT Planner
4. Updated Weighted RRT Planner
5. Extended Penalty Planner

Tracks execution time and provides performance analysis.
This script demonstrates whether running all algorithms simultaneously affects performance.
"""

import subprocess
import time
import multiprocessing
from pathlib import Path
from datetime import datetime


def run_discrete_planner():
    """Run the discrete sensor version in a subprocess."""
    print("\n" + "="*70)
    print("STARTING: Discrete RRT Planner (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map_discrete.py'],
        cwd=Path(__file__).parent,
        capture_output=False
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
    print("STARTING: Directional Planner (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map_directional.py'],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Directional Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def run_updated_rrt_planner():
    """Run the updated RRT planner version in a subprocess."""
    print("\n" + "="*70)
    print("STARTING: Updated RRT Planner (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'updated_rrt_infotaxis_igdm_large_map_discrete.py'],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Updated RRT Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def run_updated_weighted_rrt_planner():
    """Run the updated weighted RRT planner version in a subprocess."""
    print("\n" + "="*70)
    print("STARTING: Updated Weighted RRT Planner (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'updated_weighted_rrt_infotaxis_igdm_large_map_discrete.py'],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Updated Weighted RRT Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def run_extended_penalty_planner():
    """Run the extended penalty planner version in a subprocess."""
    print("\n" + "="*70)
    print("STARTING: Extended Penalty Planner (Large Map)")
    print("="*70)

    start_time = time.time()
    process = subprocess.run(
        ['python', 'rrt_infotaxis_igdm_improved_large_map_discrete_extended_penalty.py'],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    elapsed_time = time.time() - start_time

    return {
        'name': 'Extended Penalty Planner',
        'returncode': process.returncode,
        'elapsed_time': elapsed_time
    }


def main():
    """Run all planners in parallel."""
    print("\n" + "="*80)
    print("PARALLEL EXECUTION: All Large Map Implementations")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print("="*80)

    # Create processes for parallel execution
    p1 = multiprocessing.Process(target=run_discrete_planner)
    p2 = multiprocessing.Process(target=run_directional_planner)
    p3 = multiprocessing.Process(target=run_updated_rrt_planner)
    p4 = multiprocessing.Process(target=run_updated_weighted_rrt_planner)
    p5 = multiprocessing.Process(target=run_extended_penalty_planner)

    # Record overall start time
    overall_start = time.time()

    # Start all processes
    print("\nStarting all five planners simultaneously...")
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    # Wait for all to complete
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    overall_elapsed = time.time() - overall_start

    # Print results
    print("\n" + "="*80)
    print("EXECUTION COMPLETED")
    print("="*80)
    print(f"Overall execution time (all running in parallel): {overall_elapsed:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS:")
    print("="*80)
    print("\nWhen running all scripts simultaneously:")
    print("- All five utilize independent CPU cores (multiprocessing)")
    print("- Memory usage is higher due to running 5 instances")
    print("- Particle filter (400 particles) and visualization operations run in parallel")
    print("- Expected overall time ≈ max(discrete_time, directional_time, updated_rrt_time, updated_weighted_rrt_time, extended_penalty_time)")
    print("- If running sequentially, time would be the sum of all five runtimes")
    print("\n✓ You can see all outputs simultaneously in the terminal above.")
    print("✓ Log files are saved in '/home/hdd/akademia/cmpe/final-project/week-9/' for detailed analysis.")
    print("="*80)


if __name__ == "__main__":
    main()
