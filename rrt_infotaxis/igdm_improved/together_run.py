import importlib
import random
import csv
import numpy as np
import time
from occupancy_grid import OccupancyGrid

# ================= CONFIGURATION =================
# LIST YOUR 6 SCRIPTS HERE
# Format: "module_name": "ClassName"
# (Do not include .py in module_name)
SCRIPTS_CONFIG = [
    {"module": "simal_1room", "class": "RRTInfotaxisIGDM"},
    {"module": "simal_adaptive_1room", "class": "RRTInfotaxisIGDM"}, 
    {"module": "simal_3room", "class": "RRTInfotaxisIGDM"},
    {"module": "simal_adaptive_3rooms", "class": "RRTInfotaxisIGDM"},
    {"module": "simal_obstacle", "class": "RRTInfotaxisIGDM"},
    {"module": "simal_adaptive_obstacle", "class": "RRTInfotaxisIGDM"},
]

MAP_WIDTH = 25.0
MAP_HEIGHT = 25.0
RESOLUTION = 0.1
OUTPUT_FILE = "benchmark_results.csv"

# ================= HELPER FUNCTIONS =================

def get_valid_random_start_positions(num_points=3):
    """Generates valid start positions (not inside obstacles)."""
    # Initialize a temporary grid to check validity (using same obstacles as your main maps)
    # NOTE: Ensure this matches your "Large Map" obstacle layout for validity checking
    grid = OccupancyGrid(MAP_WIDTH, MAP_HEIGHT, RESOLUTION)
    
    # Add the obstacles from your environment so we don't spawn robots inside walls
    # (Copying the 2-obstacle layout for safety)
    grid.add_rectangular_obstacle(x_min=4.9, x_max=5.1, y_min=4.0, y_max=10.0, value=1)
    grid.add_rectangular_obstacle(x_min=14.9, x_max=15.1, y_min=4.0, y_max=10.0, value=1)
    
    valid_points = []
    while len(valid_points) < num_points:
        # Pick random coordinates
        rx = random.uniform(1.0, MAP_WIDTH - 1.0)
        ry = random.uniform(1.0, MAP_HEIGHT - 1.0)
        
        # Check if valid (not in obstacle)
        if grid.is_valid((rx, ry)):
            valid_points.append((rx, ry))
            
    return valid_points

def run_benchmark():
    results = []
    
    # 1. Generate the "5-5-5" Start Points (3 distinct points)
    print("Generating 3 random valid start positions...")
    start_positions = get_valid_random_start_positions(3)
    
    for i, pos in enumerate(start_positions):
        print(f"  Start Point {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")

    # 2. Iterate through each script
    for entry in SCRIPTS_CONFIG:
        module_name = entry["module"]
        class_name = entry["class"]
        
        print(f"\nBenchmark started for: {module_name}")
        
        try:
            # Dynamically import the script and class
            module = importlib.import_module(module_name)
            AlgorithmClass = getattr(module, class_name)
        except Exception as e:
            print(f"Error loading {module_name}: {e}")
            continue

        # 3. The "5-5-5" Loop
        # For each of the 3 start positions, run 5 times
        for pos_idx, start_pos in enumerate(start_positions):
            print(f"  > Testing Start Point {pos_idx+1} ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
            
            for run_idx in range(1, 6): # 5 runs per start point
                try:
                    # Instantiate algorithm with dynamic start
                    algo = AlgorithmClass(sigma_m=1.0)
                    
                    # Manually override start pos if not handled in __init__
                    # (This is a safety net if you didn't update __init__ perfectly)
                    algo.robot_start = start_pos
                    algo.robot_pos = start_pos
                    algo.trajectory = [start_pos]
                    
                    # Run
                    start_time = time.time()
                    steps, success = algo.run()
                    duration = time.time() - start_time
                    
                    # Log
                    status = "SUCCESS" if success else "TIMEOUT"
                    print(f"    Run {run_idx}/5: {status} in {steps} steps ({duration:.2f}s)")
                    
                    results.append({
                        "Script": module_name,
                        "Start_Point_ID": pos_idx + 1,
                        "Start_X": start_pos[0],
                        "Start_Y": start_pos[1],
                        "Run_ID": run_idx,
                        "Steps": steps,
                        "Success": success,
                        "Duration_Sec": duration
                    })
                    
                except Exception as e:
                    print(f"    Run {run_idx} FAILED: {e}")

    # 4. Save Results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = ["Script", "Start_Point_ID", "Start_X", "Start_Y", "Run_ID", "Steps", "Success", "Duration_Sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("Benchmark Complete!")

if __name__ == "__main__":
    run_benchmark()