# Gas Source Localization - Trajectory Plotting

This script generates Figure 5-style visualizations from your IGDM search logs.

## Features

The plotting script generates three types of plots:

1. **Trajectory Plot** - Search path with color-coded planner modes (blue=LOCAL, red=GLOBAL)
   - Initial and final robot positions
   - Estimated source location evolution
   - True source location (if available)
   - Localization error visualization

2. **Metrics Over Time** - Key performance indicators
   - Entropy evolution
   - Standard deviations (σ_x, σ_y, σ_Q)
   - Gas sensor readings and adaptive threshold
   - Highlighted global planner mode regions

3. **Dead-End Analysis** - Dead-end detection behavior
   - BI* (optimal branch information)
   - BI_threshold (adaptive dead-end threshold)
   - Dead-end detection events marked

## Usage

### Quick Start (Interactive)

```bash
# Show all plots interactively
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv
```

### Generate All Plots (Save to Files)

```bash
# Generate all three plots and save as PNG files
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --all
```

This creates:
- `igdm_log_TIMESTAMP_trajectory.png`
- `igdm_log_TIMESTAMP_metrics.png`
- `igdm_log_TIMESTAMP_deadend.png`

### Individual Plots

```bash
# Only trajectory
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --trajectory

# Only metrics
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --metrics

# Only dead-end analysis
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --deadend
```

### With Map Overlay (Optional)

```bash
# Overlay trajectory on map image
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --map /path/to/map.png --all
```

### Custom Output Directory

```bash
# Save plots to specific directory
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --all --output-dir ~/my_results/
```

### Custom Title

```bash
# Add custom title to trajectory plot
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv --trajectory --title "Environment A - Trial 1"
```

## Find Your Latest Log

```bash
# List all logs sorted by date
ls -lt ~/igdm_logs/

# Plot the most recent log
LATEST_LOG=$(ls -t ~/igdm_logs/igdm_log_*.csv | head -1)
python3 scripts/plot_search_trajectory.py $LATEST_LOG --all
```

## Setting True Source Location

The script tries to find the true source location in this order:

1. From CSV columns `true_source_x` and `true_source_y` (if logged)
2. From the summary file `igdm_log_TIMESTAMP_summary.txt`
3. If not found, only shows estimated source

### To Add True Source to CSV Logging

If you want the true source to appear in plots, you can log it by modifying the ROS2 parameters:

```bash
ros2 run igdm start --ros-args \
  -p true_source_x:=2.0 \
  -p true_source_y:=4.5
```

Or in your launch file:

```python
parameters=[{
    'true_source_x': 2.0,
    'true_source_y': 4.5,
    # ... other parameters
}]
```

## Example Output

### Trajectory Plot
- Blue line: Local planner (RRT-Infotaxis) mode
- Red line: Global planner (frontier exploration) mode
- Green circle: Initial position
- Blue square: Final position
- Red star: Estimated source location
- Yellow star: True source location (if available)
- Red dashed line: Localization error

### Metrics Plot
- Shows entropy convergence over time
- Standard deviations approaching convergence threshold (0.5m)
- Gas sensor readings with adaptive threshold
- Red shaded regions indicate global planner mode

### Dead-End Analysis
- Blue line: Optimal branch information (BI*)
- Red dashed line: Adaptive dead-end threshold
- Red X marks: Dead-end detection events (when BI* < threshold)

## Dependencies

Make sure you have these Python packages installed:

```bash
pip3 install numpy matplotlib pandas pillow
```

## Comparing Multiple Runs

To compare multiple experimental runs (like in Figure 5), you can:

1. Generate trajectory plots for each run separately
2. Manually overlay them using image editing software, OR
3. Create a custom comparison script (let me know if you need this!)

## Troubleshooting

**Q: Plot shows no data or errors**
- Check that the CSV file is not corrupted
- Ensure the CSV has all expected columns (run with `-v` for verbose)

**Q: True source location not showing**
- Set ROS2 parameters `true_source_x` and `true_source_y` before running
- Or manually edit the plotting script to hardcode your source location

**Q: Map overlay doesn't align**
- Adjust the `_get_map_extent()` method to match your map's coordinate system
- Ensure map image resolution and coordinate frame match your robot's frame

**Q: Want to customize colors/styles**
- Edit the script directly - all plotting code is in `SearchTrajectoryPlotter` class
- Look for `ax.plot()` and `ax.scatter()` calls to change colors/markers

## Citation

If you use these plots in publications, consider citing the IGDM paper:

```
Kim et al., "Gas Source Localization in Unknown Indoor Environments Using
Dual-Mode Information-Theoretic Search", IEEE Robotics and Automation Letters,
Vol. 10, No. 1, January 2025
```
