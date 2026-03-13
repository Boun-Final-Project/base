# Complete Guide: Creating Figure 5-Style Plots with GADEN

This guide shows you how to create publication-quality trajectory plots with gas distribution visualization from your GADEN simulations.

## Quick Start (3 Steps)

### 1. Run Simulation with True Source

```bash
# Your env_a has source at (2.5, 4.5)
ros2 run igdm start --ros-args \
  -p true_source_x:=2.5 \
  -p true_source_y:=4.5
```

### 2. Capture Visualization (While Running)

**Open RViz2 with pre-configured view:**
```bash
rviz2 -d ~/ros2_ws/src/base/efe_igdm/config/rviz_topdown_capture.rviz
```

**Take screenshot:**
- Click camera icon in toolbar
- Save as: `~/env_a_visualization.png`

### 3. Generate Plots (After Simulation)

```bash
cd ~/ros2_ws/src/base/efe_igdm

# Plot with background map
./scripts/plot_latest.sh ~/env_a_visualization.png

# Or manually specify log file
python3 scripts/plot_search_trajectory.py \
  ~/igdm_logs/igdm_log_20260107_HHMMSS.csv \
  --map ~/env_a_visualization.png \
  --all
```

Done! Your plots are in `~/igdm_logs/`

---

## Detailed Instructions

### Capturing Gas Distribution from GADEN

#### Method 1: RViz Screenshot (Recommended)

**While your simulation is running:**

1. **Launch RViz with config:**
   ```bash
   rviz2 -d ~/ros2_ws/src/base/efe_igdm/config/rviz_topdown_capture.rviz
   ```

2. **Adjust view if needed:**
   - Views panel → TopDownOrtho
   - Adjust scale to fit your environment
   - Center view on your environment

3. **Configure displays:**
   - Enable/disable layers as needed:
     - ✅ Occupancy Grid (`/map`)
     - ✅ Gas Concentration (`/gaden_player/gas_concentration`)
     - ⬜ Particles (optional, can be noisy)
     - ⬜ Robot model (hide for cleaner view)

4. **Take screenshot:**
   - Toolbar camera icon OR
   - `Panels` → `Screenshot` OR
   - `Ctrl+P` (if available)

5. **Save as:** `~/env_a_gas_distribution.png`

#### Method 2: Export Map Only

If you just want the occupancy grid without gas:

```bash
# While simulation is running
python3 scripts/export_map_from_ros.py --output ~/env_a_map.png
```

This creates:
- `env_a_map.png` - The map image
- `env_a_map_metadata.txt` - Resolution, origin info

### Setting True Source Location

Your GADEN sim files have the true source location. Use these values:

**For env_a** (`gaden_maps/env_a/simulations/sim1.yaml`):
```bash
ros2 run igdm start --ros-args -p true_source_x:=2.5 -p true_source_y:=4.5
```

**For env_b** (check your `sim1.yaml`):
```bash
ros2 run igdm start --ros-args -p true_source_x:=X.X -p true_source_y:=Y.Y
```

When you set these parameters:
- ✅ True source appears as yellow star in plots
- ✅ Localization error is automatically calculated
- ✅ Error is displayed in trajectory plot

### Plotting Options

#### Quick Plot (Latest Run)

```bash
# No map background
./scripts/plot_latest.sh

# With map background
./scripts/plot_latest.sh ~/env_a_gas_distribution.png
```

#### Specific Log File

```bash
# All plots with map
python3 scripts/plot_search_trajectory.py \
  ~/igdm_logs/igdm_log_20260107_022720.csv \
  --map ~/env_a_gas_distribution.png \
  --all

# Just trajectory
python3 scripts/plot_search_trajectory.py \
  ~/igdm_logs/igdm_log_20260107_022720.csv \
  --trajectory

# Just metrics
python3 scripts/plot_search_trajectory.py \
  ~/igdm_logs/igdm_log_20260107_022720.csv \
  --metrics
```

#### Custom Title

```bash
python3 scripts/plot_search_trajectory.py \
  ~/igdm_logs/igdm_log_20260107_022720.csv \
  --trajectory \
  --title "Environment A - Trial 1 - RRT-Infotaxis with IGDM"
```

---

## Output Files

After running `--all`, you get three plots:

### 1. `*_trajectory.png` - Search Path Visualization
- Blue line: Local planner (RRT-Infotaxis)
- Red line: Global planner (frontier exploration)
- Orange dots: Estimated source evolution
- Red star: Final estimated source
- Yellow star: True source (if set)
- Green circle: Start position
- Blue square: End position

### 2. `*_metrics.png` - Performance Metrics
- Entropy evolution
- Standard deviations (convergence)
- Sensor readings
- Red shaded = global planner mode

### 3. `*_deadend.png` - Dead-End Detection
- BI* (optimal branch information)
- Adaptive threshold
- Detection events (red X marks)

---

## Tips for Best Results

### For Publication-Quality Figures:

1. **High-resolution RViz screenshots:**
   - Maximize RViz window before screenshot
   - Or use image upscaling tools

2. **Clean visualization:**
   - Hide robot model
   - Set white background
   - Disable grid
   - Only show relevant layers

3. **Consistent camera view:**
   - Save RViz config for each environment
   - Use same scale/angle for all trials
   - Center on same region

4. **Multiple trials:**
   - Run same scenario multiple times
   - Save each trajectory separately
   - Overlay in post-processing

### Comparing Multiple Methods (Like Figure 5):

To create multi-method comparison plots:

1. Run different methods separately
2. Generate trajectory plots for each
3. Use image editing software to composite, OR
4. Ask me to create a comparison plotting script!

---

## Troubleshooting

**Q: Gas concentration not showing in RViz**
```bash
# Check if GADEN is publishing
ros2 topic list | grep -i gas
ros2 topic echo /gaden_player/gas_concentration --once

# Make sure GADEN player is running
ros2 node list | grep -i gaden
```

**Q: Map not showing**
```bash
# Check map topic
ros2 topic echo /map --once

# Build map first with SLAM/exploration before taking screenshot
```

**Q: Screenshot button not in RViz**
```bash
# Alternative: Use system screenshot
# Linux: Shift+PrtScn (select region)
# Or use scrot/flameshot
```

**Q: Map extent doesn't match trajectory**
- Check map metadata file
- Adjust `_get_map_extent()` in plotting script
- Or use RViz screenshot which has everything aligned

---

## Complete Example Workflow

```bash
# Terminal 1: Start GADEN simulation
cd ~/ros2_ws
source install/setup.bash
ros2 launch your_gaden_launch.py

# Terminal 2: Start IGDM with true source
ros2 run igdm start --ros-args \
  -p true_source_x:=2.5 \
  -p true_source_y:=4.5

# Terminal 3: Open RViz for visualization
rviz2 -d ~/ros2_ws/src/base/efe_igdm/config/rviz_topdown_capture.rviz

# During simulation:
# - Take RViz screenshot → save as ~/env_a_trial1.png

# After simulation completes:
cd ~/ros2_ws/src/base/efe_igdm
./scripts/plot_latest.sh ~/env_a_trial1.png

# View results:
eog ~/igdm_logs/igdm_log_*_trajectory.png
```

---

## Next Steps

- ✅ Basic trajectory plots - **DONE**
- ⬜ Multi-trial comparison plots - Want this?
- ⬜ Animated trajectory over time - Want this?
- ⬜ Statistical analysis across trials - Want this?

Let me know what else you need!
