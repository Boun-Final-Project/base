# Capturing GADEN Gas Distribution and Map for Plotting

There are several methods to capture the map and gas distribution from GADEN simulator:

## Method 1: Export Map from ROS2 (Recommended for Map)

While your simulation is running:

```bash
cd ~/ros2_ws/src/base/efe_igdm

# Export the occupancy grid map
python3 scripts/export_map_from_ros.py --output ~/gaden_map.png
```

This will create:
- `~/gaden_map.png` - The map image
- `~/gaden_map_metadata.txt` - Metadata (resolution, origin, etc.)

Then use it in plotting:

```bash
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv \
  --map ~/gaden_map.png --all
```

## Method 2: RViz2 Screenshot (Best for Gas Distribution)

### For Gas Concentration Visualization:

1. **In RViz2**, set up your view:
   - Add GADEN gas concentration markers/visualization
   - Add occupancy grid map
   - Adjust camera to top-down view (2D)
   - Hide robot model if needed

2. **Take screenshot**:
   - Method A: Built-in RViz screenshot
     - Go to RViz2 menu: `Panels` → `Screenshot`
     - Or click the camera icon in toolbar

   - Method B: Use ROS2 image saver
     ```bash
     # Save RViz camera view
     ros2 run image_view image_saver --ros-args -r image:=/rviz_camera/image
     ```

3. **Save with transparent background** (optional):
   - In RViz2, set background to white/transparent
   - Use image editing software to make white transparent

### Setting up Top-Down View in RViz2:

1. Select **"TopDownOrtho"** or **"Orbit"** view controller
2. Set camera:
   - Pitch: -90° (looking straight down)
   - Yaw: 0°
   - Distance: Adjust to fit your environment
3. Disable grid if needed for cleaner visualization

## Method 3: Export from GADEN Directly

GADEN can export gas concentration data. Check your GADEN configuration:

### Using GADEN Player:

```bash
# GADEN publishes gas concentration as markers
# You can record these and visualize offline

# Record gas markers during simulation
ros2 bag record /gaden/gas_markers /map

# Then replay and screenshot
ros2 bag play your_recording.db3
```

## Method 4: Create Gas Concentration Heatmap (Advanced)

If you want to programmatically generate gas distribution overlays:

```python
# Subscribe to GADEN gas concentration topic
# Process the gas data into a 2D heatmap
# Overlay on map image
# This requires knowing GADEN's topic structure
```

Would you like me to create this script?

## Recommended Workflow for Publication Figures

### 1. **During Simulation:**

```bash
# Terminal 1: Run GADEN simulation
ros2 launch your_gaden_launch_file

# Terminal 2: Run IGDM
ros2 run igdm start --ros-args -p true_source_x:=2.0 -p true_source_y:=4.5

# Terminal 3: Export map (once environment is built)
python3 scripts/export_map_from_ros.py --output ~/env_a_map.png
```

### 2. **Take RViz Screenshot:**

- Set up RViz2 with gas concentration visualization
- Take top-down screenshot
- Save as `~/env_a_gas_distribution.png`

### 3. **After Simulation:**

```bash
# Generate trajectory plots with map overlay
python3 scripts/plot_search_trajectory.py ~/igdm_logs/igdm_log_TIMESTAMP.csv \
  --map ~/env_a_map.png --all
```

### 4. **Post-Processing (Optional):**

Use image editing software (GIMP, Photoshop, or Python) to:
- Overlay gas distribution on map
- Composite trajectory plot on top
- Add labels and annotations

## Quick RViz2 Screenshot Setup for GADEN

Add to your RViz2 config:

```yaml
Displays:
  - Class: rviz_default_plugins/Map
    Topic: /map

  - Class: rviz_default_plugins/MarkerArray
    Topic: /gaden/gas_markers  # or your GADEN marker topic

  - Class: rviz_default_plugins/Path
    Topic: /rrt_infotaxis/best_path

Views:
  Current:
    Class: rviz_default_plugins/TopDownOrtho
    Scale: 10
```

## What Topics Does GADEN Publish?

To find GADEN gas visualization topics:

```bash
# List all active topics
ros2 topic list | grep -i gas

# Check marker topics
ros2 topic list | grep -i marker

# Common GADEN topics:
# /gaden/gas_markers
# /gaden/concentration
# /PioneerP3DX/Nose/Sensor_reading
```

## Example: Combining Map + Gas + Trajectory

I can create a script that:
1. Loads the map PNG
2. Loads gas concentration data (if available from GADEN logs)
3. Overlays trajectory from your CSV
4. Creates Figure 5-style composite

Would you like me to create this comprehensive visualization script?
