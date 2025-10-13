# Infotaxis Package

## Running the System

### Step 1: Start GADEN Environment

First, launch the GADEN preprocessing and simulation environment:

```bash
ros2 launch test_env gaden_preproc_launch.py scenario:=10x6_empty_room simulation:=uniform_infotaxis
```

This command:
- Loads the `10x6_empty_room` scenario
- Uses the `uniform_infotaxis` simulation configuration
- Starts the gas dispersion simulation

Wait until you see messages indicating that GADEN is ready and publishing gas concentration data.

### Step 2: Run Infotaxis Node

In a new terminal, start the infotaxis node:

```bash
ros2 run infotaxis infotaxis_node
```

The node will:
- Request the occupancy grid from GADEN
- Initialize the probability distribution
- Subscribe to gas sensor readings
- Begin the source localization process
- Publish probability maps for visualization

You should see log messages showing:
- Current position and entropy
- Best moves being selected
- Expected entropy decrease values

### Optional Parameters

You can customize the behavior with parameters:

```bash
ros2 run infotaxis infotaxis_node --ros-args \
  -p detection_threshold:=1.0 \
  -p step_size:=0.5 \
  -p robot_namespace:=/PioneerP3DX \
  -p infotaxis_update_interval:=0.1
```

Parameters:
- `detection_threshold`: Gas concentration threshold for detection (default: 1.0)
- `robot_namespace`: Robot's namespace (default: /PioneerP3DX)
- `infotaxis_update_interval`: Time between infotaxis updates in seconds (default: 0.1)

## Visualization

### Real-time Visualization with RViz2

The infotaxis node publishes the probability distribution as an `OccupancyGrid` on the `/infotaxis/probability_map` topic. You can visualize this in RViz2:

Add a Map display:
   - Click "Add" button
   - Select "Map" type
   - Set topic to `/infotaxis/probability_map`
   - The probability map will show hotspots where the source is most likely located

### Post-Processing Visualization

For detailed trajectory analysis, use the visualization script:

```bash
ros2 run infotaxis visualize_infotaxis
```

This script:
- Subscribes to robot position (`/PioneerP3DX/ground_truth`)
- Subscribes to gas sensor readings (`/fake_pid/Sensor_reading`)
- Subscribes to probability maps (`/infotaxis/probability_map`)
- Collects data during the search

Press `Ctrl+C` to stop data collection. The script will automatically generate:
- **Trajectory plot** overlaid on the final probability distribution
- **Gas detection markers** showing where hits occurred
- **Probability heatmap** highlighting the most probable source location
- **Trajectory-only view** with detection markers

The visualization is saved to `/tmp/infotaxis_visualization.png` and displayed interactively.

#### Visualization Features

The generated plots show:
- **White trajectory line**: Robot's path during search
- **Cyan diamonds**: Locations where gas was detected
- **Green circle**: Starting position
- **Red square**: Final position
- **Yellow star**: Most probable source location (on heatmap)
- **Hot colormap**: Probability intensity (red = high probability)

## Algorithm Parameters

The infotaxis algorithm uses the following physical parameters (defined in infotaxis_node.py:29-36):

- `dt = 0.1`: Time step duration in seconds
- `src_radius = 0.2`: Detection radius (must be within 20cm of source)
- `w = 0.1`: Mean wind speed in m/s (positive x-direction)
- `d = 0.02`: Turbulent diffusivity coefficient in m²/s
- `r = 10`: Source emission rate in Hz
- `a = 0.003`: Sensor size in meters
- `tau = 100`: Particle lifetime in seconds

These parameters define the plume model used for Bayesian inference. Adjust them in the source code if your simulation uses different physical characteristics.

## Topics

### Subscribed Topics
- `/fake_pid/Sensor_reading` (olfaction_msgs/GasSensor): Gas sensor readings
- `/PioneerP3DX/ground_truth` (geometry_msgs/PoseWithCovarianceStamped): Robot position

### Published Topics
- `/infotaxis/probability_map` (nav_msgs/OccupancyGrid): Source probability distribution (5 Hz)

## Architecture

The package is organized into several modules:

- **infotaxis_node.py**: Main node implementing the algorithm
- **grid_manager.py**: Handles occupancy grid and probability distributions
- **navigation_controller.py**: Controls robot movement and teleportation
- **sensor_handler.py**: Processes gas sensor readings
- **visualize_infotaxis.py**: Post-processing visualization tool

## Stopping Condition

The search automatically stops when the probability of the source being within `src_radius` (0.2m) exceeds 50%. At this point, the source is considered localized and the robot stops moving.

## References

The Infotaxis algorithm is based on:
- Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). 'Infotaxis' as a strategy for searching without gradients. *Nature*, 445(7126), 406-409.
- [Supplementary Materials](http://www.nature.com/nature/journal/v445/n7126/extref/nature05464-s1.pdf)
