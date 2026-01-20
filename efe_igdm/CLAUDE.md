# IGDM - Information Gain-based Dual-Mode Gas Source Localization

## Project Overview

This is a ROS2 package implementing **RRT-Infotaxis** for gas source localization. The implementation is based on the paper "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search" (PDF included in project root).

## Implementation Status

This package implements **RRT-Infotaxis with IGDM** for indoor gas source localization:
- **RRT-Infotaxis** - Information-theoretic path planning with hybrid utility
- **IGDM (Indoor Gaussian Dispersion Model)** - Obstacle-aware gas dispersion for indoor environments without airflow

## Purpose

The system enables an autonomous mobile robot to locate a gas source in an unknown environment by:
1. Using a binary gas sensor to detect presence/absence of gas
2. Employing a particle filter to estimate source location and release rate
3. Using RRT (Rapidly-exploring Random Tree) for path planning with information gain optimization
4. Implementing a measure-plan-move loop for continuous source search

## Architecture

### Main Components

#### 1. **RRTInfotaxisNode** ([igdm.py](igdm/igdm.py))
The main ROS2 node that orchestrates the entire search process.

**Key Responsibilities:**
- Receives robot pose and gas sensor readings
- Manages measure-plan-move loop
- Coordinates particle filter updates
- Triggers RRT path planning
- Sends navigation goals to Nav2
- Publishes visualization markers to RViz2

**Topics Subscribed:**
- `/PioneerP3DX/ground_truth` - Robot pose
- `/fake_pid/Sensor_reading` - Gas sensor readings

**Topics Published:**
- `/rrt_infotaxis/particles` - Particle visualization
- `/rrt_infotaxis/all_paths` - All RRT candidate paths
- `/rrt_infotaxis/best_path` - Selected best path
- `/rrt_infotaxis/estimated_source` - Estimated source location
- `/rrt_infotaxis/current_position` - Current robot position
- `/rrt_infotaxis/source_info_text` - Text information overlay

**Action Clients:**
- `/PioneerP3DX/navigate_to_pose` - Nav2 navigation

**Key Parameters:**
- `sigma_m` (default: 1.0) - IGDM constant standard deviation in meters
- `number_of_particles` (default: 1000) - Particle filter size
- `n_tn` (default: 30) - Number of RRT nodes
- `delta` (default: 1) - RRT step size in meters
- `xy_goal_tolerance` (default: 0.3) - Navigation goal tolerance
- `robot_radius` (default: 0.35) - Robot footprint radius
- `sigma_threshold` (default: 0.35) - Convergence threshold for estimation
- `success_distance` (default: 0.5) - Success distance to source
- `positive_weight` (default: 0.5) - Weight of information gain vs travel cost

#### 2. **ParticleFilterOptimized** ([particle_filter_optimized.py](igdm/particle_filter_optimized.py))
Optimized particle filter for Bayesian source term estimation.

**Features:**
- Vectorized operations for performance
- MCMC moves for particle diversity
- Adaptive resampling based on effective sample size
- Estimates source location (x, y) and release rate (Q)

**Key Methods:**
- `update(measurement, sensor_position)` - Update with binary sensor reading
- `predict(hypothetical_measurement, sensor_position)` - Predict for path planning
- `get_estimate()` - Get mean and std dev of estimated parameters

#### 3. **RRT** ([rrt.py](igdm/rrt.py))
Rapidly-exploring Random Tree for information-theoretic path planning.

**Features:**
- Generates N_tn random nodes within R_range
- Collision checking with robot radius
- Path pruning to find candidate paths
- Entropy gain calculation for each path
- Travel cost computation with discount factor

**Key Methods:**
- `sprawl(start_pos)` - Build RRT tree
- `prune()` - Extract candidate paths
- `calculate_entropy_gain(path, particle_filter)` - Compute information gain
- `get_next_move_debug(start_pos, particle_filter)` - Select best path and return debug info

#### 4. **IndoorGaussianDispersionModel (IGDM)** ([igdm_gas_model.py](igdm/igdm_gas_model.py))
Indoor Gaussian Dispersion Model for gas concentration prediction in obstacle-rich environments.

**Key Features:**
- Isotropic dispersion (no wind)
- Obstacle-aware distance via Dijkstra's algorithm
- Constant standard deviation σm
- Works with incomplete/online maps
- Optimized batch computation for particle filters

**Key Methods:**
- `compute_concentration(position, source_location, release_rate)` - Single concentration computation
- `compute_concentrations_batch(sensor_position, particle_locations, release_rates)` - Batch computation for all particles
- `compute_distance_map_from_sensor(sensor_position)` - Dijkstra distance map for optimization

#### 5. **BinarySensorModel** ([sensor_model.py](igdm/sensor_model.py))
Binary sensor model with adaptive threshold.

**Features:**
- Adaptive threshold that increases with higher measurements
- Probabilistic binary measurements
- Accounts for sensor noise

**Key Methods:**
- `update_threshold(measurement)` - Update adaptive threshold
- `get_binary_measurement(measurement)` - Convert to 0/1
- `probability_binary(binary_value, particle_concentration)` - Likelihood computation

#### 6. **OccupancyGridMap** ([occupancy_grid.py](igdm/occupancy_grid.py))
2D occupancy grid for collision checking and IGDM distance calculations.

**Features:**
- Loads from GADEN ROS2 service
- Grid-to-world coordinate conversion
- Collision checking with robot radius
- Used by IGDM for Dijkstra's algorithm
- Visualization support

#### 7. **TextVisualizer** ([text_visualizer.py](igdm/text_visualizer.py))
RViz2 text marker visualization for real-time status display.

**Displays:**
- Predicted source location
- Standard deviation
- Sensor readings
- Search status

## Algorithm Flow

### Measure-Plan-Move Loop

1. **MEASURE** (Line 447-456 in igdm.py)
   - Read gas sensor value
   - Update adaptive threshold
   - Convert to binary measurement (0 or 1)
   - Update particle filter with measurement

2. **PLAN** (Line 462-467 in igdm.py)
   - Generate RRT tree from current position
   - For each candidate path:
     - Predict particle filter evolution along path
     - Calculate entropy reduction (information gain)
     - Calculate travel cost
     - Compute utility = entropy_gain - weight * travel_cost
   - Select path with highest utility

3. **MOVE** (Line 513-515 in igdm.py)
   - Send first node of best path as navigation goal to Nav2
   - Wait for goal to be reached (with XY tolerance)
   - Repeat loop

### Termination Condition

Search completes when **estimation converges**: max(σ_x, σ_y) < σ_threshold

This indicates the particle filter has sufficiently localized the source.

## File Structure

```
igdm/
├── igdm/
│   ├── __init__.py                    # Package initialization
│   ├── igdm.py                        # Main ROS2 node
│   ├── particle_filter_optimized.py   # Optimized particle filter (IGDM-enabled)
│   ├── particle_filter.py             # Original particle filter (legacy)
│   ├── rrt.py                         # RRT path planning
│   ├── igdm_gas_model.py              # Indoor Gaussian Dispersion Model
│   ├── gaussian_plume.py              # Gaussian plume model (unused, kept for reference)
│   ├── sensor_model.py                # Binary sensor model
│   ├── occupancy_grid.py              # Occupancy grid management
│   ├── text_visualizer.py             # RViz text visualization
│   ├── dijkstra.py                    # Dijkstra pathfinding (unused)
│   └── test_ideal.py                  # Testing utilities
├── test/                              # Unit tests
├── resource/                          # ROS2 resources
├── package.xml                        # ROS2 package manifest
├── setup.py                           # Python package setup
└── Gas_Source_Localization_in_Unknown_Indoor_Environments_Using_Dual-Mode_Information-Theoretic_Search.pdf
```

## Dependencies

### ROS2 Dependencies
- `rclpy` - ROS2 Python client library
- `geometry_msgs` - Pose messages
- `visualization_msgs` - Marker messages for RViz
- `nav2_msgs` - Navigation action interface
- `olfaction_msgs` - Gas sensor messages
- `gaden_msgs` - GADEN occupancy service

### Python Dependencies
- `numpy` - Numerical computing
- `scipy` - Statistical functions
- `matplotlib` - Plotting (for occupancy grid)

## Key Algorithms

### 1. Particle Filter Update (Algorithm 2 in paper)
- Compute likelihood for each particle based on gas model
- Update weights proportionally to likelihood
- Normalize weights
- Resample if effective sample size drops
- Apply MCMC moves for diversity

### 2. RRT Path Planning
- Build random tree within R_range
- Extract max_depth paths
- For each path, predict measurements and compute entropy
- Select path maximizing: entropy_gain - positive_weight * travel_cost

### 3. Binary Sensor Model
- Adaptive threshold c̄_k increases when measurement exceeds it
- Binary measurement: 1 if measurement > threshold, else 0
- Likelihood: P(b|θ) = β if b=0, 1-β if b=1
  - Where β = Φ((threshold - predicted_concentration) / σ)

## Important Notes

### Current Implementation Details
1. **Particle filter**: Uses optimized version with vectorization
2. **Coordinate system**: Standard ROS2 (X forward, Y left, Z up)
3. **Map frame**: All positions in "map" frame
4. **Nav2 integration**: Uses NavigateToPose action with XY-only tolerance
5. **Visualization**: All markers published to RViz2 in "map" frame

### Performance Considerations
- Particle filter is main computational bottleneck
- Vectorized operations significantly improve performance
- Cache concentration computations during RRT planning
- Typical performance: ~5 Hz update rate with 1000 particles, 30 RRT nodes

### Git Status (as of 2025-10-28)
- Current branch: `main`
- Modified: `../rrt_infotaxis/rrt_infotaxis/rrt_infotaxis.py`
- Modified: `../rrt_infotaxis/rrt_infotaxis/text_visualizer.py`
- Untracked: `./` (this igdm package)

## Entry Point

**Console script**: `start`
- Command: `ros2 run igdm start`
- Entry point: `igdm.igdm:main`

## Indoor Gaussian Dispersion Model (IGDM)

The core gas dispersion model for indoor environments without airflow.

### IGDM Model ([igdm_gas_model.py](igdm/igdm_gas_model.py))

**Model Equation (Paper Equation 18):**
```
R(rk|θ) = Qm · exp(-cobs(rk, r0)² / (2σm²))
```

Where:
- **Qm**: Release rate (g/s converted to μg/s)
- **cobs**: Obstacle-aware distance computed via Dijkstra's algorithm on occupancy grid
- **σm**: Constant standard deviation (tunable parameter)

**Key Features:**
- ✅ Isotropic dispersion (no wind direction)
- ✅ Obstacle-aware distance using Dijkstra's algorithm
- ✅ Constant dispersion parameter σm
- ✅ Works with incomplete/online maps
- ✅ Optimized batch computation for particle filters
- ✅ Single Dijkstra execution computes distances to all particles

**Usage:**
```python
from igdm.igdm_gas_model import IndoorGaussianDispersionModel

# Create model (automatically used by particle filter)
igdm = IndoorGaussianDispersionModel(sigma_m=1.0, occupancy_grid=occupancy_map)

# Single concentration computation
concentration = igdm.compute_concentration(
    position=(x, y),
    source_location=(x0, y0),
    release_rate=Q0
)

# Batch computation for particle filter (optimized)
concentrations = igdm.compute_concentrations_batch(
    sensor_position=(x, y),
    particle_locations=particles[:, :2],  # (N, 2)
    release_rates=particles[:, 2]         # (N,)
)
```

**Why IGDM for Indoor Environments?**

Traditional outdoor models (Gaussian Plume) assume:
- Wind-driven dispersion (downwind/crosswind)
- Open space with no obstacles
- Euclidean distance calculations

IGDM is designed for indoor scenarios:
- No external wind (closed environment)
- Complex obstacles (walls, furniture)
- Gas travels along accessible paths

### Implemented Components:

✅ RRT tree generation with collision checking
✅ Particle filter for source estimation (IGDM-optimized)
✅ Information gain (entropy reduction) calculation
✅ Binary sensor model with adaptive threshold
✅ Indoor Gaussian Dispersion Model (IGDM)
✅ RViz visualization
✅ Nav2 integration

## Future Improvements

Potential enhancements:
1. Dual-mode planner with dead end detection and global planner
2. Multi-source localization
3. Adaptive particle count based on uncertainty
4. Dynamic wind estimation
5. Hybrid IGDM/Plume model for mixed environments
6. Real sensor calibration and noise modeling
7. Multi-robot collaborative search

## References

Paper: "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search"
- Kim et al., IEEE Robotics and Automation Letters, Vol. 10, No. 1, January 2025

Implementation combines:
- RRT-Infotaxis with hybrid utility function
- Indoor Gaussian Dispersion Model (IGDM) for obstacle-aware gas modeling
- Optimized particle filter for real-time performance
