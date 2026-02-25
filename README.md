# Autonomous Gas Source Localization on GADEN

A ROS2 research platform for autonomous gas source localization (GSL) in unknown indoor environments. This repository implements and compares multiple information-theoretic search algorithms using the [GADEN](https://github.com/MAPIRlab/gaden) gas dispersion simulator and a simulated Pioneer P3DX robot.

## Overview

Finding gas leak sources in indoor environments is challenging due to turbulent airflow, obstacles, and the absence of stable concentration gradients. This project implements several algorithms that guide a mobile robot to autonomously locate gas sources by maximizing information gain at each step.

### Algorithms Implemented

| Algorithm | Package | Based On | Description |
|-----------|---------|----------|-------------|
| **Dual-Mode IGDM** | `efe_igdm` | [Kim et al., 2025](https://ieeexplore.ieee.org/document/10777609/) | Indoor Gaussian Dispersion Model with dual-mode planning (local RRT-Infotaxis + global PRM frontier exploration) |
| **RRT-Infotaxis** | `rrt_infotaxis` | [Park & Cho, 2022](https://www.sciencedirect.com/science/article/pii/S1270963821007860) | Local development package for experimenting with RRT-Infotaxis variants and improvements |
| **Classical Infotaxis** | `infotaxis` | [Vergassola et al., 2007](https://www.nature.com/articles/nature05464) | Grid-based entropy-minimization search |
| **Ali's IGDM** | `ali_igdm` | — | ROS2 port of the best-performing algorithm from the `rrt_infotaxis` experiments |

## Repository Structure

```
base/
├── efe_igdm/              # Dual-mode IGDM system
│   ├── efe_igdm/
│   │   ├── igdm.py        # Main ROS2 node (coordinator)
│   │   ├── estimation/    # Particle filter, IGDM gas model, sensor model
│   │   ├── planning/      # RRT-Infotaxis, PRM global planner, dead-end detector
│   │   ├── mapping/       # Occupancy grid, LiDAR mapper, wind field estimation
│   │   ├── visualization/ # RViz marker publishers
│   │   └── utils/         # Experiment logger
│   ├── launch/
│   ├── config/
│   └── scripts/           # Plotting and analysis tools
│
├── rrt_infotaxis/         # RRT-Infotaxis with IGDM variants
│   ├── rrt_infotaxis/     # Core algorithm
│   ├── igdm/              # Basic IGDM integration
│   ├── igdm_improved/     # Enhanced variants (rooms, large maps, adaptive)
│   └── igdm_time_weighted/# Time-weighted variants
│
├── infotaxis/             # Classical infotaxis implementation
│   └── infotaxis/         # Grid-based infotaxis node
│
├── ali_igdm/              # Experimental IGDM variant
│
└── wind_visualizer/       # Wind field visualization utility
```

## Dependencies

### System

- **ROS2** (Humble or later)
- **Nav2** (navigation stack)
- **GADEN** simulator ([gaden](https://github.com/MAPIRlab/gaden)) with `gaden_msgs` and `olfaction_msgs`

### Python

- NumPy
- SciPy
- Matplotlib
- Numba (for JIT-accelerated particle filter operations)

### ROS2 Packages

Standard ROS2 packages:
- `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `visualization_msgs`, `nav2_msgs`

Installed automatically with GADEN:
- `gaden_msgs`, `olfaction_msgs`

## Installation

### Prerequisites

1. **ROS2 Humble** (or later) — [installation guide](https://docs.ros.org/en/humble/Installation.html)
2. **GADEN simulator** — follow the [GADEN installation instructions](https://github.com/MAPIRlab/gaden) and build it in your ROS2 workspace before proceeding.

### Build

```bash
# Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/Boun-Final-Project/base.git

# Install Python dependencies
pip install numpy scipy matplotlib numba

# Build
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Usage

### 1. Start GADEN Environment

Launch the GADEN simulation in three steps:

**Step 1 — Preprocessing:** Generate the wind field and occupancy grid.
```bash
ros2 launch test_env gaden_preproc_launch.py scenario:=<scenario_name> simulation:=<simulation_name>
```

**Step 2 — Gas simulation:** Run the filament-based gas dispersion simulation. Wait for it to finish before proceeding.
```bash
ros2 launch test_env gaden_sim_launch.py scenario:=<scenario_name> simulation:=<simulation_name>
```

**Step 3 — Robot & environment:** Spawn the robot and start the environment player.
```bash
ros2 launch test_env main_simbot_launch.py scenario:=<scenario_name> simulation:=<simulation_name>
```

### 2. Run a GSL Algorithm

**Dual-Mode IGDM** (recommended):
```bash
# Standard mode
ros2 run efe_igdm start

# Or via launch file
ros2 launch efe_igdm igdm_launch.py

# Simplified variant (no wind estimation)
ros2 run efe_igdm start_basic
```

**Classical Infotaxis:**
```bash
ros2 run infotaxis infotaxis_node
```

**RRT-Infotaxis:**
```bash
ros2 run rrt_infotaxis start
```

**Wind Visualizer:**
```bash
ros2 launch wind_visualizer wind_visualizer.launch.py
```

### 3. Visualization

Open RViz2 and add displays for the topics published by the running algorithm:

| Topic | Type | Description |
|-------|------|-------------|
| `/igdm/particles` | MarkerArray | Particle filter belief distribution |
| `/igdm/rrt_tree` | MarkerArray | RRT search tree |
| `/igdm/frontiers` | MarkerArray | Frontier exploration targets |
| `/igdm/estimated_source` | Marker | Most likely source location |
| `/infotaxis/probability_map` | OccupancyGrid | Source probability heatmap |

### 4. Post-Processing

Generate trajectory plots and analysis figures:

```bash
# Plot search trajectory with metrics
python3 src/base/efe_igdm/scripts/plot_search_trajectory.py

# Plot entropy over time
python3 src/base/efe_igdm/scripts/plot_entropy.py
```

## Architecture (efe_igdm)

The primary `efe_igdm` package uses a modular architecture:

```
                    ┌─────────────────┐
                    │   igdm.py       │
                    │  (Coordinator)  │
                    └────────┬────────┘
           ┌─────────┬──────┴──────┬───────────┐
           v         v             v            v
    ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
    │ Estimation │ │ Planning │ │ Mapping  │ │Visualization│
    ├────────────┤ ├──────────┤ ├──────────┤ ├─────────────┤
    │ Particle   │ │ RRT      │ │Occupancy │ │ RViz Markers│
    │ Filter     │ │ Infotaxis│ │ Grid     │ │ Text Overlay│
    │ IGDM Model │ │ PRM      │ │ LiDAR    │ └─────────────┘
    │ Sensor     │ │ Frontier │ │ Mapper   │
    │ Model      │ │ Dead-End │ │ Wind Map │
    └────────────┘ │ Detector │ │ GMRF     │
                   │ Navigator│ └──────────┘
                   └──────────┘
```

## References

- Kim, H. et al., "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search," *IEEE Robotics and Automation Letters*, 2025. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10777609/)
- Park, S. & Cho, H., "Receding-horizon RRT-Infotaxis for autonomous source search in urban environments," *Aerospace Science and Technology*, 2022. [[ScienceDirect]](https://www.sciencedirect.com/science/article/pii/S1270963821007860)
- Vergassola, M., Villermaux, E. & Shraiman, B. I., "'Infotaxis' as a strategy for searching without gradients," *Nature*, 445(7126), 406-409, 2007. [[Nature]](https://www.nature.com/articles/nature05464)

## Contributors

- **Efe Mantaroglu** — efemantaroglu@gmail.com
- **Ali Sonmez** — al1.sonmez.mi@gmail.com
- **Simal** — (transferred to another project)

## License

MIT
