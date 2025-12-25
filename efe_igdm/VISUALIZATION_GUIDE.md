# RViz Visualization Guide for RRT-Infotaxis IGDM

This guide explains all the visualization elements in RViz2 for the gas source localization system.

## Overview

The system publishes multiple visualization topics that help you understand the robot's decision-making process in real-time.

## Visualization Topics

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/rrt_infotaxis/particles` | MarkerArray | Particle filter particles with weights |
| `/rrt_infotaxis/all_paths` | MarkerArray | All RRT candidate paths (color-coded by utility) |
| `/rrt_infotaxis/best_path` | Marker | Selected best path (highlighted) |
| `/rrt_infotaxis/estimated_source` | Marker | Current source estimate |
| `/rrt_infotaxis/current_position` | Marker | Robot's current position |
| `/rrt_infotaxis/source_info_text` | MarkerArray | Text overlay with status info |

## Color Coding

### 1. Particle Filter Particles

**Visualization**: Colored spheres

**Color Scheme**: Blue → Yellow (viridis-like)
- **Blue spheres**: Low weight particles (less likely source locations)
- **Yellow spheres**: High weight particles (more likely source locations)
- **Sphere size**: 0.1m diameter
- **Height**: 0.5m above ground

**Interpretation**:
- Clusters of yellow particles indicate high-confidence regions
- Spread-out particles indicate high uncertainty
- As search progresses, particles should converge to the true source

### 2. RRT Candidate Paths (All Paths)

**Visualization**: Line strips with color gradient

**Color Scheme**: 🔴 Red → 🟡 Yellow → 🟢 Green
- **🔴 RED**: Low utility paths (least informative)
- **🟡 YELLOW**: Medium utility paths
- **🟢 GREEN**: High utility paths (most informative)

**Details**:
- **Line width**: 0.08m
- **Opacity**: 0.7 (semi-transparent)
- **Height**: 0.5m above ground
- **Normalization**: Colors are relative to min/max utility in current step

**Interpretation**:
- Green paths indicate directions with high expected information gain
- Red paths indicate directions with low information value
- Multiple green paths suggest multiple good options
- Only red paths might indicate approaching a dead end

### 3. Best Path (Selected Path)

**Visualization**: Thick blue line

**Color**: 🔵 **Cyan/Blue** (RGB: 0.0, 0.5, 1.0)
- **Line width**: 0.20m (thicker than other paths)
- **Opacity**: 1.0 (fully opaque)
- **Height**: 0.6m (slightly higher than other paths for visibility)

**Interpretation**:
- This is the path the robot will follow
- Should typically correspond to one of the green (high utility) paths
- First node of this path is sent as navigation goal

### 4. Estimated Source Location

**Visualization**: Orange sphere

**Color**: 🟠 **Orange** (RGB: 1.0, 0.65, 0.0)
- **Sphere size**: 0.4m diameter
- **Opacity**: 1.0 (fully opaque)
- **Height**: 0.5m above ground

**Interpretation**:
- Weighted mean of all particle locations
- Should converge toward true source location
- Standard deviation shown in text overlay

### 5. Current Robot Position

**Visualization**: Green sphere

**Color**: 🟢 **Green** (RGB: 0.0, 1.0, 0.0)
- **Sphere size**: 0.4m diameter
- **Opacity**: 1.0 (fully opaque)
- **Height**: 0.5m above ground

**Interpretation**:
- Robot's current position in map frame
- Updated from `/PioneerP3DX/ground_truth` topic

## Text Overlay Information

**Location**: Top-right corner of RViz (configurable)

**Default position**: (8.0m, 5.5m, 1.5m) in map frame

### Display Format

```
Predicted Source:
  x: [x] m
  y: [y] m
  z: [z] m
Std Dev: [σ_p]
Entropy: [H]
Sensor: [value]
Binary: [0/1]
Threshold: [c_threshold]
--- Branch Info (BI) ---
Branches: [N]
Tree Nodes: [N_tn]
Best Utility: [U]
  J1 (Entropy): [I]
  J2 (Cost): [C]
--- Dead End Detect ---
BI*: [optimal BI]
Threshold: [BI_thresh]
Margin: [+/- difference]
Status: [OK / ⚠ DEAD END!]
Search: [SEARCHING / COMPLETE]
```

### Field Descriptions

| Field | Description | Units |
|-------|-------------|-------|
| Predicted Source x,y,z | Estimated source location (mean of particles) | meters |
| Std Dev | Max of σ_x and σ_y (estimation uncertainty) | meters |
| Entropy | Shannon entropy of particle distribution | bits |
| Sensor | Raw sensor reading (converted to μg/m³) | μg/m³ |
| Binary | Binary sensor output (0 = below threshold, 1 = above) | - |
| Threshold | Adaptive binary threshold | μg/m³ |
| Branches | Number of valid RRT paths found | count |
| Tree Nodes | Total nodes in RRT tree | count |
| Best Utility | Utility of selected path | - |
| J1 (Entropy) | Information gain component | bits |
| J2 (Cost) | Travel cost component | meters |
| BI* | Optimal branch information (max utility) | - |
| BI Threshold | Dead end detection threshold | - |
| Margin | BI* - BI_thresh (positive = OK, negative = dead end) | - |
| Status | Dead end detector status | - |
| Search | Overall search status | - |

### Status Indicators

**Dead End Status**:
- `OK`: Normal operation, BI* > threshold
- `⚠ DEAD END!`: Dead end detected, BI* < threshold

**Search Status**:
- `SEARCHING`: Active source search
- `COMPLETE`: Search converged (σ_p < σ_threshold)

## Dead End Detection Visual Indicators

When a **dead end is detected**:

1. **Text Overlay**: Shows `Status: ⚠ DEAD END!`
2. **Margin**: Shows negative value (BI* < threshold)
3. **Path Colors**: Most/all paths will be RED (low utility)
4. **Console**: Warning message logged

**What this means**:
- Local RRT planner cannot find informative paths
- Robot may be stuck in a room or corner
- Global planner should activate (when implemented)

## Interpreting the Visualization

### Healthy Search Behavior

✅ **Good signs**:
- Particles converging toward source
- Multiple GREEN paths available
- Best path is GREEN or YELLOW
- BI* > BI_threshold (positive margin)
- Entropy decreasing over time
- Standard deviation decreasing

### Dead End Situation

⚠️ **Warning signs**:
- Most/all paths are RED
- BI* < BI_threshold (negative margin)
- Dead End status shows "⚠ DEAD END!"
- Few or no valid RRT branches
- Robot not making progress toward source

### Approaching Convergence

🎯 **Success indicators**:
- Particles tightly clustered
- Std Dev < 0.6m (default threshold)
- Estimated source stable
- Search status changes to "COMPLETE"

## RViz Configuration

### Recommended Display Settings

Add these displays to your RViz config:

```yaml
Displays:
  - Class: rviz/MarkerArray
    Name: Particles
    Topic: /rrt_infotaxis/particles

  - Class: rviz/MarkerArray
    Name: All Paths (Color-Coded)
    Topic: /rrt_infotaxis/all_paths

  - Class: rviz/Marker
    Name: Best Path
    Topic: /rrt_infotaxis/best_path

  - Class: rviz/Marker
    Name: Estimated Source
    Topic: /rrt_infotaxis/estimated_source

  - Class: rviz/Marker
    Name: Current Position
    Topic: /rrt_infotaxis/current_position

  - Class: rviz/MarkerArray
    Name: Status Text
    Topic: /rrt_infotaxis/source_info_text
```

### Camera Settings

For best visualization:
- **Fixed Frame**: `map`
- **View**: Top-down (2D) or slight angle
- **Camera Type**: Orbit or TopDownOrtho

## Color Blind Accessibility

The current color scheme uses:
- **Red-Yellow-Green gradient**: May be difficult for red-green color blindness
- **Alternative markers**: Use opacity and line thickness to distinguish

**Future improvement**: Option for color-blind friendly palettes (e.g., blue-white-red).

## Performance Notes

- Particle visualization: ~1000 markers at 5 Hz
- Path visualization: ~30-50 markers at 5 Hz
- High marker count may impact performance on slower systems
- Consider reducing `number_of_particles` or `n_tn` if visualization lags

## Troubleshooting

### Markers not showing in RViz

1. Check topic is being published:
   ```bash
   ros2 topic list | grep rrt_infotaxis
   ros2 topic hz /rrt_infotaxis/particles
   ```

2. Check Fixed Frame is set to `map`

3. Verify markers are not being filtered by size/distance

### Colors all the same (gray)

- `all_utilities` may not be passed correctly
- Check debug_info contains `all_utilities` field
- Verify utilities have variation (not all identical)

### Text overlay not visible

- Adjust text visualizer position in constructor
- Check text scale (default: 0.2m)
- Verify background box is visible

## References

- Paper: Kim et al., "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search", IEEE RAL 2025
- Code: `/home/efe/ros2_ws/src/base/efe_igdm/efe_igdm/igdm.py`
- Dead End Detection: `DEAD_END_DETECTION.md`
