# Dead End Detection for RRT-Infotaxis GSL

This document explains the dead end detection system implemented based on the paper "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search" by Kim et al., 2025.

## Overview

Dead end detection is a critical component of the dual-mode planner that identifies when the local RRT planner cannot find informative paths anymore. This triggers a switch to the global planner (when implemented) to escape from dead ends.

## Algorithm

The dead end detector uses an **adaptive threshold** approach based on Equations 20-21 from the paper.

### Equation 20: Optimal Branch Information

```
BI* = max{BI(V_b) | V_b ∈ V}
```

Where:
- `BI*` is the optimal (maximum) branch information among all RRT branches
- `V` is the set of all pruned branches from RRT
- `BI(V_b)` is the branch information for branch `V_b`

### Equation 21: Adaptive Threshold Update

```
BI_thresh_{k+1} = ε · BI_thresh_k + (1-ε) · BI*
```

Where:
- `BI_thresh_k` is the threshold at step k
- `ε` is a weight parameter (0 < ε < 1), typically 0.85-0.9
- This acts like a **low-pass filter**, making the threshold trail behind BI*

### Dead End Detection

A dead end is detected when:

```
BI* < BI_thresh_k
```

When the optimal branch information drops sharply (no useful information available), it falls below the delayed threshold, triggering dead end detection.

## Implementation

### 1. Dead End Detector Class

Located in: `efe_igdm/dead_end_detector.py`

```python
from efe_igdm.dead_end_detector import DeadEndDetector

# Initialize
detector = DeadEndDetector(epsilon=0.85, initial_threshold=0.1)

# Check for dead end
bi_optimal = max(all_branch_information_values)
is_dead_end = detector.is_dead_end(bi_optimal)

if is_dead_end:
    # Switch to global planner
    pass
```

### 2. Integration in Main Node

The detector is integrated in `igdm.py`:

```python
# In __init__:
self.dead_end_detector = DeadEndDetector(
    epsilon=self.get_parameter('dead_end_epsilon').value,
    initial_threshold=self.get_parameter('dead_end_initial_threshold').value
)

# In take_step (planning phase):
bi_optimal = debug_info.get("best_utility", debug_info.get("best_entropy_gain", 0.0))
dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)

if dead_end_detected:
    # TODO: Switch to global planner
    self.get_logger().warn('Dead end detected!')
```

## Parameters

### ROS2 Parameters

Add these to your launch file or set via command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dead_end_epsilon` | 0.85 | Weight for threshold update (ε). Higher = slower adaptation |
| `dead_end_initial_threshold` | 0.1 | Initial BI threshold value |

Example:
```bash
ros2 run efe_igdm start --ros-args -p dead_end_epsilon:=0.85 -p dead_end_initial_threshold:=0.1
```

### Tuning Guidelines

**Epsilon (ε)**:
- **Higher values (0.9-0.95)**: Threshold changes slowly, less sensitive to noise, may miss quick dead ends
- **Lower values (0.7-0.85)**: Threshold adapts quickly, more sensitive, may trigger false positives
- **Recommended**: Start with 0.85, adjust based on environment complexity

**Initial Threshold**:
- Should be small enough to not trigger immediately
- Typical range: 0.05 - 0.2
- **Recommended**: 0.1

## Data Logging

Dead end detection data is automatically logged to CSV files in `~/igdm_logs/`:

| Column | Description |
|--------|-------------|
| `bi_optimal` | Optimal branch information (BI*) at each step |
| `bi_threshold` | Adaptive threshold value at each step |
| `dead_end_detected` | 1 if dead end detected, 0 otherwise |

## Analysis and Visualization

### Using the Analysis Script

```bash
# Analyze a log file (displays plots)
python3 efe_igdm/analyze_dead_end.py ~/igdm_logs/igdm_log_20250102_120000.csv

# Save plots to a directory
python3 efe_igdm/analyze_dead_end.py ~/igdm_logs/igdm_log_20250102_120000.csv -o ./plots

# Also plot trajectory with dead end locations
python3 efe_igdm/analyze_dead_end.py ~/igdm_logs/igdm_log_20250102_120000.csv -t

# Analyze the latest log file
python3 efe_igdm/analyze_dead_end.py $(ls -t ~/igdm_logs/*.csv | head -1)
```

The script generates:
1. **Branch Information Plot**: Shows BI* and BI_threshold over time with dead end markers
2. **Entropy Plot**: Particle filter entropy evolution
3. **Uncertainty Plot**: Standard deviation of estimation
4. **Dead End Events**: Timeline of dead end detections
5. **Trajectory Plot** (with `-t`): Robot path with dead end locations marked

### Running the Example

Test the detector with a simulation:

```bash
cd /home/efe/ros2_ws/src/base/efe_igdm/efe_igdm
python3 dead_end_detector.py
```

This generates a plot showing how the detector responds to decreasing branch information.

## How It Works

### The Low-Pass Filter Analogy

The adaptive threshold acts like a **low-pass filter**:

1. **Normal operation**: BI* fluctuates around some value, threshold tracks it with a delay
2. **Approaching dead end**: BI* gradually decreases, threshold slowly follows
3. **Dead end reached**: BI* drops sharply, threshold hasn't caught up yet → **BI* < threshold** → detection!

### Example Timeline

```
Step | BI*  | Threshold | Status
-----|------|-----------|--------
  0  | 1.20 |   0.10    | OK (initializing)
  1  | 1.15 |   0.25    | OK
  2  | 1.10 |   0.39    | OK
  3  | 1.05 |   0.52    | OK
  4  | 0.90 |   0.64    | OK
  5  | 0.85 |   0.73    | OK (threshold rising)
  6  | 0.80 |   0.79    | OK (getting close)
  7  | 0.35 |   0.76    | DEAD END! (0.35 < 0.76)
  8  | 0.20 |   0.68    | DEAD END!
  9  | 0.15 |   0.60    | DEAD END!
```

At step 7, BI* drops sharply to 0.35 while the threshold is still at 0.76 (trailing behind), triggering detection.

## Future Work

Currently, the dead end detector only **detects** dead ends. The next step is to implement the **global planner** that will:

1. Build a PRM (Probabilistic Roadmap) graph
2. Detect frontiers (boundaries between known/unknown space)
3. Compute global utility for frontier vertices
4. Navigate to information-rich frontiers when dead end is detected
5. Switch back to local planner when high-information areas are found

See the paper Section IV.B.3 for the global planner algorithm.

## References

Kim, S., Seo, J., Jang, H., Kim, C., Kim, M., Pyo, J., & Oh, H. (2025). Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search. IEEE Robotics and Automation Letters, 10(1), 588-595.

**Key Equations**:
- Equation 19: Branch Information (BI)
- Equation 20: Optimal Branch Information (BI*)
- Equation 21: Adaptive Threshold Update
