# gaden_transfer_nav

Navigation-to-goal PPO agent deployment for GADEN.

## Overview

This sub-package deploys a trained `NavActorCritic` policy that takes a 77-dimensional observation vector (72 LiDAR rays + position + goal bearing + goal distance) and outputs a heading `(cos θ, sin θ)` which is decoded to a heading, allowing the robot to move a fixed `step_size` (0.5 m) per step via teleport to navigate the robot to a goal position.

Compatible checkpoints are in `nav_lidar-003/*.pt`.

## Observation Layout

The agent receives a 77-dimensional observation vector structured as follows:

| Slice | Dims | Content |
|---|---|---|
| `obs[:72]` | 72 | LiDAR rays, normalized by `LIDAR_MAX_RANGE = 3.0 m` |
| `obs[72:74]` | 2 | Robot position `(x/W, y/H)` normalized by environment width/height |
| `obs[74:76]` | 2 | Goal direction `(cos θ_goal, sin θ_goal)` |
| `obs[76]` | 1 | Goal distance normalized by room diagonal |

## Build

```bash
cd /home/hdd/ros2_ws
colcon build --packages-select gaden_transfer
```

## Run

```bash
ros2 run gaden_transfer gaden_rl_node_nav \
    --ros-args \
    -p checkpoint:=/home/hdd/ros2_ws/src/base/reinforcement_learning/runs/nav_lidar-003/checkpoint_00600.pt \
    -p goal_x:=3.0 \
    -p goal_y:=4.5 \
    -p goal_tolerance:=0.5 \
    -p max_steps:=600 \
    -p use_nav2:=false
```

## ROS Parameters

All parameters are configurable via `launch/params.yaml` or overridden at launch time:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | str | `""` | Absolute path to checkpoint .pt file (required) |
| `device` | str | `"cpu"` | PyTorch device (`"cpu"` or `"cuda"`) |
| `goal_x` | float | 5.0 | Goal position X (metres) |
| `goal_y` | float | 5.0 | Goal position Y (metres) |
| `goal_tolerance` | float | 0.5 | Goal reached threshold (metres) |
| `max_steps` | int | 600 | Episode length cap (steps) |
| `use_nav2` | bool | false | Drive mode: false = teleport via /PioneerP3DX/initialpose, true = Nav2 PoseStamped to /goal_pose |
| `step_size` | float | 0.5 | Step size in metres (must match training) |
| `num_episodes` | int | 1 | Number of episodes to run before shutdown |
| `step_delay` | float | 0.5 | Delay between steps (seconds) |
| `start_x` | float | -999.0 | Start position X (-999 = GADEN default spawn) |
| `start_y` | float | -999.0 | Start position Y (-999 = GADEN default spawn) |
| `occupancy_service` | str | `"/gaden_environment/occupancyMap3D"` | GADEN occupancy service endpoint |
| `occupancy_z_level` | int | 5 | Z-level for occupancy queries |
| `occupancy_timeout` | float | 60.0 | Service call timeout (seconds) |
| `step_log_every` | int | 1 | Log every N steps |
| `publish_markers` | bool | true | Publish RViz markers for debugging |
