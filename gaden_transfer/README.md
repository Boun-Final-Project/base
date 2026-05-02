# gaden_transfer

ROS 2 package that deploys PPO-trained gas-source-localization agents inside
the GADEN simulator (and on a real robot via the same node). Pulls model
classes and shared config from the sibling **`reinforcement_learning`**
package.

## What's in here

```
gaden_transfer/
├── gaden_transfer_lidar/      ── rl_old (4-channel, lidar+wind)
├── gaden_transfer_image_5ch/  ── rl_new (5-channel image, deprecated)
└── gaden_transfer_image_6ch/  ── rl_osl (5-channel out, 6-channel internal, current)
```

Each subdirectory is a self-contained ROS node:

```
gaden_transfer_lidar/
├── gaden_rl_node.py             ROS node — loads the policy, subscribes
│                                to lidar/gas/wind/pose, publishes goals
│                                or teleports per step.
├── obs_builder.py               Per-step observation assembly. Mirrors
│                                the matching training-side wrapper so
│                                the policy sees the same vector it
│                                trained on.
├── lidar_resampler.py           Downsamples raw scans to the ray count
│                                the policy expects.
└── launch/
    └── params.yaml              ROS parameter defaults.
```

## The three deployments

| Folder | Trained from | Obs | Action | Wall map | Status |
|---|---|---|---|---|---|
| `gaden_transfer_lidar/`     | `rl_old/` | 107-dim flat (gas history + lidar + pos + wind + time) for `arch ∈ {mlp, modular, dual}`, or 4-channel image (`[occupancy, gas, recency, det]`) + 2-d wind for `arch:=spatial` | Beta over heading θ (mlp/modular) or Gaussian (cos θ, sin θ) (dual/spatial); fixed 0.5 m step | n/a | working |
| `gaden_transfer_image_5ch/` | `rl_new/` | 5-channel ego-centric grid (98×98) + 4-dim ctx | Gaussian (Δx, Δy) | 0.5 m wrapper cells, center-only wall lookup (sparse — misses thin walls) | deprecated |
| `gaden_transfer_image_6ch/` | `rl_osl/` | 5-channel ego-centric grid (98×98) + 4-dim ctx | Gaussian (Δx, Δy) | 0.2 m wrapper cells, dense any-subcell wall lookup | current |

**Pick the deployment that matches the checkpoint you trained.** The
observation builders are not interchangeable; an `rl_osl` checkpoint won't run
in `gaden_transfer_lidar/` and vice versa.

### Observation channels (image arch)

For the 5-channel deployments, `spatial_obs_builder.py` produces:

| # | Channel    | Meaning |
|---|------------|---|
| 0 | is_known   | 1 where the cell has been observed (revealed by SLAM or GT raytrace) |
| 1 | is_wall    | 1 where the observed cell is occupied (any subcell wall in `image_6ch`) |
| 2 | gas        | binary detection latched into a world-frame map |
| 3 | recency    | exponential-decay heatmap of recent visits |
| 4 | det_count  | running count of binary detections, normalised |

Plus a 4-dim context vector `[wind_speed/max, cos(wind_dir), sin(wind_dir), step/MAX_STEPS]`.

### Motion modes

The 5-channel nodes support two motion modes (toggleable via parameter):

- **Teleport** — the node publishes a pose-jump per step. Fast, deterministic,
  matches training. Use for evaluation and benchmarking the policy.
- **Nav2 stop-go** — the node sends each step's target as a `NavigateToPose`
  goal. Realistic motion with planner + controller in the loop, comparable
  to ADSM-style metrics (travel distance, wall time).

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select gaden_transfer --symlink-install
source install/setup.bash
```

Dependencies (apart from ROS 2 Humble): `torch`, `numpy`, the sibling
`reinforcement_learning` package, the SLAM node `efe_igdm`, and the GADEN
test scenarios under `src/gaden/test_env/scenarios`.

## Entry points

The three nodes are exposed as console scripts (see `setup.py`):

| Console script              | Module |
|---|---|
| `gaden_rl_node_lidar`       | `gaden_transfer.gaden_transfer_lidar.gaden_rl_node` |
| `gaden_rl_node_image_5ch`   | `gaden_transfer.gaden_transfer_image_5ch.gaden_rl_node` |
| `gaden_rl_node_image_6ch`   | `gaden_transfer.gaden_transfer_image_6ch.gaden_rl_node` |

## Run a batch (recommended for benchmarking)

The workspace root has scripts that loop a checkpoint over scenarios and
collect per-run summaries:

| Script | Deployment |
|---|---|
| `run_rl_batch.sh`            | `gaden_transfer_lidar/`     (rl_old) |
| `run_rl_image_batch.sh`      | `gaden_transfer_image_5ch/` (rl_new, teleport) |
| `run_rl_image_nav2_batch.sh` | `gaden_transfer_image_5ch/` (rl_new, Nav2) |
| `run_rl_osl_nav2_batch.sh`   | `gaden_transfer_image_6ch/` (rl_osl) — supports both teleport and Nav2 via `RLO_USE_NAV2` |

Each script accepts environment-variable overrides (checkpoint, max_steps,
playback speed, etc.) and writes results to `Results/<run_name>/`.

## Adding a new checkpoint

1. Train in the matching tree (`rl_old/`, `rl_new/`, or `rl_osl/`) and
   produce a `.pt` file.
2. Make sure `reinforcement_learning/config.py` matches the training run's
   hyperparameters (cell size, grid size, sensor thresholds).
3. Pass the absolute path to the `.pt` file via `checkpoint:=` or via the
   batch script's `*_CHECKPOINT` environment variable.

The checkpoint includes `model_state_dict`. The deployment loads it into the
arch-specific actor-critic class, sets the policy to `eval()`, and never
trains during deployment.
