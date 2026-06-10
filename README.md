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
├── wind_visualizer/       # Wind field visualization utility
│
├── gaden_maps/            # Pre-configured GADEN scenario maps
│   ├── env_a/             # Single-room environment
│   ├── env_b/             # Multi-room environment
│   ├── env_c/             # Environment C
│   └── env_c_nowind/      # Environment C (no wind)
│
└── scripts/               # Helper scripts for running simulations
    ├── presim.sh          # Preprocessing + gas simulation launcher
    └── runsim.sh          # Robot & environment launcher
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

### Install GADEN Scenario Maps

This repository includes pre-configured scenario maps in the `gaden_maps/` folder. Copy them into your GADEN scenarios directory:

```bash
cp -r ~/ros2_ws/src/base/gaden_maps/* ~/ros2_ws/src/gaden/test_env/scenarios/
```

Then rebuild so GADEN picks up the new scenarios:

```bash
cd ~/ros2_ws && colcon build
```

### Apply Nav2 Configuration

This repository includes a tuned `nav2_params.yaml` in `gaden_config/navigation_config/`. Copy it over GADEN's default to prevent the robot from getting stuck near walls:

```bash
cp ~/ros2_ws/src/base/gaden_config/navigation_config/nav2_params.yaml \
   ~/ros2_ws/src/gaden/test_env/navigation_config/nav2_params.yaml
```

Then rebuild:

```bash
cd ~/ros2_ws && colcon build
```

**Why:** The default Nav2 config includes a `RotateToGoal` DWB critic and a tight `yaw_goal_tolerance: 0.1` rad. Together these cause the robot to spin in place to align with goal headings, which frequently results in the robot pressing against walls and getting stuck. The tuned config removes the `RotateToGoal` critic and sets `yaw_goal_tolerance: 6.28` (any orientation accepted), so the robot drives straight to the goal position without forced in-place rotation.

### Helper Scripts (Optional)

Two interactive launcher scripts are provided in `scripts/` to simplify running simulations. They auto-detect your workspace path.

```bash
# Add aliases to your shell (add these to ~/.bashrc for persistence)
alias presim='~/ros2_ws/src/base/scripts/presim.sh'
alias runsim='~/ros2_ws/src/base/scripts/runsim.sh'
```

- **`presim`** — Runs GADEN preprocessing and gas simulation. Presents an interactive menu to select a scenario and simulation, then executes both steps sequentially.
- **`runsim`** — Launches the robot and environment player. Shows available gas data iterations and lets you pick a start time.

## Usage

### 1. Start GADEN Environment

> **Tip:** If you set up the helper scripts (see [Helper Scripts](#helper-scripts-optional)), you can simply run `presim` then `runsim` instead of the manual steps below.

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

## RL Policy Training (RLaika)

RLaika is trained entirely in a fast Python simulator (procedural maps + a
filament-based plume model) inside the
[`reinforcement_learning`](reinforcement_learning/) package — no ROS2 or GADEN
needed at training time. Producing a deployable checkpoint is a two-stage
process: a from-scratch PPO **base training** run, followed by an optional
**CFD local-wind finetune** that closes the wind-distribution gap to GADEN.

### 1. Base training — the champion recipe

[`reinforcement_learning/train_champ.sh`](reinforcement_learning/train_champ.sh)
launches the exact PPO recipe that produced the champion checkpoint
`agent_91750400.pt` (dual-backbone architecture, seed 1; the champion is the
early-stopped best checkpoint at ~91.75M of a 200M-step budget).

```bash
# On SLURM (recommended)
sbatch reinforcement_learning/train_champ.sh

# Locally / custom python interpreter
bash reinforcement_learning/train_champ.sh
VENV_PY=/path/to/python bash reinforcement_learning/train_champ.sh
```

#### Training hyperparameters

| Category | Parameter | Value |
|---|---|---|
| **PPO core** | Learning rate | 3 × 10⁻⁴ |
| | Discount factor γ | 0.99 |
| | GAE λ | 0.95 |
| | Clip ratio ε | 0.3 |
| | Value loss coefficient | 0.5 |
| | Entropy coefficient | 0.02 |
| **Rollout & updates** | Total timesteps | 2 × 10⁸ budget (champion early-stopped at ~9.2 × 10⁷) |
| | Number of parallel envs | 256 |
| | Rollout length | 1024 |
| | Minibatches per update | 32 |
| | Update epochs | 10 |
| **Optimization** | Max gradient norm | 0.5 |
| | Learning-rate annealing | linear, from 50% of training |
| | Target KL (early stop) | 0.05 |
| **Episode budget** | Max steps per episode | 600 |
| | Step cost r_step | −1.0 |
| | Detection / collision / success | +0.75 / −5.0 / +200.0 |
| | Success distance D_success | 0.5 m |
| **Curriculum learning** | Map templates (stages) | T0–T1, T0–T3, T0–T5 |
| | Unlock at progress | 0%, 25%, 50% |
| | Room size schedule | 8×6–10×8 m → 8×6–20×15 m |
| | Curriculum fraction | 50% of training |

`train_champ.sh` sets the run-defining values (learning rate, clip ratio,
target KL, annealing, envs, rollout length, timesteps, curriculum, seed) via
CLI flags; everything else comes from `config.py` defaults (see the reward
caveat below). All of them are recorded in the committed snapshot
`champ_config.json`.

Checkpoints and TensorBoard logs land in
`reinforcement_learning/runs/<run-name>/`.

[`reinforcement_learning/champ_config.json`](reinforcement_learning/champ_config.json)
is the full hyperparameter snapshot of the original run and is the
authoritative provenance. **Reward caveat:** the champion was trained with
`R_STEP = -1.0` and `R_DETECTION = 0.75`; `config.py` on `main` carries
different values. To reproduce the champion exactly, run the script from the
`efe/champ-training-script` branch (whose `config.py` matches the snapshot),
or set those two values in
[`reinforcement_learning/config.py`](reinforcement_learning/config.py) to the
snapshot values first.

### 2. Selecting a checkpoint

Training success keeps climbing long after transfer to GADEN has peaked, so
always select checkpoints by GADEN evaluation, never by training reward:

```bash
sbatch reinforcement_learning/eval_gaden.sh reinforcement_learning/runs/<run-name>
```

### 3. Finetuning on CFD wind with local-wind observation

Uniform per-episode wind is a useful simplification for base training, but it
does not capture the recirculation, channelling, and shear that obstacles
produce — the largest remaining train/eval distribution gap to GADEN. The
[`cfd_wind_pipeline`](cfd_wind_pipeline/) package closes it: we pre-compute a
library of **~2,400 maps with CFD-solved wind fields** (OpenFOAM, generated
once offline because each solve is compute-heavy) and finetune the policy
against that library.

The finetune resumes from the uniform-wind champion at 91.75M steps with a
reduced learning rate (5 × 10⁻⁵, annealed to 1 × 10⁻⁵) and switches the wind
observation from the episode's spatial mean to the **local wind measured at
the robot's position** (`OSL_LOCAL_WIND_OBS=1`). To expose the policy to
episodes where it starts outside the plume, **20% of episodes re-place the
robot 4–14 m away from the plume** (far-plume spawns,
`cfd_wind_pipeline/far_plume_placement.py`). Every checkpoint is evaluated
inline on realistic-gas GADEN episodes and the best-scoring checkpoint is
deployed — typically reached ~15M finetune steps in; longer finetuning
degrades the policy. On the offline real-gas evaluation harness this recipe
took the champion from 57% to 76–80% overall success (20 episodes/map),
including the first non-zero results on `many_rooms`.

**Step 1 — build a CFD wind library** (one-off; needs SLURM + an OpenFOAM
container). See the [`cfd_wind_pipeline` README](cfd_wind_pipeline/README.md)
for the full walkthrough. Optionally precompute the far-spawn placement cache:

```bash
sbatch cfd_wind_pipeline/sbatch/precompute_far_placement.sh \
    /path/to/cfd_library[,/path/to/cfd_library_2] /path/to/rl-package-checkout
```

**Step 2 — launch the finetune:**

```bash
bash cfd_wind_pipeline/sbatch/finetune_local_wind.sh \
    /path/to/cfd_library[,/path/to/cfd_library_2] \
    /path/to/rl-package-checkout \
    /path/to/checkpoints/agent_91750400.pt
```

#### Fine-tuning hyperparameters

| Parameter | Value |
|---|---|
| Resume from | champion checkpoint @ 91.75M steps |
| Learning rate | 5 × 10⁻⁵, annealed linearly to 1 × 10⁻⁵ |
| Clip ratio ε / target KL | 0.3 / 0.05 |
| Parallel envs | 96 |
| Wind observation | local wind at robot cell (`OSL_LOCAL_WIND_OBS=1`) |
| Training wind | CFD library (~2,400 maps), no synthetic-wind mix |
| Far-plume spawns | 20% of episodes, 4–14 m plume-hit distance |
| Step budget | 152M absolute (~60M headroom over the resumed step) |
| Checkpoint selection | inline real-gas GADEN eval; best ≈ +15M steps |

Notes:

- The RL package checkout must support `OSL_LOCAL_WIND_OBS` in its `config.py`
  (`feature/local-wind-obs` lineage). The inline GADEN eval likewise requires
  RL-package support; without it the `OSL_INLINE_GADEN_*` flags are ignored
  harmlessly — fall back to offline checkpoint evaluation with
  `reinforcement_learning/eval_gaden.sh`.
- `--total-timesteps` is **absolute** (resumed step + extra), not additional.
- Per the checkpoint-selection rule above, the GADEN peak arrives ~15M steps
  into the finetune and drifts down afterwards — early-stop on GADEN eval, not
  training reward.
- The finetuned policy observes local wind, so eval and deployment must also
  run with `OSL_LOCAL_WIND_OBS=1`; mixing the conventions silently degrades it.

For custom recipes (mixing synthetic-wind resets back in via
`CFD_MIX_SYNTHETIC`, restricting map templates via `CFD_TEMPLATE_FILTER`, other
PPO flags), call the underlying
[`cfd_wind_pipeline/sbatch/train_cfd_library.sh`](cfd_wind_pipeline/sbatch/train_cfd_library.sh)
directly — the launcher above is a thin, recipe-pinned wrapper around it.

## RL Policy Deployment (RLaika)

Besides the information-theoretic algorithms above, the platform deploys
**RLaika** — a PPO policy trained for gas-source localization — inside the same
GADEN simulator. The deployment node lives in the [`gaden_transfer`](gaden_transfer/)
package (`gaden_rl_node_lidar`; lidar + wind observation, `arch ∈ {mlp, modular,
dual, spatial}`) and can optionally engage a SLAM-based frontier-escape stack
when the policy gets wedged in a room.

### Batch deployment / evaluation

`run_rl_lidar_batch.sh` (at the **workspace root**,
`~/ros2_ws/`, alongside the other `run_*_batch.sh` runners) deploys a checkpoint
over the 7 evaluation maps and collects
per-run summaries. Each run brings up the GADEN world
(`main_simbot_launch.py method:=none`), runs `gaden_rl_node_lidar`, harvests
`node.log → summary.txt`, and tears the sim down before the next run.

```bash
cd ~/ros2_ws
colcon build --packages-select gaden_transfer test_env --symlink-install
source install/setup.bash

# 7 maps × 5 runs, dual-backbone checkpoint, SLAM frontier-escape + video, headless
RL_CHECKPOINT=~/ros2_ws/src/base/agent_188416000.pt RL_ARCH=dual \
RL_NUM_RUNS=5 RL_ESCAPE=1 RL_RECORD=1 \
bash ~/ros2_ws/run_rl_lidar_batch.sh
```

For long sweeps run it under `tmux`:
`tmux new-session -d -s rl 'RL_CHECKPOINT=… RL_NUM_RUNS=5 bash ~/ros2_ws/run_rl_lidar_batch.sh'`.

### Options (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `RL_CHECKPOINT` | — (**required**) | absolute path to the `.pt` checkpoint |
| `RL_ARCH` | `mlp` | network arch: `mlp` / `modular` / `dual` / `spatial` (use `dual` for `agent_188416000`) |
| `RL_NUM_RUNS` | `1` | trials per map (`5` → 7×5 = 35 runs) |
| `RL_MAX_STEPS` | `600` | episode step cap (800–1000 for `many_rooms`/`ultimate`) |
| `RL_ESCAPE` | `0` | **slam**: `1` enables the SLAM frontier-escape + hybrid-drive stack |
| `RL_RECORD` | `0` | **video**: `1` records each run's RViz view to `<run_dir>/capture.mp4` (headless Xvfb + ffmpeg) |
| `RL_SLAM_VIZ` | `0` | with recording, show the robot's online SLAM map (`/rlaika/slam_map`); needs `RL_ESCAPE=1` |
| `RL_USE_NAV2` | `false` | motion: `false` teleports each step (default), `true` drives via Nav2 (pair with `RL_SPEED=1.0`) |
| `RL_SPEED` | `5.0` | GADEN playback speed multiplier |
| `RL_DEVICE` | `cpu` | `cpu` / `cuda` |
| `RL_TARGETS` | 7 eval maps | space-separated `scenario::sim` overrides (default: `curved_labrinth_{left,right}`, `10x6_u_{left,right}`, `4_rooms`, `many_rooms`, `ultimate`, all `::sim1`) |

Headless is the default (RViz only opens when `RL_RECORD=1`). Results land in
`~/ros2_ws/Results/<run_name>/` — a `summary.txt` per run plus an `aggregate.txt`
scoreboard.

## References

- Kim, H. et al., "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search," *IEEE Robotics and Automation Letters*, 2025. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10777609/)
- Park, S. & Cho, H., "Receding-horizon RRT-Infotaxis for autonomous source search in urban environments," *Aerospace Science and Technology*, 2022. [[ScienceDirect]](https://www.sciencedirect.com/science/article/pii/S1270963821007860)
- Vergassola, M., Villermaux, E. & Shraiman, B. I., "'Infotaxis' as a strategy for searching without gradients," *Nature*, 445(7126), 406-409, 2007. [[Nature]](https://www.nature.com/articles/nature05464)

## Contributors

- **Efe Mantaroglu** — efemantaroglu@gmail.com
- **Ali Sonmez** — al1.sonmez.mi@gmail.com
- **Simal Guven** — (transferred to another project)

## License

MIT
