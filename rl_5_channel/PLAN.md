# RL Gas Source Localization — Implementation Plan

## Goal

Train a PPO-based RL agent to autonomously locate a gas source in unknown indoor environments. Pretrain in a fast Python simulator (reusing IGDM components), then fine-tune in GADEN.

---

## 1. Design Summary

### State Vector (dim = 39)

| Component       | Dim | Details                                          |
|-----------------|-----|--------------------------------------------------|
| Gas history     | 10  | Binary readings for last 10 steps (0 or 1)       |
| LiDAR           | 24  | Raycast distances, 24 rays, normalized to [0, 1] |
| Position        | 2   | (x / map_width, y / map_height)                  |
| Wind            | 2   | (speed / max_speed, angle / 2pi)                 |
| Time budget     | 1   | step / max_steps                                 |

### Action Space

- **Output**: Single angle theta in [0, 2pi)
- **Step size**: Fixed 0.5 m
- **Distribution**: Beta distribution scaled to [0, 2pi)

### Reward Structure

| Event                      | Reward | Notes                                        |
|----------------------------|--------|----------------------------------------------|
| Source found (d < d_thresh) | +100   | Terminal, huge                               |
| Gas detection (binary = 1) | +1     | Encourages plume following                   |
| New cell visited           | +0.2   | Small coverage bonus, 0.5 m grid resolution  |
| Time step                  | -0.1   | Efficiency pressure                          |
| Obstacle collision          | -2     | Robot stays in place, receives penalty       |
| Max steps reached          | -10    | Failure terminal penalty                     |

### Architecture (PPO Actor-Critic)

```
Observation (39)
    |
MLP Shared Backbone [256, 256] (tanh activations)
    |
    +--- Actor Head [128] --> Beta distribution params (alpha, beta) --> angle
    |
    +--- Critic Head [128] --> V(s) scalar
```

### Training

- **Algorithm**: PPO (CleanRL-style single-file implementation)
- **Framework**: PyTorch
- **Parallelism**: Vectorized environments (multiple envs per training run)

---

## 2. Maps

6 map templates. Each episode randomly selects one template and randomizes its parameters (dimensions, wall positions, gap widths). Source, robot start, and wind are randomized per episode.

### Map 1 — Empty Room

```
+---------------------------+
|                           |
|                           |
|                           |
|                           |
+---------------------------+
```

- No internal obstacles.
- Room dimensions randomized: width in [8, 20] m, height in [6, 15] m.
- Baseline difficulty. Tests pure plume-following without navigation.

### Map 2 — Single Vertical Wall

```
+---------------------------+
|              |             |
|              |             |
|              |             |
|                            |
+---------------------------+
```

- One vertical wall segment in the interior, not spanning the full height (gap at top or bottom).
- Randomize: wall x-position, wall length, gap size and position.
- Forces the agent to navigate around a partition.

### Map 3 — U-Shaped Obstacle

```
+---------------------------+
|                           |
|          +--+             |
|          |  |             |
|          +  +             |
|                           |
+---------------------------+
```

- A U-shaped (or C-shaped) obstacle in the interior. Three connected wall segments forming a pocket.
- Randomize: U position, opening direction (up/down/left/right), arm lengths, gap width.
- Tests ability to avoid traps — gas may diffuse into the U but the source is outside it.

### Map 4a — Three Walls (Staggered)

```
+---------------------------+
|     |           |         |
|     |    +-+    |         |
|     |    | |    |         |
|          +-+              |
|                           |
+---------------------------+
```

- Two vertical walls extending from the top (with gaps at bottom) plus a central block/U obstacle.
- Creates a winding path requiring multiple direction changes.
- Randomize: wall positions, lengths, gap sizes, central obstacle shape.

### Map 4b — Complex Maze

```
+---------------------------+
|     |          |          |
|     |    +--+  |          |
|     |    |  |             |
|          +--+---+         |
|                 |         |
+---------------------------+
```

- Three walls (mix of vertical and horizontal) creating a maze-like layout with corridors.
- Horizontal wall segments at the bottom add extra routing complexity.
- Randomize: wall positions, lengths, horizontal/vertical mix, gap locations.

### Map 5 — Multi-Room (Doorways)

```
+--------+--+--------+
|        |  |        |
|  Room  +  +  Room  |
|        |  |        |
+--+  +--+--+--+  +--+
|  |  |        |  |  |
+  +--+        +--+  +
|  |  |        |  |  |
+--+  +--+--+--+  +--+
|        |  |        |
|  Room  +  +  Room  |
|        |  |        |
+--------+--+--------+
```

- 4-6 rooms connected by doorways around a central corridor.
- Randomize: number of rooms (4 or 6), room sizes, doorway widths and positions.
- Most challenging — agent must explore rooms systematically through narrow doorways.

### Randomization Per Episode

| Parameter             | Range / Method                                  |
|-----------------------|-------------------------------------------------|
| Map template          | Uniform random from 6 templates                 |
| Room dimensions       | Width: [8, 20] m, Height: [6, 15] m (template-dependent) |
| Wall positions        | Randomized within template constraints           |
| Gap sizes             | [0.8, 2.0] m minimum (robot must fit)           |
| Source position       | Random valid cell, not inside obstacle           |
| Robot start position  | Random valid cell, min distance from source > 3 m |
| Wind direction        | Uniform [0, 2pi)                                |
| Wind speed            | Uniform [0.1, 1.5] m/s                          |

---

## 3. Environment Components

### 3.1 Gas Dispersion (reuse + extend)

Reuse from `src/base/rrt_infotaxis/igdm_improved/`:
- `OccupancyGrid` — grid-based obstacle representation
- `IGDMModel` — Gaussian dispersion with Dijkstra obstacle-aware distances
- `BinarySensorModel` — binary concentration readings with adaptive threshold

**Extension needed**: Modify `IGDMModel` to support a wind bias. Approach:
- Shift the effective concentration peak downwind of the source.
- Given wind vector (v_w, theta_w), offset the Gaussian center by `wind_speed * dispersion_factor` in the wind direction.
- This creates asymmetric concentration: higher downwind, lower upwind.
- Does not need to be CFD-accurate — just enough to teach upwind-seeking behavior.

### 3.2 Raycast LiDAR

- 24 rays, evenly spaced (15 degrees apart).
- Each ray marches through the occupancy grid using Bresenham's line algorithm (or DDA).
- Returns distance to first occupied cell, capped at `max_range` (e.g., 3.0 m).
- Output: 24 floats normalized to [0, 1] by dividing by max_range.
- Fast: operates on the occupancy grid, microsecond-level per call.

### 3.3 Wind Model

- Per-episode constant uniform wind field (direction + speed).
- Wind affects gas dispersion (see 3.1) but does NOT affect robot movement.
- Robot observes wind at its current position via state vector.
- During GADEN fine-tuning, this will be replaced by the simulated anemometer readings.

### 3.4 Map Generator

- Each template is a Python function that takes randomized parameters and returns a populated `OccupancyGrid`.
- Wall thickness: 0.2 m (2 cells at 0.1 m resolution).
- Boundary walls always present.
- Validation: after generating a map, verify source and robot positions are reachable from each other (BFS/flood-fill connectivity check).

---

## 4. File Structure

```
src/base/rl_5_channel/
|-- PLAN.md                     # This document
|-- requirements.txt            # torch, gymnasium, numpy, scipy, numba, matplotlib
|-- config.py                   # All hyperparameters (single source of truth)
|
|-- envs/
|   |-- __init__.py
|   |-- gas_source_env.py       # Gymnasium env: reset(), step(), render()
|   |-- map_generator.py        # 6 map templates + randomization
|   |-- wind_model.py           # Uniform wind field + IGDM wind bias
|   |-- lidar_sim.py            # Raycast LiDAR on occupancy grid
|   |-- igdm_model.py           # Copied + extended with wind support
|   |-- occupancy_grid.py       # Copied from igdm_improved (unchanged)
|   |-- sensor_model.py         # Copied from igdm_improved (unchanged)
|
|-- models/
|   |-- __init__.py
|   |-- actor_critic.py         # PPO MLP actor-critic network
|
|-- training/
|   |-- __init__.py
|   |-- ppo.py                  # CleanRL-style PPO implementation
|   |-- train.py                # Entry point: env creation, training loop, logging
|
|-- evaluate.py                 # Load trained model, run episodes, visualize
```

### Dependency on Existing Code

The following files are **copied** from `src/base/rrt_infotaxis/igdm_improved/` into `envs/` to avoid cross-package imports and allow independent modification:
- `occupancy_grid.py` (unchanged)
- `sensor_model.py` (BinarySensorModel, unchanged)
- `igdm_model.py` (extended with wind bias)

---

## 5. Implementation Order

### Phase 1 — Environment

1. `config.py` — all hyperparameters and constants.
2. `envs/occupancy_grid.py` — copy from existing.
3. `envs/sensor_model.py` — copy BinarySensorModel from existing.
4. `envs/lidar_sim.py` — raycast LiDAR implementation.
5. `envs/wind_model.py` — uniform wind field.
6. `envs/igdm_model.py` — copy + add wind-biased dispersion.
7. `envs/map_generator.py` — 6 templates with randomization + connectivity validation.
8. `envs/gas_source_env.py` — Gymnasium env wrapping everything together.
9. **Test**: manual stepping through env, verify observations, rewards, rendering.

### Phase 2 — PPO Training

10. `models/actor_critic.py` — MLP with shared backbone, actor (Beta dist) + critic heads.
11. `training/ppo.py` — PPO algorithm: rollout buffer, GAE, policy/value loss, clipping.
12. `training/train.py` — entry point with vectorized envs, logging, checkpointing.
13. **Test**: train on empty room first, verify learning signal (reward curve should increase).

### Phase 3 — Evaluation and Iteration

14. `evaluate.py` — load checkpoint, run episodes, render trajectories, compute metrics.
15. Tune reward magnitudes based on training curves.
16. Tune map difficulty progression (curriculum: start easy, add complexity).
17. Compare binary vs. discrete sensor performance.

### Phase 4 — GADEN Integration (future)

18. Build a `GadenGymWrapper` that implements the same Gymnasium interface.
19. Same state vector (39-dim), same action space.
20. Load pretrained weights, fine-tune with lower learning rate.

---

## 6. Key Hyperparameters (initial values)

```python
# Environment
MAX_STEPS = 300
STEP_SIZE = 0.5               # meters
GRID_RESOLUTION = 0.1         # meters (occupancy grid)
VISITED_CELL_RESOLUTION = 0.5 # meters (for new-cell reward)
LIDAR_NUM_RAYS = 24
LIDAR_MAX_RANGE = 3.0         # meters
GAS_HISTORY_LENGTH = 10
D_SUCCESS = 0.5               # meters, source found threshold
ROBOT_RADIUS = 0.2            # meters, for collision checking

# Rewards
R_SUCCESS = 100.0
R_DETECTION = 1.0
R_NEW_CELL = 0.2
R_STEP = -0.1
R_COLLISION = -2.0
R_MAX_STEPS = -10.0

# Wind
WIND_SPEED_RANGE = (0.1, 1.5)     # m/s
WIND_MAX_SPEED = 2.0              # for normalization
WIND_DISPERSION_FACTOR = 2.0     # how much wind shifts the concentration

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
NUM_ENVS = 8                 # parallel environments
ROLLOUT_LENGTH = 2048        # steps per rollout per env
NUM_MINIBATCHES = 32
UPDATE_EPOCHS = 10
TOTAL_TIMESTEPS = 10_000_000

# Network
HIDDEN_DIM = 256
BACKBONE_LAYERS = 2
ACTOR_HEAD_DIM = 128
CRITIC_HEAD_DIM = 128
```

---

## 7. Success Criteria

- **Phase 1**: Env runs at > 5000 steps/sec (single env), observations and rewards are correct.
- **Phase 2**: On empty room, agent learns to find source in < 100 steps (avg) after 1M timesteps.
- **Phase 3**: On all 6 maps, agent finds source in < 200 steps (avg) after 10M timesteps.
- **Phase 4**: Fine-tuned agent in GADEN achieves comparable performance to efe_igdm baseline.
