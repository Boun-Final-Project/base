# GADEN Deployment Plan — RL Gas Source Localization

## Goal

Deploy the pretrained PPO agent (trained in the Python sim) inside GADEN via a
ROS2 node that mirrors the same 107-dim observation vector, runs the policy at
each step, and sends velocity commands to the robot.

---

## 1. Available GADEN Maps

| Map            | Cell Size | Approx Dimensions | Source Position    | Notes                        |
|----------------|-----------|-------------------|--------------------|------------------------------|
| `env_a`        | 0.1 m     | ~5×6 m            | (2.0, 4.5, 0.5)    | Simple open room             |
| `env_b`        | 0.1 m     | ~16×14 m          | (14.0, 12.0, 0.5)  | Large open space             |
| `env_c`        | 0.2 m     | ~20×10 m          | (18.0, 7.5, 0.5)   | Large, coarser grid          |
| `4_rooms`      | 0.1 m     | ~10×8 m           | (2.5, 3.0, 0.5)    | 4 rooms with doorways (sim2 has 2nd source) |
| `10x6_u_left`  | 0.1 m     | ~10×6 m           | (1.45, 3.0, 0.5)   | U-shaped obstacle            |
| `10x6_u_right` | 0.1 m     | ~10×6 m           | varies             | Mirrored U-shape             |
| `many_rooms`   | 0.1 m     | large             | (1.45, 3.0, 0.5)   | Multi-room layout            |
| `ultimate`     | 0.1 m     | large             | (1.45, 3.0, 0.5)   | Most complex layout          |

**Testing order** (easy → hard): `env_a` → `10x6_u_left` → `4_rooms` → `many_rooms` → `ultimate`

---

## 2. Architecture

```
GADEN simulation
  │
  ├── /fake_pid/Sensor_reading       (GasSensor)
  ├── /fake_anemometer/WindSensor_reading  (Anemometer)
  ├── /PioneerP3DX/laser_scanner     (LaserScan)
  └── /PioneerP3DX/ground_truth      (PoseWithCovarianceStamped)
         │
         ▼
  GadenRLNode  (new ROS2 node — this file)
  ├── _build_observation()           → 107-dim float32 vector
  ├── policy.act(obs)                → action scalar [0,1]
  ├── theta = action * 2π
  └── Navigator.send_goal(target_x, target_y)
         │
         ▼
  /PioneerP3DX/cmd_vel  (Twist)
```

The node reuses `Navigator` from `efe_igdm` for motion execution. Everything
else is new.

---

## 3. Observation Vector Construction (107 dims)

Identical to the training env `_build_observation()` in `gas_source_env.py`.

### 3.1 Gas History — dims 0–29 (30 values)

Store a rolling deque of the last 10 `(abs_x, abs_y, binary)` tuples.
At each step:
1. Read `msg.raw` from `/fake_pid/Sensor_reading`
2. Pass through `BinarySensorModel` (same adaptive threshold from `sensor_model.py`)
3. Append `(robot_x, robot_y, binary)` to the deque
4. At obs-build time, convert to relative coordinates:
   ```python
   rel_x = 0.5 + (past_x - current_x) / (2.0 * map_width)
   rel_y = 0.5 + (past_y - current_y) / (2.0 * map_height)
   ```
   Uninitialized entries: `(0.5, 0.5, 0.0)`

**Topic**: `/fake_pid/Sensor_reading` → `msg.raw` (float, ppm)

### 3.2 LiDAR — dims 30–101 (72 values)

1. Subscribe to `/PioneerP3DX/laser_scanner` (LaserScan)
2. Resample to exactly 72 uniformly-spaced rays over [0, 2π):
   ```python
   target_angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
   # Interpolate from msg.ranges using msg.angle_min + i * msg.angle_increment
   ```
3. Normalize: `distances / 3.0` (max_range = 3.0 m), clip to [0, 1]
4. Replace `inf`/`nan` with `1.0` (no obstacle within range)

**Topic**: `/PioneerP3DX/laser_scanner` → `msg.ranges`

### 3.3 Robot Position — dims 102–103 (2 values)

```python
obs[102] = robot_x / map_width    # from occupancy grid params
obs[103] = robot_y / map_height
```

**Source**: `/PioneerP3DX/ground_truth` → `msg.pose.pose.position.{x,y}`
**Map bounds**: from `/gaden_environment/occupancyMap3D` service (same as efe_igdm)

### 3.4 Wind — dims 104–105 (2 values)

```python
obs[104] = wind_speed / 2.0           # WIND_MAX_SPEED = 2.0 m/s
obs[105] = wind_direction / (2 * π)   # radians from +x, wrapped to [0,1]
```

**Topic**: `/fake_anemometer/WindSensor_reading` → `msg.wind_speed`, `msg.wind_direction`

### 3.5 Time — dim 106 (1 value)

```python
obs[106] = self.step_count / 600      # MAX_STEPS = 600
```

Internal counter, reset on episode start.

---

## 4. Action Execution

The policy outputs a scalar `a ∈ [0, 1]` (Beta distribution mean).

```python
theta = a * 2 * np.pi                        # convert to angle
dx = STEP_SIZE * np.cos(theta)               # STEP_SIZE = 0.5 m
dy = STEP_SIZE * np.sin(theta)
target = (robot_x + dx, robot_y + dy)
navigator.send_goal(target[0], target[1])    # reuse efe_igdm Navigator
```

The navigator drives the robot to the target waypoint, then calls back to
trigger the next policy step.

---

## 5. Files to Create

```
gaden_transfer/gaden_transfer_lidar/
├── PLAN.md                   ← this file
├── gaden_rl_node.py          ← main ROS2 node
├── obs_builder.py            ← builds the 107-dim observation
├── lidar_resampler.py        ← resamples LaserScan to 72 rays
└── launch/
    ├── env_a.launch.py       ← launch file for env_a
    ├── 4_rooms.launch.py     ← launch file for 4_rooms
    └── params.yaml           ← shared ROS2 parameters
```

The node imports directly from the training package:
```python
from reinforcement_learning.models.actor_critic import ActorCritic
from reinforcement_learning.envs.sensor_model import BinarySensorModel
from reinforcement_learning import config as cfg
```

And from efe_igdm for motion:
```python
from efe_igdm.planning.navigator import Navigator
from efe_igdm.mapping.occupancy_grid import load_3d_occupancy_grid_from_service
```

---

## 6. Episode Logic

```
on_reset():
    step_count = 0
    gas_history = deque([(None,None,0)] * 10, maxlen=10)
    sensor.initialize_threshold(first_reading)
    obs = build_observation()

on_navigation_complete():   ← callback from Navigator
    step_count += 1
    if step_count >= MAX_STEPS:
        log_failure(); reset()
        return
    obs = build_observation()
    action = policy.act(obs)
    execute_action(action)

on_source_found():          ← dist < D_SUCCESS = 0.5 m
    log_success(step_count)
    reset()                 ← optionally restart for multi-episode eval
```

---

## 7. Map-Specific Parameters

These go in `params.yaml` and are loaded per launch file.

| Parameter     | env_a       | 4_rooms     | many_rooms  | Notes                     |
|---------------|-------------|-------------|-------------|---------------------------|
| `map_width`   | auto        | auto        | auto        | Read from occupancy grid  |
| `map_height`  | auto        | auto        | auto        | Read from occupancy grid  |
| `true_source_x` | 2.0       | 2.5         | 1.45        | For success detection     |
| `true_source_y` | 4.5       | 3.0         | 3.0         | For success detection     |
| `max_steps`   | 600         | 600         | 800         | Larger maps need more     |
| `robot_start_x` | manual    | manual      | manual      | Set per experiment        |
| `robot_start_y` | manual    | manual      | manual      | Set per experiment        |

`map_width` and `map_height` are derived automatically from the occupancy grid
service — no hardcoding needed.

---

## 8. Testing Protocol

### Step 1 — Sanity check (env_a, 1 episode)
- Load best checkpoint from training run
- Place robot far from source (> 3 m)
- Run 1 episode, log observation vector at each step
- Verify all 107 dims are in [0, 1]
- Verify gas history accumulates correctly
- Check that action → movement direction is plausible

### Step 2 — Baseline evaluation (env_a, 20 episodes)
- Vary robot start position (e.g. 4 corners)
- Measure: success rate, steps to find, total distance
- Compare to efe_igdm baseline on same map

### Step 3 — Obstacle maps (10x6_u_left, 4_rooms)
- Same 20-episode protocol
- Watch for collision-heavy runs → may need fine-tuning

### Step 4 — Complex maps (many_rooms, ultimate)
- Increase `max_steps` to 800–1000 if needed
- Evaluate success rate only (expect lower than training)

### Step 5 — Fine-tuning (optional)
- If success rate < 30% on GADEN maps, fine-tune:
  - Load pretrained weights
  - Lower LR to 1e-4
  - Run PPO updates with GADEN as the env (slow but accurate)
  - 200k–500k timesteps should be sufficient

---

## 9. Key Differences: Sim vs GADEN

| Aspect              | Python Sim                          | GADEN                                 |
|---------------------|-------------------------------------|---------------------------------------|
| Gas model           | Filament plume (synthetic)          | CFD-based pre-simulated filaments     |
| Wind                | Constant per episode                | Spatially varying, pre-simulated      |
| LiDAR               | Perfect raycast on grid             | Noisy, real scan with occlusions      |
| Robot motion        | Instant teleport (0.5 m step)      | Nav2 / cmd_vel, takes ~1–2 s         |
| Observation rate    | 1 obs/step (synchronous)            | Async callbacks, need sync logic      |
| Map bounds          | Known exactly at reset              | Read from occupancy service           |
| Gas sensor          | Binary from filament concentration  | Binary from `GasSensor.raw` + threshold |

The most critical difference is **wind**: GADEN wind is spatially varying
while training used a constant per-episode wind. The observation passes only
the anemometer reading at the robot's current position, which is the same
in both cases — so this should transfer acceptably.

---

## 10. Success Criteria

| Stage           | Metric                       | Target     |
|-----------------|------------------------------|------------|
| Sanity check    | Obs vector valid             | 100%       |
| env_a baseline  | Success rate (20 eps)        | ≥ 50%      |
| 4_rooms         | Success rate (20 eps)        | ≥ 30%      |
| many_rooms      | Success rate (20 eps)        | ≥ 20%      |
| After fine-tune | Success rate on hardest map  | ≥ 40%      |
