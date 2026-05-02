# gaden_transfer_lidar

ROS deployment of the **lidar-based** PPO agents trained in `rl_old/`.

- **Observation**: 39-dim flat vector (gas history + lidar rays + position + wind + time).
- **Action**: Beta over heading θ ∈ [0, 2π), fixed 0.5 m step.
- **Architectures supported** via the `arch` ROS parameter:
  - `mlp` → `ActorCritic` (default, Beta action)
  - `modular` → `ActorCriticModular` (GRU + Conv, Beta)
  - `dual` → `ActorCriticDualBackbone` (GRU + Conv, Gaussian cos/sin)
  - `spatial` → `ActorCriticSpatial` (the spatial CNN; same weights as the
    image variants but consumed via this lidar-style obs builder)

## Run

```bash
ros2 run gaden_transfer gaden_rl_node_lidar \
    --ros-args \
    -p checkpoint:=/abs/path/to/agent_X.pt \
    -p arch:=mlp
```

Defaults live in `launch/params.yaml`. Set `max_steps` to 600 for
training-sized maps, 800–1000 for `many_rooms`/`ultimate`.

## Files

- `gaden_rl_node.py` — ROS node entry-point.
- `obs_builder.py` — assembles the 39-dim flat observation.
- `lidar_resampler.py` — downsamples raw `LaserScan` to the policy's ray count.
- `spatial_obs_builder.py` — alternate builder for the `spatial` arch (matches
  the image variants' channel layout).
- `PLAN.md` — original design notes.
