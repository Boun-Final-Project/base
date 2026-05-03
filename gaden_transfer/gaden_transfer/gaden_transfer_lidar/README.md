# gaden_transfer_lidar

ROS deployment of the PPO agents trained in `rl_old/`. Multi-arch dispatcher
selected by the `arch` ROS parameter.

- **Action**: Beta over heading θ ∈ [0, 2π) (mlp/modular) or Gaussian
  (cos θ, sin θ) (dual/spatial); fixed 0.5 m step.
- **Architectures supported** via the `arch` ROS parameter:
  - `mlp` *(default)* → `ActorCritic` — flat 107-dim obs (gas history + lidar + pos + wind + time), Beta action
  - `modular` → `ActorCriticModular` — same 107-dim obs through GRU + Conv, Beta
  - `dual` → `ActorCriticDualBackbone` — same 107-dim obs through GRU + Conv, Gaussian cos/sin
  - `spatial` → `ActorCriticSpatial` (**4-channel** local copy from
    `reinforcement_learning/models/`) — image-style obs `[occupancy, gas,
    recency, det]` (98×98) built from live `LaserScan` + 2-d wind, Gaussian
    cos/sin. Distinct from the 5/6-channel image variants — different input
    channel count and a 4-output actor head. This is the deployment for the
    `ali_agent_*.pt` checkpoint family.

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
- `obs_builder.py` — assembles the 107-dim flat observation (mlp/modular/dual).
- `lidar_resampler.py` — downsamples raw `LaserScan` to the policy's ray count.
- `spatial_obs_builder.py` — alternate builder for the `spatial` arch:
  4-channel image obs `[occupancy, gas, recency, det]` + 2-d wind, built
  from live lidar (not GADEN GT raycast).
- `PLAN.md` — original design notes.
