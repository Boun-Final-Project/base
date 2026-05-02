# gaden_transfer_image_5ch

ROS deployment of the **5-channel spatial-image** PPO agents trained in
`rl_new/`. **Status: deprecated** — superseded by `gaden_transfer_image_6ch`
(same obs shape, finer cell res and dense wall lookup). Kept for replaying
older `rl_new` checkpoints.

- **Observation**: `(5, 98, 98)` ego-centric grid + `(4,)` context vector.
  - Channels: `[is_known, is_wall, gas, recency, det_count]`.
  - Context: `[wind_speed/max, cos(wind_dir), sin(wind_dir), step/MAX_STEPS]`.
- **Action**: Gaussian over (Δx, Δy) per step.
- **Wall map**: 0.5 m wrapper cells, **center-only** wall lookup — misses
  thin walls aligned off-center. (Fixed in `image_6ch`.)
- **Architecture**: `ActorCriticSpatial` (dual-stream CNN + FiLM wind).

## Run

```bash
ros2 run gaden_transfer gaden_rl_node_image_5ch \
    --ros-args -p checkpoint:=/abs/path/to/agent_X.pt
```

Both motion modes (teleport / Nav2 stop-go) are supported via parameter; see
`launch/params.yaml`.

## Files

- `gaden_rl_node.py` — ROS node entry-point.
- `actor_critic_spatial.py` — local copy of the model class (matches the
  `rl_new` training-tree definition for state-dict compatibility).
- `spatial_obs_builder.py` — 5-channel observation assembly.
