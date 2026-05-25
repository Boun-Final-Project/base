# gaden_transfer_image_6ch

ROS deployment of the **current-generation** spatial-image PPO agents trained
in `rl_osl/` (and the byte-identical `test_rl/`).

- **Observation**: `(5, 98, 98)` ego-centric grid + `(4,)` context vector.
  - Channels: `[is_known, is_wall, gas, recency, det_count]`.
  - A 6th `motion` channel is maintained internally (decay 0.6/step) for
    forward-compat but not emitted, to stay byte-compatible with the
    `efe_0_2_wall_*` checkpoints.
  - Context: `[wind_speed/max, cos(wind_dir), sin(wind_dir), step/MAX_STEPS]`.
- **Action**: Gaussian over (Δx, Δy) per step.
- **Wall map**: 0.2 m wrapper cells (hard-coded — bypasses the in-tree
  `cfg.VISITED_CELL_RESOLUTION` to avoid breaking the older 5-channel
  variant), **dense any-subcell** wall lookup.
- **Origin-aware indexing**: subtracts `occ_map.origin_x/y` everywhere
  (patched 2026-04-26). Required for non-zero GADEN origins.
- **Architecture**: `ActorCriticSpatial`.

## Run

```bash
ros2 run gaden_transfer gaden_rl_node_image_6ch \
    --ros-args -p checkpoint:=/abs/path/to/agent_X.pt
```

Compatible checkpoints live at `~/ros2_ws/rl_osl/efe_0_2_wall_*.pt` and the
top-level `efe_agent_99975168_r10.pt`.

## Files

- `gaden_rl_node.py` — ROS node entry-point.
- `actor_critic_spatial.py` — local copy of the model class.
- `spatial_obs_builder.py` — observation assembly (origin-aware, dense wall).
- `test_obs_equivalence.py` — checks the deployment builder is bit-exact
  against the `rl_osl` training-time wrapper at origin (0, 0).
