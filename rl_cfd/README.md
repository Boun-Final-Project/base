# rl_cfd

6-channel spatial RL with a CFD-style spatially-varying wind field. Extension of `rl_5_channel/` that adds a motion-trail observation channel and replaces uniform wind with a per-cell flow field derived from 2D potential flow + curl noise.

## What's different from `rl_5_channel/`

**Observation channels (5 → 6):** adds a `motion` channel that records the
recent agent trajectory.

```
[is_known, is_wall, gas, recency, det_count, motion]
```

**Wind model:** [`envs/wind_field.py`](envs/wind_field.py) replaces the uniform `WindModel`.
Per episode it solves the 2D Laplace equation on the room (Dirichlet at the
outer ring as inflow/outflow, Neumann at interior obstacles), takes velocity
as `-∇φ`, and overlays an optional divergence-free curl-noise field. Walls
zero out velocity. The plume queries `local_batch(positions)` per filament so
each filament drifts according to the local flow.

The policy still consumes the **spatial mean** of the field as a 4-D ctx
vector `[speed/max, cos(θ), sin(θ), step/MAX_STEPS]`, so 5-channel
checkpoints can be fine-tuned without touching the policy heads.

## Migrating a 5-channel checkpoint

[`models/convert_5ch_to_6ch.py`](models/convert_5ch_to_6ch.py) reshapes
`dynamic_cnn.0.weight` from `(16, 3, 5, 5)` → `(16, 4, 5, 5)`, zero-init the
new motion-channel slot. At fine-tune step 0 the forward pass is
bit-identical to the 5-channel checkpoint.

```bash
python -m base.rl_cfd.models.convert_5ch_to_6ch \
    /path/to/5ch_checkpoint.pt \
    /path/to/6ch_checkpoint.pt
```

## Layout

```
config.py                    Hyperparameters and constants
envs/                        Env, wrappers, gas/wind models
  wind_field.py              Spatially-varying potential-flow wind
  spatial_obs_wrapper.py     6-channel ego-centric wrapper
  batched_obs_builder.py     GPU-side obs builder
models/                      Networks
  actor_critic_spatial.py    6-channel input variant
  convert_5ch_to_6ch.py      Checkpoint upconversion utility
training/                    PPO + train loop
test/                        Eval, viz, comparison harnesses
viz_fast_bundle.py           Fast-bundle wind/plume viz
viz_motion_channel.py        Motion-channel inspection
train_cfd.sh                 SLURM training entry
eval_cfd.sh                  SLURM eval entry
eval_gaden.sh                SLURM GADEN-maps eval entry
runs/                        Training outputs (gitignored)
```

## Train / eval

```bash
sbatch base/rl_cfd/train_cfd.sh
sbatch base/rl_cfd/eval_cfd.sh base/rl_cfd/runs/<run-dir>
```

Direct (non-SLURM):

```bash
python -m base.rl_cfd.training.train --arch spatial --batched-obs \
    --num-envs 48 --rollout-length 1024 \
    --entropy-coeff 0.02 --entropy-coeff-end 0.005 --target-kl 0.02 \
    --output-dir <out>
```

## GADEN-maps evaluation

The agent can be evaluated on the hand-curated maps in `base/gaden_maps/`
— same walls and source positions used in the GADEN simulator, with the
real CFD wind field driving the plume.

How it works:
- [test/gaden_loader.py](test/gaden_loader.py) rasterizes occupancy from
  `walls.stl` ∪ ¬`inner.stl` at z=0.5 m, builds an `(H, W, 2)` wind field
  from `wind_at_cell_centers_0.csv`, and reads source / robot positions
  from `recommended_configs.yaml`.
- `GadenWindField` is shaped as a drop-in for the native `WindField` —
  exposes `local_batch()`, `Ux/Uy` attrs, `get_observation_spatial()`, and
  `get_dispersion_offset()`. Calling code in [envs/gas_source_env.py](envs/gas_source_env.py)
  and [envs/filament_plume.py](envs/filament_plume.py) doesn't need to know
  whether it's running against potential-flow or GADEN data.
- `GasSourceEnv.reset(options={"map_data": ..., "wind_field": ...})` swaps
  `self._wind` for the GADEN field for that episode and restores the
  procedural `WindField` on the next procedural reset.

Sanity-check the rasterized maps:

```bash
python -m base.rl_cfd.test.visualize_gaden_envs
# → test/gaden_viz/<map>.png — wall + wind quiver + source/robot
```

Visualize a checkpoint's trajectories on each map:

```bash
python -m base.rl_cfd.test.visualize_gaden_trajectories \
    --run-dir base/rl_cfd/runs/<run-dir> --episodes-per-map 1
# → test/gaden_trajectories/<run>/<map>.png
```

Full eval (per-map success / return per checkpoint):

```bash
sbatch base/rl_cfd/eval_gaden.sh base/rl_cfd/runs/<run-dir>
```

Maps used by default: `4rooms`, `uleft`, `uright`, `labyrinth_left`,
`labyrinth_right`, `many_rooms`, `ultimate`.
