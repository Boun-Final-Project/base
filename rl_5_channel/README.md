# rl_5_channel

5-channel spatial RL for gas source localization. Fork of `reinforcement_learning/`
with an upgraded observation wrapper and policy network.

## What's different from `reinforcement_learning/`

**Observation channels (4 → 5):** the single occupancy channel `{-1 wall, 0 unknown, +1 free}` is split into two binary channels.

| Layout | Channels |
|---|---|
| `reinforcement_learning/` (4) | `[occupancy, gas, recency, det_count]` |
| `rl_5_channel/` (5) | `[is_known, is_wall, gas, recency, det_count]` |

**Policy context vector (2 → 4):** wind direction encoded as `cos/sin` (no wrap-around discontinuity) and step progress added.

```
ctx = [speed/max_speed, cos(direction), sin(direction), step/MAX_STEPS]
```

**Other architectural changes:**
- Reveal step uses ground-truth grid + per-cell visibility check instead of LiDAR ray-casting (fixes a quantisation-overwrite bug; see [SPATIAL_CHANGES.md](SPATIAL_CHANGES.md)).
- FiLM final layer initialised so γ=1, β=0 — starts as identity, learns a delta.
- State-independent `actor_log_std` parameter instead of a state-conditional log-std head.
- Optional batched-GPU obs path: thin worker procs send scalar state, main process batches reveal + ego embedding on GPU. Enable with `--batched-obs`.

## Layout

```
config.py                    Hyperparameters and constants
envs/                        Env, wrappers, gas/wind models
  spatial_obs_wrapper.py     5-channel ego-centric wrapper
  spatial_obs_wrapper_friend.py  Frozen 4-channel reference for compare_wrappers.py
  batched_obs_builder.py     GPU-side obs builder (used by --batched-obs)
models/                      Networks (ActorCritic*, ActorCriticSpatial)
training/                    PPO + train loop
test/                        Eval, viz, comparison harnesses
visualization_scripts/       Standalone env/agent viz
train_spatial.sh             SLURM training entry
train_spatial_ent.sh         SLURM training with entropy-coeff schedule
eval_spatial.sh              SLURM eval entry
eval_gaden.sh                SLURM GADEN-maps eval entry
SPATIAL_CHANGES.md           Design notes for the 4ch → 5ch upgrade
SPATIAL_ARCH_PLAN.md         Original spatial-arch design doc
runs/                        Training outputs (gitignored)
```

## Train / eval

```bash
sbatch base/rl_5_channel/train_spatial.sh
sbatch base/rl_5_channel/eval_spatial.sh base/rl_5_channel/runs/<run-dir>
```

Direct (non-SLURM):

```bash
python -m base.rl_5_channel.training.train --arch spatial --batched-obs \
    --num-envs 48 --rollout-length 1024 --output-dir <out>
```

## GADEN-maps evaluation

Beyond the procedural test envs, the agent can be evaluated on the
hand-curated maps in `base/gaden_maps/` — same walls and source positions
the user runs in the GADEN simulator, with the actual CFD wind field driving
the plume.

How it works:
- [test/gaden_loader.py](test/gaden_loader.py) rasterizes occupancy from
  `walls.stl` ∪ ¬`inner.stl` at z=0.5 m, builds an `(H, W, 2)` wind field
  by z-collapsing `wind_at_cell_centers_0.csv`, fills mesh-density gaps via
  `scipy.ndimage.distance_transform_edt`, and reads source / robot positions
  from `recommended_configs.yaml`.
- `GasSourceEnv.reset(options={"map_data": ..., "wind_field": ...})` injects
  these instead of running `MapGenerator`. The `WindModel` is pinned to the
  field's spatial mean for the policy ctx vector; the plume queries the
  field per filament for advection.
- All env changes are backwards-compatible — passing no options leaves the
  procedural training/eval path unchanged.

Sanity-check the rasterized maps:

```bash
python -m base.rl_5_channel.test.visualize_gaden_envs
# → test/gaden_viz/<map>.png — wall + wind quiver + source/robot
```

Visualize a checkpoint's trajectories on each map:

```bash
python -m base.rl_5_channel.test.visualize_gaden_trajectories \
    --run-dir base/rl_5_channel/runs/<run-dir> --episodes-per-map 1
# → test/gaden_trajectories/<run>/<map>.png
```

Full eval (per-map success / return per checkpoint):

```bash
sbatch base/rl_5_channel/eval_gaden.sh base/rl_5_channel/runs/<run-dir>
```

Maps used by default: `4rooms`, `uleft`, `uright`, `labyrinth_left`,
`labyrinth_right`, `many_rooms`, `ultimate`.

See [SPATIAL_CHANGES.md](SPATIAL_CHANGES.md) for the rationale behind each change.
