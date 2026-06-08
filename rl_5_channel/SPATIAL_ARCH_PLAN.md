# Spatial CNN Architecture — Implementation Plan

## Overview

Add a fourth architecture `ActorCriticSpatial` alongside the existing three. No existing code is removed or modified beyond wiring the new option into `config.py` and `train.py`.

---

## Files to Create

### 1. `envs/spatial_obs_wrapper.py`

Wraps `GasSourceEnv`. On each step, intercepts the flat 107-dim observation and instead returns an ego-centric spatial observation.

**Internal state maintained per episode:**
| Grid | Shape | Init value | Update rule |
|---|---|---|---|
| `occupancy` | 98×98 | 0.0 (unknown) | On reset: stamp known walls/free cells within current map bounds; on each step: reveal cells within LiDAR range |
| `max_gas` | 98×98 | -1.0 (unvisited) | On visit: `max(current, new_reading)` |
| `recency` | 98×98 | 0.0 (unvisited) | On every step: decay all visited cells by `exp(-λ)`; set current cell to 1.0 |
| `detection_count` | 98×98 | 0.0 | On visit with detection: increment; normalize as `count / max_count` where `max_count` is the current maximum across all cells → ∈ [0,1] |

**Ego-centric transform:**
- World origin mapped so robot is always at grid center `(49, 49)`
- Transform:
  ```
  ego_col = 49 + round((world_x - robot_x) / 0.5)
  ego_row = 49 + round((world_y - robot_y) / 0.5)
  ```
- No axis flip: the codebase uses `grid[gy, gx]` where `gx = floor(x/res)` and `gy = floor(y/res)`, so row and y are co-directional. Verified from `OccupancyGrid.world_to_grid` and `LidarSim.scan`.
- Cells outside `[0, 98)` are out-of-bounds padding, left as unknown (0.0)

**Observation returned:**
```
spatial_grid : float32 (4, 98, 98)   # [occupancy, max_gas, recency, detection_count]
wind         : float32 (2,)           # [speed / WIND_MAX_SPEED, direction / 2π]
```

Returned as a tuple `(spatial_grid, wind)` — see open questions.

**Key methods:**
- `__init__(env)` — stores wrapped env, initialises grids
- `reset()` — calls `env.reset()`, resets all grids, stamps initial occupancy from `env._grid`
- `step(action)` — calls `env.step(action)`, updates grids, returns spatial obs
- `_world_to_grid(world_xy)` → `(row, col)` — coordinate transform
- `_update_grids(robot_pos, gas_reading, lidar_hits)` — single method that updates all four channels
- `_get_obs()` → `(spatial_grid, wind)` — assembles final arrays

**Occupancy encoding:**
- `0.0`  = unknown (never seen)
- `1.0`  = free (LiDAR confirmed clear or robot visited)
- `-1.0` = wall (LiDAR hit)

**Recency decay implementation:**
- Each step: `recency_grid[visited_mask] *= exp(-λ)` — efficient vectorised decay over all previously visited cells
- Then: `recency_grid[current_cell] = 1.0`
- `λ = 0.015` (configurable via `config.py`)

---

### 2. `models/actor_critic_spatial.py`

**Module-level helpers (reuse `layer_init` pattern from `actor_critic.py`):**

```
ResBlock(dim)
    Linear(dim → dim) → LayerNorm → ReLU → Linear(dim → dim) + residual
```

**`ActorCriticSpatial(nn.Module)`**

Constructor signature:
```python
def __init__(
    self,
    static_channels=1,          # occupancy
    dynamic_channels=3,         # max_gas, recency, detection_count
    wind_dim=2,
    film_hidden=64,
    cnn_out_channels=128,       # after 1×1 fusion conv
    shared_hidden=(512, 256),
    actor_head_dim=128,
    critic_head_dim=256,
    num_res_blocks=3,
    lambda_recency=0.015,
):
```

**Forward pass structure:**

```
static_in  : (B, 1, 98, 98)
dynamic_in : (B, 3, 98, 98)
wind_in    : (B, 2)

Static stream:
    Conv2d(1→16,  5×5, stride=2, padding=2) → ReLU  # 49×49×16
    Conv2d(16→32, 3×3, stride=2, padding=1) → ReLU  # 25×25×32
    Conv2d(32→32, 3×3, stride=2, padding=1) → ReLU  # 13×13×32

Dynamic stream:
    Conv2d(3→16,  5×5, stride=2, padding=2) → ReLU  # 49×49×16
    Conv2d(16→32, 3×3, stride=2, padding=1) → ReLU  # 25×25×32
    Conv2d(32→64, 3×3, stride=2, padding=1) → ReLU  # 13×13×64

Fusion:
    Concat along channel dim → 13×13×96
    Conv2d(96→128, 1×1)                              # 13×13×128
    Flatten → 21632-dim

FiLM:
    Linear(2→64) → ReLU → Linear(64→43264)
    Split → γ (21632), β (21632)
    f = γ ⊙ f + β

Shared MLP:
    Linear(21632→512) → Tanh
    Linear(512→256)   → Tanh
    → shared (B, 256)

Actor head:
    Linear(256→128) → LayerNorm → ReLU
    ResBlock(128) × 3
    Linear(128→2) → Softplus + 1   # (α, β), both > 1 for unimodal Beta
    → Beta(α, β) → action ∈ [0,1] → θ = action × 2π

Critic head:
    Linear(256→256) → LayerNorm → ReLU
    ResBlock(256) × 3
    Linear(256→1)
    → V(s)
```

**`get_action_and_value(spatial, wind, action=None)`** — matches interface of existing models for PPO compatibility.

---

## Files to Modify

### 3. `config.py`

Add a new section at the bottom:

```python
# =============================================================================
# Spatial CNN architecture
# =============================================================================
SPATIAL_GRID_SIZE     = 98           # cells (49m / 0.5m)
SPATIAL_LAMBDA        = 0.015        # recency decay rate
SPATIAL_FILM_HIDDEN   = 64
SPATIAL_CNN_OUT_CH    = 128          # channels after 1×1 fusion conv
SPATIAL_SHARED_HIDDEN = (512, 256)
SPATIAL_ACTOR_DIM     = 128
SPATIAL_CRITIC_DIM    = 256
SPATIAL_RES_BLOCKS    = 3
```

No existing constants are changed.

---

### 4. `models/__init__.py`

Add export:
```python
from .actor_critic_spatial import ActorCriticSpatial
```

---

### 5. `training/train.py`

**a) New `--arch` choice:**
```
--arch  {mlp, modular, dual, spatial}   default: mlp
```

**b) New `make_spatial_env` factory:**
```python
def make_spatial_env(seed, rank, template_id=None):
    def _init():
        env = SpatialObsWrapper(GasSourceEnv(template_id=template_id))
        env.reset(seed=seed + rank)
        return env
    return _init
```

**c) Obs shape handling in `RolloutBuffer`:**

The existing buffer assumes flat obs (`STATE_DIM = 107`). For spatial, obs is a tuple
`(spatial_grid (4,98,98), wind (2,))`. Use **Option A**:

Add an `obs_is_spatial` flag to `RolloutBuffer`. When set, allocate:
- `self.spatial_obs : (T, N, 4, 98, 98)`
- `self.wind_obs    : (T, N, 2)`

instead of `self.obs : (T, N, 107)`. Pass both to `get_action_and_value` during the PPO update.

**d) Model selection:**
```python
if args.arch == "spatial":
    model = ActorCriticSpatial(...cfg values...).to(device)
elif args.arch == "dual":
    model = ActorCriticDualBackbone(...)
...
```

**e) No changes** to PPO update logic, GAE computation, or reward normalisation — these are obs-agnostic.

---

## Implementation Order

1. `config.py` — add spatial constants (no risk)
2. `envs/spatial_obs_wrapper.py` — core observation logic; test standalone with a single env reset+step loop
3. `models/actor_critic_spatial.py` — build and verify output shapes with a dummy forward pass
4. `training/train.py` — wire `--arch spatial`, update `RolloutBuffer`, test with `--total-timesteps 10000`
5. `models/__init__.py` — export

---

## Open Questions

1. **Occupancy reveal strategy:** Currently the plan stamps the full map at reset (the wrapper has access to `env._grid`). This means the agent sees the full layout immediately. If you want to enforce that only explored areas are visible, reveal should happen incrementally via LiDAR hits only — requires passing LiDAR hit positions from the env step. Which do you prefer?

2. **Observation interface:** Return `(spatial_grid, wind)` as a tuple or as a flat dict? Tuple is simpler for the buffer; dict is more explicit.

3. **Detection count normalisation:** Use `count / max_count` where `max_count` is the running maximum across all cells. Retroactively rescales all cells when max changes, but always produces a well-spread [0,1] range highlighting the hotspot. ✓ Decided.
