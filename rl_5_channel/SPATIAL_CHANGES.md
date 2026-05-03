# Spatial Wrapper / Model — Proposed Changes

Living notes on issues surfaced during review of `SPATIAL_ARCH_PLAN.md` and the
in-repo implementation (`envs/spatial_obs_wrapper.py`, `models/actor_critic_spatial.py`).

---

## 1. Occupancy channel — split into two

**Current:** one channel with `{-1 wall, 0 unknown, +1 free}`.
**Problem:** the first conv has to learn to disentangle "wall boundary" from
"free-area gradient" in the same feature map; semantically mixed signal.

**Change:** output two separate binary channels.
```
is_known : 1 if the cell has ever been observed, else 0
is_wall  : 1 if the cell is known to be a wall,   else 0
```
(Free is implied by `is_known=1, is_wall=0`.)

Total spatial channels become **5** instead of 4: `[is_known, is_wall, gas, recency, det_count]`.
Model `static_channels` goes from 1 → 2 (feed `is_known` and `is_wall` together
to the static stream).

---

## 2. Replace LiDAR-based reveal with ground-truth + visibility check

**Current:** wrapper re-runs `env._lidar.scan(...)` and stamps free/wall cells
from LiDAR ray samples.

**Problems this causes:**
- **Quantisation overwrite bug (confirmed):** LiDAR/true grid are at 0.1 m,
  wrapper cells at 0.5 m. One wrapper cell contains 25 sub-cells. A ray from a
  new robot pose can pass through a previously-wall-stamped cell as "free"
  (its hit quantises into the next cell) → the `-1` gets overwritten with `+1`.
  Observed **29 flips in 400 steps** in a standard template-2 episode.
- **Dotted walls at range:** 72 rays × 3 m max → ~0.26 m angular spacing at the
  far edge. Gap cells between ray hits stay `unknown`, so far-away walls look
  like a dotted line to the CNN.

**Change:** drop the ray-casting path in `_update_occupancy`. For each 0.5 m
cell in a 3×3 m box around the robot, check ground truth directly **with an
occlusion test**:

```
for each target cell (r, c) within ±3 m:
    if ray from robot_cell → (r, c) passes through a known wall in env._grid:
        skip       # occluded
    else:
        is_known[r, c] = 1
        is_wall[r, c] = (env._grid[...] != 0)
```

- Preserves realism (walls still occlude cells behind them).
- Walls are solid/continuous at wrapper resolution — no dotted line, no
  overwrite race.
- Drops a LiDAR scan per step inside the wrapper (env still scans for its own
  obs, but we no longer duplicate that work).

**Justification:** on the real robot the occupancy map is built from LiDAR
anyway; we're just building it more directly in sim, with the same occlusion
physics. No extra information is given to the policy.

---

## 3. FiLM initialisation

**Current:** `layer_init(Linear(64, 2*cnn_flat))` with orthogonal std=√2.
Initial γ ≈ N(0, 2/64), so `γ⊙f + β` nearly zeros features at step 0.

**Change:** zero the final FiLM weight, set bias so γ=1, β=0 at init. E.g.
```python
with torch.no_grad():
    final = self.film_net[-1]
    final.weight.zero_()
    final.bias[:cnn_flat].fill_(1.0)   # γ bias = 1
    final.bias[cnn_flat:].zero_()      # β bias = 0
```
FiLM starts as identity, learns a delta.

---

## 4. (Optional) Trim the shared MLP

`Linear(21632 → 512)` is 77 % of model params (11.2 M / 14.4 M total).
Options if we ever care about train throughput:
- Adaptive average pool 13×13 → 5×5 before flatten (cuts input to 3200).
- Drop fusion channels 128 → 64.
- Both → ~3 M params total.

Not urgent; flag only.

---

## Open questions

- [ ] Action distribution: plan says **Beta**, code uses **Normal(cos, sin)**.
      Align on one. (Normal is what `ActorCriticDualBackbone` uses — consistent
      with existing model, probably the right choice.)
- [ ] Gas channel: plan says "max_gas" continuous; code uses signed binary
      `{-1, 0, +1}`. Binary matches the sensor — keep the code's choice and
      fix the plan text.
- [x] `train.py` tuple-obs buffer branch: spatial path was already wired, but
      `SpatialRolloutBuffer` had channel count and ctx dim hardcoded to the
      old `(4, 2)`. Parametrised (defaults `(5, 3)`) and smoke-tested 2
      updates end-to-end on template 0.

---

## Decision log

| Date       | Change                                            | Status    |
|------------|---------------------------------------------------|-----------|
| 2026-04-21 | Split occupancy → `is_known` + `is_wall`           | **done**  |
| 2026-04-21 | Ground truth + visibility check (drop LiDAR path)  | **done**  |
| 2026-04-21 | Fix FiLM init to identity                           | **done**  |
| 2026-04-21 | Restore time (`step/MAX_STEPS`) — appended to wind  | **done**  |
| 2026-04-21 | Update `SpatialRolloutBuffer` for 5-ch + 3-d ctx    | **done**  |
| 2026-04-21 | Fix self-occlusion in `_reveal` (target cell masked) | **done**  |

### Implementation notes (done)

- `envs/spatial_obs_wrapper.py`: replaced `_update_occupancy` with `_reveal`,
  which casts rays from the robot to each cell centre inside a 3 m box,
  stamps `is_known=1, is_wall=GT` for unoccluded cells. World grids are now
  `_known_world`, `_wall_world`, `_gas_world`, `_rec_world`, `_det_world`.
  Output tuple: `((5, 98, 98), (2,))`.
- `models/actor_critic_spatial.py`: static stream now `Conv2d(2→16, ...)`;
  slicing changed to `spatial[:, 0:2]` (static) and `spatial[:, 2:5]` (dynamic).
  Param count: 14,420,693 (≈ +400 from the extra input channel).
- Verified on template 5 / seed 11: **0 wall-cell flips over 239 steps**,
  walls appear as continuous lines (see `spatial_walls.png`, `spatial_channels.png`).
- **Self-occlusion bugfix:** initial `_reveal` marked a wall-cell target as
  occluded by its own sub-cells (ray traversed the target cell's 0.1 m sub-cells
  before reaching the centre; those wall sub-samples triggered the occlusion
  test). Fixed by masking out occlusion samples whose wrapper-cell index
  matches the target. Wall-cell count in template 5 / seed 11 at step 239
  went from ~28 to 48 — roughly 40 % of walls were being silently dropped.
- FiLM final `Linear(64 → 43264)` now has `weight = 0`, `bias = [1]*cnn_flat + [0]*cnn_flat`.
  Verified: `γ ≡ 1, β ≡ 0` for any wind input at step 0. Gradients still flow
  (`∂loss/∂γ = features`, `∂loss/∂β = 1`), so the layer learns a delta from a
  working identity start.
- Wind vector extended from 2 → 3 dims: `[speed/max, dir/2π, step/MAX_STEPS]`.
  FiLM input bumped `Linear(2→64)` → `Linear(3→64)`. Restores the time signal
  that the flat arch already had; critic can bootstrap against the
  `R_MAX_STEPS` penalty.
