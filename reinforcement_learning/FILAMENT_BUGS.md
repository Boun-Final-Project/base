# Filament Plume — Bug Report

Audit of the filament-based gas dispersion model and its integration with the
RL environment. Covers `envs/filament_plume.py`, the filament paths in
`envs/gas_source_env.py`, and the filament rendering in `envs/visualizer.py`.

| #  | Severity         | Bug                                                            |
|----|------------------|----------------------------------------------------------------|
| 1  | **High**         | `reflection_energy` parameter has zero effect                  |
| 2  | **High**         | Sensor threshold initialized from empty-plume noise            |
| 3  | **High**         | Plume not warmed up → non-physical episode start               |
| 4  | Low              | Dead write to `velocities` in `_release`                       |
| 5  | Medium (design)  | Concentration bleeds through walls (no line-of-sight check)    |

Minor physics/perf notes are listed at the end.

---

## Bug 1 — `reflection_energy` parameter has no effect

**Location:** `envs/filament_plume.py`, `_handle_obstacles`, lines ~318–324.

```python
reflected_pos = pre_positions[i] + v_reflected * self.dt        # no damping
...
self.velocities[i] = v_reflected * self.reflection_energy       # damped, but unused
```

The reflected **position** uses the undamped `v_reflected`. The damped velocity
is stored in `self.velocities[i]`, but at the start of the next `update()`:

```python
self.velocities[:self._n] = wind + turbulence                    # overwritten
```

…it is overwritten before it can influence motion. Consequently,
`FILAMENT_REFLECTION_ENERGY` in `config.py` is a dead knob — changing it has
no observable effect on the simulation.

### Proposed fix

Apply the damping factor to the reflected displacement so it actually shapes
the filament trajectory:

```python
reflected_pos = pre_positions[i] + v_reflected * self.reflection_energy * self.dt
```

The separate write to `self.velocities[i]` can be dropped (or kept for
visualization only). Alternatively, if the "memoryless turbulence each step"
design is intentional, remove `reflection_energy` from the config and the
signature entirely.

---

## Bug 2 — Sensor threshold initialized from an empty plume

**Location:** `envs/gas_source_env.py`, `reset()`, lines ~192–200.

```python
if cfg.GAS_MODEL == "filament":
    conc = self._plume.concentration_at(self._robot_pos)    # 0 — plume is empty
...
noisy = conc + self._rng.normal(0, self._sensor.get_std(conc))  # ≈ N(0, sigma_env)
self._sensor.initialize_threshold(noisy)                        # threshold ≈ noise
```

At `reset()` the filament plume has just been constructed and contains **zero
filaments**, so `concentration_at` always returns `0.0`. The "initial
measurement" is therefore pure sensor noise, and the adaptive threshold is
initialized to that value (possibly even negative). Because
`BinarySensorModel.update_threshold` only ratchets the threshold **up**, this
mis-initialization pollutes the gas history and reward signal for a long
stretch of early steps: nearly any positive concentration trips `binary = 1`.

The IGDM path is unaffected because `IGDMModel` returns a meaningful
concentration at step 0 from its static field.

### Proposed fix

Warm up the plume before reading the first sensor value (see Bug 3) so that
`concentration_at` reflects a realistic steady-state reading when the
threshold is initialized. A single fix solves Bugs 2 and 3 together.

---

## Bug 3 — Plume not warmed up at episode start

**Location:** `envs/gas_source_env.py`, `reset()`.

After construction the plume is not stepped before the environment returns
its first observation. A Lagrangian plume needs roughly `FILAMENT_MAX_AGE`
steps (≈120 in the current config) to reach steady state. During this
transient:

- The robot at or near the source receives ~0 concentration.
- Downwind regions have no gas at all.
- The binary sensor and detection reward are meaningless for the warm-up
  duration.
- Every episode starts with a non-physical "dead zone" that the policy must
  learn to ignore.

### Proposed fix

In `reset()`, after constructing the `FilamentPlume` and **before** the
sensor threshold initialization, run the plume forward for enough steps to
reach steady state:

```python
if cfg.GAS_MODEL == "filament":
    self._plume = FilamentPlume(...)
    # Warm up to steady state so step 0 sees a realistic plume
    for _ in range(cfg.FILAMENT_MAX_AGE):
        self._plume.update()
```

Then the later `concentration_at(self._robot_pos)` will return a realistic
value and the sensor threshold will be initialized against a physical reading
instead of pure noise. This also fixes Bug 2.

A new config constant `FILAMENT_WARMUP_STEPS` (defaulting to
`FILAMENT_MAX_AGE`) would let this be tuned without touching code.

**Cost:** `~max_age × filaments_per_step × python-step` added to every
`reset()`. At the current values (`120 × 2`) this is negligible relative to
the cost of an episode, but worth benchmarking if training throughput is
tight.

---

## Bug 4 — Dead write to `velocities` in `_release`

**Location:** `envs/filament_plume.py`, `_release`, line ~270.

```python
self.velocities[idx] = self.wind_velocity
```

Every freshly released filament has its velocity initialized to
`wind_velocity`, but on the very next line of `update()`:

```python
self.velocities[:self._n] = wind + turbulence
```

the whole array is overwritten. The initializer is a dead write. Not a
correctness issue, but misleading to anyone reading the code.

### Proposed fix

Delete the line. No behavior change.

---

## Bug 5 — Concentration ignores walls (no line-of-sight check)

**Location:** `envs/filament_plume.py`, `concentration_at`.

`concentration_at` sums 2D Gaussian contributions from **all** active
filaments regardless of obstacles between the filament and the query point.
A filament sitting on the opposite side of a wall still contributes to the
robot's reading, so the robot can effectively "smell through walls."

This is a known limitation of simple Gaussian puff models, but it matters
for RL: the policy may learn unphysical shortcuts that exploit this
leakage, producing behavior that will not transfer to a realistic simulator
or the real world.

### Proposed fix (optional, expensive)

Add a Bresenham ray-cast from each contributing filament to the query point
against the occupancy grid; zero out contributions from filaments whose line
of sight is blocked. Rough cost: `O(N_filaments × ray_length)` per query,
which can be significant in large maps. Mitigations:

- Pre-filter filaments by `r² > k·σ²` so that negligible contributions are
  skipped before ray-casting.
- Cache ray-cast results per `(filament_cell, query_cell)` pair within a
  single step.
- Implement the raycast in NumPy / Numba if the Python loop is too slow.

If this is a priority, I'd suggest a two-phase rollout: enable the LOS check
behind a config flag (`FILAMENT_WALL_OCCLUSION = False` by default), measure
the training-throughput impact, and promote it once confirmed acceptable.

---

## Minor issues (not fixed unless requested)

These are correctness-adjacent but do not clearly break training.

- **Reflection uses full `dt` after bounce.** `reflected_pos = pre +
  v_reflected * dt` implicitly assumes the entire timestep is spent in the
  reflected direction. The filament actually spent part of `dt` reaching the
  wall, so the bounced travel should use `(1 − t_hit) * dt`. The current form
  overshoots slightly. Fixing it requires computing the ray-wall intersection
  time, which adds complexity for a small physical gain.
- **Diffusion runs for filaments that are then absorbed** by
  `_handle_obstacles`. The `np.sqrt(σ² + 2K·dt)` update is cheap, so this is
  a negligible waste of compute, not a correctness problem.
- **`visualizer.py`, filament branch (line ~142):** `max_age = max(1,
  ages.max())` is safe only because the branch is guarded by
  `len(positions) > 0`. If that guard is ever relaxed, `ages.max()` on an
  empty array will crash. Defensive-only concern.
- **`_estimate_normal` fallback for fully-surrounded filaments** picks a
  random direction; if that direction also lies in a wall the filament is
  absorbed on the next check. This is acceptable behavior for a rare corner
  case (3×3 wall pocket).

---

## Recommended fix order

1. **Bug 3 + Bug 2** in a single change (plume warm-up in `reset()`).
2. **Bug 1** (apply `reflection_energy` to the reflected displacement).
3. **Bug 4** (delete dead write).
4. **Bug 5** only if policy behavior suggests it is exploiting wall leakage.
