# ADSM port faithfulness — input/output equivalence test

Proves that this ROS2 port of ADSM makes **identical decisions** to the authors'
published ROS1 implementation, given identical inputs.

This is a claim about the **algorithm**, deliberately independent of the
simulator (Gazebo vs BasicSim), the navigation stack (move_base vs Nav2), and
the gas sensor — none of which can affect the result.

## Run

```bash
cd src/base/adsm/test
make            # builds and runs; no ROS needed, just g++
```

## What it compares

Both implementations' decision math is embedded **verbatim**; the only edit is
that member variables become explicit parameters, because the port binds
`real_x_/real_y_/real_yaw_` where the original binds `x_/y_/yaw_` (same
formulas, different variable). Identical inputs go to both; outputs are
compared **bit-exact**.

| # | Component | Cases | Result |
|---|-----------|-------|--------|
| 1 | `probability()` — Eq. 3 indoor-Gaussian source-term estimator | 2,000,000 | **0 mismatches** (max diff 0) |
| 2 | `evaluate()` — fitness `j = j_p + k1*f*j_i` + argmax goal selection | 199,769 | **0 mismatches** |
| 3 | gas binarization | 200,000 | **differs — intentional** (MOX -> PID) |
| 4 | `estimate()` angular clustering — N-class bearing split, farthest-per-class | 100,000 | **0 mismatches** |

Verdict: **decision core bit-exact equivalent.**

## Provenance

| | |
|---|---|
| Original | <https://github.com/mwanggh/An-adaptive-robot-search-algorithm> |
| Commit | `1f3c6a0b5483191d49eb4b920713ccb276ce8743` (2026-01-20) |
| Paper | Wang, Xin, Deng, Chen & Qu, *An Adaptive Robot Search Algorithm for Balancing Exploitation and Exploration in Indoor Intermittent Source Seeking*, IEEE TIE 73(4), 2026. doi:10.1109/TIE.2025.3632565 |

Line references in the test source point at the exact origin of each extracted
block (`orig::` = paper's `src/adsm.cpp`; `port::` = this package's
`src/adsm.cpp`).

## Shared modules — checked separately by semantic diff

`frontier_finder.cpp`, `rrt_sampler.cpp` and `goal.cpp` are not re-tested here
because they are **semantically identical** to the originals. Normalising away
include paths, logging macros, comments and whitespace leaves only:

- `goal.cpp` — no difference at all.
- `rrt_sampler.cpp` — member-initialiser list reordered, plus `(void)width;
  `(void)height;`. Both no-ops: C++ initialises members in *declaration* order
  and each initialiser binds by name, so the reorder cannot change which value
  lands in which member (it only silences `-Wreorder`); the `(void)` casts
  suppress unused-parameter warnings.
- `frontier_finder.cpp` — `cost_grid[x][y]` became `cost_grid[(int)x][(int)y]`.
  `x` is a `double` that always holds an exact non-negative integer (cell
  indices, with `nx < 0` guarded), so implicit and explicit truncation give
  identical values. Cosmetic; silences a narrowing warning.

Reproduce:

```bash
norm() { sed -E 's://.*::' "$1" | sed -E '/^\s*#include/d' \
  | sed -E '/(ROS_INFO|RCLCPP_INFO|ROS_ERROR|RCLCPP_ERROR)/d' \
  | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' | grep -v '^$'; }
diff <(norm <orig>/src/frontier_finder.cpp) <(norm ../src/frontier_finder.cpp)
```

## Known, intentional deviations from the paper

All environmental — none touch the algorithm:

1. **Gas sensor: MOX -> PID.** Paper reads a MOX TGS2620 (`sensor_model: 0`) and
   inverts resistance (`gas_ = gas_max - raw`, thresholds 500/2000 of
   `gas_max` 63000). This port reads a PID (`sensor_model: 30`) in ppm directly
   (`gas_ = raw`, thresholds 0.1/0.3). The thresholds are *proportionally*
   matched (~1-3% of scale both ways). Test 3 shows the expected divergence.
   Note the `in_rec` time-window branch exists to reject MOX hysteresis and is
   largely inert with a PID; `gas_max` is unused on the PID path.
2. **Pose binding.** Uses ground-truth pose (`real_*`) where the original uses
   the odom/SLAM estimate (`x_`). Formulas identical.
3. **Port-only fallback.** When `sum_probability == 0 && sum_frontier_size == 0`
   the port explores randomly instead of selecting a `j == 0` goal. Fired in
   231/200,000 (0.1%) of randomized cases.
4. `create_random_gaol` validates map bounds / non-lethal cells before
   returning a goal (the original samples blind).
5. Map/infrastructure: GADEN occupancy service + external SLAM node instead of
   gmapping; Nav2 `NavigateToPose` instead of move_base.

## Maintenance

The two implementations are **embedded copies**, so this is a snapshot proof. If
`../src/adsm.cpp` changes, re-sync the `port::` blocks (and bump the commit above
if re-checking against a newer upstream) or the test will silently go stale.
