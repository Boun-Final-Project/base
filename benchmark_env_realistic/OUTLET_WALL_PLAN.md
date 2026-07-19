# Implementation Plan — Make GADEN outlets solid to the SLAM lidar

**Date:** 2026-07-18
**Package:** `benchmark_env_realistic`
**Status:** proposed

## Problem

In the ground-truth navigation path, GADEN **outlet** cells (the map "inlets" — vent/door
openings on the boundary) are treated as **walls**. In the realistic (Cartographer) path they are
**passable**: the robot's SLAM map shows them as free space, the planner routes through them, and —
because BasicSim uses the same 2-state map for physics — the robot can physically drive out through
the opening.

### Root cause (confirmed with data, 4_rooms_start_c, z = 0.5 m)

GADEN's 3D occupancy grid uses three states: `0` free, `1` wall, `2` outlet. Two consumers diverge:

| consumer | outlet (value 2) becomes | result |
|---|---|---|
| **Ground-truth loader** `efe_igdm/mapping/occupancy_grid.py:98-119` | `grid_2d = (grid_2d > 0)` → **occupied**; plus `outlet_mask` re-stamped as `CELL_OUTLET` by `LidarMapper`, non-traversable in `is_valid`/`is_valid_traversable` | outlets = walls |
| **BasicSim `occupancy.pgm`** (what the realistic lidar ray-casts against) | GADEN preprocessing rendered all 180 outlet cells as PGM pixel `1` = **free** | lidar passes through → Cartographer maps free → planner + physics let the robot exit |

Cross-tab that produced this (GADEN state → PGM pixel): `free(0)→1`, `wall(1)→0`,
`outlet(2)→1 for all 180 cells`. The outlets are boundary/vent openings, **not** interior
doorways — the ground-truth evals navigate 4_rooms fine while treating all 180 as walls, which
proves walling them does not block room-to-room passage.

## Chosen approach

**Patch the occupancy map BasicSim loads, at launch time**, so the outlet cells are marked occupied
— the same mechanism `realistic_launch.py` already uses to patch the scene's lidar resolution into a
temp copy. Cartographer then maps a wall there because the sensor *genuinely returns a hit* (honest
SLAM — no a-priori outlet knowledge injected into the map after the fact), BasicSim collision blocks
the robot from exiting, and lidar / physics / planner / ground-truth all agree. This also matches
what the RL policies were trained against (outlets-as-walls).

Rejected alternatives: post-processing the Cartographer map (dishonest; physics still lets the robot
exit); regenerating `occupancy.pgm` at presim (mutates the *shared* `benchmark_env` map, risks
shifting non-realistic baselines); clipping planning to `env_min/env_max` (only masks boundary
outlets, doesn't stop physical exit).

## Design details

### Convention-independent occupied value
BasicSim thresholds via `Map.cpp:56` (`pixel > 255*(1-occupied_thresh)` → Free). Rather than reason
about that threshold or OpenCV's handling of a maxval-1 PGM, **copy the value existing walls already
use**: in every scenario PGM (all `P2`, maxval 1) wall cells are pixel `0`. So marking a cell
occupied = set its pixel to `0`. Robust to any threshold/scaling.

### Coordinate alignment (CSV grid ↔ PGM pixels)
Both grids are the same shape, resolution (0.1), and origin (`[-0.4,-0.4]`), so it's a pure index
map with a possible y-flip:
- GADEN CSV reshapes to `grid[z, y, x]`, cell `(x=0,y=0)` at `env_min` (bottom-left); row `y` grows
  with world +y.
- PGM row 0 = **top** = max y (ROS map_server / image convention). BasicSim confirms this by reading
  with `height - y - 1` (`Map.cpp:57-60`).
- ⇒ outlet at GADEN `(x, y)` → PGM `(row = ny-1-y, col = x)`.

**Self-calibrate the flip** instead of hard-coding it: compute the overlap of `GADEN==1` (walls)
with `PGM==0` (occupied) for both the flipped and unflipped mapping, and adopt whichever maximizes
overlap; then apply that same transform to the `GADEN==2` outlet cells. This auto-corrects if a
scenario's PGM was generated with a different flip, and fails loud if neither orientation aligns
(guard: require ≥ ~80% wall overlap, else skip patch + warn).

### Which z-levels
Outlet footprint is z-invariant (180 cells at z = 1, 5, 10 alike). Union outlets over all z-levels
(`(grid == 2).any(axis=0)`) to get the full 2D footprint — robust to the harness's chosen nav
z-level.

## Implementation steps

### 1. `launch/realistic_launch.py` — add a map-patch helper
In `launch_setup`, after the scene is loaded and `scene["map"]` is resolved to an absolute
`occupancy.yaml` path (currently lines ~93-104), insert a call that rewrites `scene["map"]` to a
patched map:

```python
def _wall_outlets_map(occ_yaml_path, scenario, configuration, namespace):
    """Return path to a temp occupancy.yaml whose PGM has GADEN outlet cells
    (OccupancyGrid3D.csv value 2) marked occupied. Returns the original path
    unchanged if the CSV is missing or wall-overlap calibration fails."""
    cfg_dir = os.path.dirname(occ_yaml_path)
    csv_path = os.path.join(cfg_dir, "OccupancyGrid3D.csv")
    if not os.path.exists(csv_path):
        return occ_yaml_path  # warn: outlets not walled
    # parse CSV header (env_min, num_cells, cell_size) + body (';'-tokenized)
    # build grid[z,y,x]; outlet2d = (grid == 2).any(axis=0); wall2d = (grid == 1).any(axis=0)
    # load P2 pgm (W,H,maxval, pixels); OCC = 0
    # calibrate flip: for m in (pgm, flipud(pgm)): overlap = (m[wall_rows]==OCC).mean()
    #   pick best; require best_overlap >= 0.8 else return original + warn
    # set pgm[row,col] = OCC for outlet cells under the chosen mapping
    # write patched .pgm (P2) + .yaml (image=abs path, copy resolution/origin/thresholds/negate)
    #   to tempfile.gettempdir()/benchmark_env_realistic_map_{scenario}_{configuration}_{ns}.*
    # return patched yaml path
```

Wire it in behind a launch arg (see step 2):

```python
if LaunchConfiguration("wall_outlets").perform(context).lower() in ("true","1"):
    scene["map"] = _wall_outlets_map(scene["map"], scenario, configuration, namespace)
```

Notes:
- Read/write PGM as `P2` ASCII (matches all current scenarios); if `P5` is ever encountered, handle
  the binary body too (a 4-line guard) or fall back to the original with a warning.
- Keep the patched files in `tempfile.gettempdir()` with a scenario/config/namespace-scoped name,
  same pattern as the existing `patched_scene`.
- No dependency on OpenCV in the launch file — plain file I/O + numpy (already imported paths use
  `yaml`; add `numpy`).

### 2. `launch/realistic_launch.py` — new launch argument
```python
DeclareLaunchArgument(
    "wall_outlets", default_value="true",
    description="Mark GADEN outlet cells (OccupancyGrid3D.csv value 2) as occupied in the "
                "map BasicSim's lidar sees, so Cartographer maps them as walls (matches the "
                "ground-truth path). Set false to keep outlets passable.")
```
Default **true** (this is the correctness fix). Off gives the old behavior for A/B.

### 3. gsl_bench passthrough (only if a full sweep must toggle it)
`gsl_bench/gsl_bench/eval/episode_runner.py` builds the launch command (`slam_backend:=` is appended
around line 426). If you want the sweep to control it, add an `--wall-outlets` arg mirroring
`--slam-backend` and append `wall_outlets:=...`. Otherwise the launch default (true) applies and no
gsl_bench change is needed. Record the value in `result.json` meta alongside `slam_backend` for
provenance.

### 4. DirectDrive `topic` / ground-truth-map mode (follow-up, optional)
`motion:=directdrive drive_pose_source:=topic` drives on a static `drive_map_yaml`. If that map is a
ground-truth occupancy, confirm it also walls outlets (the `efe_igdm` loader already does when the
map comes from the service; a hand-supplied `drive_map_yaml` should be the patched one). Not on the
default path — note only.

## Testing / verification

1. **Unit (offline), no ROS:** run the patch helper on `4_rooms_start_c`; assert the 180 outlet
   cells are pixel `0` in the output PGM, wall cells unchanged, free cells unchanged, and the
   calibrated flip reports ≥ 0.8 wall overlap. Repeat for one `10x6_u_left_*` (104×64) to cover a
   non-square map.
2. **Visual:** render original vs patched PGM side by side (reuse `slam_map_captures/` tooling) and
   eyeball that only the openings filled in.
3. **Live SLAM:** launch `4_rooms_start_a` realistic + Cartographer with `wall_outlets:=true`,
   `use_rviz:=true`; confirm the built `/PioneerP3DX/map` now shows walls across the outlet openings
   and the robot no longer plans/exits through them. Compare against `wall_outlets:=false`.
4. **Regression:** one full realistic sweep with the flag on vs off; confirm interior room-to-room
   navigation is unaffected (only boundary openings change) and success/travel metrics for maps
   without exploitable outlets are unchanged.

## Risks & mitigations

- **Wrong flip silently mis-paints** → self-calibration + the ≥0.8 wall-overlap guard; skip-and-warn
  rather than emit a corrupted map.
- **Missing `OccupancyGrid3D.csv`** (scenario not pre-simmed) → return original map + warn; launch
  still succeeds.
- **A scenario where outlets *are* meant to be traversable** → the `wall_outlets:=false` escape hatch
  preserves old behavior; default matches ground truth.
- **Shared-map contamination** → we never write back into the `benchmark_env` share tree; patched
  files live only in tempdir.

## Files touched
- `launch/realistic_launch.py` (helper + launch arg + wiring) — primary.
- `gsl_bench/gsl_bench/eval/episode_runner.py` (optional passthrough + result.json field).
- `README.md` / `NAV2_LOGIC_REVIEW_20260718.md` (document the new arg and the outlet semantics).
