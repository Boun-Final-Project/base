# SLAM Map Quality Analysis — benchmark_env_realistic vs ADSM reference

**Date:** 2026-07-17
**Reference:** [mwanggh/An-adaptive-robot-search-algorithm](https://github.com/mwanggh/An-adaptive-robot-search-algorithm) (ADSM, ROS1 Noetic + Gazebo + gmapping)
**Data analyzed:** `slam_map_captures/19_postfix.{pgm,yaml}`, `steps/step_00..15`, `steps_right/step_00..15`, ground truth `occupancy.pgm` of `4_rooms_start_a` (install tree)
**Method:** every SLAM capture was reprojected into world frame (SLAM map origin = robot spawn `(11, 2)`, yaw 0) and compared cell-by-cell against the ground-truth grid; a rigid ICP fit (occupied cells → nearest true-wall cell) separates *global frame offset* from *true map defects*.

---

## TL;DR

| Concern | Verdict |
|---|---|
| **Ghost walls** | **Fixed.** After removing the global frame offset, occupied-cell residuals are median 3.6–4.6 cm, p90 ≈ 11 cm, and **0 cells** farther than 0.42 m from a true wall in *every* capture (both runs, all 16 steps + final). No duplicated/displaced wall copies remain. |
| **Low free space ("ray passed but still unknown")** | **Not a mapping bug.** In the final map, enclosed unknown *holes* inside swept free space total **4 cells = 0.01 m²**. All other unknown area (≈6.7 m²) is frontier beyond the 3.0 m mapping range or occlusion shadow behind walls — physically correct, and identical to what ADSM's gmapping produces with the same `maxUrange: 3.0`. |
| **New finding: global yaw anchor race** | The `steps_right` run (and `19_postfix`) is **rigidly rotated +13°** relative to world — constant from `step_00`, i.e. an *initialization* error, not SLAM drift. Root cause is in `odom_tf_republisher.py` (see Finding 1). The `steps` run anchored correctly (−0.1°). |

---

## 1. Ghost walls — quantified

ICP-fit rigid transform (SLAM occupied cells → true walls), then residual distance to nearest true wall:

| capture | fitted rotation | median resid. | p90 | cells > 0.42 m from any true wall |
|---|---|---|---|---|
| `steps` (left run), all 16 steps | −0.1° … −0.0° | 3.6–4.8 cm | ~9 cm | **0 (0.0 %)** |
| `steps_right`, all 16 steps | +12.6° … +13.4° | 4.5–4.8 cm | ~11 cm | **0 (0.0 %)** |
| `19_postfix` (final) | +13.4° | 4.6 cm | 10.6 cm | **0 (0.0 %)** |

Residuals are at grid-quantization level (SLAM 0.05 m vs GT 0.1 m cells). The "double walls" visible in raw overlays are the two faces of the 0.2 m-thick ground-truth walls plus the frame rotation — not phantom copies. **The sentinel-fix + 1° lidar + gates 0.15 + `occupancy_threshold` 0.25 combination has eliminated ghost walls.**

Overlay image: `slam_map_captures/analysis_overlay_19_postfix_vs_gt.png` (black = true walls, red = SLAM occupied, green = SLAM free).

## 2. Free-space coverage — where the "unknown" actually is

Final left-run map (`steps/step_15`), within the SLAM map bounds:

- GT-free cells: 8,560 → SLAM free **76.1 %**, unknown 22.9 %, occupied 1.0 %.
- Unknown decomposition (connected components):
  - **Enclosed holes surrounded purely by free space** (the actual "ray passed but unknown" defect class): **4 cells, 0.01 m²** — negligible.
  - Everything else (2,685 cells, 6.7 m²) touches the map border or sits behind an obstacle: frontier / occlusion shadow.
- Free area grows monotonically over the run (15.8 → 17.2 m²) with no regression; occupied count grows as new wall is discovered. No decay/erasure events.

Why it *looks* sparse: the mapping range is capped at **3.0 m** (`max_laser_range`, matching the sensor), and the start room spans > 6 m — cells farther than 3 m from every scan-insertion pose can never leave unknown, and doorways occlude the rest. ADSM has the **same 3.0 m cap** (`maxUrange: 3.0`), so this is parity, not a deficiency. A mid-run snapshot like `19_postfix` additionally shows fan-shaped unknown wedges simply because only a handful of insertion poses had occurred yet.

**Conclusion: no fix needed for free-space marking.** The no-hit-ray patch in `src/BasicSim/src/LaserScanner.cpp` (publish exactly `range_max`) is verified correct against karto's inclusive range check (`reading == range_max` is kept, `isEndPointValid=false` → free raytrace without endpoint) — it is the slam_toolbox equivalent of ADSM's `LaserScanRangeFilter` clamp (their filter replaces no-hits with 3.2 m, between `maxUrange 3.0` and `maxRange 3.4`, for exactly the same effect in gmapping).

## 3. Finding 1 (needs fix): odom anchor yaw race → whole map rotated

`steps_right` is rotated **+13° from its very first snapshot**, constant thereafter, with perfect internal quality. That signature rules out SLAM drift and points at frame initialization:

[`scripts/odom_tf_republisher.py:71-79`](scripts/odom_tf_republisher.py#L71-L79) anchors the odom frame at the **first `/ground_truth` message it happens to receive**. If the agent starts rotating before that callback fires (launch/subscription race), the odom — and therefore the SLAM map frame — is permanently rotated by whatever yaw the robot had at that instant.

Impact: the agent's own frame-consistent view is unaffected (pose, map, and scans all share the rotated frame — this is honest SLAM behavior), but anything that assumes map ≈ world breaks silently: overlay/coverage evaluation against ground truth, seeding goals in world coordinates, or any world-frame wind/gas quantity consumed in map frame. It also makes run-to-run map comparisons non-reproducible.

**Fix options** (pick one):

1. *(Recommended)* Anchor deterministically from the scenario: read the spawn pose (`BasicSimScene.yaml` `position`/`angle`) via parameters and use it as `_init_*` instead of the first message.
2. Delay motion until the republisher confirms initialization (e.g., episode driver waits for the first `/odom` publish before sending commands).
3. If honest "unknown world frame" is desired, keep the race-free version of the anchor but have `gsl_bench` record the true map→world transform at t=0 for metrics.

## 4. Finding 2 (hygiene): stale `navigation_config/slam_toolbox_params.yaml`

The launch chain loads **only** `nav2_realistic_params.yaml` (verified in `nav2_realistic_launch.py`). `slam_toolbox_params.yaml` is unreferenced and still contains the *old, known-bad* values (`minimum_travel_* : 0.5`, `occupancy_threshold: 0.1`, `min_pass_through: 2`, `map_update_interval: 5.0`). Delete it or overwrite it with the tuned values to prevent someone reviving the phantom-wall regime by accident.

## 5. ADSM parity table

| aspect | ADSM (reference) | benchmark_env_realistic | parity |
|---|---|---|---|
| SLAM backend | gmapping (particle filter, 100 particles) | slam_toolbox (pose graph + Ceres) | different algorithm, equivalent map product; both are "online occupancy SLAM from lidar" |
| map resolution | `delta: 0.05` | `resolution: 0.05` | ✅ |
| mapping range | `maxUrange: 3.0` (sensor 3.5 m) | `max_laser_range: 3.0` (sensor 3.0 m) | ✅ effective range identical |
| no-hit free-space | laser filter clamps to 3.2 m (< `maxRange 3.4`) | BasicSim patch publishes `range_max` | ✅ same mechanism, verified |
| lidar | 360 rays @ 1°, 0.12–3.5 m, 5 Hz, gaussian σ = 1 cm | 360 rays @ 1° (`lidar_deg:=1.0`), 0.1–3.0 m, **no range noise** | ⚠️ minor: BasicSim lidar is noise-free |
| odometry error | Gazebo diff-drive wheel odom | synthetic noise + drift (σ 2 cm / 0.01 rad, drift 0.005/m) | ✅ comparable spirit |
| scan insertion gating | `linearUpdate: 1.0 m`, `angularUpdate: 0.2 rad` | 0.15 m / 0.15 rad | ✅ ours is *denser* (favors coverage) |
| occupancy threshold | gmapping default 0.25 | `occupancy_threshold: 0.25` | ✅ |
| occupancy grid interface | `nav_msgs/OccupancyGrid`, unknown = −1, lethal = 100 (their `gridmap.h`) | same | ✅ |

The single substantive gap is lidar range noise (ADSM: gaussian σ = 1 cm). Adding optional gaussian noise to `LaserScanner.cpp` would close it, but the current setup already injects realism through odom noise/drift, and the noise-free lidar slightly *favors* the baseline-style mapping, so it is defensible either way — just disclose it.

## 6. Action items

1. **Fix the odom yaw-anchor race** in `odom_tf_republisher.py` (Finding 1). This is the only correctness fix needed.
2. **Delete or update** stale `navigation_config/slam_toolbox_params.yaml` (Finding 2).
3. *(Optional, parity)* add gaussian range noise (σ = 1 cm) to the BasicSim lidar to match ADSM's sensor model.
4. *(Optional, docs)* README states "map rotation ≈ 0°" — true for the verified left run, but note the anchor race until item 1 lands.

No changes needed for ghost walls or free-space marking — both verified healthy on all 33 analyzed captures.
