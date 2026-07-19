# Nav2 Logic Review — `benchmark_env_realistic`

**Date:** 2026-07-18
**Scope:** `navigation_config/nav2_realistic_params.yaml`, `nav2_realistic_params_adsm_faithful.yaml`,
`cartographer_realistic.lua`, `nav2_realistic_launch.py`, `launch/realistic_launch.py`, the two BT
XMLs in `src/base/gaden_config/navigation_config/`, and `src/directdrive_nav/directdrive_nav.py`.

## Which SLAM backend is actually in use

**Cartographer**, not slam_toolbox — the eval runs pass `--slam-backend cartographer` (confirmed in
`Results/gpulaika_20260717_211900/.../result.json`), which loads
`navigation_config/cartographer_realistic.lua`.

**But the code default is `slam_toolbox`** — in `episode_runner.py:582`, `realistic_launch.py:30`,
and `nav2_realistic_launch.py:216` — and **no driver/sbatch/script in the workspace overrides it**;
only the interactive CLI invocation does. Consequences for this review:

- The whole `slam_toolbox:` block in `nav2_realistic_params.yaml` (lines 2–87) and the README's
  "SLAM tuning notes" table describe a backend you don't run. It's **dead config** in the actual
  pipeline — keep it as a fallback, but don't tune against it, and the earlier
  "transform_tolerance vs Ceres solver latency spikes" concern is **moot under Cartographer**
  (Ceres is Karto's solver; Cartographer publishes pose at 200 Hz and its `map→odom` staleness is
  bounded by the lua's `lookup_transform_timeout_sec = 0.2`, not by solver spikes).

## TL;DR

The core wiring is **sound**. In particular the Cartographer TF setup is correct and non-obvious:
`published_frame = PioneerP3DX_odom` + `provide_odom_frame = false` makes Cartographer stack its
correction as `map→odom` (letting BasicSim's native `odom→base_link` complete the chain) instead of
giving `base_link` two TF parents — the lua comment documents that the naive alternative froze Nav2
with spurious collisions. SLAM-vs-lifecycle separation, the GoalUpdater BT, and the DirectDrive
frame math also check out. No "drives through walls" bug.

Findings are **robustness / portability / verify-me**, ranked most-worth-acting-on first:

1. Code default is `slam_toolbox` but you run `cartographer` — a forgotten flag silently degrades the map.
2. Hardcoded absolute paths (BT XMLs + DirectDrive script) — breaks in containers / relocated trees.
3. Cartographer `max_range`/`missing_data_ray_length` (2.9) is manually pinned to the `lidar_max_range` arg (3.0), not derived from it — raising the arg silently stops mapping real walls past 2.9 m.
4. Cartographer frames are hardcoded `PioneerP3DX_*` — any other `namespace:=` silently breaks `map→odom`.
5. Verify the local costmap actually receives the scan (relative `$(var namespace)/laser_scanner` under `PushRosNamespace`).
6. DirectDrive live-map cost + `_nearest_free` snapping edge cases.

---

## What is correct / sound by design

- **Cartographer TF chain** (see TL;DR): `map→odom` from Cartographer, `odom→base_link` from
  `basic_sim`, no double-parent. Correct and the trickiest part of the whole setup.
- **Cartographer scan wiring is clean under the namespace.** The node is launched inside
  `PushRosNamespace`, with `remappings=[("scan","laser_scanner")]`, so it subscribes to the relative
  `laser_scanner` → `/PioneerP3DX/laser_scanner`, matching what `basic_sim` publishes. The
  occupancy-grid node's `("map","map")` → `/PioneerP3DX/map`, matching DirectDrive / costmap
  consumers. (This is cleaner than the slam_toolbox path, which needed an *absolute* scan topic to
  avoid double-namespacing — see #5, which is really only about the Nav2 costmap side now.)
- **The no-hit-ray free-space classification is handled correctly for Cartographer.** BasicSim
  publishes no-hit rays at exactly `maxDistance` (= `lidar_max_range`); the lua's
  `max_range = 2.9` sits strictly below that so those readings classify as *misses* and get
  free-space credit out to `missing_data_ray_length = 2.9`. This is the whole reason Cartographer
  was chosen over Karto (which discards the reading). Correct — modulo the fragile coupling in #3.
- **slam_toolbox excluded from the lifecycle manager**, GoalUpdater BT is a safe superset of the
  stock tree (gsl_bench sends discrete per-step `NavigateToPose` goals and blocks for arrival; only
  the ADSM node publishes to `goal_update`), `yaw_goal_tolerance: 6.28` intentional for position-only
  GSL goals, DirectDrive frame math consistent across live-SLAM and static-PGM. (All as in the first
  pass — unchanged by the backend correction.)

---

## Findings

### 1. The default backend does not match actual use (highest practical value)

`--slam-backend` defaults to `slam_toolbox` everywhere, and every real run must remember to pass
`cartographer`. A single forgotten flag (a new sbatch, a manual repro, a teammate) silently runs
Karto with its documented no-free-space-credit gap at `max_laser_range` — degraded maps, no error,
results that look valid. Make `cartographer` the default (flip the three defaults), or have the
runner refuse to start without an explicit choice. Cheap, removes a whole class of silent-bad-map
runs.

### 2. Hardcoded absolute paths — portability (medium)

Three baked-in `/home/efe/ros2_ws/...` paths:
`nav2_realistic_params.yaml:141` (`adsm_goal_update_bt.xml`),
`nav2_realistic_params_adsm_faithful.yaml:120` (`adsm_faithful_bt.xml`),
`nav2_realistic_launch.py:17` (`DIRECTDRIVE_NAV_SCRIPT`). All exist locally, but the workspace is
containerized (Pyxis/enroot) where the source prefix can differ or be stripped from the install
tree, and these fail at *runtime*. Resolve via `get_package_share_directory(...)` / share-install
the XMLs and script, or an env var.

### 3. Cartographer range coupling is manual and silently breaks if `lidar_max_range` changes (medium)

`lidar_max_range` is a real, plumbed launch arg (`realistic_launch.py:51`, default 3.0) that patches
the scene sensor at launch. But `cartographer_realistic.lua` **hardcodes** `max_range = 2.9` and
`missing_data_ray_length = 2.9` in a static file, with only a comment telling a human to keep them
0.1 below the arg. Raise `lidar_max_range:=5.0` (nothing stops it) and:
- no-hit rays now arrive at 5.0, still > 2.9, still classify as misses — but only credit free space
  out to 2.9 m, not 5 m; and
- **real walls between 2.9 m and 5 m exceed `max_range` and are classified as misses (free space) —
  they never get mapped at all.**

There's no runtime check tying the two together. Either derive the lua values from the arg
(generate the lua at launch like the scene file already is), or assert `lidar_max_range` equals the
value the lua was written for and fail loudly otherwise. Today it works only because both sit at
their defaults.

### 4. Cartographer frames hardcoded to `PioneerP3DX_*` (low, latent)

`cartographer_realistic.lua` hardcodes `tracking_frame`, `published_frame`, `odom_frame` to
`PioneerP3DX_*` (the file documents this as intentional). So `namespace:=` anything else brings up
Cartographer publishing/looking-up the wrong frames → `map→odom` never connects → Nav2 hangs or
reports phantom collisions, silently. Same latent class as the top-level `PioneerP3DX:` param key in
`nav2_realistic_params.yaml` (baseline uses `$(var namespace):`). Default namespace is
`PioneerP3DX` everywhere so neither is active, but both are footguns worth a one-line guard or a
templated lua.

### 5. Verify the *local costmap* actually receives the scan (verify — Nav2 side, backend-independent)

The costmap observation sources use a **relative** `topic: $(var namespace)/laser_scanner`
(`nav2_realistic_params.yaml` lines 267, 293) while the Nav2 nodes run inside
`PushRosNamespace(namespace)`. That can resolve to a doubled `/PioneerP3DX/PioneerP3DX/laser_scanner`
vs. the `/PioneerP3DX/laser_scanner` that `basic_sim` publishes. (The SLAM side is fine — Cartographer
uses a relative *remap* that resolves correctly; this is purely the Nav2 costmap layers.) Identical
pattern is used in the proven non-realistic `benchmark_env/nav2_params.yaml`, so it may resolve
correctly — but if it doesn't, the local voxel/obstacle layer gets no live scan and the robot coasts
on Cartographer's static layer for planning, masking the problem. Confirm during a run:

```bash
ROS_DOMAIN_ID=42 ros2 node info /PioneerP3DX/controller_server | grep laser_scanner
```

If the subscription shows the doubled path, make the costmap `topic` absolute
(`/$(var namespace)/laser_scanner`). Same question for `bt_navigator odom_topic: $(var namespace)/odom`
(line 134).

### 6. DirectDrive live-map cost + `_nearest_free` snapping (low–medium)

`directdrive_nav.py`:
- `_map_cb` runs a full `binary_dilation` on **every** map message and A* over the whole grid per
  goal; on large maps at the SLAM publish rate this can contend with the 10 Hz `_control_tick` under
  the `MultiThreadedExecutor`. Throttle to when goal/map region actually changes.
- `_nearest_free` snaps a blocked start/goal to the global nearest free cell via `np.argwhere(~occ)`;
  a goal just inside a dilated wall can snap to the **other side** of that wall → an inexecutable
  path → burns the full 5 s `goal_timeout` then aborts.
- `_control_tick` drives open-loop toward the next waypoint with no per-tick re-validation against a
  freshly updated map; BasicSim's hard collision makes this *safe* (it stalls to timeout) but slow.

### 7. Minor / cosmetic

- `slam_toolbox:` block (params 2–87), the `amcl:` block (never launched, both files), and the
  README "SLAM tuning notes" table are all inert under Cartographer. Harmless but misleading — a
  reader would reasonably think they're what governs mapping.
- Standard-params DWB `min_vel_x: 0.0` (no reverse) + BasicSim hard-collision = the wedging failure
  mode that motivated DirectDrive; `motion:=nav2` is the weaker choice for tight scenarios, prefer
  `motion:=directdrive` there. (Expected, not a bug.)

---

## Bottom line

Under the backend you actually run (Cartographer), the navigation logic is correctly structured and
the hard parts — the `map→odom` framing and the no-hit-ray free-space classification — are done
right. No functional drive bug. The top action item is #1: **make `cartographer` the default (or
force an explicit choice)** so a forgotten flag can't silently drop you onto the degraded Karto
backend. After that: the portability paths (#2) and the fragile range coupling (#3), then the two
verify-against-a-run items (#4 frames, #5 costmap scan).
