# `benchmark_env_realistic` SLAM stall — investigation notes (2026-07-16)

## Goal

Run gpulaika through `gsl_bench`'s `--realistic` mode (`--pose-source tf --map-source slam
--launch benchmark_env_realistic`) — noisy TF pose + online `slam_toolbox` mapping instead
of ground-truth map/pose — and view the resulting SLAM map.

## Status: **resolved** — root cause found and fixed (2026-07-16)

### Root cause: `slam_toolbox` publishes on `/map`, not `/PioneerP3DX/map`

`slam_toolbox`'s `setParams()` defaults `map_name_` to `"/map"` — an **absolute** topic
name. Even though the node is wrapped inside `PushRosNamespace("PioneerP3DX")` in the launch
file, an absolute topic name (leading `/`) *bypasses* namespace remapping entirely.
Therefore `slam_toolbox` was publishing its `OccupancyGrid` on `/map` while
`gsl_bench_runner` was waiting for a message on `/PioneerP3DX/map` (derived from
`/{ns}/map`). `ros2 topic info -v /PioneerP3DX/map` correctly reported 0 publishers — there
was never a publisher on that topic.

This single discrepancy (wrong topic string) is the entire cause of the stall. Everything
else — scan-count off-by-one (#3), lifecycle manager (#1), `transform_timeout` bump (C) —
were real bugs but unrelated to the /map never appearing.

**Fix applied:**
- Added `"map_name": "map"` (relative, no leading `/`) to `slam_toolbox_node`'s inline
  parameters in `nav2_realistic_launch.py`. Under `PushRosNamespace("PioneerP3DX")` the
  topic resolves to `/PioneerP3DX/map`.
- Removed `"transform_publish_period": 0.0` so `slam_toolbox` broadcasts the SLAM-corrected
  `map → PioneerP3DX_odom` TF (default 0.05 s period). Combined with BasicSim's existing
  static ground-truth `map → PioneerP3DX_odom`, the dynamic SLAM-corrected broadcast will
  take priority for timestamps after the first scan. This is the "Option A" coexistence
  approach (no BasicSim source edits).
- Wired `slam_toolbox_params.yaml` contents into `nav2_realistic_params.yaml` under the
  `$(var namespace):` key so `slam_toolbox` actually receives `map_update_interval` (5 s),
  `scan_queue_size` (10, up from default 1 — critical to reduce MessageFilter drops),
  `min_laser_range`/`max_laser_range`, and all Karto scan-matching parameters.

---

## Bugs found and fixed (confirmed real, kept)

### 1. Nav2 `lifecycle_manager` misconfigured to manage `slam_toolbox`
`navigation_config/nav2_realistic_launch.py` listed `"slam_toolbox"` in the lifecycle
manager's `node_names`. `async_slam_toolbox_node` only exposes the `get_state`/`change_state`
lifecycle services when launched with `use_lifecycle_manager: true` (not set anywhere here),
so it self-activates but never advertises those services. `lifecycle_manager` polled
`slam_toolbox/get_state` forever, so nothing downstream (`bt_navigator`, `planner_server`,
etc.) ever activated, and Nav2's `/navigate_to_pose` action server never came up
→ `env_dead` after the harness's 300s wait.

**Fix:** removed `"slam_toolbox"` from `lifecycle_manager`'s `node_names` list. Nav2 now
activates immediately (~1s).

### 2. `odom_tf_republisher.py` crashed on startup
`scripts/odom_tf_republisher.py:21` called `self.declare_parameter('use_sim_time', True)`,
but rclpy auto-declares `use_sim_time` for every node already → `ParameterAlreadyDeclaredException`,
node dies immediately on launch. It was also never wired into `launch/realistic_launch.py`
at all (dead script, no launch entry) — added an entry for it in addition to the parameter fix.

**Fix:** removed the redundant `declare_parameter` call; added a `Node(...)` entry for it in
`realistic_launch.py`. It now runs and successfully republishes a noisy, drifting
`odom → base_link` TF (`tf2_echo` confirms plausible small-noise output).

### 3. `BasicSim` laser scan-count off-by-one (in `src/BasicSim`, upstream repo — see caveat below)
`LaserScanner.cpp:23`: `int numberOfMeasurements = (maxAngleRad - minAngleRad) / angleResolutionRad;`
truncates via the implicit `int` cast. For this scenario's 360°/72-ray config
(`maxAngleRad=6.2832, angleResolutionRad=0.08727`): `6.2832/0.08727 = 71.9977` → truncated to
**71** → `ranges.resize(71)`, but `msg.angle_max`/`msg.angle_increment` are published
unchanged, so any consumer computing the expected count via
`round((angle_max−angle_min)/angle_increment)` gets **72** — a genuine metadata/data-length
mismatch. This is a documented `slam_toolbox`/Karto failure mode for full-360° lidars
(`LaserRangeScan contains N range readings, expected N+1`), known in some cases to make
`/map` "hang and not produce a map" (see Sources below).

**Fix applied:** changed to `std::lround(...)` so the published count is self-consistent
with the metadata. Confirmed: the `"contains 71 range readings, expected 72"` log line is
now gone entirely.

**Caveat:** `BasicSim` is a separate upstream repo (`github.com/PepeOjeda/BasicSim`, same
author as GADEN but a distinct package/git history), not something owned by this benchmark
suite — patching it means the fix must be re-applied after any future upstream sync. User
has asked to explore a config-only replacement for this one (choose
`angleResolutionRad`/`maxAngleRad` values that don't straddle the truncation boundary) but
that has **not** been done yet; the source patch is what's currently in place.

**Result: not sufficient on its own.** After this fix, the scan-count mismatch warning is
gone, but the SLAM stall persists unchanged.

---

## Hypotheses tested and disproven

### A. Deadlock inside `slam_toolbox`
Initial symptom (near-zero CPU, no log activity for minutes, occasional
`"Message Filter dropping message... queue is full"`) looked like a hung worker thread.

**Test:** full `gdb -p <pid> --batch -ex "thread apply all bt"` on the live process (user
ran this with `sudo`, since `ptrace_scope=1` blocks it otherwise).

**Result: disproven.** All 13 threads are in ordinary, healthy idle wait states — DDS
transport threads waiting on `recvfrom`/shared-memory, executor threads blocked in
`rmw_wait` for the next callback, and the one thread inside `slam_toolbox`'s own code
(`SlamToolbox::publishVisualizations()`) is simply sleeping between periodic ticks. Nothing
is stuck in a Ceres solve or waiting on a held lock. **There is no process-level deadlock.**

### B. TF broadcaster conflict (`BasicSim`'s ground-truth TF vs. the new noisy `odom_tf_republisher`)
`Robot.cpp` unconditionally broadcasts its own ground-truth `map→odom` (static) and
`odom→base_link` (dynamic) TF regardless of mode; `odom_tf_republisher` broadcasts a second,
independent `odom→base_link` for the same edge. Hypothesized this dual-authority conflict
could break `tf2_ros::MessageFilter` internals.

**Test:** disabled `odom_tf_republisher` entirely (commented out of the launch), reran,
using only `BasicSim`'s raw ground-truth TF for `odom→base_link` (still has the `map→odom`
static-vs-slam-dynamic conflict, since that one couldn't be isolated this way).

**Result: disproven** (at least for the `odom→base_link` half). Identical stall: scan
registered once, then nothing, ever, no map. Removing the conflict changed nothing.

### C. `slam_toolbox` transform lookup timeout too short
~~Standing theory: `BasicSim.Robot::OnUpdate()` calls `UpdatePose()` then `UpdateSensors()`
using separate `getCurrentTime()` calls, stamping odom/TF slightly before the laser in the
same tick, causing "in the future" TF lookup failures.~~

**Correction (2026-07-16):** `BasicSim::getCurrentTime()` returns the cached member
`currentTime`, which is only updated in `publishClock()` *after* `OnUpdate()` in the main
loop (`BasicSim.cpp:44-47`). So within a single tick, `UpdatePose()` and `UpdateSensors()`
both see the **same** `currentTime` value from the *previous* tick. Timestamps are
consistent within-tick; no skew. The log line
`"Lookup would require extrapolation into the future"` was confirmed (via grep) to come
from `simulated_tdlas` only — never from `slam_toolbox` itself.

**Test:** bumped `slam_toolbox`'s `transform_timeout` from 0.5s to 5.0s.

**Result: disproven** as root cause. However the `transform_timeout` bump (now a permanent
config change) is retained as it helps reduce MessageFilter drops under CPU contention.

**Additional context:** across all test runs the robot was near-stationary at its spawn
pose (the only motion was a single manual `--once` `cmd_vel` rotate). Since
`shouldProcessScan` rejects scans 2–4 unconditionally (`scan_ctr < 5`) and then requires
≥0.45 m of travel, rejecting every scan after the first is *expected, correct behavior*
for a stationary robot — not a bug. The entire stall came down to the `/map` topic name
mismatch alone.

---

## Open questions / not yet tried

- ~~**`sync_slam_toolbox_node` instead of `async_slam_toolbox_node`**~~ — Moot; the root
  cause was the topic name, not the async hand-off path. No need to switch unless new
  symptoms appear.
- ~~**Why drops are bursty, not continuous**~~ — Understood: `scan_queue_size` defaulted to
  1 (the instant fix raised it to 10). Combined with `map_update_interval` 10 s default
  (fixed to 5 s), periodic work is being done and the queue empties, then fills again.
  Not a pathological symptom.
- **Transform publish period coexistence (Option A applied):** `slam_toolbox` now publishes
  dynamic `map → PioneerP3DX_odom` at 20 Hz while BasicSim also publishes a static
  `map → PioneerP3DX_odom` once at startup. tf2 warns about multiple authorities on the
  same edge but the dynamic at newer timestamps takes priority; this is acceptable for
  developer testing. For a production-grade "no ground truth" run, a future patch to
  disable BasicSim's static `map→odom` (e.g. a new `publishGroundTruthMapOdom` YAML flag
  in `Robot.cpp`) would be cleaner.
- **Whether this reproduces outside this heavily-loaded shared machine** — Still worth
  checking with the `scan_queue_size: 10` fix, since MessageFilter drops under contention
  were a real secondary issue. The `/clock` `best_effort` QoS (`BasicSim.cpp:25`) may
  still cause sim-time stalls on a loaded box.
- ~~**The `OnUpdate()` timestamp-ordering fix**~~ — Proven unnecessary since both
  `UpdatePose` and `UpdateSensors` use the same cached `currentTime` value.

## Confirmed working, unaffected by any of this

- `gpulaika` running in **non-realistic** mode (ground-truth ROS `map_server` + sim
  ground-truth pose) — fully unaffected, not part of this investigation.
- Nav2 itself (planner/controller/behavior/bt_navigator) activates and drives correctly
  once the lifecycle-manager bug (#1) is fixed — confirmed via `NavigateToPose` action
  server coming up in ~1s in every run since that fix.
- `odom_tf_republisher`'s noise/drift injection logic itself — confirmed via `tf2_echo`
  showing small, plausible noisy transform values once it stopped crashing (bug #2).

## Sources
- [ros - Slam_toolbox message filter dropping message - Robotics Stack Exchange](https://answers.ros.org/question/357762/slam_toolbox-message-filter-dropping-message/)
- [Slam Toolbox: Message Filter dropping message for reason 'discarding message because the queue is full' - ROS Answers archive](https://answers.ros.org/question/393773/)
- [LaserRangeScan contains different readings · Issue #426 · SteveMacenski/slam_toolbox](https://github.com/SteveMacenski/slam_toolbox/issues/576)

## Log corrections (2026-07-16)

- **`"Lookup would require extrapolation into the future"`** — grep confirmed this line
  comes exclusively from `simulated_tdlas`, never from `slam_toolbox`. The earlier report
  that implied it as a slam_toolbox symptom was an overreach.
- **`process has died [pid ..., exit code -9]`** observed in runs 4–6 — this is `SIGKILL`
  sent by the user during cleanup between test iterations, not a spontaneous crash.
- **Robot was stationary in all runs.** The only motion was a single manual
  `--once` `cmd_vel` rotate nudge (run 5). The harness was stuck in "waiting for SLAM
  map" → `act()` never called → no drive goal → robot never moves. All
  `shouldProcessScan` rejections beyond the first scan are expected for a stationary
  robot, not a slam_toolbox bug.
