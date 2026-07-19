# Benchmark driving-method integration

Design record from 2026-07-18 conversation.

## Problem

Three ROS packages in the workspace implement different *motion/driving abstractions* for
gas-source localisation. A realistic benchmark that scores all of them in one harness
requires either unifying the abstraction or accepting that crossing motion models makes
step-count metrics incomparable.

## The three packages and how they drive

### `An-adaptive-robot-search-algorithm` / `src/base/adsm` — the ADSM paper

- **Fixed-rate control loop** at `iter_rate_ = 1 Hz` (`adsm.cpp:103-124`).
- Per iteration: `observe → estimate → evaluate → navigate` → then `rate.sleep()`.
- `navigate()` calls `ac_->sendGoal(goal)` and **returns immediately** (no wait for arrival).
- Waypoints sampled up to ~3 m away (`rrt_max_r=3.0`, `random_sample_r=3.0`).
- Episode budget: `max_iter` iterations at `iter_rate` Hz (e.g. 360 s).
- ROS2 port (`src/base/adsm/launch/adsm_launch.py`) adds `use_goal_update:=true` to
  avoid BT rebuilds per tick (GoalUpdater inside the Nav2 behaviour tree).
- **A "step" = one iteration of the decision loop**, regardless of motion completion.

### `gsl_bench` `RunnerNode` — every other agent (gpulaika, random_walk, surge-cast...)

- **Pose-callback tick**, rate-gated by `step_delay` (default 0.5 s) (`runner_node.py:135,532-536`).
- Per step: `observe → act`, then **hop-cap to `max_hop=1.0` m** (`runner_node.py:775-777`),
  then Nav2 drive, then **block in `_is_moving=True`** until Nav2 completes or
  `drive_timeout=30 s` (`runner_node.py:512-516,543-563`).
- `_step` is **incremented only on goal completion** (line 738).
- Episode budget: `max_steps=600` hops.
- **A "step" = one successfully executed Nav2 hop**.

### `benchmark_env_realistic` — the realistic-perception layer

- Launches `slam_toolbox` (online SLAM), Nav2 (DWB controller), gas/wind sensors.
- Two Nav2 config variants already exist in `navigation_config/`:
  - `nav2_realistic_params.yaml` (default) — DWB tuned for SLAM drift-free mapping:
    `max_vel_x=0.26`, `max_vel_theta=1.0`, `min_speed_xy=0.0`, `xy_goal_tol=0.25`,
    `inflation_radius=0.55`.
  - `nav2_realistic_params_adsm_faithful.yaml` — transplant of ADSM's TurtleBot3 DWA
    values: `max_vel_x=0.22`, `max_vel_theta=2.75`, `min_speed_xy=0.11`,
    `xy_goal_tol=0.05`, `inflation_radius=1.0`. Pairs with `adsm_faithful_bt.xml`
    (GoalUpdater tree).

## The three axes of mismatch

| axis | ADSM paper (ROS1 faithful) | gsl_bench harness (RL/random) |
|---|---|---|
| **step rule** | fixed 1 Hz iterate-and-supersede; never blocks | blocks per Nav2 hop; counter advances only on goal completion |
| **hop length** | up to ~3 m | capped at **1.0 m** along the ray |
| **Nav2 controller** | `max_vel_x=0.22`, `θ=2.75`, `min_speed_xy=0.11`, `sim_time=1.5` | `max_vel_x=0.26`, `θ=1.0`, `min_speed_xy=0.0`, `sim_time=1.7` |
| **goal transport** | preempting `sendGoal` or GoalUpdater (ROS2 faithful) | fresh `NavigateToPose` action per step → BT rebuild + rotate-to-heading |

The third axis alone is the smaller issue (changing a YAML fixes it); the first two are
paradigm-level and require a harness design change.

## Two motion models — cannot share one loop

### gpulaika (and RL-trained policies): stop-and-go

```
pose_cb → cache update → if _is_moving: check timeout; return ← block gate
        ↓ (when not _is_moving AND step_delay elapsed)
    _take_step → agent.observe(cache.snapshot)   ← at settled pose
               → agent.act() → Waypoint
               → _cap_hop(1.0 m) → _clamp_to_free → _drive(...)
                                    ↓
                        _is_moving=True; Navigator.send_goal(NavigateToPose)
                        ↓ (goal completes async)
                        _on_nav_complete → _is_moving=False
                        ↓ (next pose_cb)
                        observe at_arrival_pose ...
```

Why this is load-bearing:
- `_action_to_target( … STEP_SIZE)` produces a post-step heading; the policy's
  expected next observation matches that heading. Mid-rotation observe breaks
  the train/eval contract (`gpulaika_agent.py:33,58,126-127`).
- `b.record_step()` advances gas-history once per accepted arrival
  (`gpulaika_agent.py:97-98`). Mid-motion extra records desync the trace.
- Lidar is heading-frame: ray 0 = body forward. Mid-turn, the lidar would be
  orientation-incoherent with the action just taken.

### ADSM: continuous decide-supersede

```
while ros::ok():                      ← iter_rate=1 Hz wall-clock
    ros::spinOnce()                   ← caches drain async
    observe()                         ← reads current cache (not arrival-conditioned)
    estimate()                        ← RRT if goal reached OR 5.5 s elapsed
    evaluate()                        ← pick argmax(j) over candidate goals
    navigate()                        ← ac_->sendGoal(goal) → returns immediately
    iter_++; rate.sleep()
```

Why this is load-bearing:
- `probability()` uses live wind direction *at decision time* (`adsm.cpp:199`).
  Blocking on arrival would let gas/wind go stale.
- Re-sample is gated by wall-clock time, not by arrival (`adsm.cpp:248`).
- Waypoints are 3-m casts; 1.0-m hop-cap would clip each goal to 1/3 intent.
- Without GoalUpdater, the BT rebuild + rotate-to-heading per tick was measured at
  "3 cm/s advance, stuck-watchdog trips" (`adsm_goal_update_bt.xml:11-13`).

## Agreed direction

### Step definition → drop "step" as the budget; use sim-time + distance

**(Choice C from conversation.)** Both the harness and the ADSM launch already support
`max_sim_time_s` and `max_travel_distance_m`. These become the primary per-scenario
budgets (oracle-derived). `max_steps` degrades to a diagnostic runaway guard.
TTS is in sim-seconds; PE is in metres; PSC is wall-ms. All motion-mode agnostic.

### Nav2 config → `*_adsm_faithful` + GoalUpdater BT, with preconditions

**(Recognised as not fully independent of the motion-mode fork.)**
Canonical config adopted as `nav2_realistic_params_adsm_faithful.yaml` +
`adsm_faithful_bt.xml`, **after two preconditions**:
1. Swap footprint from TB3-burger to P3DX (`robot_radius=0.22`),
   reduce `inflation_radius` from 1.0 to 0.55–0.65.
2. One-run SLAM stability check on the canonical scenario (`4_rooms_start_a`):
   confirm median map error ≤10 cm, no phantom-wall cells.

The GoalUpdater BT is a strict superset of the stock tree; stop_go agents never
publish to `goal_update` and are unaffected.

### ADSM integration → port as a first-class `GSLAgent` (not IPC)

Pure-Python port of `adsm.cpp`'s `observe/estimate/evaluate/navigate` into
`gsl_bench/agents/adsm_agent.py`, reusing `efe_igdm`'s `OccupancyGridMap` and
RRT/frontier helpers. Follows the suite's convention (cf. `gpulaika_agent.py`).
Declares `motion_mode='continuous'`, `iter_rate=1.0`, `hop_reach=3.0`.

### Motion-mode fork → add fields to `GSLAgent` ABC, branch in `RunnerNode`

New class-level fields on `GSLAgent` (`gsl_bench/agent.py`):
- `motion_mode: str = 'stop_go'` — `'stop_go'` or `'continuous'`
- `iter_rate: float = 1.0` — target decision rate (Hz) for continuous
- `hop_reach: float = 1.0` — per-agent hop-cap request; harness still validates
  against `--max-hop`
- `needs_arrival_obs: bool = True` — gate observe() on a fresh post-drive scan

All existing agents inherit defaults → byte-identical. New `AdsmAgent` overrides.

Changes in `runner_node.py`:
- `_pose_callback`: continuous mode falls through `_is_moving` instead of
  early-returning; rate-gates by `1.0 / agent.iter_rate` instead of
  `step_delay`.
- `_drive`: continuous mode publishes `PoseStamped` to `/{ns}/goal_update`
  (GoalUpdater) instead of sending a fresh `NavigateToPose` when already
  moving. First call always sends fresh `NavigateToPose`; completion callback
  sets `_is_moving=False` so that when the stream pauses, the next call
  restarts the tree cleanly.
- `_cap_hop` uses `min(harness_max_hop, agent.hop_reach)`.
- Result schema (`write_result`) stamps `motion_mode`, `iter_rate`,
  `max_sim_time_s`, `max_travel_distance_m`.

### Fairness guard extends

`gsl_bench/eval/metrics.py` extended to flag cross-run differences in:
`motion_mode`, `iter_rate`, `max_sim_time_s`, `max_travel_distance_m`,
`nav_params_yaml`, `bt_xml`.

### Oracle updated

`gsl_bench/eval/oracle.py` derives `max_sim_time_s` and `max_travel_distance_m`
instead of `max_steps`, using the same geodesic + drive-time estimate logic.

## Files to modify / create (in execution order)

| # | File | What |
|---|---|---|
| 1 | `benchmark_env_realistic/navigation_config/nav2_realistic_params_adsm_faithful.yaml` | Precondition #1: P3DX footprint, inflation_radius 0.55 |
| 2 | `benchmark_env_realistic/navigation_config/nav2_realistic_launch.py` | Precondition #2 (manual check); then load `_adsm_faithful.yaml` + `adsm_faithful_bt.xml` |
| 3 | `gsl_bench/agent.py` | Add `motion_mode`, `iter_rate`, `hop_reach`, `needs_arrival_obs` |
| 4 | `gsl_bench/harness/runner_node.py` | Fork `_pose_callback`, `_drive`, `_on_nav_complete`; `_goal_update_pub`; relax `_cap_hop`; record new fields |
| 5 | `gsl_bench/eval/oracle.py` | Derive `max_sim_time_s`, `max_travel_distance_m` |
| 6 | `gsl_bench/eval/episode_runner.py` | Promote `--max-sim-time-s`/`--max-travel-distance-m` to primary budgets; raise `--max-steps` to guard |
| 7 | `gsl_bench/eval/metrics.py` | Extend fairness guard with new block fields |
| 8 | `gsl_bench/agents/adsm_agent.py` (new) | Pure-Python port of ADSM decision loop; `motion_mode='continuous'` |
| 9 | `benchmark_env/navigation_config/nav2_params.yaml` | Same BT swap for ground-truth path (cross-package consistency) |
