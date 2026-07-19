# gsl_bench

A NavSim-style modular benchmark for **gas source localization**. You implement two
functions — `observe()` and `act()` — and the harness handles the rest: simulation,
physical motion execution, collision handling, timeouts, termination, metrics.

```
      your agent                     the harness (gsl_bench)
  ┌────────────────┐    Observation   ┌──────────────────────────────────────┐
  │ observe(obs)   │ ◄─────────────── │ GADEN gas + BasicSim robot + Nav2    │
  │ act() ─────────│ ──────────────►  │ hop cap → clamp to free → drive      │
  └────────────────┘     Waypoint     │ success/step-cap/watchdog → result   │
                                      └──────────────────────────────────────┘
```

## Quickstart

```bash
cd /home/efe/ros2_ws
colcon build --packages-select gsl_bench --symlink-install && source install/setup.bash

# a shipped baseline on one scenario
ros2 run gsl_bench eval --agent random_walk --scenario 4_rooms_start_a --runs 2

# the flagship RL agent on the 7-map suite (its validated recipe needs --escape)
ros2 run gsl_bench eval --agent gpulaika \
    --agent-config src/gsl_bench/configs/agents/gpulaika.yaml \
    --suite nav7 --runs 5 --escape

# re-print the table for any results directory
ros2 run gsl_bench report Results/gpulaika_20260715_010101
```

Writing your own agent: **[writing_an_agent.md](writing_an_agent.md)**. It is ~20 lines.
The ADSM implementation and fidelity boundary are documented in
**[ADSM_PORT.md](ADSM_PORT.md)**.

## What you get

Per run, a `result.json`; per sweep, a `results.csv` and a `report.md`:

| scenario | success | TT (s) | TD (m) | steps | fail modes |
|---|---|---|---|---|---|
| 4_rooms_start_a | 5/5 | 120.4 | 18.2 | 41.0 | — |
| curved_labrinth_left_1 | 4/5 | 402.1 | 63.9 | 155.2 | max_stepsx1 |
| **TOTAL** | 9/10 | 245.7 | 38.6 | 92.4 | max_stepsx1 |

TT = mean time-to-source over successes. TD = mean traveled distance over successes.
Both are meaningless over failures (a failed run's time is just the step cap), so they
are computed over successes only.

## The pieces

| Path | Role |
|---|---|
| `gsl_bench/agent.py` | the public API: `GSLAgent`, `Observation`, `Waypoint`, `MapInfo`, `ScenarioInfo` |
| `gsl_bench/registry.py` | `--agent` lookup: shipped name, entry point, or `pkg.module:Class` |
| `gsl_bench/harness/runner_node.py` | the generic ROS node: sensors → observe/act → Nav2 drive → `result.json` |
| `gsl_bench/harness/obs_cache.py` | latest-value sensor cache; one coherent snapshot per step |
| `gsl_bench/eval/episode_runner.py` | the batch runner: bake, patch, launch, babysit, clean up, report |
| `gsl_bench/eval/metrics.py`, `report.py` | aggregation and the markdown/CSV tables |
| `gsl_bench/agents/` | `random_walk`, `upwind_greedy`, `gpulaika`, `zigzag`, `surge_cast` |
| `configs/harness_default.yaml` | the fairness knobs |

Scenarios come from `benchmark_env` (27+ gaden-project maps), which `gsl_bench` treats
as a read-only dependency.

## Rules of the benchmark

1. **The agent never sees the source.** Success is the harness's call:
   `distance(robot, source) < 0.5 m`.
2. **Motion is always physical Nav2 drive.** No teleport between steps — the single
   teleport per episode is the initial placement, before step 0.
3. **Fairness knobs live in the harness**, not the agent, and the whole block is
   written into every `result.json`. `report` prints a loud warning if you aggregate
   runs scored under different settings.
4. **Escape recovery is opt-in and recorded.** `--escape` turns on the harness's
   stuck-escape (SLAM + frontier). It is OFF by default because it moves scores
   materially (historically 28→32 on the 7-map suite), so a result without the flag
   recorded is uninterpretable.

## Operational notes (learned the hard way)

* **One sweep per machine per domain.** Pass `--domain-id N`; leaked ROS nodes
  republishing on a reused `ROS_DOMAIN_ID` silently poison a later sweep. The runner
  tree-kills the launch descendants (they get their own sessions, so a pgid-only kill
  cannot reach them) and reaps anything left on its own domain.
* **Sim speed is 1× real time.** The gaden speed knob is dead (the BasicSim clock is
  wall-derived); budget wall-clock time accordingly — a 600-step failure is ~20–30 min.
* **The install tree is patched, not the source tree**: no rviz/xterm, 360°/72-ray/3 m
  lidar, tuned nav2 params, looping gas playback. All idempotent, all re-applied every
  run (preproc regenerates `BasicSimScene.yaml`, so the lidar patch must come after it).
