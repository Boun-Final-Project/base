# ADSM ROS 2 and `gsl_bench` port

This document describes how ADSM was brought from the authors' ROS 1 repository into
ROS 2 and then into `gsl_bench`, what “faithful” means, and which adaptations belong to
the benchmark rather than the search algorithm.

## Provenance and implementations

The upstream source in this workspace is:

```text
An-adaptive-robot-search-algorithm/
```

It implements Wang et al.'s *An Adaptive Robot Search Algorithm for Balancing
Exploitation and Exploration in Indoor Intermittent Source Seeking*. The relevant
reference implementation is `src/adsm.cpp` plus its grid map, frontier finder, RRT
sampler, goal type, parameters, and ROS 1 `move_base` integration.

There are two ROS 2 forms in the workspace:

1. `base/adsm/src/adsm.cpp` is the monolithic ROS 2 node. It owns subscriptions,
   the ADSM loop, visualization, recording, and Nav2 communication.
2. `base/adsm/src/engine.cpp` is the ROS-independent decision core used by
   `gsl_bench`. `python_bindings.cpp` exposes it as `adsm_core_py`, and
   `gsl_bench/agents/adsm_agent.py` is the thin `GSLAgent` adapter.

The extracted engine exists so ADSM receives the same observation object and uses the
same harness as every other benchmark method. It is not a Python reimplementation of
the algorithm; the decision logic remains C++.

## Porting map

| ROS 1 ADSM responsibility | ROS 2 monolithic port | `gsl_bench` port |
|---|---|---|
| Fixed-rate `loop()` | `Adsm::loop()` | runner schedules `AdsmAgent.act()` at 1 Hz |
| Pose, gas, wind observations | ROS 2 subscriptions | harness `Observation` |
| `observe()` and gas-hit state | `Adsm::observe()` | `Engine::classifyGas()` / `Engine::step()` |
| RRT construction | `RRTSampler` | `Engine::sampleRRT()` |
| EPI/EPR construction | `Adsm::estimate()` | `Engine::estimate()` |
| Source probability | `Adsm::probability()` | `Engine::probability()` |
| Fitness and argmax goal | `Adsm::evaluate()` | `Engine::evaluate()` |
| `move_base::sendGoal()` | Nav2 action plus GoalUpdater | runner action plus GoalUpdater |
| Terminal checks and metrics | node | benchmark harness |
| RViz markers and CSV output | node | harness result and diagnostics |

## Preserved ADSM behavior

The canonical `adsm` agent preserves the following algorithm behavior and paper
defaults:

- a 1 Hz observe/evaluate/act loop;
- PID threshold hysteresis and the sliding sensor window;
- the wind-conditioned source probability equation;
- RRT sampling, collision checks, angular clustering, and frontier counting;
- EPI exploitation candidates and persistent EPR exploration candidates;
- normalized probability and information-gain fitness with `k1 = 0.2`;
- selection of the maximum-fitness candidate every iteration;
- RRT resampling after reaching a goal or after 5.5 seconds;
- stuck detection and the random-exploration branch;
- a 3 m sampling/goal horizon and 0.5 m ADSM goal-reach threshold;
- identity goal orientation, matching the upstream goal message.

The defaults are recorded in `configs/agents/adsm.yaml`. An optional seed controls the
port's random generator and is written to the result metadata for replay.

### Continuous goal replacement is intentional

Upstream ADSM executes this sequence every iteration:

```text
observe -> estimate -> evaluate -> navigate
```

It does **not** wait for the robot to reach the previous goal before evaluating and
sending the current best goal. Candidate scores can therefore make the selected goal
change once per second. This can look unstable in RViz, particularly when EPI and EPR
scores are close, but it is part of the upstream control contract rather than a port
error.

In ROS 1, calling `move_base::sendGoal()` while driving retargeted the running motion.
Naively sending a new Nav2 `NavigateToPose` action every second is not equivalent: it
preempts the action, rebuilds the behavior tree, and repeatedly restarts the controller.
The faithful ROS 2 translation is therefore:

```text
first ADSM decision       -> start one NavigateToPose action
later ADSM decisions     -> publish PoseStamped to /<namespace>/goal_update
Nav2 behavior tree       -> GoalUpdater consumes it and replans the running action
```

For canonical runs, `AdsmAgent` declares `motion_mode = "continuous"`. The evaluator
selects the matching GoalUpdater behavior tree automatically. A healthy episode should
normally show one action start, many goal updates, and no stream of action-preemption
messages.

This preserves ADSM semantics; it does not guarantee that frequent retargeting is ideal
for every Nav2 controller. A future goal-hysteresis or minimum-commitment mode should be
reported as a separate, non-faithful navigation variant rather than silently changing
the canonical agent.

## Benchmark and ROS 2 adaptations

The following differences are deliberate and do not alter ADSM's candidate scoring or
goal-selection rule:

| Adaptation | Reason |
|---|---|
| MOX reading replaced by benchmark PID reading | The benchmark sensor contract provides PID gas concentration. The same high/low hysteresis is retained. |
| Simulator ground-truth pose replaced by SLAM/TF pose | Agents may not consume privileged ground truth. Ground truth remains harness-only for success and metrics. |
| ADSM map input replaced by the live SLAM occupancy grid | All benchmark agents operate from realistic observations. |
| ROS 1 `move_base`/DWA replaced by Nav2/DWB | Required by ROS 2; the faithful profile transplants the relevant controller, tolerance, replanning, and recovery settings. |
| TurtleBot geometry replaced by a 0.25 m radius | Nav2 collision geometry must match the physical BasicSim robot. Retaining the much smaller upstream Burger footprint produces physically impossible doorway paths. |
| Random goals constrained to valid map space | Prevents a port/runtime artifact from sending Nav2 outside the usable benchmark map. |
| Harness owns termination and metrics | Ensures identical source-distance success checks, budgets, and reporting across methods. |

The geometry correction is an execution-model correction, not an algorithm change:
ADSM still chooses the same kind of point, while Nav2 accurately represents whether the
benchmark robot can fit along the path.

## Canonical runtime contract

The registered agent metadata requires:

```text
motion_mode:          continuous
decision_rate_hz:     1.0
pose source:          TF/SLAM
map source:           SLAM
motion backend:       Nav2
navigation profile:   faithful
harness escape:       disabled
```

The evaluator rejects incompatible options instead of silently running a different
experiment. Existing agents retain the default stop-and-go contract: their next
decision occurs after arrival, whereas ADSM's next decision occurs on its next 1 Hz
tick while movement continues.

## Build and run

Build and source both packages:

```bash
cd /home/efe/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select adsm gsl_bench --symlink-install
source install/setup.bash
```

Canonical example:

```bash
ros2 run gsl_bench eval \
  --agent adsm \
  --agent-config src/gsl_bench/configs/agents/adsm.yaml \
  --scenario 4_rooms_start_a \
  --runs 5 \
  --realistic \
  --motion nav2
```

For the older monolithic batch runner, the equivalent faithful selection is explicit:

```bash
cd /home/efe/ros2_ws
AE_ALGOS=adsm AE_ADSM_FAITHFUL=1 bash run_adsm_eesa_benchmark.sh
```

`AE_ADSM_FAITHFUL=1` must pair both halves: the GoalUpdater Nav2 tree and
`use_goal_update:=true` on `adsm_node`.

## Validation

The port is checked at three levels:

1. `base/adsm/test/adsm_equivalence_test.cpp` compares the extracted decision
   calculations with the reference behavior over large randomized inputs, including
   probability, scoring/argmax, and clustering.
2. `gsl_bench/test/test_adsm_agent.py` checks agent capabilities, deterministic replay,
   metadata, and engine integration.
3. `gsl_bench/test/test_motion_modes.py` checks that continuous mode starts one action
   and updates it through GoalUpdater, while stop-and-go agents retain their existing
   behavior.

Useful commands:

```bash
colcon test --packages-select adsm gsl_bench
colcon test-result --verbose
```

Each `result.json` records the random seed, motion mode, decision rate, effective goal
reach, Nav2 action-start count, GoalUpdater publication count, and final ADSM state.
Compare methods using success, simulation time, and traveled distance. An ADSM decision
tick and a stop-and-go agent's completed movement step are not equivalent units.

## Fidelity statement

The canonical `gsl_bench` ADSM agent is faithful to the upstream **search and continuous
retargeting semantics**. It is not a bit-for-bit recreation of the upstream physical
experiment: sensor type, robot body, localization, map source, and ROS navigation stack
are benchmark-controlled adaptations. Those boundaries are intentional, documented,
and kept outside the ADSM scoring and selection logic.
