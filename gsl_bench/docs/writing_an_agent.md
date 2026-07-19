# Writing a gsl_bench agent

You implement two methods. The harness owns everything else: the simulator, physical
motion execution, collision handling, timeouts, termination, and scoring.

Agents default to `motion_mode = "stop_go"`: the next observation is taken after the
current drive finishes. Methods such as ADSM that were designed to revise a goal while
moving may declare `motion_mode = "continuous"`, a `decision_rate_hz`, and a larger
`max_goal_distance`; the harness then updates one running Nav2 action through GoalUpdater.

## 1. The whole thing

```python
import math, random
from gsl_bench.agent import GSLAgent, Observation, Waypoint

class RandomWalkAgent(GSLAgent):
    """Pick a random free direction, step 0.5 m."""

    def __init__(self, config=None):
        self.rng = random.Random((config or {}).get('seed', 0))
        self.obs = None

    def observe(self, obs: Observation) -> None:
        self.obs = obs                      # integrate into your belief/filter/history

    def act(self) -> Waypoint:
        for _ in range(20):                 # rejection-sample a free target
            th = self.rng.uniform(-math.pi, math.pi)
            x = self.obs.x + 0.5 * math.cos(th)
            y = self.obs.y + 0.5 * math.sin(th)
            if self.obs.map.is_free(x, y):
                return Waypoint(x, y, theta=th)
        return Waypoint(self.obs.x, self.obs.y)   # boxed in: stay
```

That is a complete, runnable method. `act()` returns an **absolute map-frame
coordinate**; how the robot gets there is not your problem.

## 2. Lifecycle

```
initialize()                once per process — load weights, warm caches
reset(scenario)             once per episode — map bounds, start pose, step budget
  observe(obs) -> act()     once per decision step, always as a pair
  ...
(episode ends)              success | max_steps | wall_timeout | env_dead | agent_error
```

`reset()` receives a `ScenarioInfo`. It **never contains the source position** — that
is held by the harness for the success check only.

## 3. What you get: `Observation`

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | float | robot position, map frame |
| `theta` | float | heading, rad, world frame |
| `gas_ppm` | float | latest PID reading (0.0 before the first message) |
| `wind_speed` | float | m/s at the robot — **already sqrt-corrected** by the harness |
| `wind_direction` | float | rad from world +x, pointing **downwind** (the source is at `wind_direction + π`) |
| `lidar` | np.ndarray (72,) | metres, ray 0 = sensor-frame 0°, non-finite → max range |
| `lidar_max_range` | float | 3.0 |
| `step` | int | decision-step index, 0-based |
| `sim_time` | float | seconds since episode start |
| `map` | `MapInfo` | static occupancy grid + `is_free(x, y)` |
| `lidar_msg` | LaserScan or None | the raw ROS message, if you need `angle_min`/`angle_increment` |

`MapInfo`: `grid` (H×W, 0 = free), `resolution`, `origin_x`, `origin_y`, `width_m`,
`height_m`, and `is_free(x, y)`.

## 4. What you return: `Waypoint`

`Waypoint(x, y, theta=None)` — an absolute target. `theta` is the arrival heading;
`None` means "face the direction of travel".

The harness then, without telling you:

* **caps the hop** at `max_hop` (default 1.0 m), shortening along your own ray;
* **clamps into free space** — if the target is inside a wall it walks back along the
  ray until a radius-aware free cell is found (and stays put if there is none);
* **drives physically** with Nav2, cancelling after `drive_timeout` seconds if the
  controller wedges.

So the next `observe()` simply shows where the robot really ended up. Do not assume
you arrived at your waypoint — read `obs.x`/`obs.y`.

## 5. Config

Your `__init__` takes one optional `dict`, loaded from the YAML passed to
`--agent-config`:

```yaml
# my_agent.yaml
seed: 7
gas_threshold: 50.0
```

The runner injects `wind_file` (the scenario's wind CSV) automatically, since that is
a property of the scenario rather than of your method.

## 6. Running it

```bash
# by dotted path — no registration needed
ros2 run gsl_bench eval --agent my_pkg.my_agent:MyAgent \
    --scenario 4_rooms_start_a --runs 3

# a whole suite
ros2 run gsl_bench eval --agent my_pkg.my_agent:MyAgent --suite nav7 --runs 5
```

To register a short name from your own package, add an entry point in its `setup.py`
— no fork of gsl_bench required:

```python
entry_points={'gsl_bench.agents': ['my_agent = my_pkg.my_agent:MyAgent']}
```

Then `--agent my_agent` works.

## 7. Rules of the benchmark

* You never see the source position. Success is the harness's judgement:
  `distance(robot, source) < 0.5 m`.
* Motion is always physical Nav2 drive. There is no teleport between steps (the one
  teleport per episode is the initial placement, before step 0).
* Fairness knobs (success radius, step cap, drive timeout, escape on/off) live in the
  harness, not in your agent, and the full block is recorded in every `result.json`.
  `gsl_bench report` prints a loud warning if you try to aggregate runs that were
  scored under different settings.
