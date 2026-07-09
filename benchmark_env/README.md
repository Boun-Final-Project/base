# benchmark_env

A gas-source-localization benchmark built on stock **MAPIR gaden** (github.com/MAPIRlab/gaden).
Each scenario is a self-contained gaden project (map + source + wind + robot start).
No gas is shipped in the repo — you bake it locally the first time you use a scenario.

## Requirements

- ROS 2 workspace with stock `gaden` built alongside this package (`gaden_preprocessing`,
  `gaden_filament_simulator`, `gaden_player`, `gaden_environment`).

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select benchmark_env --symlink-install
source install/setup.bash
```

## Run a scenario

Pick a scenario name from `scenarios/` (e.g. `4_rooms_start_a`).

```bash
# 1. Preprocess (generates the occupancy grid) — do this once per scenario
ros2 launch benchmark_env gaden_preproc_launch.py scenario:=4_rooms_start_a

# 2. Bake the gas plume — do this once per scenario (can take a while)
ros2 launch benchmark_env gaden_sim_launch.py scenario:=4_rooms_start_a simulation:=sim1

# 3. Run it (robot + sensors + nav2 + gas playback)
ros2 launch benchmark_env main_simbot_launch.py scenario:=4_rooms_start_a simulation:=sim1
```

Steps 1 and 2 only need to be repeated if you change the scenario's map, source, or wind.

## Even simpler: use the helper scripts

```bash
alias presim='~/ros2_ws/src/benchmark_env/scripts/presim.sh'   # preproc + bake gas
alias runsim='~/ros2_ws/src/benchmark_env/scripts/runsim.sh'   # bake-if-needed + run

presim 4_rooms_start_a
runsim 4_rooms_start_a
```

`runsim` bakes gas automatically the first time, then just launches on later runs.

Or use the terminal GUI to pick a scenario interactively:

```bash
ros2 run benchmark_env benchmark_gui
```

## Adding your own scenario

Copy an existing scenario folder from `scenarios/` into `scenarios/<name>/`, then edit:
- `environment_configurations/*/simulations/*/sim.yaml` — source position
- `environment_configurations/*/config.yaml` → `empty_point` — robot start (must be a free/navigable cell)

Then preprocess + bake as above.
