# Running GPULaika with the faithful Nav2 profile

This mode combines three pieces:

- `gsl_bench` loads GPULaika, supplies observations, requests waypoints, and records results.
- `benchmark_env_realistic` launches BasicSim, GADEN, sensors, Cartographer, and Nav2.
- `nav2_realistic_params_adsm_faithful.yaml` controls physical robot movement.

The faithful profile is closer to the original ROS1 `move_base` setup than the standard
profile. In particular, it permits reverse motion (`min_vel_x: -0.22`) and permits faster
rotation (`max_vel_theta: 2.75`). This should make it better at escaping tight wedges, but
it is still an experimental profile: compare results before making it the global default.

## Build and source

The new `--nav-profile` evaluator option and launch arguments require rebuilding both
packages:

```bash
cd /home/efe/ros2_ws
colcon build --packages-select benchmark_env_realistic gsl_bench --symlink-install
source install/setup.bash
```

The shipped GPULaika configuration is:

```text
/home/efe/ros2_ws/src/gsl_bench/configs/agents/gpulaika.yaml
```

That YAML points to the trained `.pt` checkpoint. `gsl_bench` loads the checkpoint; Nav2
does not load or run the neural network.

## Recommended smoke test

Start with one run on one scenario:

```bash
cd /home/efe/ros2_ws
source install/setup.bash

ros2 run gsl_bench eval \
  --agent gpulaika \
  --agent-config src/gsl_bench/configs/agents/gpulaika.yaml \
  --scenario 4_rooms_start_a \
  --runs 1 \
  --realistic \
  --motion nav2 \
  --nav-profile faithful \
  --domain-id 42
```

`--realistic` selects SLAM-derived TF pose, the live SLAM map, and the
`benchmark_env_realistic` launch package. `--motion nav2` is required: the profile has no
effect with DirectDrive. Use a unique domain ID if another ROS experiment is running.

## Run the seven-scenario suite

```bash
cd /home/efe/ros2_ws
source install/setup.bash

ros2 run gsl_bench eval \
  --agent gpulaika \
  --agent-config src/gsl_bench/configs/agents/gpulaika.yaml \
  --suite nav7 \
  --runs 5 \
  --realistic \
  --motion nav2 \
  --nav-profile faithful \
  --domain-id 42
```

Add `--escape` only when intentionally evaluating GPULaika with the harness-level frontier
escape mechanism. It materially changes the method and is recorded in `result.json`, so do
not mix results produced with and without it.

## Launch only the environment

For manual testing without loading GPULaika:

```bash
ros2 launch benchmark_env_realistic realistic_launch.py \
  scenario:=4_rooms_start_a \
  motion:=nav2 \
  nav_profile:=faithful
```

This launches the environment and faithful Nav2 controller, but no model or benchmark
runner. Use `gsl_bench eval` when you want GPULaika to choose and send waypoints.

## Confirm the selected profile

During a run, the following values should report approximately `-0.22` and `2.75`:

```bash
ROS_DOMAIN_ID=42 ros2 param get /PioneerP3DX/controller_server FollowPath.min_vel_x
ROS_DOMAIN_ID=42 ros2 param get /PioneerP3DX/controller_server FollowPath.max_vel_theta
```

Every evaluation result also records `"nav_profile": "faithful"` inside its `harness`
section, preventing accidental comparison with the standard profile.
