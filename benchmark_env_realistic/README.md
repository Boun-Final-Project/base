# benchmark_env_realistic

A **realistic-perception layer** on top of [`benchmark_env`](../benchmark_env/README.md).
Same scenarios, same gas, same robot — but the agent no longer gets ground truth:

| | `benchmark_env` (plain) | `benchmark_env_realistic` |
|---|---|---|
| Map | static ground-truth occupancy grid | **built online by slam_toolbox** from lidar |
| Pose | `/ground_truth` from the simulator | **TF `map → base_link`** as estimated by SLAM |
| Lidar | 5° / 72 rays | **1° / 360 rays** (ADSM parity, configurable) |

It is fully **plug-and-play**: no scenario files are modified. At launch time the
scenario's `BasicSimScene.yaml` is copied to a temp file with the lidar patched, so any
`benchmark_env` scenario works as-is.

## Requirements

- `benchmark_env` built, with the scenario's gas already baked (`presim <scenario>` — see its README).
- The workspace's **patched BasicSim** (upstream BasicSim publishes `range_max + 1` for
  no-hit rays, which slam_toolbox silently discards — no free space would ever be mapped
  along rays into open areas — and it also miscounts partial-FOV rays. Both are fixed in
  `src/BasicSim`).
- `slam_toolbox` and Nav2 (`ros-humble-slam-toolbox`, `ros-humble-navigation2`).

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select benchmark_env_realistic --base-paths src --symlink-install
source install/setup.bash
```

## Run standalone

```bash
ros2 launch benchmark_env_realistic realistic_launch.py \
    scenario:=4_rooms_start_a simulation:=sim1
```

Launch arguments (all optional):

| arg | default | meaning |
|---|---|---|
| `scenario` | `10x6_central_obstacle` | any `benchmark_env` scenario |
| `configuration` / `simulation` | `config1` / `sim1` | gaden project selection |
| `namespace` | `PioneerP3DX` | robot namespace (frames, topics) |
| `motion` | `nav2` | `nav2` (DWB drive) or `directdrive` (rotate-then-go cmd_vel controller, same `navigate_to_pose` action interface) |
| `lidar_deg` | `1.0` | lidar angular resolution in degrees, patched at launch — scene files untouched |
| `use_rviz` | `False` | open the gaden RViz view |

What it starts: `gaden_player` (gas), `basic_sim` (robot physics + patched lidar,
publishing its own `<namespace>_odom` → `<namespace>_base_link` odometry TF),
namespaced `slam_toolbox` (publishes `/<namespace>/map`), the Nav2 stack **or**
DirectDrive, and the gas/wind sensors
(PID, anemometer, TDLAS).

## Run a benchmark episode (gsl_bench)

`--realistic` switches gsl_bench to this package and to honest observations
(`--pose-source tf --map-source slam`): the agent's pose comes from SLAM's TF, its map
(and the optional frontier-escape) uses the **live SLAM grid**, never ground truth.

```bash
ros2 run gsl_bench eval \
    --agent gpulaika --agent-config gpulaika.yaml \
    --scenario 4_rooms_start_a --runs 5 \
    --realistic --motion directdrive --domain-id 42
```

Example `gpulaika.yaml`:

```yaml
arch: dual
checkpoint: /path/to/gpu_cfd_localwind_step05_job25311_upd4000_gaden80_ult85_many0.pt
device: cpu
lidar_frame: heading
local_wind_obs: 1
```

Ground truth is still used *outside* the agent: episode success (distance to true
source) and travel metrics are judged on the simulator's real pose.

## Watch the SLAM map build / save it

A ready-made RViz profile (`navigation_config/slam_watch.rviz`) shows the SLAM map,
laser scan, and TF in a top-down view:

```bash
ROS_DOMAIN_ID=42 rviz2 -d \
    ~/ros2_ws/src/benchmark_env_realistic/navigation_config/slam_watch.rviz
```

```bash
# snapshot to pgm+yaml at any time during a run
ROS_DOMAIN_ID=42 ros2 run nav2_map_server map_saver_cli -f /tmp/slam_snapshot \
    --ros-args -r map:=/PioneerP3DX/map -p save_map_timeout:=15.0
```

### Viewing from another machine (headless server + VNC)

The canonical setup is a persistent virtual display `:99` served over VNC on port 5900
— **not SSH X-forwarding** (forwarded displays have crashed rviz2 and die with the SSH
session). Full guide: [`~/ros2_ws/REMOTE_VIEWING_README.md`](../../REMOTE_VIEWING_README.md).
Short version:

```bash
# server (usually already running — check with: pgrep -af "Xvfb :99|x11vnc")
Xvfb :99 -screen 0 1920x1080x24 &
x11vnc -display :99 -rfbport 5900 -rfbauth ~/.vnc/passwd -localhost -forever -shared -bg

# put rviz on it, matching the run's domain
DISPLAY=:99 ROS_DOMAIN_ID=42 rviz2 -d \
    ~/ros2_ws/src/benchmark_env_realistic/navigation_config/slam_watch.rviz &

# laptop: tunnel + macOS Screen Sharing
ssh -L 5900:localhost:5900 ros-proxy
open vnc://localhost:5900
```

## SLAM tuning notes (why these values)

`navigation_config/nav2_realistic_params.yaml` — the non-default choices that matter:

| param | value | why |
|---|---|---|
| `map_name` | `"map"` | slam_toolbox defaults to the **absolute** topic `/map`, escaping the namespace — anything waiting on `/<ns>/map` hangs forever |
| `max_laser_range` | `3.0` | readings above this are discarded entirely (no free-space raytrace); must match the sensor |
| `minimum_travel_distance/heading` | `0.15` | at 0.05 a scan-match fired every 5 cm/3° and matcher noise random-walked the pose graph (~5° yaw drift per episode → displaced "ghost wall" copies). Coverage doesn't need dense insertion since no-hit rays raytrace free space |
| `occupancy_threshold` | `0.25` | at 0.1 a single bad scan-match painted permanent phantom walls (1–2 hits never overruled by later free passes). Real walls sit near ratio 1.0, so they are unaffected |
| `map_update_interval` | `0.1` | display refresh only (does not gate scan insertion); low value = smooth live view |
| lifecycle | not managed | slam_toolbox is not in the lifecycle manager's `node_names` (it doesn't expose `get_state` unless run as a lifecycle node — listing it deadlocks Nav2 bringup) |

Result on `4_rooms_start_a` (verified against the ground-truth grid): every mapped wall
cell within 0.42 m of a true wall, median 5–6 cm, map rotation ≈ 0°, no phantom cells in
free space. The debugging history behind these values is in
`SLAM_STALL_INVESTIGATION.md` and the before/after images in `slam_map_captures/`.
