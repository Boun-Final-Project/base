import math
import os
import sys
import tempfile

import numpy as np
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
    OpaqueFunction,
    GroupAction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch.frontend.parse_substitution import parse_substitution


def launch_arguments():
    return [
        DeclareLaunchArgument("scenario", default_value="10x6_central_obstacle"),
        DeclareLaunchArgument("configuration", default_value="config1"),
        DeclareLaunchArgument("simulation", default_value="sim1"),
        DeclareLaunchArgument("namespace", default_value="PioneerP3DX"),
        DeclareLaunchArgument("motion", default_value="nav2"),
        DeclareLaunchArgument(
            "nav_profile",
            default_value="faithful",
            description="Nav2 parameter profile: standard or faithful",
        ),
        # DirectDrive frame controls (see nav2_realistic_launch.py). Defaults
        # keep DirectDrive in the SLAM frame; the DirectDrive+ground-truth
        # isolation experiment sets drive_pose_source=topic + a world-frame
        # drive_map_yaml.
        DeclareLaunchArgument("drive_pose_source", default_value="tf"),
        DeclareLaunchArgument("drive_map_yaml", default_value=""),
        # Realistic-mode lidar angular resolution in degrees. The benchmark_env
        # scenario scene files stay untouched (they keep the plain benchmark's 5
        # deg sensor); a patched copy of the scene is generated at launch time.
        # 1 deg matches ADSM's laser and gives Cartographer's scan matcher enough
        # points to stay drift-free.
        DeclareLaunchArgument("lidar_deg", default_value="1.0"),
        # SLAM mapping range. Kept well above the RL policy's trained
        # LIDAR_MAX_RANGE (3.0, see gsl_bench episode_runner.py) — the agent's
        # own observations are clipped back to 3.0 in gsl_bench's obs_cache
        # regardless of the physical sensor's range, so raising this only
        # gives Cartographer more real wall hits instead of full-range
        # "no return" scans.
        DeclareLaunchArgument("lidar_max_range", default_value="3.0"),
        # Default False to keep the automated gsl_bench_runner --realistic
        # sweep headless/unaffected; pass use_rviz:=true for interactive runs.
        DeclareLaunchArgument("use_rviz", default_value="False"),
        # Mark GADEN outlet cells (OccupancyGrid3D.csv value 2) as occupied in
        # the map BasicSim's lidar sees, so Cartographer maps them as walls —
        # matching the ground-truth path, where outlets are non-traversable
        # (see OUTLET_WALL_PLAN.md). Set false to keep outlets passable
        # (old behavior, useful for A/B).
        DeclareLaunchArgument(
            "wall_outlets",
            default_value="true",
            description="Mark GADEN outlet cells (OccupancyGrid3D.csv value 2) as "
            "occupied in the map BasicSim's lidar sees, so Cartographer maps them "
            "as walls (matches the ground-truth path). Set false to keep outlets "
            "passable.",
        ),
    ]


def _read_occupancy_grid_3d_csv(csv_path):
    """Parse GADEN's OccupancyGrid3D.csv into a (nz, ny, nx) uint8 grid.

    Format: 4 `#`-prefixed header lines (env_min, env_max, num_cells,
    cell_size), then for each z-level nx rows of ny space-separated values
    (row index = x, columns = y) followed by a lone ';' separator line.
    Transposed to (nz, ny, nx) so it lines up with PGM (row=y, col=x).
    """
    with open(csv_path) as f:
        lines = f.readlines()
    num_cells_line = lines[2]
    nx, ny, nz = (int(v) for v in num_cells_line.split()[1:4])
    raw = np.zeros((nz, nx, ny), dtype=np.uint8)
    row_idx = 4
    for z in range(nz):
        for x in range(nx):
            raw[z, x, :] = [int(v) for v in lines[row_idx].split()]
            row_idx += 1
        row_idx += 1  # skip the ';' separator line
    return np.swapaxes(raw, 1, 2)  # (nz, ny, nx)


def _read_pgm_p2(pgm_path):
    """Minimal ASCII PGM (P2) reader. Returns (pixels[H,W], maxval)."""
    with open(pgm_path) as f:
        tokens = []
        magic = None
        for line in f:
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            if magic is None:
                magic = stripped
                if magic != "P2":
                    raise ValueError(f"{pgm_path}: unsupported PGM format {magic!r}")
                continue
            tokens.extend(stripped.split())
    width, height, maxval = (int(t) for t in tokens[:3])
    pixels = np.array(tokens[3 : 3 + width * height], dtype=np.uint8).reshape(
        height, width
    )
    return pixels, maxval


def _write_pgm_p2(pgm_path, pixels):
    height, width = pixels.shape
    with open(pgm_path, "w") as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write("1\n")
        for row in pixels:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def _wall_outlets_map(occ_yaml_path, scenario, configuration, namespace):
    """Return path to a temp occupancy.yaml whose PGM has GADEN outlet cells
    (OccupancyGrid3D.csv value 2) marked occupied, so the realistic lidar
    (and hence Cartographer + BasicSim collision) treats them as walls.

    Returns occ_yaml_path unchanged if the CSV is missing or the wall/PGM
    orientation can't be confidently calibrated.
    """
    cfg_dir = os.path.dirname(occ_yaml_path)
    csv_path = os.path.join(cfg_dir, "OccupancyGrid3D.csv")
    if not os.path.exists(csv_path):
        print(
            f"[realistic_launch] wall_outlets: {csv_path} not found, "
            "outlets NOT walled (scenario not pre-simmed?)",
            file=sys.stderr,
        )
        return occ_yaml_path

    grid = _read_occupancy_grid_3d_csv(csv_path)
    wall2d = (grid == 1).any(axis=0)  # (ny, nx)
    outlet2d = (grid == 2).any(axis=0)

    with open(occ_yaml_path) as f:
        occ_yaml = yaml.safe_load(f)
    pgm_path = occ_yaml["image"]
    if not os.path.isabs(pgm_path):
        pgm_path = os.path.join(cfg_dir, pgm_path)
    pixels, maxval = _read_pgm_p2(pgm_path)
    OCC = 0  # existing wall cells in these PGMs (maxval 1) are pixel 0

    ny, nx = wall2d.shape
    wall_ys, wall_xs = np.nonzero(wall2d)

    best_overlap = -1.0
    best_rows = None
    for flip in (False, True):
        rows = (ny - 1 - wall_ys) if flip else wall_ys
        overlap = float(np.mean(pixels[rows, wall_xs] == OCC)) if len(rows) else 0.0
        if overlap > best_overlap:
            best_overlap = overlap
            best_flip = flip

    if best_overlap < 0.8:
        print(
            f"[realistic_launch] wall_outlets: wall/PGM orientation calibration "
            f"failed for {scenario}/{configuration} (best overlap "
            f"{best_overlap:.2f} < 0.8), outlets NOT walled",
            file=sys.stderr,
        )
        return occ_yaml_path

    outlet_ys, outlet_xs = np.nonzero(outlet2d)
    outlet_rows = (ny - 1 - outlet_ys) if best_flip else outlet_ys
    patched_pixels = pixels.copy()
    patched_pixels[outlet_rows, outlet_xs] = OCC

    tag = f"{scenario}_{configuration}_{namespace}"
    patched_pgm = os.path.join(
        tempfile.gettempdir(), f"benchmark_env_realistic_map_{tag}.pgm"
    )
    _write_pgm_p2(patched_pgm, patched_pixels)

    patched_yaml_data = dict(occ_yaml)
    patched_yaml_data["image"] = patched_pgm
    patched_yaml = os.path.join(
        tempfile.gettempdir(), f"benchmark_env_realistic_map_{tag}.yaml"
    )
    with open(patched_yaml, "w") as f:
        yaml.safe_dump(patched_yaml_data, f)

    print(
        f"[realistic_launch] wall_outlets: walled {int(outlet2d.sum())} outlet "
        f"cells in {scenario}/{configuration} (flip={best_flip}, "
        f"overlap={best_overlap:.2f})",
        file=sys.stderr,
    )
    return patched_yaml


def launch_setup(context, *args, **kwargs):
    share_dir = get_package_share_directory("benchmark_env")
    namespace = LaunchConfiguration("namespace").perform(context)
    scenario = LaunchConfiguration("scenario").perform(context)
    simulation = LaunchConfiguration("simulation").perform(context)
    configuration = LaunchConfiguration("configuration").perform(context)

    gaden_player = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("benchmark_env"),
                    "launch",
                    "gaden_player_launch.py",
                )
            ]
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz").perform(context),
            "scenario": scenario,
            "configuration": configuration,
            "simulation": simulation,
        }.items(),
    )

    scene_path = os.path.join(
        share_dir,
        "scenarios",
        scenario,
        "environment_configurations",
        configuration,
        "BasicSimScene.yaml",
    )
    lidar_deg = float(LaunchConfiguration("lidar_deg").perform(context))
    lidar_max_range = float(LaunchConfiguration("lidar_max_range").perform(context))
    with open(scene_path) as f:
        scene = yaml.safe_load(f)
    # BasicSim resolves a relative map path against the scene file's own
    # directory, so the patched copy must carry an absolute path.
    if "map" in scene and not os.path.isabs(str(scene["map"])):
        scene["map"] = os.path.join(os.path.dirname(scene_path), str(scene["map"]))
    if "map" in scene and LaunchConfiguration("wall_outlets").perform(context).lower() in (
        "true",
        "1",
    ):
        scene["map"] = _wall_outlets_map(
            scene["map"], scenario, configuration, namespace
        )
    for robot in scene.get("robots", []):
        robot["publishOdom"] = True
        for sensor in robot.get("sensors", []):
            if sensor.get("type") == "laser":
                sensor["angleResolutionRad"] = round(math.radians(lidar_deg), 7)
                sensor["maxDistance"] = lidar_max_range
    patched_scene = os.path.join(
        tempfile.gettempdir(),
        f"benchmark_env_realistic_scene_{scenario}_{configuration}_{namespace}.yaml",
    )
    with open(patched_scene, "w") as f:
        yaml.safe_dump(scene, f)

    robot_simulator = [
        Node(
            package="basic_sim",
            executable="basic_sim",
            parameters=[
                {"deltaTime": 0.03},
                {"speed": 1.0},
                {"worldFile": patched_scene},
            ],
        )
    ]

    nav2_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("benchmark_env_realistic"),
                    "navigation_config",
                    "nav2_realistic_launch.py",
                )
            ]
        ),
        launch_arguments={
            "namespace": LaunchConfiguration("namespace").perform(context),
            "scenario": LaunchConfiguration("scenario").perform(context),
            "motion": LaunchConfiguration("motion").perform(context),
            "nav_profile": LaunchConfiguration("nav_profile").perform(context),
            "drive_pose_source": LaunchConfiguration("drive_pose_source").perform(context),
            "drive_map_yaml": LaunchConfiguration("drive_map_yaml").perform(context),
        }.items(),
    )

    anemometer = [
        Node(
            package="simulated_anemometer",
            executable="simulated_anemometer",
            name="fake_anemometer",
            parameters=[
                {
                    "sensor_frame": parse_substitution(
                        "$(var namespace)_anemometer_frame"
                    )
                },
                {"fixed_frame": "map"},
                {"noise_std": 0.3},
                {"use_map_ref_system": False},
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="anemometer_tf_pub",
            arguments=[
                "0", "0", "0.5", "1.0", "0.0", "0", "0",
                parse_substitution("$(var namespace)_base_link"),
                parse_substitution("$(var namespace)_anemometer_frame"),
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ]

    PID = [
        Node(
            package="simulated_gas_sensor",
            executable="simulated_gas_sensor",
            name="fake_pid",
            parameters=[
                {"sensor_model": 30},
                {
                    "sensor_frame": parse_substitution(
                        "$(var namespace)_pid_frame"
                    )
                },
                {"fixed_frame": "map"},
                {"noise_std": 20.1},
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="pid_tf_pub",
            arguments=[
                "0", "0", "0.5", "1.0", "0.0", "0", "0",
                parse_substitution("$(var namespace)_base_link"),
                parse_substitution("$(var namespace)_pid_frame"),
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ]

    TDLAS = [
        Node(
            package="simulated_tdlas",
            executable="simulated_tdlas",
            name="simulated_tdlas",
            parameters=[
                {"sensor_model": 30},
                {
                    "sensor_frame": parse_substitution(
                        "$(var namespace)_pid_frame"
                    )
                },
                {"fixed_frame": "map"},
                {"noise_std": 20.1},
                {"use_sim_time": True},
                {"verbose": True},
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tdlas_tf_pub",
            arguments=[
                "0", "0", "0.75", "1.0", "0.0", "0", "0",
                parse_substitution("$(var namespace)_base_link"),
                parse_substitution("tdlas_frame"),
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ]

    namespaced_actions = [PushRosNamespace(namespace)]

    other_actions = [gaden_player]
    other_actions.extend(robot_simulator)
    other_actions.extend(anemometer)
    other_actions.extend(PID)
    # Cartographer (inside nav2_nodes) is a use_sim_time consumer that races
    # basic_sim's /clock at bootstrap: basic_sim is the clock+odom source and
    # publishes /clock best_effort/depth-1 while stamping the first odom->base_link
    # TF at Time(0) before the clock advances. Started concurrently, cartographer
    # intermittently latches non-monotonic odom timestamps (TF_OLD_DATA "from the
    # past") and — with use_imu_data=false — its pose extrapolator wedges forever,
    # never emitting map->odom (robot never localizes, agent takes 0 steps, run
    # wall-times out). Delaying the SLAM/nav stack lets basic_sim reach a steady,
    # monotonic 33 Hz /clock+odom stream before cartographer subscribes.
    other_actions.append(TimerAction(period=6.0, actions=[nav2_nodes]))
    other_actions.extend(TDLAS)
    return [
        GroupAction(actions=namespaced_actions),
        GroupAction(actions=other_actions),
    ]


def generate_launch_description():
    launch_description = [
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
    ]

    launch_description.extend(launch_arguments())
    launch_description.append(OpaqueFunction(function=launch_setup))

    return LaunchDescription(launch_description)
