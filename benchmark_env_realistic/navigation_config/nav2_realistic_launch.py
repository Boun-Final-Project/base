import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    OpaqueFunction,
    GroupAction,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory
import xacro

DIRECTDRIVE_NAV_SCRIPT = "/home/efe/ros2_ws/src/directdrive_nav/directdrive_nav.py"


def launch_setup(context, *args, **kwargs):
    bench_share = get_package_share_directory("benchmark_env")
    realistic_share = get_package_share_directory("benchmark_env_realistic")
    namespace = LaunchConfiguration("namespace").perform(context)

    nav_profile = LaunchConfiguration("nav_profile").perform(context)
    nav_params_file = {
        "standard": "nav2_realistic_params.yaml",
        "faithful": "nav2_realistic_params_adsm_faithful.yaml",
    }.get(nav_profile)
    if nav_params_file is None:
        raise RuntimeError(
            f"Unknown nav_profile '{nav_profile}'; expected 'standard' or 'faithful'"
        )
    configured_params = ParameterFile(
        os.path.join(realistic_share, "navigation_config", nav_params_file),
        allow_substs=True,
    )
    bt_name = ("adsm_faithful_bt.xml" if nav_profile == "faithful"
               else "adsm_goal_update_bt.xml")
    # gaden_config is a plain source directory, not a built ament package (no
    # package.xml) — get_package_share_directory() can never resolve it.
    bt_xml = "/home/efe/ros2_ws/src/base/gaden_config/navigation_config/" + bt_name
    use_sim_time = True

    robot_desc = xacro.process_file(
        os.path.join(bench_share, "navigation_config", "resources", "giraff.xacro"),
        mappings={"frame_ns": namespace},
    )
    robot_desc = robot_desc.toprettyxml(indent="  ")

    # Cartographer has a genuinely separate returns/misses classification
    # with a real, comfortably-sized gap (see cartographer_realistic.lua's
    # header) — unlike slam_toolbox/Karto, which always sets
    # rangeThreshold == max_laser_range, collapsing its "trace free, no
    # wall" branch to an unreachable 1e-6-wide window.
    # Frame names are hardcoded in the lua file for this project's one
    # namespace, so this only cleanly supports namespace:=PioneerP3DX.
    slam_nodes = [
        Node(
            package="cartographer_ros",
            executable="cartographer_node",
            name="cartographer_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            arguments=[
                "-configuration_directory",
                os.path.join(
                    get_package_share_directory("benchmark_env_realistic"),
                    "navigation_config",
                ),
                "-configuration_basename",
                "cartographer_realistic.lua",
            ],
            remappings=[
                ("scan", "laser_scanner"),
                ("odom", "odom"),
            ],
        ),
        Node(
            package="cartographer_ros",
            executable="cartographer_occupancy_grid_node",
            name="cartographer_occupancy_grid_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"resolution": 0.05},
            ],
            remappings=[("map", "map")],
        ),
    ]

    motion = LaunchConfiguration("motion").perform(context)

    if motion == "directdrive":
        # DirectDrive: no Nav2 stack at all. Cartographer still runs (for the
        # live map + map->odom TF); a lightweight rotate-then-go A* controller
        # replaces bt_navigator/planner_server/controller_server/behavior_server,
        # exposing the SAME NavigateToPose action name gsl_bench's runner_node
        # already calls, so no harness changes are needed.
        #
        # drive_pose_source lets an experiment run DirectDrive in the WORLD frame
        # instead of the SLAM frame — used to isolate the drive mechanism from
        # SLAM perception (DirectDrive + ground-truth pose/map vs the Nav2 +
        # ground-truth baseline). 'topic' = subscribe to the ground_truth pose +
        # a static world-frame map (drive_map_yaml); 'tf' = the default live
        # SLAM pose + SLAM map. In topic mode everything (policy obs, drive
        # target, map, success check) is coherently in world coordinates.
        drive_pose_source = LaunchConfiguration("drive_pose_source").perform(context)
        drive_map_yaml = LaunchConfiguration("drive_map_yaml").perform(context)
        dd_cmd = [
            "python3", DIRECTDRIVE_NAV_SCRIPT, "--ros-args",
            "-p", "use_sim_time:=true",
            "-p", "map_frame:=map",
            "-p", f"base_frame:={namespace}_base_link",
            "-p", f"action_name:=/{namespace}/navigate_to_pose",
            "-p", f"cmd_vel_topic:=/{namespace}/cmd_vel",
            "-p", "robot_radius:=0.25",
        ]
        if drive_pose_source == "topic":
            dd_cmd += [
                "-p", "pose_source:=topic",
                "-p", f"pose_topic:=/{namespace}/ground_truth",
                "-p", f"map_yaml:={drive_map_yaml}",
            ]
        else:
            dd_cmd += [
                "-p", "pose_source:=tf",
                "-p", f"map_topic:=/{namespace}/map",
            ]
        navigation_nodes = list(slam_nodes) + [
            ExecuteProcess(cmd=dd_cmd, output="screen"),
        ]
    else:
        navigation_nodes = list(slam_nodes) + [
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[
                    configured_params,
                    {"use_sim_time": use_sim_time,
                     "default_nav_to_pose_bt_xml": bt_xml},
                ],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[configured_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"autostart": True},
                    {
                        "node_names": [
                            "planner_server",
                            "controller_server",
                            "bt_navigator",
                            "behavior_server",
                        ]
                    },
                ],
            ),
        ]

    visualization_nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"use_sim_time": True, "robot_description": robot_desc}],
        ),
    ]

    actions = [PushRosNamespace(namespace)]
    actions.extend(visualization_nodes)
    actions.extend(navigation_nodes)
    return [GroupAction(actions=actions)]


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument(
                "log_level",
                default_value=["info"],
                description="Logging level",
            ),
            DeclareLaunchArgument("namespace", default_value="PioneerP3DX"),
            DeclareLaunchArgument("motion", default_value="nav2"),
            DeclareLaunchArgument("nav_profile", default_value="faithful"),
            DeclareLaunchArgument("drive_pose_source", default_value="tf"),
            DeclareLaunchArgument("drive_map_yaml", default_value=""),
            DeclareLaunchArgument("scenario", default_value="Exp_C"),
            DeclareLaunchArgument("configuration", default_value="config1"),
            OpaqueFunction(function=launch_setup),
        ]
    )
