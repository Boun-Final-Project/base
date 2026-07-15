"""
    Launch file to run GADEN gas dispersion simulator.
    IMPORTANT: GADEN_preprocessing should be called before!

    Parameters:
        @param scenario - The scenario where dispersal takes place
        @param simulation - The wind flow actuating in the scenario
        @param source_(xyz) - The 3D position of the release point
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, SetLaunchConfiguration, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory


# ===========================
def launch_arguments():
    return [
        DeclareLaunchArgument(
            "scenario",
            default_value=["10x6_central_obstacle"],
            description="scenario to simulate",
        ),
        DeclareLaunchArgument(
            "configuration",
            default_value=["config1"],
            description="name of the configuration",
        ),
        DeclareLaunchArgument(
            "simulation",
            default_value=["sim1"],
            description="name of the simulation",
        ),
        DeclareLaunchArgument(
            "sim_time",
            default_value=["1000.0"],
            description="gas-dispersion sim length in seconds (saved iters = sim_time / saveDeltaTime)",
        ),
    ]
# ==========================


def launch_setup(context, *args, **kwargs):
    scenario = LaunchConfiguration("scenario").perform(context)
    pkg_dir = LaunchConfiguration("pkg_dir").perform(context)

    params_yaml_file = os.path.join(
        pkg_dir, "ros_params", "gaden_params.yaml"
    )

    return [
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d' + os.path.join(pkg_dir, 'launch', 'gaden.rviz')]
        ),

        # gaden_environment (for RVIZ visualization)
        Node(
            package='gaden_environment',
            executable='environment',
            name='gaden_environment',
            output='screen',
            parameters=[ParameterFile(params_yaml_file, allow_substs=True)]
        ),

        # gaden_filament_simulator (The core). on_exit=Shutdown so the whole launch
        # terminates when the bake finishes (rviz + gaden_environment would otherwise
        # keep it alive forever, blocking presim/go). Does not change the baked gas.
        Node(
            package='gaden_filament_simulator',
            executable='filament_simulator',
            name='gaden_filament_simulator',
            output='screen',
            parameters=[ParameterFile(params_yaml_file, allow_substs=True),
                        # 1000 s * (1/saveDeltaTime 0.5) = 2000 saved iterations — enough
                        # for the longest-start scenario (ultimate starts at iter 700)
                        # plus the robot's run. Override with sim_time:=<sec> if needed.
                        {"sim_time": float(LaunchConfiguration("sim_time").perform(context))},
                        {"runRate": 0.0}
                        ],
            on_exit=Shutdown()
        )
    ]


def generate_launch_description():

    launch_description = [
        # Set env var to print messages to stdout immediately
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),

        SetLaunchConfiguration(
            name="pkg_dir",
            value=[get_package_share_directory("benchmark_env")],
        ),
        SetLaunchConfiguration(name="playback", value="none"),
    ]

    launch_description.extend(launch_arguments())
    launch_description.append(OpaqueFunction(function=launch_setup))

    return LaunchDescription(launch_description)
