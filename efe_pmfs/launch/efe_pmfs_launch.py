"""Launch file for efe_pmfs (dual-mode PMFS GSL).

Usage:
  ros2 launch efe_pmfs efe_pmfs_launch.py                       # no scenario (manual params)
  ros2 launch efe_pmfs efe_pmfs_launch.py scenario:=env_b/sim3  # auto-load map+wind+source
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('efe_pmfs')
    config_file = os.path.join(pkg_share, 'config', 'default.yaml')

    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='',
        description=(
            'test_env scenario name, e.g. "env_b" or "env_b/sim3". '
            'When set, map bbox, mean wind, and true source are auto-loaded.'
        ),
    )

    pmfs_node = Node(
        package='efe_pmfs',
        executable='start',
        name='efe_pmfs_node',
        output='screen',
        parameters=[
            config_file,
            {'scenario': LaunchConfiguration('scenario')},
        ],
    )
    return LaunchDescription([scenario_arg, pmfs_node])
