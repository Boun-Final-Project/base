"""Launch file for the EFE IGDM (RRT Infotaxis) node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the IGDM node."""
    
    # IGDM node - minimal config to match 'ros2 run efe_igdm start'
    igdm_node = Node(
        package='efe_igdm',
        executable='start',
        output='screen',
    )
    
    return LaunchDescription([
        igdm_node,
    ])
