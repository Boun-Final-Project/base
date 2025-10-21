from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wind_visualizer',
            executable='wind_viz_node',
            name='wind_visualizer_node',
            output='screen',
            parameters=[{
                'sampling_resolution': 1.,  # meters between sample points
                'arrow_scale': 0.3,           # arrow size multiplier
                'update_rate': 1.0,           # Hz
                'min_wind_magnitude': 0.01,   # minimum wind to display
            }]
        ),
    ])
