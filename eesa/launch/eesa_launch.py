"""
EESA ROS 2 launch file.
Mirrors the adsm_launch.py pattern: starts the Python SLAM node (efe_igdm)
and then the EESA search agent.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    slam_map_topic = '/slam_node/slam_map'

    return LaunchDescription([
        # ---- Launch arguments ----
        DeclareLaunchArgument('source_x', default_value='0.0'),
        DeclareLaunchArgument('source_y', default_value='0.0'),
        DeclareLaunchArgument('find_source_th', default_value='0.5'),
        DeclareLaunchArgument('iter_rate', default_value='1'),
        DeclareLaunchArgument('max_iter', default_value='360'),
        DeclareLaunchArgument('max_stuck_time', default_value='60.0'),
        DeclareLaunchArgument('data_path', default_value='/tmp/eesa_results'),
        DeclareLaunchArgument('visual', default_value='true'),

        # Topic remapping arguments (same defaults as adsm_launch)
        DeclareLaunchArgument('pose_topic', default_value='/PioneerP3DX/odom'),
        DeclareLaunchArgument('real_pose_topic', default_value='/PioneerP3DX/ground_truth'),
        DeclareLaunchArgument('laser_topic', default_value='/PioneerP3DX/laser_scanner'),
        DeclareLaunchArgument('gas_sensor_topic', default_value='/fake_pid/Sensor_reading'),
        DeclareLaunchArgument('anemometer_topic', default_value='/fake_anemometer/WindSensor_reading'),
        DeclareLaunchArgument('nav_action', default_value='/PioneerP3DX/navigate_to_pose'),

        # Algorithm-specific
        DeclareLaunchArgument('gas_sensor_hit_th', default_value='0.3'),
        DeclareLaunchArgument('anemometer_speed_th', default_value='0.2'),
        DeclareLaunchArgument('sigma', default_value='0.8'),
        DeclareLaunchArgument('beta', default_value='0.7'),

        # ---- Python SLAM node (from efe_igdm) ----
        Node(
            package='efe_igdm',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[{
                'pose_topic': LaunchConfiguration('real_pose_topic'),
                'laser_topic': LaunchConfiguration('laser_topic'),
                'slam_map_topic': slam_map_topic,
                'publish_rate': 2.0,
            }],
        ),

        # ---- EESA node ----
        Node(
            package='eesa',
            executable='eesa_node',
            name='eesa_node',
            output='screen',
            parameters=[{
                # Map
                'map_topic': slam_map_topic,
                'robot_frame': 'PioneerP3DX_base_link',

                # Source / termination
                'source_x': LaunchConfiguration('source_x'),
                'source_y': LaunchConfiguration('source_y'),
                'find_source_th': LaunchConfiguration('find_source_th'),
                'iter_rate': LaunchConfiguration('iter_rate'),
                'max_iter': 360,
                'max_stuck_time': LaunchConfiguration('max_stuck_time'),
                'data_path': LaunchConfiguration('data_path'),
                'visual': True,

                # Topics
                'real_pose_topic': LaunchConfiguration('real_pose_topic'),
                'gas_sensor_topic': LaunchConfiguration('gas_sensor_topic'),
                'anemometer_topic': LaunchConfiguration('anemometer_topic'),
                'nav_action': LaunchConfiguration('nav_action'),

                # Sensor thresholds
                'gas_sensor_hit_th': LaunchConfiguration('gas_sensor_hit_th'),
                'anemometer_speed_th': LaunchConfiguration('anemometer_speed_th'),
                'sensor_window': 4,

                # RRT parameters
                'rrt_max_iter': 200,
                'rrt_max_r': 3.0,
                'rrt_min_r': 0.5,
                'reach_waypoint_dis_th': 0.4,
                'obs_r': 0.2,

                # Algorithm parameters
                'sigma': LaunchConfiguration('sigma'),
                'beta': LaunchConfiguration('beta'),
            }],
        ),
    ])
