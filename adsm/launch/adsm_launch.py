from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    slam_map_topic = '/slam_node/slam_map'

    return LaunchDescription([
        DeclareLaunchArgument('source_x', default_value='0.0'),
        DeclareLaunchArgument('source_y', default_value='0.0'),
        DeclareLaunchArgument('source_th', default_value='0.5'),
        DeclareLaunchArgument('iter_rate', default_value='1.0'),
        DeclareLaunchArgument('max_iter', default_value='200'),
        # Oracle-derived travel-distance cap (10x an honest straight-shot Nav2
        # drive to the source). 0.0 = disabled, matches the flat max_iter default.
        DeclareLaunchArgument('distance_budget', default_value='0.0'),
        DeclareLaunchArgument('data_path', default_value='/tmp/adsm_results'),
        DeclareLaunchArgument('visual', default_value='true'),
        DeclareLaunchArgument('k1', default_value='0.2'),

        # Topic remapping arguments
        DeclareLaunchArgument('pose_topic', default_value='/PioneerP3DX/odom'),
        DeclareLaunchArgument('real_pose_topic', default_value='/PioneerP3DX/ground_truth'),
        DeclareLaunchArgument('ref_map_topic', default_value='/PioneerP3DX/map'),
        DeclareLaunchArgument('laser_topic', default_value='/PioneerP3DX/laser_scanner'),
        DeclareLaunchArgument('gas_sensor_topic', default_value='/fake_pid/Sensor_reading'),
        DeclareLaunchArgument('anemometer_topic', default_value='/fake_anemometer/WindSensor_reading'),
        DeclareLaunchArgument('nav_action', default_value='/PioneerP3DX/navigate_to_pose'),
        # Retarget the running drive via Nav2's GoalUpdater instead of re-sending
        # the action goal (which preempts + rebuilds the BT every iteration).
        # Requires a bt_navigator tree containing GoalUpdater. Opt-in because
        # a standard Nav2 tree has no consumer for the update topic.
        DeclareLaunchArgument('use_goal_update', default_value='false'),
        DeclareLaunchArgument('goal_update_topic', default_value='/PioneerP3DX/goal_update'),
        # 0.0 = original ROS1 behaviour (re-roll the stuck-recovery random
        # goal every tick). A positive value is a port-only deviation that
        # A/B-tested worse on House09 (2026-07-16) -- exposed for future
        # experiments only, not recommended.
        DeclareLaunchArgument('random_goal_hold_th', default_value='0.0'),

        # Python SLAM node (from efe_igdm) — publishes the SLAM OccupancyGrid
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

        # ADSM node — subscribes to the Python SLAM map
        Node(
            package='adsm',
            executable='adsm_node',
            name='adsm_node',
            output='screen',
            parameters=[{
                'source_x': LaunchConfiguration('source_x'),
                'source_y': LaunchConfiguration('source_y'),
                'source_th': LaunchConfiguration('source_th'),
                'iter_rate': LaunchConfiguration('iter_rate'),
                'max_iter': ParameterValue(LaunchConfiguration('max_iter'), value_type=int),
                'distance_budget': ParameterValue(
                    LaunchConfiguration('distance_budget'), value_type=float),
                'stuck_duration_th': 60.0,
                'visual': True,
                'data_path': LaunchConfiguration('data_path'),
                'k1': LaunchConfiguration('k1'),

                # Topics
                'pose_topic': LaunchConfiguration('pose_topic'),
                'real_pose_topic': LaunchConfiguration('real_pose_topic'),
                'ref_map_topic': LaunchConfiguration('ref_map_topic'),
                'laser_topic': LaunchConfiguration('laser_topic'),
                'gas_sensor_topic': LaunchConfiguration('gas_sensor_topic'),
                'anemometer_topic': LaunchConfiguration('anemometer_topic'),
                'nav_action': LaunchConfiguration('nav_action'),
                'use_goal_update': ParameterValue(
                    LaunchConfiguration('use_goal_update'), value_type=bool),
                'goal_update_topic': LaunchConfiguration('goal_update_topic'),

                # External SLAM map from Python node
                'external_slam_map_topic': slam_map_topic,

                # RRT parameters
                'rrt_max_iter': 200,
                'rrt_max_r': 3.0,
                'rrt_min_r': 0.70,
                'rrt_step_size': 0.3,
                'obs_r': 0.2,
                'goal_reach_th': 0.5,
                'resample_time_th': 5.5,
                'frontier_search_th': 3.0,
                'random_sample_r': 3.0,
                'goal_cluster_num': 20,

                # Gas sensor parameters (PID sensor, PPM scale)
                'gas_max': 10.0,
                'gas_high_th': 0.3,
                'gas_low_th': 0.1,
                'sensor_window_length': 6.0,

                'random_goal_hold_th': ParameterValue(
                    LaunchConfiguration('random_goal_hold_th'), value_type=float),
            }],
        ),
    ])
