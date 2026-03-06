#!/usr/bin/env python3
"""
EESA main node — ROS 2 port.
Exploration-Enhanced Search Algorithm for Robot Indoor Source Searching.

Wang et al., "An Exploration-Enhanced Search Algorithm for Robot Indoor
Source Searching", IEEE Trans. Robotics, 2024.
"""

import math
import os

import rclpy
from rclpy.node import Node

from eesa.map_client import MapClient
from eesa.robot_client import RobotClient
from eesa.banana import Banana


class FinishCheck:
    """Checks various termination conditions."""

    def __init__(self, node: Node, rc: RobotClient, params: dict):
        self.node = node
        self.rc = rc
        self.params = params
        self.robot_stuck_pose = None
        self.robot_stuck_last_time = None
        self.robot_stuck_dis_th = 0.2
        self.result = ''

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    def check(self, it) -> bool:
        dis = math.hypot(
            self.params['source_x'] - self.rc.real_x,
            self.params['source_y'] - self.rc.real_y,
        )
        self.node.get_logger().info(
            f"Dist to source: {dis:.2f}  find_th={self.params['find_source_th']:.2f}"
        )

        # Found source
        if dis <= self.params['find_source_th']:
            self.result = 'FIND_SOURCE'
            self.node.get_logger().info('STOP: FIND_SOURCE')
            return True

        # Max iterations
        if it >= self.params['max_iter']:
            self.result = 'REACH_MAX_ITER'
            self.node.get_logger().info(f"STOP: MAX_ITER ({it}/{self.params['max_iter']})")
            return True

        # Stuck detection
        if self.robot_stuck_pose is None:
            self.robot_stuck_pose = (self.rc.real_x, self.rc.real_y)
            self.robot_stuck_last_time = self._now()
        else:
            d = math.hypot(
                self.robot_stuck_pose[0] - self.rc.real_x,
                self.robot_stuck_pose[1] - self.rc.real_y,
            )
            if d > self.robot_stuck_dis_th:
                self.robot_stuck_pose = (self.rc.real_x, self.rc.real_y)
                self.robot_stuck_last_time = self._now()
        stuck_d = self._now() - self.robot_stuck_last_time
        self.node.get_logger().info(
            f"Stuck time={stuck_d:.2f}  max={self.params['max_stuck_time']:.2f}"
        )
        if stuck_d > self.params['max_stuck_time']:
            self.result = 'ROBOT_STUCK'
            self.node.get_logger().info('STOP: ROBOT_STUCK')
            return True

        return False


class EesaNode(Node):
    """ROS 2 node that wraps the EESA algorithm."""

    def __init__(self):
        super().__init__('eesa_node')

        # Declare all parameters with defaults
        self.declare_parameter('map_topic', '/slam_node/slam_map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('real_pose_topic', '/PioneerP3DX/ground_truth')
        self.declare_parameter('anemometer_topic', '/fake_anemometer/WindSensor_reading')
        self.declare_parameter('anemometer_speed_th', 0.2)
        self.declare_parameter('gas_sensor_topic', '/fake_pid/Sensor_reading')
        self.declare_parameter('gas_sensor_hit_th', 62500.0)
        self.declare_parameter('sensor_window', 4)
        self.declare_parameter('source_x', 0.0)
        self.declare_parameter('source_y', 0.0)
        self.declare_parameter('find_source_th', 0.5)
        self.declare_parameter('iter_rate', 1)
        self.declare_parameter('max_iter', 360)
        self.declare_parameter('max_stuck_time', 60.0)
        self.declare_parameter('data_path', '/tmp/eesa_results')
        self.declare_parameter('visual', True)
        self.declare_parameter('rrt_max_iter', 200)
        self.declare_parameter('rrt_max_r', 3.0)
        self.declare_parameter('rrt_min_r', 0.5)
        self.declare_parameter('reach_waypoint_dis_th', 0.4)
        self.declare_parameter('sigma', 0.8)
        self.declare_parameter('beta', 0.7)
        self.declare_parameter('obs_r', 0.2)
        self.declare_parameter('nav_action', '/PioneerP3DX/navigate_to_pose')

        # Topic parameters
        self.declare_parameter('pose_topic', '/PioneerP3DX/odom')
        self.declare_parameter('laser_topic', '/PioneerP3DX/laser_scanner')

    def get_params(self) -> dict:
        return {
            'map_topic': self.get_parameter('map_topic').value,
            'robot_frame': self.get_parameter('robot_frame').value,
            'real_pose_topic': self.get_parameter('real_pose_topic').value,
            'anemometer_topic': self.get_parameter('anemometer_topic').value,
            'anemometer_speed_th': float(self.get_parameter('anemometer_speed_th').value),
            'gas_sensor_topic': self.get_parameter('gas_sensor_topic').value,
            'gas_sensor_hit_th': float(self.get_parameter('gas_sensor_hit_th').value),
            'sensor_window': int(self.get_parameter('sensor_window').value),
            'source_x': float(self.get_parameter('source_x').value),
            'source_y': float(self.get_parameter('source_y').value),
            'find_source_th': float(self.get_parameter('find_source_th').value),
            'iter_rate': int(self.get_parameter('iter_rate').value),
            'max_iter': int(self.get_parameter('max_iter').value),
            'max_stuck_time': float(self.get_parameter('max_stuck_time').value),
            'data_path': self.get_parameter('data_path').value,
            'visual': bool(self.get_parameter('visual').value),
            'rrt_max_iter': int(self.get_parameter('rrt_max_iter').value),
            'rrt_max_r': float(self.get_parameter('rrt_max_r').value),
            'rrt_min_r': float(self.get_parameter('rrt_min_r').value),
            'reach_waypoint_dis_th': float(self.get_parameter('reach_waypoint_dis_th').value),
            'sigma': float(self.get_parameter('sigma').value),
            'beta': float(self.get_parameter('beta').value),
            'obs_r': float(self.get_parameter('obs_r').value),
            'nav_action': self.get_parameter('nav_action').value,
        }


def main(args=None):
    rclpy.init(args=args)
    node = EesaNode()
    params = node.get_params()

    node.get_logger().info('Initialising MapClient ...')
    mc = MapClient(node, params['map_topic'])

    node.get_logger().info('Initialising RobotClient ...')
    rc = RobotClient(
        node,
        map_frame=mc.map_frame,
        robot_base_frame=params['robot_frame'],
        real_pose_topic=params['real_pose_topic'],
        anemometer_topic=params['anemometer_topic'],
        gas_sensor_topic=params['gas_sensor_topic'],
        nav_action_name=params['nav_action'],
        sensor_window=params['sensor_window'],
    )

    fc = FinishCheck(node, rc, params)
    agent = Banana(node, mc, rc, params)

    # Build a unique run path
    import uuid
    run_id = str(uuid.uuid4())[:8]
    data_path = params['data_path'] + '/' + run_id

    iter_period = 1.0 / params['iter_rate']  # seconds between iterations
    it = 0

    node.get_logger().info('=== EESA main loop starting ===')
    while rclpy.ok():
        iter_start = node.get_clock().now().nanoseconds / 1e9
        node.get_logger().info(f'ITER {it}  TIME {iter_start:.2f}')

        # Process all pending callbacks
        rclpy.spin_once(node, timeout_sec=0.0)

        agent.observe(it)
        agent.estimate()
        agent.evaluate()
        agent.navigate()

        if params['visual']:
            agent.visualize()
        agent.record_data()

        if fc.check(it):
            agent.save_data(data_path)
            rc.cancel_move()

            # Print final summary
            travel_time = node.get_clock().now().nanoseconds / 1e9 - agent.start_time
            estimation_error = math.hypot(
                rc.real_x - params['source_x'],
                rc.real_y - params['source_y'],
            )
            result_str = 'Success' if fc.result == 'FIND_SOURCE' else fc.result
            summary = (
                '\n'
                '================================================================================\n'
                'EESA - GAS SOURCE LOCALIZATION - FINAL SUMMARY\n'
                '================================================================================\n'
                f'Result:                        {result_str}\n'
                f'ST (Search Time):              {it} steps\n'
                f'TD (Travel Distance):          {agent.total_distance:.2f} m\n'
                f'Travel Time:                   {travel_time:.2f} s\n'
                f'Estimation Error:              {estimation_error:.3f} m\n'
                f'Robot Position:                ({rc.real_x:.3f}, {rc.real_y:.3f})\n'
                f"True Source:                   ({params['source_x']:.3f}, {params['source_y']:.3f})\n"
                '================================================================================'
            )
            node.get_logger().info(summary)
            node.get_logger().info(f'Results saved to {data_path}')
            break

        it += 1

        # Sleep for the remainder of the period while spinning to keep callbacks alive
        try:
            while rclpy.ok():
                elapsed = node.get_clock().now().nanoseconds / 1e9 - iter_start
                if elapsed >= iter_period:
                    break
                rclpy.spin_once(node, timeout_sec=min(0.05, iter_period - elapsed))
        except KeyboardInterrupt:
            break

    node.get_logger().info('EESA node shutting down.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
