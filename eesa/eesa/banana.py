#!/usr/bin/env python3
"""
Banana — ROS 2 port of banana.py from the EESA package.
Contains Frontier detection, RRT sampling, goal evaluation, and the
main EESA (Exploration-Enhanced Search Algorithm) agent logic.
"""

import math
import random
import os
from timeit import default_timer as timer
from copy import deepcopy
from typing import List
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker as VMarker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point as GPoint

from eesa.map_client import MapClient
from eesa.robot_client import RobotClient


# ======================================================================
# Frontier
# ======================================================================
class FrontierNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Frontier:
    FREE = 0
    UNKNOWN = -1

    def __init__(self, obs_r: float = 0.20):
        self.obs_r = obs_r

    def find(self, map_client: MapClient, mx, my) -> List[FrontierNode]:
        obs_r = round(self.obs_r / map_client.resolution)
        frontiers: List[FrontierNode] = []
        q = []
        visited = np.full((map_client.width, map_client.height), False, dtype=bool)

        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]

        sx, sy = map_client.get_costmap_x_y(mx, my)
        q.append((sx, sy))
        visited[sx, sy] = True

        while len(q) > 0:
            node = q.pop(0)
            if node[0] < 0 or node[0] >= map_client.width or node[1] < 0 or node[1] >= map_client.height:
                continue
            if map_client.get_cost_from_costmap_x_y(node[0], node[1]) == self.UNKNOWN:
                continue
            xmin = max(0, node[0] - obs_r)
            xmax = min(map_client.width, node[0] + obs_r)
            ymin = max(0, node[1] - obs_r)
            ymax = min(map_client.height, node[1] + obs_r)
            data = map_client.grid_data[ymin:ymax, xmin:xmax]
            if data.size > 0 and np.amax(data) > self.FREE:
                continue

            for i in range(4):
                nx, ny = node[0] + dx[i], node[1] + dy[i]
                if nx < 0 or nx >= map_client.width or ny < 0 or ny >= map_client.height:
                    continue
                if not visited[nx, ny]:
                    visited[nx, ny] = True
                    if map_client.get_cost_from_costmap_x_y(nx, ny) == self.UNKNOWN:
                        frontiers.append(FrontierNode(node[0], node[1]))
                    elif map_client.get_cost_from_costmap_x_y(nx, ny) == self.FREE:
                        q.append((nx, ny))

        for node in frontiers:
            node.x, node.y = map_client.get_world_x_y(node.x, node.y)
        return frontiers


# ======================================================================
# RRT
# ======================================================================
class RRTNode:
    def __init__(self, x, y, cost=0.0, parent_ind=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent_ind


class RRTSample:
    FREE = 0

    def __init__(
        self,
        max_iter=200,
        sample_min_r=0.5,
        sample_max_r=3.0,
        step=0.3,
        path_resolution=0.12,
        obs_r=0.20,
        near_r=0.5,
    ):
        self.max_iter = max_iter
        self.max_r = sample_max_r
        self.min_r = sample_min_r
        self.step = step
        self.path_res = path_resolution
        self.obs_r = obs_r
        self.near_r = near_r

    def steer(self, to_node: RRTNode, from_node: RRTNode) -> RRTNode:
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        x = from_node.x + self.step * math.cos(theta)
        y = from_node.y + self.step * math.sin(theta)
        if math.hypot(to_node.x - x, to_node.y - y) <= self.step:
            return RRTNode(to_node.x, to_node.y)
        return RRTNode(x, y)

    def no_obs(self, to_node: RRTNode, from_node: RRTNode, mc: MapClient) -> bool:
        dis = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        n_expand = math.floor(dis / self.path_res)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        path_x = [from_node.x + i * self.path_res * math.cos(theta) for i in range(n_expand)]
        path_y = [from_node.y + i * self.path_res * math.sin(theta) for i in range(n_expand)]
        path_x.append(to_node.x)
        path_y.append(to_node.y)
        for px, py in zip(path_x, path_y):
            cx, cy = mc.get_costmap_x_y(px, py)
            obs_r = int(round(self.obs_r / mc.resolution))
            xmin = int(max(0, cx - obs_r))
            xmax = int(min(mc.width, cx + obs_r))
            ymin = int(max(0, cy - obs_r))
            ymax = int(min(mc.height, cy + obs_r))
            data = mc.grid_data[ymin:ymax, xmin:xmax]
            if data.size == 0:
                return False
            if np.amax(data) > self.FREE:
                return False
            if np.amin(data) < self.FREE:
                return False
        return True

    def sample(self, mc: MapClient, start_x, start_y) -> List[RRTNode]:
        start = RRTNode(start_x, start_y)
        node_list = [start]
        for _ in range(self.max_iter):
            random_theta = random.random() * math.pi * 2 - math.pi
            random_r = self.min_r + random.random() * (self.max_r - self.min_r)
            rx = start.x + random_r * math.cos(random_theta)
            ry = start.y + random_r * math.sin(random_theta)
            dist_list = [
                (i, math.hypot(n.x - rx, n.y - ry)) for i, n in enumerate(node_list)
            ]
            nearest_info = min(dist_list, key=lambda x: x[1])
            nearest = node_list[nearest_info[0]]
            new_node = self.steer(RRTNode(rx, ry), nearest)
            new_node.cost = nearest.cost + nearest_info[1]
            new_node.parent = nearest_info[0]

            if self.no_obs(new_node, nearest, mc):
                dist_list2 = [
                    (i, math.hypot(n.x - new_node.x, n.y - new_node.y))
                    for i, n in enumerate(node_list)
                ]
                near_nodes = [item for item in dist_list2 if item[1] < self.near_r]
                if near_nodes:
                    candidates = [
                        (item[0], node_list[item[0]].cost + item[1])
                        for item in near_nodes
                        if node_list[item[0]].cost + item[1] < new_node.cost
                    ]
                    if candidates:
                        new_parent_info = min(candidates, key=lambda x: x[1])
                        new_parent = node_list[new_parent_info[0]]
                        temp = self.steer(new_node, new_parent)
                        if self.no_obs(temp, new_parent, mc):
                            new_node = temp
                            new_node.cost = new_parent.cost + math.hypot(
                                new_parent.x - new_node.x, new_parent.y - new_node.y
                            )
                            new_node.parent = new_parent_info[0]
                    for item in near_nodes:
                        near_node = node_list[item[0]]
                        cost_t = (
                            math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
                            + new_node.cost
                        )
                        if cost_t < near_node.cost:
                            near_node.cost = cost_t
                            near_node.parent = len(node_list)
                node_list.append(new_node)
        return node_list


# ======================================================================
# Goal node
# ======================================================================
class GoalNode:
    def __init__(self, x, y, yaw=None, u=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.u = u


# ======================================================================
# EESA (Banana) Agent
# ======================================================================
class Banana:
    STATE_UPS = 'Upstreaming'
    STATE_BTK = 'Backtracking'
    STATE_DWS = 'Downstreaming'
    STATE_EPR = 'Exploration'

    def __init__(self, node: Node, mc: MapClient, rc: RobotClient, params: dict):
        self.node = node
        self.iteration = None
        self.mc = mc
        self.rc = rc
        self.params = params
        self.rrt_max_r = params['rrt_max_r']
        self.rrt_min_r = params['rrt_min_r']
        self.reach_waypoint_dis_th = params['reach_waypoint_dis_th']
        self.sigma = params['sigma']
        self.beta = params['beta']
        self.obs_r = params['obs_r']
        self.rrt_client = RRTSample(
            params['rrt_max_iter'],
            sample_min_r=self.rrt_min_r,
            sample_max_r=self.rrt_max_r,
            obs_r=self.obs_r,
        )
        self.frontier_client = Frontier(obs_r=self.obs_r)

        self.do_sample = True
        self.goal: GoalNode = None
        self.gas_hit = False
        self.wind_hit = False
        self.state = self.STATE_EPR
        self.G_epr: List[GoalNode] = []
        self.G_dws: List[GoalNode] = []
        self.G_ups: List[GoalNode] = []
        self.G_dws_last: List[RRTNode] = []
        self.rrt_nodes_raw: List[RRTNode] = []
        self.frontiers_raw: List[FrontierNode] = []

        self.robot_stuck_pose = None
        self.robot_stuck_last_time = self._now()
        self.robot_stuck_max_duration = 5.0
        self.robot_stuck_dis_th = 0.3

        self.cal_start_time = None
        self.cal_end_time = None

        # Distance / timing tracking
        self.total_distance = 0.0
        self.prev_real_x = None
        self.prev_real_y = None
        self.start_time = self._now()

        self.data_log = []
        self.data_log_map = []
        self.data_targets = []
        self.data_rrt_nodes = []
        self.data_frontiers = []

        # Publishers (lazy-init)
        self._pub_goal = None
        self._pub_rrt_tree = None
        self._pub_targets = None
        self._pub_text = None
        self._pub_source = None

        self.node.get_logger().info('EESA agent initialised.')

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------------
    # Source probability estimator  (Eq. 2)
    # ------------------------------------------------------------------
    def probability(self, x, y):
        kQ = 4.0
        D = 1.0
        tau = 250
        V = self.rc.wind_speed
        phi = (
            self.rc.yaw - self.rc.wind_direction
            if self.gas_hit
            else self.rc.yaw - self.rc.wind_direction + math.pi
        )
        lam = math.sqrt(D * tau / (1 + V ** 2 * tau / (4 * D)))
        dis = math.sqrt((x - self.rc.x) ** 2 + (y - self.rc.y) ** 2)
        dx = self.rc.x - x
        dy = self.rc.y - y

        pa = kQ / (4 * math.pi * D * (dis + 1e-4))
        pb = math.exp(-dis / lam)
        pc = math.exp(-dx * V * math.cos(phi) / (2 * D))
        pd = math.exp(-dy * V * math.sin(phi) / (2 * D))
        return pa * pb * pc * pd

    # ------------------------------------------------------------------
    def check_reach_goal(self, x, y):
        return math.hypot(self.rc.x - x, self.rc.y - y) < self.reach_waypoint_dis_th

    def _del_node_reached(self, lst: List[GoalNode]):
        del_list = []
        for ind, node in enumerate(lst):
            if self.check_reach_goal(node.x, node.y):
                del_list.append(ind)
        for ind in sorted(del_list, reverse=True):
            self.node.get_logger().info(
                f'Reach target: idx={ind} x={lst[ind].x:.2f} y={lst[ind].y:.2f}, del'
            )
            del lst[ind]

    # ------------------------------------------------------------------
    # observe -> estimate -> evaluate -> navigate
    # ------------------------------------------------------------------
    def observe(self, iteration):
        self.iteration = iteration
        self.mc.update_grid_map()
        self.rc.update_pose()
        self.rc.update_real_pose()
        self.rc.update_anemometer()
        self.rc.update_gas()

        self.cal_start_time = timer()

        self.gas_hit = self.rc.gas >= self.params['gas_sensor_hit_th']
        self.wind_hit = self.rc.wind_speed >= self.params['anemometer_speed_th']
        self.node.get_logger().info(
            f'gas_hit={self.gas_hit} (gas={self.rc.gas:.4f} >= th={self.params["gas_sensor_hit_th"]}) '
            f'wind_hit={self.wind_hit} (speed={self.rc.wind_speed:.2f} >= th={self.params["anemometer_speed_th"]})'
        )

        if self.goal is None:
            self.goal = GoalNode(self.rc.x, self.rc.y)
        self._del_node_reached(self.G_ups)
        self._del_node_reached(self.G_dws)

        # Determine state (priority)
        self.do_sample = False
        if len(self.G_ups) > 0 or (self.gas_hit and self.wind_hit):
            self.state = self.STATE_BTK
            if self.gas_hit and self.wind_hit:
                self.state = self.STATE_UPS
            if len(self.G_ups) == 0:
                self.do_sample = True
            if self.check_reach_goal(self.goal.x, self.goal.y) and self.gas_hit and self.wind_hit:
                self.do_sample = True
            self.G_dws.clear()
            self.G_epr.clear()
        elif len(self.G_dws) > 0 or self.wind_hit:
            if len(self.G_dws) == 0:
                self.do_sample = True
            self.state = self.STATE_DWS
            self.G_epr.clear()
        else:
            self.do_sample = True
            self.state = self.STATE_EPR

        self.node.get_logger().info(f'State={self.state}')
        self.node.get_logger().info(
            f'Reach goal={self.check_reach_goal(self.goal.x, self.goal.y)} Sample={self.do_sample}'
        )
        self.node.get_logger().info(
            f'Set sizes: G_ups={len(self.G_ups)} G_dws={len(self.G_dws)} G_epr={len(self.G_epr)}'
        )

    # ------------------------------------------------------------------
    def rrt_sample(self) -> List[GoalNode]:
        raw = self.rrt_client.sample(self.mc, self.rc.x, self.rc.y)
        node_list = []
        for node in raw:
            keep = True
            if math.hypot(node.x - self.rc.x, node.y - self.rc.y) < self.rrt_min_r:
                keep = False
                continue
            if self.state == self.STATE_DWS and self.G_dws_last:
                for item in self.G_dws_last:
                    if math.hypot(node.x - item.x, node.y - item.y) < self.reach_waypoint_dis_th:
                        keep = False
                        break
            if keep:
                node_list.append(GoalNode(node.x, node.y))
        self.rrt_nodes_raw = raw
        self.G_dws_last = deepcopy(raw)
        self.node.get_logger().info(f'RRT nodes: {len(node_list)}')
        return node_list

    # ------------------------------------------------------------------
    def estimate(self):
        # Eq. (5) — Upstreaming / Backtracking
        if self.state in (self.STATE_UPS, self.STATE_BTK):
            if self.do_sample:
                self.G_ups = self.rrt_sample()
            for node in self.G_ups:
                node.u = self.sigma * node.u + self.probability(node.x, node.y)

        # Eq. (9) — Downstreaming
        if self.state == self.STATE_DWS:
            if self.do_sample:
                self.G_dws = self.G_dws + self.rrt_sample()
            if not self.G_dws:
                return
            j1_pro = [self.probability(n.x, n.y) for n in self.G_dws]
            j1_max = max(j1_pro) if max(j1_pro) != 0 else 1.0
            j1 = [v / j1_max for v in j1_pro]
            j2_dis = [math.hypot(self.rc.x - n.x, self.rc.y - n.y) for n in self.G_dws]
            j2_max = max(j2_dis) if max(j2_dis) != 0 else 1.0
            j2 = [v / j2_max for v in j2_dis]
            for ind, node in enumerate(self.G_dws):
                j = self.beta * j1[ind] + (1 - self.beta) * j2[ind]
                node.u = self.sigma * node.u + j

        # Eq. (20) — Exploration
        if self.state == self.STATE_EPR:
            if self.do_sample:
                self.frontiers_raw = self.frontier_client.find(self.mc, self.rc.x, self.rc.y)
                self.G_epr = [
                    GoalNode(f.x, f.y, u=math.hypot(f.x - self.rc.x, f.y - self.rc.y))
                    for f in self.frontiers_raw
                ]

        self.node.get_logger().info(
            f'Set sizes: G_ups={len(self.G_ups)} G_dws={len(self.G_dws)} G_epr={len(self.G_epr)}'
        )

    # ------------------------------------------------------------------
    def _generate_random_goal(self) -> GoalNode:
        MAP_FREE = 0
        while True:
            theta = random.random() * math.pi * 2 - math.pi
            r = random.random() * self.rrt_max_r
            x = self.goal.x + r * math.cos(theta)
            y = self.goal.y + r * math.sin(theta)
            cx, cy = self.mc.get_costmap_x_y(x, y)
            if not (0 <= cx < self.mc.width and 0 <= cy < self.mc.height):
                continue
            if self.mc.get_cost_from_costmap_x_y(cx, cy) != MAP_FREE:
                continue
            return GoalNode(x, y)

    # ------------------------------------------------------------------
    def evaluate(self):
        # Stuck detection
        if self.robot_stuck_pose is None:
            self.robot_stuck_pose = (self.rc.real_x, self.rc.real_y)
            self.robot_stuck_last_time = self._now()
        else:
            dis = math.hypot(
                self.robot_stuck_pose[0] - self.rc.real_x,
                self.robot_stuck_pose[1] - self.rc.real_y,
            )
            if dis > self.robot_stuck_dis_th:
                self.robot_stuck_pose = (self.rc.real_x, self.rc.real_y)
                self.robot_stuck_last_time = self._now()
        if self._now() - self.robot_stuck_last_time > self.robot_stuck_max_duration:
            self.node.get_logger().info('Robot stuck -> random goal')
            self.goal = self._generate_random_goal()
            return

        if self.state in (self.STATE_UPS, self.STATE_BTK):
            if self.G_ups:
                self.goal = max(self.G_ups, key=lambda n: n.u)  # Eq. (6)
            else:
                self.state = self.STATE_EPR
                self.node.get_logger().info('Empty G_ups -> Exploration')
        if self.state == self.STATE_DWS:
            if self.G_dws:
                self.goal = min(self.G_dws, key=lambda n: n.u)  # Eq. (10)
            else:
                self.state = self.STATE_EPR
                self.node.get_logger().info('Empty G_dws -> Exploration')
        if self.state == self.STATE_EPR:
            if self.G_epr:
                self.goal = min(self.G_epr, key=lambda n: n.u)  # Eq. (21)
            else:
                self.node.get_logger().info('Empty G_epr -> random goal')
                self.goal = self._generate_random_goal()

        self.node.get_logger().info(f'Goal: x={self.goal.x:.2f} y={self.goal.y:.2f}')

    # ------------------------------------------------------------------
    def navigate(self):
        self.rc.send_goal(self.goal.x, self.goal.y, self.goal.yaw)
        self.cal_end_time = timer()

    def in_motion(self):
        return not self.rc.move_base_done

    # ------------------------------------------------------------------
    # Visualisation (RViz markers, no jsk dependency)
    # ------------------------------------------------------------------
    def visualize(self):
        now_msg = self.node.get_clock().now().to_msg()
        frame = self.mc.map_frame

        # --- Text overlay (as a text Marker) ---
        if self._pub_text is None:
            self._pub_text = self.node.create_publisher(VMarker, 'v_text', 2)
        marker = VMarker()
        marker.ns = 'v_text'
        marker.id = 0
        marker.header.frame_id = frame
        marker.header.stamp = now_msg
        marker.action = VMarker.ADD
        marker.type = VMarker.TEXT_VIEW_FACING
        marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        marker.pose.position.x = self.rc.x if self.rc.x else 0.0
        marker.pose.position.y = (self.rc.y if self.rc.y else 0.0) + 1.5
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.25
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.text = (
            f'It:{self.iteration} St:{self.state}\n'
            f'Wind:{self.wind_hit} Gas:{self.gas_hit}\n'
            f'Ups:{len(self.G_ups)} Dws:{len(self.G_dws)} Epr:{len(self.G_epr)}'
        )
        self._pub_text.publish(marker)

        # --- Source location ---
        if self._pub_source is None:
            self._pub_source = self.node.create_publisher(VMarker, 'v_source', 2)
        marker = VMarker()
        marker.ns = 'v_source'
        marker.id = 0
        marker.header.frame_id = frame
        marker.header.stamp = now_msg
        marker.action = VMarker.ADD
        marker.type = VMarker.LINE_LIST
        marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        shape_v = 4
        angle_off = 2 * math.pi / shape_v
        sx = self.params['source_x']
        sy = self.params['source_y']
        sr = 0.1
        for i in list(range(shape_v)) + [0]:
            p1 = GPoint(x=sx + sr * math.cos(angle_off * i), y=sy + sr * math.sin(angle_off * i), z=0.0)
            p2 = GPoint(x=sx + sr * math.cos(angle_off * (i + 1)), y=sy + sr * math.sin(angle_off * (i + 1)), z=0.0)
            marker.points.append(p1)
            marker.points.append(p2)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        self._pub_source.publish(marker)

        # --- Targets ---
        if self._pub_targets is None:
            self._pub_targets = self.node.create_publisher(VMarker, 'v_targets', 2)
        marker = VMarker()
        marker.ns = 'targets_v'
        marker.id = 0
        marker.header.frame_id = frame
        marker.header.stamp = now_msg
        marker.action = VMarker.ADD
        marker.type = VMarker.POINTS
        marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        lst = {
            self.STATE_UPS: self.G_ups,
            self.STATE_BTK: self.G_ups,
            self.STATE_DWS: self.G_dws,
        }.get(self.state, self.G_epr)
        if lst:
            umin = min(lst, key=lambda n: n.u).u
            umax = max(lst, key=lambda n: n.u).u
            norm = matplotlib.colors.Normalize(vmin=umin, vmax=umax, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
            for n in lst:
                marker.points.append(GPoint(x=n.x, y=n.y, z=0.0))
                c = mapper.to_rgba(n.u)
                marker.colors.append(ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(c[3])))
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
        self._pub_targets.publish(marker)

        # --- Goal ---
        if self._pub_goal is None:
            self._pub_goal = self.node.create_publisher(VMarker, 'v_goal', 2)
        marker = VMarker()
        marker.ns = 'goal_v'
        marker.id = 0
        marker.header.frame_id = frame
        marker.header.stamp = now_msg
        marker.action = VMarker.ADD
        marker.type = VMarker.POINTS
        marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        marker.points.append(GPoint(x=self.goal.x, y=self.goal.y, z=0.0))
        marker.colors.append(ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
        self._pub_goal.publish(marker)

        # --- RRT tree (skip in Exploration state) ---
        if self._pub_rrt_tree is None:
            self._pub_rrt_tree = self.node.create_publisher(VMarker, 'v_rrt_tree', 2)
        if self.state == self.STATE_EPR:
            return
        marker = VMarker()
        marker.ns = 'v_rrt_tree'
        marker.id = 0
        marker.header.frame_id = frame
        marker.header.stamp = now_msg
        marker.action = VMarker.ADD
        marker.type = VMarker.LINE_LIST
        marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        for node in self.rrt_nodes_raw:
            if node.parent is None:
                continue
            pn = self.rrt_nodes_raw[node.parent]
            marker.points.append(GPoint(x=pn.x, y=pn.y, z=0.0))
            marker.points.append(GPoint(x=node.x, y=node.y, z=0.0))
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.01, y=0.01, z=0.01)
        self._pub_rrt_tree.publish(marker)

    # ------------------------------------------------------------------
    # Data recording & saving
    # ------------------------------------------------------------------
    def record_data(self):
        # Accumulate travel distance
        if self.prev_real_x is not None and self.rc.real_x is not None:
            step_d = math.hypot(
                self.rc.real_x - self.prev_real_x,
                self.rc.real_y - self.prev_real_y,
            )
            self.total_distance += step_d
        self.prev_real_x = self.rc.real_x
        self.prev_real_y = self.rc.real_y

        self.data_log.append([
            self.iteration, self._now(), self.state,
            self.cal_start_time, self.cal_end_time,
            self.goal.x, self.goal.y, self.goal.yaw,
            self.rc.x, self.rc.y, self.rc.z, self.rc.yaw,
            self.rc.real_x, self.rc.real_y, self.rc.real_z, self.rc.real_yaw,
            self.rc.gas, self.gas_hit, self.rc.wind_speed, self.wind_hit, self.rc.wind_direction,
            self.params['source_x'], self.params['source_y'], self.do_sample,
            len(self.G_ups), len(self.G_dws), len(self.G_epr),
        ])
        self.data_log_map.append(deepcopy(self.mc.raw_grid_data))
        self.data_targets += [
            (self.iteration, 2, n.x, n.y, n.yaw, n.u) for n in self.G_ups
        ]
        self.data_targets += [
            (self.iteration, 1, n.x, n.y, n.yaw, n.u) for n in self.G_dws
        ]
        self.data_targets += [
            (self.iteration, 0, n.x, n.y, n.yaw, n.u) for n in self.G_epr
        ]
        self.data_frontiers += [
            (self.iteration, f.x, f.y) for f in self.frontiers_raw
        ]
        self.data_rrt_nodes += [
            (self.iteration, i, n.x, n.y, n.parent, n.cost)
            for i, n in enumerate(self.rrt_nodes_raw)
        ]
        if self.cal_end_time and self.cal_start_time:
            self.node.get_logger().info(
                f'Iter {self.iteration} cost={self.cal_end_time - self.cal_start_time:.5f}s'
            )

    def save_data(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        header = [
            'it', 'ros_time', 'state', 'cal_start_time', 'cal_end_time',
            'goal_x', 'goal_y', 'goal_yaw',
            'robot_x', 'robot_y', 'robot_z', 'robot_yaw',
            'robot_real_x', 'robot_real_y', 'robot_real_z', 'robot_real_yaw',
            'gas', 'gas_hit', 'wind_speed', 'wind_hit', 'wind_dir',
            'source_x', 'source_y', 'sample', 'G_ups_size', 'G_dws_size', 'G_epr_size',
        ]
        data_file = os.path.join(file_path, 'data.csv')
        self.node.get_logger().info(f'Save data to {data_file}')
        pd.DataFrame(self.data_log, columns=header).to_csv(data_file, index=False, float_format='%.6f')

        map_file = os.path.join(file_path, 'map_pickle')
        self.node.get_logger().info(f'Save map to {map_file}')
        with open(map_file, 'wb') as f:
            pickle.dump(self.data_log_map, f)

        if self.data_targets:
            tf = os.path.join(file_path, 'targets.csv')
            self.node.get_logger().info(f'Save targets to {tf}')
            pd.DataFrame(
                np.array(self.data_targets, dtype=object),
                columns=['it', 'level', 'x', 'y', 'yaw', 'u'],
            ).to_csv(tf, index=False, float_format='%.6f')

        if self.data_frontiers:
            ff = os.path.join(file_path, 'frontiers.csv')
            self.node.get_logger().info(f'Save frontiers to {ff}')
            pd.DataFrame(
                np.array(self.data_frontiers, dtype=object),
                columns=['it', 'x', 'y'],
            ).to_csv(ff, index=False, float_format='%.6f')

        if self.data_rrt_nodes:
            rf = os.path.join(file_path, 'rrt_nodes.csv')
            self.node.get_logger().info(f'Save rrt_nodes to {rf}')
            pd.DataFrame(
                np.array(self.data_rrt_nodes, dtype=object),
                columns=['it', 'ind', 'x', 'y', 'parent', 'cost'],
            ).to_csv(rf, index=False, float_format='%.6f')
