"""
RViz markers for efe_pmfs.

Adapted from efe_igdm/visualization/marker_visualizer.py. Particle cloud is
replaced by a candidate-lattice scatter colored by p(S|H). Hit probability
map is published as an OccupancyGrid for RViz's built-in grid rendering.

Topic namespace: `/efe_pmfs/...`
"""
import math

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.msg import OccupancyGrid


class MarkerVisualizer:
    def __init__(self, node, slam_map):
        self.node = node
        self.slam_map = slam_map
        self.prev_num_paths = 0

        # Core publishers
        self.source_belief_pub = node.create_publisher(
            MarkerArray, '/efe_pmfs/source_belief', 10
        )
        self.hit_prob_pub = node.create_publisher(
            OccupancyGrid, '/efe_pmfs/hit_prob', 10
        )
        self.confidence_pub = node.create_publisher(
            OccupancyGrid, '/efe_pmfs/confidence', 10
        )
        self.slam_map_pub = node.create_publisher(
            OccupancyGrid, '/efe_pmfs/slam_map', 10
        )

        # Planner visualizations
        self.all_paths_pub = node.create_publisher(
            MarkerArray, '/efe_pmfs/all_paths', 10
        )
        self.best_path_pub = node.create_publisher(
            Marker, '/efe_pmfs/best_path', 10
        )
        self.frontier_cells_pub = node.create_publisher(
            Marker, '/efe_pmfs/frontier_cells', 10
        )
        self.frontier_centroids_pub = node.create_publisher(
            MarkerArray, '/efe_pmfs/frontier_centroids', 10
        )
        self.prm_graph_pub = node.create_publisher(
            MarkerArray, '/efe_pmfs/prm_graph', 10
        )
        self.global_path_pub = node.create_publisher(
            Marker, '/efe_pmfs/global_path', 10
        )
        self.planner_mode_pub = node.create_publisher(
            Marker, '/efe_pmfs/planner_mode', 10
        )

        # Estimate markers
        self.estimated_source_pub = node.create_publisher(
            Marker, '/efe_pmfs/estimated_source', 10
        )
        self.current_pos_pub = node.create_publisher(
            Marker, '/efe_pmfs/current_position', 10
        )
        self.wind_pub = node.create_publisher(
            Marker, '/efe_pmfs/wind_arrow', 10
        )

    # ------------------------------------------------------------------
    # Grid-shaped overlays
    # ------------------------------------------------------------------
    def _grid_msg(self, values_0_1, stamp):
        """Pack a float[0,1] grid into an OccupancyGrid (0..100)."""
        msg = OccupancyGrid()
        msg.header = Header(stamp=stamp, frame_id='map')
        msg.info.resolution = float(self.slam_map.resolution)
        msg.info.width = int(self.slam_map.width)
        msg.info.height = int(self.slam_map.height)
        msg.info.origin.position.x = float(self.slam_map.origin_x)
        msg.info.origin.position.y = float(self.slam_map.origin_y)
        msg.info.origin.position.z = 0.0
        scaled = np.clip(values_0_1 * 100.0, 0, 100).astype(np.int8)
        msg.data = scaled.ravel().tolist()
        return msg

    def publish_slam_map(self):
        stamp = self.node.get_clock().now().to_msg()
        grid = self.slam_map.grid
        # Convert {-1, 0, 1, 2} → ROS {-1, 0, 100, 50}
        out = np.full_like(grid, -1, dtype=np.int8)
        out[grid == 0] = 0
        out[grid == 1] = 100
        out[grid == 2] = 50
        msg = OccupancyGrid()
        msg.header = Header(stamp=stamp, frame_id='map')
        msg.info.resolution = float(self.slam_map.resolution)
        msg.info.width = int(self.slam_map.width)
        msg.info.height = int(self.slam_map.height)
        msg.info.origin.position.x = float(self.slam_map.origin_x)
        msg.info.origin.position.y = float(self.slam_map.origin_y)
        msg.data = out.ravel().tolist()
        self.slam_map_pub.publish(msg)

    def publish_hit_prob(self, hit_prob):
        stamp = self.node.get_clock().now().to_msg()
        self.hit_prob_pub.publish(self._grid_msg(hit_prob, stamp))

    def publish_confidence(self, confidence):
        stamp = self.node.get_clock().now().to_msg()
        self.confidence_pub.publish(self._grid_msg(confidence, stamp))

    # ------------------------------------------------------------------
    # Source belief (lattice scatter)
    # ------------------------------------------------------------------
    def publish_source_belief(self, candidate_cells, p_S, valid_mask):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = 'source_belief'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15

        if p_S.max() > 0:
            normed = p_S / p_S.max()
        else:
            normed = np.zeros_like(p_S)

        for i, (gx, gy) in enumerate(candidate_cells):
            if not valid_mask[i]:
                continue
            wx, wy = self.slam_map.grid_to_world(int(gx), int(gy))
            p = Point()
            p.x, p.y, p.z = float(wx), float(wy), 0.2
            marker.points.append(p)
            v = float(normed[i])
            # red → yellow → green colormap by probability
            c = ColorRGBA()
            if v < 0.5:
                c.r = 1.0
                c.g = 2.0 * v
                c.b = 0.0
            else:
                c.r = 2.0 * (1.0 - v)
                c.g = 1.0
                c.b = 0.0
            c.a = 0.3 + 0.7 * v
            marker.colors.append(c)
        marker_array.markers.append(marker)
        self.source_belief_pub.publish(marker_array)

    # ------------------------------------------------------------------
    # RRT / paths
    # ------------------------------------------------------------------
    def visualize_all_paths(self, all_paths, all_utilities=None):
        """
        all_paths : list of list of tuple(x,y) OR list of Nodes with .position
        all_utilities : optional list of floats per path (for coloring)
        """
        marker_array = MarkerArray()
        for i in range(self.prev_num_paths):
            m = Marker()
            m.action = Marker.DELETE
            m.ns, m.id = 'all_paths', i
            marker_array.markers.append(m)

        norm_utils = None
        if all_utilities and len(all_utilities) > 0:
            utils = np.array(all_utilities, dtype=float)
            rng = utils.max() - utils.min()
            norm_utils = (utils - utils.min()) / rng if rng > 1e-6 else np.ones_like(utils) * 0.5

        count = 0
        for i, path in enumerate(all_paths):
            if len(path) < 2:
                continue
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns, marker.id = 'all_paths', i
            marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.06
            for n in path:
                pos = n.position if hasattr(n, 'position') else n
                p = Point()
                p.x, p.y, p.z = float(pos[0]), float(pos[1]), 0.5
                marker.points.append(p)
            c = ColorRGBA()
            if norm_utils is not None and i < len(norm_utils):
                val = float(norm_utils[i]) ** 0.5
                c.r = 1.0 if val < 0.5 else float(2.0 * (1.0 - val))
                c.g = float(2.0 * val) if val < 0.5 else 1.0
                c.b = 0.0
                c.a = 0.9
            else:
                c.r, c.g, c.b, c.a = 0.6, 0.6, 0.6, 0.5
            marker.color = c
            marker_array.markers.append(marker)
            count += 1
        self.prev_num_paths = count
        self.all_paths_pub.publish(marker_array)

    def visualize_best_path(self, best_path):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns, marker.id = 'best_path', 0
        marker.header.stamp = self.node.get_clock().now().to_msg()
        if len(best_path) < 2:
            marker.action = Marker.DELETE
        else:
            marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.18
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
            for n in best_path:
                pos = n.position if hasattr(n, 'position') else n
                p = Point()
                p.x, p.y, p.z = float(pos[0]), float(pos[1]), 0.6
                marker.points.append(p)
        self.best_path_pub.publish(marker)

    # ------------------------------------------------------------------
    # Frontiers / PRM / global path
    # ------------------------------------------------------------------
    def visualize_frontier_cells(self, frontier_cells):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns, marker.id = 'frontier_cells', 0
        marker.type, marker.action = Marker.CUBE_LIST, Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = self.slam_map.resolution
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
        for gx, gy in frontier_cells:
            wx, wy = self.slam_map.grid_to_world(int(gx), int(gy))
            p = Point()
            p.x, p.y, p.z = wx, wy, 0.1
            marker.points.append(p)
        self.frontier_cells_pub.publish(marker)

    def visualize_frontier_centroids(self, clusters):
        ma = MarkerArray()
        for i, c in enumerate(clusters):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns, m.id = 'frontier_centroids', i
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (
                float(c.centroid_world[0]), float(c.centroid_world[1]), 0.3
            )
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            ma.markers.append(m)
        self.frontier_centroids_pub.publish(ma)

    def visualize_prm_graph(self, vertices, vertex_dict):
        ma = MarkerArray()
        v = Marker()
        v.header.frame_id = 'map'
        v.header.stamp = self.node.get_clock().now().to_msg()
        v.ns, v.id = 'prm_vertices', 0
        v.type, v.action = Marker.SPHERE_LIST, Marker.ADD
        v.scale.x = v.scale.y = v.scale.z = 0.10
        v.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
        e = Marker()
        e.header.frame_id = 'map'
        e.header.stamp = v.header.stamp
        e.ns, e.id = 'prm_edges', 1
        e.type, e.action = Marker.LINE_LIST, Marker.ADD
        e.scale.x = 0.02
        e.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3)
        added = set()
        for vx in vertices:
            pt = Point()
            pt.x, pt.y, pt.z = vx.position[0], vx.position[1], 0.2
            v.points.append(pt)
            for nid, _ in vx.neighbors:
                edge = tuple(sorted([vx.id, nid]))
                if edge in added:
                    continue
                added.add(edge)
                if nid in vertex_dict:
                    nb = vertex_dict[nid]
                    p2 = Point()
                    p2.x, p2.y, p2.z = nb.position[0], nb.position[1], 0.2
                    e.points.append(pt)
                    e.points.append(p2)
        ma.markers.append(v)
        ma.markers.append(e)
        self.prm_graph_pub.publish(ma)

    def visualize_global_path(self, path):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id = 'global_path', 0
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.12
        m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for pos in path:
            p = Point()
            p.x, p.y, p.z = pos[0], pos[1], 0.4
            m.points.append(p)
        self.global_path_pub.publish(m)

    def visualize_planner_mode(self, mode):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id = 'planner_mode', 0
        m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x = self.slam_map.origin_x + 1.0
        m.pose.position.y = self.slam_map.origin_y + self.slam_map.real_world_height - 1.0
        m.pose.position.z = 2.0
        m.scale.z = 0.5
        if mode == 'LOCAL':
            m.text = 'MODE: LOCAL (RRT + PMFS MI)'
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            m.text = 'MODE: GLOBAL (Frontier + PMFS MI)'
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        self.planner_mode_pub.publish(m)

    # ------------------------------------------------------------------
    # Point markers
    # ------------------------------------------------------------------
    def visualize_estimated_source(self, x, y):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id = 'estimated_source', 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.5
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0)
        self.estimated_source_pub.publish(m)

    def visualize_current_position(self, position):
        if position is None:
            return
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id = 'current_position', 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(position[0]), float(position[1]), 0.5
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.current_pos_pub.publish(m)

    def visualize_wind_arrow(self, u_mean, v_mean):
        """One big arrow at the map origin showing the prevailing constant wind."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id = 'wind_arrow', 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        cx = self.slam_map.origin_x + self.slam_map.real_world_width - 1.5
        cy = self.slam_map.origin_y + self.slam_map.real_world_height - 1.5
        yaw = math.atan2(v_mean, u_mean)
        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = 0.2
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        speed = math.hypot(u_mean, v_mean)
        m.scale.x = max(0.3, min(1.5, speed))
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
        self.wind_pub.publish(m)
