from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class MarkerVisualizer:
    """
    Handles all RViz Marker visualizations for the RRT-Infotaxis node.
    """
    def __init__(self, node, slam_map):
        """
        Args:
            node: The ROS 2 node (used to create publishers and get clock)
            slam_map: The occupancy grid object (used for coordinate transforms)
        """
        self.node = node
        self.slam_map = slam_map
        self.prev_num_paths = 0

        # Create Publishers
        self.particle_pub = node.create_publisher(MarkerArray, '/rrt_infotaxis/particles', 10)
        self.all_paths_pub = node.create_publisher(MarkerArray, '/rrt_infotaxis/all_paths', 10)
        self.best_path_pub = node.create_publisher(Marker, '/rrt_infotaxis/best_path', 10)
        self.estimated_source_pub = node.create_publisher(Marker, '/rrt_infotaxis/estimated_source', 10)
        self.current_pos_pub = node.create_publisher(Marker, '/rrt_infotaxis/current_position', 10)
        
        # Global Planner Publishers
        self.frontier_cells_pub = node.create_publisher(Marker, '/rrt_infotaxis/frontier_cells', 10)
        self.frontier_centroids_pub = node.create_publisher(MarkerArray, '/rrt_infotaxis/frontier_centroids', 10)
        self.prm_graph_pub = node.create_publisher(MarkerArray, '/rrt_infotaxis/prm_graph', 10)
        self.global_path_pub = node.create_publisher(Marker, '/rrt_infotaxis/global_path', 10)
        self.planner_mode_pub = node.create_publisher(Marker, '/rrt_infotaxis/planner_mode', 10)

    def visualize_particles(self, particles, weights):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "particles", 0, Marker.POINTS, Marker.ADD
        marker.scale.x = marker.scale.y = 0.1
        marker.pose.orientation.w = 1.0
        
        norm_w = weights / weights.max() if (len(weights) > 0 and weights.max() > 0) else weights
        
        for p_val, w in zip(particles, norm_w):
            p = Point()
            p.x, p.y, p.z = float(p_val[0]), float(p_val[1]), 0.5
            marker.points.append(p)
            c = ColorRGBA()
            c.r, c.g, c.b, c.a = float(w), float(w * 0.8), float(1.0 - w), 0.8
            marker.colors.append(c)
        
        marker_array.markers.append(marker)
        self.particle_pub.publish(marker_array)

    def visualize_all_paths(self, all_paths, all_utilities=None):
        marker_array = MarkerArray()
        # Delete old markers
        for i in range(self.prev_num_paths):
            m = Marker()
            m.action = Marker.DELETE
            m.ns, m.id = "all_paths", i
            marker_array.markers.append(m)
        
        norm_utils = None
        if all_utilities and len(all_utilities) > 0:
            utils = np.array(all_utilities)
            rng = utils.max() - utils.min()
            norm_utils = (utils - utils.min()) / rng if rng > 1e-6 else np.ones_like(utils)*0.5

        count = 0
        for i, path in enumerate(all_paths):
            if len(path) < 2: continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns, marker.id, marker.type, marker.action = "all_paths", i, Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.08
            for node in path:
                p = Point()
                p.x, p.y, p.z = float(node.position[0]), float(node.position[1]), 0.5
                marker.points.append(p)
            
            c = ColorRGBA()
            if norm_utils is not None and i < len(norm_utils):
                val = norm_utils[i] ** 0.5
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
        marker.header.frame_id, marker.ns, marker.id = "map", "best_path", 0
        marker.header.stamp = self.node.get_clock().now().to_msg()
        if len(best_path) < 2:
            marker.action = Marker.DELETE
        else:
            marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = 0.20
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
            for node in best_path:
                p = Point()
                p.x, p.y, p.z = float(node.position[0]), float(node.position[1]), 0.6
                marker.points.append(p)
        self.best_path_pub.publish(marker)

    def visualize_estimated_source(self, est_x, est_y):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "estimated_source", 0
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(est_x), float(est_y), 0.5
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0)
        self.estimated_source_pub.publish(marker)

    def visualize_current_position(self, position):
        if position is None: return
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "current_position", 0
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(position[0]), float(position[1]), 0.5
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.current_pos_pub.publish(marker)

    def visualize_frontier_cells(self, frontier_cells):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "frontier_cells", 0
        marker.type, marker.action = Marker.CUBE_LIST, Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
        if hasattr(self.slam_map, 'grid_to_world'):
            for gx, gy in frontier_cells:
                wx, wy = self.slam_map.grid_to_world(gx, gy)
                p = Point()
                p.x, p.y, p.z = wx, wy, 0.1
                marker.points.append(p)
        self.frontier_cells_pub.publish(marker)

    def visualize_frontier_centroids(self, frontier_clusters):
        marker_array = MarkerArray()
        for i, cluster in enumerate(frontier_clusters):
            marker = Marker()
            marker.header.frame_id, marker.ns, marker.id = "map", "frontier_centroids", i
            marker.type, marker.action = Marker.SPHERE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = cluster.centroid_world[0], cluster.centroid_world[1], 0.3
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            marker_array.markers.append(marker)
        self.frontier_centroids_pub.publish(marker_array)

    def visualize_prm_graph(self, prm_vertices, vertex_dict):
        """
        Visualize the PRM graph.
        Args:
            prm_vertices: List of vertex objects
            vertex_dict: Dictionary mapping ID to vertex objects (needed for edges)
        """
        marker_array = MarkerArray()
        
        # Vertices
        v_marker = Marker()
        v_marker.header.frame_id, v_marker.ns, v_marker.id = "map", "prm_vertices", 0
        v_marker.type, v_marker.action = Marker.SPHERE_LIST, Marker.ADD
        v_marker.scale.x = v_marker.scale.y = v_marker.scale.z = 0.15
        v_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
        
        # Edges
        e_marker = Marker()
        e_marker.header.frame_id, e_marker.ns, e_marker.id = "map", "prm_edges", 1
        e_marker.type, e_marker.action = Marker.LINE_LIST, Marker.ADD
        e_marker.scale.x = 0.02
        e_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.3)
        
        added_edges = set()
        for v in prm_vertices:
            p = Point()
            p.x, p.y, p.z = v.position[0], v.position[1], 0.2
            v_marker.points.append(p)
            
            for nid, _ in v.neighbors:
                edge = tuple(sorted([v.id, nid]))
                if edge not in added_edges:
                    added_edges.add(edge)
                    if nid in vertex_dict:
                        neighbor = vertex_dict[nid]
                        p2 = Point()
                        p2.x, p2.y, p2.z = neighbor.position[0], neighbor.position[1], 0.2
                        e_marker.points.append(p)
                        e_marker.points.append(p2)
        
        marker_array.markers.append(v_marker)
        marker_array.markers.append(e_marker)
        self.prm_graph_pub.publish(marker_array)

    def visualize_global_path(self, global_path):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "global_path", 0
        marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
        marker.scale.x = 0.15
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        for pos in global_path:
            p = Point()
            p.x, p.y, p.z = pos[0], pos[1], 0.4
            marker.points.append(p)
        self.global_path_pub.publish(marker)

    def visualize_planner_mode(self, planner_mode):
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "planner_mode", 0
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        
        # Use slam_map properties for placement
        marker.pose.position.x = self.slam_map.origin_x + 1.0
        marker.pose.position.y = self.slam_map.origin_y + self.slam_map.real_world_height - 1.0
        marker.pose.position.z = 2.0
        marker.scale.z = 0.5
        
        if planner_mode == 'LOCAL':
            marker.text = "MODE: LOCAL (RRT-Infotaxis)"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            marker.text = "MODE: GLOBAL (Frontier Exploration)"
            marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        self.planner_mode_pub.publish(marker)

    def clear_global_planner_visualizations(self):
        m = Marker()
        m.action = Marker.DELETE
        m.header.frame_id = "map"
        
        m.ns = "frontier_cells"
        self.frontier_cells_pub.publish(m)
        m.ns = "global_path"
        self.global_path_pub.publish(m)
        
        ma = MarkerArray()
        for ns in ["prm_vertices", "prm_edges", "frontier_vertices", "frontier_centroids"]:
            for i in range(100):
                dm = Marker()
                dm.header.frame_id = "map"
                dm.ns, dm.id, dm.action = ns, i, Marker.DELETE
                ma.markers.append(dm)
        self.prm_graph_pub.publish(ma)
        self.frontier_centroids_pub.publish(ma)