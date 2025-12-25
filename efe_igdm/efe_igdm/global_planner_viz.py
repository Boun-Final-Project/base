"""
Visualization methods for Global Planner (to be integrated into igdm.py)

These methods should be added to the RRTInfotaxisNode class before take_step().
"""

def visualize_frontier_cells(self, frontier_cells):
    """Visualize frontier cells as small red cubes."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "frontier_cells"
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD

    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    marker.color = ColorRGBA()
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 0.6

    # Convert grid coordinates to world coordinates
    for gx, gy in frontier_cells:
        wx, wy = self.occupancy_grid.grid_to_world(gx, gy)
        point = Point()
        point.x = wx
        point.y = wy
        point.z = 0.1
        marker.points.append(point)

    self.frontier_cells_pub.publish(marker)


def visualize_frontier_centroids(self, frontier_clusters):
    """Visualize frontier centroids as large red spheres."""
    marker_array = MarkerArray()

    for i, cluster in enumerate(frontier_clusters):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontier_centroids"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = cluster.centroid_world[0]
        marker.pose.position.y = cluster.centroid_world[1]
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        marker_array.markers.append(marker)

    self.frontier_centroids_pub.publish(marker_array)


def visualize_prm_graph(self, prm_vertices):
    """Visualize PRM graph with vertices and edges."""
    marker_array = MarkerArray()

    # Vertices (yellow spheres)
    vertices_marker = Marker()
    vertices_marker.header.frame_id = "map"
    vertices_marker.header.stamp = self.get_clock().now().to_msg()
    vertices_marker.ns = "prm_vertices"
    vertices_marker.id = 0
    vertices_marker.type = Marker.SPHERE_LIST
    vertices_marker.action = Marker.ADD

    vertices_marker.scale.x = 0.15
    vertices_marker.scale.y = 0.15
    vertices_marker.scale.z = 0.15

    vertices_marker.color = ColorRGBA()
    vertices_marker.color.r = 1.0
    vertices_marker.color.g = 1.0
    vertices_marker.color.b = 0.0
    vertices_marker.color.a = 0.5

    for vertex in prm_vertices:
        point = Point()
        point.x = vertex.position[0]
        point.y = vertex.position[1]
        point.z = 0.2
        vertices_marker.points.append(point)

    marker_array.markers.append(vertices_marker)

    # Edges (yellow lines)
    edges_marker = Marker()
    edges_marker.header.frame_id = "map"
    edges_marker.header.stamp = self.get_clock().now().to_msg()
    edges_marker.ns = "prm_edges"
    edges_marker.id = 1
    edges_marker.type = Marker.LINE_LIST
    edges_marker.action = Marker.ADD

    edges_marker.scale.x = 0.02  # Line width

    edges_marker.color = ColorRGBA()
    edges_marker.color.r = 1.0
    edges_marker.color.g = 1.0
    edges_marker.color.b = 0.0
    edges_marker.color.a = 0.3

    # Add edges (avoid duplicates)
    added_edges = set()
    for vertex in prm_vertices:
        for neighbor_id, _ in vertex.neighbors:
            edge = tuple(sorted([vertex.id, neighbor_id]))
            if edge not in added_edges:
                added_edges.add(edge)

                # Start point
                p1 = Point()
                p1.x = vertex.position[0]
                p1.y = vertex.position[1]
                p1.z = 0.2

                # End point
                neighbor = self.global_planner.vertex_dict[neighbor_id]
                p2 = Point()
                p2.x = neighbor.position[0]
                p2.y = neighbor.position[1]
                p2.z = 0.2

                edges_marker.points.append(p1)
                edges_marker.points.append(p2)

    marker_array.markers.append(edges_marker)

    # Frontier vertices (green spheres, larger)
    frontier_vertices_marker = Marker()
    frontier_vertices_marker.header.frame_id = "map"
    frontier_vertices_marker.header.stamp = self.get_clock().now().to_msg()
    frontier_vertices_marker.ns = "frontier_vertices"
    frontier_vertices_marker.id = 2
    frontier_vertices_marker.type = Marker.SPHERE_LIST
    frontier_vertices_marker.action = Marker.ADD

    frontier_vertices_marker.scale.x = 0.25
    frontier_vertices_marker.scale.y = 0.25
    frontier_vertices_marker.scale.z = 0.25

    frontier_vertices_marker.color = ColorRGBA()
    frontier_vertices_marker.color.r = 0.0
    frontier_vertices_marker.color.g = 1.0
    frontier_vertices_marker.color.b = 0.0
    frontier_vertices_marker.color.a = 0.8

    for vertex in prm_vertices:
        if vertex.is_frontier_vertex:
            point = Point()
            point.x = vertex.position[0]
            point.y = vertex.position[1]
            point.z = 0.25
            frontier_vertices_marker.points.append(point)

    marker_array.markers.append(frontier_vertices_marker)

    self.prm_graph_pub.publish(marker_array)


def visualize_global_path(self, global_path):
    """Visualize selected global path as thick cyan line."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "global_path"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.15  # Thick line for visibility

    marker.color = ColorRGBA()
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 1.0  # Cyan
    marker.color.a = 1.0

    for position in global_path:
        point = Point()
        point.x = position[0]
        point.y = position[1]
        point.z = 0.4  # Higher than local paths
        marker.points.append(point)

    self.global_path_pub.publish(marker)


def visualize_planner_mode(self):
    """Visualize current planner mode as text marker."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "planner_mode"
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    # Position in top-left corner of map
    marker.pose.position.x = self.occupancy_grid.origin_x + 1.0
    marker.pose.position.y = self.occupancy_grid.origin_y + self.occupancy_grid.real_world_height - 1.0
    marker.pose.position.z = 2.0
    marker.pose.orientation.w = 1.0

    marker.scale.z = 0.5  # Text size

    # Color based on mode
    if self.planner_mode == 'LOCAL':
        marker.text = "MODE: LOCAL (RRT-Infotaxis)"
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    else:  # GLOBAL
        marker.text = "MODE: GLOBAL (Frontier Exploration)"
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

    marker.color.a = 1.0

    self.planner_mode_pub.publish(marker)
