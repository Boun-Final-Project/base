-- Cartographer config for benchmark_env_realistic (PioneerP3DX, single 2D
-- planar lidar via BasicSim). Frame names are hardcoded for this project's
-- one actual namespace (PioneerP3DX) rather than templated, matching how
-- Cartographer's own stock example configs hardcode a single robot's frames.
--
-- max_range / missing_data_ray_length are set with a deliberate 0.1m gap
-- below BasicSim's published sensor range (see realistic_launch.py's
-- lidar_max_range arg, currently 3.0 — keep this file's max_range 0.1
-- below whatever that arg is set to, or every no-hit reading lands at or
-- under max_range again and never gets classified as a miss). Unlike
-- slam_toolbox/Karto (which
-- collapses its "trace to threshold" branch because it always sets
-- rangeThreshold == max_laser_range, see laser_utils.cpp:162 in the
-- slam_toolbox source), Cartographer has a genuinely separate
-- returns/misses classification (range <= max_range -> real hit,
-- range > max_range -> free-only ray rescaled to missing_data_ray_length,
-- see local_trajectory_builder_2d.cc). BasicSim's no-hit convention
-- publishes exactly its sensor's maxDistance, which must land STRICTLY
-- ABOVE this file's max_range for the free-space classification to fire.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "PioneerP3DX_base_link",
  -- BasicSim already publishes odom->base_link (Robot.cpp's native odom).
  -- published_frame must be the ODOM frame, not base_link: with
  -- provide_odom_frame=false, cartographer publishes map->published_frame
  -- DIRECTLY (no extra hop), so if published_frame were base_link, base_link
  -- would get two TF parents (map from cartographer, odom from BasicSim) —
  -- an invalid tree that froze Nav2 with spurious collision reports.
  -- Setting it to the odom frame makes cartographer's correction stack as
  -- map->odom, same as slam_toolbox, letting BasicSim's existing
  -- odom->base_link complete the chain.
  published_frame = "PioneerP3DX_odom",
  odom_frame = "PioneerP3DX_odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  -- Fuse BasicSim's real odom as a motion prior (mirrors slam_toolbox's
  -- existing use of the odom TF to seed scan matching).
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05
TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 2.9
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 2.9
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

POSE_GRAPH.optimization_problem.huber_scale = 1e2
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.constraint_builder.min_score = 0.65

return options
