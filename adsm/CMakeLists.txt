cmake_minimum_required(VERSION 3.8)
project(adsm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(olfaction_msgs REQUIRED)
find_package(gaden_msgs REQUIRED)
find_package(tf2 REQUIRED)

add_executable(adsm_node
  src/adsm.cpp
  src/frontier_finder.cpp
  src/goal.cpp
  src/gridmap.cpp
  src/rrt_sampler.cpp
)

target_include_directories(adsm_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(adsm_node
  rclcpp
  rclcpp_action
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  nav2_msgs
  sensor_msgs
  olfaction_msgs
  gaden_msgs
  tf2
)

install(TARGETS adsm_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
