cmake_minimum_required(VERSION 3.8)
project(adsm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
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

add_library(adsm_engine STATIC src/engine.cpp)
set_target_properties(adsm_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(adsm_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

pybind11_add_module(adsm_core_py src/python_bindings.cpp)
target_link_libraries(adsm_core_py PRIVATE adsm_engine)
target_include_directories(adsm_core_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
ament_python_install_package(${PROJECT_NAME})
install(TARGETS adsm_engine
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(TARGETS adsm_core_py LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
