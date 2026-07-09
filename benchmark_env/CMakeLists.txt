cmake_minimum_required(VERSION 3.5)
project(benchmark_env)
find_package(ament_cmake REQUIRED)

# Self-contained gaden-project scenarios (CAD + wind + config, all real files).
# Baked gas is NOT shipped — each user generates it locally with:
#   ros2 launch benchmark_env gaden_preproc_launch.py scenario:=<name>
#   ros2 launch benchmark_env gaden_sim_launch.py     scenario:=<name>
install(DIRECTORY .
  DESTINATION share/${PROJECT_NAME}
)

# GUI exposed as a ros2 run target:  ros2 run benchmark_env benchmark_gui
install(PROGRAMS
  scripts/benchmark_gui
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
