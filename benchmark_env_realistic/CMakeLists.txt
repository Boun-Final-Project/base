cmake_minimum_required(VERSION 3.8)
project(benchmark_env_realistic)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  navigation_config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
