cmake_minimum_required(VERSION 3.8)
project(retinex_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

set(dependencies
  rclcpp
  OpenCV
  sensor_msgs
  cv_bridge
)

add_executable(retinex_node src/retinex_node.cpp)
ament_target_dependencies(retinex_node ${dependencies})

install(TARGETS retinex_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# install
install(DIRECTORY src
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
