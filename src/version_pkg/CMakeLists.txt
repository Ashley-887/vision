cmake_minimum_required(VERSION 3.0.2)
project(version_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation
)

## 添加消息文件
add_message_files(
  FILES
  Object.msg
  Objects.msg
  Point3D.msg
)

## 生成依赖
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## catkin_package 配置
catkin_package(
  # INCLUDE_DIRS include  # 如果没有 include 目录，可以注释掉或删除
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge image_transport message_runtime
)

## 安装节点脚本
catkin_install_python(PROGRAMS scripts/camera_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/inference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/clip_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/calculate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

###########
## Build ##
###########

## 指定额外的头文件位置
include_directories(
  ${catkin_INCLUDE_DIRS}
)

