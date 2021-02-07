cmake_minimum_required(VERSION 3.17)
project(rotation_solver)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")

find_package (PCL 1.8 REQUIRED)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)


find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        roscpp
        sensor_msgs
        std_msgs
        pcl_conversions
        pcl_ros
        tf
        tf_conversions
        tf2_eigen
        tf2
        rospy
        message_generation
        )

find_package(OpenCV 3.2.0 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package( Boost )



include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
)

add_executable(rotation_solver_node
        main.cpp
        )

add_dependencies(rotation_solver_node
        ${catkin_EXPORTED_TARGETS}
        )

target_link_libraries(rotation_solver_node
        Eigen3::Eigen
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )
        
add_definitions(${PCL_DEFINITIONS})

