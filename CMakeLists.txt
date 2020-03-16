cmake_minimum_required(VERSION 3.1)
project(MishMesh-demos)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
set(MSVC_RUNTIME "dynamic")

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(Eigen3 REQUIRED QUIET)
find_package(MishMesh REQUIRED QUIET)
find_package(LIBIGL REQUIRED QUIET)

add_executable(dijkstra
	dijkstra.cpp
)
target_link_libraries(dijkstra
	MishMesh
	igl::core igl::opengl_glfw
)

add_executable(cone_singularities
	cone_singularities.cpp
)
target_link_libraries(cone_singularities
	MishMesh
	igl::core igl::opengl_glfw igl::opengl_glfw_imgui
)
