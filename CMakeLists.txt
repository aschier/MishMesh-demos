cmake_minimum_required(VERSION 3.1)
project(MishMesh-demos)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
set(MSVC_RUNTIME "dynamic")

find_package(Eigen3 REQUIRED QUIET)
find_package(OpenMesh REQUIRED)

option(LIBIGL_OPENGL             "Use OpenGL"         ON)
option(LIBIGL_GLFW               "Use GLFW"           ON)
option(LIBIGL_IMGUI              "Use IMGUI"          ON)
include(libigl)

option(WITH_MISHMESH_SUBDIR          "Include MishMesh as subdir in the project" ON)
if(WITH_MISHMESH_SUBDIR)
	set(MISHMESH_SOURCE_DIR "MISHMESH_SOURCE_DIR-NOTFOUND" CACHE PATH "The MishMesh source directory")
	add_subdirectory(${MISHMESH_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/MishMesh)
else()
	find_package(MishMesh REQUIRED QUIET)
endif()

add_executable(dijkstra
	dijkstra.cpp
)
target_link_libraries(dijkstra
	MishMesh
	igl::core igl::glfw
)

add_executable(cone_singularities-demo
	cone_singularities.cpp
)
target_link_libraries(cone_singularities-demo
	MishMesh
	igl::core igl::glfw igl::imgui
)

add_executable(geodesics
	geodesics.cpp
)
target_link_libraries(geodesics
	MishMesh
	igl::core igl::glfw igl::imgui
)
