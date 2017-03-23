cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

file(GLOB mdacp_cpp_srcs ./src/*.cc)
set(mdacp_srcs ${mdacp_cpp_srcs})
include_directories(./include)

# for AVX
add_definitions(-DAVX2)
# add_definitions(-DMESH_SIMD)

# enable C++11, avx intrinsics, and OpenMP
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp -xHOST -O3 -std=c++11")

add_executable(mdacp ${mdacp_srcs})
# target_link_libraries(mdacp mpi mpi++) # for SGI MPT
target_link_libraries(mdacp mpi) # for Intel MPI
