cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_BUILD_TYPE Release)

file(GLOB mdacp_cpp_srcs ./src/*.cc)
set(mdacp_srcs ${mdacp_cpp_srcs})
include_directories(./include)

if (USE_GPU_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DUSE_GPU)

  message("-- CUDA --")
  message("Version: " ${CUDA_VERSION})
  message("Runtime: " ${CUDA_CUDART_LIBRARY})

  # set(CUDA_PATH /usr/local/cuda)
  set(CUDA_PATH $ENV{CUDA_PATH}) # for sekirei

  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    "-std=c++11 -O3 -arch=sm_35 -lineinfo -Xptxas -v"
    )

  include_directories(${CUDA_PATH}/include)
  include_directories(${CUDA_PATH}/samples/common/inc)

  cuda_compile(fcalculator_o ./src/fcalculator.cu)
  set(mdacp_srcs ${mdacp_srcs} ${fcalculator_o})

  add_subdirectory(unit_test)
else ()
  file(GLOB dev_info_cc_src ./src/device_info.cc)
  list(REMOVE_ITEM mdacp_srcs ${dev_info_cc_src})
endif()

# for AVX
add_definitions(-DAVX2)
# add_definitions(-DMESH_SIMD)

# enable C++11, avx intrinsics, and OpenMP
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -march=native -fopenmp")

add_executable(mdacp ${mdacp_srcs})
# target_link_libraries(mdacp mpi mpi++) # for SGI MPT
target_link_libraries(mdacp mpi) # for Intel MPI

if (USE_GPU_CUDA)
  link_directories(${CUDA_PATH}/lib64)
  target_link_libraries(mdacp cudart)
endif()
