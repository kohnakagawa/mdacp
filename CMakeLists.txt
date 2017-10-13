cmake_minimum_required(VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_BUILD_TYPE Release)

file(GLOB mdacp_cpp_srcs ./src/*.cc)
set(mdacp_srcs ${mdacp_cpp_srcs})
include_directories(./include)

# for SIMD optimization
if (USE_AVX2)
  add_definitions(-DAVX2)
  set(OPT_FLAGS "-xCORE-AVX2 -ipo")
elseif(USE_AVX512)
  # only support icpc
  add_definitions(-DAVX512)
  set(OPT_FLAGS "-xMIC-AVX512 -ipo")
endif()

# add warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# add OpenMP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if (USE_GPU_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DUSE_GPU)

  message("-- CUDA --")
  message("Version: " ${CUDA_VERSION})
  message("Runtime: " ${CUDA_CUDART_LIBRARY})

  # set(CUDA_PATH /usr/local/cuda)
  set(CUDA_PATH $ENV{CUDA_PATH}) # for sekirei

  # NOTE: CMake official FindCUDA.cmake does not support -isystem /path/include when compiling device codes.
  # So, we explicitly add -isystem /path/include to CUDA_NVCC_FLAGS
  set(CURT_DIR ${CUDA_PATH}/include)
  set(CUHELPER_DIR ${CUDA_PATH}/samples/common/inc)
  set(INCLUDE_SYSTEM_DIRS "-isystem ${CURT_DIR} -isystem ${CUHELPER_DIR}")

  if (DEFINED GPU_ARCH)
    if (GPU_ARCH STREQUAL KEPLER)
      set(GPU_ARCH_FLAG "-arch=sm_35")
    elseif (GPU_ARCH STREQUAL PASCAL)
      set(GPU_ARCH_FLAG "-arch=sm_60")
    else ()
      message(SEND_ERROR "GPU_ARCH is not specified.")
    endif ()
  endif ()

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${INCLUDE_SYSTEM_DIRS} -std=c++11 -O3 ${GPU_ARCH_FLAG} -Xptxas -v")

  link_directories(${CUDA_PATH}/lib64)
  include_directories(SYSTEM ${CURT_DIR})
  include_directories(SYSTEM ${CUHELPER_DIR})

  cuda_compile(fcalculator_o ./src/fcalculator.cu)
  cuda_compile(meshlist_o ./src/meshlist.cu)
  set(mdacp_srcs ${mdacp_srcs} ${fcalculator_o})
  set(mdacp_srcs ${mdacp_srcs} ${meshlist_o})
else ()
  file(GLOB dev_info_cc_src ./src/device_info.cc)
  list(REMOVE_ITEM mdacp_srcs ${dev_info_cc_src})
endif()

# enable C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# add opt flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")

add_executable(mdacp ${mdacp_srcs})

if (USE_GPU_CUDA)
  target_link_libraries(mdacp cudart)
endif()
