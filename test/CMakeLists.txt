cmake_minimum_required(VERSION 2.8)

if (USE_GPU_CUDA)
  find_package(CUDA REQUIRED)
  cuda_add_executable(test_cuda_ptrs test_cuda_ptrs.cu)
endif()


