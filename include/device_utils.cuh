//----------------------------------------------------------------------
#ifndef device_utils_cuh
#define device_utils_cuh
//----------------------------------------------------------------------
__device__ __forceinline__ int
lane_id() {
  return threadIdx.x % warpSize;
}
//----------------------------------------------------------------------
template <typename T> __device__ __forceinline__ T
warp_segment_reduce(T var) {
  var += __shfl_down(var, 0x10);
  var += __shfl_down(var, 0x8);
  var += __shfl_down(var, 0x4);
  var += __shfl_down(var, 0x2);
  var += __shfl_down(var, 0x1);
  return var;
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
