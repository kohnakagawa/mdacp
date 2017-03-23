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
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    var += __shfl_down(var, offset);
  }
  return var;
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
