//----------------------------------------------------------------------
#ifndef meshlist_kernels_cuh
#define meshlist_kernels_cuh
//----------------------------------------------------------------------
#include "device_utils.cuh"
//----------------------------------------------------------------------
__global__ void
TransposeSortedListCUDA(const int* __restrict__ sorted_list,
                        const int* __restrict__ pointer,
                        const int* __restrict__ number_of_partners,
                        int* __restrict__ transposed_list,
                        const int pn) {
  const auto ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (ptcl_id >= pn) return;

  const auto lid = lane_id();
  const auto kp  = pointer[ptcl_id];
  const auto np  = number_of_partners[ptcl_id];

  for (int k = lid; k < np; k += warpSize) {
    transposed_list[ptcl_id + pn * k] = sorted_list[kp + k];
  }
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
