//----------------------------------------------------------------------
#ifndef curand_init_cuh
#define curand_init_cuh
//----------------------------------------------------------------------
#include <curand_kernel.h>
//----------------------------------------------------------------------
__global__ void
InitXorwowState(const int seed,
                curandState *state,
                const int pn,
                const int offset = 0) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= pn) return;

  curand_init(seed, tid, offset, &state[tid]);
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
