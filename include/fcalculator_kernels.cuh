//----------------------------------------------------------------------
#ifndef fcalculator_kernels_cuh
#define fcalculator_kernels_cuh
//----------------------------------------------------------------------
#include "device_utils.cuh"
//----------------------------------------------------------------------
// NOTE: Optimized for Pascal
__global__ void
CalculateForceWarpUnroll(const VecCuda* __restrict__ q,
                         VecCuda*       __restrict__ p,
                         const int*     __restrict__ sorted_list,
                         const int*     __restrict__ number_of_partners,
                         const int*     __restrict__ pointer,
                         const double CL2,
                         const double C2,
                         const double dt,
                         const int    pn) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id < pn) {
    const auto lid = lane_id();
    const auto qi  = q[i_ptcl_id];
    const auto np  = number_of_partners[i_ptcl_id];
    const auto kp  = pointer[i_ptcl_id] + lid;
    const int ini_loop = (np / warpSize) * warpSize;

    VecCuda pf = {0.0};
    if (lid == 0) pf = p[i_ptcl_id];
    int k = 0;
    for (; k < ini_loop; k += warpSize) {
      const auto j  = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2) + C2 * 8.0) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
    }

    // remaining loop
    if (lid < (np % warpSize)) {
      const auto j  = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2) + C2 * 8.0) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
    }

    // warp reduction
    pf.x = warp_segment_reduce(pf.x);
    pf.y = warp_segment_reduce(pf.y);
    pf.z = warp_segment_reduce(pf.z);

    if (lid == 0) p[i_ptcl_id] = pf;
  }
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
