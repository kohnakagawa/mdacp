//----------------------------------------------------------------------
#ifndef fcalculator_kernels_cuh
#define fcalculator_kernels_cuh
//----------------------------------------------------------------------
#include "device_utils.cuh"
//----------------------------------------------------------------------
// NOTE: Optimized for Pascal
__global__ void
CalculateForceWarpUnrollReactlessCUDA(const VecCuda* __restrict__ q,
                                      VecCuda*       __restrict__ p,
                                      const int*     __restrict__ sorted_list,
                                      const int*     __restrict__ number_of_partners,
                                      const int*     __restrict__ pointer,
                                      const double CL2,
                                      const double C2,
                                      const double dt,
                                      const int    pn) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id >= pn) return;

  const auto lid = lane_id();
  const auto qi  = q[i_ptcl_id];
  const auto np  = number_of_partners[i_ptcl_id];

  VecCuda pf = {0.0};
  if (lid == 0) pf = p[i_ptcl_id];
  const auto kp = pointer[i_ptcl_id];
  for (int k = lid; k < np; k += warpSize) {
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
//----------------------------------------------------------------------
// NOTE: Optimized for Kepler
__global__ void
CalculateForceReactlessCUDA(const VecCuda* __restrict__ q,
                            VecCuda*       __restrict__ p,
                            const int*     __restrict__ transposed_list,
                            const int*     __restrict__ number_of_partners,
                            const double CL2,
                            const double C2,
                            const double dt,
                            const int    pn) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= pn) return;

  const auto qi = q[tid];
  const auto np = number_of_partners[tid];
  const int* ptr_list = &transposed_list[tid];

  auto pf = p[tid];

  auto j = __ldg(ptr_list);
  ptr_list += pn;
  auto dxa = q[j].x - qi.x;
  auto dya = q[j].y - qi.y;
  auto dza = q[j].z - qi.z;
  double df = 0.0, dxb = 0.0, dyb = 0.0, dzb = 0.0;
  for (int32_t k = 0; k < np; k++) {
    const auto dx = dxa;
    const auto dy = dya;
    const auto dz = dza;
    const auto r2 = dx * dx + dy * dy + dz * dz;

    j = __ldg(ptr_list);
    ptr_list += pn;

    dxa = q[j].x - qi.x;
    dya = q[j].y - qi.y;
    dza = q[j].z - qi.z;

    pf.x += df * dxb;
    pf.y += df * dyb;
    pf.z += df * dzb;

    const auto r6 = r2 * r2 * r2;
    df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2) + C2 * 8.0) * dt;
    if (r2 > CL2) df = 0.0;
    dxb = dx;
    dyb = dy;
    dzb = dz;
  }
  pf.x += df * dxb;
  pf.y += df * dyb;
  pf.z += df * dzb;

  p[tid] = pf;
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
