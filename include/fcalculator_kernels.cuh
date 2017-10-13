//----------------------------------------------------------------------
#ifndef fcalculator_kernels_cuh
#define fcalculator_kernels_cuh
//----------------------------------------------------------------------
#include "device_utils.cuh"
#include <curand_kernel.h>
//----------------------------------------------------------------------
namespace ForceCalculator {
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
    const auto C2_8 = C2 * 8.0;

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
      const auto r14 = r6 * r6 * r2;
      const auto invr14 = 1.0 / r14;
      const auto df_numera = 24.0 * r6 - 48.0;
      auto df = (df_numera * invr14 + C2_8) * dt;
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

    const auto C2_8 = C2 * 8.0;
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int* ptr_list = &transposed_list[tid];
    const auto pn_x2 = pn << 1;

    auto pf = p[tid];
    int32_t k = np & 0x1;
    if (k) {
      const auto j = *ptr_list;
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      const auto r14 = r6 * r6 * r2;
      const auto invr14 = 1.0 / r14;
      const auto df_numera = 24.0 * r6 - 48.0;
      auto df = (df_numera * invr14 + C2_8) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
      ptr_list += pn;
    }

    for (; k < np; k += 2) {
      const auto j0 = *ptr_list;
      const auto j1 = *(ptr_list + pn);

      const auto dx0 = q[j0].x - qi.x; const auto dx1 = q[j1].x - qi.x;
      const auto dy0 = q[j0].y - qi.y; const auto dy1 = q[j1].y - qi.y;
      const auto dz0 = q[j0].z - qi.z; const auto dz1 = q[j1].z - qi.z;

      const auto r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
      const auto r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

      const auto r6_0 = r2_0 * r2_0 * r2_0;
      const auto r6_1 = r2_1 * r2_1 * r2_1;

      const auto r14_0 = r6_0 * r6_0 * r2_0;
      const auto r14_1 = r6_1 * r6_1 * r2_1;

      const auto invr14_01 = 1.0 / (r14_0 * r14_1);

      const auto df_numera_0 = 24.0 * r6_0 - 48.0;
      const auto df_numera_1 = 24.0 * r6_1 - 48.0;

      auto df0 = (df_numera_0 * invr14_01 * r14_1 + C2_8) * dt;
      auto df1 = (df_numera_1 * invr14_01 * r14_0 + C2_8) * dt;

      if (r2_0 > CL2) df0 = 0.0;
      if (r2_1 > CL2) df1 = 0.0;

      pf.x += df0 * dx0; pf.x += df1 * dx1;
      pf.y += df0 * dy0; pf.y += df1 * dy1;
      pf.z += df0 * dz0; pf.z += df1 * dz1;

      ptr_list += pn_x2;
    }

    p[tid] = pf;
  }
  //----------------------------------------------------------------------
  __global__ void
  HeatbathMomentaCUDA(VecCuda* __restrict__ p,
                      const double exp1,
                      const int pn) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= pn) return;

    p[tid].x *= exp1;
    p[tid].y *= exp1;
    p[tid].z *= exp1;
  }
  //----------------------------------------------------------------------
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
