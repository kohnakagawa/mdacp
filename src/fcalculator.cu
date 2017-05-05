//----------------------------------------------------------------------
#include "fcalculator.h"
#include "fcalculator_kernels.cuh"
//----------------------------------------------------------------------
void
ForceCalculator::CalculateForceGPU(Variables* vars,
                                   MeshList *mesh,
                                   SimulationInfo *sinfo,
                                   const int pn_gpu,
                                   cudaStream_t strm) {
  const auto dt     = sinfo->TimeStep;
  const auto pn_tot = vars->GetTotalParticleNumber();
  const auto CL2    = CUTOFF_LENGTH * CUTOFF_LENGTH;
  const auto C2     = vars->GetC2();

  CudaPtr2D<double, N, D>& q = vars->q_buf;
  CudaPtr2D<double, N, D>& p = vars->p_buf;

  // enqueue GPU task
  q.Host2DevAsync(0, pn_tot, strm);
  p.Host2DevAsync(0, pn_gpu, strm);

#if 0
  const auto gr_size = (WARP_SIZE * pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CalculateForceWarpUnrollReactlessCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                                 (VecCuda*)p.GetDevPtr(),
                                                                                 mesh->GetCudaPtrSortedList().GetDevPtr(),
                                                                                 mesh->GetCudaPtrNumberOfPartners().GetDevPtr(),
                                                                                 mesh->GetCudaPtrKeyPointerP().GetDevPtr(),
                                                                                 CL2, C2, dt, pn_gpu);
#else
  const auto gr_size = (pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CalculateForceReactlessCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                       (VecCuda*)p.GetDevPtr(),
                                                                       mesh->GetDevPtrTransposedList(),
                                                                       mesh->GetCudaPtrNumberOfPartners().GetDevPtr(),
                                                                       CL2, C2, dt, pn_gpu);
#endif

  p.Dev2HostAsync(0, pn_gpu, strm);

  // NOTE:
  // Synchronized in mdmanager.cc
}
//----------------------------------------------------------------------
