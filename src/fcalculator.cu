//----------------------------------------------------------------------
#include "fcalculator.h"
#include "fcalculator_kernels.cuh"
//----------------------------------------------------------------------
void
ForceCalculator::CalculateForceGPU(Variables* vars, MeshList *mesh, SimulationInfo *sinfo) {
  const auto dt     = sinfo->TimeStep;
  const auto pn     = vars->GetParticleNumber();
  const auto pn_tot = vars->GetTotalParticleNumber();
  const auto CL2    = CUTOFF_LENGTH * CUTOFF_LENGTH;
  const auto C2     = vars->GetC2();

  const auto dev_id = vars->GetDeviceId();
  checkCudaErrors(cudaSetDevice(dev_id));

  CudaPtr2D<double, N, D>& q = vars->q_buf;
  CudaPtr2D<double, N, D>& p = vars->p_buf;

  q.Host2Dev(0, pn_tot);
  p.Host2Dev(0, pn_tot);
  const auto gr_size = (WARP_SIZE * pn - 1) / THREAD_BLOCK_SIZE + 1;
  CalculateForceWarpUnroll<<<gr_size, THREAD_BLOCK_SIZE>>>((VecCuda*)q.GetDevPtr(),
                                                           (VecCuda*)p.GetDevPtr(),
                                                           mesh->GetCudaPtrSortedList().GetDevPtr(),
                                                           mesh->GetCudaPtrNumberOfPartners().GetDevPtr(),
                                                           mesh->GetCudaPtrKeyPointerP().GetDevPtr(),
                                                           CL2, C2, dt, pn);
  q.Dev2Host(0, pn_tot);
  p.Dev2Host(0, pn_tot);
}
//----------------------------------------------------------------------
