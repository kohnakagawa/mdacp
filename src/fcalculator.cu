//----------------------------------------------------------------------
#include "fcalculator.h"
#include "fcalculator_kernels.cuh"
//----------------------------------------------------------------------
void
ForceCalculator::SendParticlesHostToDev(Variables *vars,
                                        const int pn_gpu,
                                        cudaStream_t strm) {
  const auto pn_tot = vars->GetTotalParticleNumber();
  CudaPtr2D<double, N, D>& q = vars->q_buf;
  CudaPtr2D<double, N, D>& p = vars->p_buf;
  q.Host2DevAsync(0, pn_tot, strm);
  p.Host2DevAsync(0, pn_gpu, strm);
}
//----------------------------------------------------------------------
void
ForceCalculator::SendParticlesDevToHost(Variables *vars,
                                        const int pn_gpu,
                                        cudaStream_t strm) {
  CudaPtr2D<double, N, D>& p = vars->p_buf;
  p.Dev2HostAsync(0, pn_gpu, strm);
}
//----------------------------------------------------------------------
// NOTE: for calculation @ CPU
void
ForceCalculator::CalculateForce(Variables* vars,
                                MeshList *mesh,
                                SimulationInfo *sinfo,
                                const int beg) {
#ifdef AVX2
  CalculateForceAVX2Reactless(vars, mesh, sinfo, beg);
#else
  CalculateForceReactless(vars, mesh, sinfo, beg);
#endif
}
//----------------------------------------------------------------------
// NOTE: for calculation @ GPU
void
ForceCalculator::CalculateForce(Variables* vars,
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
#if 0
  // optimized for Pascal
  const auto gr_size = (WARP_SIZE * pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CalculateForceWarpUnrollReactlessCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                                 (VecCuda*)p.GetDevPtr(),
                                                                                 mesh->GetCudaPtrSortedList().GetDevPtr(),
                                                                                 mesh->GetCudaPtrNumberOfPartners().GetDevPtr(),
                                                                                 mesh->GetCudaPtrKeyPointerP().GetDevPtr(),
                                                                                 CL2, C2, dt, pn_gpu);
#else
  // optimized for Kepler
  const auto gr_size = (pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CalculateForceReactlessCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                       (VecCuda*)p.GetDevPtr(),
                                                                       mesh->GetDevPtrTransposedList(),
                                                                       mesh->GetCudaPtrNumberOfPartners().GetDevPtr(),
                                                                       CL2, C2, dt, pn_gpu);
#endif
}
//----------------------------------------------------------------------
void
ForceCalculator::HeatbathMomenta(Variables *vars,
                                 SimulationInfo *sinfo,
                                 const int pn_gpu,
                                 cudaStream_t strm) {
  const auto dt2 = sinfo->TimeStep * 0.5;
  const auto exp1 = std::exp(-dt2 * vars->Zeta);
  const auto gr_size = (pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CudaPtr2D<double, N, D>& p = vars->p_buf;

  HeatbathMomentaCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)p.GetDevPtr(),
                                                               exp1,
                                                               pn_gpu);
}
//----------------------------------------------------------------------
