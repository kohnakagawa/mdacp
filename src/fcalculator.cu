//----------------------------------------------------------------------
#include "fcalculator.h"
#include "fcalculator_kernels.cuh"
#include "curand_init.cuh"
#include <thrust/device_vector.h>
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
  CudaPtr2D<double, N, D>& q = vars->q_buf;
  CudaPtr2D<double, N, D>& p = vars->p_buf;
  q.Dev2HostAsync(0, pn_gpu, strm);
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
ForceCalculator::UpdatePositionHalf(Variables *vars,
                                    SimulationInfo *sinfo,
                                    const int pn_gpu,
                                    cudaStream_t strm) {
  const auto dt2 = sinfo->TimeStep * 0.5;
  const auto gr_size = (pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  CudaPtr2D<double, N, D>& q = vars->q_buf;
  CudaPtr2D<double, N, D>& p = vars->p_buf;

  UpdatePositionHalfCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                  (VecCuda*)p.GetDevPtr(),
                                                                  dt2,
                                                                  pn_gpu);
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
void
ForceCalculator::Langevin(Variables *vars,
                          SimulationInfo *sinfo,
                          const int pn_gpu,
                          cudaStream_t strm) {
  thread_local bool is_first = true;
  thread_local thrust::device_vector<curandState> states(N);
  curandState* ptr_states = thrust::raw_pointer_cast(states.data());
  if (is_first) {
    const auto gr_size = (N - 1) / THREAD_BLOCK_SIZE + 1;
    const auto seed = 1234;
    InitXorwowState<<<gr_size, THREAD_BLOCK_SIZE>>>(seed, ptr_states, N);
    checkCudaErrors(cudaDeviceSynchronize());
    is_first = false;
  }

  const auto dt              = sinfo->TimeStep;
  CudaPtr2D<double, N, D>& p = vars->p_buf;
  const double hb_gamma      = sinfo->HeatbathGamma;
  const double T             = sinfo->AimedTemperature;
  const double hb_D          = std::sqrt(2.0 * hb_gamma * T / dt);
  const auto gr_size         = (pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;

  LangevinCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>((VecCuda*)p.GetDevPtr(),
                                                        ptr_states,
                                                        dt,
                                                        hb_gamma,
                                                        hb_D,
                                                        pn_gpu);
}
//----------------------------------------------------------------------
