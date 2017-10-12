//----------------------------------------------------------------------
#ifndef fcalculator_h
#define fcalculator_h
//----------------------------------------------------------------------
#include "mdconfig.h"
#include "variables.h"
#include "meshlist.h"
#include "simulationinfo.h"
//----------------------------------------------------------------------
namespace ForceCalculator {
  void CalculateForceBruteforce(Variables *vars, SimulationInfo *sinfo);
  void CalculateForceUnroll(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForceSorted(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForceNext(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForcePair(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForceReactless(Variables *vars, MeshList *mesh, SimulationInfo *sinfo,
                               const int beg = 0);

  void CalculateForceReactlessSIMD(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForceReactlessSIMD_errsafe(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
#ifdef AVX2
  void CalculateForceAVX2(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void CalculateForceAVX2Reactless(Variables *vars, MeshList *mesh, SimulationInfo *sinfo,
                                   const int beg = 0);
#endif
#ifdef AVX512
  void CalculateForceAVX512(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
#endif
#ifdef USE_GPU
  void SendParticlesHostToDev(Variables *vars, const int pn_gpu, cudaStream_t strm);
  void SendParticlesDevToHost(Variables *vars, const int pn_gpu, cudaStream_t strm);
  void CalculateForce(Variables* vars, MeshList *mesh, SimulationInfo *sinfo, const int pn_gpu,
                      cudaStream_t strm);
  void CalculateForce(Variables* vars, MeshList *mesh, SimulationInfo *sinfo, const int beg);

  void UpdatePositionHalf(Variables *vars, SimulationInfo *sinfo, const int pn_gpu,
                          cudaStream_t strm);
  void HeatbathMomenta(Variables *vars, SimulationInfo *sinfo, const int pn_gpu,
                       cudaStream_t strm);
  void Langevin(Variables *vars, SimulationInfo *sinfo, const int pn_gpu,
                cudaStream_t strm);
#endif
  void UpdatePositionHalf(Variables *vars, SimulationInfo *sinfo, const int beg = 0);
  void CalculateForce(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
  void HeatbathZeta(Variables *vars, double ct, SimulationInfo *sinfo);
  void HeatbathMomenta(Variables *vars, SimulationInfo *sinfo, const int beg = 0);
  void Langevin(Variables *vars, SimulationInfo *sinfo, const int beg = 0);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
