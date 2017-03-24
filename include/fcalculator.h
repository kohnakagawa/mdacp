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
   void CalculateForceReactless(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);

   void CalculateForceReactlessSIMD(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
   void CalculateForceReactlessSIMD_errsafe(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
#ifdef AVX2
   void CalculateForceAVX2(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
#endif
#ifdef USE_GPU
   void CalculateForceGPU(Variables* vars, MeshList *mesh, SimulationInfo *sinfo);
   void CalculateForceCPUGPUHybrid(Variables* vars, MeshList *mesh, SimulationInfo *sinfo, const double ratio = 0.75);
   void CalculateForceAVX2Reactless(const double q[][D], double p[][D],
                                    const int* sorted_list,
                                    const int* number_of_partners,
                                    const int* pointer,
                                    const double CL2, const double C2,
                                    const double dt, const int beg, const int pn);
#endif
   void UpdatePositionHalf(Variables *vars, SimulationInfo *sinfo);
   void CalculateForce(Variables *vars, MeshList *mesh, SimulationInfo *sinfo);
   void HeatbathZeta(Variables *vars, double ct, SimulationInfo *sinfo);
   void HeatbathMomenta(Variables *vars, SimulationInfo *sinfo);
   void Langevin(Variables *vars, SimulationInfo *sinfo);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
