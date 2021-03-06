//----------------------------------------------------------------------
#ifndef mdunit_h
#define mdunit_h
#include <stdio.h>
#include <vector>
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
#include "mdconfig.h"
#include "parainfo.h"
#include "simulationinfo.h"
#include "mdrect.h"
#include "parameter.h"
#include "variables.h"
#include "meshlist.h"
#include "pairlist.h"
#include "observer.h"
#include "fcalculator.h"
//----------------------------------------------------------------------
class MDUnit;
//----------------------------------------------------------------------
class Executor {
public:
  virtual void Execute(MDUnit *mdu) {};
};
//----------------------------------------------------------------------
class MDUnit {
private:
  Variables *vars;
  SimulationInfo *sinfo;
  ParaInfo *pinfo;
  MeshList *mesh;
  PairList *plist;
  const int id;
  std::vector<int> border_particles[MAX_DIR];
  MDRect myrect;
#ifdef USE_GPU
  cudaStream_t strm = 0;
  int pn_gpu = 0;
#endif
public:
  MDUnit (int id_, SimulationInfo *si, ParaInfo *pi);
  ~MDUnit(void);
  std::vector<ParticleInfo> send_buffer;
  int GetID(void) {return id;};
  MDRect * GetRect(void) {return &myrect;};
  Variables *GetVariables(void) {return vars;};
  void SaveConfiguration(void);
  void SaveAsCdview(std::ofstream &ofs);
  void AddParticle(double x[D], double v[D], int type = 1);
  void AddParticle(double x[D], int type = 1);
  double * GetSystemSize(void) {return sinfo->L;};
  double GetTimeStep(void) {return sinfo->TimeStep;};
  void SetInitialVelocity(double v0) {vars->SetInitialVelocity(v0, id);};
  int GetParticleNumber(void) {return vars->GetParticleNumber();};
  int GetTotalParticleNumber(void) {return vars->GetTotalParticleNumber();};
  void SetTotalParticleNumber(int n) {vars->SetTotalParticleNumber(n);};

  void CalculateForce(void) {ForceCalculator::CalculateForce(vars, mesh, sinfo);};
  void UpdatePositionHalf(void) {ForceCalculator::UpdatePositionHalf(vars, sinfo);};
  void HeatbathZeta(double t) {ForceCalculator::HeatbathZeta(vars, t, sinfo);};
  void HeatbathMomenta(void) {ForceCalculator::HeatbathMomenta(vars, sinfo);};
  void Langevin(void) {ForceCalculator::Langevin(vars, sinfo);};

#ifdef USE_GPU
  void SendNeighborInfoToGPUAsync(void) {mesh->SendNeighborInfoToGPUAsync(pn_gpu, strm);};
  void TransposeSortedList(void) {mesh->TransposeSortedList(pn_gpu, strm);};
  void UpdateParticleNumberGPU(const double work_balance) {
    pn_gpu = int(work_balance * vars->GetParticleNumber());
  };
  void SendParticlesHostToDev(void) {
    ForceCalculator::SendParticlesHostToDev(vars, pn_gpu, strm);
  }
  void SendParticlesDevToHost(void) {
    ForceCalculator::SendParticlesDevToHost(vars, pn_gpu, strm);
  }
  void CalculateForceCPU(void) {
    ForceCalculator::CalculateForce(vars, mesh, sinfo, pn_gpu);
  };
  void CalculateForceGPU(void) {
    ForceCalculator::CalculateForce(vars, mesh, sinfo, pn_gpu, strm);
  };
  void HeatbathMomentaCPU(void) {
    ForceCalculator::HeatbathMomenta(vars, sinfo, pn_gpu);
  };
  void HeatbathMomentaGPU(void) {
    ForceCalculator::HeatbathMomenta(vars, sinfo, pn_gpu, strm);
  };
#endif

  void MakeBufferForSendingParticle(const int dir);
  void FindBorderParticles(const int dir);
  void MakeBufferForBorderParticles(const int dir);
  void ReceiveParticles(std::vector<ParticleInfo> &recv_buffer);
  void AdjustPeriodicBoundary(void) {vars->AdjustPeriodicBoundary(sinfo);};
  void ReceiveBorderParticles(std::vector<ParticleInfo> &recv_buffer);
  double ObserveDouble(DoubleObserver *obs) {return obs->Observe(vars, mesh);};
  int IntegerDouble(IntegerObserver *obs) {return obs->Observe(vars, mesh);};
  void Execute(Executor *ex) {ex->Execute(this);};
  void MakePairList(void);
  void ShowPairs(void) {mesh->ShowPairs();};
  bool IsPairListExpired(void) {return plist->IsPairListExpired(vars, mesh, sinfo);};
  //
  void ChangeScale(double alpha);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
