//----------------------------------------------------------------------
#include "phaseflow.h"
#include "mpistream.h"
#include "confmaker.h"
//----------------------------------------------------------------------
PhaseFlow pf;
//----------------------------------------------------------------------
class CylinderAdder : public Executor {
private:
  const double *L;
  double c[D];
  double r;
  double xv;
  double density;
public:
  CylinderAdder(const double *_L, double _r, double _xv, double _density) : L(_L) {
    c[X] = L[X] * 0.5;
    c[Y] = L[Y] * 0.5;
    c[Z] = L[Z] * 0.5;
    r = _r;
    xv = _xv;
    density = _density;
  };
  void Add(double x[D], MDUnit *mdu) {
    const double dx = x[X] - c[X];
    const double dy = x[Y] - c[Y];
    const double dz = x[Z] - c[Z];
    const double r2 = dx * dx + dy * dy;
    double v[D] = {xv, 0, 0};
    if (r2  > r * r * 1.5) {
      mdu->AddParticle(x, v, 0);
    } else if (r2 < r * r) {
      mdu->AddParticle(x, v, 1);
    }
  }
  void Execute(MDUnit *mdu) {
    const double s = 1.0 / pow(density * 0.25, 1.0 / 3.0);
    const int sx = static_cast<int>(L[X] / s);
    const int sy = static_cast<int>(L[Y] / s);
    const int sz = static_cast<int>(L[Z] / s);
    const double hs = s * 0.5;
    for (int iz = 0; iz < sz; iz++) {
      for (int iy = 0; iy < sy; iy++) {
        for (int ix = 0; ix < sx; ix++) {
          double x1[D] = {ix * s, iy * s, iz * s};
          double x2[D] = {x1[X] + hs, x1[Y], x1[Z]};
          double x3[D] = {x1[X], x1[Y] + hs, x1[Z]};
          double x4[D] = {x1[X], x1[Y], x1[Z] + hs};
          Add(x1, mdu);
          Add(x2, mdu);
          Add(x3, mdu);
          Add(x4, mdu);
        }
      }
    }
  }
};
//----------------------------------------------------------------------
class GravityImposer : public Executor {
private:
  const double G;
  const double dt;
public:
  GravityImposer(const double g, const double t) : G(g), dt(t) {
  };
  void Execute(MDUnit *mdu) {
    Variables *vars = mdu->GetVariables();
    const int pn = vars->GetParticleNumber();
    double (*p)[D] = vars->p;
    for (int i = 0; i < pn; i++) {
      p[i][X] += G * dt;
    }
  }
};
//----------------------------------------------------------------------
void
PhaseFlow::Run(MDManager *mdm) {
  Parameter *param = mdm->GetParameter();
  const double *L = mdm->GetSystemSize();
  CylinderAdder ca(L, L[Y] * 0.25, 1.0, 0.374);
  mdm->ExecuteAll(&ca);
  const double v0 = param->GetDoubleDef("InitialVelocity", 1.0);
  const double dt = param->GetDoubleDef("TimeStep", 0.001);
  mdm->SetControlTemperature(true);
  mdm->SetInitialVelocity(v0);
  mdm->SaveAsCdviewSequential();
  const int OBSERVE_LOOP = param->GetIntegerDef("ObserveLoop", 100);
  const int TOTAL_LOOP = param->GetIntegerDef("TotalLoop", 1000);
  const bool save_cdview_file = param->GetBooleanDef("SaveCdviewFile", false);
  mdm->ShowSystemInformation();
  GravityImposer gi(0.2, dt);
  for (int i = 0; i < TOTAL_LOOP; i++) {
    mdm->Calculate();
    mdm->ExecuteAll(&gi);
    if (i % OBSERVE_LOOP == 0) {
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << "# observe" << std::endl;
      if (save_cdview_file) {
        mdm->SaveAsCdviewSequential();
      }
    }
  }
}
//----------------------------------------------------------------------
