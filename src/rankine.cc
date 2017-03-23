//----------------------------------------------------------------------
#include "rankine.h"
#include "mpistream.h"
#include "confmaker.h"
//----------------------------------------------------------------------
Rankine rankine;
//----------------------------------------------------------------------
class RankineConfigurationMaker: public Executor {
private:
  Parameter *param;
public:
  RankineConfigurationMaker(Parameter *param_) {
    param = param_;
  };
  void
  Execute(MDUnit *mdu) {
    double *L = mdu->GetSystemSize();
    const double density = param->GetDoubleDef("Density", 0.5);
    const double s = 1.0 / pow(density * 0.25, 1.0 / 3.0);
    const double hs = s * 0.5;
    int sx = static_cast<int>(L[X] / s);
    int sy = static_cast<int>(L[Y] / s);
    int sz = static_cast<int>(L[Z] / s);
    double x[D];
    const double e = 0.0000001;
    for (int iz = 1; iz < sz - 1; iz++) {
      for (int iy = 0; iy < sy; iy++) {
        for (int ix = 0; ix < sx; ix++) {
          x[X] = static_cast<double>(ix) * s + e;
          x[Y] = static_cast<double>(iy) * s + e;
          x[Z] = static_cast<double>(iz) * s + e;
          mdu->AddParticle(x);

          x[X] = static_cast<double>(ix) * s + e;
          x[Y] = static_cast<double>(iy) * s + hs + e;
          x[Z] = static_cast<double>(iz) * s + hs + e;
          mdu->AddParticle(x);

          x[X] = static_cast<double>(ix) * s + hs + e;
          x[Y] = static_cast<double>(iy) * s + e;
          x[Z] = static_cast<double>(iz) * s + hs + e;
          mdu->AddParticle(x);

          x[X] = static_cast<double>(ix) * s + hs + e;
          x[Y] = static_cast<double>(iy) * s + hs + e;
          x[Z] = static_cast<double>(iz) * s + e;
          mdu->AddParticle(x);
        }
      }
    }

  };
};
//----------------------------------------------------------------------
class MyPiston : public Executor {
private:
  Parameter *param;
public:
  double position;
  double impulse;
  MyPiston(Parameter *param_) {
    position = 0.0;
    impulse = 0.0;
    param = param_;
  };
  void Execute(MDUnit *mdu) {
    double *L = mdu->GetSystemSize();
    const double dt = mdu->GetTimeStep();
    const double cz = L[Z] * 0.5;
    Variables *vars = mdu->GetVariables();
    const int pn = mdu->GetVariables()->GetParticleNumber();
    double (*q)[D] = mdu->GetVariables()->q;
    double (*p)[D] = mdu->GetVariables()->p;
    const double CL = CUTOFF_LENGTH;
    const double K = 0.1;
    const double C2 = vars->GetC2();
    impulse = 0.0;
    for (int i = 0; i < pn; i++) {
      if (q[i][Z] < CL) {
        const double dz = q[i][Z];
        const double r2 = dz * dz;
        double r6 = r2 * r2 * r2;
        double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2) + C2 * 8.0) * dt;
        p[i][Z] -= df * dz;
      }
      if (q[i][Z] > position ) {
        const double dz = position - q[i][Z] + CL;
        const double r2 = dz * dz;
        double r6 = r2 * r2 * r2;
        double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2) + C2 * 8.0) * dt;
        p[i][Z] += df * dz;
        impulse -= df * dz;
      }
    }
  };
};
//----------------------------------------------------------------------
void
Rankine::Run(MDManager *mdm) {
  Parameter *param = mdm->GetParameter();
  RankineConfigurationMaker c(param);
  MyPiston mp(param);
  const double LZ = param->GetDoubleDef("SystemSizeZ", 40);
  mp.position = LZ - 2.0;
  mdm->ExecuteAll(&c);
  const double v0 = param->GetDoubleDef("InitialVelocity", 1.0);
  mdm->SetInitialVelocity(v0);
  const int LOOP = param->GetIntegerDef("TotalLoop", 1000);
  const int OBSERVE_LOOP = param->GetIntegerDef("ObserveLoop", 100);
  mdm->ShowSystemInformation();
  mdm->MakePairList();
  const double d_pos = LZ * 0.5 / LOOP;

  //Push Piston
  for (int i = 0; i < LOOP; i++) {
    mdm->Calculate();
    mdm->ExecuteAll(&mp);
    mp.position -= d_pos;
    if (i % OBSERVE_LOOP == 0) {
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " " << mp.position;
      mout << " " << mp.impulse;
      mout << " #observe" << std::endl;
      mdm->SaveAsCdviewSequential();
    }
  }
  // Cooling
  mdm->SetControlTemperature(true);
  mdm->SetAimedTemperature(0.5);
  for (int i = 0; i < LOOP; i++) {
    mdm->Calculate();
    mdm->ExecuteAll(&mp);
    if (i % OBSERVE_LOOP == 0) {
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " " << mp.position;
      mout << " " << mp.impulse;
      mout << " #observe" << std::endl;
      mdm->SaveAsCdviewSequential();
    }
  }
}
//----------------------------------------------------------------------
