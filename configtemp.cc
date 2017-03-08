//----------------------------------------------------------------------
#include "configtemp.h"
#include "mpistream.h"
#include "communicator.h"
#include "confmaker.h"
//----------------------------------------------------------------------
Configtemp configtemp;
//----------------------------------------------------------------------
void
Configtemp::Run(MDManager *mdm) {
  Parameter *param = mdm->GetParameter();
  ExactConfigurationMaker c(param, mdm->GetSystemSize());
  //ConfigurationMaker c(param);
  mdm->ExecuteAll(&c);
  const double v0 = param->GetDoubleDef("InitialVelocity", 1.0);
  mdm->SetInitialVelocity(v0);
  const int T_LOOP = param->GetIntegerDef("ThermalizeLoop", 1000);
  const int LOOP = param->GetIntegerDef("TotalLoop", 1000);
  const int OBSERVE_LOOP = param->GetIntegerDef("ObserveLoop", 100);
  mdm->SetControlTemperature(true);
  mdm->ShowSystemInformation();
  for (int i = 0; i < T_LOOP; i++) {
    if (i % OBSERVE_LOOP == 0) {
      mdm->MakePairList();
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->ConfigurationTemperature();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " #Thermalize" << std::endl;
    }
    mdm->Calculate();
  }
  mdm->SetControlTemperature(false);
  mdm->ShowSystemInformation();
  for (int i = 0; i < LOOP; i++) {
    mdm->Calculate();
    if (i % OBSERVE_LOOP == 0) {
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->ConfigurationTemperature();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " " << std::endl;
    }
  }

}
//----------------------------------------------------------------------
