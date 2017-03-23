//----------------------------------------------------------------------
#ifndef phaseflow_h
#define phaseflow_h
//----------------------------------------------------------------------
#include "projectmanager.h"
//----------------------------------------------------------------------
class PhaseFlow : public Project {
private:
public:
  PhaseFlow(void) {
    ProjectManager::GetInstance().AddProject("PhaseFlow", this);
  };
  void Run(MDManager *mdm);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------

