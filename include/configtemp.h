//----------------------------------------------------------------------
#ifndef configtemp_h
#define configtemp_h
//----------------------------------------------------------------------
#include "projectmanager.h"
//----------------------------------------------------------------------
class Configtemp : public Project {
private:
public:
  Configtemp(void) {
    ProjectManager::GetInstance().AddProject("Configtemp", this);
  };
  void Run(MDManager *mdm);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------

