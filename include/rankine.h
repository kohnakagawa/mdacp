//----------------------------------------------------------------------
#ifndef rankine_h
#define rankine_h
//----------------------------------------------------------------------
#include "projectmanager.h"
//----------------------------------------------------------------------
class Rankine : public Project {
private:
public:
  Rankine(void) {
    ProjectManager::GetInstance().AddProject("Rankine", this);
  };
  void Run(MDManager *mdm);
  void Piston(double position, MDManager *mdm);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------

