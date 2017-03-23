//----------------------------------------------------------------------
#ifndef cavitation_h
#define cavitation_h
//----------------------------------------------------------------------
#include "projectmanager.h"
//----------------------------------------------------------------------
class Cavitation : public Project {
private:
public:
  Cavitation(void) {
    ProjectManager::GetInstance().AddProject("Cavitation", this);
  };
  void Run(MDManager *mdm);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------

