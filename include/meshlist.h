//----------------------------------------------------------------------
#ifndef meshlist_h
#define meshlist_h
//----------------------------------------------------------------------
#include <vector>
#include "mdconfig.h"
#include "variables.h"
#include "mdrect.h"
#ifdef USE_GPU
#include "cuda_ptr.h"
#endif
//----------------------------------------------------------------------
class MeshList {
private:
  int number_of_pairs;
  double mesh_size_x;
  double mesh_size_y;
  double mesh_size_z;
  int mx, my, mz;
  int particle_position[N];
#ifdef MESH_SIMD
  int shfl_table[16][8];
  void MakeShflTable();
  int key_partner_pairs[PAIRLIST_SIZE][2];
#else
  int *key_particles;
  int *partner_particles;
#endif

  int * mesh_index;
  int * mesh_index2;
  int * mesh_particle_number;
  int sortbuf[N];
#ifdef USE_GPU
  CudaPtr<int> key_pointer;
#else
  int key_pointer[N];
#endif
  int key_pointer2[N];
  int number_of_mesh;
#ifdef USE_GPU
  CudaPtr<int> number_of_partners;
  CudaPtr<int> sorted_list;
#else
  int number_of_partners[N];
  int sorted_list[PAIRLIST_SIZE];
#endif
  int number_of_constructions;
  inline void RegisterPair(int index1, int index2);
  inline void RegisterInteractPair(const double q[][D], int index1, int index2, const double S2);
  int sort_interval;

  void MakeListMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void MakeMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  inline void index2pos(int index, int &ix, int &iy, int &iz);
  inline int pos2index(int ix, int iy, int iz);
  void SearchMesh(int index, Variables *vars, SimulationInfo *sinfo);
  void AppendList(int mx, int my, int mz, std::vector<int> &v);

public:
  MeshList(SimulationInfo *sinfo, MDRect &r);
  ~MeshList(void);

  void ChangeScale(SimulationInfo *sinfo, MDRect &myrect);

#ifdef MESH_SIMD
  enum {
    KEY = 0,
    PARTNER = 1,
  };
  int (*GetKeyPartnerPairs(void))[2] {return key_partner_pairs;};
#else
  int *GetKeyParticles(void) {return key_particles;};
  int *GetPartnerParticles(void) {return partner_particles;};
#endif

  int GetPairNumber(void) {return number_of_pairs;};

  int GetPartnerNumber(int i) {return number_of_partners[i];};
  int GetKeyPointer(int i) {return key_pointer[i];};
#ifdef USE_GPU
  const CudaPtr<int>& GetCudaPtrSortedList(void) const {return sorted_list;};
  const CudaPtr<int>& GetCudaPtrKeyPointerP(void) const {return key_pointer;};
  const CudaPtr<int>& GetCudaPtrNumberOfPartners(void) const {return number_of_partners;};
  void SendNeighborInfoToGPU(Variables *vars);
  int *GetSortedList(void) {return sorted_list.GetHostPtr();};
  int* GetKeyPointerP(void) {return key_pointer.GetHostPtr();};
  int* GetNumberOfPartners(void) {return number_of_partners.GetHostPtr();};
#else
  int *GetSortedList(void) {return sorted_list;};
  int* GetKeyPointerP(void) {return key_pointer;};
  int* GetNumberOfPartners(void) {return number_of_partners;};
#endif

  void Sort(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  int GetNumberOfConstructions(void) {return number_of_constructions;};
  void ClearNumberOfConstructions(void) {number_of_constructions = 0;};

  void MakeList(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void MakeListBruteforce(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void ShowPairs(void);
  void ShowSortedList(Variables *vars);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
