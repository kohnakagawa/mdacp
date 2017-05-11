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
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
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
#ifdef AVX2
  int32_t shfl_table[16][8];
  void MakeShflTable(void);
  void SearchMeshAVX2(int index, Variables *vars, SimulationInfo *sinfo);
#elif AVX512
  int64_t shfl_table[256][8];
  void MakeShflTable(void);
  void SearchMeshAVX512(int index, Variables *vars, SimulationInfo *sinfo);
#endif
  int key_partner_pairs[PAIRLIST_SIZE][2];

#ifdef USE_GPU
  CudaPtr<int> mesh_index;
  CudaPtr<int> sortbuf;
#else
  int * mesh_index;
  int sortbuf[N];
#endif
  int * mesh_index2;
  int * mesh_particle_number;

#ifdef USE_GPU
  CudaPtr<int> key_pointer;
  CudaPtr<int> number_of_partners;
  CudaPtr<int> sorted_list;
  thrust::device_ptr<int> transposed_list;
  CudaPtr<int> neigh_mesh_id;
#else
  int key_pointer[N];
  int number_of_partners[N];
  int sorted_list[PAIRLIST_SIZE];
#endif
  int key_pointer2[N];
  int number_of_mesh;

  int number_of_constructions;
  inline void RegisterPair(int index1, int index2);
  inline void RegisterInteractPair(const double q[][D], int index1, int index2, const double S2);
  int sort_interval;

  void ClearPartners(Variables *vars);
  void MakeListMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void MakeMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);

  inline void index2pos(int index, int &ix, int &iy, int &iz);
  int pos2index(int ix, int iy, int iz) {
    return mx * my * iz + mx * iy + ix;
  }
  void SearchMesh(int index, Variables *vars, SimulationInfo *sinfo);
  void AppendList(int mx, int my, int mz, std::vector<int> &v);

public:
  MeshList(SimulationInfo *sinfo, MDRect &r);
  ~MeshList(void);

  void ChangeScale(SimulationInfo *sinfo, MDRect &myrect);

  enum {
    KEY = 0,
    PARTNER = 1,
  };
  int (*GetKeyPartnerPairs(void))[2] {return key_partner_pairs;};

  int GetPairNumber(void) {return number_of_pairs;};
  int GetPartnerNumber(int i) {return number_of_partners[i];};
  int GetKeyPointer(int i) {return key_pointer[i];};
#ifdef USE_GPU
  const CudaPtr<int>& GetCudaPtrSortedList(void) const {return sorted_list;};
  const CudaPtr<int>& GetCudaPtrKeyPointerP(void) const {return key_pointer;};
  const CudaPtr<int>& GetCudaPtrNumberOfPartners(void) const {return number_of_partners;};
  int *GetSortedList(void) {return sorted_list.GetHostPtr();};
  int* GetKeyPointerP(void) {return key_pointer.GetHostPtr();};
  int* GetNumberOfPartners(void) {return number_of_partners.GetHostPtr();};
  int* GetDevPtrTransposedList(void) {return thrust::raw_pointer_cast(transposed_list);};
  void SendNeighborInfoToGPUAsync(const int pn_gpu, cudaStream_t strm = 0);
  void TransposeSortedList(const int pn_gpu, cudaStream_t strm = 0);
  void MakeTransposedList(Variables *vars, SimulationInfo *sinfo, const int pn_gpu, cudaStream_t strm = 0);
  void MakeNeighborMeshId(void);
#else
  int *GetSortedList(void) {return sorted_list;};
  int* GetKeyPointerP(void) {return key_pointer;};
  int* GetNumberOfPartners(void) {return number_of_partners;};
#endif

  void Sort(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  int GetNumberOfConstructions(void) {return number_of_constructions;};
  void ClearNumberOfConstructions(void) {number_of_constructions = 0;};

  void MakeList(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void MakeMeshForSearch(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void SearchMeshAll(Variables *vars, SimulationInfo *sinfo);
  void MakeSortedList(Variables *vars);
  void MakeListBruteforce(Variables *vars, SimulationInfo *sinfo, MDRect &myrect);
  void ShowPairs(void);
  void ShowSortedList(Variables *vars);
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
