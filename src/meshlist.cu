//----------------------------------------------------------------------
#include "meshlist.h"
#include "mpistream.h"
#include "meshlist_kernels.cuh"
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
//----------------------------------------------------------------------
void
MeshList::TransposeSortedList(const int pn_gpu, cudaStream_t strm) {
  const auto gr_size = (WARP_SIZE * pn_gpu - 1) / THREAD_BLOCK_SIZE + 1;
  TransposeSortedListCUDA<<<gr_size, THREAD_BLOCK_SIZE, 0, strm>>>(sorted_list.GetDevPtr(),
                                                                   key_pointer.GetDevPtr(),
                                                                   number_of_partners.GetDevPtr(),
                                                                   thrust::raw_pointer_cast(transposed_list),
                                                                   pn_gpu);
  // NOTE:
  // Synchronized in mdmanager.cc
}
//----------------------------------------------------------------------
void
MeshList::AllocateOnGPU(void) {
  key_pointer.Allocate(N);
  number_of_partners.Allocate(N);
  sorted_list.Allocate(PAIRLIST_SIZE);
  transposed_list = thrust::device_malloc<int>(PAIRLIST_SIZE);
  thrust::fill_n(transposed_list, PAIRLIST_SIZE, 0);
}
//----------------------------------------------------------------------
void
MeshList::DeallocateOnGPU(void) {
  thrust::device_free(transposed_list);
}
//----------------------------------------------------------------------
void
MeshList::SendNeighborInfoToGPUAsync(const int pn_gpu, cudaStream_t strm) {
  key_pointer.Host2DevAsync(0, pn_gpu, strm);
  number_of_partners.Host2DevAsync(0, pn_gpu, strm);
  const auto number_of_pairs_gpu = std::accumulate(number_of_partners.GetHostPtr(),
                                                   number_of_partners.GetHostPtr() + pn_gpu,
                                                   0);
  sorted_list.Host2DevAsync(0, number_of_pairs_gpu, strm);

  const auto max_number_of_partners = *std::max_element(number_of_partners.GetHostPtr(),
                                                        number_of_partners.GetHostPtr() + pn_gpu);
  if (max_number_of_partners * pn_gpu > PAIRLIST_SIZE) {
    mout << "# Expand transposed_list size at " << __FILE__ << " " << __LINE__ << std::endl;
    mout << "# WARNING! You should increase PAIRLIST_SIZE in mdconfig.h" << std::endl;
    checkCudaErrors(cudaStreamSynchronize(strm));
    thrust::device_free(transposed_list);
    transposed_list = thrust::device_malloc<int>(2 * max_number_of_partners * pn_gpu);
  }
}
//----------------------------------------------------------------------
