//----------------------------------------------------------------------
#include "meshlist.h"
#include "meshlist_kernels.cuh"
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
