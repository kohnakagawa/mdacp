//----------------------------------------------------------------------
#ifndef meshlist_kernels_cuh
#define meshlist_kernels_cuh
//----------------------------------------------------------------------
#include "device_utils.cuh"
//----------------------------------------------------------------------
__global__ void
TransposeSortedListCUDA(const int* __restrict__ sorted_list,
                        const int* __restrict__ pointer,
                        const int* __restrict__ number_of_partners,
                        int*       __restrict__ transposed_list,
                        const int pn) {
  const auto ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (ptcl_id >= pn) return;

  const auto lid = lane_id();
  const auto kp  = pointer[ptcl_id];
  const auto np  = number_of_partners[ptcl_id];

  for (int k = lid; k < np; k += warpSize) {
    transposed_list[ptcl_id + pn * k] = sorted_list[kp + k];
  }
}
//----------------------------------------------------------------------
__global__ void
SearchMeshAllCUDA(const VecCuda* __restrict__ q,
                  const int*     __restrict__ neigh_mesh_id,
                  const int*     __restrict__ mesh_index,
                  const int*     __restrict__ sortbuf,
                  int*           __restrict__ transposed_list,
                  int*           __restrict__ number_of_partners,
                  const double search_length2,
                  const int    pn_gpu) {
  extern __shared__ VecCuda pos_buffer[];
  const auto tid_loc   = threadIdx.x;
  const auto i_cell_id = blockIdx.x;
  const auto tid       = mesh_index[i_cell_id    ] + tid_loc;
  const auto i_end_id  = mesh_index[i_cell_id + 1];

  int i_ptcl_id = -1;
  VecCuda qi    = {0.0};
  if (tid < i_end_id) {
    i_ptcl_id = sortbuf[tid];
    qi        = q[i_ptcl_id];
  }

  int n_neigh = 0;
  for (int cid = 0; cid < 27; cid++) {
    const auto j_cell_id  = neigh_mesh_id[27 * i_cell_id + cid];
    const auto j_beg_id   = mesh_index[j_cell_id    ];
    const auto j_end_id   = mesh_index[j_cell_id + 1];
    const auto num_loop_j = j_end_id - j_beg_id;

    __syncthreads();
    if (tid_loc < num_loop_j) {
      const auto j_ptcl_id = sortbuf[j_beg_id + tid_loc];
      pos_buffer[tid_loc].x = q[j_ptcl_id].x;
      pos_buffer[tid_loc].y = q[j_ptcl_id].y;
      pos_buffer[tid_loc].z = q[j_ptcl_id].z;
    }
    __syncthreads();

    if (tid < i_end_id) {
      for (int j = 0; j < num_loop_j; j++) {
        const auto j_ptcl_id = sortbuf[j_beg_id + j];
        if (i_ptcl_id == j_ptcl_id || i_ptcl_id >= pn_gpu) continue;
        const auto drx = qi.x - pos_buffer[j].x;
        const auto dry = qi.y - pos_buffer[j].y;
        const auto drz = qi.z - pos_buffer[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > search_length2) continue;
        transposed_list[pn_gpu * n_neigh + i_ptcl_id] = j_ptcl_id;
        n_neigh++;
      }
    }
  }

  if (tid < i_end_id && i_ptcl_id < pn_gpu) number_of_partners[i_ptcl_id] = n_neigh;
}
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
