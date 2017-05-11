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
void
MeshList::MakeTransposedList(Variables *vars,
                             SimulationInfo *sinfo,
                             const int pn_gpu,
                             cudaStream_t strm) {
  const int pn_tot = vars->GetTotalParticleNumber();
  const double S2  = sinfo->SearchLength * sinfo->SearchLength;

  CudaPtr2D<double, N, D>& q = vars->q_buf;

  q.Host2DevAsync(0, pn_tot, strm);
  mesh_index.Host2DevAsync(strm);
  sortbuf.Host2DevAsync(0, pn_tot, strm);

  const auto nmax_in_mesh = *std::max_element(mesh_particle_number,
                                              mesh_particle_number + number_of_mesh);

  const int tblock_size = ((nmax_in_mesh - 1) / WARP_SIZE + 1) * WARP_SIZE;
  const int smem_size   = tblock_size * sizeof(VecCuda);

  SearchMeshAllCUDA<<<number_of_mesh, tblock_size, smem_size, strm>>>((VecCuda*)q.GetDevPtr(),
                                                                      neigh_mesh_id.GetDevPtr(),
                                                                      mesh_index.GetDevPtr(),
                                                                      sortbuf.GetDevPtr(),
                                                                      thrust::raw_pointer_cast(transposed_list),
                                                                      number_of_partners.GetDevPtr(),
                                                                      S2,
                                                                      pn_gpu);
  // NOTE:
  // Synchronized in mdmanager.cc
}
//----------------------------------------------------------------------
void
MeshList::MakeNeighborMeshId(void) {
  int imesh_id = 0;
  for (int iz = 0; iz < mz; iz++)
    for (int iy = 0; iy < my; iy++)
      for (int ix = 0; ix < mx; ix++) {
        int jmesh_id = 0;
        for (int jz = -1; jz < 2; jz++)
          for (int jy = -1; jy < 2; jy++)
            for (int jx = -1; jx < 2; jx++) {
              int pos3d[] = {ix + jx, iy + jy, iz + jz};

              if (pos3d[X] <   0) pos3d[X] += mx;
              if (pos3d[X] >= mx) pos3d[X] -= mx;
              if (pos3d[Y] <   0) pos3d[Y] += my;
              if (pos3d[Y] >= my) pos3d[Y] -= my;
              if (pos3d[Z] <   0) pos3d[Z] += mz;
              if (pos3d[Z] >= mz) pos3d[Z] -= mz;

              neigh_mesh_id[27 * imesh_id + jmesh_id] = pos2index(pos3d[X], pos3d[Y], pos3d[Z]);
              jmesh_id++;
            }
        imesh_id++;
      }
  neigh_mesh_id.Host2Dev();
}
//----------------------------------------------------------------------
