//----------------------------------------------------------------------
#include <iostream>
#include <assert.h>
#include <fstream>
#include <algorithm>
#include "meshlist.h"
#include "mpistream.h"
#ifdef MESH_SIMD
#include "simd_avx2.h"
#endif
//----------------------------------------------------------------------
MeshList::MeshList(SimulationInfo *sinfo, MDRect &r) {
#ifndef MESH_SIMD
  key_particles = new int[PAIRLIST_SIZE];
  partner_particles = new int[PAIRLIST_SIZE];
#endif

  number_of_constructions = 0;
  sort_interval = 10;

  mesh_index = NULL;
  mesh_index2 = NULL;
  mesh_particle_number = NULL;
  ChangeScale(sinfo, r);

#ifdef MESH_SIMD
  MakeShflTable();
#endif
}
//----------------------------------------------------------------------
MeshList::~MeshList(void) {

  if (NULL != mesh_index) delete [] mesh_index;
  if (NULL != mesh_index2) delete [] mesh_index2;
  if (NULL != mesh_particle_number) delete [] mesh_particle_number;

#ifndef MESH_SIMD
  delete [] key_particles;
  delete [] partner_particles;
#endif
}
//----------------------------------------------------------------------
void
MeshList::ChangeScale(SimulationInfo *sinfo, MDRect &myrect) {
  if (NULL != mesh_index) delete [] mesh_index;
  if (NULL != mesh_index2) delete [] mesh_index2;
  if (NULL != mesh_particle_number) delete [] mesh_particle_number;
  double wx = myrect.GetWidth(X);
  double wy = myrect.GetWidth(Y);
  double wz = myrect.GetWidth(Z);
  const double SL = sinfo->SearchLength;

  int msx = static_cast<int>(wx / SL);
  int msy = static_cast<int>(wy / SL);
  int msz = static_cast<int>(wz / SL);
  mesh_size_x = wx / static_cast<double>(msx);
  mesh_size_y = wy / static_cast<double>(msy);
  mesh_size_z = wz / static_cast<double>(msz);
  mx = static_cast <int> (wx / mesh_size_x) + 2;
  my = static_cast <int> (wy / mesh_size_y) + 2;
  mz = static_cast <int> (wz / mesh_size_z) + 2;

  number_of_mesh = mx * my * mz;
  mesh_index = new int[number_of_mesh];
  mesh_index2 = new int[number_of_mesh];
  mesh_particle_number = new int[number_of_mesh];

}
//----------------------------------------------------------------------
void
MeshList::MakeList(Variables *vars, SimulationInfo *sinfo, MDRect &myrect) {
  number_of_pairs = 0;
  const int pn = vars->GetTotalParticleNumber();
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
  MakeListMesh(vars, sinfo, myrect);
  //MakeListBruteforce(vars,sinfo,myrect);

#ifdef MESH_SIMD
  const int s = number_of_pairs;
  for (int k = 0; k < s; k++) {
    const int i = key_partner_pairs[k][KEY];
    number_of_partners[i]++;
  }
#endif

  int pos = 0;
  key_pointer[0] = 0;
  for (int i = 0; i < pn - 1; i++) {
    pos += number_of_partners[i];
    key_pointer[i + 1] = pos;
  }

  for (int i = 0; i < pn; i++) {
    key_pointer2[i] = 0;
  }

#ifndef MESH_SIMD
  const int s = number_of_pairs;
#endif
  for (int k = 0; k < s; k++) {
#ifdef MESH_SIMD
    int i = key_partner_pairs[k][KEY];
    int j = key_partner_pairs[k][PARTNER];
#else
    int i = key_particles[k];
    int j = partner_particles[k];
#endif
    int index = key_pointer[i] + key_pointer2[i];
    sorted_list[index] = j;
    key_pointer2[i] ++;
  }
  number_of_constructions++;
}
//----------------------------------------------------------------------
void
MeshList::MakeListBruteforce(Variables *vars, SimulationInfo *sinfo, MDRect &myrect) {
  const int pn = vars->GetParticleNumber();
  const int tn = vars->GetTotalParticleNumber();
  const double SL2 = sinfo->SearchLength * sinfo->SearchLength;
  double (*q)[D] = vars->q;
  for (int i = 0; i < pn; i++) {
    for (int j = i + 1; j < tn; j++) {
      const double dx = q[i][X] - q[j][X];
      const double dy = q[i][Y] - q[j][Y];
      const double dz = q[i][Z] - q[j][Z];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < SL2) {
        RegisterPair(i, j);
      }
    }
  }
}
//----------------------------------------------------------------------
void
MeshList::MakeListMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect) {
  MakeMesh(vars, sinfo, myrect);
  for (int i = 0; i < number_of_mesh; i++) {
    SearchMesh(i, vars, sinfo);
  }
}
//----------------------------------------------------------------------
void
MeshList::Sort(Variables *vars, SimulationInfo *sinfo, MDRect &myrect) {

  if (!sinfo->SortParticle)return;

  if (sinfo->SortParticle && number_of_constructions % sort_interval != 0) {
    return;
  }

  const int pn = vars->GetParticleNumber();
  vars->SetTotalParticleNumber(pn);

  MakeMesh(vars, sinfo, myrect);

  static double q2[N][D];
  static double p2[N][D];
  double (*q)[D] = vars->q;
  double (*p)[D] = vars->p;

  for (int i = 0; i < pn; i++) {
    for (int d = 0; d < D; d++) {
      int j = sortbuf[i];
      q2[i][d] = q[j][d];
      p2[i][d] = p[j][d];
    }
  }

  for (int i = 0; i < pn; i++) {
    for (int d = 0; d < D; d++) {
      q[i][d] = q2[i][d];
      p[i][d] = p2[i][d];
    }
  }
  mout << "# Sorted!" << std::endl;
}
//----------------------------------------------------------------------
void
MeshList::MakeMesh(Variables *vars, SimulationInfo *sinfo, MDRect &myrect) {

  const int pn = vars->GetTotalParticleNumber();
  double (*q)[D] = vars->q;

  double imx = 1.0 / mesh_size_x;
  double imy = 1.0 / mesh_size_y;
  double imz = 1.0 / mesh_size_z;
  double *s = myrect.GetStartPosition();

  for (int i = 0; i < number_of_mesh; i++) {
    mesh_particle_number[i] = 0;
  }
  for (int i = 0; i < pn; i++) {
    int ix = static_cast<int>((q[i][X] - s[X]) * imx) + 1;
    int iy = static_cast<int>((q[i][Y] - s[Y]) * imy) + 1;
    int iz = static_cast<int>((q[i][Z] - s[Z]) * imz) + 1;

    if (ix < 0 ) ix = mx - 1;
    else if (ix >= mx) ix = 0;
    if (iy < 0 ) iy = my - 1;
    else if (iy >= my) iy = 0;
    if (iz < 0 ) iz = mz - 1;
    else if (iz >= mz) iz = 0;

    int index = mx * my * iz + mx * iy + ix;
    if (index >= number_of_mesh || index < 0) {
      show_error("Invalid index");
      printf("%d %d %d %d \n", ix, iy, iz, i);
      printf("%f %f %f \n", q[i][X], q[i][Y], q[i][Z]);
      exit(1);
    }
    particle_position[i] = index;
    mesh_particle_number[index]++;

  }
  mesh_index[0] = 0;
  int sum = 0;
  for (int i = 0; i < number_of_mesh - 1; i++) {
    sum += mesh_particle_number[i];
    mesh_index[i + 1] = sum;
  }
  for (int i = 0; i < number_of_mesh; i++) {
    mesh_index2[i] = 0;
  }

  for (int i = 0; i < pn; i++) {
    int index = particle_position[i];
    int j = mesh_index[index] + mesh_index2[index];
    sortbuf[j] = i;
    mesh_index2[index]++;
  }
}
//----------------------------------------------------------------------
void
MeshList::AppendList(int ix, int iy, int iz, std::vector<int> &v) {
  if (ix < 0 || ix >= mx)return;
  if (iy < 0 || iy >= my)return;
  if (iz < 0 || iz >= mz)return;

  const int index = pos2index(ix, iy, iz);
  const int in = mesh_particle_number[index];
  const int mi = mesh_index[index];
  v.insert(v.end(), &sortbuf[mi], &sortbuf[mi + in]);
}
//----------------------------------------------------------------------
void
MeshList::SearchMesh(int index, Variables *vars, SimulationInfo *sinfo) {

  int ix, iy, iz;
  index2pos(index, ix, iy, iz);
#ifdef FX10
  static __thread std::vector<int> v;
#else
  std::vector<int> v;
#endif
  v.clear();

  AppendList(ix, iy, iz, v);
  AppendList(ix + 1,  iy, iz, v);
  AppendList(ix - 1,  iy + 1, iz, v);
  AppendList(ix,  iy + 1, iz, v);
  AppendList(ix + 1,  iy + 1, iz, v);

  AppendList(ix - 1,  iy, iz + 1, v);
  AppendList(ix,  iy, iz + 1, v);
  AppendList(ix + 1,  iy, iz + 1, v);
  AppendList(ix - 1,  iy - 1, iz + 1, v);
  AppendList(ix,  iy - 1, iz + 1, v);
  AppendList(ix + 1,  iy - 1, iz + 1, v);
  AppendList(ix - 1,  iy + 1, iz + 1, v);
  AppendList(ix,  iy + 1, iz + 1, v);
  AppendList(ix + 1,  iy + 1, iz + 1, v);

  const double S2 = sinfo->SearchLength * sinfo->SearchLength;
  const int pn = vars->GetParticleNumber();
  double (*q)[D] = vars->q;

  const int in = mesh_particle_number[index];
  const int ln = v.size();

#ifdef MESH_SIMD
  const auto vpn = _mm256_set1_epi64x(pn);
  const auto vsl2 = _mm256_set1_pd(S2);
  for (int i = 0; i < (in / 4) * 4 ; i += 4) {
    const auto i_a = v[i    ];
    const auto i_b = v[i + 1];
    const auto i_c = v[i + 2];
    const auto i_d = v[i + 3];

    const auto vqia = _mm256_loadu_pd(q[i_a]);
    const auto vqib = _mm256_loadu_pd(q[i_b]);
    const auto vqic = _mm256_loadu_pd(q[i_c]);
    const auto vqid = _mm256_loadu_pd(q[i_d]);

    auto vi_id = _mm256_set_epi64x(i_d, i_c, i_b, i_a);

    v4df vqix, vqiy, vqiz;
    transpose_4x4(vqia, vqib, vqic, vqid, vqix, vqiy, vqiz);

    const int i_less_than_pn = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(vpn, vi_id)));
    for (int k = i + 4; k < ln; k++) {
      const auto j = v[k];
      const int j_less_than_pn = (j < pn) ? 0xf : 0;

      auto vqjx = _mm256_set1_pd(q[j][X]);
      auto vqjy = _mm256_set1_pd(q[j][Y]);
      auto vqjz = _mm256_set1_pd(q[j][Z]);

      auto dvx = _mm256_sub_pd(vqjx, vqix);
      auto dvy = _mm256_sub_pd(vqjy, vqiy);
      auto dvz = _mm256_sub_pd(vqjz, vqiz);

      auto dvr2 = _mm256_fmadd_pd(dvx, dvx,
                                  _mm256_fmadd_pd(dvy, dvy,
                                                  _mm256_mul_pd(dvz, dvz))) ;

      auto dvr2_flag = _mm256_cmp_pd(dvr2, vsl2, _CMP_LE_OS);
      int less_than_sl2 = _mm256_movemask_pd(dvr2_flag);

      const int shfl_key = (i_less_than_pn | j_less_than_pn) & less_than_sl2;
      if (shfl_key == 0) continue;

      const int incr = _popcnt32(shfl_key);

      auto vj_id = _mm256_set1_epi64x(j);
      auto vkey_id = _mm256_min_epi32(vi_id, vj_id);
      auto vpart_id = _mm256_max_epi32(vi_id, vj_id);
      vpart_id = _mm256_slli_si256(vpart_id, 0x4);
      auto vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

      auto idx = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(shfl_table[shfl_key]));
      vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_pairs[number_of_pairs]),
                          vpart_key_id);
      number_of_pairs += incr;
#ifdef USE_GPU
      vpart_key_id = _mm256_shuffle_epi32(vpart_key_id, 0xb1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_pairs[number_of_pairs]),
                          vpart_key_id);
      number_of_pairs += incr;
#endif
    }

    // remaining pairs
    if (i_a < pn || i_b < pn) RegisterInteractPair(q, i_a, i_b, S2);
    if (i_a < pn || i_c < pn) RegisterInteractPair(q, i_a, i_c, S2);
    if (i_a < pn || i_d < pn) RegisterInteractPair(q, i_a, i_d, S2);
    if (i_b < pn || i_c < pn) RegisterInteractPair(q, i_b, i_c, S2);
    if (i_b < pn || i_d < pn) RegisterInteractPair(q, i_b, i_d, S2);
    if (i_c < pn || i_d < pn) RegisterInteractPair(q, i_c, i_d, S2);
  }

  // remaining i loop
  for (int i = (in / 4) * 4 ; i < in; i++) {
    const int i1 = v[i];
    const double x1 = q[i1][X];
    const double y1 = q[i1][Y];
    const double z1 = q[i1][Z];
    for (int j = i + 1; j < ln; j++) {
      const int i2 = v[j];
      if (i1 >= pn && i2 >= pn)continue;
      const double dx = x1 - q[i2][X];
      const double dy = y1 - q[i2][Y];
      const double dz = z1 - q[i2][Z];
      const double r2 = (dx * dx + dy * dy + dz * dz);
      if (r2 > S2) continue;
      RegisterPair(i1, i2);
    }
  }
#else
  for (int i = 0; i < in; i++) {
    const int i1 = v[i];
    const double x1 = q[i1][X];
    const double y1 = q[i1][Y];
    const double z1 = q[i1][Z];
    for (int j = i + 1; j < ln; j++) {
      const int i2 = v[j];
      if (i1 >= pn && i2 >= pn)continue;
      const double dx = x1 - q[i2][X];
      const double dy = y1 - q[i2][Y];
      const double dz = z1 - q[i2][Z];
      const double r2 = (dx * dx + dy * dy + dz * dz);
      if (r2 > S2) continue;
      RegisterPair(i1, i2);
    }
  }
#endif
}
//----------------------------------------------------------------------
void
MeshList::index2pos(int index, int &ix, int &iy, int &iz) {
  ix = index % mx;
  index /= mx;
  iy = index % my;
  index /= my;
  iz = index;
}
//----------------------------------------------------------------------
int
MeshList::pos2index(int ix, int iy, int iz) {
  return mx * my * iz + mx * iy + ix;
}
//----------------------------------------------------------------------
inline void
MeshList::RegisterPair(int index1, int index2) {
  int i1, i2;
  if (index1 < index2) {
    i1 = index1;
    i2 = index2;
  } else {
    i1 = index2;
    i2 = index1;
  }

#ifdef MESH_SIMD
  key_partner_pairs[number_of_pairs][KEY] = i1;
  key_partner_pairs[number_of_pairs][PARTNER] = i2;
  number_of_pairs++;
#ifdef USE_GPU
  key_partner_pairs[number_of_pairs][KEY] = i2;
  key_partner_pairs[number_of_pairs][PARTNER] = i1;
  number_of_pairs++;
#endif // end of USE_GPU
#else // else MESH_SIMD
  key_particles[number_of_pairs] = i1;
  partner_particles[number_of_pairs] = i2;
  number_of_partners[i1]++;
  number_of_pairs++;
#ifdef FX10
  key_particles[number_of_pairs] = i2;
  partner_particles[number_of_pairs] = i1;
  number_of_partners[i2]++;
  number_of_pairs++;
#endif // end of FX10
#endif // end of MESH_SIMD

  assert(number_of_pairs < PAIRLIST_SIZE);
}
//----------------------------------------------------------------------
inline void
MeshList::RegisterInteractPair(const double q[][D],
                               int index1,
                               int index2,
                               const double S2) {
  const double dx = q[index1][X] - q[index2][X];
  const double dy = q[index1][Y] - q[index2][Y];
  const double dz = q[index1][Z] - q[index2][Z];
  const double r2 = (dx * dx + dy * dy + dz * dz);
  if (r2 > S2) return;
  RegisterPair(index1, index2);
}
//----------------------------------------------------------------------
void
MeshList::ShowPairs(void) {
  for (int i = 0; i < number_of_pairs; i++) {
#ifdef MESH_SIMD
    printf("(%05d,%05d)\n",
           key_partner_pairs[i][KEY], key_partner_pairs[i][PARTNER]);
#else
    printf("(%05d,%05d)\n", key_particles[i], partner_particles[i]);
#endif
  }
}
//----------------------------------------------------------------------
void
MeshList::ShowSortedList(Variables *vars) {
  const int pn = vars->GetTotalParticleNumber();
  for (int i = 0; i < pn; i++) {
    const int np = GetPartnerNumber(i);
    const int kp = GetKeyPointer(i);
    for (int k = 0; k < np; k++) {
      const int j = sorted_list[kp + k];
      printf("(%05d,%05d)\n", i, j);
    }
  }
}
//----------------------------------------------------------------------
#ifdef MESH_SIMD
void
MeshList::MakeShflTable() {
  std::fill(shfl_table[0], shfl_table[16], 0);
  for (int i = 0; i < 16; i++) {
    int tbl_id = i;
    int cnt = 0;
    for (int j = 0; j < 4; j++) {
      if (tbl_id & 0x1) {
        shfl_table[i][cnt++] = 2 * j;
        shfl_table[i][cnt++] = 2 * j + 1;
      }
      tbl_id >>= 1;
    }
  }
}
#endif
//----------------------------------------------------------------------
