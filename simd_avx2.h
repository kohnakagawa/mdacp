//----------------------------------------------------------------------
#ifndef simd_avx2_h
#define simd_avx2_h
#include <immintrin.h>
//----------------------------------------------------------------------
typedef double v4df __attribute__((vector_size(32)));
typedef int64_t v4di __attribute__((vector_size(32)));
typedef int32_t v8si __attribute__((vector_size(32)));
//----------------------------------------------------------------------
static inline void
put(v4df r) {
  double *a = (double*)(&r);
  printf("%.10f %.10f %.10f %.10f\n", a[0], a[1], a[2], a[3]);
}
//----------------------------------------------------------------------
static inline void
transpose_4x4(const v4df& va,
              const v4df& vb,
              const v4df& vc,
              const v4df& vd,
              v4df& vx,
              v4df& vy,
              v4df& vz) {
  v4df tmp0 = _mm256_unpacklo_pd(va, vb);
  v4df tmp1 = _mm256_unpackhi_pd(va, vb);
  v4df tmp2 = _mm256_unpacklo_pd(vc, vd);
  v4df tmp3 = _mm256_unpackhi_pd(vc, vd);
  vx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
  vy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
  vz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
}
//----------------------------------------------------------------------
static inline void
transpose_4x4(v4df& va,
              v4df& vb,
              v4df& vc,
              v4df& vd) {
  v4df tmp0 = _mm256_unpacklo_pd(va, vb);
  v4df tmp1 = _mm256_unpackhi_pd(va, vb);
  v4df tmp2 = _mm256_unpacklo_pd(vc, vd);
  v4df tmp3 = _mm256_unpackhi_pd(vc, vd);

  va = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
  vb = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
  vc = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
  vd = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);
}

//----------------------------------------------------------------------
#endif
