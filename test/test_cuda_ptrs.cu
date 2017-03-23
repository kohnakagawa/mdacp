#include "cuda_ptr.h"
#include "cuda_ptr2d.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>

#include <thrust/sequence.h>

#define CHECK_EQ(a, b)                                                  \
  do {                                                                  \
    if ((a) != (b)) {                                                   \
      printf("test fails at %s %s\n", __FILE__, __LINE__);              \
      assert(0);                                                        \
    }                                                                   \
  } while (0)

__global__ void check_device_value(double* val,
                                   const int beg,
                                   const int count,
                                   const int num) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num) {
    const auto end = beg + count;
    if ((tid >= beg) && (tid < end)) {
      CHECK_EQ(val[tid], double(tid));
    } else {
      CHECK_EQ(val[tid], -1.0);
    }
  }
}

int main() {
  constexpr int tb_size = 128;
  constexpr int gr_size = 1000;
  constexpr int size = tb_size * gr_size;
  constexpr int dim = 4;

  // allocate and fill -1
  CudaPtr<double> cptr(size, -1);
  CudaPtr2D<double, size / dim, dim> cptr2d;
  thrust::fill(cptr2d.GetThrustPtr(),
               cptr2d.GetThrustPtr() + cptr2d.size(),
               -1);
  std::fill(cptr2d.GetHostPtr1D(),
            cptr2d.GetHostPtr1D() + cptr2d.size(),
            -1);

  // size() function test
  CHECK_EQ(cptr.size(), size);
  CHECK_EQ(cptr2d.size(), size);

  std::iota(cptr.GetHostPtr(),
            cptr.GetHostPtr() + size,
            0);
  std::iota(cptr2d.GetHostPtr1D(),
            cptr2d.GetHostPtr1D() + size,
            0);

  // partial copy test Host -> Device
  const int beg = 10;
  const int cnt = 100;
  cptr.Host2Dev(beg, cnt);
  cptr2d.Host2Dev(beg, cnt);

  check_device_value<<<gr_size, tb_size>>>(cptr.GetDevPtr(), beg, cnt, size);
  checkCudaErrors(cudaDeviceSynchronize());

  check_device_value<<<gr_size, tb_size>>>(cptr2d.GetDevPtr(),
                                           beg * cptr2d.ny(),
                                           cnt * cptr2d.ny(),
                                           size);
  checkCudaErrors(cudaDeviceSynchronize());

  // partial copy test Device -> Host
  cptr.Dev2Host(0, beg);
  cptr2d.Dev2Host(0, beg);

  for (int i = 0; i < beg; i++) {
    CHECK_EQ(cptr[i], -1);
  }
  double (*cptr2d_h)[dim] = cptr2d.GetHostPtr();
  for (int i = 0; i < beg; i++) {
    for (int j = 0; j < dim; j++) {
      CHECK_EQ(cptr2d_h[i][j], -1);
    }
  }
}
