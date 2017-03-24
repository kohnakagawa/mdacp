//----------------------------------------------------------------------
#ifndef cuda_ptr2d_h
#define cuda_ptr2d_h
//----------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <helper_cuda.h>
//----------------------------------------------------------------------
template <typename T, std::size_t Nx, std::size_t Ny>
class CudaPtr2D {
  __attribute__((aligned(64))) T host_ptr_[Nx][Ny];
  T* dev_ptr_ = nullptr;
  thrust::device_ptr<T> thrust_ptr_;

  void Allocate(void) {
    checkCudaErrors(cudaHostRegister((void**)&host_ptr_[0],
                                     Nx * Ny * sizeof(T),
                                     cudaHostRegisterPortable));
    checkCudaErrors(cudaMalloc((void**)&dev_ptr_,
                               Nx * Ny * sizeof(T)));
    thrust_ptr_ = thrust::device_pointer_cast(dev_ptr_);
  }
  void Deallocate(void) {
    checkCudaErrors(cudaHostUnregister((void**)&host_ptr_[0]));
    checkCudaErrors(cudaFree(dev_ptr_));
  }

public:
  CudaPtr2D(void) {
    Allocate();
  }
  ~CudaPtr2D(void) {
    Deallocate();
  }

  // disable copy constructor
  const CudaPtr2D& operator = (const CudaPtr2D& obj) = delete;
  CudaPtr2D(const CudaPtr2D& obj) = delete;

  // disable move constructor
  CudaPtr2D& operator = (CudaPtr2D&& obj) = delete;
  CudaPtr2D(CudaPtr2D&& obj) = delete;

  // TODO: remove API duplicates
  // blocking API
  // NOTE: count means # of x elements.
  void Host2Dev(const int beg,
                const int count) {
    const auto offset = beg * Ny;
    const auto count_elem = count * Ny;
    assert((offset + count_elem) < size()); // NOTE: range check

    checkCudaErrors(cudaMemcpy(dev_ptr_ + offset,
                               host_ptr_[0] + offset,
                               count_elem * sizeof(T),
                               cudaMemcpyHostToDevice));
  }
  void Host2Dev(void) {
    Host2Dev(0, Nx);
  }

  // non-blocking API
  // NOTE: count means # of x elements.
  void Host2DevAsync(const int beg,
                     const int count,
                     cudaStream_t strm = 0) {
    const auto offset = beg * Ny;
    const auto count_elem = count * Ny;
    assert((offset + count_elem) < size()); // NOTE: range check

    checkCudaErrors(cudaMemcpyAsync(dev_ptr_ + offset,
                                    host_ptr_[0] + offset,
                                    count_elem * sizeof(T),
                                    cudaMemcpyHostToDevice,
                                    strm));
  }
  void Host2DevAsync(cudaStream_t strm = 0) {
    Host2DevAsync(0, Nx, strm);
  }

  // blocking API
  // NOTE: count means # of x elements
  void Dev2Host(const int beg,
                const int count) {
    const auto offset = beg * Ny;
    const auto count_elem = count * Ny;
    assert((offset + count_elem) < size()); // NOTE: range check

    checkCudaErrors(cudaMemcpy(host_ptr_[0] + offset,
                               dev_ptr_ + offset,
                               count_elem * sizeof(T),
                               cudaMemcpyDeviceToHost));
  }
  void Dev2Host(void) {
    Dev2Host(0, Nx);
  }

  // non-blocking API
  // NOTE: count means # of x elements
  void Dev2HostAsync(const int beg,
                     const int count,
                     cudaStream_t strm = 0) {
    const auto offset = beg * Ny;
    const auto count_elem = count * Ny;
    assert((offset + count_elem) < size()); // NOTE: range check

    checkCudaErrors(cudaMemcpyAsync(host_ptr_[0] + offset,
                                    dev_ptr_ + offset,
                                    count_elem * sizeof(T),
                                    cudaMemcpyDeviceToHost,
                                    strm));
  }
  void Dev2HostAsync(cudaStream_t strm = 0) {
    Dev2HostAsync(0, Nx, strm);
  }

  T* GetHostPtr1D() {return host_ptr_[0];}
  const T* GetHostPtr1D() const {return host_ptr_[0];}

  T (*GetHostPtr())[Ny] {return host_ptr_;}
  const T (*GetHostPtr() const)[Ny] {return host_ptr_;}

  T* GetDevPtr() {return dev_ptr_;}
  const T* GetDevPtr() const {return dev_ptr_;}

  const thrust::device_ptr<T>& GetThrustPtr() const {return thrust_ptr_;}
  thrust::device_ptr<T>& GetThrustPtr() {return thrust_ptr_;}

  constexpr int nx() const {return Nx;}
  constexpr int ny() const {return Ny;}
  constexpr int size() const {return Nx * Ny;}
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
