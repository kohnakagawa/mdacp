//----------------------------------------------------------------------
#ifndef cuda_ptr_h
#define cuda_ptr_h
//----------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <helper_cuda.h>
//----------------------------------------------------------------------
template <typename T>
class CudaPtr {
  T* dev_ptr_  = nullptr;
  T* host_ptr_ = nullptr;
  int size_ = -1;
  thrust::device_ptr<T> thrust_ptr_;

  void Deallocate(void) {
    checkCudaErrors(cudaFree(dev_ptr_));
    checkCudaErrors(cudaFreeHost(host_ptr_));
  }
public:
  CudaPtr(void) {}
  CudaPtr(const int s) {
    Allocate(s);
  }
  CudaPtr(const int s, const T val) {
    Allocate(s);
    SetVal(val);
  }
  ~CudaPtr(void) {
    Deallocate();
  }

  // disable copy constructor
  const CudaPtr& operator = (const CudaPtr& obj) = delete;
  CudaPtr(const CudaPtr& obj) = delete;

  CudaPtr& operator = (CudaPtr&& obj) noexcept {
    this->dev_ptr_ = obj.dev_ptr_;
    this->host_ptr_ = obj.host_ptr_;

    obj.dev_ptr_ = nullptr;
    obj.host_ptr_ = nullptr;

    return *this;
  }
  CudaPtr(CudaPtr&& obj) noexcept {
    *this = std::move(obj);
  }

  int size(void) const {
    return size_;
  }

  void Allocate(const int s) {
    size_ = s;
    checkCudaErrors(cudaMalloc((void**)&dev_ptr_,
                               size_ * sizeof(T)));
    checkCudaErrors(cudaMallocHost((void**)&host_ptr_,
                                   size_ * sizeof(T)));
    thrust_ptr_ = thrust::device_pointer_cast(dev_ptr_);
  }

  void Host2Dev(const int beg,
                const int count) {
    checkCudaErrors(cudaMemcpy(dev_ptr_ + beg,
                               host_ptr_ + beg,
                               count * sizeof(T),
                               cudaMemcpyHostToDevice));
  }
  void Host2Dev(void) {this->Host2Dev(0, size_);}
  void Host2DevAsync(const int beg,
                     const int count,
                     cudaStream_t strm = 0) {
    checkCudaErrors(cudaMemcpyAsync(dev_ptr_ + beg,
                                    host_ptr_ + beg,
                                    count * sizeof(T),
                                    cudaMemcpyHostToDevice,
                                    strm));
  }
  void Host2DevAsync(cudaStream_t strm = 0) {
    Host2DevAsync(0, size_, strm);
  }

  void Dev2Host(const int beg,  const int count) {
    checkCudaErrors(cudaMemcpy(host_ptr_ + beg,
                               dev_ptr_ + beg,
                               count * sizeof(T),
                               cudaMemcpyDeviceToHost));
  }
  void Dev2Host(void) {this->Dev2Host(0, size_);}
  void Dev2HostAsync(const int beg,
                     const int count,
                     cudaStream_t strm = 0) {
    checkCudaErrors(cudaMemcpyAsync(host_ptr_ + beg,
                                    dev_ptr_ + beg,
                                    count * sizeof(T),
                                    cudaMemcpyDeviceToHost,
                                    strm));
  }
  void Dev2HostAsync(cudaStream_t strm = 0) {
    Dev2HostAsync(0, size_, strm);
  }

  void SetVal(const T val) {
    std::fill(host_ptr_, host_ptr_ + size_, val);
    thrust::fill(thrust_ptr_, thrust_ptr_ + size_, val);
  }
  void SetVal(const int beg,
              const int count,
              const T val){
    T* end_ptr = host_ptr_ + beg + count;
    std::fill(host_ptr_ + beg, end_ptr, val);
    thrust::device_ptr<T> beg_ptr = thrust_ptr_ + beg;
    thrust::fill(beg_ptr, beg_ptr + count, val);
  }

  const T& operator [] (const int i) const {
    return host_ptr_[i];
  }

  T& operator [] (const int i) {
    return host_ptr_[i];
  }

  const T* GetHostPtr(void) const {
    return host_ptr_;
  }
  T* GetHostPtr(void) {
    return host_ptr_;
  }

  const T* GetDevPtr(void) const {
    return dev_ptr_;
  }
  T* GetDevPtr(void) {
    return dev_ptr_;
  }

  const thrust::device_ptr<T>& GetThrustPtr(void) const {
    return thrust_ptr_;
  }
  thrust::device_ptr<T>& GetThrustPtr(void) {
    return thrust_ptr_;
  }
};
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
