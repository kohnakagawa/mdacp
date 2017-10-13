#include "device_info.h"
#include "mpistream.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
//----------------------------------------------------------------------
static void show_driver_ver(void);
static void show_runtime_ver(void);
//----------------------------------------------------------------------
void
device_query_all(void) {
  mout << "\n";
  mout << "******** GPU Information ********\n";
  show_driver_ver();
  show_runtime_ver();
  int num_dev = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_dev));
  for (int i = 0; i < num_dev; i++) device_query(i);
}
//----------------------------------------------------------------------
static void
show_driver_ver(void) {
  int driver_ver = 0;
  checkCudaErrors(cudaDriverGetVersion(&driver_ver));
  mout << "CUDA Driver Version, "
       << driver_ver / 1000 << "." << (driver_ver % 100) / 10 << "\n";
}
//----------------------------------------------------------------------
static void
show_runtime_ver(void) {
  int runtime_ver;
  checkCudaErrors(cudaRuntimeGetVersion(&runtime_ver));
  mout << "CUDA Runtime Version, "
       << runtime_ver / 1000 << "." << (runtime_ver % 100) / 10 << "\n";
}
//----------------------------------------------------------------------
void
device_query(const int dev_id) {
  cudaDeviceProp dev_prop;
  checkCudaErrors(cudaGetDeviceProperties(&dev_prop, dev_id));
  const float peak_mem_bw = (float) dev_prop.memoryClockRate * 1e3 * 2 * dev_prop.memoryBusWidth / 8;

  mout << "GPU - id " << dev_id << "\n";
  mout << "GPU - name, " << dev_prop.name << "\n";
  mout << "GPU - # SMs, " << dev_prop.multiProcessorCount << "\n";
  mout << "GPU - core freq (GHz), " << dev_prop.clockRate * 1e-6 << "\n";
  mout << "GPU - L2 size (MiB), " << (float) dev_prop.l2CacheSize / 1024 / 1024 << "\n";
  mout << "GPU - memory size (GiB), "
       << (float) dev_prop.totalGlobalMem / 1024 / 1024 / 1024 << "\n";
  mout << "GPU - memory bus width (bit), " << dev_prop.memoryBusWidth << "\n";
  mout << "GPU - memory bus freq (GHz), "
       << (float) dev_prop.memoryClockRate * 1e-6 << "\n";
  mout << "GPU - theoretical peak memory bandwidth (GB/s), "
       << peak_mem_bw * 1e-9 << "\n";
  const auto ecc_enabled = (dev_prop.ECCEnabled) ? "On\n" : "Off\n";
  mout << "GPU - ECC, " << ecc_enabled;
  mout << "\n";
}
//----------------------------------------------------------------------
int
get_number_of_devices(void) {
  int dev_cnt = 0;
  checkCudaErrors(cudaGetDeviceCount(&dev_cnt));
  return dev_cnt;
}
//----------------------------------------------------------------------
