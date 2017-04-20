#ifndef stopwatch_h
#define stopwatch_h
//----------------------------------------------------------------------
#include <vector>
#include <fstream>
#include <mpi.h>
#ifdef USE_GPU
#include <helper_cuda.h>
#include <cuda_runtime.h>
#endif
//----------------------------------------------------------------------
struct TimerCPUImpl {
  typedef double Dtype;

  static void Initialize(...) {
    // dummy
  }
  static void Finalize(...) {
    // dummy
  }

  static void Start(Dtype& start) {
    start = Communicator::GetTime();
  }
  static void Stop(Dtype& stop) {
    stop = Communicator::GetTime();
  }

  static void RecordNow(const Dtype& start,
                        const Dtype& stop,
                        std::vector<double>& data) {
    data.push_back(stop - start);
  }
  static void Record(const Dtype& start,
                     const Dtype& stop,
                     std::vector<double>& data) {
    // dummy
  }
};
//----------------------------------------------------------------------
#ifdef USE_GPU
struct TimerGPUImpl {
  typedef cudaEvent_t Dtype;

  static void Initialize(Dtype& start, Dtype& stop) {
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
  }
  static void Finalize(Dtype& start, Dtype& stop) {
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
  }

  static void Start(Dtype& start) {
    checkCudaErrors(cudaEventRecord(start));
  }
  static void Stop(Dtype& stop) {
    checkCudaErrors(cudaEventRecord(stop));
  }

  static void RecordNow(const Dtype& start,
                        const Dtype& stop,
                        std::vector<double>& data) {
    // dummy
  }
  static void Record(const Dtype& start,
                     const Dtype& stop,
                     std::vector<double>& data) {
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsed_time = 0.0;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    data.push_back(elapsed_time * 1.0e-3);
  }
};
#endif
//----------------------------------------------------------------------
template <class Impl>
class Timer {
 private:
  typename Impl::Dtype start, stop;
  const char *basename;
  int id;
  std::vector<double> data;
 public:
  Timer(int rank, const char* bname) {
    basename = bname;
    id = rank;
    Impl::Initialize(start, stop);
  };
  ~Timer(void) {
    if (id == 0)SaveToFile();
    // SaveToFile();
    Impl::Finalize(start, stop);
  }
  void Start(void) {
    Impl::Start(start);
  };
  void Stop(void) {
    Impl::Stop(stop);
    Impl::RecordNow(start, stop, data);
  };
  void Record(void) {
    Impl::Record(start, stop, data);
  }

  void SaveToFile(void) {
    char filename[256];
    sprintf(filename, "%s%05d.dat", basename, id);
    std::ofstream ofs(filename);
    // ofs.write((const char *)&data[0], sizeof(double)*data.size());
    for (auto d : data) ofs << d << "\n";
  };

  double GetBackData(void) const {
    return data.back();
  }
};
//----------------------------------------------------------------------
typedef Timer<TimerCPUImpl> StopWatch;
#ifdef USE_GPU
typedef Timer<TimerGPUImpl> StopWatchCuda;
#endif
//----------------------------------------------------------------------
#endif
