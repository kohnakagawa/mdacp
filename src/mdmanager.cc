#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "device_info.h"
#endif
#include "communicator.h"
#include "mpistream.h"
#include "mdmanager.h"
#include "observer.h"
#include "stopwatch.h"
#include "cmdline.h"
#include "helper_macros.h"
//----------------------------------------------------------------------
MDManager::MDManager(int &argc, char ** &argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mout.SetRank(rank);
  mout << "# " << MDACP_VERSION << std::endl;

  cmdline::parser arg_parser;
  arg_parser.add<std::string>("in", 'i', "input file name", false, "input.cfg");
#ifdef USE_GPU
  const auto num_gpus_avail = get_number_of_devices();
  mout << "# " << num_gpus_avail << "GPUs are found." << std::endl;
  arg_parser.add<int>("num_gpus_per_node", 'g', "number of gpus per one node",
                      false, num_gpus_avail, cmdline::range(1, num_gpus_avail));
  arg_parser.add<int>("num_procs_per_gpu", 'p', "number of processes per one gpu",
                      true);
#endif
  arg_parser.parse_check(argc, argv);
  const std::string inputfile = arg_parser.get<std::string>("in");
  param.LoadFromFile(inputfile.c_str());

  num_threads = omp_get_max_threads();
  mout << "# " << num_procs << " MPI Process(es), " << num_threads
       << " OpenMP Thread(s), Total " << num_procs * num_threads << " Unit(s)" << std::endl;

#ifdef USE_GPU
  const auto ngpus_per_node  = arg_parser.get<int>("num_gpus_per_node");
  mout << "# Will use " << ngpus_per_node << "GPU(s) / node." << std::endl;
  for (int i = 0; i < ngpus_per_node; i++) device_query(i);
  const auto nprocs_per_gpu  = arg_parser.get<int>("num_procs_per_gpu");
  const auto gpu_id_global   = rank / nprocs_per_gpu;
  const auto gpu_id_local    = gpu_id_global % ngpus_per_node;
  const auto nprocs_per_node = nprocs_per_gpu * ngpus_per_node;
  if ((num_procs % nprocs_per_node) != 0) {
    show_error("# # of processes should be devidable by # of processes/node.");
    exit(1);
  }
  checkCudaErrors(cudaSetDevice(gpu_id_local));
#endif

  pinfo = new ParaInfo(num_procs, num_threads, param);
  int grid_size[D];
  pinfo->GetGridSize(grid_size);
  sinfo = new SimulationInfo(param, grid_size);
  int tid;
  MDUnit *mdp;
  std::vector <MDUnit *> v;
  #pragma omp parallel shared(v) private(tid,mdp)
  {
#ifdef USE_GPU
    checkCudaErrors(cudaSetDevice(gpu_id_local));
#endif
    tid = omp_get_thread_num();
    mdp = new MDUnit(tid + rank * num_threads, sinfo, pinfo);
    #pragma omp critical
    v.push_back(mdp);
  }
  mdv.resize(num_threads);
  for (unsigned int i = 0; i < v.size(); i++) {
    const int local_id = GetLocalID(v[i]->GetID());
    mdv[local_id] = v[i];
  }
  s_time = 0.0;

#ifdef USE_GPU
  checkCudaErrors(cudaSetDevice(gpu_id_local));
#endif
}
//----------------------------------------------------------------------
MDManager::~MDManager(void) {
  for (unsigned int i = 0; i < mdv.size(); i++) {
    delete mdv[i];
  }
  delete pinfo;
  delete sinfo;
  MPI_Finalize();
}
//----------------------------------------------------------------------
bool
MDManager::IsValid(void) {
  return Communicator::AllReduceBoolean(pinfo->IsValid() && param.IsValid());
}
//----------------------------------------------------------------------
bool
MDManager::IsMyUnit(int id) {
  return (rank == (id / num_threads));
}
//----------------------------------------------------------------------
int
MDManager::GetLocalID(int id) {
  return (id % num_threads);
}
//----------------------------------------------------------------------
void
MDManager::SetInitialVelocity(double v0) {
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->SetInitialVelocity(v0);
  }
}
//----------------------------------------------------------------------
void
MDManager::SaveAsCdviewSequential(void) {
  static int index = 0;
  char filename[256];
  sprintf(filename, "conf%04d.cd", index);
  index++;
  SaveAsCdview(filename);
}
//----------------------------------------------------------------------
void
MDManager::SaveAsCdview(const char *filename) {
  if (0 == GetRank()) {
    std::ofstream ofs(filename);
    ofs.close();
  }
  for (int r = 0; r < GetTotalProcs(); r++) {
    Communicator::Barrier();
    if (r == GetRank()) {
      std::ofstream ofs(filename, std::ios::app);
      for (int i = 0; i < num_threads; i++) {
        mdv[i]->SaveAsCdview(ofs);
      }
      ofs.close();
    }
  }
}
//----------------------------------------------------------------------
void
MDManager::SaveConfiguration(void) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->SaveConfiguration();
  }
}
//----------------------------------------------------------------------
void
MDManager::Calculate(void) {
  static StopWatch swAll(GetRank(), "all");
  static StopWatch swForce(GetRank(), "force");
  static StopWatch swComm(GetRank(), "comm");
  static StopWatch swPair(GetRank(), "pair");
  swAll.Start();
  if (IsPairListExpired()) {
    //mout << "# " << GetSimulationTime() << " # Expired!" << std::endl;
    swPair.Start();
    MakePairList();
    swPair.Stop();
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->UpdatePositionHalf();
  }
  swComm.Start();
  SendBorderParticles();
  swComm.Stop();
  if (sinfo->ControlTemperature) {
    if (sinfo->HeatbathType == HT_NOSEHOOVER) {
      CalculateNoseHoover();
    } else {
      CalculateLangevin();
    }
  } else {
    swForce.Start();
    CalculateForce();
    swForce.Stop();
  }
  s_time += sinfo->TimeStep;
  swAll.Stop();
}
//----------------------------------------------------------------------
#ifdef USE_GPU
#define GPU_CUDA_ENTER                                      \
  static StopWatch swForce_cpu(GetRank(), "force_cpu");     \
  static StopWatchCuda swForce_gpu(GetRank(), "force_gpu");	\
  static int profile_cnt = 0;                               \
  swForce_gpu.Start(); swForce_cpu.Start()
//----------------------------------------------------------------------
#define GPU_CUDA_EXIT                                         \
  swForce_cpu.Stop();                                         \
                                                              \
  checkCudaErrors(cudaDeviceSynchronize());                   \
  swForce_gpu.Record();                                       \
                                                              \
  const auto tgpu = swForce_gpu.GetSumOfLastElements();       \
  const auto tcpu = swForce_cpu.GetSumOfLastElements();       \
  tgpu_per_tcpu = tgpu / tcpu;                                \
  if (profile_cnt == 100) {                                   \
    mout << "gpu_time/cpu_time = " << tgpu_per_tcpu << "\n";	\
    profile_cnt = 0;                                          \
  }                                                           \
  profile_cnt++
//----------------------------------------------------------------------
#define INNER_LOOP_TEMPLATE_GPU(LOOP_BODY)      \
  mdv[i]->SendParticlesHostToDev();             \
  LOOP_BODY;                                    \
  mdv[i]->SendParticlesDevToHost()
//----------------------------------------------------------------------
#define GPU_TIMER_STOP swForce_gpu.Stop()
#define HOST_NAME CPU
//----------------------------------------------------------------------
#else
#define GPU_CUDA_ENTER
#define GPU_CUDA_EXIT
#define INNER_LOOP_TEMPLATE_GPU(LOOP_BODY)
#define GPU_TIMER_STOP
#define HOST_NAME
#endif
//----------------------------------------------------------------------
void
MDManager::CalculateForce(void) {
#undef LOOP_BODY_INNER
#define LOOP_BODY_INNER(DEVICE_T)                       \
  MDACP_CONCAT(mdv[i]->CalculateForce, DEVICE_T)()

  // calculate @ GPU
  GPU_CUDA_ENTER;
  for (int i = 0; i < num_threads; i++) {
    INNER_LOOP_TEMPLATE_GPU(LOOP_BODY_INNER(GPU));
  }
  GPU_TIMER_STOP;

  // calculate @ CPU
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    LOOP_BODY_INNER(HOST_NAME);
  }

  // sync CPU and GPU
  GPU_CUDA_EXIT;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->UpdatePositionHalf();
  }
}
//----------------------------------------------------------------------
void
MDManager::CalculateNoseHoover(void) {
#undef LOOP_BODY_INNER
#define LOOP_BODY_INNER(DEVICE_T)                     \
  MDACP_CONCAT(mdv[i]->HeatbathMomenta, DEVICE_T)();  \
  MDACP_CONCAT(mdv[i]->CalculateForce, DEVICE_T)();   \
  MDACP_CONCAT(mdv[i]->HeatbathMomenta, DEVICE_T)()

  double t = Temperature();
  for (int i = 0; i < num_threads; i++) { mdv[i]->HeatbathZeta(t); }

  // calculate @ GPU
  GPU_CUDA_ENTER;
  for (int i = 0; i < num_threads; i++) {
    INNER_LOOP_TEMPLATE_GPU(LOOP_BODY_INNER(GPU));
  }
  GPU_TIMER_STOP;

  // calculate @ CPU
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    LOOP_BODY_INNER(HOST_NAME);
  }

  // sync CPU and GPU
  GPU_CUDA_EXIT;

  t = Temperature();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->HeatbathZeta(t);
    mdv[i]->UpdatePositionHalf();
  }
}
//----------------------------------------------------------------------
void
MDManager::CalculateLangevin(void) {
#undef LOOP_BODY_INNER
#define LOOP_BODY_INNER(DEVICE_T)                   \
  MDACP_CONCAT(mdv[i]->CalculateForce, DEVICE_T)()

  // calculate @ GPU
  GPU_CUDA_ENTER;
  for (int i = 0; i < num_threads; i++) {
    INNER_LOOP_TEMPLATE_GPU(LOOP_BODY_INNER(GPU));
  }
  GPU_TIMER_STOP;

  // calculate @ CPU
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    LOOP_BODY_INNER(HOST_NAME);
  }

  // sync CPU and GPU
  GPU_CUDA_EXIT;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->Langevin();
  }
}
//----------------------------------------------------------------------
void
MDManager::SendParticlesSub(const int dir) {
  const int o_dir = OppositeDir[dir];
  debug_printf("dir = %s o_dir = %s\n", Direction::Name(dir), Direction::Name(o_dir));
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->MakeBufferForSendingParticle(dir);
  }
  std::vector<int> send_number;
  std::vector<int> recv_number;
  std::vector<ParticleInfo> send_buffer;
  std::vector<ParticleInfo> recv_buffer;
  for (int i = 0; i < num_threads; i++) {
    const int id = mdv[i]->GetID();
    const int id_dest = pinfo->GetNeighborID(id, dir);
    const int id_src = pinfo->GetNeighborID(id, o_dir);
    if (IsMyUnit(id_src)) {
      const int local_id_src  = GetLocalID(id_src);
      mdv[i]->ReceiveParticles(mdv[local_id_src]->send_buffer);
    }
    if (!IsMyUnit(id_dest)) {
      send_number.push_back(mdv[i]->send_buffer.size());
      send_buffer.insert(send_buffer.end(), mdv[i]->send_buffer.begin(), mdv[i]->send_buffer.end());
    }
  }
  const int dest_rank = pinfo->GetNeighborRank(rank, dir);
  const int src_rank = pinfo->GetNeighborRank(rank, o_dir);
  const int number = (int)send_number.size();
  Communicator::SendRecvVector(send_number, number, dest_rank, recv_number, number, src_rank);

  int send_sum = 0;
  int recv_sum = 0;
  for (int i = 0; i < number; i++) {
    send_sum += send_number[i];
    recv_sum += recv_number[i];
  }
  debug_printf("!!%03d:Send %d Recv %d\n", rank, send_sum, recv_sum);
  Communicator::SendRecvVector(send_buffer, send_sum, dest_rank, recv_buffer, recv_sum, src_rank);
  debug_printf("!!%03d:Sent %d Recved %d\n", rank, (int)send_buffer.size(), (int)recv_buffer.size());

  std::vector<ParticleInfo>::iterator it1 = recv_buffer.begin();
  std::vector<ParticleInfo>::iterator it2 = recv_buffer.begin();
  int index = 0;
  std::vector<ParticleInfo> temp_buffer;
  for (int i = 0; i < num_threads; i++) {
    const int id = mdv[i]->GetID();
    const int id_src = pinfo->GetNeighborID(id, o_dir);
    if (IsMyUnit(id_src)) {
      continue;
    }
    it1 = it2;
    it2 += recv_number[index];
    index++;
    temp_buffer.clear();
    temp_buffer.insert(temp_buffer.begin(), it1, it2);
    mdv[i]->ReceiveParticles(temp_buffer);
  }
}
//----------------------------------------------------------------------
void
MDManager::SendParticles(void) {
  for (int dir = 0; dir < MAX_DIR; dir++) {
    SendParticlesSub(dir);
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->AdjustPeriodicBoundary();
  }
}
//----------------------------------------------------------------------
void
MDManager::MakePairList(void) {
  SendParticles();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    const int pn = mdv[i]->GetParticleNumber();
    mdv[i]->SetTotalParticleNumber(pn);
  }
  for (int dir = 0; dir < MAX_DIR; dir++) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
      mdv[i]->FindBorderParticles(dir);
    }
    SendBorderParticlesSub(dir);
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->MakePairList();
  }

#ifdef USE_GPU
  AdjustCPUGPUWorkBalance();
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->SendNeighborInfoToGPUAsync();
#if !defined(GPU_ARCH_PASCAL)
    mdv[i]->TransposeSortedList();
#endif
  }
  checkCudaErrors(cudaDeviceSynchronize());
#endif
}
//----------------------------------------------------------------------
void
MDManager::SendBorderParticles(void) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    const int pn = mdv[i]->GetParticleNumber();
    mdv[i]->SetTotalParticleNumber(pn);
  }
  for (int dir = 0; dir < MAX_DIR; dir++) {
    SendBorderParticlesSub(dir);
  }
}
//----------------------------------------------------------------------
void
MDManager::SendBorderParticlesSub(const int dir) {
  const int o_dir = OppositeDir[dir];
  debug_printf("dir = %s o_dir = %s\n", Direction::Name(dir), Direction::Name(o_dir));
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->MakeBufferForBorderParticles(dir);
  }
  std::vector<int> send_number;
  std::vector<int> recv_number;
  std::vector<ParticleInfo> send_buffer;
  std::vector<ParticleInfo> recv_buffer;
  for (int i = 0; i < num_threads; i++) {
    const int id = mdv[i]->GetID();
    const int id_dest = pinfo->GetNeighborID(id, dir);
    const int id_src = pinfo->GetNeighborID(id, o_dir);
    if (IsMyUnit(id_src)) {
      const int local_id_src  = GetLocalID(id_src);
      mdv[i]->ReceiveBorderParticles(mdv[local_id_src]->send_buffer);
    }
    if (!IsMyUnit(id_dest)) {
      send_number.push_back(mdv[i]->send_buffer.size());
      send_buffer.insert(send_buffer.end(), mdv[i]->send_buffer.begin(), mdv[i]->send_buffer.end());
    }
  }
  const int dest_rank = pinfo->GetNeighborRank(rank, dir);
  const int src_rank = pinfo->GetNeighborRank(rank, o_dir);
  const int number = (int)send_number.size();
  Communicator::SendRecvVector(send_number, number, dest_rank, recv_number, number, src_rank);

  int send_sum = 0;
  int recv_sum = 0;
  debug_printf("%03d: Send %d Blocks\n", rank, (int)send_number.size());
  for (int i = 0; i < number; i++) {
    send_sum += send_number[i];
    recv_sum += recv_number[i];
  }
  debug_printf("!!%03d:Send %d Recv %d\n", rank, send_sum, recv_sum);
  Communicator::SendRecvVector(send_buffer, send_sum, dest_rank, recv_buffer, recv_sum, src_rank);
  debug_printf("!!%03d:Sent %d Recved %d\n", rank, (int)send_buffer.size(), (int)recv_buffer.size());

  std::vector<ParticleInfo>::iterator it1 = recv_buffer.begin();
  std::vector<ParticleInfo>::iterator it2 = recv_buffer.begin();
  int index = 0;
  std::vector<ParticleInfo> temp_buffer;
  for (int i = 0; i < num_threads; i++) {
    const int id = mdv[i]->GetID();
    const int id_src = pinfo->GetNeighborID(id, o_dir);
    if (IsMyUnit(id_src)) {
      continue;
    }
    it1 = it2;
    it2 += recv_number[index];
    temp_buffer.clear();
    temp_buffer.insert(temp_buffer.begin(), it1, it2);
    mdv[i]->ReceiveBorderParticles(temp_buffer);
    index++;
  }
}
//----------------------------------------------------------------------
void
MDManager::ShowSystemInformation(void) {
  const unsigned long int pn = GetTotalParticleNumber();
  sinfo->ShowAll(pn);
}
//----------------------------------------------------------------------
// For Observation
//----------------------------------------------------------------------
unsigned long int
MDManager::GetTotalParticleNumber(void) {
  unsigned long int pn = 0;
  for (int i = 0; i < num_threads; i++) {
    pn += static_cast<unsigned long int>(mdv[i]->GetParticleNumber());
  }
  pn = Communicator::AllReduceUnsignedLongInteger(pn);
  return pn;
}
//----------------------------------------------------------------------
bool
MDManager::IsPairListExpired(void) {
  bool expired = false;
  #pragma omp parallel for reduction(|:expired)
  for (int i = 0; i < num_threads; i++) {
    expired |= mdv[i]->IsPairListExpired();
  }
  expired = Communicator::AllReduceBoolean(expired);
  return expired;
}
//----------------------------------------------------------------------
double
MDManager::ObserveDouble(DoubleObserver *obs) {
  double e = 0.0;
  #pragma omp parallel for reduction(+:e)
  for (int i = 0; i < num_threads; i++) {
    e += mdv[i]->ObserveDouble(obs);
  }
  return Communicator::AllReduceDouble(e);
}
//----------------------------------------------------------------------
double
MDManager::Temperature(void) {
  return KineticEnergy() / 1.5;
}
//----------------------------------------------------------------------
double
MDManager::ConfigurationTemperature(void) {
  VirialObserver obs;
  const double pn = static_cast<double>(GetTotalParticleNumber());
  const double phi = ObserveDouble(&obs);
  return phi / pn;
}
//----------------------------------------------------------------------
double
MDManager::KineticEnergy(void) {
  KineticEnergyObserver obs;
  const double pn = static_cast<double>(GetTotalParticleNumber());
  return ObserveDouble(&obs) / pn;
}
//----------------------------------------------------------------------
double
MDManager::PotentialEnergy(void) {
  PotentialEnergyObserver obs(sinfo);
  const double pn = static_cast<double>(GetTotalParticleNumber());
  return ObserveDouble(&obs) / pn;
}
//----------------------------------------------------------------------
double
MDManager::TotalEnergy(void) {
  return KineticEnergy() + PotentialEnergy();
}
//----------------------------------------------------------------------
double
MDManager::Pressure(void) {
  VirialObserver obs;
  const double pn = static_cast<double>(GetTotalParticleNumber());
  const double phi = ObserveDouble(&obs) / pn;
  const double T = Temperature();
  const double V = sinfo->L[X] * sinfo->L[Y] * sinfo->L[Z];
  return (T - phi) * pn / V;
}
//----------------------------------------------------------------------
void
MDManager::ChangeScale(double alpha) {
  MakePairList();
  sinfo->L[X] = sinfo->L[X] * alpha;
  sinfo->L[Y] = sinfo->L[Y] * alpha;
  sinfo->L[Z] = sinfo->L[Z] * alpha;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->ChangeScale(alpha);
  }
  MakePairList();
  mout << "# Change Scale " << alpha << std::endl;
}
//----------------------------------------------------------------------
void
MDManager::ExecuteAll(Executor *ex) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->Execute(ex);
  }
}
//----------------------------------------------------------------------
#ifdef USE_GPU
void
MDManager::AdjustCPUGPUWorkBalance(void) {
  static double work_balance = 0.7;
  work_balance /= (1.0 - tgpu_per_tcpu) * work_balance + tgpu_per_tcpu;

  if (work_balance <= 0.0) work_balance = 0.0;
  if (work_balance >= 1.0) work_balance = 1.0;
  for (int i = 0; i < num_threads; i++) {
    mdv[i]->UpdateParticleNumberGPU(work_balance);
  }
}
#endif
//----------------------------------------------------------------------
