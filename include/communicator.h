//----------------------------------------------------------------------
// MPI Communication Class
//----------------------------------------------------------------------

#ifndef communicator_h
#define communicator_h
#include <mpi.h>
#include <vector>
#include "mdconfig.h"

namespace Communicator {
void Barrier(void);
void SendRecvInteger(int &sendnumber, int dest_rank, int &recvnumber, int src_rank);
void SendRecvDouble(void *sendbuf, int sendnumber, int dest_rank, void *recvbuf, int recvnumber, int src_rank);
template <class C>void SendRecvVector(C &send_buffer, int send_number, int dest_rank, C &recv_buffer, int recv_number, int src_rank) {
  MPI_Status st;
  recv_buffer.resize(recv_number);
  MPI_Sendrecv(&send_buffer[0], send_number * sizeof(send_buffer[0]), MPI_BYTE, dest_rank, 0, &recv_buffer[0], recv_number * sizeof(recv_buffer[0]), MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, &st);
}

double GetTime(void);
double FindMaxDouble(double value);
bool AllReduceBoolean(bool flag);
double AllReduceDouble(double value);
void AllReduceDoubleBuffer(double *sendbuf, int size, double *recvbuf);
int AllReduceInteger(int value);
unsigned long int AllReduceUnsignedLongInteger(unsigned long int value);
void AllGatherInteger(int *sendbuf, int number, int *recvbuf);
void AllGatherDouble(double *sendbuf, int number, double *recvbuf);
void AllGatherIntegerVector(std::vector<int> &send_buffer, std::vector<int> &recv_buffer, int num_procs);
void AllGatherUCharVector(std::vector<unsigned char> &send_buffer, std::vector<unsigned char> &recv_buffer, int num_procs);
void GatherIntegerVector(std::vector<int> &send_buffer, std::vector<int> &recv_buffer, int num_procs, int root);
void GatherUCharVector(std::vector<unsigned char> &send_buffer, std::vector<unsigned char> &recv_buffer, int num_procs, int root);
void BroadcastInteger(int &value, int root);

};
#endif
