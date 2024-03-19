#ifndef F_COMMUNICATER_H
#define F_COMMUNICATER_H
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

using namespace std;

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


void g_init();
int g_rank();
int g_size();
void g_barriar();

class Communicator {
public:
    Communicator(int nstreams/*=1*/);
    ~Communicator();

    void _init();
    void destroy();
    void reload();

	int reduce(torch::Tensor tensor, int root);
	int bcast(torch::Tensor tensor, int root);

	void reduceScatter(torch::Tensor send_tensor, torch::Tensor recv_tensor);
	void allGather(torch::Tensor send_tensor, torch::Tensor recv_tensor);
	void allReduce(torch::Tensor tensor);
	void allReduceRB(torch::Tensor tensor);
	void allReduceRSAG(torch::Tensor tensor);

	void multiBcast(vector<torch::Tensor> &tensor_list, vector<torch::Tensor> &output_list, const std::function<void(torch::Tensor, torch::Tensor)> &op);

    void sendrecv(torch::Tensor send_tensor, torch::Tensor recv_tensor, int peer);

    void barrier();
    void synchronize();
    void syncStream(int handler);

    int getNumOfFreeStreams();
    void _extendComms(int n_comms);

private:
    //ncclUniqueId m_nccl_id;
    //ncclComm_t m_nccl_comm;
	//cudaStream_t m_stream;
    ncclUniqueId* m_nccl_ids;
    ncclComm_t* m_nccl_comms;
	cudaStream_t* m_streams;
    std::vector<at::cuda::CUDAStream> m_torchstreams;
    cudaEvent_t *m_events;
    bool m_destroyed;
    bool* m_sync_flags;
    int m_rank;
    int m_size;
    int m_current_comm;
    int m_num_comms;
};

#endif //F_COMMUNICATER_H
