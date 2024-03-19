#include <torch/extension.h>
#include "communicator.h"


void g_init() {
    MPICHECK(MPI_Init(NULL, NULL));
}

int g_rank() {
    int r;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &r));
    return r;
}

int g_size() {
    int s;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &s));
    return s;
}

void g_barriar() {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

Communicator::Communicator(int nstreams=1) {
    m_current_comm = 0;
    m_num_comms = nstreams;
    m_rank = g_rank();
    m_size = g_size();
    m_destroyed = true;
    _init();
}

Communicator::~Communicator() {
    destroy();
    int finalized;
    MPICHECK(MPI_Finalized(&finalized));
    if (!finalized) {
        MPICHECK(MPI_Finalize());
    }
}

void Communicator::_init() {
    if (!m_destroyed) {
        return;
    }
    int nstreams = m_num_comms;
    m_nccl_ids = new ncclUniqueId[nstreams];
    m_streams = new cudaStream_t[nstreams];
    m_events = new cudaEvent_t[nstreams];
    m_nccl_comms = new ncclComm_t[nstreams];
    m_sync_flags = new bool[nstreams];
    for (int i = 0; i < m_num_comms; i++) {
	    if (m_rank == 0) ncclGetUniqueId(&m_nccl_ids[i]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i], sizeof(m_nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
    }
    for (int i = 0; i < m_num_comms; i++) {
	    //CUDACHECK(cudaStreamCreate(&m_streams[i]));
        at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
        m_streams[i] = myStream.stream();
        m_torchstreams.push_back(myStream);
	    CUDACHECK(cudaEventCreate(&m_events[i]));
	    NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i], m_size, m_nccl_ids[i], m_rank));
    }
    m_destroyed = false;
}

void Communicator::destroy() {
    for (int i = 0; i < m_num_comms; i++) {
	    NCCLCHECK(ncclCommDestroy(m_nccl_comms[i]));
        //CUDACHECK(cudaStreamDestroy(m_streams[i]));
        CUDACHECK(cudaEventDestroy(m_events[i]));
    }
    m_destroyed = true;
    delete m_streams;
    delete m_events;
    delete m_nccl_comms;
    delete m_sync_flags;
}

void Communicator::reload() {
    _init();
}

void Communicator::_extendComms(int n_comms) {
    if (m_num_comms >= n_comms) return;
    for (int i = 0; i < n_comms-m_num_comms; i++) {
	    if (m_rank == 0) ncclGetUniqueId(&m_nccl_ids[i+m_num_comms]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i+m_num_comms], sizeof(m_nccl_ids[i+m_num_comms]), MPI_BYTE, 0, MPI_COMM_WORLD));

	    CUDACHECK(cudaStreamCreate(&m_streams[i+m_num_comms]));
	    NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i+m_num_comms], m_size, m_nccl_ids[i+m_num_comms], m_rank));
    }
    m_num_comms = n_comms;
}

void Communicator::barrier() {

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

}

void Communicator::synchronize() {
    //CUDACHECK(cudaStreamSynchronize(m_stream));
    for (int i = 0; i < m_num_comms; i++) {
        CUDACHECK(cudaStreamSynchronize(m_streams[i]));
        //CUDACHECK(cudaStreamWaitEvent(m_streams[i], m_events[i], 0));
        m_sync_flags[i] = true;
    }
}
void Communicator::syncStream(int handler) {
    if (handler < m_num_comms && !m_sync_flags[handler]) {
        CUDACHECK(cudaStreamSynchronize(m_streams[handler]));
        m_sync_flags[handler] = true;
    }
}

int Communicator::getNumOfFreeStreams() {
    int num_free_streams = 0;
    for (int i = 0; i < m_num_comms; i++) {
        if (m_sync_flags[i]) num_free_streams++;
        else {
            cudaError_t status = cudaStreamQuery(m_streams[i]);
            if (status == cudaSuccess) num_free_streams++;
        }
    }
    return num_free_streams;
}

int Communicator::reduce(torch::Tensor tensor, int root) {
    int current_comm = m_current_comm;
    NCCLCHECK(ncclReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, root, m_nccl_comms[current_comm], m_streams[current_comm]));
    //CUDACHECK(cudaEventRecord(m_events[current_comm], m_streams[current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return current_comm;
}

int Communicator::bcast(torch::Tensor tensor, int root) {
    ncclDataType_t nccl_type;
    int current_comm = m_current_comm;
    if (torch::kFloat32 == tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclBroadcast(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), nccl_type, root, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclBroadcast(tensor.data_ptr<long>(), tensor.data_ptr<long>(), tensor.numel(), nccl_type, root, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return current_comm;
}

void Communicator::reduceScatter(torch::Tensor send_tensor, torch::Tensor recv_tensor) {
    ncclDataType_t nccl_type;
    if (torch::kFloat32 == send_tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclReduceScatter(send_tensor.data_ptr<float>(), recv_tensor.data_ptr<float>(), recv_tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclReduceScatter(send_tensor.data_ptr<long>(), recv_tensor.data_ptr<long>(), recv_tensor.numel(), ncclInt64, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::allGather(torch::Tensor send_tensor, torch::Tensor recv_tensor) {
    ncclDataType_t nccl_type;
    if (torch::kFloat32 == send_tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclAllGather(send_tensor.data_ptr<float>(), recv_tensor.data_ptr<float>(), send_tensor.numel(), ncclFloat, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclAllGather(send_tensor.data_ptr<long>(), recv_tensor.data_ptr<long>(), send_tensor.numel(), ncclInt64, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::allReduceRB(torch::Tensor tensor) {
    int current_comm = m_current_comm;
    int root = 0;
    ncclDataType_t nccl_type = ncclFloat;
    NCCLCHECK(ncclReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, root, m_nccl_comms[current_comm], m_streams[current_comm]));
    NCCLCHECK(ncclBroadcast(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), nccl_type, root, m_nccl_comms[current_comm], m_streams[current_comm]));

    //CUDACHECK(cudaEventRecord(m_events[current_comm], m_streams[current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
}
	
void Communicator::allReduceRSAG(torch::Tensor tensor) {
    int current_comm = m_current_comm;
    int n = tensor.numel();
    int p  = m_size;
    if (n < p) {
        NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else {
        int n_per_worker = (n+p-1)/p;
        int padded_n = n_per_worker * p;

        auto options_float =
            torch::TensorOptions()
            .dtype(tensor.dtype())
            .device(tensor.device().type())
            .requires_grad(false);

        at::cuda::CUDAStream cuStream = m_torchstreams[current_comm];
        {
            at::cuda::CUDAStreamGuard guard(cuStream);
            torch::Tensor temp_result = torch::zeros(n_per_worker, options_float); 
            if (n < padded_n) {
                // should be padded to the multiple of p
                torch::Tensor padded_tensor = torch::zeros(padded_n, options_float); 
                padded_tensor.narrow(0, 0, n).copy_(tensor);
                NCCLCHECK(ncclReduceScatter(padded_tensor.data_ptr<float>(), temp_result.data_ptr<float>(), temp_result.numel(), ncclFloat, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
                NCCLCHECK(ncclAllGather(temp_result.data_ptr<float>(), padded_tensor.data_ptr<float>(), temp_result.numel(), ncclFloat, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
                tensor.copy_(padded_tensor.narrow(0, 0, n));
            } else {
                NCCLCHECK(ncclReduceScatter(tensor.data_ptr<float>(), temp_result.data_ptr<float>(), temp_result.numel(), ncclFloat, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
                NCCLCHECK(ncclAllGather(temp_result.data_ptr<float>(), tensor.data_ptr<float>(), temp_result.numel(), ncclFloat, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
            }
        }
    }

    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::allReduce(torch::Tensor tensor) {
    NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, vector<torch::Tensor> &output_list, const std::function<void(torch::Tensor, torch::Tensor)> &op) {
    vector<int> tensor_ranks;
    int assigned_rank = 0;
    int num_comm_tensors = 0;
    int min_tensor_size = 512*512;
    for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        if (tensor.numel() < min_tensor_size) {
            tensor_ranks.push_back(-1);
        } else {
            tensor_ranks.push_back(assigned_rank);
            assigned_rank++;
            assigned_rank %= m_size;
            num_comm_tensors++;
        }
    }
    if (m_size > 1) {
        _extendComms(num_comm_tensors);
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (assigned_rank == -1) {
            op(tensor, output);
        } else {
            if (assigned_rank == m_rank) {
                op(tensor, output);
            } 
        }
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (m_size > 1 and assigned_rank >= 0) {
            NCCLCHECK(ncclBroadcast(output.data_ptr<float>(), output.data_ptr<float>(), output.numel(), ncclFloat, assigned_rank, m_nccl_comms[m_current_comm], m_streams[m_current_comm])); 
            //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
            m_current_comm++;
            m_current_comm %= m_num_comms;
        }
    }
}

void Communicator::sendrecv(torch::Tensor send_tensor, torch::Tensor recv_tensor, int peer) {
    NCCLCHECK(ncclGroupStart());
    ncclDataType_t nccl_type;
    if (torch::kFloat32 == send_tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclSend(send_tensor.data_ptr<float>(), send_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
        NCCLCHECK(ncclRecv(recv_tensor.data_ptr<float>(), recv_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclSend(send_tensor.data_ptr<long>(), send_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
        NCCLCHECK(ncclRecv(recv_tensor.data_ptr<long>(), recv_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    NCCLCHECK(ncclGroupEnd());
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));

    m_current_comm++;
    m_current_comm %= m_num_comms;
}
