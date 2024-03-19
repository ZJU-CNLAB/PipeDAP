# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import comm_core
import time
torch.random.manual_seed(10)

comm_core.init()


def allreduce():
    rank = comm_core.rank()
    torch.cuda.set_device(rank%4)
    communicator = comm_core.Communicator(1)
    t = torch.rand(2).cuda()
    tensor = t * t.t()
    communicator.allReduce(tensor)
    communicator.synchronize()
    print('before rank: %d' % rank, t*t.t())
    print('after rank: %d' % rank, tensor)

def reducescatter():
    rank = comm_core.rank()
    nworkers = comm_core.size()
    torch.cuda.set_device(rank%4)
    communicator = comm_core.Communicator(1)
    send_tensor = torch.rand(16).cuda()
    results = torch.zeros_like(send_tensor)
    recv_tensor = send_tensor.new_zeros(send_tensor.numel()//nworkers)
    communicator.reduceScatter(send_tensor, recv_tensor)
    communicator.allGather(recv_tensor, results)
    communicator.allReduce(send_tensor)
    #communicator.allReduceRSAG(send_tensor)
    #communicator.allReduceRB(send_tensor)
    communicator.synchronize()
    print('before rank: %d' % rank, (send_tensor).norm())
    print('after rank: %d' % rank, (results-send_tensor).norm())

def decoupleallreduce():
    rank = comm_core.rank()
    nworkers = comm_core.size()
    torch.cuda.set_device(rank%4)
    communicator = comm_core.Communicator(1)
    send_tensor1 = torch.rand(17).cuda()
    send_tensor2 = torch.zeros_like(send_tensor1)
    send_tensor2.copy_(send_tensor1)

    communicator.allReduce(send_tensor1)
    #communicator.allReduceRSAG(send_tensor2)
    communicator.allReduceRB(send_tensor2)
    communicator.synchronize()
    print('before rank: %d' % rank, (send_tensor2).norm())
    print('after rank: %d' % rank, (send_tensor1-send_tensor2).norm())

def bcast():
    rank = comm_core.rank()
    torch.cuda.set_device(rank%4)
    communicator = comm_core.Communicator(1)
    tensor = (torch.rand(2) * 100).float().cuda()
    if rank == 0:
        tensor.fill_(10)
    #print('before rank: %d' % rank, tensor)
    communicator.bcast(tensor, 0)
    print('after rank: %d' % rank, tensor)


def multi_bcast():
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    torch.cuda.set_device(local_rank)
    communicator = tcmm.Communicator(rank, size)
    ntensors = 2
    tensors = []
    for i in range(ntensors):
        t = torch.rand(2).cuda()
        tensors.append(t)
    def _op(tensor):
        tensor.mul_(2)
        return None
    print('before rank: %d' % rank, tensors)
    communicator.multiBcast(tensors, _op)
    print('after rank: %d' % rank, tensors)

def reduce():
    rank = comm_core.rank()
    size = comm_core.size()
    torch.cuda.set_device(rank%4)

    nstreams = 4
    communicator = comm_core.Communicator(nstreams)

    n_elements = 1*1024
    iterations = 1000
    tensors = []
    for i in range(nstreams): 
        tensors.append(torch.rand(n_elements).cuda())
    if rank == 0:
        print('before rank: %d' % rank, time.time())
    for i in range(nstreams):
        #communicator.reduce(tensor, 0)
        communicator.allReduce(tensors[i])
    #hvd.allreduce(tensor)
    communicator.synchronize()
    start = time.time()
    previous = start
    for i in range(iterations):
        #communicator.reduce(tensor, 0)
        for j in range(nstreams):
            communicator.allReduce(tensors[j])
        #hvd.allreduce(tensor)
        current = time.time()
        #if rank ==0:
        #    print('i: ', i, current-previous)
        previous = current
    communicator.synchronize()
    end = time.time()
    if rank == 0:
        print('after rank: %d' % rank, time.time(), (end-start)/iterations)
        print('throughput: ', nstreams * n_elements * 4 *1e-9/ ((end-start)/iterations), 'GB/s')

def sendrecv():
    rank = comm_core.rank()
    size = comm_core.size()
    torch.cuda.set_device(rank%4)

    nstreams = 1
    communicator = comm_core.Communicator(nstreams)

    n_elements = 4
    iterations = 1
    send_tensor = torch.rand(n_elements).cuda()
    recv_tensor = torch.zeros_like(send_tensor)
    if rank % 2 == 0:
        print('before rank: %d' % rank, send_tensor)
    if rank == 0:
        peer = 1
    elif rank == 1:
        peer = 0
    elif rank == 2:
        peer = 3
    else:
        peer = 2
    communicator.sendrecv(send_tensor, recv_tensor, peer)
    if rank %2 == 1:
        print('rank: %d' % rank, recv_tensor)

def perf_benchmarks():
    rank = comm_core.rank()
    size = comm_core.size()
    torch.cuda.set_device(rank%4)

    nstreams = 1
    communicator = comm_core.Communicator(nstreams)

    n_elements = 1*1024*1024
    iterations = 100
    tensors = []
    for i in range(nstreams): 
        tensors.append(torch.rand(n_elements).cuda())
    if rank == 0:
        print('before rank: %d' % rank, time.time())
    for i in range(nstreams):
        communicator.bcast(tensors[i], 0)
    communicator.synchronize()
    start = time.time()
    previous = start
    for i in range(iterations):
        for j in range(nstreams):
            communicator.bcast(tensors[j], 0)
        current = time.time()
        previous = current
    communicator.synchronize()
    end = time.time()
    if rank == 0:
        print('after rank: %d' % rank, time.time(), (end-start)/iterations)
        print('size:', n_elements * 4, ',throughput: ', nstreams * n_elements * 4 *1e-9/ ((end-start)/iterations), 'GB/s')

if __name__ == '__main__':
    #perf_benchmarks()
    #reducescatter()
    decoupleallreduce()
    #allreduce()
    #bcast()
    #multi_bcast()
    #reduce()
    #sendrecv()
