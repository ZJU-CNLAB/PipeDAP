#!/bin/bash
MPI_HOME=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.8/bin/python
# 100GbIB
rdma=0

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -mca mpi_warn_on_fork 0 \
    -x RDMA=$rdma \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -mca mpi_warn_on_fork 0 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=VERSION \
    -x HOROVOD_CACHE_CAPACITY=0"
fi

$MPI_HOME/bin/mpirun --prefix $MPI_HOME --oversubscribe -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    $params \
    $PY tests/test_comm.py
