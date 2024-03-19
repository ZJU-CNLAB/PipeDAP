#!/bin/bash
nworkers="${nworkers:-4}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
compressor="${compressor:-none}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
nstreams="${nstreams:-1}"
mgwfbp="${mgwfbp:-0}"
asc="${asc:-0}"
threshold="${threshold:-0}"
exclude_parts="${exclude_parts:-''}"
baseline="${baseline:-PipeDAP}"
horovodrun=horovodrun
source ../configs/envs.conf

if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    benchfile="bert_benchmark.py --model $dnn --sentence-len $senlen --exclude-parts $exclude_parts --baseline $baseline"
elif [ "$dnn" = "gpt2" ]; then
    benchfile="gpt2_benchmark.py --model $dnn --exclude-parts $exclude_parts --baseline $baseline"
else
    benchfile="imagenet_benchmark.py --model $dnn --exclude-parts $exclude_parts --baseline $baseline"
fi

if [ "$compressor" = "none" ]; then 
    cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold $threshold"
    if [ "$asc" = "1" ]; then 
        cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --nstreams $nstreams --asc"
    fi
else
    cmd="$PY $benchfile --density 0.001 --compressor $compressor --batch-size $bs --nstreams $nstreams --threshold 67108864"
fi
echo $cmd

#10GbE Config
if [ "$rdma" = "0" ]; then
  $horovodrun -np $nworkers $cmd

elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
else
#100GbIB Config with Ethernet
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_NET_GDR_LEVEL=0 \
    -x NCCL_NET_GDR_READ=0 \
    -x NCCL_IB_CUDA_SUPPORT=0 \
    -x NCCL_LAUNCH_MODE=PARALLEL \
    -x WFSGD_TIMELINE=${WFSGD_TIMELINE} \
    $cmd
fi

