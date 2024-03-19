from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import PipeDAP
import os
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'

from compression import compressors
import timeit
import numpy as np
from profiling import benchmark

import time

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,7"

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use mixed precision')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=5,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=20,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=5,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')
parser.add_argument('--threshold', type=int, default=536870912, help='Set threshold if mgwfbp is False')
parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')
parser.add_argument('--compressor', type=str, default='none', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=1, help='Density for sparsification')
parser.add_argument('--exclude-parts', type=str, default='', help='choices: reducescatter, allgather')

parser.add_argument('--baseline', type=str, default='PipeDAP', help='choices: PipeDAP, DeAR')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['HOROVOD_NUM_NCCL_STREAMS'] = str(args.nstreams)

# choose baseline: ['PipeDAP', 'DeAR']
# baseline = 'PipeDAP'
#import hv_distributed_optimizer as hvd
if args.baseline == 'PipeDAP':
    import opt_PipeDAP_sumcomm as hvd
    # import opt_PipeDAP as hvd
elif args.baseline == 'DeAR':
    import opt_DeAR as hvd
#import dopt_rsag_bo as hvd
if hvd.rank() == 0:
    print("Running " + args.baseline + "...")

schedule_order = PipeDAP.PipeDAP(args.model)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.rank()%8)
    #torch.cuda.set_device(hvd.rank()%8)

if args.fp16:
    import apex
else:
    apex = None

hvd.init()

cudnn.benchmark = True

# Set up standard model.
if args.model == 'inceptionv4':
    from inceptionv4 import inceptionv4
    model = inceptionv4()
else:
    model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

# Set up fixed fake data
image_size = 224
if args.model == 'inception_v3':
    image_size = 227
data = torch.randn(args.batch_size, 3, image_size, image_size)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()



if args.mgwfbp:
    seq_layernames, layerwise_times, _ = benchmark(model, (data, target), F.cross_entropy, task='imagenet')
    layerwise_times = comm.bcast(layerwise_times, root=0)
else:
    seq_layernames, layerwise_times = None, None

if hvd.size() > 1:
    if args.baseline == 'PipeDAP':
        optimizer = hvd.DistributedOptimizer(optimizer, model=model, compression=compressors[args.compressor](), is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=args.threshold, writer=None, gradient_path='./', fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, multi_job_scheduling=False, exclude_parts=args.exclude_parts, schedule_order=schedule_order)
    elif args.baseline == 'DeAR':
        optimizer = hvd.DistributedOptimizer(optimizer, model=model, compression=compressors[args.compressor](), is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=args.threshold, writer=None, gradient_path='./', fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, multi_job_scheduling=False, exclude_parts=args.exclude_parts)

if apex is not None:
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O2', loss_scale=128.0)

if hvd.size() > 0:
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    #hvd.broadcast_optimizer_state(optimizer, root_rank=0)



def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    if apex is not None:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('FP16: %s' % args.fp16)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
iter_times = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)
    iter_times.append(time / args.num_batches_per_iter)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Iteraction time: %.3f +-%.3f' % (np.mean(iter_times), 1.96*np.std(iter_times)))
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
