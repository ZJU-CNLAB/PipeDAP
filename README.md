# PipeDAP: An Efficient Communication Framework for Scheduling Decoupled All-Reduce Primitives in Distributed DNN Training #  
## Introduction ##
PipeDAP is a communication mechanism for scheduling decoupled all-reduce primitives in optimal order. PipeDAP is implemented on the PyTorch and DeAR framework. PipeDAP outperforms the state-of-the-art communication scheduling mechanisms including [DeAR](https://github.com/lzhangbv/dear_pytorch), [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler) and WFBP.
<div align=center><img src="system%20architecture.png" width="500"/></div> 

## Installation ##
### Prerequisites ###
* CUDA-10.2  
* [Python 3.6](https://www.python.org/ftp/python/)  
* [cudnn-7.6.5](https://developer.nvidia.com/rdp/cudnn-archive)  
* [NCCL-2.15.5](https://developer.nvidia.com/nccl/nccl-legacy-downloads)  
* [PyTorch-1.8.+](https://download.pytorch.org/whl/torch_stable.html)  

We highly recommend using Docker images for installing the above prerequisites. You can download Dockerfile in the directory of docker and run the following scripts:  
```
docker build -t PipeDAP:v1 . -f Dockerfile --build-arg FRAMEWORK=pytorch  
nvidia-docker run -it --net=host --shm-size=32768m -v /data:/data PipeDAP:v1 bash  
```
* [OpenMPI-4.0.+](https://www.open-mpi.org/software/ompi/v4.0/)  
### Quick Start ###
You can download this code to /root/code folder and first run the following script：
```
cd /root/code/PipeDAP  
pip install -r requirements.txt  
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.21.3  
```
If Horovod installation with NCCL failed, please check the [installation guide](https://horovod.readthedocs.io/en/stable/install_include.html).  
After that, please carefully configure MPIPATH in the file configs/envs.conf and compile the communication package:  
```
cd common/comm_core  
bash compile.sh  
```
Then, you can run the following scripts:  
```
cd PipeDAP  
dnn=resnet50 bs=64 nworkers=4 baseline=PipeDAP ./dist.sh  
```  
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the ResNet50 model with a fake ImageNet2012 dataset (i.e., randomly generate the input image of 224\*224\*3) using the PipeDAP mechanism.
