FROM nvidia/cuda:10.2-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ARG DEBIAN_FRONTEND=noninteractive
RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list;
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com A4B469963BF863CC
RUN apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
RUN apt-get update --fix-missing
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends --fix-missing \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        cmake \
        lsb-release \
        libcudnn7=7.6.5.32-1+cuda10.2 \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libnccl2=2.15.5-1+cuda10.2 \
        libnccl-dev=2.15.5-1+cuda10.2

# install framework
# note: for tf <= 1.14, you need gcc-4.9
ARG FRAMEWORK=pytorch
RUN if [ "$FRAMEWORK" = "tensorflow" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U tensorflow-gpu==1.15.0; \
    elif [ "$FRAMEWORK" = "pytorch" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U numpy==1.18.1 torchvision==0.9.1 torch==1.8.1; \
    elif [ "$FRAMEWORK" = "mxnet" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U mxnet-cu100==1.5.0; \
    else \
        echo "unknown framework: $FRAMEWORK"; \
        exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH