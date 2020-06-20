# @author hyunwoong
# @see "https://hub.docker.com/r/gusdnd852/chatbot"

# 1. Load cuda-ubuntu
ARG UBUNTU_VERSION=18.04
ARG ARCH
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

# 2. python set up
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools
RUN ln -s $(which python3) /usr/local/bin/python

# 3. java set up
RUN apt install openjdk-11-jdk -y

# 4. set up python packages
COPY requirements.txt /
RUN pip3 install -r requirements.txt