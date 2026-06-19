FROM python:3.12.13-bookworm

LABEL Description="JaxILI Docker Image with Python 3.12"
ENV SHELL /bin/bash

# Install system dependencies
RUN apt-get update -y --quiet --fix-missing && \
    apt-get dist-upgrade -y --quiet --fix-missing && \
    apt-cache policy autconf && \
    apt-get install -y --quiet \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    g++ \
    gcc  \
    gfortran \
    git \
    git-lfs \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libcfitsio-dev \
    libfftw3-bin \
    libfftw3-dev \
    libgl1-mesa-glx \
    libgsl-dev \
    libhealpix-cxx-dev \
    libtool \
    libtool-bin \
    libtool-doc \
    locales \
    locate \
    make \
    openmpi-bin \
    libopenmpi-dev \
    pkg-config \
    protobuf-compiler \
    vim \
    xterm && \
    apt-get clean -y && \
    apt-get autoremove --purge --quiet -y && \
    rm -rf /var/lib/apt/lists/* /var/tmp/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jaxili==0.1.3
