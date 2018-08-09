# -*- mode: dockerfile -*-
# dockerfile to build libmxnet.so on GPU
# Use cuda 9.0
# FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 
FROM nvidia/cuda:latest
MAINTAINER SonayangZhang

COPY install/cpp.sh install/
RUN chmod +x install/cpp.sh
RUN install/cpp.sh

# ENV BUILD_OPTS "USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
# RUN git clone --recursive https://github.com/dmlc/mxnet && cd mxnet && \
#    make -j$(nproc) $BUILD_OPTS

# OpenCV
RUN apt-get update && \
        apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev

WORKDIR /
ENV OPENCV_VERSION="3.4.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python2.7 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python2.7) \
  -DPYTHON_INCLUDE_DIR=$(python2.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python2.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install -j ${nproc} \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

# -*- mode: dockerfile -*-
# part of the dockerfile to install the python binding

COPY install/python.sh install/
RUN chmod +x install/python.sh
RUN install/python.sh

RUN pip2 install nose numpy==1.14.0 nose-timer requests==2.18.4 Pillow easydict pyyaml sacred visdom Cython matplotlib scikit-image tqdm mxnet-cu90 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install nose pylint numpy==1.14.0 nose-timer requests==2.18.4 Pillow easydict pyyaml sacred visdom Cython matplotlib scikit-image tqdm mxnet-cu90 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip2 install opencv-python==3.4.1.15 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install opencv-python==3.4.1.15 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get -y install python-tk
RUN apt-get -y install python3-tk

ENV PYTHONPATH=/mxnet/python 
CMD sh -c 'ln -s /dev/null /dev/raw1394'; bash
