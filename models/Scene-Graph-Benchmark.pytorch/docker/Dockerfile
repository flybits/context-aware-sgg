FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils vim nano git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg libsm6 libxext6 libqt5gui5

# Create a Python environment
RUN conda install -y conda-build ipython \
 && conda clean -ya \
 && pip install requests ninja yacs cython matplotlib opencv-python tqdm h5py

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && git fetch --all --tags --force \
 && git checkout tags/v0.9.0 -b v0.9.0-branch \
 && python setup.py install

# install pycocotools
RUN git clone  https://github.com/ppwwyyxx/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && git fetch --all --tags --force \
 && git checkout tags/22.03 -b v22.03-branch \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
COPY . scenegraph
RUN cd scenegraph \
 && pip install -r requirements.txt scipy \
 && python setup.py build develop
