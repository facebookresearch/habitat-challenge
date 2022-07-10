# Base image
FROM nvidia/cudagl:10.1-runtime-ubuntu18.04

# Setup basic packages 1.09GB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libglm-dev \
    libegl1-mesa-dev \
    xorg-dev \
    freeglut3-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*


# Install conda 1.1GB
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install pyyaml mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Conda environment
RUN conda create -n habitat python=3.7 cmake=3.14.0

# Disable cache to have fresh challenge-2022 tags
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Setup habtiat-sim 1.81GB
RUN git clone --branch hab2_challenge_2022 https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; pip install -r requirements.txt;python setup.py install --headless"
# RUN /bin/bash -c ". activate habitat; conda install habitat-sim-challenge-2022 -c conda-forge -c aihabitat"

# Install habitat-lab 2.65GB
RUN git clone --branch challenge_tasks https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate habitat; cd habitat-lab; pip install -r requirements.txt; python setup.py develop --all"


# Silence habitat-sim logs
ENV HABITAT_SIM_LOG="quiet"
ENV MAGNUM_LOG="quiet"

# Install challenge dependencies
RUN /bin/bash -c ". activate habitat; pip install grpcio"
ADD evalai-remote-evaluation evalai-remote-evaluation
