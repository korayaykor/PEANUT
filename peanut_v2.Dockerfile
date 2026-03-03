# =============================================================================
# PEANUT v2 Dockerfile
# Python 3.8 + PyTorch 2.1 + CUDA 11.8 + Habitat 0.2.1
# Supports: YOLOv8/v9/v10/v11, SAM, Detectron2, MaskRCNN, Cascade
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# NVIDIA runtime: must include 'graphics' for EGL headless rendering
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget unzip \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
    libjpeg-dev libpng-dev libtiff-dev \
    libx11-6 libxau6 libxcb1 libxdmcp6 libxext6 \
    libopengl0 libegl1 libgles2 libgl1-mesa-glx libegl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Accept conda TOS and configure channels
RUN conda config --set auto_activate_base false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create habitat environment with Python 3.8
RUN conda create -n habitat -c conda-forge python=3.8 -y

# Install habitat-sim 0.2.1 (headless + bullet, Python 3.8)
# Must use exact build string to get headless_bullet variant (not headless_nobullet)
RUN conda install -n habitat -c aihabitat -c conda-forge \
    'habitat-sim=0.2.1=py3.8_headless_bullet_linux_fc7fb11ccec407753a73ab810d1dbb5f57d0f9b9' \
    -y

# Install PyTorch 2.1 with CUDA 11.8
RUN conda run -n habitat pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install habitat-lab (challenge-2022 branch, exact commit from original container)
RUN cd / && git clone https://github.com/facebookresearch/habitat-lab.git && \
    cd /habitat-lab && \
    git checkout e934b15c35233457cc3cb9c90ba0e207610dbd19 && \
    conda run -n habitat pip install -e .

# Patch rearrange task to be fault-tolerant (FetchRobotNoWheels not in habitat-sim 0.2.1)
# We only use ObjectNav, not rearrange tasks
RUN cd /habitat-lab && python3 -c "import re; path='habitat/tasks/rearrange/__init__.py'; content=open(path).read(); content=content.replace('def _try_register_rearrange_task():\n','def _try_register_rearrange_task():\n  try:\n'); content=content.rstrip()+chr(10)+'  except (ImportError, AttributeError) as e:'+chr(10)+'    import warnings; warnings.warn(str(e))'+chr(10); open(path,'w').write(content); print('Patched rearrange __init__')"

# Install Detectron2 (built for PyTorch 2.1 + CUDA 11.8)
RUN conda run -n habitat pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install core project dependencies
# Note: scikit-fmm==2019.1.30 fails with modern setuptools, use newer version
RUN conda run -n habitat pip install \
    opencv-python>=4.5.0 \
    cython \
    numpy \
    matplotlib \
    seaborn \
    "scikit-fmm>=2022.3.26" \
    scikit-image \
    imageio \
    scikit-learn \
    ifcfg \
    "gym==0.22.0"

# NOTE: mmcv/mmseg not installed globally.
# The /prediction module is a custom mmseg fork requiring mmcv-full 1.3-1.7,
# which is incompatible with PyTorch 2.x. The prediction module (SegFormer
# for semantic map prediction) is separate from segmentation detection models.
# If needed later, install mmcv-full with a compatible PyTorch version.

# Install Ultralytics (latest - supports YOLOv8/v9/v10/v11)
RUN conda run -n habitat pip install ultralytics

# Install SAM (Segment Anything)
RUN conda run -n habitat pip install git+https://github.com/facebookresearch/segment-anything.git

# Pre-download popular model weights (optional, saves time later)
# YOLOv11 seg models
RUN conda run -n habitat python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')" || true
RUN conda run -n habitat python -c "from ultralytics import YOLO; YOLO('yolo11l-seg.pt')" || true
RUN conda run -n habitat python -c "from ultralytics import YOLO; YOLO('yolo11x-seg.pt')" || true

# Copy config files
ADD configs/challenge_objectnav2021.local.rgbd.yaml /challenge_objectnav2021.local.rgbd.yaml
ADD configs/challenge_objectnav2021.remote.rgbd.yaml /challenge_objectnav2021.remote.rgbd.yaml
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ADD configs/challenge_objectnav2022noisy.local.rgbd.yaml /challenge_objectnav2022noisy.local.rgbd.yaml

# Copy prediction module
ADD prediction /prediction
RUN conda run -n habitat bash -c "cd /prediction && pip install -e ." || true

# Copy nav experiment script
ARG INCUBATOR_VER=unknown
ADD nav_exp.sh /nav_exp.sh

# Environment setup
ENV AGENT_EVALUATION_TYPE=remote
ENV PYTHONPATH="/nav"
ENV TRACK_CONFIG_FILE="/challenge_objectnav2022.local.rgbd.yaml"

# Shell setup: conda init + auto-activate habitat env
RUN conda init bash && \
    echo "conda activate habitat" >> /root/.bashrc

CMD ["/bin/bash", "-c", "source /root/.bashrc && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash /nav_exp.sh"]
