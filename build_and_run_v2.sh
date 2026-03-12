#!/usr/bin/env bash
# ============================================================
# build_and_run_v2.sh - Build and run the PEANUT v2 Docker image.
#
# Builds peanut_v2.Dockerfile (Python 3.9, PyTorch 2.1, CUDA 11.8)
# with support for YOLOv8/v11/v26, SAM, and Detectron2.
# After building, starts a container with GPU access and mounts
# habitat-challenge-data, data, and nav directories.
# ============================================================

DOCKER_NAME="peanut_v2"

echo "============================================="
echo "  Building PEANUT v2 Docker Image"
echo "  Python 3.9 + PyTorch 2.1 + CUDA 11.8"
echo "  Supports: YOLOv8/v9/v10/v11, SAM, Detectron2"
echo "============================================="

DOCKER_BUILDKIT=1 docker build . \
    --build-arg INCUBATOR_VER=$(date +%Y%m%d-%H%M%S) \
    --file peanut_v2.Dockerfile \
    -t ${DOCKER_NAME}

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed!"
    exit 1
fi

echo ""
echo "============================================="
echo "  Build successful! Starting container..."
echo "============================================="

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/habitat-challenge-data/data/scene_datasets/hm3d \
    -v $(realpath habitat-challenge-data/data/scene_datasets/hm3d):/data/scene_datasets/hm3d \
    -v $(pwd)/data:/data \
    -v $(pwd)/nav:/nav \
    --gpus='all' \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml" \
    --ipc=host \
    --name peanut_v2 \
    ${DOCKER_NAME}
