#!/usr/bin/env bash
# ============================================================
#  run_3model_comparison.sh
#  Run Mask R-CNN, YOLOv8, and Cascade Mask R-CNN evaluations
#  across ALL object categories (10 scenes) inside crazy_leavitt.
#
#  All models use pretrained weights only.
#
#  Usage:
#    bash run_3model_comparison.sh [NUM_SCENES] [START_SCENE]
#    Defaults: 10 scenes starting from 0
# ============================================================

set -e

CONTAINER="26959a4d38b8"
NUM_SCENES="${1:-10}"
START_SCENE="${2:-0}"

echo "============================================================"
echo "  3-Model All-Category Comparison"
echo "  Models: Mask R-CNN | YOLOv8 | Cascade Mask R-CNN"
echo "  Scenes: ${START_SCENE} to $((NUM_SCENES - 1))"
echo "  Container: ${CONTAINER}"
echo "  Visualization: ENABLED (saved per-model)"
echo "============================================================"

# ----------------------------------------------------------
# 0) Copy scripts into container & create output dirs
# ----------------------------------------------------------
echo "[0/6] Copying scripts into container..."
docker cp /home/koray/PEANUT/nav/collect_all_categories.py ${CONTAINER}:/nav/collect_all_categories.py
docker cp /home/koray/PEANUT/nav/compare_3models.py        ${CONTAINER}:/nav/compare_3models.py
docker cp /home/koray/PEANUT/nav/arguments.py               ${CONTAINER}:/nav/arguments.py
docker cp /home/koray/PEANUT/nav/constants.py               ${CONTAINER}:/nav/constants.py

# Create per-model visualization directories inside container
docker exec ${CONTAINER} mkdir -p /nav/data/comparison_run/maskrcnn
docker exec ${CONTAINER} mkdir -p /nav/data/comparison_run/yolo
docker exec ${CONTAINER} mkdir -p /nav/data/comparison_run/cascade

# ----------------------------------------------------------
# 1) Run Mask R-CNN (pretrained R-101 cat9 weights)
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [1/6] Running Mask R-CNN (pretrained) + visualization"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=1 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type maskrcnn \
        --seg_model_wts /nav/agent/utils/mask_rcnn_R_101_cat9.pth \
        --exp_name maskrcnn_allcat \
        --dump_location /nav/data/comparison_run/maskrcnn/ \
        --start_ep ${START_SCENE} \
        --end_ep ${NUM_SCENES} \
        -v 2 \
        --sem_gpu_id 0 \
        --evaluation local
"

echo ""
echo "[1/6] Mask R-CNN evaluation complete."

# ----------------------------------------------------------
# 2) Run YOLOv8 (pretrained COCO weights)
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [2/6] Running YOLOv8 (pretrained) + visualization"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=1 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type yolo \
        --seg_model_wts /nav/yolov8x-seg.pt \
        --exp_name yolo_allcat \
        --dump_location /nav/data/comparison_run/yolo/ \
        --start_ep ${START_SCENE} \
        --end_ep ${NUM_SCENES} \
        -v 2 \
        --sem_gpu_id 0 \
        --evaluation local
"

echo ""
echo "[2/6] YOLOv8 evaluation complete."

# ----------------------------------------------------------
# 3) Run Cascade Mask R-CNN (pretrained COCO weights)
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [3/6] Running Cascade Mask R-CNN (pretrained) + visualization"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=1 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type cascade \
        --exp_name cascade_allcat \
        --dump_location /nav/data/comparison_run/cascade/ \
        --start_ep ${START_SCENE} \
        --end_ep ${NUM_SCENES} \
        -v 2 \
        --sem_gpu_id 0 \
        --evaluation local
"

echo ""
echo "[3/6] Cascade Mask R-CNN evaluation complete."

# ----------------------------------------------------------
# 4) Generate comprehensive comparison charts
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [4/6] Generating comprehensive comparison charts"
echo "============================================================"

docker exec ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/compare_3models.py \
        --maskrcnn_results /nav/data/comparison_run/maskrcnn/maskrcnn_allcat_all_categories_results.txt \
        --yolo_results /nav/data/comparison_run/yolo/yolo_allcat_all_categories_results.txt \
        --cascade_results /nav/data/comparison_run/cascade/cascade_allcat_all_categories_results.txt \
        --output_dir /nav/data/comparison_run/charts
"

echo ""
echo "[4/6] Charts generated."

# ----------------------------------------------------------
# 5) Copy results back to host
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [5/6] Copying results to host"
echo "============================================================"

mkdir -p /home/koray/PEANUT/data/comparison_run/

# Copy entire comparison_run tree (results + visualizations + charts)
docker cp ${CONTAINER}:/nav/data/comparison_run/ \
    /home/koray/PEANUT/data/comparison_run/

# ----------------------------------------------------------
# 6) Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [6/6] DONE!"
echo "============================================================"
echo ""
echo "  All output: /home/koray/PEANUT/data/comparison_run/"
echo ""
echo "  Visualizations:"
echo "    data/comparison_run/maskrcnn/dump/maskrcnn_allcat/episodes/"
echo "    data/comparison_run/yolo/dump/yolo_allcat/episodes/"
echo "    data/comparison_run/cascade/dump/cascade_allcat/episodes/"
echo ""
echo "  Raw results:"
echo "    data/comparison_run/maskrcnn/maskrcnn_allcat_all_categories_results.txt"
echo "    data/comparison_run/yolo/yolo_allcat_all_categories_results.txt"
echo "    data/comparison_run/cascade/cascade_allcat_all_categories_results.txt"
echo ""
echo "  Charts:"
echo "    data/comparison_run/charts/comprehensive_comparison.png"
echo "    data/comparison_run/charts/per_category_comparison.png"
echo "    data/comparison_run/charts/per_episode_detail.png"
echo "    data/comparison_run/charts/time_comparison.png"
echo "    data/comparison_run/charts/scene_category_heatmap.png"
echo "============================================================"
