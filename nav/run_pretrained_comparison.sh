#!/usr/bin/env bash
# ============================================================
#  run_pretrained_comparison.sh
#  Run Mask R-CNN (COCO pretrained), YOLOv8 (COCO pretrained),
#  and Cascade Mask R-CNN (COCO pretrained) evaluations
#  across ALL object categories (10 scenes) inside crazy_leavitt.
#
#  ALL models use ONLY pretrained COCO weights — no finetuning.
#
#  Usage:
#    bash run_pretrained_comparison.sh [NUM_SCENES] [START_SCENE]
#    Defaults: 10 scenes starting from 0
# ============================================================

set -e

CONTAINER="26959a4d38b8"
NUM_SCENES="${1:-10}"
START_SCENE="${2:-0}"
OUT_BASE="/nav/data/comparison_with_pretrained"

echo "============================================================"
echo "  3-Model PRETRAINED-ONLY Comparison"
echo "  Models: Mask R-CNN R-101 (COCO) | YOLOv8 (COCO) | Cascade (COCO)"
echo "  Scenes: ${START_SCENE} to $((START_SCENE + NUM_SCENES - 1))"
echo "  Container: ${CONTAINER}"
echo "  Visualization: ENABLED (saved per-model)"
echo "  Output: ${OUT_BASE}"
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
docker exec ${CONTAINER} mkdir -p ${OUT_BASE}/r101coco
docker exec ${CONTAINER} mkdir -p ${OUT_BASE}/yolo
docker exec ${CONTAINER} mkdir -p ${OUT_BASE}/cascade

# ----------------------------------------------------------
# 1) Run Mask R-CNN R-101 COCO pretrained (80 classes -> 9 PEANUT)
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [1/6] Running Mask R-CNN R-101 (COCO pretrained) + viz"
echo "        seg_model_type=r101coco  (NO finetuning)"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=0 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type r101coco \
        --exp_name r101coco_allcat \
        --dump_location ${OUT_BASE}/r101coco/ \
        --start_ep ${START_SCENE} \
        --end_ep ${NUM_SCENES} \
        -v 2 \
        --sem_gpu_id 0 \
        --evaluation local
"

echo ""
echo "[1/6] Mask R-CNN R-101 (COCO pretrained) evaluation complete."

# ----------------------------------------------------------
# 2) Run YOLOv8 (pretrained COCO weights)
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [2/6] Running YOLOv8 (COCO pretrained) + viz"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=0 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type yolo \
        --seg_model_wts /nav/yolov8x-seg.pt \
        --exp_name yolo_allcat \
        --dump_location ${OUT_BASE}/yolo/ \
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
echo "  [3/6] Running Cascade Mask R-CNN (COCO pretrained) + viz"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    -e CUDA_VISIBLE_DEVICES=0 \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect_all_categories.py \
        --seg_model_type cascade \
        --exp_name cascade_allcat \
        --dump_location ${OUT_BASE}/cascade/ \
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
        --maskrcnn_results ${OUT_BASE}/r101coco/r101coco_allcat_all_categories_results.txt \
        --yolo_results ${OUT_BASE}/yolo/yolo_allcat_all_categories_results.txt \
        --cascade_results ${OUT_BASE}/cascade/cascade_allcat_all_categories_results.txt \
        --output_dir ${OUT_BASE}/charts
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

HOST_OUT="/home/koray/PEANUT/data/comparison_with_pretrained"
mkdir -p "${HOST_OUT}"

# Copy entire comparison_with_pretrained tree (results + visualizations + charts)
docker cp ${CONTAINER}:${OUT_BASE}/. "${HOST_OUT}/"

# ----------------------------------------------------------
# 6) Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  [6/6] DONE! — All models used PRETRAINED COCO weights only"
echo "============================================================"
echo ""
echo "  All output: ${HOST_OUT}/"
echo ""
echo "  Models & weights used:"
echo "    Mask R-CNN R-101:      detectron2 model_zoo COCO-InstSeg/mask_rcnn_R_101_FPN_3x (COCO pretrained)"
echo "    YOLOv8:                yolov8x-seg.pt (COCO pretrained)"
echo "    Cascade Mask R-CNN:    detectron2 model_zoo cascade_mask_rcnn_X_152 (COCO pretrained)"
echo ""
echo "  Visualizations:"
echo "    ${HOST_OUT}/r101coco/dump/r101coco_allcat/episodes/"
echo "    ${HOST_OUT}/yolo/dump/yolo_allcat/episodes/"
echo "    ${HOST_OUT}/cascade/dump/cascade_allcat/episodes/"
echo ""
echo "  Raw results:"
echo "    ${HOST_OUT}/r101coco/r101coco_allcat_all_categories_results.txt"
echo "    ${HOST_OUT}/yolo/yolo_allcat_all_categories_results.txt"
echo "    ${HOST_OUT}/cascade/cascade_allcat_all_categories_results.txt"
echo ""
echo "  Charts:"
echo "    ${HOST_OUT}/charts/comprehensive_comparison.png"
echo "    ${HOST_OUT}/charts/per_category_comparison.png"
echo "    ${HOST_OUT}/charts/per_episode_detail.png"
echo "    ${HOST_OUT}/charts/time_comparison.png"
echo "    ${HOST_OUT}/charts/scene_category_heatmap.png"
echo "============================================================"
