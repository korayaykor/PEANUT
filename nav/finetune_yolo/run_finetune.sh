#!/usr/bin/env bash
# =============================================================================
# Fine-tune YOLOv8-seg on Habitat data — PAPER SCALE
#
# Paper-scale dataset (matching PEANUT arXiv:2212.02497):
#   - 80 train scenes × 1000 imgs = 80,000 train images
#   - 20 val scenes   × 1000 imgs = 20,000 val images
#   - Labels from Habitat GT semantic annotations (NOT Mask R-CNN teacher)
#
# This script uses BOTH GPUs for parallel data collection and training.
#
# Usage:
#   docker exec -it 26959a4d38b8 bash /nav/finetune_yolo/run_finetune.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="/data/yolo_dataset_v2"
YOLO_WEIGHTS="/nav/yolov8x-seg.pt"
GPUS="0,1"
WORKERS_PER_GPU=4
TARGET_PER_SCENE=1000
EPOCHS=50
BATCH=32
IMGSZ=640
FREEZE=15
LR=0.001
PROJECT="/data/yolo_runs"
NAME="habitat_gt_finetune_v2"

echo "============================================"
echo "  YOLO Fine-Tuning Pipeline — PAPER SCALE"
echo "  GPUs: ${GPUS}  Workers/GPU: ${WORKERS_PER_GPU}"
echo "  Target: ${TARGET_PER_SCENE} imgs/scene"
echo "============================================"

# ---- Step 1: Collect training data (80 scenes × 1000 = 80K) ----
echo ""
echo "[Step 1/3] Collecting TRAIN data (80 scenes × ${TARGET_PER_SCENE} imgs)..."
echo "  Using both GPUs with ${WORKERS_PER_GPU} workers each"
echo ""

conda run --no-capture-output -n habitat python "${SCRIPT_DIR}/collect_yolo_data.py" \
    --split train \
    --save_dir "${SAVE_DIR}" \
    --gpus "${GPUS}" \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --target_per_scene ${TARGET_PER_SCENE} \
    --max_steps 500 \
    2>&1 | tee /data/yolo_collect_train.log

# ---- Step 2: Collect validation data (20 scenes × 1000 = 20K) ----
echo ""
echo "[Step 2/3] Collecting VAL data (20 scenes × ${TARGET_PER_SCENE} imgs)..."
echo ""

conda run --no-capture-output -n habitat python "${SCRIPT_DIR}/collect_yolo_data.py" \
    --split val \
    --save_dir "${SAVE_DIR}" \
    --gpus "${GPUS}" \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --target_per_scene ${TARGET_PER_SCENE} \
    --max_steps 500 \
    2>&1 | tee /data/yolo_collect_val.log

# ---- Step 3: Fine-tune YOLO (both GPUs) ----
echo ""
echo "[Step 3/3] Fine-tuning YOLOv8x-seg on GPU 0 (batch=${BATCH})..."
echo "  epochs=${EPOCHS}, freeze=${FREEZE}, lr=${LR}"
echo ""

conda run --no-capture-output -n habitat python "${SCRIPT_DIR}/train_yolo.py" \
    --data_dir "${SAVE_DIR}" \
    --base_weights "${YOLO_WEIGHTS}" \
    --epochs ${EPOCHS} \
    --batch ${BATCH} \
    --imgsz ${IMGSZ} \
    --freeze ${FREEZE} \
    --lr0 ${LR} \
    --device "0,1" \
    --project "${PROJECT}" \
    --name "${NAME}" \
    2>&1 | tee /data/yolo_train_v2.log

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo ""
echo "  Dataset : ${SAVE_DIR}"
echo "  Weights : ${PROJECT}/${NAME}/weights/best.pt"
echo ""
echo "  To benchmark:"
echo "    python nav/collect_all_categories.py \\"
echo "        --seg_model_type yolo \\"
echo "        --seg_model_wts ${PROJECT}/${NAME}/weights/best.pt"
echo "============================================"
