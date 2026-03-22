#!/bin/bash
set -e
cd /nav

echo "============================================"
echo "  Starting COCO-80 Semantic Map Generation  "
echo "  3 Models x 10 Scenes = 30 runs            "
echo "============================================"
echo "Start time: $(date)"

# ──── YOLOv8x-seg ────
echo ""
echo "########################################"
echo "  MODEL 1/3: YOLOv8x-seg"
echo "########################################"
conda run --no-capture-output -n habitat python vlmaps_dataloader_coco80.py \
    --scene_dir all \
    --seg_model_type yolo \
    --seg_model_wts /nav/yolov8x-seg.pt \
    --visualize 0 \
    --only_explore 1 \
    --map_size_cm 9600 \
    -d /nav/data/recommended_settings/coco80_yolov8/ \
    --frame_step 1

echo ""
echo "YOLOv8x-seg completed at: $(date)"

# ──── YOLO11x-seg ────
echo ""
echo "########################################"
echo "  MODEL 2/3: YOLO11x-seg"
echo "########################################"
conda run --no-capture-output -n habitat python vlmaps_dataloader_coco80.py \
    --scene_dir all \
    --seg_model_type yolo11 \
    --seg_model_wts /nav/yolo11x-seg.pt \
    --visualize 0 \
    --only_explore 1 \
    --map_size_cm 9600 \
    -d /nav/data/recommended_settings/coco80_yolo11/ \
    --frame_step 1

echo ""
echo "YOLO11x-seg completed at: $(date)"

# ──── YOLO26x-seg ────
echo ""
echo "########################################"
echo "  MODEL 3/3: YOLO26x-seg"
echo "########################################"
conda run --no-capture-output -n habitat python vlmaps_dataloader_coco80.py \
    --scene_dir all \
    --seg_model_type yolo26 \
    --seg_model_wts /nav/yolo26x-seg.pt \
    --visualize 0 \
    --only_explore 1 \
    --map_size_cm 9600 \
    -d /nav/data/recommended_settings/coco80_yolo26/ \
    --frame_step 1

echo ""
echo "YOLO26x-seg completed at: $(date)"

echo ""
echo "============================================"
echo "  ALL 3 MODELS COMPLETE!"
echo "  End time: $(date)"
echo "============================================"
