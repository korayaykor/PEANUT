#!/bin/bash
# ============================================================================
# ObjectNav Evaluation - First 25 Episodes - YOLOv8 / YOLO11 / YOLO26
# ============================================================================
# Runs inside the peanut_v2 Docker container (74e7fcaec6f3)
#
# Usage from host:
#   bash /home/koray/PEANUT/nav/run_objectnav_25ep.sh 2>&1 | tee /home/koray/PEANUT/data/objectnav_25ep.log
# ============================================================================

CONTAINER="74e7fcaec6f3"
DUMP_DIR="/data/objectnav_25ep"
START_EP=0
END_EP=25
VIS=2   # 2 = save visualization images, 0 = no visualization

echo "============================================================"
echo "ObjectNav 25-Episode Evaluation"
echo "Container: $CONTAINER"
echo "Output: $DUMP_DIR"
echo "Episodes: $START_EP -> $END_EP"
echo "Started: $(date)"
echo "============================================================"

# ---- YOLOv8 ----
echo ""
echo "============================================================"
echo "[1/3] Running YOLOv8 (yolov8x-seg.pt)"
echo "Started: $(date)"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    $CONTAINER bash -c "source activate habitat && cd / && \
    python /nav/collect_25ep.py \
        --seg_model_type yolo \
        --seg_model_wts /nav/yolov8x-seg.pt \
        --exp_name yolov8_25ep \
        --dump_location $DUMP_DIR/ \
        --start_ep $START_EP --end_ep $END_EP \
        -v $VIS --sem_gpu_id 0 --evaluation local"

echo "[1/3] YOLOv8 completed: $(date)"

# ---- YOLO11 ----
echo ""
echo "============================================================"
echo "[2/3] Running YOLO11 (yolo11x-seg.pt)"
echo "Started: $(date)"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    $CONTAINER bash -c "source activate habitat && cd / && \
    python /nav/collect_25ep.py \
        --seg_model_type yolo11 \
        --yolo11_model_path /nav/yolo11x-seg.pt \
        --exp_name yolo11_25ep \
        --dump_location $DUMP_DIR/ \
        --start_ep $START_EP --end_ep $END_EP \
        -v $VIS --sem_gpu_id 0 --evaluation local"

echo "[2/3] YOLO11 completed: $(date)"

# ---- YOLO26 ----
echo ""
echo "============================================================"
echo "[3/3] Running YOLO26 (yolo26x-seg.pt)"
echo "Started: $(date)"
echo "============================================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    $CONTAINER bash -c "source activate habitat && cd / && \
    python /nav/collect_25ep.py \
        --seg_model_type yolo26 \
        --yolo26_model_path /nav/yolo26x-seg.pt \
        --exp_name yolo26_25ep \
        --dump_location $DUMP_DIR/ \
        --start_ep $START_EP --end_ep $END_EP \
        -v $VIS --sem_gpu_id 0 --evaluation local"

echo "[3/3] YOLO26 completed: $(date)"

# ---- Summary ----
echo ""
echo "============================================================"
echo "ALL MODELS COMPLETED: $(date)"
echo "============================================================"
echo ""
echo "Results location: $DUMP_DIR/"
echo "  - yolov8_25ep_results.txt"
echo "  - yolo11_25ep_results.txt"
echo "  - yolo26_25ep_results.txt"
echo ""
echo "Semantic maps:"
echo "  - $DUMP_DIR/yolov8_25ep_semantic_maps/"
echo "  - $DUMP_DIR/yolo11_25ep_semantic_maps/"
echo "  - $DUMP_DIR/yolo26_25ep_semantic_maps/"
echo ""
echo "Summaries:"
echo "  - $DUMP_DIR/yolov8_25ep_summary.json"
echo "  - $DUMP_DIR/yolo11_25ep_summary.json"
echo "  - $DUMP_DIR/yolo26_25ep_summary.json"
echo ""

# Quick comparison
echo "--- Quick Metric Comparison ---"
for model in yolov8 yolo11 yolo26; do
    summary="$DUMP_DIR/${model}_25ep_summary.json"
    # $DUMP_DIR is inside container, need to map to host path
    host_summary="/home/koray/PEANUT/data/objectnav_25ep/${model}_25ep_summary.json"
    if [ -f "$host_summary" ]; then
        python3 -c "
import json
with open('$host_summary') as f:
    s = json.load(f)
print(f\"  {s['seg_model_type']:10s}: SR={s['success_rate']*100:.1f}%  SPL={s['spl']*100:.1f}%  SoftSPL={s['softspl']*100:.1f}%  AvgDist={s['avg_distance_to_goal']:.2f}m  AvgLen={s['avg_episode_length']:.0f}\")
"
    fi
done

echo ""
echo "Done!"
