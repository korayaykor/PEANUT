       #!/usr/bin/env bash
# ============================================================
#  run_comparison.sh  –  Run Mask R-CNN and YOLOv8 evaluations
#                        inside the crazy_leavitt container.
#
#  Usage (from host):
#    bash run_comparison.sh [NUM_EPISODES] [START_EP]
#
#  Defaults: 10 episodes starting from 0
# ============================================================

set -e

CONTAINER="26959a4d38b8"
NUM_EP="${1:-10}"
START_EP="${2:-0}"

echo "============================================"
echo "  Running comparison: ${NUM_EP} episodes"
echo "  Container: ${CONTAINER}"
echo "============================================"

# ----------------------------------------------------------
# 1) Copy updated collect.py and compare_results.py into container
# ----------------------------------------------------------
echo "[1/5] Copying updated scripts into container..."
docker cp /home/koray/PEANUT/nav/collect.py      ${CONTAINER}:/nav/collect.py
docker cp /home/koray/PEANUT/nav/compare_results.py ${CONTAINER}:/nav/compare_results.py

# ----------------------------------------------------------
# 2) Run Mask R-CNN evaluation
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "  [2/5] Running Mask R-CNN evaluation"
echo "============================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect.py \
        --seg_model_type maskrcnn \
        --seg_model_wts /nav/agent/utils/mask_rcnn_R_101_cat9.pth \
        --pred_model_wts /nav/pred_model_wts.pth \
        --pred_model_cfg /nav/pred_model_cfg.py \
        --exp_name maskrcnn_seg \
        --dump_location /nav/data/tmp/ \
        --start_ep ${START_EP} \
        --end_ep ${NUM_EP} \
        -v 0 \
        --evaluation local
"

echo ""
echo "[2/5] Mask R-CNN evaluation complete."

# ----------------------------------------------------------
# 3) Run YOLOv8 evaluation
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "  [3/5] Running YOLOv8 evaluation"
echo "============================================"

docker exec -e CHALLENGE_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml \
    ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/collect.py \
        --seg_model_type yolo \
        --seg_model_wts /nav/yolov8x-seg.pt \
        --pred_model_wts /nav/pred_model_wts.pth \
        --pred_model_cfg /nav/pred_model_cfg.py \
        --exp_name yolo_seg \
        --dump_location /nav/data/tmp/ \
        --start_ep ${START_EP} \
        --end_ep ${NUM_EP} \
        -v 0 \
        --evaluation local
"

echo ""
echo "[3/5] YOLOv8 evaluation complete."

# ----------------------------------------------------------
# 4) Generate comparison chart
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "  [4/5] Generating comparison chart"
echo "============================================"

docker exec ${CONTAINER} bash -c "
    source activate habitat && \
    cd / && \
    python /nav/compare_results.py \
        --maskrcnn_results /nav/data/tmp/maskrcnn_seg_results.txt \
        --yolo_results /nav/data/tmp/yolo_seg_results.txt \
        --output /nav/data/tmp/comparison_chart.png
"

# ----------------------------------------------------------
# 5) Copy results back to host
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "  [5/5] Copying results to host"
echo "============================================"

mkdir -p /home/koray/PEANUT/nav/data/tmp/
docker cp ${CONTAINER}:/nav/data/tmp/maskrcnn_seg_results.txt /home/koray/PEANUT/nav/data/tmp/
docker cp ${CONTAINER}:/nav/data/tmp/yolo_seg_results.txt     /home/koray/PEANUT/nav/data/tmp/
docker cp ${CONTAINER}:/nav/data/tmp/comparison_chart.png     /home/koray/PEANUT/nav/data/tmp/

echo ""
echo "============================================"
echo "  DONE!"
echo "  Results:  /home/koray/PEANUT/nav/data/tmp/"
echo "    - maskrcnn_seg_results.txt"
echo "    - yolo_seg_results.txt"
echo "    - comparison_chart.png"
echo "============================================"
