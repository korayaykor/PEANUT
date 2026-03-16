"""
Create a 2×3 comparison image of all 6 segmentation models.
Each sub-image is the majority-vote semantic map with legend.
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──
RESULTS = {
    "MaskRCNN Trained\n(9 PEANUT cats)": "/nav/data/tmp_coco80_maskrcnn/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
    "MaskRCNN Pretrained\n(80 COCO cats)": "/nav/data/tmp_coco80_maskrcnn_pretrained/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
    "Cascade X-152\n(80 COCO cats)": "/nav/data/tmp_coco80_cascade/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
    "YOLOv8x-seg\n(80 COCO cats)": "/nav/data/tmp_coco80_yolo/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
    "YOLOv11x-seg\n(80 COCO cats)": "/nav/data/tmp_coco80_yolo11/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
    "YOLOv26x-seg\n(80 COCO cats)": "/nav/data/tmp_coco80_yolo26/coco80_replay_5LpN3gDmAk7_1_chair/semantic_map_coco80.png",
}

# ── Stats (from logs) ──
STATS = {
    "MaskRCNN Trained\n(9 PEANUT cats)": "1,138 cells | 5 cats",
    "MaskRCNN Pretrained\n(80 COCO cats)": "13,811 cells | 21 cats",
    "Cascade X-152\n(80 COCO cats)": "12,584 cells | 25 cats",
    "YOLOv8x-seg\n(80 COCO cats)": "14,987 cells | 26 cats",
    "YOLOv11x-seg\n(80 COCO cats)": "9,081 cells | 26 cats",
    "YOLOv26x-seg\n(80 COCO cats)": "10,310 cells | 26 cats",
}

def main():
    """Load 6 model semantic map PNGs and arrange them in a 2x3 comparison figure."""
    fig = plt.figure(figsize=(36, 24))
    fig.suptitle(
        "Semantic Map — 6 Model Comparison\n"
        "Scene: 5LpN3gDmAk7_1 | 1159 frames | Majority-vote + 5×5 spatial smoothing",
        fontsize=20, fontweight='bold', y=0.98
    )
    
    gs = gridspec.GridSpec(2, 3, wspace=0.05, hspace=0.12,
                           left=0.02, right=0.98, top=0.93, bottom=0.02)
    
    for idx, (title, path) in enumerate(RESULTS.items()):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(np.array(img))
        else:
            ax.text(0.5, 0.5, f"NOT FOUND:\n{path}", 
                    ha='center', va='center', fontsize=12, color='red',
                    transform=ax.transAxes)
        
        stats = STATS.get(title, "")
        ax.set_title(f"{title}\n{stats}", fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')
    
    out_path = "/nav/data/6model_comparison_coco80_vote.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
