"""
Run all 3 YOLO models with best sweep config (conf=0.05, cat_thresh=3.0)
with outdoor false-positive filtering, then create a 1×3 comparison image.
"""
import os
import sys
import subprocess
import time
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ──
SCENE_DIR = "/nav/vlmaps_data/5LpN3gDmAk7_1"
BEST_CONF = 0.05
BEST_CAT_THRESH = 3.0
MIN_VOTES = 3

MODELS = [
    {"seg_type": "yolo",   "label": "YOLOv8x-seg",  "out": "tmp_coco80_yolo_best"},
    {"seg_type": "yolo11", "label": "YOLOv11x-seg", "out": "tmp_coco80_yolo11_best"},
    {"seg_type": "yolo26", "label": "YOLOv26x-seg", "out": "tmp_coco80_yolo26_best"},
]

def run_model(seg_type, out_name):
    """Run vlmaps_dataloader_coco80.py for one model."""
    cmd = [
        sys.executable, "/nav/vlmaps_dataloader_coco80.py",
        "--scene_dir", SCENE_DIR,
        "--target_object", "chair",
        "--filter_outdoor", "1",
        "--min_votes", str(MIN_VOTES),
        "--seg_model_type", seg_type,
        "--yolo_conf", str(BEST_CONF),
        "--cat_pred_threshold", str(BEST_CAT_THRESH),
        "--num_sem_categories", "81",
        "--only_explore", "1",
        "-d", f"/nav/data/{out_name}/",
    ]
    print(f"\n{'#'*70}")
    print(f"# Running: {seg_type} | conf={BEST_CONF} | cat_thresh={BEST_CAT_THRESH} | min_votes={MIN_VOTES}")
    print(f"# Output:  /nav/data/{out_name}/")
    print(f"{'#'*70}\n")
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"\n  [{seg_type}] Finished in {elapsed:.1f}s (exit code {result.returncode})")
    return result.returncode == 0


def make_comparison():
    """Create 1×3 comparison image from the 3 YOLO model results."""
    print(f"\n{'='*70}")
    print(f"  Creating 3-model comparison image")
    print(f"{'='*70}\n")

    fig = plt.figure(figsize=(42, 16))
    fig.suptitle(
        "YOLO Model Comparison — Best Config (conf=0.05, cat_thresh=3.0, min_votes=3)\n"
        "Scene: 5LpN3gDmAk7_1 | 1159 frames | Majority-vote + 5×5 spatial | Outdoor FP filtered | min_votes=3",
        fontsize=22, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(1, 3, wspace=0.03,
                           left=0.01, right=0.99, top=0.92, bottom=0.01)

    for idx, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, idx])
        scene_name = os.path.basename(SCENE_DIR.rstrip("/"))
        png_path = f"/nav/data/{model['out']}/coco80_replay_{scene_name}_chair/semantic_map_coco80.png"

        if os.path.exists(png_path):
            img = Image.open(png_path)
            ax.imshow(np.array(img))
            # Read stats from the npy if available
            sem_path = png_path.replace("semantic_map_coco80.png", "semantic_map_coco80.npy")
            if os.path.exists(sem_path):
                sem = np.load(sem_path)
                n_cats = int((sem.max(axis=(1, 2)) > 0.5).sum())
                total = int((sem > 0.5).any(axis=0).sum())
                stats = f"{total:,} cells | {n_cats} categories"
            else:
                stats = ""
            ax.set_title(f"{model['label']}\n{stats}", fontsize=16, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, f"NOT FOUND:\n{png_path}",
                    ha='center', va='center', fontsize=12, color='red',
                    transform=ax.transAxes)
            ax.set_title(model['label'], fontsize=16, fontweight='bold', pad=10)

        ax.axis('off')

    out_path = "/nav/data/3yolo_comparison_best.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved comparison: {out_path}")
    return out_path


def main():
    """Run all 3 YOLO models sequentially with best config, then create comparison image."""
    print(f"╔{'═'*68}╗")
    print(f"║  3-YOLO Model Comparison with Best Sweep Config                    ║")
    print(f"║  conf={BEST_CONF}, cat_thresh={BEST_CAT_THRESH}, outdoor FP filtering ON            ║")
    print(f"╚{'═'*68}╝")

    # Run each model
    results = {}
    for model in MODELS:
        ok = run_model(model["seg_type"], model["out"])
        results[model["seg_type"]] = ok
        # Clear GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"  Run Summary:")
    for seg_type, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"    {seg_type:10s}: {status}")

    # Create comparison
    if all(results.values()):
        comp_path = make_comparison()
        print(f"\n  Comparison image: {comp_path}")
    else:
        print("\n  Some models failed — skipping comparison image")

    print(f"\n{'='*70}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
