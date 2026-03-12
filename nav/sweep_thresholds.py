"""
Sweep confidence and cat_pred_threshold for YOLO v8, v11, v26.
For each combination, run full 1159-frame replay (no frame skipping)
and collect stats. Output a summary table at the end.

Usage (inside container):
  cd /nav
  python sweep_thresholds.py
"""

import argparse
import os
import sys
import time
import json
import copy
import gc

import numpy as np
import torch

_nav_dir = os.path.dirname(os.path.abspath(__file__))
if _nav_dir not in sys.path:
    sys.path.insert(0, _nav_dir)


# ── Confidence values to sweep ──
CONF_VALUES = [0.05, 0.10, 0.20, 0.50]

# ── cat_pred_threshold values to sweep ──
CAT_THRESH_VALUES = [3.0, 3.0, 10.00, 5.0, 10.0]

# ── Models ──
MODELS = [
    ("yolo",   "yolov8x-seg.pt"),
    ("yolo11", "yolo11x-seg.pt"),
    ("yolo26", "yolo26x-seg.pt"),
]

SCENE_DIR = "/nav/vlmaps_data/5LpN3gDmAk7_1"
TARGET = "chair"


def run_one_config(seg_type, model_path, conf, cat_thresh):
    """Run a single replay and return stats dict. Returns None on failure."""

    # We need to reimport each time because agent state is stateful
    # Clear GPU memory first
    torch.cuda.empty_cache()
    gc.collect()

    from arguments import get_args

    # Build args
    sys.argv = [
        "sweep",
        "--seg_model_type", seg_type,
        "--visualize", "0",
        "--only_explore", "1",
        "-d", f"/nav/data/sweep_tmp/",
    ]
    args = get_args()

    # Override key parameters
    args.num_sem_categories = 81
    args.only_explore = 1
    args.hfov = 90.0
    args.camera_height = 1.5
    args.min_depth = 0.5
    args.max_depth = 5.0
    args.env_frame_width = 640
    args.env_frame_height = 480
    args.cat_pred_threshold = cat_thresh

    # YOLO-specific
    args._coco80_model_path = model_path
    args._coco80_conf = conf

    # Fix seg_model_wts for agent_helper init
    if seg_type == 'yolo':
        args.seg_model_wts = model_path

    scene_name = os.path.basename(SCENE_DIR.rstrip("/"))
    out_dir = f"/nav/data/sweep_tmp/{seg_type}_c{conf}_t{cat_thresh}"
    os.makedirs(out_dir, exist_ok=True)
    args.dump_location = out_dir
    args.exp_name = f"sweep_{seg_type}"
    args.timestep_limit = 1200

    # Import replay components
    from vlmaps_dataloader_coco80 import (
        VLMapsSceneLoader, make_observation, SemanticPredYOLO_COCO80,
        COCO_80_NAMES
    )
    from constants import hm3d_names

    name_to_id = {v: k for k, v in hm3d_names.items()}
    objectgoal_id = name_to_id[TARGET]

    loader = VLMapsSceneLoader(SCENE_DIR)
    frame_indices = list(range(loader.num_frames))  # ALL frames, no skipping

    # Mock task config
    class _Cfg:
        pass
    task_cfg = _Cfg()
    task_cfg.TASK = _Cfg()
    task_cfg.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    # Create agent
    from agent.peanut_agent import PEANUT_Agent
    agent = PEANUT_Agent(args=args, task_config=task_cfg)
    agent.reset()

    # Replace seg model
    coco80_seg = SemanticPredYOLO_COCO80(args)
    agent.agent_helper.seg_model = coco80_seg
    agent.agent_helper.seg_model.n_cats = 80

    # Replay
    t0 = time.time()
    for step_i, fidx in enumerate(frame_indices):
        rgb, depth_m, position, rotation = loader.load_frame(fidx)
        gps, compass = loader.get_gps_compass(fidx)
        obs = make_observation(rgb, depth_m, gps, compass, objectgoal_id,
                               target_h=args.env_frame_height,
                               target_w=args.env_frame_width,
                               min_depth=args.min_depth,
                               max_depth=args.max_depth)
        action = agent.act(obs)
        act_id = action.get("action", -1)
        if act_id == 0:
            act_id = 1

    elapsed = time.time() - t0

    # Extract map
    agent_state = agent.agent_states
    agent_state.full_map[:, agent_state.lmb[0]:agent_state.lmb[1],
                            agent_state.lmb[2]:agent_state.lmb[3]] = agent_state.local_map
    agent_state.full_vote_map[:, agent_state.lmb[0]:agent_state.lmb[1],
                                agent_state.lmb[2]:agent_state.lmb[3]] = agent_state.local_vote_map

    full_map = agent_state.full_map.cpu().numpy()
    full_vote = agent_state.full_vote_map.cpu().numpy()

    n_sem = 80
    explored_map = full_map[1]
    semantic_channels = full_map[4:4+n_sem]
    vote_channels = full_vote[:n_sem]

    # Crop to explored region
    explored_mask = explored_map > 0.5
    if explored_mask.any():
        rows = np.any(explored_mask, axis=1)
        cols = np.any(explored_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(full_map.shape[1], rmax + pad + 1)
        cmin = max(0, cmin - pad)
        cmax = min(full_map.shape[2], cmax + pad + 1)
    else:
        rmin, rmax = 0, full_map.shape[1]
        cmin, cmax = 0, full_map.shape[2]

    sem_crop = semantic_channels[:, rmin:rmax, cmin:cmax]
    vote_crop = vote_channels[:, rmin:rmax, cmin:cmax]
    explored_crop = explored_map[rmin:rmax, cmin:cmax]

    h, w = sem_crop.shape[1], sem_crop.shape[2]

    # Majority vote
    has_sem = vote_crop.max(axis=0) > 0
    vote_winner = np.argmax(vote_crop, axis=0)

    # Spatial mode filter
    from scipy.ndimage import generic_filter

    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[has_sem] = vote_winner[has_sem]

    def _spatial_mode(values):
        valid = values[values >= 0]
        if len(valid) == 0:
            return -1
        counts = np.bincount(valid.astype(np.int32), minlength=n_sem)
        return int(np.argmax(counts))

    label_smoothed = generic_filter(
        label_map.astype(np.float64), _spatial_mode, size=5, mode='constant', cval=-1
    ).astype(np.int32)
    label_smoothed[~has_sem] = -1

    # Count stats
    total_sem_cells = int(has_sem.sum())
    explored_cells = int((explored_crop > 0.5).sum())

    # Count per-category cells from smoothed result
    cat_counts = {}
    valid_mask = label_smoothed >= 0
    if valid_mask.any():
        for cat_i in np.unique(label_smoothed[valid_mask]):
            cat_counts[int(cat_i)] = int((label_smoothed == cat_i).sum())

    n_cats_detected = len(cat_counts)
    total_final = sum(cat_counts.values())

    # Semantic coverage: what fraction of explored area is labeled
    coverage = total_final / max(explored_cells, 1)

    # Cleanup
    del agent, agent_state, coco80_seg, full_map, full_vote
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "seg_type": seg_type,
        "conf": conf,
        "cat_thresh": cat_thresh,
        "total_cells": total_final,
        "n_cats": n_cats_detected,
        "explored": explored_cells,
        "coverage": round(coverage, 4),
        "time": round(elapsed, 1),
        "categories": {COCO_80_NAMES[k]: v for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])},
    }
    return result


def main():
    all_results = []

    for seg_type, model_path in MODELS:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {seg_type} ({model_path})")
        print(f"{'#'*70}")

        for conf in CONF_VALUES:
            for cat_thresh in CAT_THRESH_VALUES:
                tag = f"{seg_type} | conf={conf:.2f} | cat_thresh={cat_thresh:.1f}"
                print(f"\n>>> {tag}")
                sys.stdout.flush()

                try:
                    result = run_one_config(seg_type, model_path, conf, cat_thresh)
                    if result:
                        all_results.append(result)
                        print(f"    → {result['total_cells']:,} cells | "
                              f"{result['n_cats']} cats | "
                              f"coverage={result['coverage']:.3f} | "
                              f"{result['time']:.0f}s")
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")
                    import traceback; traceback.print_exc()

                sys.stdout.flush()

    # ── Summary ──
    print(f"\n\n{'='*90}")
    print(f"  SWEEP RESULTS SUMMARY — {len(all_results)} runs")
    print(f"{'='*90}")
    print(f"{'Model':<8} {'Conf':>6} {'CatTh':>6} {'Cells':>8} {'Cats':>5} {'Coverage':>9} {'Time':>6}")
    print(f"{'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*9} {'-'*6}")

    for r in sorted(all_results, key=lambda x: (x['seg_type'], -x['total_cells'])):
        print(f"{r['seg_type']:<8} {r['conf']:>6.2f} {r['cat_thresh']:>6.1f} "
              f"{r['total_cells']:>8,} {r['n_cats']:>5} {r['coverage']:>9.4f} {r['time']:>6.1f}")

    # Best per model
    print(f"\n{'='*90}")
    print(f"  BEST CONFIG PER MODEL (by total semantic cells)")
    print(f"{'='*90}")
    for seg_type, _ in MODELS:
        model_results = [r for r in all_results if r['seg_type'] == seg_type]
        if model_results:
            best = max(model_results, key=lambda r: r['total_cells'])
            print(f"\n  {seg_type}: conf={best['conf']:.2f}, cat_thresh={best['cat_thresh']:.1f}")
            print(f"    → {best['total_cells']:,} cells | {best['n_cats']} categories | coverage={best['coverage']:.4f}")
            top5 = sorted(best['categories'].items(), key=lambda x: -x[1])[:10]
            for name, cnt in top5:
                print(f"      {name:20s}: {cnt:>6,}")

    # Save JSON
    out_path = "/nav/data/sweep_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved: {out_path}")


if __name__ == "__main__":
    main()
