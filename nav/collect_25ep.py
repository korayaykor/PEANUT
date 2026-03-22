"""
ObjectNav evaluation on first 25 episodes with semantic map archival.

Runs the PEANUT agent through Habitat ObjectNav episodes and:
  1. Collects standard metrics (SR, SPL, SoftSPL, distance_to_goal)
  2. Saves the final semantic map (full_map tensor) after each episode
  3. Saves a color-coded semantic map visualization (PNG)
  4. Writes per-episode JSONL results

Usage (inside peanut_v2 container):
  python /nav/collect_25ep.py \
      --seg_model_type yolo \
      --seg_model_wts /nav/yolov8x-seg.pt \
      --exp_name yolov8_25ep \
      --dump_location /data/objectnav_25ep/ \
      --start_ep 0 --end_ep 25 \
      -v 0 --sem_gpu_id 0 --evaluation local
"""

import argparse
import os
import random
import habitat
import torch
import sys
import cv2
import time
import numpy as np
import json
from PIL import Image

from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names, color_palette
from agent.peanut_agent import PEANUT_Agent


# PEANUT category names for semantic map channels 4..13
CATEGORY_NAMES = [
    'chair', 'sofa', 'plant', 'bed', 'toilet',
    'tv_monitor', 'fireplace', 'bathtub', 'mirror', 'other'
]

# Map layer legends (index -> color from palette, label)
MAP_LAYERS = [
    (0, "Unknown"),
    (1, "Obstacle"),
    (2, "Explored"),
]


def add_legend(img_rgb, detected_indices):
    """Add a semantic legend bar below the image."""
    color_pal = [int(x * 255.) for x in color_palette]
    H, W = img_rgb.shape[:2]

    entries = []
    for idx, label in MAP_LAYERS:
        r, g, b = color_pal[idx*3], color_pal[idx*3+1], color_pal[idx*3+2]
        entries.append(((r, g, b), label))
    for cat_i, cat_name in enumerate(CATEGORY_NAMES):
        pal_idx = cat_i + 5
        r, g, b = color_pal[pal_idx*3], color_pal[pal_idx*3+1], color_pal[pal_idx*3+2]
        marker = " *" if (cat_i + 1) in detected_indices else ""
        entries.append(((r, g, b), cat_name + marker))

    num_cols = 7
    num_rows = (len(entries) + num_cols - 1) // num_cols
    row_h = 30
    col_w = max(W // num_cols, 160)
    legend_w = max(W, col_w * num_cols)
    legend_h = num_rows * row_h + 10
    swatch_size = 16

    legend = np.ones((legend_h, legend_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1

    for i, (color_rgb, label) in enumerate(entries):
        row = i // num_cols
        col = i % num_cols
        x = col * col_w + 8
        y = row * row_h + 8

        cv2.rectangle(legend, (x, y), (x + swatch_size, y + swatch_size),
                      (color_rgb[2], color_rgb[1], color_rgb[0]), -1)
        cv2.rectangle(legend, (x, y), (x + swatch_size, y + swatch_size),
                      (0, 0, 0), 1)

        cv2.putText(legend, label, (x + swatch_size + 5, y + swatch_size - 2),
                    font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    if legend_w != W:
        center_pad = (legend_w - W) // 2
        padded = np.ones((H, legend_w, 3), dtype=np.uint8) * 255
        padded[:, center_pad:center_pad+W] = img_rgb
        img_rgb = padded

    sep = np.ones((2, legend_w, 3), dtype=np.uint8) * 128
    result = np.vstack([img_rgb, sep, legend])
    return result


def save_semantic_map(agent, ep_idx, scene_short, target_name, save_dir):
    """
    Save the agent's full semantic map as:
      - .npy (raw tensor for later analysis)
      - .png (color-coded visualization)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Access the full map from agent internals
    # agent -> agent_helper -> agent_states -> full_map
    helper = agent.agent_helper
    state = helper.agent_states
    full_map = state.full_map.cpu().numpy()  # shape: (C, H, W)

    # Save raw map
    prefix = f"ep{ep_idx:03d}_{scene_short}_{target_name}"
    np.save(os.path.join(save_dir, f"{prefix}_fullmap.npy"), full_map)

    # Build color visualization of semantic channels
    # Channels: 0=obstacle, 1=explored, 2=current_pos, 3=trajectory, 4..13=semantic categories
    num_sem = full_map.shape[0] - 4  # semantic channels start at index 4
    sem_channels = full_map[4:4+num_sem]  # (num_sem, H, W)

    # Argmax over semantic channels (0 = background/no detection)
    sem_max = sem_channels.max(axis=0)
    sem_label = sem_channels.argmax(axis=0) + 1  # 1-indexed categories
    sem_label[sem_max < 0.01] = 0  # background

    # Build a combined map: 0=unknown, 1=obstacle, 2=explored, 3=visited, 4..=categories
    obstacle = (full_map[0] > 0.5).astype(np.uint8)
    explored = (full_map[1] > 0.5).astype(np.uint8)
    
    vis_map = np.zeros_like(sem_label, dtype=np.uint8)
    vis_map[explored == 1] = 2  # explored free space
    vis_map[obstacle == 1] = 1  # obstacles
    vis_map[sem_label > 0] = sem_label[sem_label > 0] + 4  # semantic categories offset by 4
    
    # Use the PEANUT color palette
    color_pal = [int(x * 255.) for x in color_palette]
    sem_vis = Image.new("P", (vis_map.shape[1], vis_map.shape[0]))
    sem_vis.putpalette(color_pal)
    sem_vis.putdata(vis_map.flatten().astype(np.uint8))
    sem_vis = sem_vis.convert("RGB")
    sem_vis = np.flipud(np.array(sem_vis))
    
    # Find which categories are detected
    detected_cats = set(np.unique(sem_label)) - {0}

    # Add legend to full map
    sem_vis_with_legend = add_legend(sem_vis, detected_cats)
    cv2.imwrite(
        os.path.join(save_dir, f"{prefix}_semmap.png"),
        sem_vis_with_legend[:, :, ::-1]  # RGB -> BGR for cv2
    )
    
    # Also save a cropped version (only the explored area)
    explored_full = full_map[1] > 0.01
    rows = np.any(explored_full, axis=1)
    cols = np.any(explored_full, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Add padding
        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(full_map.shape[1] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(full_map.shape[2] - 1, cmax + pad)
        
        cropped = sem_vis[rmin:rmax+1, cmin:cmax+1]
        cropped_with_legend = add_legend(cropped, detected_cats)
        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_semmap_cropped.png"),
            cropped_with_legend[:, :, ::-1]
        )
    
    print(f"  Saved semantic map: {prefix} (shape={full_map.shape})")
    return full_map.shape


def main():
    args = get_args()
    args.only_explore = 0  # We want navigation, not just exploration

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    config.freeze()

    hab_env = Env(config=config)
    nav_agent = PEANUT_Agent(args=args, task_config=config)
    
    print("=" * 60)
    print(f"ObjectNav Evaluation: {args.exp_name}")
    print(f"Model type: {args.seg_model_type}")
    print(f"Split: {config.DATASET.SPLIT}")
    print(f"Total episodes: {len(hab_env.episodes)}")
    print(f"Episode range: {args.start_ep} -> {args.end_ep}")
    print(f"Dump location: {args.dump_location}")
    print("=" * 60)
    sys.stdout.flush()

    # Prepare output directories
    os.makedirs(args.dump_location, exist_ok=True)
    results_file = os.path.join(args.dump_location, f"{args.exp_name}_results.txt")
    semmap_dir = os.path.join(args.dump_location, f"{args.exp_name}_semantic_maps")
    os.makedirs(semmap_dir, exist_ok=True)
    
    # Clear results file
    open(results_file, 'w').close()

    num_episodes = len(hab_env.episodes)
    start = args.start_ep
    end = args.end_ep if args.end_ep > 0 else num_episodes
    end = min(end, num_episodes)

    sucs, spls, softspls = [], [], []
    dists = []
    ep_lens = []
    times = []

    ep_i = 0
    while ep_i < min(num_episodes, end):
        observations = hab_env.reset()
        nav_agent.reset()

        if ep_i >= start and ep_i < end:
            target_name = hm3d_names[observations['objectgoal'][0]]
            scene_id = hab_env._current_episode.scene_id
            scene_short = scene_id.split('/')[-2] if '/' in scene_id else scene_id
            
            print('-' * 60)
            print(f'Episode {ep_i}/{end-1} | Target: {target_name} | Scene: {scene_short}')
            sys.stdout.flush()

            step_i = 0
            ep_start_time = time.time()

            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)

                if step_i % 100 == 0:
                    print(f'  step {step_i}...')
                    sys.stdout.flush()

                step_i += 1

            ep_elapsed_time = time.time() - ep_start_time
            
            # Get metrics
            metrics = hab_env.get_metrics()
            
            suc = metrics.get('success', 0.0)
            spl = metrics.get('spl', 0.0)
            sspl = metrics.get('softspl', 0.0)
            dtg = metrics.get('distance_to_goal', -1)
            
            sucs.append(suc)
            spls.append(spl)
            softspls.append(sspl)
            dists.append(dtg)
            ep_lens.append(step_i)
            times.append(ep_elapsed_time)

            status = "✓ SUCCESS" if suc > 0 else "✗ FAIL"
            print(f'  {status} | steps={step_i} | time={ep_elapsed_time:.1f}s | '
                  f'dist={dtg:.2f}m | spl={spl:.3f}')
            
            # Running averages
            n = len(sucs)
            print(f'  Running avg ({n} eps): SR={np.mean(sucs)*100:.1f}% | '
                  f'SPL={np.mean(spls)*100:.1f}% | SoftSPL={np.mean(softspls)*100:.1f}%')
            sys.stdout.flush()
            
            # Save semantic map
            try:
                save_semantic_map(nav_agent, ep_i, scene_short, target_name, semmap_dir)
            except Exception as e:
                print(f"  WARNING: Failed to save semantic map: {e}")

            # Write result record
            record = {
                'episode': ep_i,
                'scene_id': scene_id,
                'scene_short': scene_short,
                'target': target_name,
                'episode_length': step_i,
                'time': ep_elapsed_time,
                'distance_to_goal': dtg,
                'success': suc,
                'spl': spl,
                'softspl': sspl,
            }
            try:
                with open(results_file, 'a') as _f:
                    _f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"  WARNING: Failed to write results: {e}")

        ep_i += 1

    # Final summary
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {args.exp_name}")
    print(f"Model: {args.seg_model_type}")
    print(f"Episodes evaluated: {len(sucs)}")
    print(f"  Success Rate (SR):  {np.mean(sucs)*100:.1f}%")
    print(f"  SPL:                {np.mean(spls)*100:.1f}%")
    print(f"  Soft SPL:           {np.mean(softspls)*100:.1f}%")
    print(f"  Avg Distance:       {np.mean(dists):.2f}m")
    print(f"  Avg Episode Length: {np.mean(ep_lens):.0f} steps")
    print(f"  Avg Time/Episode:   {np.mean(times):.1f}s")
    print(f"  Total Time:         {sum(times):.0f}s ({sum(times)/60:.1f}min)")
    print(f"\nResults saved to: {results_file}")
    print(f"Semantic maps saved to: {semmap_dir}")
    print("=" * 60)
    
    # Also save a summary JSON
    summary = {
        'exp_name': args.exp_name,
        'seg_model_type': args.seg_model_type,
        'num_episodes': len(sucs),
        'start_ep': start,
        'end_ep': end,
        'success_rate': float(np.mean(sucs)),
        'spl': float(np.mean(spls)),
        'softspl': float(np.mean(softspls)),
        'avg_distance_to_goal': float(np.mean(dists)),
        'avg_episode_length': float(np.mean(ep_lens)),
        'avg_time_per_episode': float(np.mean(times)),
        'total_time': float(sum(times)),
        'per_episode': [
            {
                'episode': start + i,
                'success': float(sucs[i]),
                'spl': float(spls[i]),
                'softspl': float(softspls[i]),
                'distance_to_goal': float(dists[i]),
                'episode_length': int(ep_lens[i]),
                'time': float(times[i]),
            }
            for i in range(len(sucs))
        ]
    }
    summary_file = os.path.join(args.dump_location, f"{args.exp_name}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
