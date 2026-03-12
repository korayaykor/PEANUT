#!/usr/bin/env python
"""
Collect RGB frames from Habitat with GT semantic labels for YOLO fine-tuning.

Paper-scale collection:
  - 80 train scenes x ~1000 imgs/scene = 80,000 train images
  - 20 val scenes   x ~1000 imgs/scene = 20,000 val images

Parallel mode: multiple worker processes, each assigned a subset of scenes
and a specific GPU.  Workers operate fully independently (no shared state).

Usage:
    # Parallel (2 GPUs, 4 workers per GPU = 8 total workers)
    python collect_yolo_data.py --split train --workers_per_gpu 4 \
        --gpus 0,1 --save_dir /data/yolo_dataset_v2

    # Single worker (debug)
    python collect_yolo_data.py --split train --workers_per_gpu 1 \
        --gpus 0 --save_dir /data/yolo_dataset_v2

Categories (PEANUT 9-class):
    0: chair, 1: sofa, 2: plant, 3: bed, 4: toilet,
    5: tv_monitor, 6: fireplace, 7: bathtub, 8: mirror
"""

import argparse
import gzip
import glob
import json
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue

import cv2
import numpy as np

# Add nav directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_nav_dir = os.path.dirname(_script_dir)
if not os.path.isdir(os.path.join(_nav_dir, 'agent')):
    _nav_dir = os.path.join(_nav_dir, 'nav')
sys.path.insert(0, _nav_dir)


# ---- PEANUT 9-class category definitions ----
PEANUT_CATEGORIES = [
    'chair',       # 0
    'sofa',        # 1
    'plant',       # 2
    'bed',         # 3
    'toilet',      # 4
    'tv_monitor',  # 5
    'fireplace',   # 6
    'bathtub',     # 7
    'mirror',      # 8
]

# Map HM3D semantic category names -> PEANUT class index
HM3D_TO_PEANUT = {
    # 0: chair
    'chair': 0, 'dining chair': 0, 'desk chair': 0, 'computer chair': 0,
    'office chair': 0, 'lounge chair': 0, 'rocking chair': 0,
    'folding chair': 0, 'high chair': 0, 'stool': 0, 'bar stool': 0,
    'swivel chair': 0, 'bench': 0, 'beanbag': 0, 'bean bag': 0,
    # 1: sofa
    'sofa': 1, 'couch': 1, 'sofa chair': 1, 'sofa seat': 1,
    'sofa set': 1, 'circular sofa': 1, 'sectional sofa': 1,
    'loveseat': 1, 'armchair': 1, 'recliner': 1, 'futon': 1,
    # 2: plant
    'plant': 2, 'potted plant': 2, 'flower': 2, 'flower pot': 2,
    'flower vase': 2, 'indoor plant': 2, 'tree': 2, 'vase': 2,
    # 3: bed
    'bed': 3, 'bunk bed': 3, 'baby bed': 3, 'crib': 3,
    # 4: toilet
    'toilet': 4, 'urinal': 4,
    # 5: tv_monitor
    'tv': 5, 'tv monitor': 5, 'tv_monitor': 5, 'television': 5,
    'monitor': 5, 'computer monitor': 5, 'screen': 5,
    'wall tv': 5, 'computer': 5,
    # 6: fireplace
    'fireplace': 6,
    # 7: bathtub
    'bathtub': 7, 'bath': 7, 'bath tub': 7, 'hot tub': 7, 'jacuzzi': 7,
    # 8: mirror
    'mirror': 8, 'bathroom mirror': 8, 'wall mirror': 8,
}


# -- Helpers -------------------------------------------------------------------

def build_obj_id_to_peanut(scene):
    """Build {semantic_index: peanut_class} from scene semantic annotations.

    Habitat's semantic sensor returns per-pixel *list indices* into
    ``scene.objects``, NOT the string ``obj.id``.  So we key by the
    integer index.
    """
    mapping = {}
    for idx, obj in enumerate(scene.objects):
        if obj is None or not hasattr(obj, 'category') or obj.category is None:
            continue
        name = obj.category.name().lower().strip()
        if name in HM3D_TO_PEANUT:
            mapping[idx] = HM3D_TO_PEANUT[name]
    return mapping


def mask_to_polygon(mask, tolerance=2.0):
    """Binary mask -> YOLO polygon (normalised 0-1), or None."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, tolerance, True)
    if len(contour) < 3:
        return None
    H, W = mask.shape
    return [(np.clip(pt[0][0] / W, 0., 1.),
             np.clip(pt[0][1] / H, 0., 1.)) for pt in contour]


def process_gt_frame(semantic_obs, obj_id_to_peanut, min_mask_area=100):
    """Extract YOLO-format labels from a GT semantic frame."""
    labels = []
    for obj_id in np.unique(semantic_obs):
        if obj_id not in obj_id_to_peanut:
            continue
        mask = (semantic_obs == obj_id).astype(np.uint8)
        if mask.sum() < min_mask_area:
            continue
        poly = mask_to_polygon(mask)
        if poly is not None:
            labels.append((obj_id_to_peanut[obj_id], poly))
    return labels


def save_yolo_label(label_path, labels):
    """Save labels in YOLO segmentation format."""
    with open(label_path, 'w') as f:
        for cls_id, poly in labels:
            coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly)
            f.write(f'{cls_id} {coords}\n')


def get_scene_list(split):
    """Return sorted list of scene IDs available for a given split."""
    content_dir = f'/habitat-challenge-data/objectgoal_hm3d/{split}/content/'
    files = glob.glob(os.path.join(content_dir, '*.json.gz'))
    scenes = sorted(os.path.basename(f).replace('.json.gz', '') for f in files)
    return scenes


# -- Per-scene worker ----------------------------------------------------------

def collect_scene(scene_id, split, gpu_id, save_dir,
                  target_images, max_steps_per_ep,
                  min_mask_area, seed, result_queue):
    """Collect up to *target_images* images from a single scene.

    Runs in its own process.  Creates its own Habitat env on the given GPU.
    """
    import habitat
    from habitat.core.env import Env

    tag = f'[GPU{gpu_id} {scene_id}]'

    try:
        scene_dataset_map = {
            'train': '/habitat-challenge-data/data/scene_datasets/hm3d/'
                     'hm3d_annotated_train_basis.scene_dataset_config.json',
            'val':   '/habitat-challenge-data/data/scene_datasets/hm3d/'
                     'hm3d_annotated_val_basis.scene_dataset_config.json',
        }

        config = habitat.get_config('/challenge_objectnav2022.local.rgbd.yaml')
        config.defrost()
        config.SEED = seed
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        config.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')
        config.SIMULATOR.SCENE_DATASET = scene_dataset_map[split]
        config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 99999
        config.DATASET.SPLIT = split
        config.freeze()

        env = Env(config=config)

        img_dir = os.path.join(save_dir, 'images', split)
        lbl_dir = os.path.join(save_dir, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        total_images = 0
        total_labels = 0
        per_class = [0] * len(PEANUT_CATEGORIES)
        ep_idx = 0
        prev_scene = None
        obj_id_to_peanut = {}
        max_episodes = len(env.episodes)

        while total_images < target_images:
            if ep_idx >= max_episodes:
                ep_idx = 0  # wrap around

            obs = env.reset()
            cur_scene = env._current_episode.scene_id.split('/')[-1].split('.')[0]

            if cur_scene != prev_scene:
                scene_ann = env.sim.semantic_annotations()
                obj_id_to_peanut = build_obj_id_to_peanut(scene_ann)
                prev_scene = cur_scene

            if not obj_id_to_peanut:
                ep_idx += 1
                continue

            for step in range(max_steps_per_ep):
                if env.episode_over:
                    break
                action = np.random.choice([1, 2, 3])
                obs = env.step(action)

                rgb = obs['rgb']
                semantic = obs['semantic']
                labels = process_gt_frame(semantic, obj_id_to_peanut,
                                          min_mask_area)
                if not labels:
                    continue

                fname = f'{scene_id}_ep{ep_idx:05d}_s{step:04d}'
                cv2.imwrite(os.path.join(img_dir, f'{fname}.jpg'),
                            rgb[:, :, ::-1])
                save_yolo_label(os.path.join(lbl_dir, f'{fname}.txt'), labels)

                total_images += 1
                total_labels += len(labels)
                for cid, _ in labels:
                    per_class[cid] += 1

                if total_images >= target_images:
                    break

            ep_idx += 1

            if ep_idx % 20 == 0:
                print(f'{tag}  ep {ep_idx} | {total_images}/{target_images} imgs',
                      flush=True)

        env.close()

        result_queue.put({
            'scene': scene_id, 'images': total_images, 'labels': total_labels,
            'per_class': per_class, 'status': 'ok',
        })
        print(f'{tag}  DONE  {total_images} images, {total_labels} labels',
              flush=True)

    except Exception as exc:
        traceback.print_exc()
        result_queue.put({
            'scene': scene_id, 'images': 0, 'labels': 0,
            'per_class': [0] * len(PEANUT_CATEGORIES), 'status': f'ERROR: {exc}',
        })


# -- Orchestrator --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Paper-scale GT data collection for YOLO fine-tuning')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'])
    parser.add_argument('--save_dir', type=str, default='/data/yolo_dataset_v2')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--workers_per_gpu', type=int, default=4,
                        help='Parallel Habitat workers per GPU')
    parser.add_argument('--target_per_scene', type=int, default=1000,
                        help='Target images per scene (paper=1000)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode (paper=500)')
    parser.add_argument('--min_mask_area', type=int, default=100,
                        help='Min mask area in pixels')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    total_workers = len(gpu_ids) * args.workers_per_gpu

    scenes = get_scene_list(args.split)
    print(f'Split          : {args.split}')
    print(f'Scenes found   : {len(scenes)}')
    print(f'Target/scene   : {args.target_per_scene}')
    print(f'Total target   : {len(scenes) * args.target_per_scene}')
    print(f'GPUs           : {gpu_ids}')
    print(f'Workers/GPU    : {args.workers_per_gpu}')
    print(f'Total workers  : {total_workers}')
    print(f'Max steps/ep   : {args.max_steps}')
    print()

    os.makedirs(os.path.join(args.save_dir, 'images', args.split), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'labels', args.split), exist_ok=True)

    # ---- Build scene -> process list ----
    result_queue = Queue()
    all_procs = []

    for idx, scene_id in enumerate(scenes):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        seed = args.seed + idx * 1000

        p = Process(target=collect_scene, args=(
            scene_id, args.split, gpu_id, args.save_dir,
            args.target_per_scene, args.max_steps,
            args.min_mask_area, seed, result_queue,
        ))
        all_procs.append(p)

    # ---- Launch in batches of total_workers ----
    active = []
    pending = list(all_procs)
    finished = []

    t0 = time.time()
    print(f'Launching {len(pending)} scene workers '
          f'(max {total_workers} concurrent)...\n')

    while pending or active:
        # Start new workers up to limit
        while pending and len(active) < total_workers:
            proc = pending.pop(0)
            proc.start()
            active.append(proc)

        # Check for finished
        still_active = []
        for proc in active:
            proc.join(timeout=0.5)
            if proc.is_alive():
                still_active.append(proc)
        active = still_active

        # Drain results
        while not result_queue.empty():
            finished.append(result_queue.get_nowait())

        elapsed = time.time() - t0
        done = len(finished)
        total = len(scenes)
        if done > 0:
            eta = elapsed / done * (total - done)
            sys.stdout.write(
                f'\r  Progress: {done}/{total} scenes  '
                f'[{elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min ETA]   ')
            sys.stdout.flush()

        time.sleep(1.0)

    # Final drain
    while not result_queue.empty():
        finished.append(result_queue.get_nowait())

    elapsed = time.time() - t0
    print(f'\n\nAll workers finished in {elapsed/60:.1f} minutes.')

    # ---- Summary ----
    grand_imgs = sum(r['images'] for r in finished)
    grand_lbls = sum(r['labels'] for r in finished)
    grand_cls = [0] * len(PEANUT_CATEGORIES)
    for r in finished:
        for i, c in enumerate(r['per_class']):
            grand_cls[i] += c

    errors = [r for r in finished if r['status'] != 'ok']

    print(f'\n{"="*60}')
    print(f'  Collection Summary  ({args.split})')
    print(f'{"="*60}')
    print(f'  Total images : {grand_imgs}')
    print(f'  Total labels : {grand_lbls}')
    print(f'  Scenes done  : {len(finished)}/{len(scenes)}')
    if errors:
        print(f'  ERRORS       : {len(errors)}')
        for e in errors:
            print(f'    {e["scene"]}: {e["status"]}')
    print(f'\n  Per-class counts:')
    for i, name in enumerate(PEANUT_CATEGORIES):
        print(f'    {i}: {name:12s} -- {grand_cls[i]:>6d}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
