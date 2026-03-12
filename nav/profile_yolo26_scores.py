"""
Diagnostic: Profile YOLO26-seg confidence scores on Habitat-rendered images.
Runs a few episodes, captures every frame, runs YOLO26, and logs ALL detections
(no filtering) to understand score distributions per PEANUT category.
"""
import os
import sys
import json
import numpy as np
import habitat
import torch
from collections import defaultdict

sys.path.insert(0, '/nav')
from arguments import get_args


def main():
    """Run YOLO26 on Habitat frames and log all detections with confidence scores."""
    from ultralytics import YOLO

    args = get_args()

    # Load YOLO26 model
    model_path = getattr(args, 'yolo26_model_path', 'yolo26x-seg.pt')
    model = YOLO(model_path)
    device = f'cuda:{args.sem_gpu_id}'

    # COCO class names we care about
    names = model.names
    peanut_coco_map = {
        56: ('chair', 0),
        57: ('couch/sofa', 1),
        58: ('potted plant', 2),
        59: ('bed', 3),
        61: ('toilet', 4),
        62: ('tv', 5),
        60: ('dining table/fireplace', 6),
        69: ('oven/bathtub', 7),
        71: ('sink/mirror', 8),
    }

    # Setup habitat env
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    config.freeze()

    from habitat.core.env import Env
    hab_env = Env(config=config)

    # Collect scores across episodes
    all_scores = defaultdict(list)  # peanut_name -> [conf_scores]
    det_per_episode = defaultdict(lambda: defaultdict(int))

    num_episodes = min(20, len(hab_env.episodes))  # 20 episodes across different scenes
    
    for ep_i in range(num_episodes):
        obs = hab_env.reset()
        ep = hab_env.current_episode
        target = ep.object_category if hasattr(ep, 'object_category') else 'unknown'
        scene = ep.scene_id.split('/')[-2] if '/' in ep.scene_id else ep.scene_id
        
        print(f"\n=== Episode {ep_i} | Target: {target} | Scene: {scene} ===")
        
        step = 0
        done = False
        ep_detections = defaultdict(list)
        
        while not done and step < 200:  # max 200 steps per episode for profiling
            rgb = obs['rgb']  # (H, W, 3)
            
            # Run YOLO26 with very low conf to capture ALL detections
            results = model.predict(
                rgb,
                conf=0.001,  # extremely low to see everything
                device=device,
                verbose=False,
                retina_masks=True,
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    conf = boxes.conf[j].item()
                    
                    if cls_id in peanut_coco_map:
                        cat_name, peanut_idx = peanut_coco_map[cls_id]
                        all_scores[cat_name].append(conf)
                        ep_detections[cat_name].append(conf)
                        
                        if step % 50 == 0:  # print sample detections
                            # Get mask area if available
                            mask_area = 0
                            if results[0].masks is not None:
                                mask = results[0].masks.data[j]
                                mask_area = mask.sum().item()
                            print(f"  step {step}: {cat_name} conf={conf:.4f} mask_area={mask_area:.0f}px")
            
            # Take random action to explore
            action = np.random.randint(1, 4)  # forward, left, right
            obs = hab_env.step(action)
            done = hab_env.episode_over
            step += 1
        
        # Episode summary
        for cat, scores in ep_detections.items():
            if scores:
                print(f"  {cat}: {len(scores)} dets, "
                      f"min={min(scores):.4f}, max={max(scores):.4f}, "
                      f"mean={np.mean(scores):.4f}, median={np.median(scores):.4f}")

    hab_env.close()
    
    # Global summary
    print("\n" + "="*80)
    print("GLOBAL CONFIDENCE SCORE DISTRIBUTION")
    print("="*80)
    
    for cat_name in sorted(all_scores.keys()):
        scores = np.array(all_scores[cat_name])
        if len(scores) == 0:
            continue
        print(f"\n{cat_name} ({len(scores)} total detections):")
        print(f"  min={scores.min():.4f}  max={scores.max():.4f}")
        print(f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}")
        print(f"  std={scores.std():.4f}")
        
        # Percentiles
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"  P{p:2d} = {np.percentile(scores, p):.4f}", end="")
        print()
        
        # How many survive at various thresholds
        for thr in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            n = (scores >= thr).sum()
            pct = 100 * n / len(scores)
            print(f"  >= {thr:.2f}: {n:5d} ({pct:5.1f}%)", end="")
            if thr in [0.10, 0.30]:
                print()
        print()


if __name__ == '__main__':
    main()
