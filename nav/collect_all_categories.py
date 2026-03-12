"""
collect_all_categories.py - Run ObjectNav evaluation across all PEANUT categories.

Iterates over all 6 PEANUT object categories (chair, bed, plant, toilet,
tv_monitor, sofa) and runs evaluation episodes using the Habitat simulator.
Saves per-episode results as JSONL files for later analysis by the
compare_*.py scripts.

Functions:
    main() - Parse arguments, set up Habitat environment, run episodes,
             and save per-episode results to JSONL files.

Usage (inside container):
    python collect_all_categories.py --seg_model_type yolo --split val_mini
"""
import argparse
import os
import random
import habitat
import torch
import sys
import cv2
import time
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict

from agent.peanut_agent import PEANUT_Agent


def main():

    args = get_args()
    args.only_explore = 1  # Set to 0 if PEANUT_Prediction_Model (mmcv) is available  
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = "val"
    config.freeze()
    
    hab_env = Env(config=config)
    nav_agent = PEANUT_Agent(args=args, task_config=config)
    print(config.DATASET.SPLIT, "split")
    print(len(hab_env.episodes), "episodes in dataset")

    # ── Build a map: scene_id -> { category: first_episode_index } ──
    all_categories = list(hm3d_names.values())  # [chair, bed, plant, toilet, tv_monitor, sofa]
    print("Target categories:", all_categories)
    
    # Map scene -> {category -> episode_index} (first occurrence)
    scene_cat_to_ep = OrderedDict()
    for ep_idx, ep in enumerate(hab_env.episodes):
        scene_id = ep.scene_id
        cat = ep.goals[0].object_category
        if scene_id not in scene_cat_to_ep:
            scene_cat_to_ep[scene_id] = {}
        if cat not in scene_cat_to_ep[scene_id]:
            scene_cat_to_ep[scene_id][cat] = ep_idx

    scenes = list(scene_cat_to_ep.keys())
    print("\nFound %d scenes in dataset" % len(scenes))
    for s in scenes:
        cats = list(scene_cat_to_ep[s].keys())
        print("  Scene %s: categories %s" % (s.split("/")[-2], cats))

    # ── Determine which scenes to evaluate ──
    num_scenes = args.end_ep if args.end_ep > 0 else len(scenes)
    start_scene = args.start_ep
    selected_scenes = scenes[start_scene:num_scenes]
    
    print("\n" + "=" * 60)
    print("Will evaluate %d scenes (scene index %d to %d)" % (
        len(selected_scenes), start_scene, num_scenes - 1))
    print("Each scene will be tested with ALL available categories")
    print("=" * 60)

    # prepare results file in dump location
    os.makedirs(args.dump_location, exist_ok=True)
    results_file = os.path.join(args.dump_location, "%s_all_categories_results.txt" % args.exp_name)
    # Clear the results file for a fresh run
    with open(results_file, "w") as _f:
        pass

    sucs, spls, ep_lens = [], [], []
    total_runs = 0

    for scene_idx, scene_id in enumerate(selected_scenes):
        scene_short = scene_id.split("/")[-2]
        cat_map = scene_cat_to_ep[scene_id]
        
        print("\n" + "#" * 60)
        print("Scene %d/%d: %s" % (scene_idx + 1, len(selected_scenes), scene_short))
        print("Available categories: %s" % list(cat_map.keys()))
        print("#" * 60)

        for cat_name in all_categories:
            if cat_name not in cat_map:
                print("  [SKIP] Category %s not in scene %s" % (cat_name, scene_short))
                continue

            ep_index = cat_map[cat_name]
            
            # Jump to the specific episode
            hab_env._current_episode = hab_env.episodes[ep_index]
            hab_env._episode_from_iter_on_reset = False
            
            observations = hab_env.reset()
            nav_agent.reset()
            
            actual_target = hm3d_names[observations["objectgoal"][0]]
            print("\n" + "-" * 40)
            print("Scene: %s | Target: %s (ep_index=%d)" % (scene_short, actual_target, ep_index))
            print("-" * 40)
            sys.stdout.flush()

            step_i = 0
            ep_start_time = time.time()
            
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                          
                if step_i % 100 == 0:
                    print("step %d..." % step_i)
                    sys.stdout.flush()

                step_i += 1
                    
            ep_elapsed_time = time.time() - ep_start_time
            print("ended at step %d (%.2fs)" % (step_i, ep_elapsed_time))
            
            # Navigation metrics
            metrics = hab_env.get_metrics()
            print(metrics)
            
            # Append per-episode metrics to results file
            record = {
                "scene_idx": start_scene + scene_idx,
                "episode_index": ep_index,
                "scene_id": scene_id,
                "scene_short": scene_short,
                "target": actual_target,
                "episode_length": step_i,
                "time": ep_elapsed_time,
            }
            try:
                record.update(metrics)
            except Exception:
                record["metrics"] = str(metrics)

            try:
                with open(results_file, "a") as _f:
                    _f.write(json.dumps(record) + "\n")
            except Exception as e:
                print("Failed to write metrics to %s: %s" % (results_file, e))
            
            sucs.append(metrics["success"])
            spls.append(metrics["spl"])
            ep_lens.append(step_i)
            total_runs += 1
            
            print("-" * 40)
            print("Running Avg | Success: %.4f, SPL: %.4f (%d runs)" % (
                np.mean(sucs), np.mean(spls), total_runs))
            print("-" * 40)
            sys.stdout.flush()

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS (%d total runs across %d scenes)" % (total_runs, len(selected_scenes)))
    print("Overall Success: %.4f" % np.mean(sucs))
    print("Overall SPL: %.4f" % np.mean(spls))
    print("Average Episode Length: %.1f" % np.mean(ep_lens))
    
    # Per-category summary
    print("\nPer-category breakdown:")
    with open(results_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for cat in all_categories:
        cat_records = [r for r in records if r["target"] == cat]
        if cat_records:
            cat_suc = np.mean([r["success"] for r in cat_records])
            cat_spl = np.mean([r["spl"] for r in cat_records])
            print("  %s: Success=%.4f, SPL=%.4f (%d episodes)" % (
                cat, cat_suc, cat_spl, len(cat_records)))
        else:
            print("  %s: No episodes" % cat)
    print("=" * 60)
    print("Results saved to: %s" % results_file)


if __name__ == "__main__":
    main()
