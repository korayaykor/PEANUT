"""
Run single-category evaluation across all scenes that contain that category.
Usage:
  python collect_single_category.py --seg_model_type yolo11 --target_category sofa ...
"""
import os
import sys
import time
import json
import habitat
import numpy as np
from collections import OrderedDict

from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
from agent.peanut_agent import PEANUT_Agent


def main():
    """Run single-category ObjectNav evaluation across matching scenes."""
    args = get_args()
    args.only_explore = 1

    # Target category to evaluate (default: sofa)
    target_cat = getattr(args, 'target_category', 'sofa')
    print("=" * 60)
    print("SINGLE-CATEGORY EVALUATION: %s" % target_cat)
    print("=" * 60)

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
    print(len(hab_env.episodes), "episodes in dataset")

    # Build scene -> {category -> episode_index}
    scene_cat_to_ep = OrderedDict()
    for ep_idx, ep in enumerate(hab_env.episodes):
        scene_id = ep.scene_id
        cat = ep.goals[0].object_category
        if scene_id not in scene_cat_to_ep:
            scene_cat_to_ep[scene_id] = {}
        if cat not in scene_cat_to_ep[scene_id]:
            scene_cat_to_ep[scene_id][cat] = ep_idx

    # Filter scenes that have the target category
    target_scenes = []
    for scene_id, cat_map in scene_cat_to_ep.items():
        if target_cat in cat_map:
            target_scenes.append((scene_id, cat_map[target_cat]))

    print("\nFound %d scenes with '%s' episodes" % (len(target_scenes), target_cat))

    # Apply scene range limits
    num_scenes = args.end_ep if args.end_ep > 0 else len(target_scenes)
    start_scene = args.start_ep
    target_scenes = target_scenes[start_scene:num_scenes]

    print("Will evaluate %d scenes (index %d to %d)" % (
        len(target_scenes), start_scene,
        min(num_scenes, len(target_scenes)) - 1))
    print("=" * 60)

    # Prepare results file
    os.makedirs(args.dump_location, exist_ok=True)
    results_file = os.path.join(
        args.dump_location,
        "%s_%s_results.txt" % (args.exp_name, target_cat))
    with open(results_file, "w") as _f:
        pass

    sucs, spls, ep_lens = [], [], []

    for scene_idx, (scene_id, ep_index) in enumerate(target_scenes):
        scene_short = scene_id.split("/")[-2]

        # Jump to specific episode
        hab_env._current_episode = hab_env.episodes[ep_index]
        hab_env._episode_from_iter_on_reset = False
        observations = hab_env.reset()
        nav_agent.reset()

        actual_target = hm3d_names[observations["objectgoal"][0]]
        print("\n" + "-" * 50)
        print("[%d/%d] Scene: %s | Target: %s (ep=%d)" % (
            scene_idx + 1, len(target_scenes), scene_short,
            actual_target, ep_index))
        print("-" * 50)
        sys.stdout.flush()

        step_i = 0
        t0 = time.time()
        while not hab_env.episode_over:
            action = nav_agent.act(observations)
            observations = hab_env.step(action)
            if step_i % 100 == 0:
                print("  step %d..." % step_i)
                sys.stdout.flush()
            step_i += 1

        elapsed = time.time() - t0
        metrics = hab_env.get_metrics()
        status = "SUCCESS" if metrics["success"] > 0.5 else "FAIL"
        print("  %s  steps=%d  spl=%.3f  (%.1fs)" % (
            status, step_i, metrics["spl"], elapsed))

        record = {
            "scene_idx": start_scene + scene_idx,
            "episode_index": ep_index,
            "scene_id": scene_id,
            "scene_short": scene_short,
            "target": actual_target,
            "episode_length": step_i,
            "time": elapsed,
        }
        try:
            record.update(metrics)
        except Exception:
            record["metrics"] = str(metrics)
        with open(results_file, "a") as _f:
            _f.write(json.dumps(record) + "\n")

        sucs.append(metrics["success"])
        spls.append(metrics["spl"])
        ep_lens.append(step_i)

        print("  Running: Success=%.1f%% SPL=%.1f%% (%d/%d)" % (
            np.mean(sucs) * 100, np.mean(spls) * 100,
            scene_idx + 1, len(target_scenes)))
        sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS: %s (%d scenes)" % (target_cat, len(target_scenes)))
    print("  Success: %.2f%%" % (np.mean(sucs) * 100))
    print("  SPL:     %.2f%%" % (np.mean(spls) * 100))
    print("  Avg Len: %.1f" % np.mean(ep_lens))
    print("Results saved to: %s" % results_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
