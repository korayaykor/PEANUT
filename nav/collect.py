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

from agent.peanut_agent import PEANUT_Agent


def main():

    args = get_args()
    args.only_explore = 0  
    
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
    nav_agent = PEANUT_Agent(args=args,task_config=config)
    print(config.DATASET.SPLIT, 'split')
    print(len(hab_env.episodes), 'episodes in dataset')
    # prepare results file in dump location
    os.makedirs(args.dump_location, exist_ok=True)
    results_file = os.path.join(args.dump_location, f"{args.exp_name}_results.txt")
    # ensure file exists
    open(results_file, 'a').close()
    
    num_episodes = 500
    start = args.start_ep
    end = args.end_ep if args.end_ep > 0 else num_episodes
    
    sucs, spls, ep_lens = [], [], []
    
    ep_i = 0
    while ep_i < min(num_episodes, end):
        observations = hab_env.reset()
        nav_agent.reset()
        print('-' * 40)
        sys.stdout.flush()
        
        if ep_i >= start and ep_i < end:
            print('Episode %d | Target: %s' % (ep_i, hm3d_names[observations['objectgoal'][0]]))
            target_name = hm3d_names[observations['objectgoal'][0]]
            print('Episode %d | Target: %s' % (ep_i, target_name))
            print('Scene: %s' % hab_env._current_episode.scene_id)

            step_i = 0
            seq_i = 0
            ep_start_time = time.time()
            
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                          
                if step_i % 100 == 0:
                    print('step %d...' % step_i)
                    sys.stdout.flush()

                step_i += 1
                    
            if args.only_explore == 0:
                
                ep_elapsed_time = time.time() - ep_start_time
                print('ended at step %d (%.2fs)' % (step_i, ep_elapsed_time))
                
                # Navigation metrics
                metrics = hab_env.get_metrics()
                print(metrics)
                # Append per-episode metrics to results file (one JSON object per line)
                record = {
                    'episode': ep_i,
                    'scene_id': hab_env._current_episode.scene_id,
                    'target': target_name,
                    'episode_length': step_i,
                    'time': ep_elapsed_time,
                }
                # merge metrics into record (metrics may contain nested values)
                try:
                    record.update(metrics)
                except Exception:
                    # fallback: stringify metrics
                    record['metrics'] = str(metrics)

                try:
                    with open(results_file, 'a') as _f:
                        _f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"Failed to write metrics to {results_file}: {e}")
                
                # Log the metrics (save them however you want)
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                ep_lens.append(step_i)
                print('-' * 40)
                print('Average Success: %.4f, Average SPL: %.4f' % (np.mean(sucs), np.mean(spls)))
                print('-' * 40)
                sys.stdout.flush()
                
        ep_i += 1
        

if __name__ == "__main__":
    main()
