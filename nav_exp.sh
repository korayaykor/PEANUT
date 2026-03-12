#!/usr/bin/env bash

# Original: runs sequential episodes (1 object per episode)
# python nav/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 0 --end_ep 10 --evaluation $AGENT_EVALUATION_TYPE $@ 

# All categories: --start_ep = start scene index, --end_ep = end scene index
# For first 10 scenes, each tested with ALL 6 hm3d categories (chair, bed, plant, toilet, tv_monitor, sofa)
python nav/collect_all_categories.py -v 0 --dump_location ./data/tmp --exp_name debug --start_ep 0 --end_ep 10 --evaluation $AGENT_EVALUATION_TYPE $@
sleep infinity
