"""
vote_analysis.py - Analyze per-category vote count distributions.

Loads the saved vote map (vote_map_coco80.npy) from a YOLOv8 semantic
mapping run and shows how many cells survive at different minimum-vote
thresholds (v>=1 through v>=10) for each COCO-80 category. Helps
determine the optimal min_votes parameter to suppress low-evidence
false positives (e.g. walls misclassified as refrigerator).

Usage (inside container):
    python vote_analysis.py
"""
import numpy as np

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

vote = np.load("/nav/data/tmp_coco80_yolo_best/coco80_replay_5LpN3gDmAk7_1_chair/vote_map_coco80.npy")

print("Effect of min_vote threshold on each category (YOLOv8):")
header = "%-22s | %6s | %6s | %6s | %6s | %6s | %6s" % ("Category","v>=1","v>=2","v>=3","v>=5","v>=8","v>=10")
print(header)
print("=" * 85)
for i in range(80):
    active = vote[i] > 0
    n1 = int(active.sum())
    if n1 == 0:
        continue
    n2 = int((vote[i]>=2).sum())
    n3 = int((vote[i]>=3).sum())
    n5 = int((vote[i]>=5).sum())
    n8 = int((vote[i]>=8).sum())
    n10 = int((vote[i]>=10).sum())
    print("  %-20s | %6d | %6d | %6d | %6d | %6d | %6d" % (COCO[i], n1, n2, n3, n5, n8, n10))

print()
line = "%-22s" % "TOTAL"
for thr in [1,2,3,5,8,10]:
    total = int((vote.max(axis=0)>=thr).sum())
    line += " | %6d" % total
print(line)
