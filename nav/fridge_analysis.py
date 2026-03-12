"""
Quick test: check refrigerator cells at different confidence levels
by looking at existing sweep results.
"""
import numpy as np
import json

data = json.load(open("/nav/data/sweep_results.json"))

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

outdoor = {"train","airplane","boat","horse","cow","sheep","elephant","bear","zebra","giraffe","kite","surfboard","snowboard","skateboard","frisbee","skis","sports ball","baseball bat","baseball glove","tennis racket","fire hydrant","stop sign","parking meter","traffic light"}

print("YOLOv8 — refrigerator cells at different confidence levels:")
print("=" * 70)
for r in data:
    if r["seg_type"] != "yolo":
        continue
    cats = r.get("categories", {})
    fridge = cats.get("refrigerator", 0)
    # Sum all non-outdoor cells for reference
    fp_total = sum(cats.get(c, 0) for c in outdoor)
    clean_total = r["total_cells"] - fp_total
    print("  conf=%.2f ct=%.1f | fridge=%5d | clean_total=%6d | fridge/clean=%.1f%%" % (
        r["conf"], r["cat_thresh"], fridge, clean_total, 100*fridge/max(1,clean_total)))
