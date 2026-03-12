"""
analyze_sweep.py - Analyze threshold sweep results for YOLO models.

Reads sweep_results.json produced by sweep_thresholds.py and prints a
table showing total cells, false-positive outdoor categories, and clean
(indoor-only) cell counts for every (model, confidence, cat_threshold)
combination. Prints the best config per model ranked by clean cell count.

Usage (inside container):
    python analyze_sweep.py
"""
import json

data = json.load(open("/nav/data/sweep_results.json"))

outdoor_cats = {"train", "airplane", "boat", "horse", "cow", "sheep", "elephant", "bear",
                "zebra", "giraffe", "kite", "surfboard", "snowboard", "skateboard", "frisbee",
                "skis", "sports ball", "baseball bat", "baseball glove", "tennis racket",
                "fire hydrant", "stop sign", "parking meter", "traffic light"}

header = "%-8s %5s %5s | %7s | %5s | %7s | %4s | FP details" % ("Model","conf","ct","total","FP","clean","cats")
print(header)
print("=" * 110)

best = {}
for r in data:
    cats = r.get("categories", {})
    fp_total = sum(cats.get(c, 0) for c in outdoor_cats)
    fp_detail = {c: cats[c] for c in outdoor_cats if cats.get(c, 0) > 0}
    model = r["seg_type"]
    conf = r["conf"]
    ct = r["cat_thresh"]
    total = r["total_cells"]
    nc = r["n_cats"]
    clean = total - fp_total
    fp_cats_n = len(fp_detail)
    clean_cats = nc - fp_cats_n
    print("%-8s %5.2f %5.1f | %7d | %5d | %7d | %4d | %s" % (model, conf, ct, total, fp_total, clean, clean_cats, fp_detail))

    if model not in best or clean > best[model]["clean"]:
        best[model] = {"conf": conf, "ct": ct, "total": total, "fp": fp_total, "clean": clean, "clean_cats": clean_cats, "fp_detail": fp_detail}

print("\n" + "=" * 110)
print("BEST CONFIG PER MODEL (by clean cells, excluding outdoor FP):")
print("=" * 110)
for m in ["yolo", "yolo11", "yolo26"]:
    b = best[m]
    print("  %s: conf=%.2f, cat_thresh=%.1f => clean=%d cells, %d cats (FP removed: %d cells from %s)" % (
        m, b["conf"], b["ct"], b["clean"], b["clean_cats"], b["fp"], b["fp_detail"]))
