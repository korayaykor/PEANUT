#!/usr/bin/env python
"""Generate comparison charts for YOLO Finetuned vs Mask R-CNN (R101-COCO) benchmarks."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_results(filepath):
    """Load per-episode JSONL results from the given file path."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def compute_stats(results):
    """Group episodes by target category and compute per-category success/SPL."""
    cats = {}
    for r in results:
        t = r["target"]
        if t not in cats:
            cats[t] = {"success": [], "spl": []}
        cats[t]["success"].append(r["success"])
        cats[t]["spl"].append(r["spl"])
    
    stats = {}
    for cat in sorted(cats.keys()):
        n = len(cats[cat]["success"])
        stats[cat] = {
            "count": n,
            "success": sum(cats[cat]["success"]) / n * 100,
            "spl": sum(cats[cat]["spl"]) / n * 100,
        }
    
    total = len(results)
    stats["OVERALL"] = {
        "count": total,
        "success": sum(r["success"] for r in results) / total * 100,
        "spl": sum(r["spl"] for r in results) / total * 100,
    }
    return stats

# --- Load all results ---
base = "/nav/data/comparison_with_pretrained"

models = {
    "Mask R-CNN\n(R101-COCO)": os.path.join(base, "r101coco", "r101coco_allcat_all_categories_results.txt"),
    "Cascade\nMask R-CNN": os.path.join(base, "cascade", "cascade_allcat_all_categories_results.txt"),
    "YOLOv8x-seg\n(COCO)": os.path.join(base, "yolo_pretrained_rerun", "yolo_pretrained_rerun_all_categories_results.txt"),
    "YOLOv8x-seg\n(Finetuned)": os.path.join(base, "yolo_finetuned_v2_best", "yolo_finetuned_v2_best_all_categories_results.txt"),
}

all_stats = {}
for name, path in models.items():
    if os.path.exists(path):
        all_stats[name] = compute_stats(load_results(path))
        print(f"Loaded {name}: {path}")
    else:
        print(f"MISSING: {path}")

# Categories to plot
categories = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa", "OVERALL"]
cat_labels = ["Chair", "Bed", "Plant", "Toilet", "TV Monitor", "Sofa", "OVERALL"]

model_names = list(all_stats.keys())
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # blue, orange, green, red

# ============================================================
# Chart 1: Success Rate comparison
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(categories))
width = 0.18
offsets = np.arange(len(model_names)) - (len(model_names) - 1) / 2

for i, model in enumerate(model_names):
    vals = []
    for cat in categories:
        if cat in all_stats[model]:
            vals.append(all_stats[model][cat]["success"])
        else:
            vals.append(0)
    bars = ax.bar(x + offsets[i] * width, vals, width * 0.9, label=model, color=colors[i], edgecolor='white', linewidth=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Object Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('ObjectNav Success Rate: All Models Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)  # Separator before OVERALL

plt.tight_layout()
plt.savefig(os.path.join(base, "charts", "all_models_success.png"), dpi=150)
print("Saved: all_models_success.png")

# ============================================================
# Chart 2: SPL comparison
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

for i, model in enumerate(model_names):
    vals = []
    for cat in categories:
        if cat in all_stats[model]:
            vals.append(all_stats[model][cat]["spl"])
        else:
            vals.append(0)
    bars = ax.bar(x + offsets[i] * width, vals, width * 0.9, label=model, color=colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Object Category', fontsize=13, fontweight='bold')
ax.set_ylabel('SPL (%)', fontsize=13, fontweight='bold')
ax.set_title('ObjectNav SPL: All Models Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=11)
ax.set_ylim(0, 60)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(base, "charts", "all_models_spl.png"), dpi=150)
print("Saved: all_models_spl.png")

# ============================================================
# Chart 3: Head-to-head YOLO Finetuned vs Mask R-CNN (grouped)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

focus_models = ["Mask R-CNN\n(R101-COCO)", "YOLOv8x-seg\n(Finetuned)"]
focus_colors = ["#4C72B0", "#C44E52"]
focus_labels = ["Mask R-CNN (R101-COCO)", "YOLOv8x-seg (Finetuned)"]

width2 = 0.3

# Success Rate
for i, model in enumerate(focus_models):
    vals = []
    for cat in categories:
        if cat in all_stats[model]:
            vals.append(all_stats[model][cat]["success"])
        else:
            vals.append(0)
    bars = ax1.bar(x + (i - 0.5) * width2, vals, width2 * 0.9, label=focus_labels[i], color=focus_colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Object Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(cat_labels, fontsize=10, rotation=15)
ax1.set_ylim(0, 115)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)

# SPL
for i, model in enumerate(focus_models):
    vals = []
    for cat in categories:
        if cat in all_stats[model]:
            vals.append(all_stats[model][cat]["spl"])
        else:
            vals.append(0)
    bars = ax2.bar(x + (i - 0.5) * width2, vals, width2 * 0.9, label=focus_labels[i], color=focus_colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Object Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('SPL (%)', fontsize=12, fontweight='bold')
ax2.set_title('SPL (Success weighted by Path Length)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(cat_labels, fontsize=10, rotation=15)
ax2.set_ylim(0, 55)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)

fig.suptitle('YOLOv8x-seg (Finetuned) vs Mask R-CNN (R101-COCO)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(base, "charts", "yolo_ft_vs_maskrcnn.png"), dpi=150, bbox_inches='tight')
print("Saved: yolo_ft_vs_maskrcnn.png")

# ============================================================
# Chart 4: Overall summary bar chart (all 4 models side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

model_short_names = ["Mask R-CNN\nR101-COCO", "Cascade\nMask R-CNN", "YOLOv8x\nCOCO", "YOLOv8x\nFinetuned"]
success_vals = [all_stats[m]["OVERALL"]["success"] for m in model_names]
spl_vals = [all_stats[m]["OVERALL"]["spl"] for m in model_names]

bars1 = ax1.bar(model_short_names, success_vals, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars1, success_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
ax1.set_title('Overall Success Rate', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 70)
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(model_short_names, spl_vals, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars2, spl_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('SPL (%)', fontsize=13, fontweight='bold')
ax2.set_title('Overall SPL', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 35)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('ObjectNav Benchmark: Overall Performance (50 episodes, 10 scenes)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base, "charts", "overall_summary.png"), dpi=150, bbox_inches='tight')
print("Saved: overall_summary.png")

# Print summary table
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)
header = "%-12s" % "Category"
for m in model_names:
    short = m.replace("\n", " ")
    header += " | %s" % short
print(header)
print("-" * 90)

for cat, label in zip(categories, cat_labels):
    row = "%-12s" % label
    for m in model_names:
        if cat in all_stats[m]:
            s = all_stats[m][cat]["success"]
            sp = all_stats[m][cat]["spl"]
            row += " | %4.0f%% / %4.1f%%" % (s, sp)
        else:
            row += " |    -  /   -  "
    print(row)

print("=" * 90)
