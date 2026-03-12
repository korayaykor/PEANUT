#!/usr/bin/env python3
"""
Generate comparison charts: Finetuned Mask R-CNN (cat9) vs Finetuned YOLOv8
Also includes pretrained models for context.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = '/nav/data/comparison_with_pretrained'
OUTPUT_DIR = os.path.join(RESULTS_BASE, 'comparison_charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────
def load_results(path):
    """Load per-episode JSONL results from the given file path."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes

def summarize(episodes):
    """Group episodes by target category and compute per-category success, SPL, and avg length."""
    cats = {}
    for ep in episodes:
        t = ep['target']
        if t not in cats:
            cats[t] = {'success': [], 'spl': [], 'length': []}
        cats[t]['success'].append(ep['success'])
        cats[t]['spl'].append(ep['spl'])
        cats[t]['length'].append(ep['episode_length'])
    overall_success = np.mean([ep['success'] for ep in episodes])
    overall_spl = np.mean([ep['spl'] for ep in episodes])
    overall_length = np.mean([ep['episode_length'] for ep in episodes])
    per_cat = {}
    for t, v in cats.items():
        per_cat[t] = {
            'success': np.mean(v['success']),
            'spl': np.mean(v['spl']),
            'length': np.mean(v['length']),
            'n': len(v['success'])
        }
    return {
        'overall_success': overall_success,
        'overall_spl': overall_spl,
        'overall_length': overall_length,
        'per_cat': per_cat,
        'n_episodes': len(episodes)
    }

# All 5 models
models = {
    'Mask R-CNN R101\n(COCO pretrained)': os.path.join(RESULTS_BASE, 'r101coco', 'r101coco_allcat_all_categories_results.txt'),
    'Cascade MR-CNN\n(COCO pretrained)': os.path.join(RESULTS_BASE, 'cascade', 'cascade_allcat_all_categories_results.txt'),
    'YOLOv8x-seg\n(COCO pretrained)': os.path.join(RESULTS_BASE, 'yolo_pretrained_rerun', 'yolo_pretrained_rerun_all_categories_results.txt'),
    'Mask R-CNN R101\n(PEANUT finetuned)': os.path.join(RESULTS_BASE, 'maskrcnn_cat9', 'maskrcnn_cat9_all_categories_results.txt'),
    'YOLOv8x-seg\n(HM3D finetuned)': os.path.join(RESULTS_BASE, 'yolo_finetuned_v2_fixed', 'yolo_finetuned_v2_fixed_all_categories_results.txt'),
    'Grounded-SAM\n(PEANUT eval)': os.path.join(RESULTS_BASE, 'grounded_sam_sanity', 'grounded_sam_sanity_all_categories_results.txt'),
    'YOLOv11x-seg\n(COCO pretrained)': os.path.join(RESULTS_BASE, 'yolo11_pretrained', 'yolo11_pretrained_all_categories_results.txt'),
    'YOLO26x-seg\n(COCO pretrained)': os.path.join(RESULTS_BASE, 'yolo26_pretrained', 'yolo26_pretrained_all_categories_results.txt'),
}

# Colors: pretrained = lighter, finetuned = darker/vivid
colors = {
    'Mask R-CNN R101\n(COCO pretrained)': '#93c5fd',       # light blue
    'Cascade MR-CNN\n(COCO pretrained)': '#a5b4fc',        # light indigo
    'YOLOv8x-seg\n(COCO pretrained)': '#86efac',           # light green
    'Mask R-CNN R101\n(PEANUT finetuned)': '#2563eb',      # vivid blue
    'YOLOv8x-seg\n(HM3D finetuned)': '#16a34a',           # vivid green
    'Grounded-SAM\n(PEANUT eval)': '#f59e0b',            # amber
    'YOLOv11x-seg\n(COCO pretrained)': '#e11d48',         # rose
    'YOLO26x-seg\n(COCO pretrained)': '#7c3aed',          # violet
}

summaries = {}
for name, path in models.items():
    if os.path.exists(path):
        episodes = load_results(path)
        if len(episodes) == 0:
            print(f"  WARNING: {path} is empty, skipping {name}")
            continue
        summaries[name] = summarize(episodes)
        print(f"  Loaded {name}: {summaries[name]['n_episodes']} episodes, "
              f"Success={summaries[name]['overall_success']:.1%}, SPL={summaries[name]['overall_spl']:.1%}")
    else:
        print(f"  WARNING: {path} not found, skipping {name}")

model_names = list(summaries.keys())
categories = ['chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa']

if len(model_names) == 0:
    print("No valid result files found. Nothing to plot.")
    raise SystemExit(0)

# ── Chart 1: Overall Success & SPL ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

success_vals = [summaries[m]['overall_success'] * 100 for m in model_names]
spl_vals = [summaries[m]['overall_spl'] * 100 for m in model_names]
bar_colors = [colors[m] for m in model_names]

# Success rate
bars1 = axes[0].bar(range(len(model_names)), success_vals, color=bar_colors, edgecolor='#333', linewidth=0.8)
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, fontsize=8)
axes[0].set_ylabel('Success Rate (%)', fontsize=11)
axes[0].set_title('Overall Success Rate', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, max(success_vals) * 1.2)
for bar, val in zip(bars1, success_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# SPL
bars2 = axes[1].bar(range(len(model_names)), spl_vals, color=bar_colors, edgecolor='#333', linewidth=0.8)
axes[1].set_xticks(range(len(model_names)))
axes[1].set_xticklabels(model_names, fontsize=8)
axes[1].set_ylabel('SPL (%)', fontsize=11)
axes[1].set_title('Overall SPL (Success weighted by Path Length)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(spl_vals) * 1.2)
for bar, val in zip(bars2, spl_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('PEANUT ObjectNav — 5-Model Comparison (50 episodes, 10 HM3D val scenes)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5model_overall.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/5model_overall.png")
plt.close()

# ── Chart 2: Per-category comparison (grouped bars) ───────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

x = np.arange(len(categories))
n_models = len(model_names)
width = 0.15

for i, m in enumerate(model_names):
    cat_success = []
    cat_spl = []
    for cat in categories:
        if cat in summaries[m]['per_cat']:
            cat_success.append(summaries[m]['per_cat'][cat]['success'] * 100)
            cat_spl.append(summaries[m]['per_cat'][cat]['spl'] * 100)
        else:
            cat_success.append(0)
            cat_spl.append(0)
    offset = (i - n_models/2 + 0.5) * width
    axes[0].bar(x + offset, cat_success, width, label=m, color=colors[m], edgecolor='#333', linewidth=0.5)
    axes[1].bar(x + offset, cat_spl, width, label=m, color=colors[m], edgecolor='#333', linewidth=0.5)

for ax, title in zip(axes, ['Per-Category Success Rate (%)', 'Per-Category SPL (%)']):
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('PEANUT ObjectNav — Per-Category Breakdown (5 Models)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5model_per_category.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/5model_per_category.png")
plt.close()

if 'Mask R-CNN R101\n(PEANUT finetuned)' in summaries and 'YOLOv8x-seg\n(HM3D finetuned)' in summaries:
    finetuned = ['Mask R-CNN R101\n(PEANUT finetuned)', 'YOLOv8x-seg\n(HM3D finetuned)']
    ft_colors = [colors[m] for m in finetuned]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, m in enumerate(finetuned):
        axes[0].bar(i, summaries[m]['overall_success'] * 100, color=ft_colors[i], edgecolor='#333', linewidth=0.8, label=m)
        axes[0].text(i, summaries[m]['overall_success'] * 100 + 1,
                    f"{summaries[m]['overall_success']*100:.1f}%", ha='center', fontweight='bold', fontsize=12)
    axes[0].set_xticks(range(len(finetuned)))
    axes[0].set_xticklabels([m.replace('\n', ' ') for m in finetuned], fontsize=8)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 75)

    for i, m in enumerate(finetuned):
        axes[1].bar(i, summaries[m]['overall_spl'] * 100, color=ft_colors[i], edgecolor='#333', linewidth=0.8, label=m)
        axes[1].text(i, summaries[m]['overall_spl'] * 100 + 0.5,
                    f"{summaries[m]['overall_spl']*100:.1f}%", ha='center', fontweight='bold', fontsize=12)
    axes[1].set_xticks(range(len(finetuned)))
    axes[1].set_xticklabels([m.replace('\n', ' ') for m in finetuned], fontsize=8)
    axes[1].set_ylabel('SPL (%)')
    axes[1].set_title('SPL', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 40)

    x = np.arange(len(categories))
    width = 0.35
    for i, m in enumerate(finetuned):
        vals = []
        for cat in categories:
            if cat in summaries[m]['per_cat']:
                vals.append(summaries[m]['per_cat'][cat]['success'] * 100)
            else:
                vals.append(0)
        axes[2].bar(x + (i - 0.5) * width, vals, width, label=m, color=ft_colors[i], edgecolor='#333', linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, fontsize=9)
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Per-Category Success', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Finetuned Models Head-to-Head: Mask R-CNN (PEANUT cat9) vs YOLOv8 (HM3D GT)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'finetuned_head_to_head.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/finetuned_head_to_head.png")
    plt.close()

# ── Chart 4: Improvement from finetuning ──────────────────────
if ('Mask R-CNN R101\n(COCO pretrained)' in summaries and
        'Mask R-CNN R101\n(PEANUT finetuned)' in summaries and
        'YOLOv8x-seg\n(COCO pretrained)' in summaries and
        'YOLOv8x-seg\n(HM3D finetuned)' in summaries):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mr_pre = summaries['Mask R-CNN R101\n(COCO pretrained)']
    mr_ft = summaries['Mask R-CNN R101\n(PEANUT finetuned)']

    for cat_idx, cat in enumerate(categories):
        pre_s = mr_pre['per_cat'].get(cat, {}).get('success', 0) * 100
        ft_s = mr_ft['per_cat'].get(cat, {}).get('success', 0) * 100
        delta = ft_s - pre_s
        color = '#16a34a' if delta >= 0 else '#dc2626'
        axes[0].barh(cat_idx, delta, color=color, edgecolor='#333', linewidth=0.5)
        axes[0].text(delta + (1 if delta >= 0 else -1), cat_idx,
                    f'{delta:+.1f}%', va='center', fontsize=10, fontweight='bold')

    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Change in Success Rate (%)')
    axes[0].set_title('Mask R-CNN R101: Finetuning Gain\n(PEANUT cat9 − COCO pretrained)', fontsize=11, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    yolo_pre = summaries['YOLOv8x-seg\n(COCO pretrained)']
    yolo_ft = summaries['YOLOv8x-seg\n(HM3D finetuned)']

    for cat_idx, cat in enumerate(categories):
        pre_s = yolo_pre['per_cat'].get(cat, {}).get('success', 0) * 100
        ft_s = yolo_ft['per_cat'].get(cat, {}).get('success', 0) * 100
        delta = ft_s - pre_s
        color = '#16a34a' if delta >= 0 else '#dc2626'
        axes[1].barh(cat_idx, delta, color=color, edgecolor='#333', linewidth=0.5)
        axes[1].text(delta + (1 if delta >= 0 else -1), cat_idx,
                    f'{delta:+.1f}%', va='center', fontsize=10, fontweight='bold')

    axes[1].set_yticks(range(len(categories)))
    axes[1].set_yticklabels(categories)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Change in Success Rate (%)')
    axes[1].set_title('YOLOv8x-seg: Finetuning Gain\n(HM3D GT − COCO pretrained)', fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle('Effect of Finetuning on Navigation Success Rate',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'finetuning_improvement.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/finetuning_improvement.png")
    plt.close()

# ── Summary table ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("FULL 5-MODEL COMPARISON SUMMARY")
print("=" * 80)
print(f"{'Model':<35} {'Success':>10} {'SPL':>10} {'Avg Len':>10}")
print("-" * 65)
for m in model_names:
    s = summaries[m]
    print(f"{m.replace(chr(10), ' '):<35} {s['overall_success']*100:>9.1f}% {s['overall_spl']*100:>9.1f}% {s['overall_length']:>10.1f}")
print("-" * 65)
print("\nPer-category Success Rate:")
print(f"{'Model':<35}", end="")
for cat in categories:
    print(f" {cat:>10}", end="")
print()
print("-" * 95)
for m in model_names:
    print(f"{m.replace(chr(10), ' '):<35}", end="")
    for cat in categories:
        val = summaries[m]['per_cat'].get(cat, {}).get('success', 0) * 100
        print(f" {val:>9.1f}%", end="")
    print()
print("=" * 95)
print(f"\nAll charts saved to: {OUTPUT_DIR}/")
