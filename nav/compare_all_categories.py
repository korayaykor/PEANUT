#!/usr/bin/env python3
"""
Compare Mask R-CNN vs YOLOv8 across ALL object categories and scenes.

Reads per-episode JSONL result files from collect_all_categories.py and generates:
  1. Overall comparison bar chart
  2. Per-category comparison bar chart (Success & SPL per object type)
  3. Per-scene comparison heatmap
  4. Per-scene x per-category detailed heatmap
  5. Console summary tables

Usage:
    python compare_all_categories.py \
        --maskrcnn_results ./data/tmp/maskrcnn_allcat_all_categories_results.txt \
        --yolo_results ./data/tmp/yolo_allcat_all_categories_results.txt \
        --output_dir ./data/tmp/allcat_comparison
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ALL_CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]


def load_results(path):
    """Load per-episode JSONL results from the given file path."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("[WARN] skipping non-JSON line in %s" % path)
    return records


def compute_metrics(records):
    """Compute aggregated success, SPL, avg episode length and time from episode records."""
    if not records:
        return {"num_episodes": 0, "success": 0, "spl": 0,
                "avg_episode_length": 0, "avg_time": 0}
    return {
        "num_episodes": len(records),
        "success": np.mean([r.get("success", 0) for r in records]) * 100,
        "spl": np.mean([r.get("spl", 0) for r in records]) * 100,
        "avg_episode_length": np.mean([r.get("episode_length", 0) for r in records]),
        "avg_time": np.mean([r.get("time", 0) for r in records]),
    }


def print_table(title, headers, rows, col_widths=None):
    """Print a formatted ASCII table with title, headers, and rows."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    def fmt_row(row):
        return "|" + "|".join(str(row[i]).center(col_widths[i]) for i in range(len(row))) + "|"
    
    print("\n" + title)
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


def chart_overall(model_data, output_dir):
    """Overall comparison bar chart."""
    labels = ["Success Rate (%)", "SPL (%)", "Avg Ep Length", "Avg Time (s)"]
    n = len(model_data)
    x = np.arange(len(labels))
    width = 0.7 / n
    colors = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, m) in enumerate(model_data):
        vals = [m["success"], m["spl"], m["avg_episode_length"], m["avg_time"]]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate("%.1f" % h, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Overall Comparison  -  All Categories (10 Scenes)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "overall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


def chart_per_category(model_records, output_dir):
    """Per-category Success & SPL comparison."""
    n_models = len(model_records)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_i, metric_name in enumerate(["success", "spl"]):
        x = np.arange(len(ALL_CATEGORIES))
        width = 0.7 / n_models
        colors = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6"]

        for i, (name, records) in enumerate(model_records):
            vals = []
            for cat in ALL_CATEGORIES:
                cat_recs = [r for r in records if r["target"] == cat]
                if cat_recs:
                    vals.append(np.mean([r.get(metric_name, 0) for r in cat_recs]) * 100)
                else:
                    vals.append(0)
            offset = (i - (n_models - 1) / 2) * width
            bars = axes[ax_i].bar(x + offset, vals, width, label=name,
                                  color=colors[i % len(colors)])
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    axes[ax_i].annotate("%.0f" % h,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(ALL_CATEGORIES, fontsize=10, rotation=15)
        axes[ax_i].set_ylabel("%s (%%)" % metric_name.upper(), fontsize=11)
        axes[ax_i].set_title("Per-Category %s" % metric_name.upper(), fontsize=13, fontweight="bold")
        axes[ax_i].legend(fontsize=10)
        axes[ax_i].grid(axis="y", alpha=0.3)
        axes[ax_i].set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_category_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


def chart_per_scene(model_records, output_dir):
    """Per-scene Success & SPL comparison."""
    # Collect all scenes
    all_scenes = []
    seen = set()
    for _, records in model_records:
        for r in records:
            s = r.get("scene_short", "")
            if s and s not in seen:
                seen.add(s)
                all_scenes.append(s)
    all_scenes.sort()
    short_names = [s.split("-")[-1][:8] if "-" in s else s[:8] for s in all_scenes]

    n_models = len(model_records)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_i, metric_name in enumerate(["success", "spl"]):
        x = np.arange(len(all_scenes))
        width = 0.7 / n_models
        colors = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6"]

        for i, (name, records) in enumerate(model_records):
            vals = []
            for scene in all_scenes:
                scene_recs = [r for r in records if r.get("scene_short", "") == scene]
                if scene_recs:
                    vals.append(np.mean([r.get(metric_name, 0) for r in scene_recs]) * 100)
                else:
                    vals.append(0)
            offset = (i - (n_models - 1) / 2) * width
            bars = axes[ax_i].bar(x + offset, vals, width, label=name,
                                  color=colors[i % len(colors)])

        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
        axes[ax_i].set_ylabel("%s (%%)" % metric_name.upper(), fontsize=11)
        axes[ax_i].set_title("Per-Scene %s" % metric_name.upper(), fontsize=13, fontweight="bold")
        axes[ax_i].legend(fontsize=10)
        axes[ax_i].grid(axis="y", alpha=0.3)
        axes[ax_i].set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_scene_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


def chart_heatmap(model_records, output_dir):
    """Detailed heatmap: scenes x categories, one subplot per model."""
    # Collect all scenes
    all_scenes = []
    seen = set()
    for _, records in model_records:
        for r in records:
            s = r.get("scene_short", "")
            if s and s not in seen:
                seen.add(s)
                all_scenes.append(s)
    all_scenes.sort()
    short_names = [s.split("-")[-1][:10] if "-" in s else s[:10] for s in all_scenes]

    n_models = len(model_records)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, max(6, len(all_scenes) * 0.5 + 2)))
    if n_models == 1:
        axes = [axes]

    cmap_green_red = LinearSegmentedColormap.from_list("gr", ["#fee2e2", "#fef9c3", "#bbf7d0"])

    for i, (name, records) in enumerate(model_records):
        data = np.full((len(all_scenes), len(ALL_CATEGORIES)), np.nan)
        for r in records:
            scene = r.get("scene_short", "")
            cat = r.get("target", "")
            if scene in all_scenes and cat in ALL_CATEGORIES:
                si = all_scenes.index(scene)
                ci = ALL_CATEGORIES.index(cat)
                data[si, ci] = r.get("success", 0)

        ax = axes[i]
        im = ax.imshow(data, cmap=cmap_green_red, aspect="auto", vmin=0, vmax=1)

        for si in range(len(all_scenes)):
            for ci in range(len(ALL_CATEGORIES)):
                val = data[si, ci]
                if np.isnan(val):
                    ax.text(ci, si, "N/A", ha="center", va="center", fontsize=8, color="gray")
                else:
                    label = "OK" if val > 0.5 else "X"
                    color = "#166534" if val > 0.5 else "#991b1b"
                    ax.text(ci, si, label, ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)

        ax.set_xticks(range(len(ALL_CATEGORIES)))
        ax.set_xticklabels(ALL_CATEGORIES, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(all_scenes)))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(name, fontsize=12, fontweight="bold")

    fig.suptitle("Success per Scene x Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "scene_category_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Compare all-category results: MaskRCNN vs YOLOv8")
    parser.add_argument("--maskrcnn_results", type=str, help="Path to Mask R-CNN all_categories results")
    parser.add_argument("--yolo_results", type=str, help="Path to YOLOv8 all_categories results")
    parser.add_argument("--output_dir", type=str, default="./data/tmp/allcat_comparison",
                        help="Output directory for charts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_data = []
    model_records = []

    if args.maskrcnn_results and os.path.exists(args.maskrcnn_results):
        r = load_results(args.maskrcnn_results)
        print("Loaded %d Mask R-CNN records from %s" % (len(r), args.maskrcnn_results))
        model_data.append(("Mask R-CNN", compute_metrics(r)))
        model_records.append(("Mask R-CNN", r))
    if args.yolo_results and os.path.exists(args.yolo_results):
        r = load_results(args.yolo_results)
        print("Loaded %d YOLOv8 records from %s" % (len(r), args.yolo_results))
        model_data.append(("YOLOv8", compute_metrics(r)))
        model_records.append(("YOLOv8", r))

    if not model_data:
        print("No results files found!")
        sys.exit(1)

    # ── Console Tables ──
    
    # 1. Overall
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    headers = ["Metric"] + [name for name, _ in model_data]
    rows = []
    for key, label, fmt in [
        ("num_episodes", "Episodes", "d"),
        ("success", "Success Rate (%)", ".2f"),
        ("spl", "SPL (%)", ".2f"),
        ("avg_episode_length", "Avg Episode Length", ".1f"),
        ("avg_time", "Avg Time/Ep (s)", ".2f"),
    ]:
        row = [label]
        for _, m in model_data:
            row.append(("%" + fmt) % m[key])
        rows.append(row)
    print_table("Overall", headers, rows)

    # 2. Per-category
    print("\n" + "=" * 70)
    print("PER-CATEGORY COMPARISON")
    print("=" * 70)
    for cat in ALL_CATEGORIES:
        headers = ["Metric"] + [name for name, _ in model_records]
        rows = []
        for name, records in model_records:
            cat_recs = [r for r in records if r["target"] == cat]
            if not cat_recs:
                continue
        cat_data = []
        for name, records in model_records:
            cat_recs = [r for r in records if r["target"] == cat]
            cat_data.append((name, compute_metrics(cat_recs)))
        
        if all(m["num_episodes"] == 0 for _, m in cat_data):
            continue
        
        headers = ["Metric"] + [name for name, _ in cat_data]
        rows = [
            ["Episodes"] + [str(m["num_episodes"]) for _, m in cat_data],
            ["Success (%)"] + ["%.1f" % m["success"] for _, m in cat_data],
            ["SPL (%)"] + ["%.1f" % m["spl"] for _, m in cat_data],
        ]
        print_table("Category: %s" % cat.upper(), headers, rows)

    # 3. Per-scene
    print("\n" + "=" * 70)
    print("PER-SCENE COMPARISON (Success %)")
    print("=" * 70)
    all_scenes = sorted(set(
        r.get("scene_short", "") for _, records in model_records for r in records
    ))
    headers = ["Scene"] + [name for name, _ in model_records]
    rows = []
    for scene in all_scenes:
        row = [scene.split("-")[-1][:12] if "-" in scene else scene[:12]]
        for name, records in model_records:
            scene_recs = [r for r in records if r.get("scene_short", "") == scene]
            if scene_recs:
                suc = np.mean([r.get("success", 0) for r in scene_recs]) * 100
                row.append("%.0f" % suc)
            else:
                row.append("N/A")
        rows.append(row)
    print_table("Per-Scene Success", headers, rows)

    # 4. Detailed: scene x category
    print("\n" + "=" * 70)
    print("DETAILED: SCENE x CATEGORY (Success: OK/X)")
    print("=" * 70)
    for name, records in model_records:
        print("\n--- %s ---" % name)
        headers = ["Scene"] + ALL_CATEGORIES
        rows = []
        for scene in all_scenes:
            row = [scene.split("-")[-1][:12] if "-" in scene else scene[:12]]
            for cat in ALL_CATEGORIES:
                matching = [r for r in records
                            if r.get("scene_short", "") == scene and r.get("target", "") == cat]
                if matching:
                    suc = matching[0].get("success", 0)
                    row.append("OK" if suc > 0.5 else "X")
                else:
                    row.append("N/A")
            rows.append(row)
        print_table(name, headers, rows)

    # ── Generate Charts ──
    chart_overall(model_data, args.output_dir)
    chart_per_category(model_records, args.output_dir)
    chart_per_scene(model_records, args.output_dir)
    chart_heatmap(model_records, args.output_dir)

    print("\nAll charts saved to: %s" % args.output_dir)


if __name__ == "__main__":
    main()
