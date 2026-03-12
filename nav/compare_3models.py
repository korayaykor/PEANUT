#!/usr/bin/env python3
"""
compare_3models.py — Comprehensive comparison of Mask R-CNN, YOLOv8, and
Cascade Mask R-CNN for ObjectNav.

Generates:
  1. comprehensive_comparison.png  — Overall metrics bar chart
  2. per_category_comparison.png   — Per-category Success/SPL/Time bars
  3. per_episode_detail.png        — Per-episode step & SPL scatter
  4. time_comparison.png           — Time breakdown charts
  5. scene_category_heatmap.png    — Success heatmap (scene × category)
  6. Console tables with ALL requested metrics

Usage:
    python compare_3models.py \
        --maskrcnn_results  <path> \
        --yolo_results      <path> \
        --cascade_results   <path> \
        --output_dir        <path>
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

COLORS = {
    "Mask R-CNN":          "#3b82f6",
    "YOLOv8":              "#f97316",
    "Cascade Mask R-CNN":  "#10b981",
}


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
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
    """Compute all requested metrics from a list of episode records."""
    if not records:
        return {
            "num_episodes": 0,
            "total_steps": 0,
            "total_time": 0,
            "success_rate": 0,
            "avg_success_rate": 0,
            "spl": 0,
            "avg_spl": 0,
            "avg_step": 0,
            "avg_time": 0,
        }
    steps = [r.get("episode_length", 0) for r in records]
    times = [r.get("time", 0) for r in records]
    succs = [r.get("success", 0) for r in records]
    spls  = [r.get("spl", 0) for r in records]
    return {
        "num_episodes":    len(records),
        "total_steps":     int(np.sum(steps)),
        "total_time":      float(np.sum(times)),
        "success_rate":    float(np.mean(succs) * 100),
        "avg_success_rate": float(np.mean(succs) * 100),   # same as success_rate overall
        "spl":             float(np.mean(spls) * 100),
        "avg_spl":         float(np.mean(spls) * 100),
        "avg_step":        float(np.mean(steps)),
        "avg_time":        float(np.mean(times)),
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


# ──────────────────────────────────────────────────────────────
#  Chart 1: Comprehensive Overall Comparison
# ──────────────────────────────────────────────────────────────
def chart_comprehensive(model_data, output_dir):
    """
    Single figure with 6 subplots:
      Row 1: Success Rate (%), SPL (%), Avg SPL (%)
      Row 2: Avg Steps, Avg Time (s), Total Time (s)
    """
    metrics_grid = [
        [("success_rate", "Success Rate (%)", ".1f"),
         ("spl",          "SPL (%)",          ".1f"),
         ("avg_spl",      "Avg SPL (%)",      ".1f")],
        [("avg_step",     "Avg Steps",        ".1f"),
         ("avg_time",     "Avg Time / Ep (s)",".1f"),
         ("total_time",   "Total Time (s)",   ".1f")],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Comprehensive 3-Model Comparison  —  All Categories (Pretrained Weights)",
                 fontsize=16, fontweight="bold", y=0.98)

    model_names = [n for n, _ in model_data]
    colors = [COLORS.get(n, "#999") for n in model_names]

    for ri, row_metrics in enumerate(metrics_grid):
        for ci, (key, label, fmt) in enumerate(row_metrics):
            ax = axes[ri][ci]
            vals = [m[key] for _, m in model_data]
            x = np.arange(len(model_names))
            bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
            for bar, v in zip(bars, vals):
                ax.annotate(("%" + fmt) % v,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, fontsize=10, rotation=15, ha="right")
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "comprehensive_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


# ──────────────────────────────────────────────────────────────
#  Chart 2: Per-Category Comparison
# ──────────────────────────────────────────────────────────────
def chart_per_category(model_records, output_dir):
    """Per-category bar charts for Success, SPL, Avg Steps, Avg Time."""
    metrics = [
        ("success", "Success Rate (%)", True),
        ("spl",     "SPL (%)",          True),
        ("episode_length", "Avg Steps", False),
        ("time",    "Avg Time (s)",     False),
    ]
    n_models = len(model_records)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Per-Category Comparison (Pretrained Weights)",
                 fontsize=15, fontweight="bold", y=0.98)

    for ax_i, (metric_key, ylabel, is_pct) in enumerate(metrics):
        ax = axes[ax_i // 2][ax_i % 2]
        x = np.arange(len(ALL_CATEGORIES))
        width = 0.7 / n_models

        for i, (name, records) in enumerate(model_records):
            vals = []
            for cat in ALL_CATEGORIES:
                cat_recs = [r for r in records if r.get("target", "") == cat]
                if cat_recs:
                    v = np.mean([r.get(metric_key, 0) for r in cat_recs])
                    if is_pct:
                        v *= 100
                    vals.append(v)
                else:
                    vals.append(0)
            offset = (i - (n_models - 1) / 2) * width
            color = COLORS.get(name, "#999")
            bars = ax.bar(x + offset, vals, width, label=name, color=color,
                          edgecolor="white", linewidth=0.8)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate("%.0f" % h if h >= 1 else "%.1f" % h,
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CATEGORIES, fontsize=10, rotation=15)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "per_category_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


# ──────────────────────────────────────────────────────────────
#  Chart 3: Per-Episode Detail (scatter)
# ──────────────────────────────────────────────────────────────
def chart_per_episode(model_records, output_dir):
    """Per-episode scatter: Steps vs SPL, colored by model, marker by success."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Per-Episode Detail", fontsize=15, fontweight="bold", y=1.02)

    subtitles = ["Steps per Episode", "SPL per Episode", "Time per Episode (s)"]
    keys      = ["episode_length",    "spl",             "time"]

    for ax_i, (key, title) in enumerate(zip(keys, subtitles)):
        ax = axes[ax_i]
        for name, records in model_records:
            color = COLORS.get(name, "#999")
            vals = [r.get(key, 0) for r in records]
            succs = [r.get("success", 0) for r in records]
            xs = list(range(len(vals)))
            # Plot successes as circles, failures as X
            suc_x  = [j for j, s in zip(xs, succs) if s > 0.5]
            suc_v  = [v for v, s in zip(vals, succs) if s > 0.5]
            fail_x = [j for j, s in zip(xs, succs) if s <= 0.5]
            fail_v = [v for v, s in zip(vals, succs) if s <= 0.5]

            ax.scatter(suc_x, suc_v, c=color, marker="o", s=50, alpha=0.8,
                       label="%s (success)" % name, edgecolors="black", linewidths=0.5)
            ax.scatter(fail_x, fail_v, c=color, marker="x", s=50, alpha=0.5,
                       label="%s (fail)" % name)

        ax.set_xlabel("Episode #", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_episode_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % path)


# ──────────────────────────────────────────────────────────────
#  Chart 4: Time Comparison
# ──────────────────────────────────────────────────────────────
def chart_time(model_data, model_records, output_dir):
    """Dedicated time chart: total time, avg time, time per step."""
    model_names = [n for n, _ in model_data]
    colors = [COLORS.get(n, "#999") for n in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Time Comparison (Pretrained Weights)", fontsize=15, fontweight="bold", y=1.02)

    # Total Time
    vals = [m["total_time"] for _, m in model_data]
    axes[0].bar(model_names, vals, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(vals):
        axes[0].annotate("%.1fs" % v, xy=(i, v), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Total Time (s)", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Avg Time per Episode
    vals = [m["avg_time"] for _, m in model_data]
    axes[1].bar(model_names, vals, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(vals):
        axes[1].annotate("%.1fs" % v, xy=(i, v), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_title("Avg Time / Episode (s)", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    # Avg Time per Step
    vals = []
    for _, m in model_data:
        if m["avg_step"] > 0:
            vals.append(m["avg_time"] / m["avg_step"])
        else:
            vals.append(0)
    axes[2].bar(model_names, vals, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(vals):
        axes[2].annotate("%.3fs" % v, xy=(i, v), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    axes[2].set_title("Avg Time / Step (s)", fontsize=13, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(output_dir, "time_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % path)


# ──────────────────────────────────────────────────────────────
#  Chart 5: Scene × Category Heatmap
# ──────────────────────────────────────────────────────────────
def chart_heatmap(model_records, output_dir):
    all_scenes = sorted(set(
        r.get("scene_short", "") for _, recs in model_records for r in recs
    ))
    if not all_scenes:
        return
    short_names = [s.split("-")[-1][:10] if "-" in s else s[:10] for s in all_scenes]

    n_models = len(model_records)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, max(6, len(all_scenes) * 0.5 + 2)))
    if n_models == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list("gr", ["#fee2e2", "#fef9c3", "#bbf7d0"])

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
        ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        for si in range(len(all_scenes)):
            for ci in range(len(ALL_CATEGORIES)):
                val = data[si, ci]
                if np.isnan(val):
                    ax.text(ci, si, "N/A", ha="center", va="center", fontsize=8, color="gray")
                else:
                    label = "✓" if val > 0.5 else "✗"
                    color = "#166534" if val > 0.5 else "#991b1b"
                    ax.text(ci, si, label, ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)

        ax.set_xticks(range(len(ALL_CATEGORIES)))
        ax.set_xticklabels(ALL_CATEGORIES, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(all_scenes)))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(name, fontsize=12, fontweight="bold")

    fig.suptitle("Success per Scene × Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "scene_category_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % path)


# ──────────────────────────────────────────────────────────────
#  Console Tables
# ──────────────────────────────────────────────────────────────
def print_all_tables(model_data, model_records):

    # ── 1. Overall Comparison ──
    print("\n" + "=" * 80)
    print("  OVERALL COMPARISON  (Pretrained Weights Only)")
    print("=" * 80)
    headers = ["Metric"] + [n for n, _ in model_data]
    rows = []
    for key, label, fmt in [
        ("num_episodes",    "Episodes",               "d"),
        ("total_steps",     "Total Steps",             "d"),
        ("total_time",      "Total Time (s)",          ".1f"),
        ("avg_step",        "Avg Steps / Episode",     ".1f"),
        ("avg_time",        "Avg Time / Episode (s)",  ".2f"),
        ("success_rate",    "Success Rate (%)",        ".2f"),
        ("avg_success_rate","Avg Success Rate (%)",    ".2f"),
        ("spl",             "SPL (%)",                 ".2f"),
        ("avg_spl",         "Avg SPL (%)",             ".2f"),
    ]:
        row = [label]
        for _, m in model_data:
            row.append(("%" + fmt) % m[key])
        rows.append(row)
    print_table("Overall", headers, rows)

    # ── 2. Per-Category ──
    print("\n" + "=" * 80)
    print("  PER-CATEGORY COMPARISON")
    print("=" * 80)
    for cat in ALL_CATEGORIES:
        cat_data = []
        for name, records in model_records:
            cat_recs = [r for r in records if r.get("target", "") == cat]
            cat_data.append((name, compute_metrics(cat_recs)))

        if all(m["num_episodes"] == 0 for _, m in cat_data):
            continue

        headers = ["Metric"] + [n for n, _ in cat_data]
        rows = [
            ["Episodes"]       + [str(m["num_episodes"]) for _, m in cat_data],
            ["Success (%)"]    + ["%.1f" % m["success_rate"] for _, m in cat_data],
            ["SPL (%)"]        + ["%.1f" % m["spl"] for _, m in cat_data],
            ["Avg Steps"]      + ["%.1f" % m["avg_step"] for _, m in cat_data],
            ["Avg Time (s)"]   + ["%.2f" % m["avg_time"] for _, m in cat_data],
        ]
        print_table("Category: %s" % cat.upper(), headers, rows)

    # ── 3. Per-Episode Detail ──
    print("\n" + "=" * 80)
    print("  PER-EPISODE DETAIL")
    print("=" * 80)
    # Find the model with most episodes for indexing
    max_eps = max(len(recs) for _, recs in model_records)
    headers = ["Ep#"] 
    for name, _ in model_records:
        headers += ["%s Step" % name[:8], "%s SPL" % name[:8],
                    "%s Suc" % name[:8], "%s Time" % name[:8]]
    rows = []
    for ep_i in range(max_eps):
        row = [str(ep_i)]
        for name, records in model_records:
            if ep_i < len(records):
                r = records[ep_i]
                row.append(str(r.get("episode_length", "?")))
                row.append("%.3f" % r.get("spl", 0))
                row.append("%.0f" % r.get("success", 0))
                row.append("%.1f" % r.get("time", 0))
            else:
                row += ["N/A"] * 4
        rows.append(row)
    print_table("Per-Episode", headers, rows)

    # ── 4. Per-Scene Summary ──
    print("\n" + "=" * 80)
    print("  PER-SCENE SUCCESS RATE (%)")
    print("=" * 80)
    all_scenes = sorted(set(
        r.get("scene_short", "") for _, recs in model_records for r in recs
    ))
    headers = ["Scene"] + [n for n, _ in model_records]
    rows = []
    for scene in all_scenes:
        row = [scene.split("-")[-1][:14] if "-" in scene else scene[:14]]
        for name, records in model_records:
            scene_recs = [r for r in records if r.get("scene_short", "") == scene]
            if scene_recs:
                suc = np.mean([r.get("success", 0) for r in scene_recs]) * 100
                row.append("%.0f" % suc)
            else:
                row.append("N/A")
        rows.append(row)
    print_table("Per-Scene Success", headers, rows)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 3-model comparison: MaskRCNN vs YOLOv8 vs Cascade MaskRCNN")
    parser.add_argument("--maskrcnn_results", type=str, help="Mask R-CNN results JSONL")
    parser.add_argument("--yolo_results",     type=str, help="YOLOv8 results JSONL")
    parser.add_argument("--cascade_results",  type=str, help="Cascade Mask R-CNN results JSONL")
    parser.add_argument("--output_dir",       type=str, default="./data/tmp/3model_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_data    = []   # [(name, metrics_dict)]
    model_records = []   # [(name, [records])]

    for label, path in [
        ("Mask R-CNN",         args.maskrcnn_results),
        ("YOLOv8",             args.yolo_results),
        ("Cascade Mask R-CNN", args.cascade_results),
    ]:
        if path and os.path.exists(path):
            recs = load_results(path)
            print("Loaded %d %s records from %s" % (len(recs), label, path))
            model_data.append((label, compute_metrics(recs)))
            model_records.append((label, recs))
        else:
            print("[SKIP] %s results not found: %s" % (label, path))

    if not model_data:
        print("ERROR: No result files found!")
        sys.exit(1)

    # Console output
    print_all_tables(model_data, model_records)

    # Charts
    chart_comprehensive(model_data, args.output_dir)
    chart_per_category(model_records, args.output_dir)
    chart_per_episode(model_records, args.output_dir)
    chart_time(model_data, model_records, args.output_dir)
    chart_heatmap(model_records, args.output_dir)

    print("\n" + "=" * 60)
    print("All charts saved to: %s" % args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
