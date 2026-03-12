#!/usr/bin/env python3
"""
Compare segmentation models for ObjectNav evaluation.

Reads per-episode JSONL result files produced by collect.py and generates
a grouped bar chart and per-episode heatmap comparing:
  - Success Rate (%)
  - SPL (%)
  - Average Episode Length
  - Average Time per Episode (s)

Usage:
    python compare_results.py \
        --maskrcnn_results ./data/tmp/maskrcnn_seg_results.txt \
        --yolo_results     ./data/tmp/yolo_seg_results.txt \
        --yolo_tuned_results ./data/tmp/yolo_tuned_results.txt \
        --output           ./data/tmp/comparison_chart.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt


def load_results(path):
    """Load per-episode result records (one JSON object per line)."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] skipping non-JSON line in {path}: {line[:80]}")
    return records


def compute_metrics(records):
    """Return dict with aggregated metrics from a list of episode records."""
    successes = [r.get('success', 0) for r in records]
    spls = [r.get('spl', 0) for r in records]
    ep_lens = [r.get('episode_length', 0) for r in records]
    times = [r.get('time', 0) for r in records]

    n = len(records)
    return {
        'num_episodes': n,
        'success': np.mean(successes) * 100 if n else 0,
        'spl': np.mean(spls) * 100 if n else 0,
        'avg_episode_length': np.mean(ep_lens) if n else 0,
        'avg_time': np.mean(times) if n else 0,
    }


def make_chart(model_data, output_path):
    """
    Generate a grouped bar chart comparing multiple models.
    model_data: list of (name, metrics_dict) tuples
    """
    labels = [
        'Success\nRate (%)',
        'SPL\n(%)',
        'Avg Episode\nLength',
        'Avg Time\nper Ep (s)',
    ]

    n_models = len(model_data)
    n_metrics = len(labels)
    colors = ['#3b82f6', '#f97316', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4']

    x = np.arange(n_metrics)
    total_bar_width = 0.75
    bar_width = total_bar_width / n_models

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, (name, metrics) in enumerate(model_data):
        values = [
            metrics['success'],
            metrics['spl'],
            metrics['avg_episode_length'],
            metrics['avg_time'],
        ]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=name,
                      color=colors[i % len(colors)], edgecolor='white', linewidth=0.8)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Segmentation Model Comparison  —  ObjectNav Evaluation',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    caption_parts = [f"{name}: {m['num_episodes']} eps" for name, m in model_data]
    fig.text(0.5, 0.01, '  |  '.join(caption_parts),
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    plt.close(fig)


def make_per_episode_chart(model_records, output_path):
    """Generate a per-episode success comparison heatmap."""
    all_episodes = set()
    for _, records in model_records:
        for r in records:
            all_episodes.add(r['episode'])
    episodes = sorted(all_episodes)
    if not episodes:
        return

    n_models = len(model_records)
    n_eps = len(episodes)

    fig, ax = plt.subplots(figsize=(max(8, n_eps * 0.8), 2 + n_models * 0.8))

    data = np.zeros((n_models, n_eps))
    targets = [''] * n_eps

    for i, (name, records) in enumerate(model_records):
        ep_map = {r['episode']: r for r in records}
        for j, ep in enumerate(episodes):
            if ep in ep_map:
                data[i, j] = ep_map[ep].get('success', 0)
                if not targets[j]:
                    targets[j] = ep_map[ep].get('target', '')

    cmap = plt.cm.colors.ListedColormap(['#fee2e2', '#bbf7d0'])
    ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    for i in range(n_models):
        for j in range(n_eps):
            val = 'OK' if data[i, j] > 0.5 else 'X'
            color = '#166534' if data[i, j] > 0.5 else '#991b1b'
            ax.text(j, i, val, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    ax.set_xticks(range(n_eps))
    ax.set_xticklabels([f"Ep{ep}\n({targets[j][:6]})" for j, ep in enumerate(episodes)], fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([name for name, _ in model_records], fontsize=10)
    ax.set_title('Per-Episode Success Comparison', fontsize=13, fontweight='bold')

    plt.tight_layout()
    per_ep_path = output_path.replace('.png', '_per_episode.png')
    fig.savefig(per_ep_path, dpi=150, bbox_inches='tight')
    print(f"Per-episode chart saved to {per_ep_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare segmentation model results')
    parser.add_argument('--maskrcnn_results', type=str, help='Path to Mask R-CNN results')
    parser.add_argument('--yolo_results', type=str, help='Path to YOLOv8 (default) results')
    parser.add_argument('--yolo_tuned_results', type=str, help='Path to YOLOv8 (tuned) results')
    parser.add_argument('--yolo_finetuned_results', type=str, help='Path to YOLOv8 (finetuned) results')
    parser.add_argument('--output', type=str, default='./data/tmp/comparison_chart.png',
                        help='Output path for the chart image')
    args = parser.parse_args()

    model_data = []       # (name, metrics_dict)
    model_records = []    # (name, records_list)

    if args.maskrcnn_results:
        print(f"Loading Mask R-CNN results from {args.maskrcnn_results}")
        r = load_results(args.maskrcnn_results)
        model_data.append(('Mask R-CNN', compute_metrics(r)))
        model_records.append(('Mask R-CNN', r))
    if args.yolo_results:
        print(f"Loading YOLOv8 (default) results from {args.yolo_results}")
        r = load_results(args.yolo_results)
        model_data.append(('YOLOv8 (default)', compute_metrics(r)))
        model_records.append(('YOLOv8 (default)', r))
    if args.yolo_tuned_results:
        print(f"Loading YOLOv8 (tuned) results from {args.yolo_tuned_results}")
        r = load_results(args.yolo_tuned_results)
        model_data.append(('YOLOv8 (tuned)', compute_metrics(r)))
        model_records.append(('YOLOv8 (tuned)', r))
    if args.yolo_finetuned_results:
        print(f"Loading YOLOv8 (finetuned) results from {args.yolo_finetuned_results}")
        r = load_results(args.yolo_finetuned_results)
        model_data.append(('YOLOv8 (finetuned)', compute_metrics(r)))
        model_records.append(('YOLOv8 (finetuned)', r))

    if not model_data:
        print("No results provided.")
        sys.exit(1)

    # Print summary table
    col_w = 18
    header = f"{'Metric':<25}" + ''.join(f"{name:>{col_w}}" for name, _ in model_data)
    sep = '=' * len(header)
    print(f'\n{sep}')
    print(header)
    print('-' * len(header))
    for key, label, fmt in [
        ('num_episodes', 'Episodes', 'd'),
        ('success', 'Success Rate (%)', '.2f'),
        ('spl', 'SPL (%)', '.2f'),
        ('avg_episode_length', 'Avg Episode Length', '.1f'),
        ('avg_time', 'Avg Time / Ep (s)', '.2f'),
    ]:
        row = f"{label:<25}"
        for _, m in model_data:
            row += f"{m[key]:>{col_w}{fmt}}"
        print(row)
    print(f'{sep}\n')

    # Generate charts
    make_chart(model_data, args.output)
    make_per_episode_chart(model_records, args.output)


if __name__ == '__main__':
    main()
