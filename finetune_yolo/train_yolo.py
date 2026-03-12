#!/usr/bin/env python
"""
Fine-tune YOLOv8-seg on the collected Habitat data.

Uses multi-GPU DDP training on both GPUs.

Usage (inside the container):
    conda run -n habitat python finetune_yolo/train_yolo.py \
        --data_dir /data/yolo_dataset_v2 \
        --base_weights /nav/yolov8x-seg.pt \
        --epochs 50 --batch 32 --device 0,1

Produces weights at:
    /data/yolo_runs/habitat_gt_finetune_v2/weights/best.pt
"""

import os
import sys
import argparse
import yaml


PEANUT_CATEGORIES = [
    'chair',       # 0
    'sofa',        # 1
    'plant',       # 2
    'bed',         # 3
    'toilet',      # 4
    'tv_monitor',  # 5
    'fireplace',   # 6
    'bathtub',     # 7
    'mirror',      # 8
]


def create_data_yaml(data_dir, output_path):
    """Create the YOLO data.yaml configuration file."""
    data_config = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(PEANUT_CATEGORIES),
        'names': PEANUT_CATEGORIES,
    }
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f'Created data config: {output_path}')
    return output_path


def count_dataset(data_dir):
    """Count images and labels in the dataset."""
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, 'images', split)
        lbl_dir = os.path.join(data_dir, 'labels', split)
        n_imgs = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        n_lbls = len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else 0
        print(f'  {split}: {n_imgs} images, {n_lbls} labels')


def main():
    """Parse args, create data config, and launch YOLO fine-tuning with DDP."""
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8-seg on Habitat data (multi-GPU)')
    parser.add_argument('--data_dir', type=str, default='/data/yolo_dataset_v2')
    parser.add_argument('--base_weights', type=str,
                        default='/nav/yolov8x-seg.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32,
                        help='Total batch size across all GPUs')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--freeze', type=int, default=15,
                        help='Freeze first N layers (15=backbone)')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='0,1',
                        help='GPU device(s) - use "0,1" for multi-GPU DDP')
    parser.add_argument('--project', type=str, default='/data/yolo_runs')
    parser.add_argument('--name', type=str, default='habitat_gt_finetune_v2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader workers per GPU')
    args = parser.parse_args()

    print('Dataset summary:')
    count_dataset(args.data_dir)

    val_img_dir = os.path.join(args.data_dir, 'images', 'val')
    if not os.path.isdir(val_img_dir) or len(os.listdir(val_img_dir)) == 0:
        print('\nWARNING: No validation data found!')

    data_yaml = os.path.join(args.data_dir, 'data.yaml')
    create_data_yaml(args.data_dir, data_yaml)

    from ultralytics import YOLO

    print(f'\nLoading base weights: {args.base_weights}')
    model = YOLO(args.base_weights)

    # Freeze backbone layers
    if args.freeze > 0:
        print(f'Freezing first {args.freeze} layers...')
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if i < args.freeze:
                param.requires_grad = False
        frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.model.parameters())
        print(f'  {frozen}/{total} parameters frozen')

    # Parse device: "0,1" -> [0,1] for multi-GPU DDP
    if ',' in args.device:
        device = [int(x) for x in args.device.split(',')]
    else:
        device = int(args.device)

    print(f'\nTraining config:')
    print(f'  epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}')
    print(f'  freeze={args.freeze}, lr0={args.lr0}')
    print(f'  device={device}, workers={args.workers}')
    print(f'  Output: {args.project}/{args.name}/')
    print('=' * 60)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        workers=args.workers,
        # ---- No visualisation ----
        plots=False,
        save=True,
        save_period=10,
        # ---- Data augmentation ----
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    best_wts = os.path.join(args.project, args.name, 'weights', 'best.pt')
    last_wts = os.path.join(args.project, args.name, 'weights', 'last.pt')

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'Best weights: {best_wts}')
    print(f'Last weights: {last_wts}')
    print(f'\nTo benchmark:')
    print(f'  python collect_all_categories.py --seg_model_type yolo \\')
    print(f'      --seg_model_wts {best_wts}')


if __name__ == '__main__':
    main()
