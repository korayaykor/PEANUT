"""
YOLO26-seg instance segmentation backend for PEANUT.

Drop-in replacement for SemanticPredMaskRCNN / SemanticPredYOLO / SemanticPredYOLO11.
Requires ultralytics >= 8.4.20 (Python 3.8+).
Supports COCO-pretrained models (80 classes) and fine-tuned models (9 classes).

Key improvements over YOLO11:
  - NMS-free end-to-end inference (no post-processing needed)
  - Semantic segmentation loss + multi-scale proto modules for better masks
  - Up to 43% faster CPU inference
"""

import torch
import torch.nn.functional as F
import numpy as np


class SemanticPredYOLO26:
    """
    YOLO26-seg based instance segmentation for PEANUT ObjectNav.

    Maps YOLO class predictions to PEANUT's 9 semantic categories:
        0: chair, 1: sofa, 2: plant, 3: bed, 4: toilet,
        5: tv_monitor, 6: fireplace, 7: bathtub, 8: mirror
    """

    def __init__(self, args):
        from ultralytics import YOLO

        self.args = args
        self.n_cats = 9  # PEANUT's 9 semantic categories

        # Model path: use yolo26_model_path if set, else fall back to seg_model_wts
        model_path = getattr(args, 'yolo26_model_path', None) or args.seg_model_wts
        self.model = YOLO(model_path)

        # Device
        self.device = f'cuda:{args.sem_gpu_id}' if torch.cuda.is_available() else 'cpu'

        # Build YOLO-class-index → PEANUT-index mapping.
        # Works for both COCO-pretrained (80 classes) and fine-tuned (9 classes).
        names = getattr(self.model, 'names', {})

        yolo_name_to_peanut = {
            # COCO-pretrained names
            'chair': 0,
            'couch': 1,
            'potted plant': 2,
            'bed': 3,
            'toilet': 4,
            'tv': 5,
            'dining table': 6,   # mapped to fireplace slot (6)
            'oven': 7,           # mapped to bathtub slot (7)
            'sink': 8,           # mapped to mirror slot (8)
            # Fine-tuned PEANUT names (aliases)
            'sofa': 1,
            'plant': 2,
            'tv_monitor': 5,
            'fireplace': 6,
            'bathtub': 7,
            'mirror': 8,
        }

        self.yolo_to_peanut = {}
        for yidx, yname in names.items():
            if yname in yolo_name_to_peanut:
                self.yolo_to_peanut[yidx] = yolo_name_to_peanut[yname]

        print(f"[SemanticPredYOLO26] Model: {model_path}")
        print(f"[SemanticPredYOLO26] ultralytics YOLO26, {len(names)} classes")
        print(f"[SemanticPredYOLO26] YOLO->PEANUT map: {self.yolo_to_peanut}")
        print(f"[SemanticPredYOLO26] Device: {self.device}")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """
        Run YOLO26-seg on an RGB image and return per-category mask tensor.

        Args:
            img: (H, W, 3) uint8 numpy array, RGB format
            depth: unused (kept for API compatibility)
            goal_cat: int, goal category index (applies stricter threshold)

        Returns:
            semantic_input: (H, W, n_cats+1) float numpy array
            img_bgr: (H, W, 3) uint8 numpy array in BGR format
        """
        args = self.args
        H, W = img.shape[0], img.shape[1]

        # Thresholds
        yolo_conf = getattr(args, 'yolo26_conf', 0.08)
        yolo_goal_conf = getattr(args, 'yolo26_goal_conf', 0.15)

        # Run inference — YOLO26 is natively end-to-end (NMS-free)
        # Use a low pre-filter to capture borderline detections
        results = self.model.predict(
            img,
            conf=max(0.005, yolo_conf * 0.3),
            device=self.device,
            verbose=False,
            retina_masks=True,   # full-resolution masks
        )

        semantic_input = torch.zeros(H, W, self.n_cats + 1, device=self.device)

        if len(results) > 0 and getattr(results[0], 'masks', None) is not None:
            result = results[0]
            boxes = result.boxes
            masks = result.masks

            num_instances = len(boxes) if boxes is not None else 0
            for j in range(num_instances):
                try:
                    yolo_cls = int(boxes.cls[j].item())
                    confscore = boxes.conf[j].item()
                except Exception:
                    continue

                if yolo_cls not in self.yolo_to_peanut:
                    continue

                peanut_idx = self.yolo_to_peanut[yolo_cls]

                # Apply threshold
                if confscore < yolo_conf:
                    continue
                if peanut_idx == goal_cat and confscore < yolo_goal_conf:
                    continue

                # Get mask and resize if needed
                mask = masks.data[j]
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                semantic_input[:, :, peanut_idx] += mask.to(self.device)

        img_bgr = img[:, :, ::-1]
        return semantic_input.cpu().numpy(), img_bgr
