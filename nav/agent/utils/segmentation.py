import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import argparse
import time
import numpy as np

from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.model_zoo import get_config
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
import detectron2.data.transforms as T


def debug_tensor(label, tensor):
    print(label, tensor.size(), tensor.mean().item(), tensor.std().item())


class SemanticPredMaskRCNN():

    def __init__(self, args):
        cfg = get_cfg()
        cfg.merge_from_file('nav/agent/utils/COCO-InstSeg/mask_rcnn_R_101_cat9.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.sem_pred_prob_thr
        cfg.MODEL.WEIGHTS = args.seg_model_wts
        cfg.MODEL.DEVICE = args.sem_gpu_id
        
        self.n_cats = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.predictor = DefaultPredictor(cfg)
        self.args = args

    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
    
        img = img[:, :, ::-1]
        pred_instances = self.predictor(img)["instances"]
        
        semantic_input = torch.zeros(img.shape[0], img.shape[1], self.n_cats + 1, device=args.sem_gpu_id)
        for j, class_idx in enumerate(pred_instances.pred_classes.cpu().numpy()):
            if class_idx in range(self.n_cats):
                idx = class_idx
                confscore = pred_instances.scores[j]
                
                # Higher threshold for target category
                if (confscore < args.sem_pred_prob_thr): 
                    continue
                if idx == goal_cat:
                    if confscore < args.goal_thr:
                        continue
                obj_mask = pred_instances.pred_masks[j] * 1.
                semantic_input[:, :, idx] += obj_mask
        
        return semantic_input.cpu().numpy(), img



class SemanticPredYOLO():
    """
    YOLO-based instance segmentation model (drop-in replacement for SemanticPredMaskRCNN).

    Uses ultralytics YOLOv8 segmentation model. The model outputs instance masks
    with COCO class IDs, which are mapped to the PEANUT category scheme via
    an internal name-to-index mapping.
    """

    def __init__(self, args):
        # Import here to avoid hard dependency at module import time
        try:
            from ultralytics import YOLO
        except Exception:
            raise ImportError('ultralytics is required for SemanticPredYOLO')

        self.args = args
        self.n_cats = 9  # match Mask R-CNN's 9 categories used in PEANUT

        # Load YOLO segmentation model (path provided via args.seg_model_wts)
        model_path = args.seg_model_wts
        self.model = YOLO(model_path)

        # Device string for ultralytics (either 'cpu' or 'cuda:0')
        self.device = f'cuda:{args.sem_gpu_id}' if torch.cuda.is_available() else 'cpu'

        # Build mapping from YOLO class indices to PEANUT indices using YOLO names.
        # Supports BOTH COCO-pretrained models (80 classes with COCO names)
        # AND finetuned models (9 classes with PEANUT names).
        names = getattr(self.model, 'names', {})
        yolo_name_to_peanut = {
            # COCO names (pretrained model)
            'chair': 0,
            'couch': 1,
            'potted plant': 2,
            'bed': 3,
            'toilet': 4,
            'tv': 5,
            'dining table': 6,
            'oven': 7,
            'sink': 8,
            # Finetuned PEANUT names (aliases)
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

        print(f"[SemanticPredYOLO] YOLO->PEANUT map: {self.yolo_to_peanut}")
        print(f"[SemanticPredYOLO] Model classes ({len(names)}): {names}")

    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
        H, W = img.shape[0], img.shape[1]

        # thresholds (can be tuned via args)
        yolo_conf = getattr(args, 'yolo_conf', 0.15)
        yolo_goal_conf = getattr(args, 'yolo_goal_conf', 0.5)

        # Run YOLO; use a low pre-filter then post-filter by our thresholds
        results = self.model.predict(
            img,
            conf=max(0.01, yolo_conf * 0.5),
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        semantic_input = torch.zeros(H, W, self.n_cats + 1, device=self.device)

        if len(results) > 0 and getattr(results[0], 'masks', None) is not None:
            result = results[0]
            boxes = getattr(result, 'boxes', None)
            masks = getattr(result, 'masks', None)

            # iterate instances
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

                if confscore < yolo_conf:
                    continue
                if peanut_idx == goal_cat and confscore < yolo_goal_conf:
                    continue

                # get mask, resize if necessary
                mask = masks.data[j]
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                semantic_input[:, :, peanut_idx] += mask.to(self.device)

        img_bgr = img[:, :, ::-1]
        return semantic_input.cpu().numpy(), img_bgr


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map
