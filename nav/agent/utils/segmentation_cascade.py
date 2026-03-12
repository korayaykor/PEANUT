"""
Cascade Mask R-CNN X-152 segmentation model for PEANUT.

Uses the Detectron2 model zoo Cascade Mask R-CNN with ResNeXt-152 backbone,
pretrained on COCO (80 classes). COCO classes are mapped to the 9 PEANUT
categories at inference time.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


# COCO class index -> PEANUT index (9 categories)
# COCO indices: chair=56, couch=57, potted plant=58, bed=59,
#   dining table=60, toilet=61, tv=62, oven=69, sink=71
COCO_TO_PEANUT = {
    56: 0,   # chair
    57: 1,   # couch -> sofa
    58: 2,   # potted plant
    59: 3,   # bed
    61: 4,   # toilet
    62: 5,   # tv
    60: 6,   # dining table
    69: 7,   # oven
    71: 8,   # sink
}


class SemanticPredCascade():
    """
    Cascade Mask R-CNN X-152-32x8d-FPN (ImageNet-5k pretrained, COCO finetuned).
    Drop-in replacement for SemanticPredMaskRCNN.
    """

    def __init__(self, args):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
            )
        )
        # Use COCO-trained weights from model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # will filter further below
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = "cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu"
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.n_cats = 9   # match PEANUT category count
        self.predictor = DefaultPredictor(cfg)
        self.args = args
        print("[SemanticPredCascade] Cascade Mask R-CNN X-152 loaded (80 COCO classes -> 9 PEANUT)")
        print("[SemanticPredCascade] COCO->PEANUT map: %s" % COCO_TO_PEANUT)

    def get_prediction(self, img, depth=None, goal_cat=None):
        """Run Cascade Mask R-CNN on RGB image, return (H,W,10) semantic tensor and BGR image."""
        args = self.args

        img_bgr = img[:, :, ::-1]
        pred_instances = self.predictor(img_bgr)["instances"]

        semantic_input = torch.zeros(
            img.shape[0], img.shape[1], self.n_cats + 1,
            device=torch.device("cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu")
        )

        for j, coco_cls in enumerate(pred_instances.pred_classes.cpu().numpy()):
            coco_cls = int(coco_cls)
            if coco_cls not in COCO_TO_PEANUT:
                continue

            peanut_idx = COCO_TO_PEANUT[coco_cls]
            confscore = pred_instances.scores[j].item()

            # Apply thresholds matching the Mask R-CNN logic
            if confscore < args.sem_pred_prob_thr:
                continue
            if peanut_idx == goal_cat:
                if confscore < args.goal_thr:
                    continue

            obj_mask = pred_instances.pred_masks[j].float()
            semantic_input[:, :, peanut_idx] += obj_mask

        return semantic_input.cpu().numpy(), img_bgr
