"""
Mask R-CNN R-101 FPN (COCO pretrained, 80 classes) for PEANUT.
Uses detectron2 model zoo weights directly - no finetuning.
Maps COCO 80-class predictions to PEANUT 9 categories.
"""
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# COCO class index -> PEANUT index (9 categories)
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

class SemanticPredR101COCO():
    """
    Mask R-CNN R-101-FPN 3x, pretrained on COCO (80 classes).
    Drop-in replacement for SemanticPredMaskRCNN.
    """
    def __init__(self, args):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = "cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu"
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.n_cats = 9
        self.predictor = DefaultPredictor(cfg)
        self.args = args
        print("[SemanticPredR101COCO] Mask R-CNN R-101-FPN-3x loaded (COCO 80 -> 9 PEANUT)")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """Run Mask R-CNN R-101 on RGB image, return (H,W,10) semantic tensor and BGR image."""
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
            if confscore < args.sem_pred_prob_thr:
                continue
            if peanut_idx == goal_cat:
                if confscore < args.goal_thr:
                    continue
            obj_mask = pred_instances.pred_masks[j].float()
            semantic_input[:, :, peanut_idx] += obj_mask

        return semantic_input.cpu().numpy(), img_bgr
