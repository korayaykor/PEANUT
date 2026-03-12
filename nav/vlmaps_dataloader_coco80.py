"""
Offline replay with ALL 80 COCO categories on the semantic map.

Instead of mapping YOLO's 80 classes down to PEANUT's 9 categories,
this script uses all 80 COCO classes directly. Each COCO class gets
its own map channel so the semantic map shows every detected object type.

Usage (inside container):
  cd /nav
  python vlmaps_dataloader_coco80.py \
      --scene_dir /nav/vlmaps_data/5LpN3gDmAk7_1 \
      --target_object chair \
      --seg_model_type yolo \
      --visualize 2 --only_explore 1 \
      -d /nav/data/tmp_coco80_yolo/
"""

import argparse
import os
import sys
import time
import json
import glob
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R
from PIL import Image

_nav_dir = os.path.dirname(os.path.abspath(__file__))
if _nav_dir not in sys.path:
    sys.path.insert(0, _nav_dir)

from arguments import get_args
from constants import hm3d_names, hm3d_to_coco

# ═══════════════════════════════════════════════════════════════════════════════
#  Full COCO-80 category names (matching ultralytics YOLO class index 0..79)
# ═══════════════════════════════════════════════════════════════════════════════

COCO_80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Build map_category_names for all 80 COCO categories
COCO80_MAP_NAMES = {i: name for i, name in enumerate(COCO_80_NAMES)}

# Categories that are impossible in indoor scenes (false positives)
OUTDOOR_FP_NAMES = {
    "train", "airplane", "boat", "horse", "cow", "sheep", "elephant",
    "bear", "zebra", "giraffe", "kite", "surfboard", "snowboard",
    "skateboard", "frisbee", "skis", "sports ball", "baseball bat",
    "baseball glove", "tennis racket", "fire hydrant", "stop sign",
    "parking meter", "traffic light",
}
OUTDOOR_FP_INDICES = {i for i, name in enumerate(COCO_80_NAMES) if name in OUTDOOR_FP_NAMES}

# ═══════════════════════════════════════════════════════════════════════════════
#  Generate 80 distinct colors for visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_80_colors():
    """Generate 80 visually distinct colors using HSV spacing."""
    import colorsys
    colors = []
    n = 80
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio spacing
        sat = 0.65 + 0.35 * ((i % 3) / 2.0)
        val = 0.75 + 0.25 * ((i % 5) / 4.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((r, g, b))
    return colors

COCO80_COLORS = _generate_80_colors()


# ═══════════════════════════════════════════════════════════════════════════════
#  Monkey-patched segmentation models that map ALL 80 COCO classes
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: biggest-mask-wins logic shared by all detectron2 models ──────────

def _detectron2_biggest_mask_wins(pred_instances, H, W, n_cats, cls_to_channel,
                                   conf_thr, device):
    """
    Given detectron2 Instances, build (H, W, n_cats+1) tensor with
    biggest-mask-wins assignment. `cls_to_channel` maps model class id → channel.
    """
    semantic_input = torch.zeros(H, W, n_cats + 1, device=device)
    best_area = torch.zeros(H, W, device=device)
    best_cat  = torch.full((H, W), -1, dtype=torch.long, device=device)
    max_area  = float(H * W)

    inst_info = []
    for j, cls_id in enumerate(pred_instances.pred_classes.cpu().numpy()):
        cls_id = int(cls_id)
        if cls_id not in cls_to_channel:
            continue
        confscore = pred_instances.scores[j].item()
        if confscore < conf_thr:
            continue
        mask = pred_instances.pred_masks[j].float().to(device)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False,
            ).squeeze(0).squeeze(0)
        mask_bool = mask > 0.5
        area = mask_bool.sum().item()
        if area < 1:
            continue
        ch = cls_to_channel[cls_id]
        inst_info.append((area, ch, mask_bool))

    inst_info.sort(key=lambda x: x[0])  # smallest first

    for area, ch, mask_bool in inst_info:
        better = mask_bool & (area > best_area)
        if better.any():
            old_cats = best_cat[better]
            for old_c in old_cats.unique():
                if old_c >= 0:
                    revert = better & (best_cat == old_c)
                    semantic_input[:, :, old_c.item()][revert] = 0.0
            area_value = 0.1 + 0.9 * (area / max_area) ** 0.5
            semantic_input[:, :, ch][better] = area_value
            best_area[better] = area
            best_cat[better] = ch

    return semantic_input


# ── MaskRCNN trained (9 PEANUT categories) ──────────────────────────────────

# The trained model only knows 9 classes. We map them to the corresponding
# COCO-80 channel indices so they appear in the right place on the 80-channel
# semantic map.
_PEANUT9_TO_COCO80 = {
    0: 56,   # chair
    1: 57,   # couch/sofa
    2: 58,   # potted plant
    3: 59,   # bed
    4: 61,   # toilet
    5: 62,   # tv
    6: 60,   # dining table
    7: 69,   # oven
    8: 71,   # sink
}


class SemanticPredMaskRCNN_COCO80:
    """Trained MaskRCNN R-101 (9-cat) mapped onto the 80-channel COCO map."""

    def __init__(self, args):
        """Load trained 9-class MaskRCNN weights and build class→channel mapping."""
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file('/nav/agent/utils/COCO-InstSeg/mask_rcnn_R_101_cat9.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.sem_pred_prob_thr
        cfg.MODEL.WEIGHTS = args.seg_model_wts  # mask_rcnn_R_101_cat9.pth
        cfg.MODEL.DEVICE = "cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu"

        self.n_cats = 80
        self.predictor = DefaultPredictor(cfg)
        self.args = args
        self.device_str = cfg.MODEL.DEVICE

        # Map: model class id (0-8) → COCO-80 channel
        self.cls_to_channel = dict(_PEANUT9_TO_COCO80)
        print(f"[MaskRCNN-trained COCO80] Loaded 9-cat weights, mapped to 80-channel map")
        print(f"[MaskRCNN-trained COCO80] cls→ch: {self.cls_to_channel}")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """Run trained MaskRCNN, return (H,W,81) biggest-mask-wins tensor and BGR image."""
        args = self.args
        H, W = img.shape[:2]
        img_bgr = img[:, :, ::-1]
        pred_instances = self.predictor(img_bgr)["instances"]
        device = torch.device(self.device_str)
        sem = _detectron2_biggest_mask_wins(
            pred_instances, H, W, self.n_cats, self.cls_to_channel,
            args.sem_pred_prob_thr, device)
        return sem.cpu().numpy(), img_bgr


# ── MaskRCNN pretrained (80 COCO classes) ───────────────────────────────────

class SemanticPredR101COCO_COCO80:
    """MaskRCNN R-101-FPN-3x pretrained on COCO, identity-mapped to 80 channels."""

    def __init__(self, args):
        """Load pretrained R-101-FPN-3x from model zoo with identity class mapping."""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = "cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu"
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.n_cats = 80
        self.predictor = DefaultPredictor(cfg)
        self.args = args
        self.device_str = cfg.MODEL.DEVICE

        # Identity map: COCO class i → channel i
        self.cls_to_channel = {i: i for i in range(80)}
        print(f"[MaskRCNN-pretrained COCO80] R-101-FPN-3x loaded (80→80)")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """Run pretrained R-101-FPN-3x, return (H,W,81) biggest-mask-wins tensor and BGR image."""
        args = self.args
        H, W = img.shape[:2]
        img_bgr = img[:, :, ::-1]
        pred_instances = self.predictor(img_bgr)["instances"]
        device = torch.device(self.device_str)
        sem = _detectron2_biggest_mask_wins(
            pred_instances, H, W, self.n_cats, self.cls_to_channel,
            0.3, device)
        return sem.cpu().numpy(), img_bgr


# ── Cascade Mask R-CNN pretrained (80 COCO classes) ─────────────────────────

class SemanticPredCascade_COCO80:
    """Cascade MaskRCNN X-152 pretrained on COCO, identity-mapped to 80 channels."""

    def __init__(self, args):
        """Load pretrained Cascade X-152 from model zoo with identity class mapping."""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = "cuda:%d" % args.sem_gpu_id if torch.cuda.is_available() else "cpu"
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        self.n_cats = 80
        self.predictor = DefaultPredictor(cfg)
        self.args = args
        self.device_str = cfg.MODEL.DEVICE

        self.cls_to_channel = {i: i for i in range(80)}
        print(f"[Cascade-pretrained COCO80] X-152 loaded (80→80)")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """Run Cascade X-152, return (H,W,81) biggest-mask-wins tensor and BGR image."""
        args = self.args
        H, W = img.shape[:2]
        img_bgr = img[:, :, ::-1]
        pred_instances = self.predictor(img_bgr)["instances"]
        device = torch.device(self.device_str)
        sem = _detectron2_biggest_mask_wins(
            pred_instances, H, W, self.n_cats, self.cls_to_channel,
            0.3, device)
        return sem.cpu().numpy(), img_bgr


# ── YOLO COCO-80 (shared by v8, v11, v26) ──────────────────────────────────

class SemanticPredYOLO_COCO80:
    """YOLO-based seg that keeps all 80 COCO classes (no collapsing to 9)."""

    def __init__(self, args):
        """Load YOLO seg model (v8/v11/v26) and build identity class mapping."""
        from ultralytics import YOLO
        self.args = args
        self.n_cats = 80

        model_path = getattr(args, '_coco80_model_path', args.seg_model_wts)
        self.model = YOLO(model_path)
        self.device = f'cuda:{args.sem_gpu_id}' if torch.cuda.is_available() else 'cpu'

        names = getattr(self.model, 'names', {})
        # Build identity mapping: YOLO class i → PEANUT channel i
        self.yolo_to_peanut = {}
        for yidx, yname in names.items():
            if yidx < 80:
                self.yolo_to_peanut[yidx] = yidx

        print(f"[COCO80] Model: {model_path}")
        print(f"[COCO80] {len(names)} YOLO classes → {len(self.yolo_to_peanut)} mapped to COCO80")
        print(f"[COCO80] Device: {self.device}")

    def get_prediction(self, img, depth=None, goal_cat=None):
        """
        Biggest-mask-wins per-pixel segmentation:
        Each pixel is assigned to the category whose instance mask is the
        LARGEST (most pixels). Bigger masks = closer/clearer view of the
        object, so they produce more accurate labels. Small noisy detections
        from far away get ignored when a bigger detection covers the same area.
        """
        args = self.args
        H, W = img.shape[:2]

        yolo_conf = getattr(args, '_coco80_conf', 0.15)

        results = self.model.predict(
            img,
            conf=max(0.01, yolo_conf * 0.5),
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        semantic_input = torch.zeros(H, W, self.n_cats + 1, device=self.device)

        # Track the biggest mask area that claimed each pixel
        best_area = torch.zeros(H, W, device=self.device)
        best_cat  = torch.full((H, W), -1, dtype=torch.long, device=self.device)

        # Max possible mask area for normalization
        max_area = float(H * W)

        if len(results) > 0 and getattr(results[0], 'masks', None) is not None:
            result = results[0]
            boxes = result.boxes
            masks = result.masks

            num_inst = len(boxes) if boxes is not None else 0

            # Pre-compute mask areas and sort: smallest first so bigger
            # masks overwrite smaller ones
            inst_info = []
            for j in range(num_inst):
                try:
                    yolo_cls = int(boxes.cls[j].item())
                    confscore = boxes.conf[j].item()
                except Exception:
                    continue

                if yolo_cls not in self.yolo_to_peanut:
                    continue
                if confscore < yolo_conf:
                    continue

                mask = masks.data[j]
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode='bilinear', align_corners=False,
                    ).squeeze(0).squeeze(0)

                mask_bool = (mask > 0.5)
                area = mask_bool.sum().item()
                if area < 1:
                    continue

                peanut_idx = self.yolo_to_peanut[yolo_cls]
                inst_info.append((area, peanut_idx, mask_bool.to(self.device)))

            # Sort by area ascending → biggest masks processed last → they win
            inst_info.sort(key=lambda x: x[0])

            for area, peanut_idx, mask_bool in inst_info:
                # This mask overwrites any pixel where it is bigger
                better = mask_bool & (area > best_area)
                if better.any():
                    # Remove old category for these pixels
                    old_cats = best_cat[better]
                    for old_c in old_cats.unique():
                        if old_c >= 0:
                            revert = better & (best_cat == old_c)
                            semantic_input[:, :, old_c.item()][revert] = 0.0

                    # Encode mask area as the channel value so that
                    # mapping.py's torch.max merge across frames naturally
                    # prefers the observation where the object was seen
                    # with the biggest mask (closest/best camera view).
                    # Scale: sqrt(area/max_area) to compress range while
                    # preserving ordering. Add small base (0.1) so even
                    # small detections register.
                    area_value = 0.1 + 0.9 * (area / max_area) ** 0.5
                    semantic_input[:, :, peanut_idx][better] = area_value
                    best_area[better] = area
                    best_cat[better] = peanut_idx

        return semantic_input.cpu().numpy(), img[:, :, ::-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose conversion helpers (same as vlmaps_dataloader.py)
# ═══════════════════════════════════════════════════════════════════════════════

def pose_to_gps(position, start_position, start_rotation_xyzw):
    """Convert absolute 3D position to 2D GPS relative to the start pose."""
    r_start_inv = R.from_quat(start_rotation_xyzw).inv()
    delta = np.array(position) - np.array(start_position)
    rotated = r_start_inv.apply(delta)
    return np.array([-rotated[2], rotated[0]], dtype=np.float32)

def pose_to_compass(rotation_xyzw, start_rotation_xyzw):
    """Convert absolute quaternion rotation to scalar compass heading relative to start."""
    q_agent = R.from_quat(rotation_xyzw)
    q_start = R.from_quat(start_rotation_xyzw)
    q_rel = q_agent.inv() * q_start
    direction = np.array([0.0, 0.0, -1.0])
    heading_vector = q_rel.apply(direction)
    phi = np.arctan2(heading_vector[0], -heading_vector[2])
    return np.array([phi], dtype=np.float32)

def depth_meters_to_habitat(depth_m, min_depth=0.5, max_depth=5.0):
    """Normalize raw depth (meters) to [0,1] range matching Habitat convention."""
    out = np.zeros_like(depth_m, dtype=np.float32)
    valid = depth_m > 0
    out[valid] = (depth_m[valid] - min_depth) / (max_depth - min_depth)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene data loader (same)
# ═══════════════════════════════════════════════════════════════════════════════

class VLMapsSceneLoader:
    """Load pre-recorded RGB, depth, and pose data from a VLMaps scene folder."""

    def __init__(self, scene_dir):
        """Initialize loader with scene directory containing rgb/, depth/, poses.txt."""
        self.scene_dir = scene_dir
        self.rgb_dir = os.path.join(scene_dir, "rgb")
        self.depth_dir = os.path.join(scene_dir, "depth")
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        self.num_frames = len(self.rgb_files)
        assert self.num_frames > 0
        poses_path = os.path.join(scene_dir, "poses.txt")
        self.poses = np.loadtxt(poses_path)
        assert len(self.poses) == self.num_frames
        self.start_position = self.poses[0, :3].copy()
        self.start_rotation = self.poses[0, 3:7].copy()
        print(f"[VLMapsLoader] {os.path.basename(scene_dir)}: {self.num_frames} frames")

    def load_frame(self, idx):
        """Return (rgb, depth_m, position, rotation) for frame idx."""
        rgb = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(self.depth_dir, f"{idx:06d}.npy")).astype(np.float32)
        position = self.poses[idx, :3]
        rotation = self.poses[idx, 3:7]
        return rgb, depth, position, rotation

    def get_gps_compass(self, idx):
        """Return (gps, compass) for frame idx in Habitat convention."""
        position = self.poses[idx, :3]
        rotation = self.poses[idx, 3:7]
        gps = pose_to_gps(position, self.start_position, self.start_rotation)
        compass = pose_to_compass(rotation, self.start_rotation)
        return gps, compass


def make_observation(rgb, depth_m, gps, compass, objectgoal_id,
                     target_h=480, target_w=640, min_depth=0.5, max_depth=5.0):
    """Build a Habitat-style observation dict from raw RGB, depth, GPS, compass."""
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    depth_norm = depth_meters_to_habitat(depth_m, min_depth, max_depth)
    depth_resized = cv2.resize(depth_norm, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    depth_resized = depth_resized[:, :, np.newaxis]
    return {
        "rgb": rgb_resized,
        "depth": depth_resized.astype(np.float32),
        "gps": gps,
        "compass": compass,
        "objectgoal": np.array([objectgoal_id]),
    }


ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Main offline replay with 80 COCO categories
# ═══════════════════════════════════════════════════════════════════════════════

def run_coco80_replay(scene_dir, target_object, peanut_args, num_frames=None, frame_step=1, filter_outdoor=False, min_votes=3):
    """Replay pre-recorded frames with ALL 80 COCO categories in the semantic map.

    Processes each frame through the chosen segmentation model (YOLO/MaskRCNN/Cascade),
    applies biggest-mask-wins per frame, cross-frame majority voting, 5×5 spatial mode
    smoothing, optional outdoor false-positive filtering, and min-votes thresholding.
    Saves final semantic map as .npy and .png.

    Args:
        scene_dir:       Path to VLMaps scene folder with rgb/, depth/, poses.txt.
        target_object:   HM3D goal object name (e.g. 'chair').
        peanut_args:     PEANUT argument namespace from arguments.py.
        num_frames:      Max frames to process (None = all).
        frame_step:      Process every N-th frame.
        filter_outdoor:  If True, zero-out outdoor/impossible categories.
        min_votes:       Minimum cross-frame votes to keep a cell label (default 3).
    """

    name_to_id = {v: k for k, v in hm3d_names.items()}
    if target_object not in name_to_id:
        raise ValueError(f"Unknown target '{target_object}'. Choose from {list(name_to_id.keys())}")
    objectgoal_id = name_to_id[target_object]

    print(f"\n{'='*60}")
    print(f"  COCO-80 Replay | Target: {target_object} (id={objectgoal_id})")
    print(f"  Seg model: {peanut_args.seg_model_type}")
    print(f"{'='*60}\n")

    loader = VLMapsSceneLoader(scene_dir)
    total = loader.num_frames if num_frames is None else min(loader.num_frames, num_frames)
    frame_indices = list(range(0, total, frame_step))

    # ── Override args for 80 categories ──
    peanut_args.num_sem_categories = 81  # 80 COCO classes + 1 "other"
    peanut_args.only_explore = 1
    peanut_args.hfov = 90.0
    peanut_args.camera_height = 1.5
    peanut_args.min_depth = 0.5
    peanut_args.max_depth = 5.0
    peanut_args.env_frame_width = 640
    peanut_args.env_frame_height = 480
    peanut_args.timestep_limit = len(frame_indices) + 10

    # ── Determine model path & fix seg_model_wts for YOLO types ──
    seg_type = peanut_args.seg_model_type
    if seg_type == 'yolo':
        model_path = peanut_args.seg_model_wts
        if not model_path.endswith('.pt'):
            model_path = os.path.join(_nav_dir, 'yolov8x-seg.pt')
        peanut_args.seg_model_wts = model_path  # fix for agent_helper init too
        peanut_args._coco80_model_path = model_path
        peanut_args._coco80_conf = getattr(peanut_args, 'yolo_conf', 0.15)
    elif seg_type == 'yolo11':
        peanut_args._coco80_model_path = getattr(peanut_args, 'yolo11_model_path', 'yolo11x-seg.pt')
        peanut_args._coco80_conf = getattr(peanut_args, 'yolo11_conf', 0.15)
    elif seg_type == 'yolo26':
        peanut_args._coco80_model_path = getattr(peanut_args, 'yolo26_model_path', 'yolo26x-seg.pt')
        peanut_args._coco80_conf = getattr(peanut_args, 'yolo26_conf', 0.08)
    elif seg_type in ('maskrcnn', 'maskrcnn_pretrained', 'cascade'):
        # Detectron2-based models — no YOLO-specific config needed
        pass
    else:
        raise ValueError(f"COCO-80 mode supports: yolo/yolo11/yolo26/maskrcnn/maskrcnn_pretrained/cascade, not '{seg_type}'")

    scene_name = os.path.basename(scene_dir.rstrip("/"))
    out_dir = os.path.join(peanut_args.dump_location,
                           f"coco80_replay_{scene_name}_{target_object}")
    os.makedirs(out_dir, exist_ok=True)
    peanut_args.dump_location = out_dir
    peanut_args.exp_name = f"coco80_{scene_name}"

    # ── Mock task config ──
    class _Cfg:
        pass
    task_cfg = _Cfg()
    task_cfg.TASK = _Cfg()
    task_cfg.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    # ── Create PEANUT agent (with 81 sem channels) ──
    from agent.peanut_agent import PEANUT_Agent
    agent = PEANUT_Agent(args=peanut_args, task_config=task_cfg)
    agent.reset()

    # ── Replace the segmentation model with our COCO-80 version ──
    if seg_type in ('yolo', 'yolo11', 'yolo26'):
        coco80_seg = SemanticPredYOLO_COCO80(peanut_args)
    elif seg_type == 'maskrcnn':
        coco80_seg = SemanticPredMaskRCNN_COCO80(peanut_args)
    elif seg_type == 'maskrcnn_pretrained':
        coco80_seg = SemanticPredR101COCO_COCO80(peanut_args)
    elif seg_type == 'cascade':
        coco80_seg = SemanticPredCascade_COCO80(peanut_args)
    else:
        raise ValueError(f"Unknown seg_model_type: {seg_type}")
    # Monkey-patch the agent's helper to use our 80-class seg model
    agent.agent_helper.seg_model = coco80_seg
    agent.agent_helper.seg_model.n_cats = 80

    print(f"  Replaying {len(frame_indices)} frames (step={frame_step})\n")

    # ── Replay loop ──
    results = []
    stop_count = 0
    t_start = time.time()

    for step_i, fidx in enumerate(frame_indices):
        rgb, depth_m, position, rotation = loader.load_frame(fidx)
        gps, compass = loader.get_gps_compass(fidx)

        obs = make_observation(rgb, depth_m, gps, compass, objectgoal_id,
                               target_h=peanut_args.env_frame_height,
                               target_w=peanut_args.env_frame_width,
                               min_depth=peanut_args.min_depth,
                               max_depth=peanut_args.max_depth)

        action = agent.act(obs)
        act_id = action.get("action", -1)

        if act_id == 0:
            stop_count += 1
            act_id = 1

        act_name = ACTION_NAMES.get(act_id, f"UNKNOWN({act_id})")

        if step_i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step_i:4d}/{len(frame_indices)} | Frame {fidx:5d} | "
                  f"Action: {act_name:14s} | Time: {elapsed:.1f}s")
            sys.stdout.flush()

        results.append({"step": step_i, "frame_idx": fidx, "action": act_id, "action_name": act_name})

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Replay done: {len(frame_indices)} steps in {elapsed:.1f}s "
          f"({len(frame_indices)/elapsed:.1f} fps)")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  Extract & save semantic map
    # ══════════════════════════════════════════════════════════════════════════

    agent_state = agent.agent_states
    agent_state.full_map[:, agent_state.lmb[0]:agent_state.lmb[1],
                            agent_state.lmb[2]:agent_state.lmb[3]] = agent_state.local_map
    agent_state.full_vote_map[:, agent_state.lmb[0]:agent_state.lmb[1],
                                agent_state.lmb[2]:agent_state.lmb[3]] = agent_state.local_vote_map

    full_map = agent_state.full_map.cpu().numpy()
    full_vote = agent_state.full_vote_map.cpu().numpy()  # (num_sem, H, W)
    n_sem = 80  # we use 80 COCO categories
    obstacle_map = full_map[0]
    explored_map = full_map[1]
    trajectory_map = full_map[3]
    semantic_channels = full_map[4:4+n_sem]  # (80, H, W)
    vote_channels = full_vote[:n_sem]  # (80, H, W) — vote counts per category

    # Crop to explored region
    explored_mask = explored_map > 0.5
    if explored_mask.any():
        rows = np.any(explored_mask, axis=1)
        cols = np.any(explored_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(full_map.shape[1], rmax + pad + 1)
        cmin = max(0, cmin - pad)
        cmax = min(full_map.shape[2], cmax + pad + 1)
    else:
        rmin, rmax = 0, full_map.shape[1]
        cmin, cmax = 0, full_map.shape[2]

    obstacle_crop = obstacle_map[rmin:rmax, cmin:cmax]
    explored_crop = explored_map[rmin:rmax, cmin:cmax]
    traj_crop = trajectory_map[rmin:rmax, cmin:cmax]
    sem_crop = semantic_channels[:, rmin:rmax, cmin:cmax]
    vote_crop = vote_channels[:, rmin:rmax, cmin:cmax]   # (80, h, w)

    # ── Filter outdoor / impossible categories ──
    if filter_outdoor:
        removed_names = []
        for fp_idx in sorted(OUTDOOR_FP_INDICES):
            fp_cells = int((vote_crop[fp_idx] > 0).sum())
            if fp_cells > 0:
                removed_names.append(f"{COCO_80_NAMES[fp_idx]}({fp_cells})")
            vote_crop[fp_idx] = 0
            sem_crop[fp_idx] = 0
        if removed_names:
            print(f"  [FP filter] Removed {len(removed_names)} outdoor categories: {', '.join(removed_names)}")
        else:
            print(f"  [FP filter] No outdoor false positives found.")

    # ══════════════════════════════════════════════════════════════════════════
    # Per-orit:y-veewinningsct-ogory'e vosesciunt must mget:min_s
   max_vtes = vote_cop.max(axis=0)  # (h, w) —mx vote cout each cll
    #  Form =  ax_votese>acmin_h ces   # (h, w)

    beforelfilte, = int((max_v tes > 0)hsue())
     fter_filter = intahbs_sem.sum())
    print(f"  [mln_vote i{min_votes}] {before_filter} cells had any vote, "
       tecf"{after_filter}asurvivedeg{before_filter - after_filter}yremoved)"that was observed the most
    #  times across all frames (highest vote count). Ties broken by the
    #  accumulated semantic value (area-based, from get_prediction).
    # ══════════════════════════════════════════════════════════════════════════

    h, w = sem_crop.shape[1], sem_crop.shape[2]

    # Per-cell: the winning category's vote count must meet min_votes
    max_votes = vote_crop.max(axis=0)  # (h, w) — max vote count at each cell
    has_sem = max_votes >= min_votes   # (h, w)

    before_filter = int((max_votes > 0).sum())
    after_filter = int(has_sem.sum())
    print(f"  [min_votes={min_votes}] {before_filter} cells had any vote, "
          f"{after_filter} survived ({before_filter - after_filter} removed)")

    # Majority vote: argmax of vote counts → winning category per cell
    vote_winner = np.argmax(vote_crop, axis=0)  # (h, w) — category index

    # Build clean single-label semantic map from vote winner (vectorized)
    sem_voted = np.zeros_like(sem_crop)   # (80, h, w)
    rows_sem, cols_sem = np.where(has_sem)
    cats_sem = vote_winner[rows_sem, cols_sem]
    sem_voted[cats_sem, rows_sem, cols_sem] = 1.0

    # ── Spatial mode filter for neighborhood consistency ──
    # For each cell, look at a 5×5 neighborhood and assign the label
    # that appears most frequently among neighbors (mode filter).
    from scipy.ndimage import generic_filter

    # Convert to label map: -1 = no label, 0..79 = category
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[has_sem] = vote_winner[has_sem]

    def _spatial_mode(values):
        """Return the most common non-negative label in the window."""
        valid = values[values >= 0]
        if len(valid) == 0:
            return -1
        counts = np.bincount(valid.astype(np.int32), minlength=n_sem)
        return int(np.argmax(counts))

    label_smoothed = generic_filter(
        label_map.astype(np.float64), _spatial_mode, size=5, mode='constant', cval=-1
    ).astype(np.int32)

    # Only keep smoothed labels where there was already a semantic detection
    # (don't create new labels where none existed)
    label_smoothed[~has_sem] = -1

    # Rebuild one-hot semantic from smoothed label map (vectorized)
    sem_final = np.zeros_like(sem_crop)   # (80, h, w)
    valid_mask = label_smoothed >= 0
    rows_v, cols_v = np.where(valid_mask)
    cats_v = label_smoothed[rows_v, cols_v]
    sem_final[cats_v, rows_v, cols_v] = 1.0

    # Use the voted + smoothed result for visualization
    sem_crop = sem_final

    print(f"\n  Majority-vote post-processing:")
    print(f"    Cells with any semantic: {int(has_sem.sum())}")
    print(f"    After spatial smoothing (5×5 mode filter): "
          f"{int((label_smoothed >= 0).sum())} cells")

    # Save numpy arrays
    np.save(os.path.join(out_dir, "full_map.npy"), full_map)
    np.save(os.path.join(out_dir, "vote_map_coco80.npy"), vote_crop)
    np.save(os.path.join(out_dir, "semantic_map_coco80.npy"), sem_crop)
    print(f"  Saved full_map: shape={full_map.shape}")
    print(f"  Saved vote_map_coco80: shape={vote_crop.shape}")
    print(f"  Saved semantic_map_coco80: shape={sem_crop.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Build colorized semantic map with ALL COCO categories legend
    # ══════════════════════════════════════════════════════════════════════════

    h, w = sem_crop.shape[1], sem_crop.shape[2]

    # Background
    canvas = np.ones((h, w, 3), dtype=np.float32)
    exp_mask = explored_crop > 0.5
    obs_mask = obstacle_crop > 0.5
    canvas[exp_mask] = [0.92, 0.92, 0.92]
    canvas[obs_mask] = [0.45, 0.45, 0.45]
    traj_mask = traj_crop > 0.5
    canvas[traj_mask] = [0.70, 0.85, 1.0]

    # Paint semantic categories
    detected_cats = []
    cat_cell_counts = {}
    for cat_i in range(n_sem):
        mask = sem_crop[cat_i] > 0.5
        count = int(mask.sum())
        if count > 0:
            canvas[mask] = COCO80_COLORS[cat_i]
            detected_cats.append(cat_i)
            cat_cell_counts[cat_i] = count

    canvas = np.flipud(canvas)

    # ── Create figure with map + legend ──
    # Sort detected categories by cell count (largest first)
    detected_sorted = sorted(detected_cats, key=lambda c: cat_cell_counts[c], reverse=True)
    n_detected = len(detected_sorted)

    total_sem = sum(cat_cell_counts.values())

    # Layout: map on the left, legend on the right
    fig = plt.figure(figsize=(20, 14))

    # If many categories, need more legend space
    legend_width_ratio = 0.35 if n_detected > 20 else 0.25
    gs = gridspec.GridSpec(1, 2, width_ratios=[1 - legend_width_ratio, legend_width_ratio],
                           wspace=0.02, left=0.02, right=0.98, top=0.92, bottom=0.02)

    # Map
    ax_map = fig.add_subplot(gs[0])
    ax_map.imshow(canvas, interpolation='nearest')
    ax_map.set_title(f"PEANUT Semantic Map — {scene_name}\n"
                     f"{seg_type.upper()} | {len(frame_indices)} frames | "
                     f"Majority-vote + 5×5 spatial | {n_detected} detected | {total_sem:,} cells",
                     fontsize=13, fontweight='bold')
    ax_map.set_aspect('equal')
    ax_map.axis('off')

    # Legend panel
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    # Structural legend at top
    struct_items = [
        ([0.92, 0.92, 0.92], "Free space"),
        ([0.45, 0.45, 0.45], "Obstacle"),
        ([0.70, 0.85, 1.0], "Trajectory"),
    ]

    y_pos = 0.98
    line_h = 0.022  # height per legend line

    ax_leg.text(0.0, y_pos, "Structure:", fontsize=11, fontweight='bold',
                va='top', transform=ax_leg.transAxes)
    y_pos -= line_h * 1.2
    for color, label in struct_items:
        rect = mpatches.FancyBboxPatch((0.02, y_pos - line_h * 0.8), 0.06, line_h * 0.7,
                                        boxstyle="round,pad=0.002",
                                        facecolor=color, edgecolor='gray', linewidth=0.5,
                                        transform=ax_leg.transAxes)
        ax_leg.add_patch(rect)
        ax_leg.text(0.10, y_pos - line_h * 0.35, label, fontsize=9, va='center',
                    transform=ax_leg.transAxes)
        y_pos -= line_h

    y_pos -= line_h * 0.5
    ax_leg.text(0.0, y_pos, f"Detected Categories ({n_detected}):", fontsize=11,
                fontweight='bold', va='top', transform=ax_leg.transAxes)
    y_pos -= line_h * 1.2

    # If many categories, use two columns
    if n_detected > 25:
        col_width = 0.5
        items_per_col = (n_detected + 1) // 2
    else:
        col_width = 1.0
        items_per_col = n_detected

    col = 0
    row_in_col = 0
    y_start = y_pos

    for rank, cat_i in enumerate(detected_sorted):
        cat_name = COCO_80_NAMES[cat_i]
        count = cat_cell_counts[cat_i]
        color = COCO80_COLORS[cat_i]

        x_offset = col * col_width
        y = y_start - row_in_col * line_h

        if y < 0.01 and col == 0 and n_detected > 25:
            # Switch to second column
            col = 1
            row_in_col = 0
            y = y_start

        rect = mpatches.FancyBboxPatch((x_offset + 0.02, y - line_h * 0.8), 0.04, line_h * 0.7,
                                        boxstyle="round,pad=0.002",
                                        facecolor=color, edgecolor='gray', linewidth=0.5,
                                        transform=ax_leg.transAxes)
        ax_leg.add_patch(rect)
        ax_leg.text(x_offset + 0.08, y - line_h * 0.35,
                    f"{cat_name} ({count:,})", fontsize=8, va='center',
                    transform=ax_leg.transAxes)

        row_in_col += 1
        if row_in_col >= items_per_col and col == 0 and n_detected > 25:
            col = 1
            row_in_col = 0

    # Total at bottom
    ax_leg.text(0.0, 0.0, f"Total: {total_sem:,} semantic cells",
                fontsize=10, fontweight='bold', va='bottom', transform=ax_leg.transAxes)

    png_path = os.path.join(out_dir, "semantic_map_coco80.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # Also save to scene directory
    scene_png = os.path.join(scene_dir, f"peanut_coco80_{seg_type}.png")
    import shutil
    shutil.copy2(png_path, scene_png)
    print(f"  Saved to scene dir: {scene_png}")

    # ── Stats ──
    print(f"\n  Semantic map statistics (COCO-80):")
    print(f"    Map size: {full_map.shape[1]}x{full_map.shape[2]} → cropped {h}x{w}")
    print(f"    Explored cells: {int(exp_mask.sum())}")
    print(f"    Obstacle cells: {int(obs_mask.sum())}")
    for cat_i in detected_sorted:
        print(f"    {COCO_80_NAMES[cat_i]:20s}: {cat_cell_counts[cat_i]:6d} cells")
    print(f"    {'TOTAL':20s}: {total_sem:6d} cells")
    print(f"    STOP suppressed: {stop_count} times")
    print(f"\n  Output directory: {out_dir}")

    return detected_cats, cat_cell_counts, out_dir.)")
    parseradd_argument("--min_votes", type=int, default=3,
                        help="Min cross-frame votes to keep a cell label (default 3


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point: parse scene_dir, target_object, filter flags, then run replay."""
    parser = argparse.ArgumentParser(description="COCO-80 offline replay", add_help=False)
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--target_object", type=str, default="chair")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--filter_outdoor", type=int, default=0,
                        help="1 = suppress outdoor/impossible categories (train, airplane, etc.)")
    parser.add_argument("--min_votes", type=int, default=3,
                        help="Min cross-frame votes to keep a cell label (default 3)")
    our_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    peanut_args = get_args()

    run_coco80_replay(
        scene_dir=our_args.scene_dir,
        target_object=our_args.target_object,
        peanut_args=peanut_args,
        num_frames=our_args.num_frames,
        frame_step=our_args.frame_step,
        filter_outdoor=bool(our_args.filter_outdoor),
        min_votes=our_args.min_votes,
    )


if __name__ == "__main__":
    main()
