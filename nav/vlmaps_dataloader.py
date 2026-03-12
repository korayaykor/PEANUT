"""
Offline replay of pre-recorded VLMaps frames through the PEANUT ObjectNav agent.

This script bypasses the Habitat simulator entirely:
  - Loads RGB, depth, and pose data from a VLMaps scene folder
  - Converts them to the exact observation format PEANUT expects
    (Habitat GPS sensor, compass sensor, depth normalization)
  - Feeds frames sequentially to the PEANUT agent
  - The agent builds its internal map, runs segmentation, and plans actions
    (actions are logged but don't control anything since it's a replay)

Usage (inside container):
  cd /nav
  python vlmaps_dataloader.py \
      --scene_dir /nav/vlmaps_data/5LpN3gDmAk7_1 \
      --target_object chair \
      --seg_model_type yolo \
      --num_frames 200 \
      --visualize 2
"""

import argparse
import os
import sys
import time
import json
import glob

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation as R
from PIL import Image

# ── Ensure /nav is on the path ──────────────────────────────────────────────
_nav_dir = os.path.dirname(os.path.abspath(__file__))
if _nav_dir not in sys.path:
    sys.path.insert(0, _nav_dir)

from arguments import get_args
from constants import hm3d_names, hm3d_to_coco, map_category_names, color_palette


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose conversion helpers  (exactly matches Habitat sensor implementations)
# ═══════════════════════════════════════════════════════════════════════════════

def quaternion_rotate_vector(quat_xyzw, v):
    """Rotate vector *v* by quaternion (x, y, z, w) using scipy."""
    r = R.from_quat(quat_xyzw)          # scipy expects [x,y,z,w]
    return r.apply(v)


def pose_to_gps(position, start_position, start_rotation_xyzw):
    """
    Replicate Habitat's EpisodicGPSSensor.get_observation (2-D mode).

    GPS = R_start^{-1} @ (position - start_position)
    Then return [-z_rotated, x_rotated]   (Habitat convention)
    """
    r_start_inv = R.from_quat(start_rotation_xyzw).inv()
    delta = np.array(position) - np.array(start_position)
    rotated = r_start_inv.apply(delta)
    return np.array([-rotated[2], rotated[0]], dtype=np.float32)


def pose_to_compass(rotation_xyzw, start_rotation_xyzw):
    """
    Replicate Habitat's EpisodicCompassSensor.get_observation.

    1) q_rel = q_agent^{-1} * q_start
    2) heading_vector = q_rel @ [0, 0, -1]
    3) phi = arctan2(heading_vector[0], -heading_vector[2])
    """
    q_agent = R.from_quat(rotation_xyzw)
    q_start = R.from_quat(start_rotation_xyzw)
    q_rel = q_agent.inv() * q_start                     # relative rotation
    direction = np.array([0.0, 0.0, -1.0])
    heading_vector = q_rel.apply(direction)
    phi = np.arctan2(heading_vector[0], -heading_vector[2])
    return np.array([phi], dtype=np.float32)


def depth_meters_to_habitat(depth_m, min_depth=0.5, max_depth=5.0):
    """
    Convert depth in **meters** to Habitat's normalised 0-1 range.

    Habitat depth sensor: depth_normalized = (depth_m - min_d) / (max_d - min_d)
    Values outside [min_d, max_d] are clipped.  Pixels with depth==0 stay 0.
    """
    out = np.zeros_like(depth_m, dtype=np.float32)
    valid = depth_m > 0
    out[valid] = (depth_m[valid] - min_depth) / (max_depth - min_depth)
    out = np.clip(out, 0.0, 1.0)
    # Pixels that were exactly 0 (invalid) stay 0
    out[~valid] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene data loader
# ═══════════════════════════════════════════════════════════════════════════════

class VLMapsSceneLoader:
    """Load pre-recorded RGB, depth, semantic and poses from a VLMaps folder."""

    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.rgb_dir = os.path.join(scene_dir, "rgb")
        self.depth_dir = os.path.join(scene_dir, "depth")
        self.sem_dir = os.path.join(scene_dir, "semantic")

        # Count frames
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        self.num_frames = len(self.rgb_files)
        assert self.num_frames > 0, f"No RGB frames in {self.rgb_dir}"

        # Load poses (x, y, z, qx, qy, qz, qw)  -- one per line
        poses_path = os.path.join(scene_dir, "poses.txt")
        self.poses = np.loadtxt(poses_path)
        assert len(self.poses) == self.num_frames, \
            f"Pose count {len(self.poses)} != frame count {self.num_frames}"

        # Start pose (frame 0) for GPS/compass reference
        self.start_position = self.poses[0, :3].copy()
        self.start_rotation = self.poses[0, 3:7].copy()     # qx, qy, qz, qw

        print(f"[VLMapsLoader] Scene: {os.path.basename(scene_dir)}")
        print(f"  Frames: {self.num_frames}")
        print(f"  Start position: {self.start_position}")
        print(f"  Start rotation: {self.start_rotation}")

    def load_frame(self, idx):
        """Return (rgb, depth_m, semantic, position, rotation) for frame *idx*."""
        rgb = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)      # H×W×3  uint8

        depth_path = os.path.join(self.depth_dir, f"{idx:06d}.npy")
        depth = np.load(depth_path).astype(np.float32)   # H×W  meters

        sem_path = os.path.join(self.sem_dir, f"{idx:06d}.npy")
        semantic = np.load(sem_path) if os.path.exists(sem_path) else None

        position = self.poses[idx, :3]
        rotation = self.poses[idx, 3:7]
        return rgb, depth, semantic, position, rotation

    def get_gps_compass(self, idx):
        """Return (gps, compass) for frame *idx* in Habitat convention."""
        position = self.poses[idx, :3]
        rotation = self.poses[idx, 3:7]
        gps = pose_to_gps(position, self.start_position, self.start_rotation)
        compass = pose_to_compass(rotation, self.start_rotation)
        return gps, compass


# ═══════════════════════════════════════════════════════════════════════════════
#  Build an observation dict identical to what Habitat returns
# ═══════════════════════════════════════════════════════════════════════════════

def make_observation(rgb, depth_m, gps, compass, objectgoal_id,
                     target_h=480, target_w=640,
                     min_depth=0.5, max_depth=5.0):
    """
    Convert raw VLMaps data → Habitat-format observation dict.

    Parameters
    ----------
    rgb        : (H, W, 3) uint8  – original resolution (e.g. 720×1080)
    depth_m    : (H, W) float32   – depth in metres
    gps        : (2,) float32     – Habitat GPS
    compass    : (1,) float32     – Habitat compass
    objectgoal_id : int           – HM3D goal index (0-5)
    target_h, target_w : int      – PEANUT env frame size (480, 640)
    min_depth, max_depth : float  – depth sensor range

    Returns
    -------
    obs : dict  with keys 'rgb', 'depth', 'gps', 'compass', 'objectgoal'
    """
    # Resize RGB  (original → env_frame)
    rgb_resized = cv2.resize(rgb, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)     # (H, W, 3) uint8

    # Convert & resize depth
    depth_norm = depth_meters_to_habitat(depth_m, min_depth, max_depth)  # (H, W) 0-1
    depth_resized = cv2.resize(depth_norm, (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST)  # (H, W)
    depth_resized = depth_resized[:, :, np.newaxis]               # (H, W, 1)

    obs = {
        "rgb": rgb_resized,                                # (H, W, 3) uint8
        "depth": depth_resized.astype(np.float32),         # (H, W, 1) float32
        "gps": gps,                                        # (2,) float32
        "compass": compass,                                # (1,) float32
        "objectgoal": np.array([objectgoal_id]),            # (1,) int
    }
    return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main offline replay
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def run_offline_replay(scene_dir, target_object, num_frames=None,
                       frame_step=1, peanut_args=None):
    """
    Feed pre-recorded frames through PEANUT agent.

    Parameters
    ----------
    scene_dir     : str   – path to VLMaps scene folder
    target_object : str   – one of hm3d_names values: chair, bed, plant, toilet, tv_monitor, sofa
    num_frames    : int   – how many frames to replay (None = all)
    frame_step    : int   – step between frames (1 = every frame)
    peanut_args   : Namespace – PEANUT arguments (from get_args())
    """

    # ── resolve target ──
    name_to_id = {v: k for k, v in hm3d_names.items()}
    if target_object not in name_to_id:
        raise ValueError(f"Unknown target '{target_object}'. "
                         f"Choose from {list(name_to_id.keys())}")
    objectgoal_id = name_to_id[target_object]
    print(f"\n{'='*60}")
    print(f"  Target object: {target_object}  (HM3D id={objectgoal_id})")
    print(f"{'='*60}\n")

    # ── load scene data ──
    loader = VLMapsSceneLoader(scene_dir)
    total = loader.num_frames
    if num_frames is not None:
        total = min(total, num_frames)
    frame_indices = list(range(0, total, frame_step))
    print(f"  Replaying {len(frame_indices)} frames (step={frame_step})\n")

    # ── build PEANUT args ──
    if peanut_args is None:
        peanut_args = get_args()

    # Override key settings for VLMaps data
    peanut_args.only_explore = 1       # no prediction model needed
    peanut_args.hfov = 90.0            # VLMaps uses 90° HFOV
    peanut_args.camera_height = 1.5    # VLMaps camera height
    peanut_args.min_depth = 0.5
    peanut_args.max_depth = 5.0
    peanut_args.env_frame_width = 640
    peanut_args.env_frame_height = 480
    peanut_args.timestep_limit = len(frame_indices) + 10  # don't auto-stop

    # Create output directory
    scene_name = os.path.basename(scene_dir.rstrip("/"))
    out_dir = os.path.join(peanut_args.dump_location,
                           f"vlmaps_replay_{scene_name}_{target_object}")
    os.makedirs(out_dir, exist_ok=True)
    peanut_args.dump_location = out_dir
    peanut_args.exp_name = f"vlmaps_{scene_name}"

    # ── Construct a minimal task_config (mock) ──
    class _Cfg:
        """Minimal mock of habitat.Config to satisfy PEANUT_Agent.__init__."""
        pass

    task_cfg = _Cfg()
    task_cfg.TASK = _Cfg()
    task_cfg.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    # ── Create agent ──
    from agent.peanut_agent import PEANUT_Agent
    agent = PEANUT_Agent(args=peanut_args, task_config=task_cfg)
    agent.reset()

    # ── Replay loop (agent never stops — STOP is overridden) ──
    results = []
    stop_count = 0
    t_start = time.time()

    for step_i, fidx in enumerate(frame_indices):
        rgb, depth_m, semantic, position, rotation = loader.load_frame(fidx)
        gps, compass = loader.get_gps_compass(fidx)

        obs = make_observation(
            rgb, depth_m, gps, compass, objectgoal_id,
            target_h=peanut_args.env_frame_height,
            target_w=peanut_args.env_frame_width,
            min_depth=peanut_args.min_depth,
            max_depth=peanut_args.max_depth,
        )

        action = agent.act(obs)
        act_id = action.get("action", -1)

        # ── Never stop: override STOP → MOVE_FORWARD so map keeps building ──
        if act_id == 0:
            stop_count += 1
            act_id = 1  # MOVE_FORWARD instead

        act_name = ACTION_NAMES.get(act_id, f"UNKNOWN({act_id})")

        if step_i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step_i:4d}/{len(frame_indices)} | Frame {fidx:5d} | "
                  f"Action: {act_name:14s} | "
                  f"GPS: ({gps[0]:+7.2f}, {gps[1]:+7.2f}) | "
                  f"Compass: {float(compass):+6.2f} rad | "
                  f"Time: {elapsed:.1f}s")
            sys.stdout.flush()

        results.append({
            "step": step_i,
            "frame_idx": fidx,
            "action": act_id,
            "action_name": act_name,
            "gps": gps.tolist(),
            "compass": float(compass),
        })

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Replay finished: {len(frame_indices)} steps in {elapsed:.1f}s "
          f"({len(frame_indices)/elapsed:.1f} fps)")
    print(f"  STOP suppressed {stop_count} times")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    #  Extract and save the semantic map from the agent's internal state
    # ══════════════════════════════════════════════════════════════════════════

    # Flush local map back to full map
    agent_state = agent.agent_states
    agent_state.full_map[:, agent_state.lmb[0]:agent_state.lmb[1],
                            agent_state.lmb[2]:agent_state.lmb[3]] = agent_state.local_map

    full_map = agent_state.full_map.cpu().numpy()   # (nc, H, W)
    # Channels: 0=obstacle, 1=explored, 2=cur_loc, 3=past_locs, 4..4+N=semantics
    n_sem = peanut_args.num_sem_categories
    obstacle_map  = full_map[0]          # (H, W)
    explored_map  = full_map[1]          # (H, W)
    trajectory_map = full_map[3]         # (H, W)
    semantic_channels = full_map[4:4+n_sem]  # (N, H, W)

    # Crop to explored region (with padding)
    explored_mask = explored_map > 0.5
    if explored_mask.any():
        rows = np.any(explored_mask, axis=1)
        cols = np.any(explored_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin, rmax = max(0, rmin - pad), min(full_map.shape[1], rmax + pad + 1)
        cmin, cmax = max(0, cmin - pad), min(full_map.shape[2], cmax + pad + 1)
    else:
        rmin, rmax, cmin, cmax = 0, full_map.shape[1], 0, full_map.shape[2]

    # Crop all channels
    obstacle_crop  = obstacle_map[rmin:rmax, cmin:cmax]
    explored_crop  = explored_map[rmin:rmax, cmin:cmax]
    traj_crop      = trajectory_map[rmin:rmax, cmin:cmax]
    sem_crop       = semantic_channels[:, rmin:rmax, cmin:cmax]   # (N, h, w)

    # Save raw full map numpy
    full_map_path = os.path.join(out_dir, "full_map.npy")
    np.save(full_map_path, full_map)
    print(f"  Saved full map tensor: {full_map_path}  shape={full_map.shape}")

    # Save cropped semantic map numpy
    sem_map_path = os.path.join(out_dir, "semantic_map.npy")
    np.save(sem_map_path, sem_crop)
    print(f"  Saved semantic map: {sem_map_path}  shape={sem_crop.shape}")

    # ── Build a colorized semantic map image ──
    h, w = sem_crop.shape[1], sem_crop.shape[2]

    # Category colors (RGB, 0-1)
    cat_colors = [
        (0.96, 0.36, 0.26),   # 0: chair       — red
        (0.12, 0.47, 0.71),   # 1: sofa        — blue
        (0.20, 0.80, 0.20),   # 2: plant       — green
        (0.94, 0.78, 0.66),   # 3: bed         — tan
        (0.94, 0.89, 0.26),   # 4: toilet      — yellow
        (0.66, 0.94, 0.85),   # 5: tv_monitor  — cyan
        (0.94, 0.50, 0.16),   # 6: fireplace   — orange
        (0.50, 0.00, 0.80),   # 7: bathtub     — purple
        (0.80, 0.80, 1.00),   # 8: mirror      — lavender
        (0.60, 0.60, 0.60),   # 9: other       — gray
    ]

    # Background: white=unexplored, light gray=explored free, dark gray=obstacle
    canvas = np.ones((h, w, 3), dtype=np.float32)   # white = unexplored
    exp_mask_crop = explored_crop > 0.5
    obs_mask_crop = obstacle_crop > 0.5
    canvas[exp_mask_crop] = [0.92, 0.92, 0.92]      # explored free space
    canvas[obs_mask_crop] = [0.45, 0.45, 0.45]      # obstacles

    # Paint trajectory
    traj_mask = traj_crop > 0.5
    canvas[traj_mask] = [0.70, 0.85, 1.0]            # light blue trajectory

    # Paint semantic categories on top
    legend_entries = []
    for cat_i in range(min(n_sem, len(cat_colors))):
        mask = sem_crop[cat_i] > 0.5
        if mask.any():
            canvas[mask] = cat_colors[cat_i]
            cat_name = map_category_names.get(cat_i, f"cat_{cat_i}")
            legend_entries.append((cat_name, cat_colors[cat_i]))

    # Flip vertically to match visual convention (origin at bottom)
    canvas = np.flipud(canvas)

    # ── Save with matplotlib (includes legend) ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(canvas, interpolation='nearest')
    ax.set_title(f"PEANUT Semantic Map — {scene_name}\n"
                 f"({len(frame_indices)} frames, {peanut_args.seg_model_type} seg, "
                 f"res={peanut_args.map_resolution}cm/px)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Map size: {w}×{h} cells  |  "
                  f"Resolution: {peanut_args.map_resolution} cm/cell")

    # Add fixed legend entries for structural elements
    struct_legend = [
        mpatches.Patch(color=[0.92, 0.92, 0.92], label='Free space'),
        mpatches.Patch(color=[0.45, 0.45, 0.45], label='Obstacle'),
        mpatches.Patch(color=[0.70, 0.85, 1.0],  label='Trajectory'),
    ]
    cat_legend = [mpatches.Patch(color=c, label=n) for n, c in legend_entries]
    ax.legend(handles=struct_legend + cat_legend,
              loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=11, framealpha=0.9)

    ax.set_aspect('equal')
    plt.tight_layout()
    png_path = os.path.join(out_dir, "semantic_map.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved semantic map image: {png_path}")

    # ── Also save a clean version in the scene directory ──
    scene_png = os.path.join(scene_dir, f"peanut_semantic_map_{peanut_args.seg_model_type}.png")
    scene_npy = os.path.join(scene_dir, f"peanut_semantic_map_{peanut_args.seg_model_type}.npy")
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    ax2.imshow(canvas, interpolation='nearest')
    ax2.set_title(f"PEANUT Semantic Map — {scene_name}\n"
                  f"{peanut_args.seg_model_type} | {len(frame_indices)} frames",
                  fontsize=14, fontweight='bold')
    struct_legend2 = [
        mpatches.Patch(color=[0.92, 0.92, 0.92], label='Free space'),
        mpatches.Patch(color=[0.45, 0.45, 0.45], label='Obstacle'),
        mpatches.Patch(color=[0.70, 0.85, 1.0],  label='Trajectory'),
    ]
    cat_legend2 = [mpatches.Patch(color=c, label=n) for n, c in legend_entries]
    ax2.legend(handles=struct_legend2 + cat_legend2,
               loc='upper left', bbox_to_anchor=(1.02, 1.0),
               fontsize=11, framealpha=0.9)
    ax2.set_aspect('equal')
    plt.tight_layout()
    fig2.savefig(scene_png, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    np.save(scene_npy, sem_crop)
    print(f"  Saved to scene dir: {scene_png}")
    print(f"  Saved to scene dir: {scene_npy}")

    # ── Per-category stats ──
    print(f"\n  Semantic map statistics:")
    print(f"    Map size: {full_map.shape[1]}×{full_map.shape[2]} → cropped {h}×{w}")
    print(f"    Explored cells: {int(exp_mask_crop.sum())}")
    print(f"    Obstacle cells: {int(obs_mask_crop.sum())}")
    total_sem = 0
    for cat_i in range(min(n_sem, len(cat_colors))):
        count = int((sem_crop[cat_i] > 0.5).sum())
        if count > 0:
            cat_name = map_category_names.get(cat_i, f"cat_{cat_i}")
            print(f"    {cat_name:15s}: {count:6d} cells")
            total_sem += count
    print(f"    {'TOTAL semantic':15s}: {total_sem:6d} cells")

    # ── Save results JSON ──
    results_path = os.path.join(out_dir, "replay_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "scene": scene_name,
            "target": target_object,
            "objectgoal_id": objectgoal_id,
            "num_frames": len(frame_indices),
            "frame_step": frame_step,
            "elapsed_time": elapsed,
            "stop_suppressed": stop_count,
            "args": {
                "hfov": peanut_args.hfov,
                "camera_height": peanut_args.camera_height,
                "min_depth": peanut_args.min_depth,
                "max_depth": peanut_args.max_depth,
                "seg_model_type": peanut_args.seg_model_type,
                "env_frame_width": peanut_args.env_frame_width,
                "env_frame_height": peanut_args.env_frame_height,
                "map_resolution": peanut_args.map_resolution,
                "map_size_cm": peanut_args.map_size_cm,
            },
            "semantic_map_shape": list(sem_crop.shape),
            "actions": results,
        }, f, indent=2)

    # ── Action summary ──
    from collections import Counter
    action_counts = Counter(r["action_name"] for r in results)
    print("\n  Action distribution:")
    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count} ({100*count/len(results):.1f}%)")
    if stop_count:
        print(f"    (STOP was suppressed {stop_count} times)")

    print(f"\n  Output directory: {out_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point: parse scene/target args, forward rest to PEANUT, then run replay."""
    # We parse our own args first, then forward the rest to PEANUT's get_args
    parser = argparse.ArgumentParser(
        description="Offline replay of VLMaps frames through PEANUT agent",
        add_help=False)
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to VLMaps scene folder")
    parser.add_argument("--target_object", type=str, default="chair",
                        help="Target object name: chair, bed, plant, toilet, tv_monitor, sofa")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of frames to replay (default: all)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Step between frames (1=every frame, 5=every 5th)")

    our_args, remaining = parser.parse_known_args()

    # Pass remaining args to PEANUT's arg parser
    sys.argv = [sys.argv[0]] + remaining
    peanut_args = get_args()

    run_offline_replay(
        scene_dir=our_args.scene_dir,
        target_object=our_args.target_object,
        num_frames=our_args.num_frames,
        frame_step=our_args.frame_step,
        peanut_args=peanut_args,
    )


if __name__ == "__main__":
    main()
