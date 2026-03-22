This agent is an embodied AI agent for **Object Goal Navigation (ObjectNav)** in indoor environments. Given a target object category (e.g., "bed", "toilet", "tv"), the agent navigates through an unseen 3D environment to find and reach an instance of that object.

The agent operates as a modular pipeline:

1. **Perception** — Semantic segmentation of RGB frames (multiple swappable backends)
2. **Mapping** — RGB-D observations projected into a top-down 2D semantic map
3. **Prediction** — An encoder-decoder network predicts likely locations of unseen target objects from the partial map
4. **Planning** — Fast Marching Method (FMM) planner converts goal maps into discrete actions

Built on the [Habitat](https://aihabitat.org/) simulator and designed for the Habitat ObjectNav Challenge on HM3D scenes.

## Architecture

```
Observations (RGB-D, GPS, Compass, Goal)
        │
        ▼
┌─────────────────┐
│   Perception    │  Mask R-CNN / YOLOv8 / YOLO11 / YOLO26 / Cascade / Grounded-SAM
│  (Segmentation) │  Selected via --seg_model_type
└────────┬────────┘
         │ per-pixel semantic labels
         ▼
┌─────────────────┐
│ Semantic Mapping │  Depth → Point Cloud → Voxel Splatting → 2D Grid
│   (nn.Module)   │  Multi-channel: obstacles, explored, trajectory, semantics
└────────┬────────┘
         │ partial semantic map
         ▼
┌─────────────────┐
│   Prediction    │  ResNet50 + PSPHead (MMSegmentation)
│    Network      │  14-channel input → 6-channel target probability map
└────────┬────────┘
         │ goal location
         ▼
┌─────────────────┐
│    Planning     │  FMM shortest path → deterministic local policy
│   (FMMPlanner)  │  Actions: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
└─────────────────┘
```

### Segmentation Backends

The perception module supports multiple segmentation models, selected at runtime via `--seg_model_type`:

| Backend | Flag | Description |
|---------|------|-------------|
| Mask R-CNN (R-101) | `maskrcnn` | Detectron2, fine-tuned on 9 PEANUT categories |
| Mask R-CNN (Pretrained) | `maskrcnn_pretrained` | Detectron2, COCO-pretrained |
| YOLOv8x-seg | `yolo` | Ultralytics YOLOv8 instance segmentation |
| YOLO11x-seg | `yolo11` | Ultralytics YOLO11 instance segmentation |
| YOLO26x-seg | `yolo26` | Ultralytics YOLO26 instance segmentation |
| Cascade R-CNN | `cascade` | Detectron2 Cascade instance segmentation |
| R-101 COCO | `r101coco` | Detectron2 R-101 with full COCO-80 categories |
| Grounded-SAM | `grounded_sam` | GroundingDINO + SAM (open-vocabulary) |

## Requirements

### Platform

- **Linux only** (required by Habitat simulator + nvidia-docker)
- NVIDIA GPU with CUDA 11.1+ support
- [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) (NVIDIA Container Toolkit)
- Docker with BuildKit support

### File Setup

Download and place the following files before building:

1. **HM3D Scenes**
   - [train](https://api.matterport.com/resources/habitat/hm3d-train-habitat.tar) and [val](https://api.matterport.com/resources/habitat/hm3d-val-habitat.tar) splits
   - Extract into `habitat-challenge-data/data/scene_datasets/hm3d/{train,val}/`

2. **ObjectNav Episode Dataset**
   - [Download](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip)
   - Extract into `habitat-challenge-data/` so that `habitat-challenge-data/objectgoal_hm3d/val/` exists

3. **Model Weights**
   - [Mask R-CNN weights](https://drive.google.com/file/d/1tJ9MFK6Th7SY1iJTPrtpOmXNHB4ztPxC/view?usp=share_link) → `nav/agent/utils/mask_rcnn_R_101_cat9.pth`
   - [Prediction network weights](https://drive.google.com/file/d/1Xvly65BKVyy92Ja5GL7YwxryDrsnyO05/view?usp=share_link) → `nav/pred_model_wts.pth`
   - YOLO weights (auto-downloaded by Ultralytics, or place manually): `nav/yolov8x-seg.pt`, `nav/yolo11x-seg.pt`, `nav/yolo26x-seg.pt`

Expected file structure after setup:

```
PEANUT/
├── habitat-challenge-data/
│   ├── objectgoal_hm3d/
│   │   ├── train/
│   │   ├── val/
│   │   └── val_mini/
│   └── data/
│       └── scene_datasets/
│           └── hm3d/
│               ├── train/
│               └── val/
└── nav/
    ├── pred_model_wts.pth
    ├── yolov8x-seg.pt          # optional, for YOLO backends
    ├── yolo11x-seg.pt          # optional
    ├── yolo26x-seg.pt          # optional
    └── agent/
        └── utils/
            └── mask_rcnn_R_101_cat9.pth
```

## Usage

### Quick Start

Edit `nav_exp.sh` to configure the desired script and arguments, then:

```bash
sh build_and_run.sh
```

This builds the Docker image and runs the experiment. Use `build_and_run_v2.sh` for the v2 environment (PyTorch 2.1, CUDA 11.8).

> **Note:** Depending on your Docker setup, you may need `sudo`.

### Evaluating the Navigation Agent

Run ObjectNav evaluation across episodes:

```bash
# Inside nav_exp.sh, set:
python collect.py --seg_model_type yolo --exp_name my_eval
```

- `nav/collect.py` — Primary evaluation loop with per-episode metric logging (Success, SPL) to JSONL
- `nav/collect_all_categories.py` — Evaluate across all 6 HM3D object categories
- `nav/eval.py` — Habitat Challenge submission entry point (uses `habitat.Challenge`)

See `nav/arguments.py` for the full set of command-line arguments.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seg_model_type` | `yolo` | Segmentation backend (see table above) |
| `--seg_model_wts` | `mask_rcnn_R_101_cat9.pth` | Path to segmentation model weights |
| `--pred_model_wts` | `nav/pred_model_wts.pth` | Prediction network weights |
| `--visualize` | `2` | 0: off, 1: on-screen, 2: dump to files |
| `--max_episode_length` | `500` | Maximum steps per episode |
| `--map_size_cm` | `4800` | Global map size in centimeters |
| `--yolo_conf` | `0.15` | YOLO detection confidence threshold |
| `--yolo_goal_conf` | `0.50` | YOLO goal category confidence threshold |
| `--sem_pred_prob_thr` | `0.95` | Semantic prediction probability threshold |
| `--exp_name` | `yolo_seg` | Experiment name for output logging |
| `-d` / `--dump_location` | `./data/tmp/` | Output directory for results and visualizations |

### Collecting Semantic Maps

Collect semantic maps from exploration episodes and save as `.npz` files:

```bash
python collect_maps.py --start_ep 0 --end_ep 100
```

Maps are saved to `data/saved_maps/` and can be used to train the prediction model.

### Training the Prediction Model

The prediction network is trained using a custom fork of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) included in `prediction/`:

```bash
cd prediction
pip install -e .
python train_prediction_model.py
```

- Architecture: ResNet50-v1c encoder + PSPHead decoder
- Input: 14-channel partial semantic map
- Output: 6-channel target object probability map
- Dataset: 5000 episodes (4000 train / 1000 val) of exploration in HM3D

### Fine-tuning YOLO

Fine-tune YOLO segmentation models on Habitat-rendered data with ground-truth semantic labels:

```bash
# 1. Collect training data from Habitat scenes
python finetune_yolo/collect_yolo_data.py

# 2. Fine-tune
python finetune_yolo/train_yolo.py
```

Fine-tuned weights can then be used via `--seg_model_wts <path_to_weights>`.

## Project Structure

```
PEANUT/
├── nav/                           # Core navigation agent
│   ├── agent/
│   │   ├── peanut_agent.py        # Top-level Habitat agent (reset/act)
│   │   ├── agent_state.py         # Map, pose, and prediction state
│   │   ├── agent_helper.py        # Planning, preprocessing, visualization
│   │   ├── mapping.py             # Depth-to-map projection (nn.Module)
│   │   ├── prediction.py          # Prediction model wrapper
│   │   └── utils/
│   │       ├── segmentation.py    # Mask R-CNN + YOLO backends
│   │       ├── segmentation_yolo11.py
│   │       ├── segmentation_yolo26.py
│   │       ├── segmentation_grounded_sam.py
│   │       ├── fmm_planner.py     # Fast Marching Method planner
│   │       ├── depth_utils.py     # Depth → 3D point cloud transforms
│   │       └── visualization.py   # Debug visualization
│   ├── eval.py                    # Habitat Challenge submission
│   ├── collect.py                 # Episode evaluation with metrics
│   ├── collect_all_categories.py  # Multi-category evaluation
│   ├── collect_maps.py            # Semantic map data collection
│   ├── arguments.py               # CLI argument definitions
│   ├── constants.py               # Category mappings (HM3D ↔ COCO ↔ PEANUT)
│   └── pred_model_cfg.py          # MMSeg prediction model config
├── prediction/                    # Custom MMSegmentation fork
│   ├── mmseg/                     # MMSeg library
│   ├── train_prediction_model.py  # Prediction model training
│   └── setup.py                   # pip install -e .
├── finetune_yolo/                 # YOLO fine-tuning pipeline
│   ├── collect_yolo_data.py       # Render RGB + GT from Habitat
│   └── train_yolo.py             # YOLO training launcher
├── configs/                       # Habitat environment YAML configs
├── data/                          # Datasets, outputs, analysis
│   └── saved_maps/                # Collected semantic map dataset
├── habitat-challenge-data/        # HM3D scenes + ObjectNav episodes
├── build_and_run.sh               # Docker build + run (v1)
├── build_and_run_v2.sh            # Docker build + run (v2)
├── nav_exp.sh                     # In-container experiment launcher
├── peanut.Dockerfile              # Docker image (PyTorch 1.10, CUDA 11.1)
└── peanut_v2.Dockerfile           # Docker image (PyTorch 2.1, CUDA 11.8)
```

## Docker Environments

Two Docker configurations are provided:

| Version | Dockerfile | Build Script | PyTorch | CUDA | Notes |
|---------|-----------|-------------|---------|------|-------|
| v1 | `peanut.Dockerfile` | `build_and_run.sh` | 1.10 | 11.1 | Original, Detectron2 + MMSeg |
| v2 | `peanut_v2.Dockerfile` | `build_and_run_v2.sh` | 2.1 | 11.8 | + Ultralytics YOLO + SAM |

## Semantic Map Dataset

The original map dataset used in the paper can be downloaded from [this Google Drive link](https://drive.google.com/file/d/134omZAAu_zYUaOYuNQcDMPhZCdxV0zbZ/view?usp=sharing).

It contains sequences of semantic maps from 5000 episodes (4000 train, 1000 val) of [Stubborn](https://github.com/Improbable-AI/Stubborn)-based exploration in HM3D. This dataset can be directly used to train the prediction model with `prediction/train_prediction_model.py`.

## Technology Stack

- **Simulation:** [Habitat-sim](https://github.com/facebookresearch/habitat-sim) 0.2.1 + [Habitat-lab](https://github.com/facebookresearch/habitat-lab)
- **Deep Learning:** PyTorch 2.1 / 1.10
- **Segmentation:** [Detectron2](https://github.com/facebookresearch/detectron2), [Ultralytics](https://github.com/ultralytics/ultralytics), [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **Prediction:** [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (custom fork)
- **Path Planning:** [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm) (Fast Marching Method)
- **Infrastructure:** Docker + nvidia-docker, EGL headless rendering

## Acknowledgments

This project builds upon code from [PEANUT](https://github.com/ajzhai/PEANUT) ([Zhai & Wang, ICCV 2023](https://arxiv.org/abs/2212.02497)), [Stubborn](https://github.com/Improbable-AI/Stubborn), [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank the authors of these projects for their amazing work!

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
