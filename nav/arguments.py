import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='PEANUT')

    # General arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--sem_gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--start_ep', type=int, default=0,
                        help='start episode for data collection')
    parser.add_argument('--end_ep', type=int, default=-1,
                        help='end episode for data collection')
    parser.add_argument('--target_category', type=str, default='sofa',
                        help='target category for single-category evaluation')

    parser.add_argument('-v', '--visualize', type=int, default=2,
                        help="""1: Show visualization on screen
                                2: Dump visualizations as image files
                                (default: 0)""")
    parser.add_argument('--exp_name', type=str, default="yolo_seg",
                        help='experiment name (default: exp1)')
    parser.add_argument('-d', '--dump_location', type=str, default="./data/tmp/",
                        help='path to dump models and log (default: ./data/tmp/)')
    
    # Segmentation model
    parser.add_argument('--seg_model_type', type=str, default='yolo',
                        choices=['maskrcnn', 'maskrcnn_pretrained', 'yolo', 'yolo11', 'yolo26', 'cascade', 'r101coco', 'grounded_sam'],
                        help='segmentation model type: maskrcnn, maskrcnn_pretrained, yolo, yolo11, yolo26, cascade, r101coco, or grounded_sam')
    _nav_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--seg_model_wts', type=str,
                        default=os.path.join(_nav_dir, 'agent', 'utils', 'mask_rcnn_R_101_cat9.pth'),
                        help='path to segmentation model weights '
                             '(for yolo: e.g. yolov8x-seg.pt or path to custom weights)')
    parser.add_argument('--yolo_conf', type=float, default=0.15,
                        help='YOLO confidence threshold for detections '
                             '(lower than sem_pred_prob_thr because YOLO scores '
                             'are much lower on synthetic Habitat images, default: 0.15)')
    parser.add_argument('--yolo_goal_conf', type=float, default=0.5,
                        help='YOLO confidence threshold for goal category '
                             '(default: 0.25)')
    parser.add_argument('--grounded_box_thresh', type=float, default=0.15,
                        help='GroundingDINO box threshold for grounded_sam')
    parser.add_argument('--grounded_text_thresh', type=float, default=0.15,
                        help='GroundingDINO text threshold for grounded_sam')
    parser.add_argument('--grounded_conf', type=float, default=0.15,
                        help='Minimum detection confidence accepted by grounded_sam')
    parser.add_argument('--grounded_goal_conf', type=float, default=0.15,
                        help='Minimum detection confidence for goal category in grounded_sam')
    parser.add_argument('--grounded_aux_conf', type=float, default=0.45,
                        help='Minimum confidence for auxiliary categories (fireplace/bathtub/mirror) in grounded_sam')
    parser.add_argument('--grounded_chair_sofa_conf', type=float, default=0.20,
                        help='Minimum confidence for focused chair/sofa disambiguation prompts in grounded_sam')
    parser.add_argument('--grounded_open_vocab_prompt', type=str,
                        default='chair . sofa . couch . plant . potted plant . bed . toilet . tv monitor . television . fireplace . dining table . bathtub . oven . mirror . sink .',
                        help='Open-vocabulary prompt string for grounded_sam')
    parser.add_argument('--grounded_config_path', type=str,
                        default='/nav/agent/utils/grounded_sam_weights/GroundingDINO_SwinT_OGC.py',
                        help='GroundingDINO config path for grounded_sam')
    parser.add_argument('--grounded_checkpoint', type=str,
                        default='/nav/agent/utils/grounded_sam_weights/groundingdino_swint_ogc.pth',
                        help='GroundingDINO checkpoint path for grounded_sam')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type for grounded_sam')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='/nav/agent/utils/grounded_sam_weights/sam_vit_b_01ec64.pth',
                        help='SAM checkpoint path for grounded_sam')
    
    # YOLOv11-seg arguments
    parser.add_argument('--yolo11_model_path', type=str, default='yolo11x-seg.pt',
                        help='YOLOv11-seg model path (e.g. yolo11n-seg.pt, yolo11x-seg.pt, or custom weights)')
    parser.add_argument('--yolo11_conf', type=float, default=0.15,
                        help='YOLOv11 confidence threshold for detections (default: 0.15)')
    parser.add_argument('--yolo11_goal_conf', type=float, default=0.25,
                        help='YOLOv11 confidence threshold for goal category (default: 0.25)')

    # YOLO26-seg arguments
    parser.add_argument('--yolo26_model_path', type=str, default='yolo26x-seg.pt',
                        help='YOLO26-seg model path (e.g. yolo26n-seg.pt, yolo26x-seg.pt, or custom weights)')
    parser.add_argument('--yolo26_conf', type=float, default=0.08,
                        help='YOLO26 confidence threshold for detections (default: 0.08)')
    parser.add_argument('--yolo26_goal_conf', type=float, default=0.15,
                        help='YOLO26 confidence threshold for goal category (default: 0.15)')

    # Prediction model
    parser.add_argument('--pred_model_wts', type=str, default="./nav/pred_model_wts.pth",
                        help='path to prediction model weights')
    parser.add_argument('--pred_model_cfg', type=str, default="./nav/pred_model_cfg.py",
                        help='path to prediction model config')
    parser.add_argument('--prediction_window', type=int, default=720,
                        help='size of prediction (in pixels)')
    
    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/objectnav_gibson.yaml",
                        help="path to config yaml containing task information")

    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")


    parser.add_argument('--num_local_steps', type=int, default=20,
                        help="""Number of steps between local map position updates""")

    # Mapping
    parser.add_argument('--num_sem_categories', type=int, default=10)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.95)
    parser.add_argument('--goal_thr', type=float, default=0.985)
    parser.add_argument('--global_downscaling', type=int, default=2)  # local map relative size
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=4800)  # the global downscaling may also need to change
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=0.1)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    
    parser.add_argument('--col_rad', type = float, default = 4) 
    parser.add_argument('--goal_erode', type = int, default = 3) 
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument(
        "--evaluation", type=str, required=False, choices=["local", "remote"]
    )
    
    # Small details from Stubborn
    parser.add_argument('--timestep_limit', type = int, default=499)
    parser.add_argument('--grid_resolution',type = int, default = 24)
    parser.add_argument('--magnify_goal_when_hard',type = int, default = 100) #originally 100
    parser.add_argument("--move_forward_after_stop",type = int, default = 1) #originally 1

    # Long-term goal selection
    parser.add_argument('--dist_weight_temperature', type = float, default = 500,
                        help="Temperature for exponential distance weight (lambda in paper)")
    parser.add_argument('--goal_reached_dist', type = float, default = 75,
                        help="Distance at which goal is considered reached")
    parser.add_argument('--update_goal_freq', type = float, default = 10,
                        help="How often to update long-term goal")
    parser.add_argument('--switch_step', type = float, default = 0,
                        help="For switching from Stubborn goal selection to PEANUT")
    
    # For data collection 
    parser.add_argument('--use_gt_seg', type = int, default = 0)
    parser.add_argument('--only_explore', type = int, default = 1)

    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
