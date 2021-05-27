from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data

from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets.kitti_v2 import KittiDepthTrain_v2
from detectron2.data.build import build_detection_train_loader

from detectron2.modeling.meta_arch.MotionLearning import MotionLearningModel

from detectron2.utils.logger import setup_logger

logger = setup_logger()
logger.info("???")

if __name__ == '__main__':

    cfg = CN()
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = CN()

    cfg.DATASETS.TRAIN.SPLIT = "E:\data\KITTI_tiny\kitti_tiny_train.txt"
    cfg.DATASETS.TRAIN.DATA_ROOT = "E:\data\KITTI_tiny"
    cfg.DATASETS.TRAIN.DEPTH_ROOT = "E:\data\KITTI_tiny"
    cfg.DATASETS.TRAIN.IMG_WIDTH = 384
    cfg.DATASETS.TRAIN.IMG_HEIGHT = 192

    # `True` if cropping is used for data augmentation during training
    cfg.DATASETS.TRAIN.KB_CROP = False
    cfg.DATASETS.TRAIN.RESIZE = False
    cfg.DATASETS.TRAIN.DEPTH_TYPE = "velodyne"
    cfg.DATASETS.TRAIN.FORWARD_CONTEXT = 0
    cfg.DATASETS.TRAIN.BACKWARD_CONTEXT = 1
    cfg.DATASETS.TRAIN.STRIDE = 1
    cfg.DATASETS.TRAIN.WITH_POSE = False
    cfg.DATASETS.TRAIN.WITH_CONTEXT_DEPTH = True

    cfg.MODEL = CN()
    cfg.MODEL.DEPTH_NET = CN()

    cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.PIXEL_STD = [0.229, 0.224, 0.225]

    cfg.MODEL.DATASET = "kitti"
    cfg.MODEL.MAX_DEPTH = 80

    cfg.MODEL.DEPTH_NET.NAME = "DepthResNet"
    cfg.MODEL.DEPTH_NET.ENCODER_NAME = "18pt"

    cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH = False
    cfg.MODEL.DEPTH_NET.LEARN_SCALE = False
    cfg.MODEL.DEPTH_NET.FLIP_PROB = 0.5

    cfg.MODEL.POSE_NET = CN()
    cfg.MODEL.POSE_NET.NAME = 'GoogleMotionNet'
    cfg.MODEL.POSE_NET.NUM_CONTEXTS = 0
    cfg.MODEL.POSE_NET.USE_DEPTH = True
    cfg.MODEL.POSE_NET.GROUP_NORM = False
    cfg.MODEL.POSE_NET.MASK_MOTION = False
    cfg.MODEL.POSE_NET.LEARN_SCALE = False

    cfg.LOSS = CN()
    cfg.LOSS.SSIM_WEIGHT = 0.5
    cfg.LOSS.C1 = 0.1
    cfg.LOSS.C2 = 0.1
    cfg.LOSS.CLIP = 0.0
    cfg.LOSS.AUTOMASK = False
    cfg.LOSS.SMOOTHNESS_WEIGHT = 0.0
    cfg.LOSS.PHOTOMETRIC_REDUCE = 'mean'
    cfg.LOSS.SUPERVISED_WEIGHT = 0.1
    cfg.LOSS.VARIANCE_FOCUS = 0.85
    cfg.LOSS.VAR_LOSS_WEIGHT = 0.1
    cfg.LOSS.MOTION_SMOOTHNESS_WEIGHT = 0.1
    cfg.LOSS.MOTION_SPARSITY_WEIGHT = 0.1
    cfg.LOSS.SCALE_NORMALIZE = True

    dataset = KittiDepthTrain_v2(cfg.DATASETS.TRAIN, cfg)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=0,
                                             batch_size=2,
                                             shuffle=False,
                                             collate_fn=dataset.batch_collator)

    model = MotionLearningModel(cfg)

    for data in tqdm(dataloader):
        output = model(data)
