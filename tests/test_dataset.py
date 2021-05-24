from tqdm import tqdm

import torch
import torch.nn as nn

from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets.kitti_v2 import KittiDepthTrain_v2
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
    cfg.MODEL.MAX_DEPTH = 80

    dataset = KittiDepthTrain_v2(cfg.DATASETS.TRAIN, cfg)

    for data in tqdm(dataset):
        pass
