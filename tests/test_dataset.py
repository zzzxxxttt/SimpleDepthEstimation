import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tqdm import tqdm

import torch
import torch.nn as nn

from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets.kitti_v2 import KittiDepthV2
from detectron2.utils.logger import setup_logger

logger = setup_logger()
logger.info("???")

if __name__ == '__main__':

    cfg = CN()
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = CN()

    cfg.DATASETS.TRAIN.SPLIT = "E:\A\splits\eigen_train_files.txt"
    cfg.DATASETS.TRAIN.DATA_ROOT = "E:\A\kitti_raw"
    cfg.DATASETS.TRAIN.DEPTH_ROOT = "E:\A\KITTI_raw_groundtruth"
    cfg.DATASETS.TRAIN.IMG_WIDTH = 384
    cfg.DATASETS.TRAIN.IMG_HEIGHT = 192

    # `True` if cropping is used for data augmentation during training
    cfg.DATASETS.TRAIN.KB_CROP = False
    cfg.DATASETS.TRAIN.RESIZE = False
    cfg.DATASETS.TRAIN.DEPTH_TYPE = "groundtruth"
    cfg.DATASETS.TRAIN.FORWARD_CONTEXT = 0
    cfg.DATASETS.TRAIN.BACKWARD_CONTEXT = 1
    cfg.DATASETS.TRAIN.STRIDE = 1
    cfg.DATASETS.TRAIN.WITH_POSE = True
    cfg.DATASETS.TRAIN.WITH_CONTEXT_DEPTH = True

    cfg.MODEL = CN()
    cfg.MODEL.MAX_DEPTH = 80

    dataset = KittiDepthV2(cfg.DATASETS.TRAIN, cfg)

    for data in tqdm(dataset):
        pass
