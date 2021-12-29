import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets.kitti_v2 import KittiDepthV2
from detectron2.utils.logger import setup_logger

from detectron2.geometry.camera import view_synthesis

logger = setup_logger()
logger.info("???")

if __name__ == '__main__':

    cfg = CN()
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = CN()

    cfg.DATASETS.TRAIN.SPLIT = "E:\A\splits\eigen_train_files.txt"
    cfg.DATASETS.TRAIN.DATA_ROOT = "E:\A\kitti_raw"
    cfg.DATASETS.TRAIN.DEPTH_ROOT = "E:\A\kitti_depth_groundtruth"
    cfg.DATASETS.TRAIN.IMG_WIDTH = 640
    cfg.DATASETS.TRAIN.IMG_HEIGHT = 192

    # `True` if cropping is used for data augmentation during training
    cfg.DATASETS.TRAIN.KB_CROP = False
    cfg.DATASETS.TRAIN.RESIZE = True
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
        img1 = data['image']
        img2 = data['context'][0]
        depth1 = data['depth_gt']
        depth2 = data['context_depth_gt'][0]
        pose1 = data['pose_gt']
        pose2 = data['context_pose_gt'][0]
        intrinsics = data['intrinsics']

        plt.imshow(img2.permute(1, 2, 0).numpy())
        plt.show()

        plt.imshow(img1.permute(1, 2, 0).numpy())
        plt.show()

        # plt.imshow(depth1[0].numpy(), cmap='gray')
        # plt.show()
        #
        # plt.imshow(depth2[0].numpy(), cmap='gray')
        # plt.show()

        T = pose1 @ np.linalg.inv(pose2)

        sampled_values, depth_in_B, points_A_coords_in_B, proj_mask = view_synthesis(img2[None, ...],
                                                                                     depth1[None, ...],
                                                                                     intrinsics[None, ...],
                                                                                     T[None, :3, :3],
                                                                                     T[None, :3, [3], None])

        plt.imshow(sampled_values[0].permute(1, 2, 0).numpy())
        plt.show()

        pass
