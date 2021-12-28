import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # todo


from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data

from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets.kitti_v2 import KittiDepthV2
from detectron2.data.build import build_detection_train_loader

from detectron2.modeling.meta_arch.MotionLearning import MotionLearningModel

from detectron2.utils.logger import setup_logger

logger = setup_logger()
logger.info("???")

if __name__ == '__main__':

    cfg = CN()
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = CN()

    cfg.DATASETS.TRAIN.SPLIT = "E:/splits/eigen_train_files_d3.txt"
    cfg.DATASETS.TRAIN.DATA_ROOT = "E:/kitti_d3"
    cfg.DATASETS.TRAIN.DEPTH_ROOT = "E:/KITTI_raw_groundtruth_d3"
    cfg.DATASETS.TRAIN.IMG_WIDTH = 320
    cfg.DATASETS.TRAIN.IMG_HEIGHT = 96

    # `True` if cropping is used for data augmentation during training
    cfg.DATASETS.TRAIN.KB_CROP = False
    cfg.DATASETS.TRAIN.RESIZE = True
    cfg.DATASETS.TRAIN.DEPTH_TYPE = "none"
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
    cfg.MODEL.DEPTH_NET.FLIP_PROB = 0.0

    cfg.MODEL.POSE_NET = CN()
    cfg.MODEL.POSE_NET.NAME = 'GooglePoseNet'
    cfg.MODEL.POSE_NET.NUM_CONTEXTS = 0
    cfg.MODEL.POSE_NET.USE_DEPTH = False
    cfg.MODEL.POSE_NET.GROUP_NORM = True
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
    cfg.LOSS.ROT_CYCLE_WEIGHT = 0.0
    cfg.LOSS.TRANS_CYCLE_WEIGHT = 0.0

    dataset = KittiDepthV2(cfg.DATASETS.TRAIN, cfg)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             num_workers=0,
                                             batch_size=2,
                                             shuffle=True,
                                             collate_fn=dataset.batch_collator)

    model = MotionLearningModel(cfg)
    model.load_state_dict(torch.load(
        '../output/debug2/model_0000009.pth', map_location='cpu')['model'])
    model.eval()

    for data in tqdm(dataloader):
        output = model(data)

        plt.imshow(data['image_orig'][0].permute(1, 2, 0).cpu().numpy())
        plt.show()

        depth = output['depth_pred'][0, 0]
        depth = np.clip(depth / 80, 0, 1.0)
        plt.imshow(depth, cmap='plasma_r')
        plt.show()
        pass

