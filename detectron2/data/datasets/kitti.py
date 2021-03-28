import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from detectron2.data.build import DATASET_REGISTRY
from .preprocessing import resize_depth_np


def random_crop(img, depth, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0]
    assert img.shape[1] == depth.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    depth = depth[y:y + height, x:x + width] if depth is not None else None
    return img, depth


def augment_image(image):
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug


@DATASET_REGISTRY.register()
class KittiDepthTrain(Dataset):
    def __init__(self, cfg):
        self.data_root = cfg.DATASETS.DATA_ROOT
        self.depth_root = cfg.DATASETS.DEPTH_ROOT

        self.filenames = [line.strip().split() for line in open(cfg.DATASETS.SPLIT_TRAIN, 'r')]

        self.metadatas = []
        for file in self.filenames:
            date = file[0].split('/')[0]
            drive = file[0].split('/')[1].replace(f'{date}_drive_', '').replace('_sync', '')
            img_id = file[0].split('/')[-1].replace('.png', '')
            self.metadatas.append((date, drive, img_id))

        self.input_w = cfg.INPUT.IMG_WIDTH_TRAIN
        self.input_h = cfg.INPUT.IMG_HEIGHT_TRAIN

        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.use_right = False
        self.kb_crop = cfg.INPUT.KB_CROP
        self.rand_rot = cfg.INPUT.RAND_ROT
        self.to_tensor = transforms.ToTensor()

        self.resize = cfg.INPUT.RESIZE
        self.with_pose = cfg.INPUT.WITH_POSE
        self.forward_context = cfg.INPUT.FORWARD_CONTEXT
        self.backward_context = cfg.INPUT.BACKWARD_CONTEXT
        self.stride = cfg.INPUT.STRIDE

        # If using context, filter file list
        self.context_list = [[] for _ in range(len(self.metadatas))]
        if self.backward_context != 0 or self.forward_context != 0:
            self.valid_inds = []
            for idx, (date, drive, img_id) in enumerate(self.metadatas):
                for offset in range(-self.backward_context * self.stride,
                                    self.forward_context * self.stride + 1,
                                    self.stride):
                    new_idx = idx + offset
                    if 0 <= new_idx < len(self.metadatas) \
                            and offset != 0 \
                            and self.metadatas[new_idx][0] == date \
                            and self.metadatas[new_idx][1] == drive \
                            and int(self.metadatas[new_idx][2]) == int(img_id) + offset:
                        self.context_list[idx].append(new_idx)
                if len(self.context_list[idx]) == self.backward_context + self.forward_context:
                    self.valid_inds.append(idx)
        else:
            self.valid_inds = list(range(len(self.metadatas)))

    def __len__(self):
        return len(self.valid_inds)

    def __getitem__(self, idx):
        idx = self.valid_inds[idx]

        line = self.filenames[idx]

        focal = float(line[2])

        if self.use_right and random.random() > 0.5:
            image_dir = os.path.join(self.data_root, line[3])
            depth_dir = os.path.join(self.depth_root, line[4])
        else:
            image_dir = os.path.join(self.data_root, line[0])
            depth_dir = os.path.join(self.depth_root, line[1])

        image = Image.open(image_dir)
        depth = Image.open(depth_dir)

        top_margin, left_margin = 0, 0
        if self.kb_crop:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        if self.rand_rot > 0:
            random_angle = (random.random() - 0.5) * 2 * self.rand_rot
            image = image.rotate(random_angle, resample=Image.BILINEAR)
            depth = depth.rotate(random_angle, resample=Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth = np.asarray(depth, dtype=np.float32) / 256.0
        depth = np.clip(depth, 0, self.max_depth)

        if self.resize:
            image = cv2.resize(image, (self.input_w, self.input_h))
            depth = resize_depth_np(depth, (self.input_h, self.input_w))
        else:
            image, depth = random_crop(image, depth, self.input_h, self.input_w)

        # Random flipping
        if random.random() > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth = (depth[:, ::-1]).copy()

        # Random gamma, brightness, color augmentation
        if random.random() > 0.5:
            image = augment_image(image)

        return {'image': self.to_tensor(image),
                'focal': focal,
                'depth_gt': torch.from_numpy(depth)[None, :, :],
                'top_margin': top_margin,
                'left_margin': left_margin}

    def batch_collator(self, batch):
        return default_collate(batch)


@DATASET_REGISTRY.register()
class KittiDepthVal(KittiDepthTrain):
    def __init__(self, cfg):
        super(KittiDepthVal, self).__init__(cfg)
        self.filenames = [line.strip().split() for line in open(cfg.DATASETS.SPLIT_TEST, 'r')]

        self.input_w = cfg.INPUT.IMG_WIDTH_TEST
        self.input_h = cfg.INPUT.IMG_HEIGHT_TEST

    def __getitem__(self, idx):
        line = self.filenames[idx]

        focal = float(line[2])

        image_dir = os.path.join(self.data_root, line[0])
        depth_dir = os.path.join(self.depth_root, line[1])

        image = Image.open(image_dir)
        depth = Image.open(depth_dir) if os.path.exists(depth_dir) else None

        top_margin, left_margin = 0, 0
        if self.kb_crop:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352)) if depth else None

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth = np.asarray(depth, dtype=np.float32) / 256.0 if depth is not None else np.zeros(image.shape[:2])

        image = self.to_tensor(image)

        return {'image': image,
                'focal': focal,
                'depth_gt': torch.from_numpy(depth)[None, :, :],
                'top_margin': top_margin,
                'left_margin': left_margin}
