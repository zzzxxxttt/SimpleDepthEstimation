import os
import cv2
import random
import numpy as np

import torch
import torch.utils.data.distributed

from torchvision import transforms

from detectron2.data.build import DATASET_REGISTRY, DatasetBase
from ...geometry.camera import resize_depth_np


def kb_crop(data):
    top_margin = int(data['image'].shape[0] - 352)
    left_margin = int((data['image'].shape[1] - 1216) / 2)
    data['image'] = data['image'][top_margin:top_margin + 352, left_margin:left_margin + 1216]

    data['intrinsics'][0, 2] -= left_margin
    data['intrinsics'][1, 2] -= top_margin

    if 'depth_gt' in data:
        data['depth_gt'] = data['depth_gt'][top_margin:top_margin + 352, left_margin:left_margin + 1216]

    if 'context' in data:
        data['context'] = [img[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                           for img in data['context']]

    data['top_margin'] = top_margin
    data['left_margin'] = left_margin
    return data


def resize(data, img_h, img_w):
    H, W, _ = data['image'].shape
    data['image'] = cv2.resize(data['image'], (img_w, img_h))

    data['intrinsics'][0, 0] *= img_w / W
    data['intrinsics'][0, 2] *= img_w / W
    data['intrinsics'][1, 1] *= img_h / H
    data['intrinsics'][1, 2] *= img_h / H

    if 'context' in data:
        data['context'] = [cv2.resize(img, (img_w, img_h)) for img in data['context']]

    if 'depth_gt' in data:
        data['depth_gt'] = resize_depth_np(data['depth_gt'], (img_h, img_w))

    return data


def random_crop(data, height, width):
    assert data['image'].shape[0] >= height
    assert data['image'].shape[1] >= width
    x = random.randint(0, data['image'].shape[1] - width)
    y = random.randint(0, data['image'].shape[0] - height)
    data['image'] = data['image'][y:y + height, x:x + width, :]

    data['intrinsics'][0, 2] -= x
    data['intrinsics'][1, 2] -= y

    if 'context' in data:
        data['context'] = [img[y:y + height, x:x + width, :] for img in data['context']]

    if 'depth_gt' in data:
        data['depth_gt'] = data['depth_gt'][y:y + height, x:x + width]

    return data


def flip(data):
    data['image'] = data['image'][:, ::-1, :].copy()

    if 'context' in data:
        data['context'] = [img[:, ::-1, :].copy() for img in data['context']]

    if 'depth_gt' in data:
        data['depth_gt'] = data['depth_gt'][:, ::-1].copy()

    data['flip'] = True

    return data


def image_augment(image, gamma, brightness, colors):
    # gamma augmentation
    image = image ** gamma

    # brightness augmentation
    image = image * brightness

    # color augmentation
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image *= color_image
    image = np.clip(image, 0, 1)

    return image


def random_image_augment(data):
    gamma = random.uniform(0.9, 1.1)
    brightness = random.uniform(0.9, 1.1)
    colors = np.random.uniform(0.9, 1.1, size=3)

    data['image'] = image_augment(data['image'], gamma, brightness, colors)

    if 'context' in data:
        data['context'] = [image_augment(img, gamma, brightness, colors) for img in data['context']]

    return data


def read_img(filepath):
    img = cv2.imread(filepath)
    assert img is not None, f'{filepath}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255


def read_npz_depth(file):
    depth = np.load(file)['velodyne_depth']
    return depth.astype(np.float32) / 256


def read_calib_file(filepath):
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


@DATASET_REGISTRY.register()
class KittiDepthTrain_v2(DatasetBase):
    def __init__(self, dataset_cfg, cfg):
        self.mode = 'train'

        self.data_root = dataset_cfg.DATA_ROOT
        self.depth_root = dataset_cfg.DEPTH_ROOT
        self.split_file = dataset_cfg.SPLIT

        self.input_w = dataset_cfg.IMG_WIDTH
        self.input_h = dataset_cfg.IMG_HEIGHT

        self.resize = dataset_cfg.RESIZE
        self.kb_crop = dataset_cfg.KB_CROP
        self.with_pose = dataset_cfg.WITH_POSE
        self.with_depth = dataset_cfg.WITH_DEPTH

        self.forward_context = dataset_cfg.FORWARD_CONTEXT
        self.backward_context = dataset_cfg.BACKWARD_CONTEXT
        self.stride = dataset_cfg.STRIDE

        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.to_tensor = transforms.ToTensor()

        self.metadatas = []
        for line in open(self.split_file, 'r'):
            line = line.strip().split()
            date = line[0].split('/')[0]
            drive = line[0].split('/')[1].replace(f'{date}_drive_', '').replace('_sync', '')
            img_id = line[0].split('/')[-1].replace('.png', '')
            self.metadatas.append((date, drive, img_id))

        self.metadatas = sorted(self.metadatas, key=lambda x: (x[0], x[1], x[2]))

        # If using context, filter file list
        self.context_list = [[] for _ in range(len(self.metadatas))]
        self.with_context = self.backward_context != 0 or self.forward_context != 0
        if self.with_context:
            self.valid_inds = []
            for idx, (date, drive, img_id) in enumerate(self.metadatas):
                for offset in range(-self.backward_context * self.stride,
                                    self.forward_context * self.stride + 1,
                                    self.stride):
                    new_idx = idx + offset
                    if offset != 0:
                        if 0 <= new_idx < len(self.metadatas) \
                                and self.metadatas[new_idx][0] == date \
                                and self.metadatas[new_idx][1] == drive \
                                and int(self.metadatas[new_idx][2]) == int(img_id) + offset:
                            self.context_list[idx].append(new_idx)
                if len(self.context_list[idx]) == self.backward_context + self.forward_context:
                    self.valid_inds.append(idx)
        else:
            self.valid_inds = list(range(len(self.metadatas)))

        self.calib_cache = {}

    def __len__(self):
        return len(self.valid_inds)

    def _get_img_dir(self, date, drive, img_id):
        return os.path.join(self.data_root, date, f'{date}_drive_{drive}_sync',
                            'image_02', 'data', f'{img_id}.png')

    def _get_npz_depth_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                            'proj_depth', 'velodyne', 'image_02', f'{img_id}.npz')

    def _get_png_depth_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                            'proj_depth', 'groundtruth', 'image_02', f'{img_id}.png')

    def __getitem__(self, idx):
        idx = self.valid_inds[idx]

        date, drive, img_id = self.metadatas[idx]

        data = {'metadata': {'date': date, 'drive': drive, 'img_id': img_id},
                'image': read_img(self._get_img_dir(date, drive, img_id)),
                'flip': False,
                'top_margin': 0,
                'left_margin': 0}

        if date in self.calib_cache:
            intrinsics = self.calib_cache[date]
        else:
            intrinsics = read_calib_file(os.path.join(self.data_root, date, 'calib_cam_to_cam.txt'))
            self.calib_cache[date] = intrinsics
        data['intrinsics'] = np.array(intrinsics['P_rect_02']).reshape([3, 4])[:3, :3]

        if self.with_depth:
            data['depth_gt'] = read_npz_depth(self._get_npz_depth_dir(date, drive, img_id))
            if self.mode == 'val':
                data['depth_gt_orig'] = data['depth_gt'].copy()

        # Add context information if requested
        if self.with_context:
            # Add context images
            data['context'] = [read_img(self._get_img_dir(*self.metadatas[ctx_idx]))
                               for ctx_idx in self.context_list[idx]]

        if self.kb_crop:
            data = kb_crop(data)

        if 'depth_gt' in data:
            data['depth_gt'] = np.clip(data['depth_gt'], 0, self.max_depth)

        if self.resize:
            data = resize(data, self.input_h, self.input_w)
        elif self.mode == 'train':
            data = random_crop(data, self.input_h, self.input_w)

        # Random flipping
        if self.mode == 'train' and random.random() > 0.5:
            data = flip(data)

        # Random gamma, brightness, color augmentation
        if self.mode == 'train' and random.random() > 0.5:
            data = random_image_augment(data)

        data['image'] = self.to_tensor(data['image'])
        if 'context' in data:
            data['context'] = [self.to_tensor(img) for img in data['context']]
        if 'depth_gt' in data:
            data['depth_gt'] = torch.from_numpy(data['depth_gt'])[None, :, :]
        return data


@DATASET_REGISTRY.register()
class KittiDepthVal_v2(KittiDepthTrain_v2):
    def __init__(self, dataset_cfg, cfg):
        super(KittiDepthVal_v2, self).__init__(dataset_cfg, cfg)

        self.mode = 'val'
