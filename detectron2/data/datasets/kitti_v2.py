import os
import logging
import numpy as np

import torch
import torch.utils.data.distributed

from torchvision import transforms

from detectron2.data.build import DATASET_REGISTRY, DatasetBase
from .preprocessing import read_img, read_npz_depth, read_png_depth, read_kitti_calib_file, read_bin
from .preprocessing import resize, random_crop, random_image_augment_v2, kb_crop, flip

logger = logging.getLogger(__name__)


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
        self.depth_type = dataset_cfg.DEPTH_TYPE

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

            # check file exists
            if (not os.path.isfile(self._get_img_dir(date, drive, img_id))) \
                    or (self.depth_type == 'velodyne'
                        and not os.path.isfile(self._get_npz_depth_dir(date, drive, img_id))) \
                    or (self.depth_type == 'groundtruth'
                        and not os.path.isfile(self._get_png_depth_dir(date, drive, img_id))) \
                    or (self.depth_type == 'refined'
                        and not os.path.isfile(self._get_refined_png_depth_dir(date, drive, img_id))):
                continue

            self.metadatas.append((date, drive, img_id))

        self.metadatas = sorted(self.metadatas, key=lambda x: (x[0], x[1], x[2]))

        logger.info(f'Loaded {len(self.metadatas)} samples')

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

        logger.info(f'After context filtering, {len(self.valid_inds)} samples left')

        self.calib_cache = {}

    def __len__(self):
        return len(self.valid_inds)

    def __getitem__(self, idx):
        idx = self.valid_inds[idx]

        date, drive, img_id = self.metadatas[idx]

        data = {'metadata': {'date': date, 'drive': drive, 'img_id': img_id},
                'image': read_img(self._get_img_dir(date, drive, img_id)),
                'top_margin': 0,
                'left_margin': 0}

        if date in self.calib_cache:
            cam_calib = self.calib_cache[date]['cam_calib']
            lidar_calib = self.calib_cache[date]['lidar_calib']
        else:
            cam_calib = read_kitti_calib_file(os.path.join(self.data_root, date, 'calib_cam_to_cam.txt'))
            lidar_calib = read_kitti_calib_file(os.path.join(self.data_root, date, 'calib_velo_to_cam.txt'))
            self.calib_cache[date] = {'cam_calib': cam_calib,
                                      'lidar_calib': lidar_calib}
        P2 = np.eye(4, dtype=np.float32)
        P2[:3, :] = np.array(cam_calib['P_rect_02']).reshape([3, 4])
        R0 = np.eye(4, dtype=np.float32)
        R0[:3, :3] = np.array(cam_calib['R_rect_00']).reshape([3, 3])
        data['intrinsics'] = P2[:3, :3]

        if self.depth_type == 'velodyne':
            data['depth_gt'] = read_npz_depth(self._get_npz_depth_dir(date, drive, img_id))
        elif self.depth_type == 'groundtruth':
            data['depth_gt'] = read_png_depth(self._get_png_depth_dir(date, drive, img_id))
        elif self.depth_type == 'refined':
            data['depth_gt'] = read_png_depth(self._get_refined_png_depth_dir(date, drive, img_id))

        if self.mode == 'val' and 'depth_gt' in data:
            data['depth_gt_orig'] = data['depth_gt'].copy()

        # Add context information if requested
        if self.with_context:
            # Add context images
            data['context'] = [read_img(self._get_img_dir(*self.metadatas[ctx_idx]))
                               for ctx_idx in self.context_list[idx]]

        # data['lidar'] = read_bin(self._get_lidar_dir(date, drive, img_id))

        if self.kb_crop:
            data = kb_crop(data)

        if 'depth_gt' in data:
            data['depth_gt'] = np.clip(data['depth_gt'], 0, self.max_depth)

        if self.resize:
            data = resize(data, self.input_h, self.input_w)
        elif self.mode == 'train':
            data = random_crop(data, self.input_h, self.input_w)

        data['image_orig'] = data['image'].copy()
        if 'context' in data:
            data['context_orig'] = [d.copy() for d in data['context']]

        if self.mode == 'train':
            # data = random_image_augment(data)  # Random gamma, brightness, color augmentation
            data = random_image_augment_v2(data)

        data['image'] = self.to_tensor(data['image'])
        data['image_orig'] = self.to_tensor(data['image_orig'])
        if 'context' in data:
            data['context'] = [self.to_tensor(img) for img in data['context']]
            data['context_orig'] = [self.to_tensor(img) for img in data['context_orig']]
        if 'depth_gt' in data:
            data['depth_gt'] = torch.from_numpy(data['depth_gt'])[None, :, :]

        return data

    def _get_img_dir(self, date, drive, img_id):
        return os.path.join(self.data_root, date, f'{date}_drive_{drive}_sync',
                            'image_02', 'data', f'{img_id}.png')

    def _get_npz_depth_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                            'proj_depth', 'velodyne', 'image_02', f'{img_id}.npz')

    def _get_png_depth_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                            'proj_depth', 'groundtruth', 'image_02', f'{img_id}.png')

    def _get_refined_png_depth_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, f'{date}_drive_{drive}_sync',
                            'proj_depth', 'groundtruth', 'image_02', f'{img_id}.png')

    def _get_lidar_dir(self, date, drive, img_id):
        return os.path.join(self.depth_root, f'{date}_drive_{drive}_sync',
                            'velodyne_points', 'data', f'{img_id}.bin')


@DATASET_REGISTRY.register()
class KittiDepthVal_v2(KittiDepthTrain_v2):
    def __init__(self, dataset_cfg, cfg):
        super(KittiDepthVal_v2, self).__init__(dataset_cfg, cfg)

        self.mode = 'val'
