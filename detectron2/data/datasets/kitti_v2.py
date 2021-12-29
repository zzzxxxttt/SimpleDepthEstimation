import os
import logging
import numpy as np
from collections import defaultdict

import torch

from detectron2.data.build import DATASET_REGISTRY, DatasetBase
from detectron2.geometry.pose_utils import pose_from_oxts_packet_np, T_from_R_t_np

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class KittiDepthV2(DatasetBase):
    def __init__(self, dataset_cfg, cfg):
        super(KittiDepthV2, self).__init__(dataset_cfg, cfg)

        self.data_root = dataset_cfg.DATA_ROOT
        self.depth_root = dataset_cfg.DEPTH_ROOT
        self.split_file = dataset_cfg.SPLIT

        self.depth_type = dataset_cfg.DEPTH_TYPE
        self.with_depth = self.depth_type != 'none'

        self.forward_context = dataset_cfg.get('FORWARD_CONTEXT', 0)
        self.backward_context = dataset_cfg.get('BACKWARD_CONTEXT', 0)
        self.stride = dataset_cfg.get('STRIDE', 0)

        self.with_pose = dataset_cfg.get('WITH_POSE', False)
        self.with_context_depth = dataset_cfg.get('WITH_CONTEXT_DEPTH', False)

        self.metadatas = []
        count = 0
        for line in open(self.split_file, 'r'):
            line = line.strip().split()
            date = line[0].split('/')[0]
            drive = line[0].split('/')[1].replace(f'{date}_drive_', '').replace('_sync', '')
            img_id = line[0].split('/')[-1].replace('.png', '')
            count += 1

            # check file exists
            if (not os.path.isfile(self._get_img_dir(date, drive, img_id))) \
                    or (self.depth_type != 'none' and (
                    not os.path.isfile(self._get_depth_dir(date, drive, img_id)))):
                continue

            self.metadatas.append((date, drive, img_id))

        self.metadatas = sorted(self.metadatas, key=lambda x: (x[0], x[1], x[2]))

        logger.info(f'Loaded {count} samples')
        logger.info(f'After existence filtering, {len(self.metadatas)} samples left')

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
        if len(self.metadatas) == 0:
            logger.warning('Empty dataset!')

        self.calib_cache = {}

    def __len__(self):
        return len(self.valid_inds)

    def __getitem__(self, idx_):
        idx = self.valid_inds[idx_]

        date, drive, img_id = self.metadatas[idx]

        data = {'metadata': {'date': date,
                             'drive': drive,
                             'img_id': img_id,
                             'img_dir': self._get_img_dir(date, drive, img_id),
                             'depth_dir': self._get_depth_dir(date, drive, img_id),
                             'lidar_dir': self._get_lidar_dir(date, drive, img_id),
                             'ctx_img_dir': [self._get_img_dir(*self.metadatas[ctx_idx])
                                             for ctx_idx in self.context_list[idx]],
                             'ctx_depth_dir': [self._get_depth_dir(*self.metadatas[ctx_idx])
                                               for ctx_idx in self.context_list[idx]],
                             'ctx_lidar_dir': [self._get_lidar_dir(*self.metadatas[ctx_idx])
                                               for ctx_idx in self.context_list[idx]]}}

        if date in self.calib_cache:
            cam_calib = self.calib_cache[date]['cam_calib']
            lidar_calib = self.calib_cache[date]['lidar_calib']
            imu_calib = self.calib_cache[date]['imu_calib']
        else:
            cam_calib = self._read_calib(os.path.join(self.data_root, date, 'calib_cam_to_cam.txt'))
            lidar_calib = self._read_calib(os.path.join(self.data_root, date, 'calib_velo_to_cam.txt'))
            imu_calib = self._read_calib(os.path.join(self.data_root, date, 'calib_imu_to_velo.txt'))
            self.calib_cache[date] = {'cam_calib': cam_calib,
                                      'lidar_calib': lidar_calib,
                                      'imu_calib': imu_calib}

        P2 = np.eye(4, dtype=np.float32)
        P2[:3, :] = np.array(cam_calib['P_rect_02']).reshape([3, 4])
        R0 = np.eye(4, dtype=np.float32)
        R0[:3, :3] = np.array(cam_calib['R_rect_00']).reshape([3, 3])
        data['intrinsics'] = P2[:3, :3]

        if self.with_pose:
            velo2cam = T_from_R_t_np(lidar_calib['R'], lidar_calib['T'])
            imu2velo = T_from_R_t_np(imu_calib['R'], imu_calib['T'])
            imu2cam = R0 @ velo2cam @ imu2velo
            data['pose_gt'] = self._get_pose(date, drive, img_id, imu2cam)

        data = self.preprocess(data)

        return data

    def _get_img_dir(self, date, drive, img_id):
        return os.path.join(self.data_root, date, f'{date}_drive_{drive}_sync',
                            'image_02', 'data', f'{img_id}.png')

    def _get_depth_dir(self, date, drive, img_id):
        if self.depth_type == 'none':
            return ''
        elif self.depth_type == 'velodyne':
            return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                                'proj_depth', 'velodyne', 'image_02', f'{img_id}.npz')
        elif self.depth_type == 'groundtruth':
            return os.path.join(self.depth_root, date, f'{date}_drive_{drive}_sync',
                                'proj_depth', 'groundtruth', 'image_02', f'{img_id}.png')
        elif self.depth_type == 'refined':
            return os.path.join(self.depth_root, f'{date}_drive_{drive}_sync',
                                'proj_depth', 'groundtruth', 'image_02', f'{img_id}.png')
        else:
            raise NotImplementedError

    def _get_lidar_dir(self, date, drive, img_id):
        return os.path.join(self.data_root, date, f'{date}_drive_{drive}_sync',
                            'velodyne_points', 'data', f'{img_id}.bin')

    def _get_oxts_dir(self, date, drive, img_id):
        return os.path.join(self.data_root, date, f'{date}_drive_{drive}_sync',
                            'oxts', 'data', f'{img_id}.txt')

    def _read_calib(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        return data

    def _get_pose(self, date, drive, img_id, imu2cam):
        """Gets the pose information from an image file."""
        # Get origin data
        origin_oxts_data = np.loadtxt(self._get_oxts_dir(date, drive, '0000000000'), delimiter=' ', skiprows=0)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet_np(origin_oxts_data, scale)
        origin_pose = T_from_R_t_np(origin_R, origin_t)
        # Compute current pose
        oxts_data = np.loadtxt(self._get_oxts_dir(date, drive, img_id), delimiter=' ', skiprows=0)
        R, t = pose_from_oxts_packet_np(oxts_data, scale)
        pose = T_from_R_t_np(R, t)
        # Compute odometry pose
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        return odo_pose

    def batch_collator(self, batch_list):
        # convert list of dict into dict of list
        example_merged = defaultdict(list)
        for example in batch_list:
            for k, v in example.items():
                example_merged[k].append(v)

        ret = {}
        for key, value in example_merged.items():
            if key in ['img', 'img_orig']:
                ret[key] = torch.stack(value, 0)
            elif key in ['intrinsics', 'pose_gt']:
                ret[key] = torch.from_numpy(np.stack(value, 0))
            elif key in ['depth']:
                ret[key] = torch.from_numpy(np.stack(value, 0)[:, None, ...])
            elif key in ['ctx_img', 'ctx_img_orig']:
                value = np.stack([np.stack(v, 0) for v in value])
                ret[key] = [value[:, i] for i in range(value.shape[1])]
            elif key in ['ctx_depth']:
                value = np.stack([np.stack(v, 0)[:, None, ...] for v in value])
                ret[key] = [value[:, i] for i in range(value.shape[1])]
            elif key == 'flip':
                ret[key] = value[0]
            else:
                ret[key] = value
        return ret
