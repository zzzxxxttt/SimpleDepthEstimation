import os
import logging
import pickle

import numpy as np
from collections import defaultdict

import torch

from detectron2.data.build import DATASET_REGISTRY, DatasetBase

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class WaymoDepth(DatasetBase):
    def __init__(self, dataset_cfg, cfg):
        super(WaymoDepth, self).__init__(dataset_cfg, cfg)

        self.data_root = dataset_cfg.DATA_ROOT
        self.depth_root = dataset_cfg.DEPTH_ROOT
        self.split_file = dataset_cfg.SPLIT

        self.downsample = dataset_cfg.get('DOWNSAMPLE', 1)

        self.use_cams = dataset_cfg.get('USE_CAM', 'FRONT')
        self.with_depth = dataset_cfg.get('WITH_DEPTH', False)

        self.forward_context = dataset_cfg.get('FORWARD_CONTEXT', 0)
        self.backward_context = dataset_cfg.get('BACKWARD_CONTEXT', 0)
        self.stride = dataset_cfg.get('STRIDE', 0)

        self.with_pose = dataset_cfg.get('WITH_POSE', False)
        self.with_context_depth = dataset_cfg.get('WITH_CONTEXT_DEPTH', False)

        with open(self.split_file, 'rb') as f:
            infos = pickle.load(f)

        self.metadatas = []
        self.calib_cache = {}
        for segment, seg_info in infos.items():
            metadatas = []
            for frame, frame_info in seg_info['frames'].items():
                metadatas.append((segment, frame, frame_info['cams']))
            self.metadatas.extend(sorted(metadatas, key=lambda x: x[1])[:len(metadatas) // self.downsample])
            self.calib_cache[segment] = seg_info['cams']

        if self.downsample > 1:
            logger.info(f'Downsample dataset to 1/{self.downsample}!')
        logger.info(f'Loaded {len(self.metadatas)} samples')

        # If using context, filter file list
        self.context_list = [[] for _ in range(len(self.metadatas))]
        self.with_context = self.backward_context != 0 or self.forward_context != 0

        if self.with_context:
            self.valid_inds = []
            for idx, (segment, frame, _) in enumerate(self.metadatas):
                for offset in range(-self.backward_context * self.stride,
                                    self.forward_context * self.stride + 1,
                                    self.stride):
                    new_idx = idx + offset
                    if offset != 0:
                        if 0 <= new_idx < len(self.metadatas) \
                                and self.metadatas[new_idx][0] == segment:
                            self.context_list[idx].append(new_idx)
                if len(self.context_list[idx]) == self.backward_context + self.forward_context:
                    self.valid_inds.append(idx)
        else:
            self.valid_inds = list(range(len(self.metadatas)))

        logger.info(f'After context filtering, {len(self.valid_inds)} samples left')
        if len(self.metadatas) == 0:
            logger.warning('Empty dataset!')

    def __len__(self):
        return len(self.valid_inds)

    def __getitem__(self, idx_):
        idx = self.valid_inds[idx_]

        segment, frame_time, img_time = self.metadatas[idx]

        data_allcams = []
        for cam in self.use_cams:
            data = {'metadata': {'segment': segment,
                                 'frame_time': frame_time,
                                 'cam': cam,
                                 'use_cams': self.use_cams,
                                 'img_time': img_time,
                                 'img_dir': self._get_img_dir(segment, img_time[cam], cam),
                                 'depth_dir': self._get_depth_dir(segment, img_time[cam], cam),
                                 'ctx_img_dir': [self._get_img_dir(self.metadatas[ctx_idx][0],
                                                                   self.metadatas[ctx_idx][2][cam], cam)
                                                 for ctx_idx in self.context_list[idx]],
                                 'ctx_depth_dir': [self._get_depth_dir(self.metadatas[ctx_idx][0],
                                                                       self.metadatas[ctx_idx][2][cam], cam)
                                                   for ctx_idx in self.context_list[idx]]},
                    'intrinsics': self.calib_cache[segment][cam]['intrinsics'][:3, :3].astype(np.float32)}

            # T = np.array([[0, -1, 0, 0],
            #               [0, 0, -1, 0],
            #               [1, 0, 0, 0],
            #               [0, 0, 0, 1]])
            # data['extrinsics'] = T @ self.calib_cache[segment][cam]['extrinsics']

            data_allcams.append(self.preprocess(data))

        return data_allcams

    def _get_img_dir(self, segment, img_time, cam):
        return os.path.join(self.data_root, segment, cam, f'{img_time}.jpg')

    def _get_depth_dir(self, segment, img_time, cam):
        return os.path.join(self.depth_root, segment, cam, f'{img_time}.png')

    def batch_collator(self, batch_list):
        batch_list = [d for data in batch_list for d in data] # absorb camera dim into batch

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
