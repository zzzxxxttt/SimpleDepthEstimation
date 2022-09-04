import os.path

import cv2
import numpy as np
from collections import namedtuple

from .build import PREPROCESS_REGISTRY, Preprocess

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


@PREPROCESS_REGISTRY.register()
class LoadImg(Preprocess):
    def __init__(self, cfg):
        super(LoadImg, self).__init__(cfg)
        self.load_ctx = cfg.get('WITH_CTX', False)

    def _load(self, img_dir):
        img = cv2.imread(img_dir)
        assert img is not None, f"'{img_dir} does not exist!'"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def forward(self, data_dict):

        data_dict['img'] = self._load(data_dict['metadata']['img_dir'])

        if self.load_ctx:
            data_dict['ctx_img'] = []
            for img_dir in data_dict['metadata']['ctx_img_dir']:
                data_dict['ctx_img'].append(self._load(img_dir))

        return data_dict


@PREPROCESS_REGISTRY.register()
class LoadDepth(Preprocess):
    def __init__(self, cfg):
        super(LoadDepth, self).__init__(cfg)
        self.load_ctx = cfg.get('WITH_CTX', False)
        self.keep_orig_for_eval = cfg.get('KEEP_ORIG', False)

    def _load(self, depth_dir):
        if os.path.splitext(depth_dir)[-1] == '.npz':
            depth = np.load(depth_dir)['velodyne_depth'].astype(np.float32)
        elif os.path.splitext(depth_dir)[-1] == '.png':
            depth = cv2.imread(depth_dir, -1).astype(np.float32) / 255
        else:
            raise NotImplementedError
        assert depth is not None, f"'{depth_dir} does not exist!'"
        return depth

    def forward(self, data_dict):
        data_dict['depth'] = self._load(data_dict['metadata']['depth_dir'])
        if self.keep_orig_for_eval:
            data_dict['depth_orig'] = data_dict['depth'].copy()

        if self.load_ctx:
            data_dict['ctx_depth'] = []
            for depth_dir in data_dict['metadata']['ctx_depth_dir']:
                data_dict['ctx_depth'].append(self._load(depth_dir))

        return data_dict


@PREPROCESS_REGISTRY.register()
class LoadMask(Preprocess):
    def __init__(self, cfg):
        super(LoadMask, self).__init__(cfg)

    def _load(self, mask_dir):
        mask = cv2.imread(mask_dir, -1).astype(np.float32)
        assert mask is not None, f"'{mask_dir} does not exist!'"
        return mask

    def forward(self, data_dict):

        data_dict['mask'] = self._load(data_dict['metadata']['mask_dir'])

        data_dict['ctx_mask'] = []
        for mask_dir in data_dict['metadata']['ctx_mask_dir']:
            data_dict['ctx_mask'].append(self._load(mask_dir))

        return data_dict


@PREPROCESS_REGISTRY.register()
class LoadLidar(Preprocess):
    def __init__(self, cfg):
        super(LoadLidar, self).__init__(cfg)
        self.load_ctx = cfg.get('WITH_CTX', False)
        self.load_dim = cfg.get('LOAD_DIM', 4)
        self.use_dim = cfg.get('USE_DIM', 3)

    def _load(self, lidar_dir):
        if os.path.splitext(lidar_dir)[-1] == '.bin':
            scan = np.fromfile(lidar_dir, dtype=np.float32).reshape(-1, self.load_dim)
        else:
            raise NotImplementedError
        scan = scan[:, :self.use_dim] if isinstance(self.use_dim, int) else scan[:, self.use_dim]
        return scan

    def forward(self, data_dict):
        data_dict['lidar'] = self._load(data_dict['metadata']['lidar_dir'])

        if self.load_ctx:
            data_dict['ctx_lidar'] = []
            for lidar_dir in data_dict['metadata']['ctx_lidar_dir']:
                data_dict['ctx_lidar'].append(self._load(lidar_dir))

        return data_dict
