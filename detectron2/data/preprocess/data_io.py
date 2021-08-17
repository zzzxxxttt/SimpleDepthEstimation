import cv2
import numpy as np
from collections import namedtuple

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


def read_img(filepath):
    img = cv2.imread(filepath)
    assert img is not None, f'{filepath}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_npz_depth(file):
    depth = np.load(file)['velodyne_depth']
    return depth.astype(np.float32)


def read_png_depth(filepath):
    img = cv2.imread(filepath, -1)
    return img.astype(np.float32) / 255


def read_bin(filepath, dims=4, out_dims=3):
    scan = np.fromfile(filepath, dtype=np.float32).reshape(-1, dims).T
    return scan[:out_dims]


def read_kitti_calib_file(filepath):
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


def write_depth(depth, save_path):
    pred_depth_scaled = (depth * 255).astype(np.uint16)
    cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return
