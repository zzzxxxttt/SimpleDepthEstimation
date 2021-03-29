import cv2
import random
import numpy as np


def resize_depth_np(depth, dst_size):
    if depth.shape[-2] == dst_size[-2] and depth.shape[-1] == dst_size[-1]:
        return depth
    else:
        H, W = depth.shape
        y, x = np.nonzero(depth)
        resized_depth = np.zeros(dst_size, dtype=np.float32)
        resized_depth[(dst_size[0] * y / H).astype(np.int),
                      (dst_size[1] * x / W).astype(np.int)] = depth[y, x]
        return resized_depth


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
    if random.random() > 0.5:
        data['image'] = data['image'][:, ::-1, :].copy()

        if 'context' in data:
            data['context'] = [img[:, ::-1, :].copy() for img in data['context']]

        if 'depth_gt' in data:
            data['depth_gt'] = data['depth_gt'][:, ::-1].copy()

        data['flip'] = True

    else:
        data['flip'] = False

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


def random_image_augment(data, p=0.5):
    if random.random() < p:
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
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
