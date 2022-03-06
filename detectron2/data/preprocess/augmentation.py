import cv2
import random
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image

from .build import PREPROCESS_REGISTRY, Preprocess


def resize_depth(depth, dst_size):
    if depth.shape[-2] == dst_size[-2] and depth.shape[-1] == dst_size[-1]:
        return depth
    else:
        H, W = depth.shape
        y, x = np.nonzero(depth)
        resized_depth = np.zeros(dst_size, dtype=np.float32)
        resized_depth[(dst_size[0] * y / H).astype(np.int),
                      (dst_size[1] * x / W).astype(np.int)] = depth[y, x]
        return resized_depth


@PREPROCESS_REGISTRY.register()
class KBCrop(Preprocess):
    def __init__(self, cfg):
        super(KBCrop, self).__init__(cfg)

    def forward(self, data_dict):
        img_h, img_w = data_dict['img'].shape[:2]

        x_start = int((img_w - 1216) / 2)
        y_start = int(img_h - 352)

        x_end = x_start + 1216
        y_end = y_start + 352

        data_dict['img'] = data_dict['img'][y_start:y_end, x_start:x_end]

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 2] -= x_start
            data_dict['intrinsics'][1, 2] -= y_start

        if 'depth' in data_dict:
            data_dict['depth'] = data_dict['depth'][y_start:y_end, x_start:x_end]

        if 'ctx_img' in data_dict:
            data_dict['ctx_img'] = [img[y_start:y_end, x_start:x_end] for img in data_dict['ctx_img']]

        if 'ctx_depth' in data_dict:
            data_dict['ctx_depth'] = [depth[y_start:y_end, x_start:x_end] for depth in data_dict['ctx_depth']]

        data_dict['metadata']['kb_y_start'] = y_start
        data_dict['metadata']['kb_x_start'] = x_start
        data_dict['metadata']['h_before_kb_crop'] = img_h
        data_dict['metadata']['w_before_kb_crop'] = img_w
        return data_dict

    def backward(self, data_dict):
        depth_pred = data_dict['depth_pred']
        x_start, y_start = data_dict['metadata']['kb_x_start'], data_dict['metadata']['kb_y_start']
        img_h, img_w = data_dict['metadata']['h_before_kb_crop'], data_dict['metadata']['w_before_kb_crop']
        uncropped = np.zeros((img_h, img_w), dtype=np.float32)
        uncropped[y_start:y_start + depth_pred.shape[-2], x_start:x_start + depth_pred.shape[-1]] = depth_pred
        data_dict['depth_pred'] = uncropped
        return data_dict


@PREPROCESS_REGISTRY.register()
class CropTopTo(Preprocess):
    def __init__(self, cfg):
        super(CropTopTo, self).__init__(cfg)
        self.height = cfg.IMG_H

    def forward(self, data_dict):
        img_h, img_w = data_dict['img'].shape[:2]

        y_start = int(img_h - self.height)

        data_dict['img'] = data_dict['img'][y_start:]

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][1, 2] -= y_start

        if 'depth' in data_dict:
            data_dict['depth'] = data_dict['depth'][y_start:]

        if 'ctx_img' in data_dict:
            data_dict['ctx_img'] = [img[y_start:] for img in data_dict['ctx_img']]

        if 'ctx_depth' in data_dict:
            data_dict['ctx_depth'] = [depth[y_start:] for depth in data_dict['ctx_depth']]

        data_dict['metadata']['crop_y_start'] = y_start
        data_dict['metadata']['h_before_crop'] = img_h
        data_dict['metadata']['w_before_crop'] = img_w
        return data_dict

    def backward(self, data_dict):
        depth_pred = data_dict['depth_pred']
        y_start = data_dict['metadata']['crop_y_start']
        img_h, img_w = data_dict['metadata']['h_before_crop'], data_dict['metadata']['w_before_crop']
        uncropped = np.zeros((img_h, img_w), dtype=np.float32)
        uncropped[y_start:] = depth_pred
        data_dict['depth_pred'] = uncropped
        return data_dict


@PREPROCESS_REGISTRY.register()
class Resize(Preprocess):
    def __init__(self, cfg):
        super(Resize, self).__init__(cfg)
        self.img_h = cfg.IMG_H
        self.img_w = cfg.IMG_W

    def forward(self, data_dict):
        H, W, _ = data_dict['img'].shape
        data_dict['img'] = cv2.resize(data_dict['img'], (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 0] *= self.img_w / W
            data_dict['intrinsics'][0, 2] *= self.img_w / W
            data_dict['intrinsics'][1, 1] *= self.img_h / H
            data_dict['intrinsics'][1, 2] *= self.img_h / H

        if 'depth' in data_dict:
            data_dict['depth'] = resize_depth(data_dict['depth'], (self.img_h, self.img_w))

        if 'ctx_img' in data_dict:
            data_dict['ctx_img'] = [cv2.resize(img, (self.img_w, self.img_h)) for img in data_dict['ctx_img']]

        if 'ctx_depth' in data_dict:
            data_dict['ctx_depth'] = [resize_depth(depth, (self.img_h, self.img_w))
                                      for depth in data_dict['ctx_depth']]

        data_dict['metadata']['h_before_resize'] = H
        data_dict['metadata']['w_before_resize'] = W
        return data_dict

    def backward(self, data_dict):
        img_h, img_w = data_dict['metadata']['h_before_resize'], data_dict['metadata']['w_before_resize']
        data_dict['depth_pred'] = cv2.resize(data_dict['depth_pred'], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return data_dict


@PREPROCESS_REGISTRY.register()
class RandomCrop(Preprocess):
    def __init__(self, cfg):
        super(RandomCrop, self).__init__(cfg)
        self.img_h = cfg.IMG_H
        self.img_w = cfg.IMG_W

    def forward(self, data_dict):
        assert data_dict['img'].shape[0] >= self.img_h
        assert data_dict['img'].shape[1] >= self.img_w

        img_h, img_w = data_dict['img'].shape[:2]
        x_start = random.randint(0, img_w - self.img_w)
        y_start = random.randint(0, img_h - self.img_h)
        x_end = x_start + self.img_w
        y_end = y_start + self.img_h

        data_dict['img'] = data_dict['img'][y_start:y_end, x_start:x_end]

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 2] -= x_start
            data_dict['intrinsics'][1, 2] -= y_start

        if 'depth' in data_dict:
            data_dict['depth'] = data_dict['depth'][y_start:y_end, x_start:x_end]

        if 'ctx_img' in data_dict:
            data_dict['ctx_img'] = [img[y_start:y_end, x_start:x_end] for img in data_dict['ctx_img']]

        if 'ctx_depth' in data_dict:
            data_dict['ctx_depth'] = [depth[y_start:y_end, x_start:x_end] for depth in data_dict['ctx_depth']]

        data_dict['metadata']['rand_y_start'] = y_start
        data_dict['metadata']['rand_x_start'] = x_start
        data_dict['metadata']['h_before_rand_crop'] = img_h
        data_dict['metadata']['w_before_rand_crop'] = img_w
        return data_dict

    def backward(self, data_dict):
        depth_pred = data_dict['depth_pred']
        x_start, y_start = data_dict['metadata']['rand_x_start'], data_dict['metadata']['rand_y_start']
        img_h, img_w = data_dict['metadata']['h_before_rand_crop'], data_dict['metadata']['w_before_rand_crop']
        uncropped = np.zeros((img_h, img_w), dtype=np.float32)
        uncropped[y_start:depth_pred.shape[-2], x_start:depth_pred.shape[-1]] = depth_pred
        data_dict['depth_pred'] = uncropped
        return data_dict


@PREPROCESS_REGISTRY.register()
class RandomFlip(Preprocess):
    def __init__(self, cfg):
        super(RandomFlip, self).__init__(cfg)

    def forward(self, data_dict):
        data_dict['flip'] = random.random() > 0.5
        return data_dict


@PREPROCESS_REGISTRY.register()
class ClipDepth(Preprocess):
    def __init__(self, cfg):
        super(ClipDepth, self).__init__(cfg)
        self.max_depth = cfg.MAX_DEPTH

    def forward(self, data_dict):
        if 'depth' in data_dict:
            data_dict['depth'] = np.clip(data_dict['depth'], 0, self.max_depth)

        if 'ctx_depth' in data_dict:
            data_dict['ctx_depth'] = [np.clip(depth, 0, self.max_depth) for depth in data_dict['ctx_depth']]

        return data_dict


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


@PREPROCESS_REGISTRY.register()
class RandomImageAug(Preprocess):
    def __init__(self, cfg):
        super(RandomImageAug, self).__init__(cfg)
        self.jitter_prob = cfg.get('JITTER_PROB', 1.0)
        jitter_params = cfg.get('JITTER_PARAMS', (0.2, 0.2, 0.2, 0.05))
        self.brightness = [max(1 - float(jitter_params[0]), 0.0), 1 + float(jitter_params[0])]
        self.contrast = [max(1 - float(jitter_params[1]), 0.0), 1 + float(jitter_params[1])]
        self.saturation = [max(1 - float(jitter_params[2]), 0.0), 1 + float(jitter_params[2])]
        self.hue = [-float(jitter_params[3]), float(jitter_params[3])]

        self.fn_idx = None
        self.b = None
        self.c = None
        self.s = None
        self.h = None
        self.get_params()

    def get_params(self):
        self.fn_idx = torch.randperm(4)
        self.b = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        self.c = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
        self.s = float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
        self.h = float(torch.empty(1).uniform_(self.hue[0], self.hue[1]))

    def augment(self, img):
        for fn_id in self.fn_idx:
            if fn_id == 0:
                img = F.adjust_brightness(img, self.b)
            elif fn_id == 1:
                img = F.adjust_contrast(img, self.c)
            elif fn_id == 2:
                img = F.adjust_saturation(img, self.s)
            elif fn_id == 3:
                img = F.adjust_hue(img, self.h)
        return img

    def forward(self, data_dict):
        data_dict['img_orig'] = data_dict['img'].copy()

        if 'ctx_img' in data_dict:
            data_dict['ctx_img_orig'] = [img.copy() for img in data_dict['ctx_img']]

        if random.random() < self.jitter_prob:
            self.get_params()

            data_dict['img'] = np.array(self.augment(Image.fromarray(data_dict['img'])))

            # Jitter context
            if 'ctx_img' in data_dict:
                data_dict['ctx_img'] = [np.array(self.augment(Image.fromarray(cxt)))
                                        for cxt in data_dict['ctx_img']]

        # Return jittered (?) sample
        return data_dict
