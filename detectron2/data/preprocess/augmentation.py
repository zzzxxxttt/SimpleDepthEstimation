import cv2
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

import torch

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
        img_h, img_w = data_dict['image'].shape[:2]
        y_start = int(data_dict['image'].shape[0] - 352)
        x_start = int((data_dict['image'].shape[1] - 1216) / 2)
        data_dict['image'] = data_dict['image'][y_start:y_start + 352, x_start:x_start + 1216]

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 2] -= x_start
            data_dict['intrinsics'][1, 2] -= y_start

        if 'depth_gt' in data_dict:
            data_dict['depth_gt'] = data_dict['depth_gt'][y_start:y_start + 352, x_start:x_start + 1216]

        if 'context' in data_dict:
            data_dict['context'] = [img[y_start:y_start + 352, x_start:x_start + 1216]
                                    for img in data_dict['context']]

        if 'context_depth_gt' in data_dict:
            data_dict['context_depth_gt'] = [depth[y_start:y_start + 352, x_start:x_start + 1216]
                                             for depth in data_dict['context_depth_gt']]

        if 'metadata' not in data_dict:
            data_dict['metadata'] = {}

        data_dict['metadata']['kb_y_start'] = y_start
        data_dict['metadata']['kb_x_start'] = x_start
        data_dict['metadata']['h_before_kb_crop'] = img_h
        data_dict['metadata']['w_before_kb_crop'] = img_w
        return data_dict

    def inverse(self, data_dict):
        depth_pred = data_dict['depth_pred']
        x_start, y_start = data_dict['metadata']['kb_x_start'], data_dict['metadata']['kb_y_start']
        img_h, img_w = data_dict['metadata']['h_before_kb_crop'], data_dict['metadata']['w_before_kb_crop']
        uncropped = np.zeros((img_h, img_w), dtype=np.float32)
        uncropped[y_start:depth_pred.shape[-2], x_start:depth_pred.shape[-1]] = depth_pred
        data_dict['depth_pred'] = uncropped
        return data_dict


@PREPROCESS_REGISTRY.register()
class Resize(Preprocess):
    def __init__(self, cfg):
        super(Resize, self).__init__(cfg)
        self.img_h = cfg.IMG_H
        self.img_w = cfg.IMG_W

    def forward(self, data_dict):
        H, W, _ = data_dict['image'].shape
        data_dict['image'] = cv2.resize(data_dict['image'], (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 0] *= self.img_w / W
            data_dict['intrinsics'][0, 2] *= self.img_w / W
            data_dict['intrinsics'][1, 1] *= self.img_h / H
            data_dict['intrinsics'][1, 2] *= self.img_h / H

        if 'depth_gt' in data_dict:
            data_dict['depth_gt'] = resize_depth(data_dict['depth_gt'], (self.img_h, self.img_w))

        if 'context' in data_dict:
            data_dict['context'] = [cv2.resize(img, (self.img_w, self.img_h))
                                    for img in data_dict['context']]

        if 'context_depth_gt' in data_dict:
            data_dict['context_depth_gt'] = [resize_depth(depth, (self.img_h, self.img_w))
                                             for depth in data_dict['context_depth_gt']]

        if 'metadata' not in data_dict:
            data_dict['metadata'] = {}

        data_dict['metadata']['h_before_resize'] = H
        data_dict['metadata']['w_before_resize'] = W
        return data_dict

    def inverse(self, data_dict):
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
        assert data_dict['image'].shape[0] >= self.img_h
        assert data_dict['image'].shape[1] >= self.img_w
        img_h, img_w = data_dict['image'].shape[:2]
        x_start = random.randint(0, data_dict['image'].shape[1] - self.img_w)
        y_start = random.randint(0, data_dict['image'].shape[0] - self.img_h)
        data_dict['image'] = data_dict['image'][y_start:y_start + self.img_h, x_start:x_start + self.img_w, :]

        if 'intrinsics' in data_dict:
            data_dict['intrinsics'][0, 2] -= x_start
            data_dict['intrinsics'][1, 2] -= y_start

        if 'depth_gt' in data_dict:
            data_dict['depth_gt'] = data_dict['depth_gt'][y_start:y_start + self.img_h, x_start:x_start + self.img_w]

        if 'context' in data_dict:
            data_dict['context'] = [img[y_start:y_start + self.img_h, x_start:x_start + self.img_w, :]
                                    for img in data_dict['context']]

        if 'context_depth_gt' in data_dict:
            data_dict['context_depth_gt'] = [depth[y_start:y_start + self.img_h, x_start:x_start + self.img_w]
                                             for depth in data_dict['context_depth_gt']]

        if 'metadata' not in data_dict:
            data_dict['metadata'] = {}

        data_dict['metadata']['rand_y_start'] = y_start
        data_dict['metadata']['rand_x_start'] = x_start
        data_dict['metadata']['h_before_rand_crop'] = img_h
        data_dict['metadata']['w_before_rand_crop'] = img_w
        return data_dict

    def inverse(self, data_dict):
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
        if 'depth_gt' in data_dict:
            data_dict['depth_gt'] = np.clip(data_dict['depth_gt'], 0, self.max_depth)

        if 'context_depth_gt' in data_dict:
            data_dict['context_depth_gt'] = [np.clip(depth, 0, self.max_depth)
                                             for depth in data_dict['context_depth_gt']]

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
        self.jitter_params = cfg.get('JITTER_PARAMS', (0.2, 0.2, 0.2, 0.05))

    def forward(self, data_dict):
        data_dict['image_orig'] = data_dict['image'].copy()

        if 'context' in data_dict:
            data_dict['context_orig'] = [img.copy() for img in data_dict['context']]

        if random.random() < self.jitter_prob:
            # Prepare transformation
            color_augmentation = transforms.ColorJitter()
            brightness, contrast, saturation, hue = self.jitter_params
            augment_image = color_augmentation.get_params(brightness=[max(0., 1 - brightness), 1 + brightness],
                                                          contrast=[max(0., 1 - contrast), 1 + contrast],
                                                          saturation=[max(0., 1 - saturation), 1 + saturation],
                                                          hue=[-hue, hue])
            # Jitter single items
            data_dict['image'] = np.array(augment_image(Image.fromarray(data_dict['image'])))

            # Jitter context
            if 'context' in data_dict:
                data_dict['context'] = [np.array(augment_image(Image.fromarray(cxt)))
                                        for cxt in data_dict['context']]
        # Return jittered (?) sample
        return data_dict


@PREPROCESS_REGISTRY.register()
class ToTensor(Preprocess):
    def __init__(self, cfg):
        super(ToTensor, self).__init__(cfg)
        self.to_tensor = transforms.ToTensor()

    def forward(self, data_dict):
        for key in data_dict:
            if key in ['image', 'image_orig']:
                data_dict[key] = self.to_tensor(data_dict[key])
            elif key in ['intrinsics', 'pose_gt', 'depth_gt']:
                if key == 'depth_gt':
                    data_dict[key] = data_dict[key][None, :, :]
                data_dict[key] = torch.from_numpy(data_dict[key])
            elif key in ['context', 'context_orig']:
                data_dict[key] = [self.to_tensor(img) for img in data_dict[key]]
            elif key in ['context_pose_gt', 'context_depth_gt']:
                if key == 'depth_gt':
                    data_dict[key] = [d[None, :, :] for d in data_dict[key]]
                data_dict[key] = [torch.from_numpy(p) for p in data_dict[key]]
        return data_dict
