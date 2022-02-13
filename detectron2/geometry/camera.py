# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


def inv_intrinsics(K):
    assert K.dim() == 3
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    K_inv = K.clone()
    K_inv[:, 0, 0] = 1. / fx
    K_inv[:, 1, 1] = 1. / fy
    K_inv[:, 0, 2] = -1. * cx / fx
    K_inv[:, 1, 2] = -1. * cy / fy
    return K_inv


def resize_img(image, dst_size, mode='bilinear'):
    if image.shape[-2] == dst_size[-2] and image.shape[-1] == dst_size[-1]:
        return image
    else:
        resized_image = F.interpolate(image, size=dst_size, mode=mode,
                                      align_corners=True if mode != 'nearest' else None)
        return resized_image


def resize_img_avgpool(image, dst_size):
    if image.shape[-2] == dst_size[-2] and image.shape[-1] == dst_size[-1]:
        return image
    else:
        resized_image = F.adaptive_avg_pool2d(image, dst_size)
        return resized_image


@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid


def img_to_points(depth, R, t):
    B, C, H, W = depth.shape
    assert C == 1
    assert R.dim() == 3
    assert t.dim() == 3

    # Create flat index grid
    grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
    grid = grid * depth
    flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

    points = R.bmm(flat_grid) + t

    return points.view(B, 3, H, W)


def points_to_img(points, R, t):
    B, C, H, W = points.shape
    assert C == 3
    assert R.dim() == 3
    assert t.dim() == 3

    proj = R.bmm(points.view(B, 3, -1)) + t

    # Normalize points
    X = proj[:, 0] / (proj[:, 2] + 1e-6)
    Y = proj[:, 1] / (proj[:, 2] + 1e-6)
    Z = proj[:, 2]

    valid_proj_mask = (X >= 0) & (X < W) & \
                      (Y >= 0) & (Y < H) & \
                      (Z > 0)

    Z = Z.clamp(min=1e-5)

    Xnorm = 2 * X / (W - 1) - 1.
    Ynorm = 2 * Y / (H - 1) - 1.

    # Clamp out-of-bounds pixels
    # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
    # Xnorm[Xmask] = 2.
    # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
    # Ynorm[Ymask] = 2.

    # Return pixel coordinates
    return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2), \
           Z.view(B, H, W, 1), \
           valid_proj_mask.view(B, H, W, 1)


def view_synthesis(image_B, depth_A, intrinsics, R_A_to_B, t_A_to_B):
    R = R_A_to_B.clone()  # [B, 3, 3]
    t = t_A_to_B.clone()  # [B, 3, 1, 1] or [B, 3, H, W]
    B, _, H, W = t.shape

    # Reconstruct world points from target_camera
    points_A = img_to_points(depth_A,
                             R=inv_intrinsics(intrinsics),
                             t=torch.zeros([image_B.shape[0], 3, 1], device=image_B.device))

    # Project world points onto reference camera
    R = intrinsics.bmm(R)
    t = intrinsics.bmm(t.view(B, 3, H * W))

    points_A_coords_in_B, points_A_depth_in_B, valid_proj_mask = points_to_img(points_A, R, t)

    # View-synthesis given the projected reference points
    sampled_B = F.grid_sample(image_B, points_A_coords_in_B,
                              mode='bilinear', padding_mode='zeros', align_corners=True)

    return (sampled_B,
            points_A_depth_in_B[:, None, :, :, 0],
            points_A_coords_in_B,
            valid_proj_mask[:, None, :, :, 0])
