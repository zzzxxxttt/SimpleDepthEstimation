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


def img_to_points(depth, T):
    B, C, H, W = depth.shape
    assert C == 1
    assert T.dim() == 3

    # Create flat index grid
    grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
    grid = grid * depth
    flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

    points = T[:, :3, :3].bmm(flat_grid) + T[:, :3, [3]]

    return points.view(B, 3, H, W)


def points_to_img(points, T):
    B, C, H, W = points.shape
    assert C == 3
    assert T.dim() == 3

    proj = T[:, :3, :3].bmm(points.view(B, 3, -1)) + T[:, :3, [3]]

    # Normalize points
    X = proj[:, 0]
    Y = proj[:, 1]
    Z = proj[:, 2].clamp(min=1e-5)
    Xnorm = 2 * (X / Z) / (W - 1) - 1.
    Ynorm = 2 * (Y / Z) / (H - 1) - 1.

    # Clamp out-of-bounds pixels
    # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
    # Xnorm[Xmask] = 2.
    # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
    # Ynorm[Ymask] = 2.

    # Return pixel coordinates
    return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
