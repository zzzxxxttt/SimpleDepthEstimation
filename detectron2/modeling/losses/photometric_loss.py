# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssim_loss import SSIM
from ...geometry.camera import img_to_points, points_to_img, inv_intrinsics


class PhotometricLoss(nn.Module):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    ssim_loss_weight : float
        Weight for the SSIM loss
    C1,C2 : float
        SSIM parameters
    clip_loss : float
        Threshold for photometric loss clipping
    """

    def __init__(self, ssim_loss_weight=0.85, C1=1e-4, C2=9e-4, clip_loss=0.5):
        super().__init__()
        self.ssim_loss_weight = ssim_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.clip_loss = clip_loss

    def forward(self,
                image_src,
                image_dst,
                depth_src=None,
                intrinsics_src=None,
                T_src_to_dst=None):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image_src : torch.Tensor [B,3,H,W]
            Original image
        image_dst : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        depth_src : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        intrinsics_src : torch.Tensor [B,3,3]
            Original camera intrinsics
        T_src_to_dst : torch.Tensor [B,4,4]
            Camera transformation between original and context

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        if depth_src is not None:
            # Reconstruct world points from target_camera
            T = torch.eye(4, device=image_src.device)
            T[:3, :3] = inv_intrinsics(intrinsics_src)
            points = img_to_points(depth_src, T)

            # Project world points onto reference camera
            T = T_src_to_dst.clone()
            T[:3, :3] = T[:3, :3].bmm(intrinsics_src)
            coords_dst = points_to_img(points, T)

            # View-synthesis given the projected reference points
            warped_dst = F.grid_sample(image_dst, coords_dst,
                                       mode='bilinear', padding_mode='zeros', align_corners=True)
        else:
            warped_dst = image_dst

        l1_loss = torch.abs(warped_dst - image_src)

        # SSIM loss
        if self.ssim_loss_weight > 0.0:

            ssim_loss = SSIM(warped_dst, image_src, C1=self.C1, C2=self.C2, kernel_size=3)
            ssim_loss = torch.clamp((1. - ssim_loss) / 2., 0., 1.)

            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = self.ssim_loss_weight * ssim_loss.mean(1, True) + \
                               (1 - self.ssim_loss_weight) * l1_loss.mean(1, True)
        else:
            photometric_loss = l1_loss

        # Clip loss
        if self.clip_loss > 0.0:
            mean, std = photometric_loss.mean(), photometric_loss.std()
            photometric_loss = torch.clamp(photometric_loss, max=float(mean + self.clip_loss * std))

        # Return losses and metrics
        return photometric_loss
