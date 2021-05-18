# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssim_loss import SSIM, WeightedSSIM
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

    def forward(self, real_image, synthesised_image):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        real_image : torch.Tensor [B,3,H,W]
            Original image
        synthesised_image : torch.Tensor [B,3,H,W]
            Context containing a list of reference images

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        l1_loss = torch.abs(synthesised_image - real_image)

        # SSIM loss
        if self.ssim_loss_weight > 0.0:

            ssim_loss = SSIM(synthesised_image, real_image, C1=self.C1, C2=self.C2, kernel_size=3)
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


class OcclusionAwarePhotometricLoss(nn.Module):
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
        self.ssim = WeightedSSIM(C1, C2)
        self.clip_loss = clip_loss

    def forward(self, real_image, synthesised_image, sampled_depth, synthesised_depth, proj_mask):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        real_image : torch.Tensor [B,3,H,W]
            Original image
        synthesised_image : torch.Tensor [B,3,H,W]
            Context containing a list of reference images

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        mask = (synthesised_depth < sampled_depth).float()

        sqared_depth_error = (sampled_depth - synthesised_depth) ** 2
        depth_error_second_moment = ((sqared_depth_error * mask).sum([1, 2, 3]) / (mask.sum([1, 2, 3]) + 1.0)) + 1e-4

        depth_proximity_weight = (depth_error_second_moment / (sqared_depth_error + depth_error_second_moment))
        depth_proximity_weight = (depth_proximity_weight * proj_mask.float()).detach()

        l1_loss = torch.abs(synthesised_image - real_image)

        # SSIM loss
        if self.ssim_loss_weight > 0.0:

            ssim_loss, avg_weight = self.ssim(synthesised_image, real_image, depth_proximity_weight)

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
