import logging
from collections import defaultdict

import torch
import torch.nn as nn

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from ..pose_net import build_pose_net
from ..losses.smoothness_loss import smoothness_loss_fn
from ..losses.photometric_loss import PhotometricLoss
from ..losses.losses import silog_loss, variance_loss_fn
from ..losses.ssim_loss import SSIM
from ...utils.memory import to_cuda
from ...geometry.camera import resize_img, scale_intrinsics, view_synthesis

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MonoDepth2Model(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__()

        self.depth_net = build_depth_net(cfg)
        self.pose_net = build_pose_net(cfg)
        self.ssim = SSIM(cfg.LOSS.C1, cfg.LOSS.C2)
        self.self_supervise_loss = PhotometricLoss(ssim_loss_weight=cfg.LOSS.SSIM_WEIGHT,
                                                   C1=cfg.LOSS.C1,
                                                   C2=cfg.LOSS.C2,
                                                   clip_loss=cfg.LOSS.CLIP)

        self.ssim_loss_weight = cfg.LOSS.SSIM_WEIGHT
        self.photometric_reduce = cfg.LOSS.PHOTOMETRIC_REDUCE
        self.use_automask = cfg.LOSS.AUTOMASK
        self.clip_loss = cfg.LOSS.CLIP

        self.var_loss_w = cfg.LOSS.VAR_LOSS_WEIGHT
        self.sup_loss_w = cfg.LOSS.SUPERVISED_WEIGHT
        self.smooth_loss_w = cfg.LOSS.SMOOTHNESS_WEIGHT

        self.supervise_loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):
        batch = to_cuda(batch, self.device)

        output = {}

        batch['depth_net_input'] = (batch["img"] - self.pixel_mean) / self.pixel_std

        batch = self.depth_net(batch)

        if self.training:
            batch['pose_net_input'] = torch.cat([batch['img']] + batch['ctx_img'], 1)
            batch = self.pose_net(batch)  # num_ctx * [B, 4, 4]

            image = batch['img_orig']
            contexts = batch['ctx_img_orig']
            intrinsics = batch['intrinsics']
            depth_pred = batch['depth_pred']

            num_scales = len(depth_pred)

            losses = defaultdict(lambda: 0)
            photo_losses = [[] for _ in range(num_scales)]

            for i in range(num_scales):
                # scale_w = 1.0 / 2 ** i
                scale_w = 1.0 / 2 ** (num_scales - i - 1)

                resized_image = resize_img(image, dst_size=depth_pred[i].shape[-2:])
                resized_intrinsics = scale_intrinsics(intrinsics.clone(),
                                                      x_scale=depth_pred[i].shape[-1] / image.shape[-1],
                                                      y_scale=depth_pred[i].shape[-2] / image.shape[-2])

                for j, (img_target, pose) in enumerate(zip(contexts, batch['pose_pred'])):
                    resized_target = resize_img(img_target, dst_size=depth_pred[i].shape[-2:])
                    photo_losses[i].append(self.rgb_consistency_loss(resized_image,
                                                                     resized_target,
                                                                     depth_pred[i],
                                                                     resized_intrinsics,
                                                                     pose[:, :3, :3],
                                                                     pose[:, :3, [3], None]))

                    if self.use_automask:
                        photo_losses[i].append(self.rgb_consistency_loss(resized_image,
                                                                         resized_target,
                                                                         depth_pred[i],
                                                                         resized_intrinsics,
                                                                         None, None))

                if self.smooth_loss_w > 0.0:
                    losses['smooth_loss'] += smoothness_loss_fn(depth_pred[i], resized_image) * \
                                             scale_w * self.smooth_loss_w / num_scales

                if self.sup_loss_w > 0.0:
                    depth_gt = resize_img(batch['depth'], depth_pred[i].shape[-2:], mode='nearest')
                    losses['sup_loss'] += self.supervise_loss(depth_pred[i], depth_gt) * \
                                          scale_w * self.smooth_loss_w / num_scales

                if self.var_loss_w > 0.0:
                    losses['var_loss'] += variance_loss_fn(depth_pred[i]) * scale_w * self.var_loss_w / num_scales

            # Calculate reduced photometric loss
            if self.photometric_reduce == 'mean':
                photo_losses = [sum([l.mean() for l in losses]) / len(losses) for losses in photo_losses]
            elif self.photometric_reduce == 'min':
                photo_losses = [torch.cat(losses, 1).min(1, True)[0].mean() for losses in photo_losses]
            else:
                raise NotImplementedError

            # average over scales
            output['rec_loss'] = sum(photo_losses) / num_scales
            output.update(losses)
        else:
            output['depth_pred'] = batch['depth_pred'][0]
        return output

    def rgb_consistency_loss(self, frame_A, frame_B, depth_A, intrinsics, R_A2B=None, t_A2B=None):

        if R_A2B is not None and t_A2B is not None:
            sampled_frame_B, _, _, _ = view_synthesis(frame_B, depth_A, intrinsics, R_A2B, t_A2B)
        else:
            sampled_frame_B = frame_B

        photometric_loss = (torch.abs(sampled_frame_B - frame_A)).mean(1, True)

        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            rgb_ssim_loss = self.ssim(sampled_frame_B, frame_A).mean(1, True)
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = rgb_ssim_loss * self.ssim_loss_weight + \
                               photometric_loss * (1 - self.ssim_loss_weight)

        # Clip loss
        if self.clip_loss > 0.0:
            mean, std = photometric_loss.mean(), photometric_loss.std()
            photometric_loss = torch.clamp(photometric_loss, max=float(mean + self.clip_loss * std))

        return photometric_loss
