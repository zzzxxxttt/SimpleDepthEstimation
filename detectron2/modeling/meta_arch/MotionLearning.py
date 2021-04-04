import logging

import torch
import torch.nn as nn

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from ..pose_net import build_pose_net
from ...utils.memory import to_cuda
from ...geometry.pose_utils import pose_vec2mat
from ...geometry.camera import resize_img, scale_intrinsics
from ..losses.smoothness_loss import cal_smoothness_loss
from ..losses.photometric_loss import PhotometricLoss
from ..losses.losses import silog_loss, variance_loss
from .SupDepth import post_process

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MotionLearningModel(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__()

        self.depth_net = build_depth_net(cfg)
        self.pose_net = build_pose_net(cfg)
        self.self_supervise_loss = PhotometricLoss(ssim_loss_weight=cfg.LOSS.SSIM_WEIGHT,
                                                   C1=cfg.LOSS.C1,
                                                   C2=cfg.LOSS.C2,
                                                   clip_loss=cfg.LOSS.CLIP)

        self.use_automask = cfg.LOSS.AUTOMASK
        self.smooth_loss_weight = cfg.LOSS.SMOOTHNESS_WEIGHT
        self.photometric_reduce = cfg.LOSS.PHOTOMETRIC_REDUCE
        self.sup_loss_weight = cfg.LOSS.SUPERVISED_WEIGHT
        self.supervise_loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)
        self.var_loss_weight = cfg.LOSS.VAR_LOSS_WEIGHT

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):
        batch = to_cuda(batch, self.device)

        image_no_norm = batch["image"]
        batch['image'] = (batch["image"] - self.pixel_mean) / self.pixel_std

        output = self.depth_net(batch)

        if self.training:
            # [B, N, 6]
            pose_vec = self.pose_net(image_no_norm, batch['context'])
            poses = [pose_vec2mat(pose_vec[:, i]) for i in range(pose_vec.shape[1])]

            photometric_loss, smoothness_loss = self.calc_self_sup_losses(batch['image_orig'],
                                                                          batch['context_orig'],
                                                                          output['depth_pred'],
                                                                          batch['intrinsics'],
                                                                          poses)
            output['rec_loss'] = photometric_loss
            output['smooth_loss'] = self.smooth_loss_weight * smoothness_loss

            if self.sup_loss_weight > 0.0:
                depth_gt = [resize_img(batch['depth_gt'], pred.shape[-2:], mode='nearest')
                            for pred in output['depth_pred']]
                sup_losses = [self.supervise_loss(pred, gt) for pred, gt in zip(output['depth_pred'], depth_gt)]
                output['silog_loss'] = self.sup_loss_weight * sum(sup_losses) / len(sup_losses)

            if self.var_loss_weight > 0.0:
                var_losses = [variance_loss(d) for d in output['depth_pred']]
                output['var_loss'] = self.var_loss_weight * sum(var_losses) / len(var_losses)

        else:
            output['depth_pred'] = post_process(output['depth_pred'][0], batch)
        return output

    def calc_self_sup_losses(self, image, contexts, depth_pred, intrinsics, poses):
        num_scales = len(depth_pred)

        photo_losses = [[] for _ in range(num_scales)]
        smooth_losses = []

        # todo understand this!
        for i in range(num_scales):

            resized_image = resize_img(image, dst_size=depth_pred[i].shape[-2:])
            resized_intrinsics = scale_intrinsics(intrinsics.clone(),
                                                  x_scale=depth_pred[i].shape[-1] / image.shape[-1],
                                                  y_scale=depth_pred[i].shape[-2] / image.shape[-2])

            for j, (img_target, pose) in enumerate(zip(contexts, poses)):
                resized_target = resize_img(img_target, dst_size=depth_pred[i].shape[-2:])

                photo_losses[i].append(self.self_supervise_loss(resized_image,
                                                                resized_target,
                                                                depth_pred[i],
                                                                resized_intrinsics,
                                                                pose))

                if self.use_automask:
                    photo_losses[i].append(self.self_supervise_loss(resized_image, resized_target))

            if self.smooth_loss_weight > 0.0:
                smooth_losses.append(cal_smoothness_loss(depth_pred[i], resized_image))

        # Calculate reduced photometric loss
        if self.photometric_reduce == 'mean':
            photo_losses = [sum([l.mean() for l in losses]) / len(losses) for losses in photo_losses]
        elif self.photometric_reduce == 'min':
            photo_losses = [torch.cat(losses, 1).min(1, True)[0].mean() for losses in photo_losses]
        else:
            raise NotImplementedError

        # average over scales
        photo_loss = sum(photo_losses) / num_scales

        smooth_loss = sum([s / 2 ** i for i, s in enumerate(smooth_losses)]) / num_scales

        return photo_loss, smooth_loss
