import logging

import torch
import torch.nn as nn

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from ..pose_net import build_pose_net
from ...utils.memory import to_cuda
from ...geometry.pose_utils import pose_vec2mat
from ...geometry.camera import resize_img, scale_intrinsics, resize_img_avgpool
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
        self.scale_normalize = cfg.LOSS.SCALE_NORMALIZE

        self.pose_use_depth = cfg.MODEL.POSE_NET.USE_DEPTH
        self.cycle_consistency = cfg.MODEL.POSE_NET.CYCLE_CONSISTENCY

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):
        batch = to_cuda(batch, self.device)

        # currently only support two frames
        frame1 = batch["image"]
        frame2 = batch["context"][0]

        if self.training:
            batch['depth_net_input'] = torch.cat([(frame1 - self.pixel_mean) / self.pixel_std,
                                                  (frame2 - self.pixel_mean) / self.pixel_std], 0)
            output = self.depth_net(batch)

            depth1, depth2 = zip(*[torch.chunk(d, 2, dim=0) for d in output['depth_pred']])

            if self.pose_use_depth:
                frame1 = torch.cat([frame1, depth1[0]], 1)
                frame2 = torch.cat([frame2, depth2[0]], 1)

            if self.cycle_consistency:
                frame1, frame2 = torch.cat([frame1, frame2], 0), \
                                 torch.cat([frame2, frame1], 0)

            pose_out = self.pose_net(frame1, frame2)  # [B, 6]

            poses = pose_vec2mat(pose_out['pose'])  # [B, 4, 4]

            photometric_loss, smoothness_loss = self.calc_self_sup_losses(frame1, frame2,
                                                                          depth1, depth2,
                                                                          batch['intrinsics'],
                                                                          poses, pose_out['motion'])
            output['rec_loss'] = photometric_loss
            output['smooth_loss'] = smoothness_loss

            if self.sup_loss_weight > 0.0:
                depth_gt = [resize_img(batch['depth_gt'], pred.shape[-2:], mode='nearest')
                            for pred in output['depth_pred']]
                sup_losses = [self.supervise_loss(pred, gt) for pred, gt in zip(output['depth_pred'], depth_gt)]
                output['silog_loss'] = self.sup_loss_weight * sum(sup_losses) / len(sup_losses)

            if self.var_loss_weight > 0.0:
                var_losses = [variance_loss(d) / (2 ** i) for i, d in enumerate(output['depth_pred'])]
                output['var_loss'] = self.var_loss_weight * sum(var_losses) / len(var_losses)

        else:
            batch['depth_net_input'] = (frame1 - self.pixel_mean) / self.pixel_std
            output = self.depth_net(batch)
            output['depth_pred'] = post_process(output['depth_pred'][0], batch)

        return output

    def calc_self_sup_losses(self, frame1, frame2, depth1, depth2, intrinsics, pose, motion=None):
        num_scales = len(depth1)

        photometric_losses = []
        smooth_losses = []

        depth_mean = [torch.mean(torch.cat([d1, d2], 0)) for d1, d2 in zip(depth1, depth2)]
        if self.scale_normalize:
            depth1 = [d / depth_mean for d in depth1]
            depth2 = [d / depth_mean for d in depth2]
            pose[:, :3, 3] /= depth_mean
            if motion is not None:
                motion /= depth_mean

        pose_1to2 = pose
        pose_2to1 = None
        motion_1to2 = None
        motion_2to1 = None
        if motion is not None:
            motion_1to2 = motion
            if self.cycle_consistency:
                pose_1to2, pose_2to1 = torch.chunk(pose, 2, dim=0)
                motion_1to2, motion_2to1 = torch.chunk(motion, 2, dim=0)

        for i in range(num_scales):

            resized_frame1 = resize_img_avgpool(frame1, dst_size=depth1[i].shape[-2:])
            resized_frame2 = resize_img_avgpool(frame2, dst_size=depth2[i].shape[-2:])

            resized_intrinsics = scale_intrinsics(intrinsics.clone(),
                                                  x_scale=depth1[i].shape[-1] / frame1.shape[-1],
                                                  y_scale=depth1[i].shape[-2] / frame1.shape[-2])

            photometric_losses[i].append(self.self_supervise_loss(resized_frame1,
                                                                  resized_frame2,
                                                                  depth1[i],
                                                                  resized_intrinsics,
                                                                  pose_1to2))

            if self.smooth_loss_weight > 0.0:
                smooth_losses.append(cal_smoothness_loss(depth1[i], resized_frame1))

            if motion_1to2 is not None:
                resized_motion_1to2 = resize_img_avgpool(motion_1to2, depth1[i].shape[-2:])
                translation_1to2 = resized_motion_1to2 * pose_1to2[:, :3, [3], None]
            if motion_2to1 is not None:
                resized_motion_2to1 = resize_img_avgpool(motion_2to1, depth2[i].shape[-2:])
                translation_2to1 = resized_motion_2to1 * pose_2to1[:, :3, [3], None]

        # average over scales
        photo_loss = sum(photometric_losses) / num_scales

        smooth_loss = sum([s / (2 ** i) for i, s in enumerate(smooth_losses)]) / num_scales

        return photo_loss, self.smooth_loss_weight * smooth_loss
