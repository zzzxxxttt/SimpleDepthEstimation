import logging

import torch
import torch.nn as nn

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from ..pose_net import build_pose_net
from ...utils.memory import to_cuda
from ...geometry.pose_utils import pose_vec2mat
from ...geometry.camera import resize_img, scale_intrinsics, resize_img_avgpool, view_synthesis
from ..losses.smoothness_loss import cal_smoothness_loss, cal_motion_smoothness_loss
from ..losses.photometric_loss import PhotometricLoss
from ..losses.ssim_loss import WeightedSSIM
from ..losses.losses import silog_loss, variance_loss, sparsity_loss
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

        self.ssim_loss_weight = cfg.LOSS.SSIM_WEIGHT
        self.ssim = WeightedSSIM(cfg.LOSS.C1, cfg.LOSS.C2)
        self.clip_loss = cfg.LOSS.CLIP

        self.use_automask = cfg.LOSS.AUTOMASK
        self.smooth_loss_weight = cfg.LOSS.SMOOTHNESS_WEIGHT
        self.photometric_reduce = cfg.LOSS.PHOTOMETRIC_REDUCE
        self.sup_loss_weight = cfg.LOSS.SUPERVISED_WEIGHT
        self.supervise_loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)
        self.var_loss_weight = cfg.LOSS.VAR_LOSS_WEIGHT
        self.motion_smooth_loss_weight = cfg.LOSS.MOTION_SMOOTHNESS_WEIGHT
        self.motion_sparsity_loss_weight = cfg.LOSS.MOTION_SPARSITY_WEIGHT
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

            pose_out = self.pose_net(frame1, frame2)  # pose: [2B, 6] motion: [2B, 3, H, W]
            poses = pose_vec2mat(pose_out['pose'])  # [2B, 4, 4]

            pose_1to2 = poses
            pose_2to1 = None
            motion_1to2 = None
            motion_2to1 = None
            if pose_out['motion'] is not None:
                motion_1to2 = pose_out['motion']
                if self.cycle_consistency:
                    pose_1to2, pose_2to1 = torch.chunk(poses, 2, dim=0)
                    motion_1to2, motion_2to1 = torch.chunk(motion_1to2, 2, dim=0)

            losses = {}
            num_scales = len(depth1)

            for i in range(num_scales):

                resized_frame1 = resize_img_avgpool(frame1, dst_size=depth1[i].shape[-2:])
                resized_frame2 = resize_img_avgpool(frame2, dst_size=depth2[i].shape[-2:])

                resized_intrinsics = scale_intrinsics(batch['intrinsics'].clone(),
                                                      x_scale=depth1[i].shape[-1] / frame1.shape[-1],
                                                      y_scale=depth1[i].shape[-2] / frame1.shape[-2])

                if motion_1to2 is not None:
                    resized_motion_1to2 = resize_img_avgpool(motion_1to2, depth1[i].shape[-2:])
                else:
                    resized_motion_1to2 = None

                losses_A2B = self.calc_loss_(resized_frame1, resized_frame2,
                                             depth1[i], depth2[i],
                                             resized_intrinsics,
                                             pose_1to2[:, :3, :3],
                                             pose_1to2[:, :3, [3], None],
                                             resized_motion_1to2,
                                             batch['depth_1_gt'])

                losses = {k: v if k not in losses else v + losses[k] for k, v in losses_A2B.items()}

                if pose_2to1 is not None:
                    if motion_2to1 is not None:
                        resized_motion_2to1 = resize_img_avgpool(motion_2to1, depth2[i].shape[-2:])
                    else:
                        resized_motion_2to1 = None

                    losses_B2A = self.calc_loss_(resized_frame2, resized_frame1,
                                                 depth2[i], depth1[i],
                                                 resized_intrinsics,
                                                 pose_2to1[:, :3, :3],
                                                 pose_2to1[:, :3, [3], None],
                                                 resized_motion_2to1,
                                                 batch['depth_2_gt'])

                    losses = {k: v if k not in losses else v + losses[k] for k, v in losses_B2A.items()}

            output.update(losses)

        else:
            batch['depth_net_input'] = (frame1 - self.pixel_mean) / self.pixel_std
            output = self.depth_net(batch)
            output['depth_pred'] = post_process(output['depth_pred'][0], batch)

        return output

    def calc_loss_(self,
                   frame_A, frame_B,
                   depth_A, depth_B,
                   intrinsics,
                   R_A_to_B, t_A_to_B, t_A_to_B_res=None,
                   depth_B_gt=None,
                   scale=0):

        if self.scale_normalize:
            depth_mean = torch.mean(torch.cat([depth_A, depth_B], 0))
            depth_A_normalized = depth_A / depth_mean
            depth_B_normalized = depth_B / depth_mean
            t_A_to_B /= depth_mean
            if t_A_to_B_res is not None:
                t_A_to_B_res /= depth_mean
        else:
            depth_A_normalized = depth_A
            depth_B_normalized = depth_B

        translation_map = t_A_to_B if t_A_to_B_res is None else t_A_to_B + t_A_to_B_res

        losses = {}

        sampled_values, depth_in_B, proj_mask = view_synthesis(torch.cat([frame_B, depth_B_normalized], 1),
                                                               depth_A_normalized, intrinsics,
                                                               R_A_to_B, translation_map)

        sampled_frame_B, sampled_depth_B = torch.split(sampled_values, [3, 1], dim=1)

        mask = (depth_in_B < sampled_depth_B).float()

        depth_error = (depth_in_B - sampled_depth_B) ** 2
        depth_error_2nd_moment = ((depth_error * mask).sum([1, 2, 3]) / (mask.sum([1, 2, 3]) + 1.0)) + 1e-4
        depth_proximity_weight = (depth_error_2nd_moment / (depth_error + depth_error_2nd_moment))
        depth_proximity_weight = (depth_proximity_weight * proj_mask.float()).detach()

        scale_w = 1.0 / 2 ** scale

        l1_loss = torch.abs(sampled_frame_B - frame_A)
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss, avg_weight = self.ssim(sampled_frame_B, frame_A, depth_proximity_weight)
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            losses['photometric_losses'] = (self.ssim_loss_weight * ssim_loss.mean(1, True) +
                                            (1 - self.ssim_loss_weight) * l1_loss.mean(1, True)) * scale_w
        else:
            losses['photometric_losses'] = l1_loss.mean(1, True) * scale_w

        if self.smooth_loss_weight > 0.0:
            losses['smooth_losses'] = cal_smoothness_loss(depth_B_normalized, frame_B) * scale_w

        t_norm = (t_A_to_B.norm(2, dim=[1, 2, 3], keepdim=True) * 3.0).mean()
        t_A_to_B_res_norm = t_A_to_B_res / torch.sqrt(t_norm + 1e-5)

        if t_A_to_B.shape[-1] > 1 and self.motion_smooth_loss_weight > 0.0:
            losses['motion_smooth_losses'] = cal_motion_smoothness_loss(t_A_to_B_res_norm) * scale_w

        if t_A_to_B.shape[-1] > 1 and self.motion_sparsity_loss_weight > 0.0:
            losses['motion_sparsity_losses'] = sparsity_loss(t_A_to_B_res_norm) * scale_w

        if self.sup_loss_weight > 0.0:
            depth_gt = resize_img(depth_B_gt, depth_B.shape[-2:], mode='nearest')
            losses['sup_losses'] = self.supervise_loss(depth_B, depth_gt) * scale_w

        if self.var_loss_weight > 0.0:
            losses['var_loss'] = variance_loss(depth_B) * scale_w

        losses = {k: sum(v) / len(v) for k, v in losses.items()}
        return losses
