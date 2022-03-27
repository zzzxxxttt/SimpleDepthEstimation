import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from ..pose_net import build_pose_net
from ...utils.memory import to_cuda
from ...geometry.camera import resize_img, scale_intrinsics, resize_img_avgpool, view_synthesis
from ..losses.smoothness_loss import smoothness_loss
from ..losses.motion_loss import motion_consistency_loss, motion_smoothness_loss_fn, motion_sparsity_loss_fn
from ..losses.ssim_loss import WeightedSSIM, SSIM
from ..losses.losses import silog_loss, variance_loss

logger = logging.getLogger(__name__)


def merge_loss(losses, new_losses, w=1.0):
    for k, v in new_losses.items():
        if 'loss' in k:
            losses[k] += v * w
    return losses


@META_ARCH_REGISTRY.register()
class MotionLearningModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.depth_net = build_depth_net(cfg)
        self.pose_net = build_pose_net(cfg)

        self.num_scales = cfg.LOSS.NUM_SCALES

        self.depth_l1_loss_w = cfg.LOSS.DEPTH_L1_WEIGHT

        self.ssim_loss_w = cfg.LOSS.SSIM_WEIGHT
        self.ssim = WeightedSSIM(cfg.LOSS.C1, cfg.LOSS.C2)
        # self.ssim = SSIM(cfg.LOSS.C1, cfg.LOSS.C2)
        self.clip_loss = cfg.LOSS.CLIP

        self.smooth_loss_w = cfg.LOSS.SMOOTHNESS_WEIGHT

        self.sup_loss_w = cfg.LOSS.SUPERVISED_WEIGHT
        self.supervise_loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)

        self.var_loss_w = cfg.LOSS.VAR_LOSS_WEIGHT

        self.motion_smooth_loss_w = cfg.LOSS.MOTION_SMOOTHNESS_WEIGHT
        self.motion_sparsity_loss_w = cfg.LOSS.MOTION_SPARSITY_WEIGHT
        self.rot_cycle_loss_w = cfg.LOSS.ROT_CYCLE_WEIGHT
        self.trans_cycle_loss_w = cfg.LOSS.TRANS_CYCLE_WEIGHT
        self.motion_sparsity_loss_w = cfg.LOSS.MOTION_SPARSITY_WEIGHT

        self.scale_normalize = cfg.LOSS.SCALE_NORMALIZE

        self.pose_use_depth = cfg.MODEL.POSE_NET.USE_DEPTH

        self.with_mask = cfg.MODEL.get('WITH_MASK', False)
        self.mask_dilation = cfg.MODEL.get('MASK_DILATION', 8)

        self.return_loss = cfg.MODEL.get('RETURN_LOSS', False)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):

        batch = to_cuda(batch, self.device)

        if self.training or self.return_loss:
            # currently only support two frames
            frame1 = batch["img"]
            frame2 = batch["ctx_img"][0]

            batch['depth_net_input'] = torch.cat([(frame1 - self.pixel_mean) / self.pixel_std,
                                                  (frame2 - self.pixel_mean) / self.pixel_std], 0)
            batch = self.depth_net(batch)

            # convert list of [2B, 1, H, W] to two list of [B, 1, H, W]
            depth1, depth2 = zip(*[torch.chunk(d, 2, dim=0) for d in batch['depth_pred']])

            pose_net_input_1 = frame1
            pose_net_input_2 = frame2
            if self.pose_use_depth:
                pose_net_input_1 = torch.cat([pose_net_input_1, depth1[0]], 1)
                pose_net_input_2 = torch.cat([pose_net_input_2, depth2[0]], 1)

            # [2B, 2C, H, W]
            batch['pose_net_input'] = torch.cat([torch.cat([pose_net_input_1, pose_net_input_2], 1),
                                                 torch.cat([pose_net_input_2, pose_net_input_1], 1)], 0)

            batch = self.pose_net(batch)

            # pose_pred: [2B, 4, 4], motion_pred: [2B, 3, H, W]
            pose_1to2, pose_2to1 = torch.chunk(batch['pose_pred'], 2, dim=0)
            motion_1to2, motion_2to1 = None, None
            if 'motion_pred' in batch:
                motion_1to2, motion_2to1 = torch.chunk(batch['motion_pred'], 2, dim=0)

                if self.with_mask:
                    mask1 = (batch["mask"] > 0).float()
                    mask2 = (batch['ctx_mask'][0] > 0).float()
                    if self.mask_dilation > 0:
                        pool_size = self.mask_dilation * 2 + 1
                        mask1 = F.max_pool2d(mask1, pool_size, stride=1, padding=self.mask_dilation)
                        mask2 = F.max_pool2d(mask2, pool_size, stride=1, padding=self.mask_dilation)
                    motion_1to2 = motion_1to2 * mask1
                    motion_2to1 = motion_2to1 * mask2

            batch['depth_proximity_weight'] = []
            batch['overall_motion'] = []
            losses = defaultdict(lambda: 0)
            for i in reversed(range(self.num_scales)):
                scale_w = 1.0 / 2 ** i
                H = int(depth1[0].shape[-2] * scale_w)
                W = int(depth1[0].shape[-1] * scale_w)

                resized_frame1 = resize_img_avgpool(frame1, dst_size=(H, W))
                resized_frame2 = resize_img_avgpool(frame2, dst_size=(H, W))

                resized_intrinsics = scale_intrinsics(batch['intrinsics'].clone(),
                                                      x_scale=scale_w, y_scale=scale_w)

                resized_depth1 = resize_img_avgpool(depth1[0], dst_size=(H, W))
                resized_depth2 = resize_img_avgpool(depth2[0], dst_size=(H, W))

                # [B, 3, 3]
                R_1to2 = pose_1to2[:, :3, :3]
                R_2to1 = pose_2to1[:, :3, :3]
                # [B, 3, 1, 1]
                t_1to2 = pose_1to2[:, :3, [3], None]
                t_2to1 = pose_2to1[:, :3, [3], None]

                if 'motion_pred' in batch:
                    resized_motion_1to2 = resize_img_avgpool(motion_1to2, (H, W))
                    resized_motion_2to1 = resize_img_avgpool(motion_2to1, (H, W))
                    # [B, 3, 1, 1] --> [B, 3, H, W]
                    t_1to2 = t_1to2 + resized_motion_1to2
                    t_2to1 = t_2to1 + resized_motion_2to1
                else:
                    resized_motion_1to2 = None
                    resized_motion_2to1 = None
                    t_1to2 = t_1to2.expand(-1, -1, H, W)
                    t_2to1 = t_2to1.expand(-1, -1, H, W)

                batch['overall_motion'].append((t_1to2, t_2to1))

                if self.scale_normalize:
                    depth_mean = torch.mean(torch.cat([resized_depth1, resized_depth2], 0))
                    # normalize depth
                    depth1_normalized = resized_depth1 / depth_mean
                    depth2_normalized = resized_depth2 / depth_mean
                    # normalize translation
                    t_1to2 = t_1to2 / depth_mean
                    t_2to1 = t_2to1 / depth_mean
                    if 'motion_pred' in batch:
                        resized_motion_1to2 = resized_motion_1to2 / depth_mean
                        resized_motion_2to1 = resized_motion_2to1 / depth_mean
                else:
                    depth1_normalized = resized_depth1
                    depth2_normalized = resized_depth2

                output_1_to_2 = self.rgbd_consistency_loss(resized_frame1, resized_frame2,
                                                           depth1_normalized, depth2_normalized,
                                                           resized_intrinsics,
                                                           R_1to2, t_1to2)

                losses = merge_loss(losses, output_1_to_2, scale_w)

                output_2_to_1 = self.rgbd_consistency_loss(resized_frame2, resized_frame1,
                                                           depth2_normalized, depth1_normalized,
                                                           resized_intrinsics,
                                                           R_2to1, t_2to1)

                losses = merge_loss(losses, output_2_to_1, scale_w)

                batch['depth_proximity_weight'].append((output_1_to_2['depth_proximity_weight'],
                                                        output_2_to_1['depth_proximity_weight']))

                if self.rot_cycle_loss_w > 0 or self.trans_cycle_loss_w > 0:
                    rot_loss, trans_loss = motion_consistency_loss(output_1_to_2['coords_A_in_B'],
                                                                   output_1_to_2['occlusion_mask'],
                                                                   R_1to2, R_2to1,
                                                                   t_1to2, t_2to1)
                    losses['rot_loss'] += rot_loss * scale_w * self.rot_cycle_loss_w
                    losses['trans_loss'] += trans_loss * scale_w * self.trans_cycle_loss_w

                    rot_loss, trans_loss = motion_consistency_loss(output_2_to_1['coords_A_in_B'],
                                                                   output_2_to_1['occlusion_mask'],
                                                                   R_2to1, R_1to2,
                                                                   t_2to1, t_1to2)
                    losses['rot_loss'] += rot_loss * scale_w * self.rot_cycle_loss_w
                    losses['trans_loss'] += trans_loss * scale_w * self.trans_cycle_loss_w

                if 'motion_pred' in batch:

                    t_1to2_scale = t_1to2.pow(2).mean([1, 2, 3], keepdim=True) * 3.0
                    t_2to1_scale = t_2to1.pow(2).mean([1, 2, 3], keepdim=True) * 3.0
                    m_1to2_normalized = resized_motion_1to2 / torch.sqrt(t_1to2_scale + 1e-12)
                    m_2to1_normalized = resized_motion_2to1 / torch.sqrt(t_2to1_scale + 1e-12)

                    if self.motion_smooth_loss_w > 0.0:
                        losses['motion_smooth_loss'] += \
                            motion_smoothness_loss_fn(m_1to2_normalized) * scale_w * self.motion_smooth_loss_w
                        losses['motion_smooth_loss'] += \
                            motion_smoothness_loss_fn(m_2to1_normalized) * scale_w * self.motion_smooth_loss_w

                    if self.motion_sparsity_loss_w > 0.0:
                        losses['motion_sparsity_loss'] += \
                            motion_sparsity_loss_fn(m_1to2_normalized) * scale_w * self.motion_sparsity_loss_w
                        losses['motion_sparsity_loss'] += \
                            motion_sparsity_loss_fn(m_2to1_normalized) * scale_w * self.motion_sparsity_loss_w

                if self.sup_loss_w > 0.0:
                    depth1_gt = resize_img(batch['depth'], (H, W), mode='nearest')
                    depth2_gt = resize_img(batch['ctx_depth'][0], (H, W), mode='nearest')

                    losses['sup_loss'] += \
                        self.supervise_loss(resized_depth1, depth1_gt) * scale_w * self.sup_loss_w
                    losses['sup_loss'] += \
                        self.supervise_loss(resized_depth2, depth2_gt) * scale_w * self.sup_loss_w

                if self.smooth_loss_w > 0.0:
                    losses['smooth_loss'] += \
                        smoothness_loss(depth1_normalized, resized_frame1) * scale_w * self.smooth_loss_w
                    losses['smooth_loss'] += \
                        smoothness_loss(depth2_normalized, resized_frame2) * scale_w * self.smooth_loss_w

                if self.var_loss_w > 0.0:
                    losses['var_loss'] += variance_loss(resized_depth1) * scale_w * self.var_loss_w
                    losses['var_loss'] += variance_loss(resized_depth2) * scale_w * self.var_loss_w

            batch.update(losses)

        else:
            batch['depth_net_input'] = (batch["img"] - self.pixel_mean) / self.pixel_std
            batch = self.depth_net(batch)
        return batch

    def rgbd_consistency_loss(self, frame_A, frame_B, depth_A, depth_B, intrinsics, R_A2B, t_A2B):

        return_dict = {}

        sampled_values, depth_in_B, points_A_coords_in_B, proj_mask = view_synthesis(
            torch.cat([frame_B, depth_B], 1), depth_A, intrinsics, R_A2B, t_A2B)

        return_dict['coords_A_in_B'] = points_A_coords_in_B

        sampled_frame_B, sampled_depth_B = torch.split(sampled_values, [3, 1], dim=1)

        occlusion_mask = ((depth_in_B < sampled_depth_B).float() * proj_mask.float())

        return_dict['occlusion_mask'] = occlusion_mask

        normalizer = occlusion_mask.sum([1, 2, 3]) + 1

        # todo depth l1 not work, why?
        if self.depth_l1_loss_w > 0:
            depth_l1_loss = torch.abs(sampled_depth_B.detach() - depth_in_B) * occlusion_mask
            return_dict['depth_l1_loss'] = (depth_l1_loss.sum([1, 2, 3]) / normalizer).mean() * self.depth_l1_loss_w

        rgb_l1_loss = torch.abs(sampled_frame_B - frame_A) * occlusion_mask
        # return_dict['rgb_l1_loss'] = (rgb_l1_loss.sum([1, 2, 3]) / normalizer).mean()  # todo test this
        return_dict['rgb_l1_loss'] = rgb_l1_loss.mean()

        # SSIM loss
        if self.ssim_loss_w > 0.0:
            # [B, 1, H, W]
            depth_error = (depth_in_B - sampled_depth_B) ** 2
            # [B]
            depth_err_2nd_mom = ((depth_error * occlusion_mask).sum([1, 2, 3]) / normalizer) + 1e-4
            # [B, 1, H, W]
            depth_proximity_weight = \
                (depth_err_2nd_mom.view(-1, 1, 1, 1) / (depth_error + depth_err_2nd_mom.view(-1, 1, 1, 1)))
            depth_proximity_weight = (depth_proximity_weight * proj_mask.float()).detach()

            rgb_ssim_loss, avg_weight = self.ssim(sampled_frame_B, frame_A, depth_proximity_weight)
            rgb_ssim_loss = (rgb_ssim_loss * avg_weight).mean()

            return_dict['depth_proximity_weight'] = depth_proximity_weight
            return_dict['ssim_loss'] = rgb_ssim_loss * self.ssim_loss_w * 0.5

        return return_dict
