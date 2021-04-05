import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import POSE_NET_REGISTRY
from .PoseNet import conv_gn


@POSE_NET_REGISTRY.register()
class GooglePoseNet(nn.Module):
    """Pose network """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        group_norm = cfg.MODEL.POSE_NET.GROUP_NORM
        self.learn_scale = cfg.MODEL.POSE_NET.LEARN_SCALE
        channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn(3 * 2, channels[0], kernel_size=7, group_norm=group_norm)
        self.conv2 = conv_gn(channels[0], channels[1], kernel_size=5, group_norm=group_norm)
        self.conv3 = conv_gn(channels[1], channels[2], group_norm=group_norm)
        self.conv4 = conv_gn(channels[2], channels[3], group_norm=group_norm)
        self.conv5 = conv_gn(channels[3], channels[4], group_norm=group_norm)
        self.conv6 = conv_gn(channels[4], channels[5], group_norm=group_norm)
        self.conv7 = conv_gn(channels[5], channels[6], group_norm=group_norm)

        self.pose_pred = nn.Conv2d(channels[6], 6, kernel_size=1, padding=0)

        if self.learn_scale:
            self.register_parameter('rot_scale', nn.Parameter(torch.tensor(0.01)))
            self.register_parameter('trans_scale', nn.Parameter(torch.tensor(0.01)))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image, context):
        B, _, _, _ = image.shape
        inputs = torch.cat([image, *context], 1)
        out_conv1 = self.conv1(inputs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7.mean([2, 3], keepdim=True)).view(B, 6)
        rot, trans = pose[:, :3], pose[:, 3:]

        if self.learn_scale:
            rot_scale = torch.relu(self.rot_scale - 0.001) + 0.001
            trans_scale = torch.relu(self.trans_scale - 0.001) + 0.001
            pose = torch.stack([rot * rot_scale, trans * trans_scale], -1)
        else:
            pose = torch.stack([rot * 0.01, trans * 0.01], -1)

        return pose


@POSE_NET_REGISTRY.register()
class GoogleMotionNet(nn.Module):
    """Pose network """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        group_norm = cfg.MODEL.POSE_NET.GROUP_NORM
        self.learn_scale = cfg.MODEL.POSE_NET.LEARN_SCALE
        self.mask_motion = cfg.MODEL.POSE_NET.MASK_MOTION

        in_channels = 4 * 2 if cfg.MODEL.POSE_NET.USE_DEPTH else 3 * 2
        channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn(in_channels, channels[0], kernel_size=7, group_norm=group_norm)
        self.conv2 = conv_gn(channels[0], channels[1], kernel_size=5, group_norm=group_norm)
        self.conv3 = conv_gn(channels[1], channels[2], group_norm=group_norm)
        self.conv4 = conv_gn(channels[2], channels[3], group_norm=group_norm)
        self.conv5 = conv_gn(channels[3], channels[4], group_norm=group_norm)
        self.conv6 = conv_gn(channels[4], channels[5], group_norm=group_norm)
        self.conv7 = conv_gn(channels[5], channels[6], group_norm=group_norm)

        self.pose_pred = nn.Conv2d(channels[6], 6, kernel_size=1, padding=0)

        self.conv8 = conv_gn(3, 3, kernel_size=1, group_norm=group_norm)

        self.refiner7 = MotionRefiner(trans_channel=3, skip_channel=channels[6], group_norm=group_norm)
        self.refiner6 = MotionRefiner(trans_channel=3, skip_channel=channels[5], group_norm=group_norm)
        self.refiner5 = MotionRefiner(trans_channel=3, skip_channel=channels[4], group_norm=group_norm)
        self.refiner4 = MotionRefiner(trans_channel=3, skip_channel=channels[3], group_norm=group_norm)
        self.refiner3 = MotionRefiner(trans_channel=3, skip_channel=channels[2], group_norm=group_norm)
        self.refiner2 = MotionRefiner(trans_channel=3, skip_channel=channels[1], group_norm=group_norm)
        self.refiner1 = MotionRefiner(trans_channel=3, skip_channel=channels[0], group_norm=group_norm)
        self.refiner0 = MotionRefiner(trans_channel=3, skip_channel=in_channels, group_norm=group_norm)

        if self.learn_scale:
            self.register_parameter('rot_scale', nn.Parameter(torch.tensor(0.01)))
            self.register_parameter('trans_scale', nn.Parameter(torch.tensor(0.01)))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image, context):
        B, _, _, _ = image.shape
        inputs = torch.cat([image, *context], 1)
        out_conv1 = self.conv1(inputs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7.mean([2, 3], keepdim=True))
        rot, trans = pose[:, :3, :, :], pose[:, 3:, :, :]

        residual_motion = self.conv8(trans)
        residual_motion = self.refiner7(residual_motion, out_conv7)
        residual_motion = self.refiner6(residual_motion, out_conv6)
        residual_motion = self.refiner5(residual_motion, out_conv5)
        residual_motion = self.refiner4(residual_motion, out_conv4)
        residual_motion = self.refiner3(residual_motion, out_conv3)
        residual_motion = self.refiner2(residual_motion, out_conv2)
        residual_motion = self.refiner1(residual_motion, out_conv1)
        residual_motion = self.refiner0(residual_motion, image)

        if self.learn_scale:
            rot_scale = torch.relu(self.rot_scale - 0.001) + 0.001
            trans_scale = torch.relu(self.trans_scale - 0.001) + 0.001
            pose = torch.stack([rot * rot_scale, trans * trans_scale], -1)
            residual_motion = residual_motion * trans_scale
        else:
            pose = torch.stack([rot * 0.01, trans * 0.01], -1)
            residual_motion = residual_motion * 0.01

        if self.mask_motion:
            sq_residual_motion = torch.sqrt((residual_motion ** 2).sum(dim=1, keepdim=True))
            mean_sq_residual_motion = sq_residual_motion.mean()
            # A mask of shape [B, h, w, 1]
            residual_motion *= (sq_residual_motion > mean_sq_residual_motion).float()

        return pose, residual_motion


class MotionRefiner(nn.Module):
    def __init__(self, trans_channel, skip_channel, group_norm):
        super().__init__()
        self.conv1 = conv_gn(trans_channel + skip_channel, skip_channel, kernel_size=3, group_norm=group_norm)
        self.conv21 = conv_gn(trans_channel + skip_channel, skip_channel, kernel_size=3, group_norm=group_norm)
        self.conv22 = conv_gn(skip_channel, skip_channel, kernel_size=3, group_norm=group_norm)
        self.conv3 = conv_gn(skip_channel * 2, trans_channel, kernel_size=1, group_norm=group_norm)

    def forward(self, trans, trans_skip):
        upsampled_trans = F.interpolate(trans, size=trans_skip.shape[-2:], mode='bilinear', align_corners=True)
        inputs = torch.cat([upsampled_trans, trans_skip], 1)
        out1 = self.conv1(inputs)
        out2 = self.conv22(self.conv21(inputs))
        out = torch.cat([out1, out2], 1)
        out = trans + self.conv3(out)
        return out
