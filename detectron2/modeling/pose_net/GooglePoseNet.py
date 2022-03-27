import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import POSE_NET_REGISTRY
# from .PoseNet import conv_gn_relu
from ...geometry.pose_utils import pose_vec2mat
from detectron2.layers.conv_tf import Conv2dTF


def conv_gn_relu(in_planes, out_planes, kernel_size=3, stride=2, group_norm=True):
    layers = [
        # Conv2dTF(in_planes, out_planes, kernel_size=kernel_size, stride=stride),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride),
        nn.ReLU(inplace=True)
    ]
    if group_norm:
        layers.insert(1, nn.GroupNorm(16, out_planes))
    return nn.Sequential(*layers)


# conv2d = Conv2dTF
conv2d = nn.Conv2d


def clip_STE(x, min_value):
    return (torch.clamp_min(x, min_value) - x).detach() + x


@POSE_NET_REGISTRY.register()
class GooglePoseNet(nn.Module):
    """Pose network """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        group_norm = cfg.MODEL.POSE_NET.GROUP_NORM
        self.learn_scale = cfg.MODEL.POSE_NET.LEARN_SCALE

        in_channels = 4 * 2 if cfg.MODEL.POSE_NET.USE_DEPTH else 3 * 2
        channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn_relu(in_channels, channels[0], kernel_size=7, group_norm=group_norm)
        self.conv2 = conv_gn_relu(channels[0], channels[1], kernel_size=5, group_norm=group_norm)
        self.conv3 = conv_gn_relu(channels[1], channels[2], group_norm=group_norm)
        self.conv4 = conv_gn_relu(channels[2], channels[3], group_norm=group_norm)
        self.conv5 = conv_gn_relu(channels[3], channels[4], group_norm=group_norm)
        self.conv6 = conv_gn_relu(channels[4], channels[5], group_norm=group_norm)
        self.conv7 = conv_gn_relu(channels[5], channels[6], group_norm=group_norm)

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

    def forward(self, batch):
        B = batch['pose_net_input'].shape[0]

        out_conv1 = self.conv1(batch['pose_net_input'])
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7.mean([2, 3], keepdim=True)).view(B, 6)
        trans, rot = pose[:, :3], pose[:, 3:]

        if self.learn_scale:
            rot_scale = torch.relu(self.rot_scale - 0.001) + 0.001
            trans_scale = torch.relu(self.trans_scale - 0.001) + 0.001
            pose = torch.cat([trans * trans_scale, rot * rot_scale], -1)
        else:
            pose = torch.cat([trans * 0.01, rot * 0.01], -1)

        batch['pose_pred'] = pose_vec2mat(pose)
        return batch


class MotionRefiner(nn.Module):
    def __init__(self, channel_out, channel_mid, group_norm):
        super().__init__()
        self.conv1 = conv_gn_relu(channel_out + channel_mid, channel_mid,
                                  kernel_size=3, group_norm=group_norm, stride=1)
        self.conv21 = conv_gn_relu(channel_out + channel_mid, channel_mid,
                                   kernel_size=3, group_norm=group_norm, stride=1)
        self.conv22 = conv_gn_relu(channel_mid, channel_mid,
                                   kernel_size=3, group_norm=group_norm, stride=1)
        self.conv3 = conv2d(channel_mid * 2, channel_out, kernel_size=1, bias=False)

    def forward(self, trans, trans_skip):
        upsampled_trans = F.interpolate(trans, size=trans_skip.shape[-2:], mode='bilinear', align_corners=True)
        inputs = torch.cat([upsampled_trans, trans_skip], 1)
        out1 = self.conv1(inputs)
        out2 = self.conv22(self.conv21(inputs))
        out = torch.cat([out1, out2], 1)
        out = upsampled_trans + self.conv3(out)
        return out


@POSE_NET_REGISTRY.register()
class GoogleMotionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        group_norm = cfg.MODEL.POSE_NET.GROUP_NORM
        self.learn_scale = cfg.MODEL.POSE_NET.LEARN_SCALE
        self.mask_motion = cfg.MODEL.POSE_NET.MASK_MOTION
        self.scale_constrain = cfg.MODEL.POSE_NET.get('SCALE_CONSTRAIN', 'clip')
        self.motion_weight = 1.0

        in_channels = 4 * 2 if cfg.MODEL.POSE_NET.USE_DEPTH else 3 * 2
        channels = [16, 32, 64, 128, 256, 512, 1024]
        self.conv1 = conv_gn_relu(in_channels, channels[0], group_norm=group_norm)
        self.conv2 = conv_gn_relu(channels[0], channels[1], group_norm=group_norm)
        self.conv3 = conv_gn_relu(channels[1], channels[2], group_norm=group_norm)
        self.conv4 = conv_gn_relu(channels[2], channels[3], group_norm=group_norm)
        self.conv5 = conv_gn_relu(channels[3], channels[4], group_norm=group_norm)
        self.conv6 = conv_gn_relu(channels[4], channels[5], group_norm=group_norm)
        self.conv7 = conv_gn_relu(channels[5], channels[6], group_norm=group_norm)

        self.pose_pred = conv2d(channels[6], 6, kernel_size=1, bias=False)

        self.conv8 = conv2d(6, 3, kernel_size=1)

        self.refiner7 = MotionRefiner(channel_out=3, channel_mid=channels[6], group_norm=group_norm)
        self.refiner6 = MotionRefiner(channel_out=3, channel_mid=channels[5], group_norm=group_norm)
        self.refiner5 = MotionRefiner(channel_out=3, channel_mid=channels[4], group_norm=group_norm)
        self.refiner4 = MotionRefiner(channel_out=3, channel_mid=channels[3], group_norm=group_norm)
        self.refiner3 = MotionRefiner(channel_out=3, channel_mid=channels[2], group_norm=group_norm)
        self.refiner2 = MotionRefiner(channel_out=3, channel_mid=channels[1], group_norm=group_norm)
        self.refiner1 = MotionRefiner(channel_out=3, channel_mid=channels[0], group_norm=group_norm)
        self.refiner0 = MotionRefiner(channel_out=3, channel_mid=in_channels, group_norm=False)

        if self.learn_scale:
            if self.scale_constrain == 'softplus':
                self.register_parameter('rot_scale', nn.Parameter(torch.tensor(0.4)))
                self.register_parameter('trans_scale', nn.Parameter(torch.tensor(0.4)))
            else:
                self.register_parameter('rot_scale', nn.Parameter(torch.tensor(0.01)))
                self.register_parameter('trans_scale', nn.Parameter(torch.tensor(0.01)))

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, batch):
        out_conv1 = self.conv1(batch['pose_net_input'])
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7.mean([2, 3], keepdim=True))
        rot, trans = pose[:, :3, :, :], pose[:, 3:, :, :]

        residual_motion = self.conv8(pose)
        residual_motion = self.refiner7(residual_motion, out_conv7)
        residual_motion = self.refiner6(residual_motion, out_conv6)
        residual_motion = self.refiner5(residual_motion, out_conv5)
        residual_motion = self.refiner4(residual_motion, out_conv4)
        residual_motion = self.refiner3(residual_motion, out_conv3)
        residual_motion = self.refiner2(residual_motion, out_conv2)
        residual_motion = self.refiner1(residual_motion, out_conv1)
        residual_motion = self.refiner0(residual_motion, batch['pose_net_input'])

        if self.learn_scale:
            if self.scale_constrain == 'clip_ste':
                trans_scale = clip_STE(self.trans_scale, 0.001)
                rot_scale = clip_STE(self.rot_scale, 0.001)
            elif self.scale_constrain == 'clip':
                trans_scale = torch.relu(self.trans_scale - 0.001) + 0.001
                rot_scale = torch.relu(self.rot_scale - 0.001) + 0.001
            elif self.scale_constrain == 'softplus':
                trans_scale = F.softplus(self.trans_scale) * 0.01 + 0.001
                rot_scale = F.softplus(self.rot_scale) * 0.01 + 0.001
            else:
                raise NotImplementedError

            pose = torch.cat([trans[:, :, 0, 0] * trans_scale, rot[:, :, 0, 0] * rot_scale], -1)
            residual_motion = residual_motion * trans_scale
        else:
            pose = torch.cat([trans[:, :, 0, 0] * 0.01, rot[:, :, 0, 0] * 0.01], -1)
            residual_motion = residual_motion * 0.01

        if self.mask_motion:
            sq_residual_motion = torch.sqrt((residual_motion ** 2).sum(dim=1, keepdim=True))
            mean_sq_residual_motion = sq_residual_motion.mean()
            # A mask of shape [B, h, w, 1]
            residual_motion *= (sq_residual_motion > mean_sq_residual_motion).float()

        batch['pose_pred'] = pose_vec2mat(pose)
        batch['motion_pred'] = residual_motion * self.motion_weight
        return batch
