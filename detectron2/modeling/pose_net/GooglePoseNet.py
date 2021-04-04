import torch
import torch.nn as nn

from .build import POSE_NET_REGISTRY
from .PoseNet import conv_gn


@POSE_NET_REGISTRY.register()
class PoseNet(nn.Module):
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
        inputs = torch.cat([image, *context], 1)
        out_conv1 = self.conv1(inputs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6).mean([2, 3], keepdim=True)

        pose = self.pose_pred(out_conv7)

        if self.learn_scale:
            scale = torch.cat([self.rot_scale.expand(3), self.trans_scale.expand(3)], -1)[None, :]
            scale = torch.relu(scale - 0.001) + 0.001
            pose = scale * pose.view(pose.size(0), 6)
        else:
            pose = 0.01 * pose.view(pose.size(0), 6)

        return pose
