# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from SfmLearner
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/models/PoseExpNet.py

import torch
import torch.nn as nn

from .build import POSE_NET_REGISTRY
from ...geometry.pose_utils import pose_vec2mat


def conv_gn_relu(in_planes, out_planes, kernel_size=3, stride=2, group_norm=True):
    layers = [nn.Conv2d(in_planes, out_planes,
                        kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride),
              nn.ReLU(inplace=True)]
    if group_norm:
        layers.insert(1, nn.GroupNorm(16, out_planes))
    return nn.Sequential(*layers)


@POSE_NET_REGISTRY.register()
class PoseNet(nn.Module):
    """Pose network """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.nb_ref_imgs = cfg.MODEL.POSE_NET.NUM_CONTEXTS

        channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn_relu(3 * (1 + self.nb_ref_imgs), channels[0], kernel_size=7)
        self.conv2 = conv_gn_relu(channels[0], channels[1], kernel_size=5)
        self.conv3 = conv_gn_relu(channels[1], channels[2])
        self.conv4 = conv_gn_relu(channels[2], channels[3])
        self.conv5 = conv_gn_relu(channels[3], channels[4])
        self.conv6 = conv_gn_relu(channels[4], channels[5])
        self.conv7 = conv_gn_relu(channels[5], channels[6])

        self.pose_pred = nn.Conv2d(channels[6], 6 * self.nb_ref_imgs, kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        batch['pose_pred'] = [pose_vec2mat(pose[:, i]) for i in range(pose.shape[1])]

        return batch
