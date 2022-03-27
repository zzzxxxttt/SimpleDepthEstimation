import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import DEPTH_NET_REGISTRY

from detectron2.layers.layer_norm import RandLayerNorm
from detectron2.layers.conv_tf import Conv2dTF, ConvTranspose2dTF, MaxPool2dTF

logger = logging.getLogger(__name__)


# def conv1x1(in_planes, out_planes, stride=1, bias=False):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
#
#
# def conv3x3(in_planes, out_planes, stride=1, bias=False):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=bias)
#
#
# def conv7x7(in_planes, out_planes, stride=1, bias=False):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=0, bias=bias)

# def deconv3x3(in_planes, out_planes, stride=2, bias=False):
#     return ConvTranspose2dTF(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv7x7(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=bias)


def deconv3x3(in_planes, out_planes, stride=2, bias=False):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        self.downsample = None
        if inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        elif stride != 1:
            self.downsample = nn.MaxPool2d(stride, stride, padding=stride // 2)
            # self.downsample = MaxPool2dTF(stride, stride)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.stride = stride

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d):
        super(ResNetEncoder, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.conv1 = conv7x7(3, self.inplanes, stride=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = MaxPool2dTF(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, RandLayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, self._norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out0 = F.relu(out, inplace=True)
        out = self.maxpool(out0)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return [out0, out1, out2, out3, out4]


class UpsampleBlock(nn.Module):
    def __init__(self, channel_in, channel_out, channel_cat):
        super(UpsampleBlock, self).__init__()
        self.upconv = deconv3x3(channel_in, channel_out, bias=True)
        self.iconv = conv3x3(channel_out + channel_cat if channel_cat else channel_out, channel_out, bias=True)

    def forward(self, x, y=None):
        out = F.relu(self.upconv(x), inplace=True)
        if y is not None:
            out = torch.cat([out, y], 1)
        out = F.relu(self.iconv(out), inplace=True)
        return out


class DepthDecoder(nn.Module):
    def __init__(self, learn_scale=False):
        super(DepthDecoder, self).__init__()

        self.channels = [512, 256, 128, 64, 32, 16]
        self.enc_channels = [256, 128, 64, 64, None]

        self.scale = nn.Parameter(torch.ones(1), requires_grad=True) if learn_scale else None

        # decoder
        self.blocks = nn.ModuleList()
        for C_in, C_out, C_mid in zip(self.channels[:-1], self.channels[1:], self.enc_channels):
            self.blocks.append(UpsampleBlock(C_in, C_out, C_mid))

        self.out_conv = conv3x3(self.channels[-1], 1, bias=True)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input_features):
        out = input_features[-1]
        for y, block in zip(reversed([None] + input_features[:-1]), self.blocks):
            out = block(out, y)
        out = F.softplus(self.out_conv(out))
        return out


@DEPTH_NET_REGISTRY.register()
class GoogleResNetv2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        version = cfg.MODEL.DEPTH_NET.ENCODER_NAME
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])  # First two characters are the number of layers
        # pretrained = version[2:] == 'pt'  # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18], 'ResNet version {} not available'.format(num_layers)

        norm = {'BN': nn.BatchNorm2d,
                'randLN': RandLayerNorm,
                None: None}[cfg.MODEL.DEPTH_NET.NORM]

        self.encoder = {18: ResNetEncoder(BasicBlock, [2, 2, 2, 2], norm)}[num_layers]

        self.decoder = DepthDecoder(learn_scale=cfg.MODEL.DEPTH_NET.LEARN_SCALE)

        self.upsample_depth = cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH

    def set_stddev(self, stddev):

        def set_stddev(m):
            if isinstance(m, RandLayerNorm):
                m.stddev = stddev

        self.apply(set_stddev)

    def forward(self, batch):
        image = batch['depth_net_input']

        if batch.get('flip', False):
            image = torch.flip(image, [3])

        x = self.encoder(image)
        x = self.decoder(x)

        if batch.get('flip', False):
            x = torch.flip(x, [3])

        batch['depth_pred'] = [x]
        return batch
