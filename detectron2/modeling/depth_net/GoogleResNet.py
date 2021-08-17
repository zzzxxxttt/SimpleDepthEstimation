from collections import OrderedDict

import random
import logging
import numpy as np
from functools import partial

import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url

from .build import DEPTH_NET_REGISTRY

from ...layers.depth_decoder import upsample, ConvBlock, Conv3x3
from ...geometry.camera import resize_img

from ...layers.layer_norm import LayerNorm

logger = logging.getLogger(__name__)


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](False, norm_layer=LayerNorm)

        if pretrained:
            pretrained_dict = load_state_dict_from_url(model_urls[f'resnet{num_layers}'], progress=True)
            model_dict = self.encoder.state_dict()

            for k in model_dict:
                if k not in pretrained_dict:
                    logger.info(f'missing key: {k} in pretrained model!')
                elif model_dict[k].shape != pretrained_dict[k].shape:
                    logger.info(f'shape mismatch: {k} '
                                f'{model_dict[k].shape} in model vs {pretrained_dict[k].shape} in pretrained!')

            for k in pretrained_dict:
                if k not in model_dict:
                    logger.info(f'unexpected key: {k} in pretrained model!')

            self.encoder.load_state_dict(pretrained_dict, strict=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []

        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)

        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 use_skips=True, learn_scale=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.scale = nn.Parameter(torch.ones(1), requires_grad=True) if learn_scale else None

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.softplus = nn.Softplus()

    def forward(self, input_features):
        outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.softplus(self.convs[("dispconv", i)](x))
                if self.scale is not None:
                    outputs[("disp", i)] *= self.scale

        return outputs


@DEPTH_NET_REGISTRY.register()
class GoogleResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        version = cfg.MODEL.DEPTH_NET.ENCODER_NAME
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])  # First two characters are the number of layers
        pretrained = version[2:] == 'pt'  # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    learn_scale=cfg.MODEL.DEPTH_NET.LEARN_SCALE)

        self.upsample_depth = cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH
        self.flip_prob = cfg.MODEL.DEPTH_NET.FLIP_PROB

    def set_stddev(self, stddev):
        for m in self.modules():
            if isinstance(m, LayerNorm):
                m.stddev = stddev

    def forward(self, data):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        image = data['depth_net_input']

        flip = random.random() < self.flip_prob
        if self.training and flip:
            image = torch.flip(image, [3])

        x = self.encoder(image)
        x = self.decoder(x)
        disps = [x[('disp', 0)]]

        if flip:
            disps = [torch.flip(d, [3]) for d in disps]

        if self.upsample_depth:
            disps = [resize_img(d, data['depth_net_input'].shape[-2:], mode='nearest') for d in disps]

        return {'depth_pred': disps}
