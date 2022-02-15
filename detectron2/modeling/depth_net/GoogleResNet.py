from collections import OrderedDict
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url

from .build import DEPTH_NET_REGISTRY

from ...layers.layer_norm import RandLayerNorm

logger = logging.getLogger(__name__)


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, norm_layer=nn.BatchNorm2d):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](False, norm_layer=norm_layer)

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
    def __init__(self, num_ch_enc, learn_scale=False):
        super(DepthDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.scale = nn.Parameter(torch.ones(1), requires_grad=True) if learn_scale else None

        # decoder
        self.blocks = nn.ModuleList()
        for i in range(4, -1, -1):
            # upconv_0
            C_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            C_out = self.num_ch_dec[i]
            C_cat = num_ch_enc[i - 1] if i > 0 else None
            self.blocks.append(UpsampleBlock(C_in, C_out, C_cat))

        self.out_conv = nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, stride=1, padding=1)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input_features):
        out = input_features[-1]
        for y, block in zip(input_features[-2::-1] + [None], self.blocks):
            out = block(out, y)
        out = F.softplus(self.out_conv(out))
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channel_in, channel_out, channel_cat=None):
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1)
        if channel_cat:
            self.iconv = nn.Conv2d(channel_out + channel_cat, channel_out, kernel_size=3, stride=1, padding=1)
        else:
            self.iconv = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y=None):
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        out = F.relu(self.upconv(out))
        if y is not None:
            out = torch.cat([out, y], 1)
        out = F.relu(self.iconv(out))
        return out


@DEPTH_NET_REGISTRY.register()
class GoogleResNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        version = cfg.MODEL.DEPTH_NET.ENCODER_NAME
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])  # First two characters are the number of layers
        pretrained = version[2:] == 'pt'  # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        norms = {'BN': nn.BatchNorm2d,
                 'randLN': RandLayerNorm,
                 None: None}

        self.encoder = ResnetEncoder(num_layers=num_layers,
                                     pretrained=pretrained,
                                     norm_layer=norms[cfg.MODEL.DEPTH_NET.NORM])
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    learn_scale=cfg.MODEL.DEPTH_NET.LEARN_SCALE)

        self.upsample_depth = cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH

    def set_stddev(self, stddev):

        def set_stddev(m):
            if isinstance(m, RandLayerNorm):
                m.stddev = stddev

        self.apply(set_stddev)

    def forward(self, batch):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        image = batch['depth_net_input']

        if batch.get('flip', False):
            image = torch.flip(image, [3])

        x = self.encoder(image)
        x = self.decoder(x)

        if batch.get('flip', False):
            x = torch.flip(x, [3])

        batch['depth_pred'] = [x]
        return batch
