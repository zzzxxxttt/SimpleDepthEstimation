import random

import torch
import torch.nn as nn
from functools import partial

from .build import DEPTH_NET_REGISTRY

from ...layers.resnet_encoder import ResnetEncoder
from ...layers.depth_decoder import DepthDecoder, disp_to_depth
from ...geometry.camera import resize_img


@DEPTH_NET_REGISTRY.register()
class DepthResNet(nn.Module):
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
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=cfg.MODEL.MAX_DEPTH)

        self.upsample_depth = cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH
        self.flip_prob = cfg.MODEL.DEPTH_NET.FLIP_PROB

    def forward(self, data):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        image = data['image']
        flip = False
        if self.training and random.random() < self.flip_prob:
            image = torch.flip(image, [3])
            flip = True

        x = self.encoder(image)
        x = self.decoder(x)
        disps = [self.scale_inv_depth(x[('disp', i)])[1] for i in range(4)]

        if flip:
            disps = [torch.flip(d, [3]) for d in disps]

        if self.upsample_depth:
            disps = [resize_img(d, data['image'].shape[-2:], mode='nearest') for d in disps]

        return {'res2': disps[3],
                'res3': disps[2],
                'res4': disps[1],
                'depth_pred': disps}

########################################################################################################################
