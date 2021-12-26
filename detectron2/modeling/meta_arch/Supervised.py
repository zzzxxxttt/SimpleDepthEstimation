import logging

import numpy as np

import torch
import torch.nn as nn

from .build import META_ARCH_REGISTRY
from ..depth_net import build_depth_net
from detectron2.modeling.losses.losses import silog_loss
from ...utils.memory import to_cuda, to_numpy
from ...geometry.camera import resize_img

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class SupDepthModel(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__()

        self.depth_net = build_depth_net(cfg)
        self.loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):
        batch = to_cuda(batch, self.device)

        batch["depth_net_input"] = (batch["img"] - self.pixel_mean) / self.pixel_std

        output = self.depth_net(batch)

        if self.training:
            depth_gt = [resize_img(batch['depth'], pred.shape[-2:], mode='nearest') for pred in output['depth_pred']]
            sup_losses = [self.loss(pred, gt) for pred, gt in zip(output['depth_pred'], depth_gt)]
            output['silog_loss'] = sum(sup_losses) / len(sup_losses)
        else:
            output['depth_pred'] = output['depth_pred'][0]
        return output
