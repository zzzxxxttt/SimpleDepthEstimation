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
        self.supervise_loss = silog_loss(cfg.LOSS.VARIANCE_FOCUS)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch):
        batch = to_cuda(batch, self.device)

        batch["image"] = (batch["image"] - self.pixel_mean) / self.pixel_std

        output = self.depth_net(batch)

        if self.training:
            depth_gt = [resize_img(batch['depth_gt'], pred.shape[-2:], mode='nearest')
                        for pred in output['depth_pred']]
            sup_losses = [self.supervise_loss(pred, gt) for pred, gt in zip(output['depth_pred'], depth_gt)]
            output['silog_loss'] = sum(sup_losses) / len(sup_losses)
        else:
            output['depth_pred'] = self.post_process(output['depth_pred'][0], batch)
        return output


def post_process(depth_pred, batch):
    orig_h, orig_w = batch['img_h'][0].item(), batch['img_w'][0].item()

    top_margin = int(batch['top_margin'][0])
    left_margin = int(batch['left_margin'][0])
    B, _, H, W = depth_pred.shape

    croped_h, croped_w = orig_h - top_margin, orig_w - left_margin * 2
    if croped_h != H or croped_w != W:
        depth_pred = resize_img(depth_pred, (croped_h, croped_w), mode='nearest')

    pred_uncropped = np.zeros((B, 1, orig_h, orig_w), dtype=np.float32)
    pred_uncropped[:, :, top_margin:croped_h, left_margin:croped_w] = to_numpy(depth_pred)
    return pred_uncropped
