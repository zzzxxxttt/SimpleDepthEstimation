# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn

from detectron2.utils.registry import Registry

POSE_NET_REGISTRY = Registry("POSE_NET")
POSE_NET_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_pose_net(cfg, input_shape=None):
    """
    Build a depth_net from `cfg.MODEL.DEPTH_NET.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """

    network_name = cfg.MODEL.POSE_NET.NAME
    pose_net = POSE_NET_REGISTRY.get(network_name)(cfg)
    assert isinstance(pose_net, nn.Module)
    return pose_net
