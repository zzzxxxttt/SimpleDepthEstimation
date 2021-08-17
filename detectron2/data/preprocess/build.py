# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from easydict import EasyDict

from detectron2.utils.registry import Registry

PREPROCESS_REGISTRY = Registry("PREPROCESS")  # noqa F401 isort:skip
PREPROCESS_REGISTRY.__doc__ = """
Registry for preprocesses.

The registered object will be called with `obj(cfg)`
and expected to return a Preprocess object.
"""


class Preprocess:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, data_dict):
        return data_dict

    def inverse(self, data_dict):
        return data_dict


def build_preprocess(cfg):
    cfg = EasyDict(cfg)
    preprocess = PREPROCESS_REGISTRY.get(cfg.NAME)(cfg)
    assert isinstance(preprocess, Preprocess)
    return preprocess
